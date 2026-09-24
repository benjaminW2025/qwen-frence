"""Real Qwen forward from scheduler-owned GPU metadata; no second KV allocator."""
from __future__ import annotations
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import torch

from design import select_action
from naive_forward import apply_rms_norm, apply_rope, apply_swiglu
from ragged_prefill import _rope_factors


def allocate_pool(cfg, blocks, device):
    shape = (blocks, 16, cfg.n_kv_heads, cfg.d_head)
    return SimpleNamespace(
        k_pool=[torch.empty(shape, device=device, dtype=torch.float16) for _ in range(cfg.n_layers)],
        v_pool=[torch.empty(shape, device=device, dtype=torch.float16) for _ in range(cfg.n_layers)],
        block_size=16,
    )


class ModelAdapter:
    def __init__(self, model, pool, loop, policy=None):
        self.model, self.pool, self.loop, self.policy = model, pool, loop, policy
        self.decisions = {}
        self.step_calls = []
        self.observer = None

    @torch.no_grad()
    def __call__(self, ids, positions, slots, cu, context, table, max_query, decode):
        from kernel_dispatch import packed_paged_prefill_attention
        from paged_decode_attention import paged_decode_attention

        cfg, model, pool = self.model.cfg, self.model, self.pool
        tokens, heads, kvheads, dim = ids.numel(), cfg.n_heads, cfg.n_kv_heads, cfg.d_head
        action = None
        if decode and self.policy:
            action = select_action(self.policy, tokens, self.loop.max_decode_context_length())
        if decode:
            key = "production" if action is None else f"H1-K{action[0]}-S{action[1]}"
            self.decisions[key] = self.decisions.get(key, 0) + 1
        self.step_calls.append((bool(decode), tokens, context.numel(), max_query))
        x = model.embed(ids)
        cos, sin = (None, None) if cfg.use_custom_kernels else _rope_factors(cfg, positions, x.dtype)
        for i, layer in enumerate(model.layers):
            residual = x
            h = apply_rms_norm(x, layer.input_norm, cfg)
            q, k, v = layer.project_qkv(h)
            q = q.view(tokens, heads, dim).transpose(0, 1).unsqueeze(0)
            k = k.view(tokens, kvheads, dim).transpose(0, 1).unsqueeze(0)
            v = v.view(tokens, kvheads, dim).transpose(0, 1).unsqueeze(0)
            q = apply_rope(q, cos, sin, cfg, positions[None, :])
            k = apply_rope(k, cos, sin, cfg, positions[None, :])
            pool.k_pool[i].view(-1, kvheads, dim).index_copy_(0, slots, k[0].transpose(0, 1).contiguous())
            pool.v_pool[i].view(-1, kvheads, dim).index_copy_(0, slots, v[0].transpose(0, 1).contiguous())
            if decode:
                query = q[0].transpose(0, 1).contiguous()
                if action is None:
                    attention = paged_decode_attention(query, pool.k_pool[i], pool.v_pool[i], table, context)
                else:
                    from paged_decode_grouped_splitk_pipelined import grouped_splitk_attention
                    attention = grouped_splitk_attention(
                        query, pool.k_pool[i], pool.v_pool[i], table, context,
                        heads_per_program=1, split_k=action[0], num_stages=action[1],
                        num_warps=4, pipelined=action[1] > 1)
                attention = attention.reshape(tokens, heads * dim)
            else:
                attention = packed_paged_prefill_attention(
                    q, pool.k_pool[i], pool.v_pool[i], cu, table, context,
                    max_query_len=max_query, page_size=16, tile_policy="static")
                attention = attention.transpose(1, 2).reshape(tokens, heads * dim)
            x = residual + layer.o_proj(attention)
            residual = x
            h = apply_rms_norm(x, layer.post_attn_norm, cfg)
            gate, up = layer.project_gate_up(h)
            x = residual + layer.down_proj(apply_swiglu(gate, up, cfg))
        x = apply_rms_norm(x, model.norm, cfg)
        if not decode:
            x = x.index_select(0, cu[1:].to(torch.long) - 1)
        logits = model.lm_head(x)
        if self.observer is not None:
            checked = self.observer((ids, positions, slots, cu, context, table, max_query, decode), logits)
            if checked is not None:
                return checked
        return logits


class GraphModelAdapter(ModelAdapter):
    """Run C++-scheduled decode through bucketed full-model CUDA graphs.

    Prefill keeps the existing eager adapter. The C++ loop owns and transfers
    metadata; the graph decoder copies the active views into fixed capture inputs.
    Capture happens at construction, before requests are submitted or timed.
    """

    def __init__(self, model, pool, loop, *, max_running, max_context_length,
                 decode_attention_policy="production", decode_buckets=None,
                 enable_residual_rmsnorm=False,
                 enable_native_decode_rope_kv=False,
                 enable_native_decode_qkv_postprocess=False,
                 enable_fused_qkv_rope_cache=False,
                 enable_packed_qkv_rope_cache=False,
                 enable_stable_decode_table_cache=False):
        super().__init__(model, pool, loop)
        if decode_attention_policy not in ("production", "splitk", "fa3"):
            raise ValueError("graph decode policy must be 'production', 'splitk', or 'fa3'")
        graph_dir = Path(__file__).resolve().parents[2] / "engine/graph"
        if str(graph_dir) not in sys.path:
            sys.path.insert(0, str(graph_dir))
        from bucketed_graph_decoder import BucketedGraphDecoder

        self.decode_attention_policy = decode_attention_policy
        self.enable_residual_rmsnorm = bool(enable_residual_rmsnorm)
        self.enable_native_decode_rope_kv = bool(enable_native_decode_rope_kv)
        self.enable_native_decode_qkv_postprocess = bool(
            enable_native_decode_qkv_postprocess
        )
        self.enable_fused_qkv_rope_cache = bool(enable_fused_qkv_rope_cache)
        self.enable_packed_qkv_rope_cache = bool(enable_packed_qkv_rope_cache)
        self.max_context_length = max_context_length
        self.max_blocks = (max_context_length + pool.block_size - 1) // pool.block_size
        if self.max_blocks < 1:
            raise ValueError("graph decode requires a positive context bound")
        from paged_decode_attention import (resolve_decode_attention_policy,
                                            select_splitk_config)
        effective = resolve_decode_attention_policy(decode_attention_policy,
                                                    max_context_length)
        if effective == "splitk":
            config = select_splitk_config(max_context_length, page_size=pool.block_size)
            self.action = f"H1-K{config['split_k']}-S{config['num_stages']}"
        elif effective == "fa3":
            self.action = "FA3-auto"
        else:
            self.action = "production"
        self.graph_decoder = BucketedGraphDecoder(
            model, pool, max_running, self.max_blocks,
            pool.k_pool[0].device, pool.k_pool[0].dtype,
            buckets=decode_buckets,
            decode_attention_policy=decode_attention_policy,
            max_decode_context_length=max_context_length,
            enable_residual_rmsnorm=self.enable_residual_rmsnorm,
            enable_native_decode_rope_kv=self.enable_native_decode_rope_kv,
            enable_native_decode_qkv_postprocess=(
                self.enable_native_decode_qkv_postprocess
            ),
            enable_fused_qkv_rope_cache=self.enable_fused_qkv_rope_cache,
            enable_packed_qkv_rope_cache=self.enable_packed_qkv_rope_cache,
            enable_stable_decode_table_cache=enable_stable_decode_table_cache,
        )

    @torch.no_grad()
    def __call__(self, ids, positions, slots, cu, context, table, max_query, decode):
        if not decode:
            return super().__call__(ids, positions, slots, cu, context, table,
                                    max_query, decode)
        if ids.ndim != 1 or max_query != 1 or cu.numel() != 0:
            raise ValueError("graph decode requires one token per sequence")
        if table.shape[1] > self.max_blocks:
            raise ValueError("decode block table exceeds captured width")

        self.decisions[self.action] = self.decisions.get(self.action, 0) + 1
        self.step_calls.append((True, ids.numel(), context.numel(), max_query))
        logits = self.graph_decoder.decode(
            ids.view(-1, 1), positions, context, table, slots
        ).squeeze(1)
        if self.observer is not None:
            checked = self.observer((ids, positions, slots, cu, context, table, max_query, decode), logits)
            if checked is not None:
                return checked
        return logits


class PiecewiseGraphModelAdapter(GraphModelAdapter):
    """Captured decode plus bucketed prefill pieces around eager attention."""

    def __init__(self, model, pool, loop, *, max_running, max_context_length,
                 decode_attention_policy="production", max_capture_tokens=2048,
                 max_prefill_shapes=8, prefill_buckets=None, decode_buckets=None,
                 enable_residual_rmsnorm=False,
                 enable_native_decode_rope_kv=False,
                 enable_native_decode_qkv_postprocess=False,
                 enable_fused_qkv_rope_cache=False,
                 enable_packed_qkv_rope_cache=False,
                 enable_stable_decode_table_cache=False,
                 enable_prefill_packed_qkv_rope_cache=False,
                 enable_prefill_residual_rmsnorm=False,
                 enable_prefill_swiglu_fusion=False):
        super().__init__(model, pool, loop, max_running=max_running,
                         max_context_length=max_context_length,
                         decode_attention_policy=decode_attention_policy,
                         decode_buckets=decode_buckets,
                         enable_residual_rmsnorm=enable_residual_rmsnorm,
                         enable_native_decode_rope_kv=enable_native_decode_rope_kv,
                         enable_native_decode_qkv_postprocess=(
                             enable_native_decode_qkv_postprocess
                         ),
                         enable_fused_qkv_rope_cache=enable_fused_qkv_rope_cache,
                         enable_packed_qkv_rope_cache=enable_packed_qkv_rope_cache,
                         enable_stable_decode_table_cache=enable_stable_decode_table_cache)
        graph_dir = Path(__file__).resolve().parents[2] / "engine/graph"
        if str(graph_dir) not in sys.path:
            sys.path.insert(0, str(graph_dir))
        from piecewise_prefill import PiecewisePrefill
        self.enable_prefill_packed_qkv_rope_cache = bool(
            enable_prefill_packed_qkv_rope_cache
        )
        self.enable_prefill_residual_rmsnorm = bool(enable_prefill_residual_rmsnorm)
        self.enable_prefill_swiglu_fusion = bool(enable_prefill_swiglu_fusion)
        self.piecewise_prefill = PiecewisePrefill(
            model, pool, max_capture_tokens=max_capture_tokens,
            max_shapes=max_prefill_shapes, token_buckets=prefill_buckets,
            enable_packed_qkv_rope_cache=self.enable_prefill_packed_qkv_rope_cache,
            enable_residual_rmsnorm=self.enable_prefill_residual_rmsnorm,
            enable_swiglu_fusion=self.enable_prefill_swiglu_fusion)

    @torch.no_grad()
    def __call__(self, ids, positions, slots, cu, context, table, max_query, decode):
        if decode:
            return super().__call__(ids, positions, slots, cu, context, table,
                                    max_query, decode)
        logits = self.piecewise_prefill.forward(
            ids, positions, slots, cu, context, table, max_query)
        if logits is None:
            return ModelAdapter.__call__(self, ids, positions, slots, cu, context,
                                         table, max_query, False)
        self.step_calls.append((False, ids.numel(), context.numel(), max_query))
        if self.observer is not None:
            checked = self.observer((ids, positions, slots, cu, context, table, max_query, False), logits)
            if checked is not None:
                return checked
        return logits


def combine_mixed_metadata(decode, prefill):
    """Pack one-token decode rows before ragged prompt chunks.

    Both use the same paged attention contract: context length includes the
    newly written query tokens, and cu_seqlens marks each sequence's query
    chunk. The decode metadata must have been cloned before the C++ scheduler
    reuses its H2D staging buffer for prefill.
    """
    d_ids, d_positions, d_slots, d_cu, d_context, d_table, _, _ = decode
    p_ids, p_positions, p_slots, p_cu, p_context, p_table, p_max_query, _ = prefill
    if d_cu.numel() != 0 or d_ids.numel() != d_context.numel():
        raise ValueError("mixed decode metadata must contain one token per sequence")
    if p_cu.numel() != p_context.numel() + 1:
        raise ValueError("mixed prefill cu_seqlens does not match request count")
    width = max(d_table.shape[1], p_table.shape[1])
    import torch.nn.functional as F
    if d_table.shape[1] < width:
        d_table = F.pad(d_table, (0, width - d_table.shape[1]))
    if p_table.shape[1] < width:
        p_table = F.pad(p_table, (0, width - p_table.shape[1]))
    tables = torch.cat((d_table, p_table), dim=0)
    decode_rows = d_ids.numel()
    cu = torch.cat((torch.arange(decode_rows + 1, device=d_ids.device,
                                 dtype=p_cu.dtype), p_cu[1:] + decode_rows))
    return (torch.cat((d_ids, p_ids)),
            torch.cat((d_positions, p_positions)),
            torch.cat((d_slots, p_slots)), cu,
            torch.cat((d_context, p_context)), tables, p_max_query)


class PackedMixedPiecewiseGraphModelAdapter(PiecewiseGraphModelAdapter):
    """Use one packed model pass for a C++-scheduled mixed iteration.

    The first C++ callback stages decode metadata and returns a logits tensor
    that is filled during the subsequent prefill callback, before C++ samples.
    Pure decode and pure prefill retain their established dispatch paths.
    """

    def __init__(self, *args, mixed_attention_policy="packed_paged",
                 full_mixed_graph=False, clone_decode_metadata=True, **kwargs):
        super().__init__(*args, **kwargs)
        if mixed_attention_policy not in ("packed_paged", "fa3_hybrid", "fa3_varlen"):
            raise ValueError("unsupported mixed attention policy")
        self.mixed_attention_policy = mixed_attention_policy
        if full_mixed_graph and mixed_attention_policy != "fa3_varlen":
            raise ValueError("full mixed graph currently requires packed varlen FA3")
        self.full_mixed_graph = bool(full_mixed_graph)
        self.clone_decode_metadata = bool(clone_decode_metadata)
        self.full_mixed_graphs = {}
        self.full_mixed_hits = 0
        self.full_mixed_events = []
        self._pending_mixed = None
        self.enable_packed_mixed = True

    @torch.no_grad()
    def __call__(self, ids, positions, slots, cu, context, table, max_query, decode):
        args = (ids, positions, slots, cu, context, table, max_query, decode)
        if decode:
            if self._pending_mixed is not None:
                raise RuntimeError("previous mixed decode was not consumed")
            if (not self.enable_packed_mixed or self.loop is None
                    or not self.loop.current_step_is_mixed()):
                return super().__call__(*args)
            # Default to defensive snapshots. The opt-in no-clone arm relies on
            # C++ keeping decode and prefill in separate tensors and retaining
            # the active decode buffer until this mixed step completes.
            saved = (tuple(tensor.clone() for tensor in args[:6])
                     if self.clone_decode_metadata else args[:6]) + (max_query, True)
            placeholder = torch.empty((ids.numel(), self.model.cfg.vocab),
                                      device=ids.device, dtype=self.model.lm_head.weight.dtype)
            self._pending_mixed = (saved, placeholder)
            self.step_calls.append((True, ids.numel(), context.numel(), max_query))
            return placeholder

        if self._pending_mixed is None:
            return super().__call__(*args)
        decode_args, placeholder = self._pending_mixed
        self._pending_mixed = None
        combined = combine_mixed_metadata(decode_args, args)
        decode_rows = decode_args[0].numel()
        logits = None
        if self.full_mixed_graph:
            key = (combined[0].numel(), combined[3].numel(),
                   combined[5].shape, combined[6])
            graph = self.full_mixed_graphs.get(key)
            capture_ms = 0.0
            outcome = "replay" if graph is not None else "cache_full"
            if graph is None and len(self.full_mixed_graphs) < 4:
                from full_mixed import FullMixedGraph
                started = time.perf_counter()
                graph = FullMixedGraph(
                    self.model, self.pool, *combined,
                    enable_packed_qkv_rope_cache=self.enable_prefill_packed_qkv_rope_cache,
                    enable_residual_rmsnorm=self.enable_prefill_residual_rmsnorm,
                    enable_swiglu_fusion=self.enable_prefill_swiglu_fusion)
                capture_ms = (time.perf_counter() - started) * 1000
                self.full_mixed_graphs[key] = graph
                outcome = "capture"
            if graph is not None:
                logits = graph.forward(*combined)
                if logits is None:
                    outcome = "incompatible"
                else:
                    self.full_mixed_hits += 1
            self.full_mixed_events.append({
                "key": (key[0], key[1], tuple(key[2]), key[3]),
                "decode_tokens": decode_rows,
                "prefill_tokens": combined[0].numel() - decode_rows,
                "outcome": outcome,
                "capture_ms": capture_ms,
            })
        if logits is None:
            logits = self.piecewise_prefill.forward(
                *combined, mixed_decode_count=(decode_rows
                                               if self.mixed_attention_policy == "fa3_hybrid"
                                               else 0),
                mixed_attention_policy=self.mixed_attention_policy)
        if logits is None:
            observer = self.observer
            self.observer = None
            try:
                logits = ModelAdapter.__call__(self, *combined, False)
            finally:
                self.observer = observer
        decode_logits, prefill_logits = logits[:decode_rows], logits[decode_rows:]
        if self.observer is not None:
            checked_decode = self.observer(decode_args, decode_logits)
            if checked_decode is not None:
                decode_logits = checked_decode
            checked_prefill = self.observer(args, prefill_logits)
            if checked_prefill is not None:
                prefill_logits = checked_prefill
        placeholder.copy_(decode_logits)
        self.step_calls.append((False, ids.numel(), context.numel(), max_query))
        action = "packed_mixed_" + self.mixed_attention_policy
        self.decisions[action] = self.decisions.get(action, 0) + 1
        return prefill_logits


class CppPackedMixedModelAdapter(PiecewiseGraphModelAdapter):
    """One C++-assembled mixed callback; no Python metadata concatenation.

    ``step_calls`` retains the two logical cohorts for the existing workload
    checker. The C++ trace has one ``cpp/callback_packed_mixed`` range.
    """

    @torch.no_grad()
    def __call__(self, ids, positions, slots, cu, context, table, max_query, decode):
        if (decode or self.loop is None or not self.loop.current_step_is_mixed()
                or not self.loop.uses_packed_mixed_step()):
            return super().__call__(ids, positions, slots, cu, context,
                                    table, max_query, decode)
        decode_rows = self.loop.current_mixed_decode_rows()
        sequences = context.numel()
        if not 0 < decode_rows < sequences or cu.numel() != sequences + 1:
            raise ValueError("C++ packed mixed metadata has invalid cohort boundaries")
        self.step_calls.extend(((True, decode_rows, decode_rows, 1),
                                (False, ids.numel() - decode_rows,
                                 sequences - decode_rows, max_query)))
        self.decisions["packed_mixed_cpp_varlen"] = (
            self.decisions.get("packed_mixed_cpp_varlen", 0) + 1)
        logits = self.piecewise_prefill.forward(
            ids, positions, slots, cu, context, table, max_query,
            mixed_attention_policy="fa3_varlen")
        if logits is None:
            observer = self.observer
            self.observer = None
            logical_calls = len(self.step_calls)
            try:
                logits = ModelAdapter.__call__(self, ids, positions, slots, cu,
                                               context, table, max_query, False)
            finally:
                self.observer = observer
                del self.step_calls[logical_calls:]
        if self.observer is not None:
            # Reconstruct logical views only for untimed validation. The hot
            # path returns one packed logits tensor without a cat/copy.
            decode_args = (ids[:decode_rows], positions[:decode_rows],
                           slots[:decode_rows], cu[:0], context[:decode_rows],
                           table[:decode_rows], 1, True)
            prefill_args = (ids[decode_rows:], positions[decode_rows:],
                            slots[decode_rows:], cu[decode_rows:] - decode_rows,
                            context[decode_rows:], table[decode_rows:],
                            max_query, False)
            self.observer(decode_args, logits[:decode_rows])
            self.observer(prefill_args, logits[decode_rows:])
        return logits
