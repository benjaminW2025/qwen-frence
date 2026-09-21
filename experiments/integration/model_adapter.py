"""Real Qwen forward from scheduler-owned GPU metadata; no second KV allocator."""
from __future__ import annotations
from pathlib import Path
import sys
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
                 enable_native_decode_qkv_postprocess=False,
                 enable_fused_qkv_rope_cache=False,
                 enable_packed_qkv_rope_cache=False):
        super().__init__(model, pool, loop)
        if decode_attention_policy not in ("production", "splitk", "fa3"):
            raise ValueError("graph decode policy must be 'production', 'splitk', or 'fa3'")
        graph_dir = Path(__file__).resolve().parents[2] / "engine/graph"
        if str(graph_dir) not in sys.path:
            sys.path.insert(0, str(graph_dir))
        from bucketed_graph_decoder import BucketedGraphDecoder

        self.decode_attention_policy = decode_attention_policy
        self.enable_residual_rmsnorm = bool(enable_residual_rmsnorm)
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
            enable_native_decode_qkv_postprocess=(
                self.enable_native_decode_qkv_postprocess
            ),
            enable_fused_qkv_rope_cache=self.enable_fused_qkv_rope_cache,
            enable_packed_qkv_rope_cache=self.enable_packed_qkv_rope_cache,
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
                 enable_native_decode_qkv_postprocess=False,
                 enable_fused_qkv_rope_cache=False,
                 enable_packed_qkv_rope_cache=False):
        super().__init__(model, pool, loop, max_running=max_running,
                         max_context_length=max_context_length,
                         decode_attention_policy=decode_attention_policy,
                         decode_buckets=decode_buckets,
                         enable_residual_rmsnorm=enable_residual_rmsnorm,
                         enable_native_decode_qkv_postprocess=(
                             enable_native_decode_qkv_postprocess
                         ),
                         enable_fused_qkv_rope_cache=enable_fused_qkv_rope_cache,
                         enable_packed_qkv_rope_cache=enable_packed_qkv_rope_cache)
        graph_dir = Path(__file__).resolve().parents[2] / "engine/graph"
        if str(graph_dir) not in sys.path:
            sys.path.insert(0, str(graph_dir))
        from piecewise_prefill import PiecewisePrefill
        self.piecewise_prefill = PiecewisePrefill(
            model, pool, max_capture_tokens=max_capture_tokens,
            max_shapes=max_prefill_shapes, token_buckets=prefill_buckets)

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
