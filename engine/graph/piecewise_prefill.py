"""Token-bucket CUDA graphs spanning the work between eager attentions.

For N layers there are N+1 graphs: embedding/pre-attention for layer zero,
post-attention of layer i plus pre-attention of layer i+1, and one final
post-attention graph. Packed paged attention remains eager with live ragged
metadata and launch geometry. Each graph retains its fixed-address inputs and
outputs across replays.
"""

from __future__ import annotations

import torch

from naive_forward import apply_rms_norm, apply_rope, apply_swiglu
from ragged_prefill import _rope_factors


class _AttentionBoundary:
    """One captured segment ending at (or starting just after) an attention call."""

    def __init__(self, model, pool, index, tokens, positions, slots, valid_tokens,
                 enable_packed_qkv_rope_cache=False,
                 enable_residual_rmsnorm=False, enable_swiglu_fusion=False,
                 capture_graph=True):
        cfg = model.cfg
        device, dtype = pool.k_pool[0].device, pool.k_pool[0].dtype
        layers = model.layers
        self.index = index
        self.last = index == len(layers)
        self.enable_packed_qkv_rope_cache = bool(enable_packed_qkv_rope_cache)
        self.enable_residual_rmsnorm = bool(enable_residual_rmsnorm)
        self.enable_swiglu_fusion = bool(enable_swiglu_fusion)
        if index == 0:
            self.ids = torch.zeros(tokens, device=device, dtype=torch.long)
        else:
            self.residual = torch.zeros((tokens, cfg.d_model), device=device, dtype=dtype)
            self.attention = torch.zeros((tokens, cfg.n_heads * cfg.d_head),
                                         device=device, dtype=dtype)
        if not self.last:
            self.positions = positions
            self.slots = slots
            self.valid_tokens = valid_tokens

        def run_segment():
            if index == 0:
                x = model.embed(self.ids)
                next_h = None
            else:
                previous = layers[index - 1]
                projected = previous.o_proj(self.attention)
                if self.enable_residual_rmsnorm:
                    from kernel_dispatch import residual_add_rms_norm
                    x, h = residual_add_rms_norm(
                        self.residual, projected,
                        previous.post_attn_norm.weight, cfg.rms_norm_eps,
                    )
                else:
                    x = self.residual + projected
                    h = apply_rms_norm(x, previous.post_attn_norm, cfg)
                gate, up = previous.project_gate_up(h)
                branch = previous.down_proj(apply_swiglu(
                    gate, up, cfg, enable_regime_fusions=self.enable_swiglu_fusion))
                if self.enable_residual_rmsnorm:
                    next_norm = model.norm if self.last else layers[index].input_norm
                    x, next_h = residual_add_rms_norm(
                        x, branch, next_norm.weight, cfg.rms_norm_eps,
                    )
                else:
                    x = x + branch
                    next_h = None
            if self.last:
                # The fused path has already applied the final model RMSNorm.
                return next_h if self.enable_residual_rmsnorm else x

            layer = layers[index]
            h = (next_h if self.enable_residual_rmsnorm and index > 0 else
                 apply_rms_norm(x, layer.input_norm, cfg))
            if self.enable_packed_qkv_rope_cache:
                from kernel_dispatch import packed_qkv_rope_cache
                if layer.qkv_proj is None:
                    raise ValueError("packed prefill QKV epilogue requires packed QKV weights")
                q_rows = packed_qkv_rope_cache(
                    layer.qkv_proj(h), self.positions, self.slots,
                    pool.k_pool[index], pool.v_pool[index],
                    base=cfg.rope_theta, valid_tokens=self.valid_tokens,
                )
                q = q_rows.transpose(0, 1).unsqueeze(0)
            else:
                q, k, v = layer.project_qkv(h)
                q = q.view(tokens, cfg.n_heads, cfg.d_head).transpose(0, 1).unsqueeze(0)
                k = k.view(tokens, cfg.n_kv_heads, cfg.d_head).transpose(0, 1).unsqueeze(0)
                v = v.view(tokens, cfg.n_kv_heads, cfg.d_head).transpose(0, 1).unsqueeze(0)
                cos, sin = ((None, None) if cfg.use_custom_kernels else
                            _rope_factors(cfg, self.positions, x.dtype))
                q = apply_rope(q, cos, sin, cfg, self.positions[None, :])
                k = apply_rope(k, cos, sin, cfg, self.positions[None, :])
                from kernel_dispatch import masked_kv_write
                masked_kv_write(k, v, self.slots, self.valid_tokens,
                                pool.k_pool[index], pool.v_pool[index])
            return q, x

        self.run_segment = run_segment
        if not capture_graph:
            return
        # Warm up on a side stream for allocator setup and Triton JIT, then
        # capture. The caller's stream waits before the first replay.
        current = torch.cuda.current_stream(device)
        side = torch.cuda.Stream(device=device)
        side.wait_stream(current)
        with torch.cuda.stream(side):
            for _ in range(3):
                run_segment()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=side):
                self.result = run_segment()
        current.wait_stream(side)

    def run_initial(self, ids, tokens):
        self.ids[:tokens].copy_(ids)
        self.graph.replay()
        return self.result

    def run_from_attention(self, residual, attention, tokens):
        self.residual.copy_(residual)
        self.attention[:tokens].copy_(attention)
        self.graph.replay()
        return self.result


class PiecewisePrefill:
    """Capture a bounded set of packed-token buckets; report eager misses."""

    def __init__(self, model, pool, *, max_capture_tokens=2048, max_shapes=8,
                 token_buckets=None, enable_packed_qkv_rope_cache=False,
                 enable_residual_rmsnorm=False, enable_swiglu_fusion=False):
        if max_capture_tokens < 1 or max_shapes < 1:
            raise ValueError("capture token and shape limits must be positive")
        if pool.k_pool[0].device.type != "cuda":
            raise ValueError("piecewise prefill requires CUDA")
        self.model, self.pool = model, pool
        self.enable_packed_qkv_rope_cache = bool(enable_packed_qkv_rope_cache)
        self.enable_residual_rmsnorm = bool(enable_residual_rmsnorm)
        self.enable_swiglu_fusion = bool(enable_swiglu_fusion)
        self.max_capture_tokens, self.max_shapes = max_capture_tokens, max_shapes
        if token_buckets is None:
            buckets = [b for b in (128, 256, 512, 1024, 2048) if b <= max_capture_tokens]
            buckets.append(max_capture_tokens)
        else:
            buckets = list(token_buckets)
            if not buckets or any(not isinstance(b, int) or b < 1 or b > max_capture_tokens
                                  for b in buckets):
                raise ValueError("token buckets must be positive integers within the capture limit")
        self.buckets = tuple(sorted(set(buckets)))
        self.shapes = {}
        self.captured_calls = 0
        self.eager_calls = 0
        self.graph_replays = 0

    def pieces(self, tokens):
        bucket = next((b for b in self.buckets if b >= tokens), None)
        if bucket is None or (bucket not in self.shapes and
                              len(self.shapes) >= self.max_shapes):
            self.eager_calls += 1
            return None
        if bucket not in self.shapes:
            device = self.pool.k_pool[0].device
            positions = torch.zeros(bucket, device=device, dtype=torch.long)
            slots = torch.zeros(bucket, device=device, dtype=torch.long)
            valid_tokens = torch.zeros((), device=device, dtype=torch.int32)
            self.shapes[bucket] = [
                _AttentionBoundary(self.model, self.pool, i, bucket,
                                   positions, slots, valid_tokens,
                                   getattr(self, "enable_packed_qkv_rope_cache", False),
                                   getattr(self, "enable_residual_rmsnorm", False),
                                   getattr(self, "enable_swiglu_fusion", False))
                for i in range(len(self.model.layers) + 1)
            ]
        self.captured_calls += 1
        return self.shapes[bucket]

    @torch.no_grad()
    def forward(self, ids, positions, slots, cu, context, table, max_query,
                *, mixed_decode_count=0, mixed_attention_policy="packed_paged"):
        from kernel_dispatch import packed_paged_prefill_attention

        cfg, model, pool = self.model.cfg, self.model, self.pool
        tokens = ids.numel()
        pieces = self.pieces(tokens)
        if pieces is None:
            return None
        pieces[0].positions[:tokens].copy_(positions)
        pieces[0].slots[:tokens].copy_(slots)
        pieces[0].valid_tokens.fill_(tokens)
        q, residual = pieces[0].run_initial(ids, tokens)
        self.graph_replays += 1
        for i in range(len(model.layers)):
            if mixed_attention_policy == "fa3_varlen":
                from kernel_dispatch import fa3_paged_varlen_attention
                query = q[0].transpose(0, 1).contiguous()
                rows = fa3_paged_varlen_attention(
                    query, pool.k_pool[i], pool.v_pool[i], cu, table, context,
                    max_query_len=max_query)
                attention = rows.transpose(0, 1).unsqueeze(0)
            elif mixed_decode_count:
                from kernel_dispatch import fa3_paged_decode_attention
                decode_q = q[0, :, :mixed_decode_count, :].transpose(0, 1).contiguous()
                decode_attention = fa3_paged_decode_attention(
                    decode_q, pool.k_pool[i], pool.v_pool[i],
                    table[:mixed_decode_count], context[:mixed_decode_count],
                ).transpose(0, 1).unsqueeze(0)
                prefill_attention = packed_paged_prefill_attention(
                    q[:, :, mixed_decode_count:tokens, :], pool.k_pool[i], pool.v_pool[i],
                    cu[mixed_decode_count:] - mixed_decode_count,
                    table[mixed_decode_count:], context[mixed_decode_count:],
                    max_query_len=max_query, page_size=pool.block_size,
                    tile_policy="static")
                attention = torch.cat((decode_attention, prefill_attention), dim=2)
            else:
                attention = packed_paged_prefill_attention(
                    q[:, :, :tokens, :], pool.k_pool[i], pool.v_pool[i], cu, table, context,
                    max_query_len=max_query, page_size=pool.block_size,
                    tile_policy="static")
            attention = attention.transpose(1, 2).reshape(tokens, cfg.n_heads * cfg.d_head)
            result = pieces[i + 1].run_from_attention(residual, attention, tokens)
            self.graph_replays += 1
            if i + 1 < len(model.layers):
                q, residual = result
            else:
                x = result
        if not self.enable_residual_rmsnorm:
            x = apply_rms_norm(x, model.norm, cfg)
        return model.lm_head(x.index_select(0, cu[1:].to(torch.long) - 1))
