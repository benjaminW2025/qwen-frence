"""Real Qwen forward from scheduler-owned GPU metadata; no second KV allocator."""
from __future__ import annotations
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
            q = layer.q_proj(h).view(tokens, heads, dim).transpose(0, 1).unsqueeze(0)
            k = layer.k_proj(h).view(tokens, kvheads, dim).transpose(0, 1).unsqueeze(0)
            v = layer.v_proj(h).view(tokens, kvheads, dim).transpose(0, 1).unsqueeze(0)
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
            x = residual + layer.down_proj(apply_swiglu(layer.gate_proj(h), layer.up_proj(h), cfg))
        x = apply_rms_norm(x, model.norm, cfg)
        if not decode:
            x = x.index_select(0, cu[1:].to(torch.long) - 1)
        logits = model.lm_head(x)
        if self.observer is not None:
            self.observer((ids, positions, slots, cu, context, table, max_query, decode), logits)
        return logits
