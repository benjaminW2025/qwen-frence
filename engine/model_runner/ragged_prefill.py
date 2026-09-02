"""Packed ragged model forward for multiple admitted or resumed requests.

Non-attention work runs over one flattened token dimension, giving GEMMs and MLPs a
single larger workload. Variable-length attention uses cumulative sequence offsets to
process the packed batch in one launch without allowing cross-request attention.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch

from naive_forward import (
    apply_packed_rope_kv_write,
    apply_rms_norm,
    apply_rope,
    apply_swiglu,
)
from packed_prefill_attention import packed_prefill_attention


def _rope_factors(cfg, positions, dtype):
    """Build factors for arbitrary flattened positions, shaped for (1, H, T, D)."""
    inv_freq = 1.0 / (
        cfg.rope_theta
        ** (
            torch.arange(
                0,
                cfg.d_head,
                2,
                device=positions.device,
                dtype=torch.float32,
            )
            / cfg.d_head
        )
    )
    freqs = positions.to(torch.float32)[:, None] * inv_freq[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos()[None, None].to(dtype), emb.sin()[None, None].to(dtype)


def _validate(cache, slots, prompt_ids, prompt_starts):
    if not slots or len(slots) != len(prompt_ids):
        raise ValueError("slots and prompt_ids must be non-empty and equally sized")
    if len(prompt_starts) != len(prompt_ids):
        raise ValueError("prompt_starts must match prompt_ids")
    if len(set(slots)) != len(slots):
        raise ValueError("ragged prefill slots must be unique")
    for slot, tokens, start in zip(slots, prompt_ids, prompt_starts):
        if not 0 <= slot < cache.batch_size:
            raise ValueError(f"slot {slot} is outside cache batch size {cache.batch_size}")
        if start < 0 or cache.cur_lens[slot] != start:
            raise ValueError(
                f"slot {slot} starts at {cache.cur_lens[slot]}, expected {start}"
            )
        if start == 0 and cache.block_tables[slot]:
            raise ValueError(f"new prefill slot {slot} must not own cache blocks")
        if not tokens:
            raise ValueError("prompt chunks must contain at least one token")


def _paged_sdpa_attention(q, cache, layer_index, block_table, context_lens, offsets, group):
    """Correctness/reference path for query chunks attending through paged KV."""
    outputs = []
    scale = q.shape[-1] ** -0.5
    for row, context_length in enumerate(context_lens):
        query_start, query_end = offsets[row:row + 2]
        query_length = query_end - query_start
        page_count = (context_length + cache.block_size - 1) // cache.block_size
        pages = block_table[row, :page_count].to(torch.long)
        keys = cache.k_pool[layer_index].index_select(0, pages).reshape(
            -1, cache.n_kv_heads, cache.d_head
        )[:context_length]
        values = cache.v_pool[layer_index].index_select(0, pages).reshape(
            -1, cache.n_kv_heads, cache.d_head
        )[:context_length]
        keys = keys.transpose(0, 1).repeat_interleave(group, dim=0).unsqueeze(0)
        values = values.transpose(0, 1).repeat_interleave(group, dim=0).unsqueeze(0)
        query = q[:, :, query_start:query_end]
        prefix_length = context_length - query_length
        query_positions = prefix_length + torch.arange(query_length, device=q.device)
        key_positions = torch.arange(context_length, device=q.device)
        causal_mask = key_positions[None, :] <= query_positions[:, None]
        outputs.append(
            torch.nn.functional.scaled_dot_product_attention(
                query, keys, values, attn_mask=causal_mask, scale=scale
            )
        )
    return torch.cat(outputs, dim=2)


@torch.no_grad()
def ragged_prefill(
    model,
    cache,
    slots,
    prompt_ids,
    prompt_starts=None,
    profile_region=None,
    attention_backend="triton",
    prefill_tile_policy="static",
    enable_regime_fusions=False,
):
    """Prefill new ragged requests and return final-position logits shaped (B, vocab)."""
    region = profile_region or (lambda _name: nullcontext())
    prompt_starts = [0] * len(prompt_ids) if prompt_starts is None else list(prompt_starts)
    _validate(cache, slots, prompt_ids, prompt_starts)
    cfg = model.cfg
    device = cache.device
    lengths = [len(tokens) for tokens in prompt_ids]
    total_tokens = sum(lengths)
    n_kv, d_head = cfg.n_kv_heads, cfg.d_head
    group = cfg.n_heads // n_kv
    block_size = cache.block_size

    with region("prefill/cache_and_input_setup"):
        n_news = [0] * cache.batch_size
        for slot, length in zip(slots, lengths):
            n_news[slot] = length
        cache.allocate_block(n_news)

        flat_ids = [token for tokens in prompt_ids for token in tokens]
        flat_positions = [
            position
            for start, length in zip(prompt_starts, lengths)
            for position in range(start, start + length)
        ]
        flat_slots = []
        for slot, start, length in zip(slots, prompt_starts, lengths):
            table = cache.block_tables[slot]
            flat_slots.extend(
                table[position // block_size] * block_size + (position % block_size)
                for position in range(start, start + length)
            )

        input_ids = torch.tensor(flat_ids, device=device, dtype=torch.long)
        positions = torch.tensor(flat_positions, device=device, dtype=torch.float32)
        slot_mapping = torch.tensor(flat_slots, device=device, dtype=torch.long)
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        cu_seqlens = torch.tensor(offsets, device=device, dtype=torch.int32)
        ends = cu_seqlens[1:].to(torch.long) - 1
        context_lens_list = [start + length for start, length in zip(prompt_starts, lengths)]
        context_lens = torch.tensor(context_lens_list, device=device, dtype=torch.int32)
        max_blocks = max(len(cache.block_tables[slot]) for slot in slots)
        padded_tables = [
            cache.block_tables[slot] + [0] * (max_blocks - len(cache.block_tables[slot]))
            for slot in slots
        ]
        block_table = torch.tensor(padded_tables, device=device, dtype=torch.int32)

        x = model.embed(input_ids)  # (T, d_model)
        cos, sin = (
            (None, None)
            if cfg.use_custom_kernels
            else _rope_factors(cfg, positions, x.dtype)
        )
        rope_positions = positions[None, :]

    for layer_index, layer in enumerate(model.layers):
        with region("prefill/layer"):
            residual = x
            with region("prefill/input_rmsnorm"):
                h = apply_rms_norm(x, layer.input_norm, cfg)
            with region("prefill/qkv_projection"):
                q = (
                    layer.q_proj(h)
                    .view(total_tokens, cfg.n_heads, d_head)
                    .transpose(0, 1)
                    .unsqueeze(0)
                )
                k = (
                    layer.k_proj(h)
                    .view(total_tokens, n_kv, d_head)
                    .transpose(0, 1)
                    .unsqueeze(0)
                )
                v = (
                    layer.v_proj(h)
                    .view(total_tokens, n_kv, d_head)
                    .transpose(0, 1)
                    .unsqueeze(0)
                )
            fused_cache_write = False
            with region("prefill/rope"):
                # A resumed prefill reads K/V back from the paged cache, so RoPE and
                # placement can be fused. Fresh attention still needs rotated K as a
                # local tensor and deliberately retains the separate reference path.
                fused_q = None
                if any(prompt_starts):
                    fused_q = apply_packed_rope_kv_write(
                        q,
                        k,
                        v,
                        rope_positions,
                        slot_mapping,
                        cache.k_pool[layer_index],
                        cache.v_pool[layer_index],
                        cfg,
                        enable_regime_fusions=enable_regime_fusions,
                    )
                if fused_q is None:
                    q = apply_rope(q, cos, sin, cfg, rope_positions)
                    k = apply_rope(k, cos, sin, cfg, rope_positions)
                else:
                    q = fused_q
                    fused_cache_write = True

            if not fused_cache_write:
                with region("prefill/kv_cache_write"):
                    k_flat = cache.k_pool[layer_index].view(-1, n_kv, d_head)
                    v_flat = cache.v_pool[layer_index].view(-1, n_kv, d_head)
                    k_flat.index_copy_(
                        0, slot_mapping, k[0].transpose(0, 1).contiguous()
                    )
                    v_flat.index_copy_(
                        0, slot_mapping, v[0].transpose(0, 1).contiguous()
                    )

            with region("prefill/attention"):
                if any(prompt_starts):
                    if attention_backend == "sdpa":
                        attention = _paged_sdpa_attention(
                            q,
                            cache,
                            layer_index,
                            block_table,
                            context_lens_list,
                            offsets,
                            group,
                        )
                    else:
                        from kernel_dispatch import packed_paged_prefill_attention

                        attention = packed_paged_prefill_attention(
                            q,
                            cache.k_pool[layer_index],
                            cache.v_pool[layer_index],
                            cu_seqlens,
                            block_table,
                            context_lens,
                            max_query_len=max(lengths),
                            page_size=block_size,
                            tile_policy=prefill_tile_policy,
                            max_prefix_length=max(prompt_starts),
                        )
                else:
                    attention = packed_prefill_attention(
                        q,
                        k,
                        v,
                        offsets if attention_backend == "sdpa" else cu_seqlens,
                        max_seqlen=max(lengths),
                        group=group,
                        backend=attention_backend,
                    )
                attention = attention.transpose(1, 2).reshape(
                    total_tokens, cfg.n_heads * d_head
                )
            with region("prefill/output_projection"):
                x = residual + layer.o_proj(attention)

            residual = x
            with region("prefill/post_attention_rmsnorm"):
                h = apply_rms_norm(x, layer.post_attn_norm, cfg)
            with region("prefill/mlp"):
                h = layer.down_proj(
                    apply_swiglu(
                        layer.gate_proj(h),
                        layer.up_proj(h),
                        cfg,
                        enable_regime_fusions=enable_regime_fusions,
                    )
                )
                x = residual + h

    with region("prefill/final_norm_and_lm_head"):
        x = apply_rms_norm(x, model.norm, cfg)
        return model.lm_head(x.index_select(0, ends))
