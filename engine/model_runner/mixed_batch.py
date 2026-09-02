"""One packed model forward containing decode tokens and prefill chunks.

All tokens share the embedding, projection, output-projection, and MLP launches.  The
attention result is assembled from the two semantics that cannot be merged naively:
single-token paged decode attention and causal packed prefill attention over paged KV.
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
from paged_decode_attention import (
    paged_decode_attention_dispatch,
    resolve_decode_attention_policy,
)
from ragged_prefill import _paged_sdpa_attention, _rope_factors


def _block_table(cache, slots, device):
    max_blocks = max(len(cache.block_tables[slot]) for slot in slots)
    rows = [
        cache.block_tables[slot]
        + [0] * (max_blocks - len(cache.block_tables[slot]))
        for slot in slots
    ]
    return torch.tensor(rows, device=device, dtype=torch.int32)


@torch.no_grad()
def mixed_batch_forward(
    model,
    cache,
    *,
    decode_slots,
    decode_tokens,
    prefill_slots,
    prefill_chunks,
    prefill_starts,
    profile_region=None,
    prefill_attention_backend="triton",
    prefill_tile_policy="static",
    decode_attention_policy="production",
    enable_regime_fusions=False,
):
    """Run one packed pass and return ``(decode_logits, prefill_logits)``.

    ``prefill_logits`` contains the final query position for every scheduled chunk;
    callers must sample only rows whose prompt completed in this iteration.
    """
    region = profile_region or (lambda _name: nullcontext())
    decode_slots = list(decode_slots)
    prefill_slots = list(prefill_slots)
    decode_tokens = list(decode_tokens)
    prefill_chunks = [list(chunk) for chunk in prefill_chunks]
    prefill_starts = list(prefill_starts)
    if not decode_slots or not prefill_slots:
        raise ValueError("mixed_batch_forward requires both decode and prefill work")
    if len(decode_slots) != len(decode_tokens):
        raise ValueError("decode slots and tokens must have equal length")
    if not (
        len(prefill_slots) == len(prefill_chunks) == len(prefill_starts)
    ):
        raise ValueError("prefill slots, chunks, and starts must have equal length")
    if len(set(decode_slots + prefill_slots)) != len(decode_slots) + len(prefill_slots):
        raise ValueError("a cache slot cannot decode and prefill in the same iteration")
    if prefill_attention_backend not in ("triton", "sdpa"):
        raise ValueError("prefill_attention_backend must be 'triton' or 'sdpa'")
    if any(not chunk for chunk in prefill_chunks):
        raise ValueError("prefill chunks must be non-empty")
    for slot, start in zip(prefill_slots, prefill_starts):
        if cache.cur_lens[slot] != start:
            raise ValueError(
                f"prefill slot {slot} starts at {cache.cur_lens[slot]}, expected {start}"
            )

    cfg = model.cfg
    device = cache.device
    n_decode = len(decode_slots)
    prefill_lengths = [len(chunk) for chunk in prefill_chunks]
    n_prefill_tokens = sum(prefill_lengths)
    total_tokens = n_decode + n_prefill_tokens
    n_kv, d_head = cfg.n_kv_heads, cfg.d_head
    block_size = cache.block_size

    with region("mixed/cache_and_input_setup"):
        old_decode_lens = [cache.cur_lens[slot] for slot in decode_slots]
        n_news = [0] * cache.batch_size
        for slot in decode_slots:
            n_news[slot] = 1
        for slot, length in zip(prefill_slots, prefill_lengths):
            n_news[slot] = length
        cache.allocate_block(n_news)

        flat_prefill = [token for chunk in prefill_chunks for token in chunk]
        input_ids = torch.tensor(
            decode_tokens + flat_prefill, device=device, dtype=torch.long
        )
        positions_list = old_decode_lens + [
            position
            for start, length in zip(prefill_starts, prefill_lengths)
            for position in range(start, start + length)
        ]
        positions = torch.tensor(positions_list, device=device, dtype=torch.float32)

        slot_mapping = []
        for slot, start, length in zip(
            decode_slots + prefill_slots,
            old_decode_lens + prefill_starts,
            [1] * n_decode + prefill_lengths,
        ):
            table = cache.block_tables[slot]
            slot_mapping.extend(
                table[position // block_size] * block_size + position % block_size
                for position in range(start, start + length)
            )
        slot_mapping = torch.tensor(slot_mapping, device=device, dtype=torch.long)

        decode_block_table = _block_table(cache, decode_slots, device)
        decode_context_lens = torch.tensor(
            [cache.cur_lens[slot] for slot in decode_slots],
            device=device,
            dtype=torch.int32,
        )
        max_decode_context_length = max(
            cache.cur_lens[slot] for slot in decode_slots
        )
        prefill_block_table = _block_table(cache, prefill_slots, device)
        prefill_context_list = [
            start + length for start, length in zip(prefill_starts, prefill_lengths)
        ]
        prefill_context_lens = torch.tensor(
            prefill_context_list, device=device, dtype=torch.int32
        )
        offsets = [0]
        for length in prefill_lengths:
            offsets.append(offsets[-1] + length)
        cu_seqlens = torch.tensor(offsets, device=device, dtype=torch.int32)
        prefill_ends = torch.tensor(
            [n_decode + end - 1 for end in offsets[1:]],
            device=device,
            dtype=torch.long,
        )

        x = model.embed(input_ids)
        cos, sin = (
            (None, None)
            if cfg.use_custom_kernels
            else _rope_factors(cfg, positions, x.dtype)
        )
        rope_positions = positions[None, :]

    effective_decode_attention_policy = resolve_decode_attention_policy(
        decode_attention_policy, max_decode_context_length
    )
    for layer_index, layer in enumerate(model.layers):
        with region("mixed/layer"):
            residual = x
            with region("mixed/input_rmsnorm"):
                h = apply_rms_norm(x, layer.input_norm, cfg)
            with region("mixed/qkv_projection"):
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
            with region("mixed/rope"):
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

            if fused_q is None:
                with region("mixed/kv_cache_write"):
                    k_flat = cache.k_pool[layer_index].view(-1, n_kv, d_head)
                    v_flat = cache.v_pool[layer_index].view(-1, n_kv, d_head)
                    k_flat.index_copy_(
                        0, slot_mapping, k[0].transpose(0, 1).contiguous()
                    )
                    v_flat.index_copy_(
                        0, slot_mapping, v[0].transpose(0, 1).contiguous()
                    )

            with region("mixed/decode_attention"):
                decode_attention = paged_decode_attention_dispatch(
                    q[0, :, :n_decode, :].transpose(0, 1).contiguous(),
                    cache.k_pool[layer_index],
                    cache.v_pool[layer_index],
                    decode_block_table,
                    decode_context_lens,
                    policy=effective_decode_attention_policy,
                    max_context_length=max_decode_context_length,
                ).transpose(0, 1).unsqueeze(0)

            prefill_q = q[:, :, n_decode:, :]
            with region("mixed/prefill_attention"):
                if prefill_attention_backend == "sdpa":
                    prefill_attention = _paged_sdpa_attention(
                        prefill_q,
                        cache,
                        layer_index,
                        prefill_block_table,
                        prefill_context_list,
                        offsets,
                        cfg.n_heads // n_kv,
                    )
                else:
                    from kernel_dispatch import packed_paged_prefill_attention

                    prefill_attention = packed_paged_prefill_attention(
                        prefill_q,
                        cache.k_pool[layer_index],
                        cache.v_pool[layer_index],
                        cu_seqlens,
                        prefill_block_table,
                        prefill_context_lens,
                        max_query_len=max(prefill_lengths),
                        page_size=block_size,
                        tile_policy=prefill_tile_policy,
                        max_prefix_length=max(prefill_starts),
                    )
            with region("mixed/output_projection"):
                attention = torch.cat((decode_attention, prefill_attention), dim=2)
                attention = attention.transpose(1, 2).reshape(
                    total_tokens, cfg.n_heads * d_head
                )
                x = residual + layer.o_proj(attention)

            residual = x
            with region("mixed/post_attention_rmsnorm"):
                h = apply_rms_norm(x, layer.post_attn_norm, cfg)
            with region("mixed/mlp"):
                h = layer.down_proj(
                    apply_swiglu(
                        layer.gate_proj(h),
                        layer.up_proj(h),
                        cfg,
                        enable_regime_fusions=enable_regime_fusions,
                    )
                )
                x = residual + h

    with region("mixed/final_norm_and_lm_head"):
        x = apply_rms_norm(x, model.norm, cfg)
        decode_logits = model.lm_head(x[:n_decode])
        prefill_logits = model.lm_head(x.index_select(0, prefill_ends))
    return decode_logits, prefill_logits
