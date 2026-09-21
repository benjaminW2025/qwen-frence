"""Optional FlashAttention-3 paged-decode oracle for the vLLM environment.

This adapter deliberately depends on vLLM's bundled FlashAttention extension.
It is an experimental kernel oracle, not an independent production backend.
"""

from __future__ import annotations

import torch


def _load_fa3():
    try:
        from vllm.vllm_flash_attn import flash_attn_with_kvcache
    except (ImportError, AttributeError) as exc:
        try:
            from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_with_kvcache
        except (ImportError, AttributeError) as nested:
            raise RuntimeError(
                "the installed vLLM wheel does not expose its bundled "
                "flash_attn_with_kvcache interface"
            ) from nested
    return flash_attn_with_kvcache


def fa3_paged_decode_attention(
    q,
    k_pool,
    v_pool,
    block_table,
    seq_lens,
    *,
    num_splits=0,
    scale=None,
):
    """Run FA3 directly on the engine's page-16 KV layout without copies."""
    if q.ndim != 3 or k_pool.ndim != 4 or k_pool.shape != v_pool.shape:
        raise ValueError("expected Q rank three and matching rank-four K/V pools")
    if block_table.ndim != 2 or seq_lens.ndim != 1:
        raise ValueError("block table and sequence lengths must have rank two and one")
    if q.shape[0] != block_table.shape[0] or q.shape[0] != seq_lens.shape[0]:
        raise ValueError("batch dimensions must match")
    if q.shape[-1] != k_pool.shape[-1] or q.shape[1] % k_pool.shape[2]:
        raise ValueError("Q/K/V head dimensions or GQA grouping do not match")
    if block_table.dtype != torch.int32 or seq_lens.dtype != torch.int32:
        raise ValueError("FA3 page table and sequence lengths must be int32")
    if not isinstance(num_splits, int) or num_splits < 0:
        raise ValueError("num_splits must be a non-negative integer")
    if scale is None:
        scale = q.shape[-1] ** -0.5

    result = _load_fa3()(
        q=q.unsqueeze(1),
        k_cache=k_pool,
        v_cache=v_pool,
        cache_seqlens=seq_lens,
        block_table=block_table,
        softmax_scale=scale,
        causal=False,
        num_splits=num_splits,
        fa_version=3,
    )
    if isinstance(result, tuple):
        result = result[0]
    return result.squeeze(1)
