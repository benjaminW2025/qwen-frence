"""Packed causal prefill attention for query chunks over a paged KV cache."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


PRODUCTION_TILE = (64, 32, 4, 2)
SHORT_QUERY_TILE = (64, 64, 4, 2)
ADAPTIVE_QUERY_TOKEN_THRESHOLD = 1280
FRESH_QUERY_TOKEN_THRESHOLD = 2048


def select_paged_prefill_tile(
    total_query_tokens, policy="static", *, max_prefix_length=None,
):
    """Select a measured H100 tile without inspecting device data.

    The adaptive rule is fitted to the dense H100 sweep over B=1/2/4,
    Q=64..2048, and prefixes through 16K. It uses only host-known packed query and
    prefix extents, so dispatch adds no device synchronization.
    """
    if total_query_tokens < 1:
        raise ValueError("total_query_tokens must be positive")
    if policy == "static":
        return PRODUCTION_TILE
    if policy == "adaptive":
        use_short_tile = total_query_tokens <= ADAPTIVE_QUERY_TOKEN_THRESHOLD
        if max_prefix_length == 0 and total_query_tokens < FRESH_QUERY_TOKEN_THRESHOLD:
            use_short_tile = True
        return SHORT_QUERY_TILE if use_short_tile else PRODUCTION_TILE
    raise ValueError("tile policy must be 'static' or 'adaptive'")


@triton.jit
def _packed_paged_prefill_attention_kernel(
    q_ptr, k_ptr, v_ptr, offsets_ptr, block_table_ptr, context_lens_ptr, out_ptr,
    stride_qh, stride_qt, stride_qd,
    stride_kt, stride_kh, stride_kd,
    stride_vt, stride_vh, stride_vd,
    stride_bt,
    stride_oh, stride_ot, stride_od,
    scale,
    GROUP: tl.constexpr,
    D_HEAD: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    """One program owns one (sequence, query head, query-chunk tile)."""
    sequence = tl.program_id(0)
    query_head = tl.program_id(1)
    query_block = tl.program_id(2)
    kv_head = query_head // GROUP

    packed_start = tl.load(offsets_ptr + sequence)
    packed_end = tl.load(offsets_ptr + sequence + 1)
    query_length = packed_end - packed_start
    context_length = tl.load(context_lens_ptr + sequence)
    prefix_length = context_length - query_length
    query_start = query_block * BLOCK_M

    offs_m = query_start + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    query_mask = offs_m < query_length
    dim_mask = offs_d < D_HEAD

    q_ptrs = (
        q_ptr
        + query_head * stride_qh
        + (packed_start + offs_m[:, None]) * stride_qt
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=query_mask[:, None] & dim_mask[None, :], other=0.0)

    running_max = tl.full([BLOCK_M], float("-inf"), tl.float32)
    running_sum = tl.zeros([BLOCK_M], tl.float32)
    accumulator = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)

    attended_length = tl.where(
        query_start < query_length,
        tl.minimum(context_length, prefix_length + query_start + BLOCK_M),
        0,
    )
    for kv_block in range(tl.cdiv(attended_length, BLOCK_N)):
        key_positions = kv_block * BLOCK_N + offs_n
        key_mask = key_positions < context_length
        page_ids = tl.load(
            block_table_ptr + sequence * stride_bt + key_positions // PAGE_SIZE,
            mask=key_mask,
            other=0,
        )
        physical_positions = page_ids * PAGE_SIZE + key_positions % PAGE_SIZE

        k_ptrs = (
            k_ptr
            + physical_positions[None, :] * stride_kt
            + kv_head * stride_kh
            + offs_d[:, None] * stride_kd
        )
        k = tl.load(k_ptrs, mask=dim_mask[:, None] & key_mask[None, :], other=0.0)
        scores = tl.dot(q, k) * scale

        absolute_queries = prefix_length + offs_m
        causal_mask = key_positions[None, :] <= absolute_queries[:, None]
        score_mask = query_mask[:, None] & key_mask[None, :] & causal_mask
        scores = tl.where(score_mask, scores, float("-inf"))
        scores = tl.where(query_mask[:, None], scores, 0.0)

        block_max = tl.max(scores, axis=1)
        new_max = tl.maximum(running_max, block_max)
        correction = tl.exp(running_max - new_max)
        probabilities = tl.exp(scores - new_max[:, None])
        probabilities = tl.where(score_mask, probabilities, 0.0)

        v_ptrs = (
            v_ptr
            + physical_positions[:, None] * stride_vt
            + kv_head * stride_vh
            + offs_d[None, :] * stride_vd
        )
        v = tl.load(v_ptrs, mask=key_mask[:, None] & dim_mask[None, :], other=0.0)
        probabilities_for_dot = (
            probabilities.to(tl.bfloat16) if IS_BF16 else probabilities.to(tl.float16)
        )
        accumulator = accumulator * correction[:, None] + tl.dot(
            probabilities_for_dot, v
        )
        running_sum = running_sum * correction + tl.sum(probabilities, axis=1)
        running_max = new_max

    output = accumulator / running_sum[:, None]
    out_ptrs = (
        out_ptr
        + query_head * stride_oh
        + (packed_start + offs_m[:, None]) * stride_ot
        + offs_d[None, :] * stride_od
    )
    tl.store(
        out_ptrs,
        output.to(out_ptr.dtype.element_ty),
        mask=query_mask[:, None] & dim_mask[None, :],
    )


def packed_paged_prefill_attention_triton(
    q,
    k_pool,
    v_pool,
    cu_seqlens,
    block_table,
    context_lens,
    max_query_len,
    page_size,
    scale=None,
    block_m=64,
    block_n=32,
    num_warps=4,
    num_stages=2,
    tile_policy="static",
    max_prefix_length=None,
):
    """Attend packed query chunks to their complete logical paged-KV contexts."""
    if q.ndim != 4 or q.shape[0] != 1:
        raise ValueError("q must have shape (1, query_heads, total_query_tokens, head_dim)")
    if k_pool.ndim != 4 or k_pool.shape != v_pool.shape:
        raise ValueError("k_pool and v_pool must have equal (pages, page, heads, dim) shapes")
    if block_table.ndim != 2 or context_lens.ndim != 1:
        raise ValueError("block_table and context_lens must be rank two and one")
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() != context_lens.numel() + 1:
        raise ValueError("cu_seqlens must contain one more entry than context_lens")
    if block_table.shape[0] != context_lens.numel():
        raise ValueError("block_table rows must match the number of query sequences")
    if q.shape[-1] != k_pool.shape[-1] or q.shape[1] % k_pool.shape[2] != 0:
        raise ValueError("query and KV head dimensions are incompatible")
    if page_size != k_pool.shape[1]:
        raise ValueError("page_size must match the KV-pool page dimension")
    tensors = (q, k_pool, v_pool, cu_seqlens, block_table, context_lens)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("packed paged prefill attention requires CUDA tensors")
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("all packed paged prefill tensors must share a device")
    if q.dtype not in (torch.float16, torch.bfloat16) or k_pool.dtype != q.dtype or v_pool.dtype != q.dtype:
        raise ValueError("q and KV pools must share a float16 or bfloat16 dtype")
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError("cu_seqlens must have an integer dtype")
    if block_table.dtype not in (torch.int32, torch.int64) or context_lens.dtype not in (torch.int32, torch.int64):
        raise ValueError("block_table and context_lens must have integer dtypes")
    if max_query_len < 1:
        raise ValueError("max_query_len must be positive")

    if tile_policy == "adaptive":
        block_m, block_n, num_warps, num_stages = select_paged_prefill_tile(
            q.shape[2], tile_policy, max_prefix_length=max_prefix_length
        )
    elif tile_policy != "static":
        raise ValueError("tile policy must be 'static' or 'adaptive'")

    _, query_heads, _, d_head = q.shape
    kv_heads = k_pool.shape[2]
    if scale is None:
        scale = 1.0 / (d_head ** 0.5)
    out = torch.empty_like(q)
    grid = (context_lens.numel(), query_heads, triton.cdiv(max_query_len, block_m))
    _packed_paged_prefill_attention_kernel[grid](
        q, k_pool, v_pool, cu_seqlens, block_table, context_lens, out,
        q.stride(1), q.stride(2), q.stride(3),
        k_pool.stride(1), k_pool.stride(2), k_pool.stride(3),
        v_pool.stride(1), v_pool.stride(2), v_pool.stride(3),
        block_table.stride(0),
        out.stride(1), out.stride(2), out.stride(3),
        scale,
        GROUP=query_heads // kv_heads,
        D_HEAD=d_head,
        BLOCK_D=triton.next_power_of_2(d_head),
        PAGE_SIZE=page_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        IS_BF16=q.dtype == torch.bfloat16,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
