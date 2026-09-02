"""Tuned paged-decode candidate, including experimental GQA-head sharing."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


DECODE_LOW_BATCH_THRESHOLD = 16
DECODE_HIGH_BATCH_THRESHOLD = 64
DECODE_LOW_BATCH_KV_TOKEN_THRESHOLD = 32768
DECODE_HIGH_BATCH_KV_TOKEN_THRESHOLD = 16384


def select_paged_decode_candidate_config(batch_size, max_context_length, policy="adaptive"):
    """Return the measured H100 candidate config, or ``None`` for production.

    Head sharing itself did not win the 30-shape sweep. The selected candidate keeps
    one query head per program and only changes the program shape/warp configuration.
    """
    if batch_size < 1 or max_context_length < 1:
        raise ValueError("batch size and context length must be positive")
    if policy == "production":
        return None
    if policy != "adaptive":
        raise ValueError("decode attention policy must be 'production' or 'adaptive'")
    total_kv_tokens = batch_size * max_context_length
    if batch_size >= DECODE_HIGH_BATCH_THRESHOLD:
        use_candidate = True
    elif batch_size >= DECODE_LOW_BATCH_THRESHOLD:
        use_candidate = total_kv_tokens >= DECODE_HIGH_BATCH_KV_TOKEN_THRESHOLD
    else:
        use_candidate = total_kv_tokens >= DECODE_LOW_BATCH_KV_TOKEN_THRESHOLD
    if not use_candidate:
        return None
    return {
        "heads_per_program": 1,
        "num_warps": 8 if batch_size < DECODE_LOW_BATCH_THRESHOLD else 4,
        "num_stages": 2,
    }


@triton.jit
def _grouped_gqa_decode_kernel(
    q_ptr, k_pool_ptr, v_pool_ptr, block_table_ptr, seq_lens_ptr, out_ptr,
    stride_qb, stride_qh, stride_qd,
    stride_pblk, stride_pt, stride_pkv, stride_pd,
    stride_btb, stride_btm,
    stride_ob, stride_oh, stride_od,
    scale,
    GROUP: tl.constexpr,
    HEADS_PER_PROGRAM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    sequence = tl.program_id(0)
    kv_head = tl.program_id(1)
    head_tile = tl.program_id(2)

    head_offsets = tl.arange(0, BLOCK_H)
    dim_offsets = tl.arange(0, D_HEAD)
    token_offsets = tl.arange(0, PAGE_SIZE)
    heads_in_group = head_tile * HEADS_PER_PROGRAM + head_offsets
    head_mask = (
        (head_offsets < HEADS_PER_PROGRAM)
        & (heads_in_group < GROUP)
    )
    query_heads = kv_head * GROUP + heads_in_group
    q = tl.load(
        q_ptr
        + sequence * stride_qb
        + query_heads[:, None] * stride_qh
        + dim_offsets[None, :] * stride_qd,
        mask=head_mask[:, None],
        other=0.0,
    )

    seq_len = tl.load(seq_lens_ptr + sequence)
    running_max = tl.full([BLOCK_H], float("-inf"), tl.float32)
    running_sum = tl.zeros([BLOCK_H], tl.float32)
    accumulator = tl.zeros([BLOCK_H, D_HEAD], tl.float32)

    for page_index in range(tl.cdiv(seq_len, PAGE_SIZE)):
        page_id = tl.load(
            block_table_ptr
            + sequence * stride_btb
            + page_index * stride_btm
        )
        positions = page_index * PAGE_SIZE + token_offsets
        token_mask = positions < seq_len
        base = page_id * stride_pblk + kv_head * stride_pkv
        pool_offsets = (
            token_offsets[:, None] * stride_pt
            + dim_offsets[None, :] * stride_pd
        )
        keys = tl.load(
            k_pool_ptr + base + pool_offsets,
            mask=token_mask[:, None],
            other=0.0,
        )
        values = tl.load(
            v_pool_ptr + base + pool_offsets,
            mask=token_mask[:, None],
            other=0.0,
        )

        scores = tl.sum(
            q[:, None, :].to(tl.float32)
            * keys[None, :, :].to(tl.float32),
            axis=2,
        ) * scale
        score_mask = head_mask[:, None] & token_mask[None, :]
        scores = tl.where(score_mask, scores, float("-inf"))
        scores = tl.where(head_mask[:, None], scores, 0.0)

        page_max = tl.max(scores, axis=1)
        new_max = tl.maximum(running_max, page_max)
        correction = tl.exp(running_max - new_max)
        probabilities = tl.exp(scores - new_max[:, None])
        probabilities = tl.where(score_mask, probabilities, 0.0)
        accumulator = (
            accumulator * correction[:, None]
            + tl.sum(
                probabilities[:, :, None] * values[None, :, :],
                axis=1,
            )
        )
        running_sum = (
            running_sum * correction + tl.sum(probabilities, axis=1)
        )
        running_max = new_max

    output = accumulator / running_sum[:, None]
    tl.store(
        out_ptr
        + sequence * stride_ob
        + query_heads[:, None] * stride_oh
        + dim_offsets[None, :] * stride_od,
        output.to(out_ptr.dtype.element_ty),
        mask=head_mask[:, None],
    )


def paged_decode_attention_candidate(
    q,
    k_pool,
    v_pool,
    block_table,
    seq_lens,
    *,
    heads_per_program,
    scale=None,
    num_warps=8,
    num_stages=2,
):
    """Attend one decode query while sharing each K/V load across GQA heads."""
    if q.ndim != 3:
        raise ValueError("q must have shape (batch, query_heads, head_dim)")
    if k_pool.ndim != 4 or k_pool.shape != v_pool.shape:
        raise ValueError("K/V pools must have matching rank-four shapes")
    if block_table.ndim != 2 or seq_lens.ndim != 1:
        raise ValueError("block table and sequence lengths must be rank two and one")
    if q.shape[0] != seq_lens.numel() or block_table.shape[0] != q.shape[0]:
        raise ValueError("batch dimensions must match")
    if q.shape[-1] != k_pool.shape[-1]:
        raise ValueError("query and KV head dimensions must match")
    query_heads = q.shape[1]
    kv_heads = k_pool.shape[2]
    if query_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    group = query_heads // kv_heads
    if heads_per_program not in (1, 2, 3, 6) or group % heads_per_program:
        raise ValueError("heads_per_program must be 1, 2, 3, or 6 and divide GROUP")
    tensors = (q, k_pool, v_pool, block_table, seq_lens)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("grouped decode requires CUDA tensors")
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("all tensors must share a device")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("grouped decode supports float16 and bfloat16")
    if k_pool.dtype != q.dtype or v_pool.dtype != q.dtype:
        raise ValueError("Q/K/V dtypes must match")
    if block_table.dtype not in (torch.int32, torch.int64):
        raise ValueError("block table must use an integer dtype")
    if seq_lens.dtype not in (torch.int32, torch.int64):
        raise ValueError("sequence lengths must use an integer dtype")
    if num_warps not in (1, 2, 4, 8):
        raise ValueError("num_warps must be 1, 2, 4, or 8")
    if num_stages < 1:
        raise ValueError("num_stages must be positive")

    if scale is None:
        scale = q.shape[-1] ** -0.5
    output = torch.empty_like(q)
    block_h = triton.next_power_of_2(heads_per_program)
    grid = (q.shape[0], kv_heads, triton.cdiv(group, heads_per_program))
    _grouped_gqa_decode_kernel[grid](
        q, k_pool, v_pool, block_table, seq_lens, output,
        q.stride(0), q.stride(1), q.stride(2),
        k_pool.stride(0), k_pool.stride(1), k_pool.stride(2), k_pool.stride(3),
        block_table.stride(0), block_table.stride(1),
        output.stride(0), output.stride(1), output.stride(2),
        scale,
        GROUP=group,
        HEADS_PER_PROGRAM=heads_per_program,
        BLOCK_H=block_h,
        PAGE_SIZE=k_pool.shape[1],
        D_HEAD=q.shape[-1],
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output
