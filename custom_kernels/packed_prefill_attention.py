"""Triton variable-length causal attention over packed prefill Q/K/V tensors."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _packed_prefill_attention_kernel(
    q_ptr, k_ptr, v_ptr, offsets_ptr, out_ptr,
    stride_qh, stride_qt, stride_qd,
    stride_kh, stride_kt, stride_kd,
    stride_vh, stride_vt, stride_vd,
    stride_oh, stride_ot, stride_od,
    scale,
    GROUP: tl.constexpr,
    D_HEAD: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    """One program owns one (sequence, query head, query-token tile)."""
    sequence = tl.program_id(0)
    query_head = tl.program_id(1)
    query_block = tl.program_id(2)
    kv_head = query_head // GROUP

    sequence_start = tl.load(offsets_ptr + sequence)
    sequence_end = tl.load(offsets_ptr + sequence + 1)
    sequence_length = sequence_end - sequence_start
    query_start = query_block * BLOCK_M

    offs_m = query_start + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    query_mask = offs_m < sequence_length
    dim_mask = offs_d < D_HEAD

    q_ptrs = (
        q_ptr
        + query_head * stride_qh
        + (sequence_start + offs_m[:, None]) * stride_qt
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=query_mask[:, None] & dim_mask[None, :], other=0.0)

    # FlashAttention-style online softmax state for each query row.
    running_max = tl.full([BLOCK_M], float("-inf"), tl.float32)
    running_sum = tl.zeros([BLOCK_M], tl.float32)
    accumulator = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)

    # A causal query tile never needs K/V blocks strictly to its right.
    attended_length = tl.where(
        query_start < sequence_length,
        tl.minimum(sequence_length, query_start + BLOCK_M),
        0,
    )
    for kv_block in range(tl.cdiv(attended_length, BLOCK_N)):
        key_positions = kv_block * BLOCK_N + offs_n
        key_mask = key_positions < sequence_length

        k_ptrs = (
            k_ptr
            + kv_head * stride_kh
            + (sequence_start + key_positions[None, :]) * stride_kt
            + offs_d[:, None] * stride_kd
        )
        k = tl.load(k_ptrs, mask=dim_mask[:, None] & key_mask[None, :], other=0.0)
        scores = tl.dot(q, k) * scale

        # Causality is local to the sequence. Invalid query rows are assigned a
        # finite dummy score so the online-softmax recurrence cannot form NaNs;
        # those rows are masked when the output is stored.
        causal_mask = key_positions[None, :] <= offs_m[:, None]
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
            + kv_head * stride_vh
            + (sequence_start + key_positions[:, None]) * stride_vt
            + offs_d[None, :] * stride_vd
        )
        v = tl.load(v_ptrs, mask=key_mask[:, None] & dim_mask[None, :], other=0.0)
        probabilities_for_dot = (
            probabilities.to(tl.bfloat16) if IS_BF16 else probabilities.to(tl.float16)
        )
        accumulator = (
            accumulator * correction[:, None]
            + tl.dot(probabilities_for_dot, v)
        )
        running_sum = (
            running_sum * correction + tl.sum(probabilities, axis=1)
        )
        running_max = new_max

    output = accumulator / running_sum[:, None]
    out_ptrs = (
        out_ptr
        + query_head * stride_oh
        + (sequence_start + offs_m[:, None]) * stride_ot
        + offs_d[None, :] * stride_od
    )
    tl.store(
        out_ptrs,
        output.to(out_ptr.dtype.element_ty),
        mask=query_mask[:, None] & dim_mask[None, :],
    )


def packed_prefill_attention_triton(
    q,
    k,
    v,
    cu_seqlens,
    max_seqlen,
    scale=None,
    block_m=64,
    block_n=32,
    num_warps=4,
    num_stages=2,
):
    """Run packed causal GQA attention.

    q: (1, query_heads, total_tokens, head_dim)
    k/v: (1, kv_heads, total_tokens, head_dim)
    cu_seqlens: cumulative packed-token offsets shaped (num_sequences + 1,)
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must be four-dimensional")
    if q.shape[0] != 1 or k.shape[0] != 1 or v.shape[0] != 1:
        raise ValueError("packed prefill expects a singleton physical batch dimension")
    if k.shape != v.shape:
        raise ValueError("k and v must have identical shapes")
    if q.shape[2:] != k.shape[2:]:
        raise ValueError("q, k, and v must share token and head dimensions")
    if q.shape[1] % k.shape[1] != 0:
        raise ValueError("query heads must be a multiple of KV heads")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("packed Triton attention supports float16 and bfloat16")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, and v must have the same dtype")
    if not q.is_cuda or not k.is_cuda or not v.is_cuda or not cu_seqlens.is_cuda:
        raise ValueError("packed Triton attention requires CUDA tensors")
    if not (q.device == k.device == v.device == cu_seqlens.device):
        raise ValueError("q, k, v, and cu_seqlens must share a device")
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("head_dim must be contiguous")
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must contain at least one sequence")
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError("cu_seqlens must use an integer dtype")
    if max_seqlen < 1:
        raise ValueError("max_seqlen must be positive")
    for name, value in (("block_m", block_m), ("block_n", block_n)):
        if value < 16 or value & (value - 1):
            raise ValueError(f"{name} must be a power of two greater than or equal to 16")
    if num_warps not in (1, 2, 4, 8):
        raise ValueError("num_warps must be one of 1, 2, 4, or 8")
    if num_stages < 1:
        raise ValueError("num_stages must be positive")

    _, query_heads, _, d_head = q.shape
    kv_heads = k.shape[1]
    num_sequences = cu_seqlens.numel() - 1
    if scale is None:
        scale = 1.0 / (d_head ** 0.5)

    block_d = triton.next_power_of_2(d_head)
    out = torch.empty_like(q)
    grid = (num_sequences, query_heads, triton.cdiv(max_seqlen, block_m))
    _packed_prefill_attention_kernel[grid](
        q, k, v, cu_seqlens, out,
        q.stride(1), q.stride(2), q.stride(3),
        k.stride(1), k.stride(2), k.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        out.stride(1), out.stride(2), out.stride(3),
        scale,
        GROUP=query_heads // kv_heads,
        D_HEAD=d_head,
        BLOCK_D=block_d,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        IS_BF16=q.dtype == torch.bfloat16,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
