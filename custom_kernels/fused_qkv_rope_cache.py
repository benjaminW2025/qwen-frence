"""Experimental packed QKV projection with RoPE and direct paged-cache stores."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _qk_projection_rope_cache_kernel(
    x_ptr, weight_ptr, bias_ptr, positions_ptr, slots_ptr,
    q_ptr, k_pool_ptr,
    stride_xm, stride_xk, stride_wn, stride_wk,
    stride_qm, stride_qh, stride_qd,
    stride_pm, stride_ph, stride_pd,
    stride_position, stride_slot,
    rows, log_theta,
    HIDDEN: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    projection_head = tl.program_id(1)  # Q heads 0..11, then K heads 12..13.
    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = tl.arange(0, BLOCK_N)
    accumulator_first = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    accumulator_second = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    output_base = projection_head * (2 * BLOCK_N)

    for start in range(0, HIDDEN, BLOCK_K):
        offsets_k = start + tl.arange(0, BLOCK_K)
        x = tl.load(
            x_ptr + offsets_m[:, None] * stride_xm + offsets_k[None, :] * stride_xk,
            mask=(offsets_m[:, None] < rows) & (offsets_k[None, :] < HIDDEN),
            other=0.0,
        )
        first_weight = tl.load(
            weight_ptr + (output_base + offsets_n[:, None]) * stride_wn
            + offsets_k[None, :] * stride_wk,
            mask=offsets_k[None, :] < HIDDEN,
            other=0.0,
        )
        second_weight = tl.load(
            weight_ptr + (output_base + BLOCK_N + offsets_n[:, None]) * stride_wn
            + offsets_k[None, :] * stride_wk,
            mask=offsets_k[None, :] < HIDDEN,
            other=0.0,
        )
        accumulator_first += tl.dot(x, tl.trans(first_weight))
        accumulator_second += tl.dot(x, tl.trans(second_weight))

    first_bias = tl.load(bias_ptr + output_base + offsets_n)
    second_bias = tl.load(bias_ptr + output_base + BLOCK_N + offsets_n)
    first = accumulator_first + first_bias[None, :]
    second = accumulator_second + second_bias[None, :]
    position = tl.load(
        positions_ptr + offsets_m * stride_position,
        mask=offsets_m < rows, other=0,
    ).to(tl.float32)
    inverse_frequency = tl.exp(
        offsets_n.to(tl.float32) * (-log_theta / BLOCK_N)
    )
    angle = position[:, None] * inverse_frequency[None, :]
    cosine, sine = tl.cos(angle), tl.sin(angle)
    rotated_first = first * cosine - second * sine
    rotated_second = second * cosine + first * sine
    row_mask = offsets_m[:, None] < rows

    is_query = projection_head < 12
    query_head = projection_head
    query_base = (offsets_m[:, None] * stride_qm
                  + query_head * stride_qh)
    tl.store(q_ptr + query_base + offsets_n[None, :] * stride_qd,
             rotated_first, mask=row_mask & is_query)
    tl.store(q_ptr + query_base + (BLOCK_N + offsets_n[None, :]) * stride_qd,
             rotated_second, mask=row_mask & is_query)

    kv_head = projection_head - 12
    slot = tl.load(slots_ptr + offsets_m * stride_slot,
                   mask=offsets_m < rows, other=0)
    cache_base = slot[:, None] * stride_pm + kv_head * stride_ph
    tl.store(k_pool_ptr + cache_base + offsets_n[None, :] * stride_pd,
             rotated_first, mask=row_mask & ~is_query)
    tl.store(k_pool_ptr + cache_base + (BLOCK_N + offsets_n[None, :]) * stride_pd,
             rotated_second, mask=row_mask & ~is_query)


@triton.jit
def _v_projection_cache_kernel(
    x_ptr, weight_ptr, bias_ptr, slots_ptr, v_pool_ptr,
    stride_xm, stride_xk, stride_wn, stride_wk,
    stride_pm, stride_ph, stride_pd, stride_slot,
    rows,
    HIDDEN: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    part = tl.program_id(1)  # Two 64-wide halves for each of two V heads.
    kv_head, half = part // 2, part % 2
    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = tl.arange(0, BLOCK_N)
    output_base = 14 * 128 + kv_head * 128 + half * BLOCK_N
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for start in range(0, HIDDEN, BLOCK_K):
        offsets_k = start + tl.arange(0, BLOCK_K)
        x = tl.load(
            x_ptr + offsets_m[:, None] * stride_xm + offsets_k[None, :] * stride_xk,
            mask=(offsets_m[:, None] < rows) & (offsets_k[None, :] < HIDDEN),
            other=0.0,
        )
        weight = tl.load(
            weight_ptr + (output_base + offsets_n[:, None]) * stride_wn
            + offsets_k[None, :] * stride_wk,
            mask=offsets_k[None, :] < HIDDEN,
            other=0.0,
        )
        accumulator += tl.dot(x, tl.trans(weight))
    bias = tl.load(bias_ptr + output_base + offsets_n)
    value = accumulator + bias[None, :]
    slot = tl.load(slots_ptr + offsets_m * stride_slot,
                   mask=offsets_m < rows, other=0)
    cache_base = slot[:, None] * stride_pm + kv_head * stride_ph
    tl.store(
        v_pool_ptr + cache_base + (half * BLOCK_N + offsets_n[None, :]) * stride_pd,
        value,
        mask=offsets_m[:, None] < rows,
    )


def fused_qkv_rope_cache(
    x, weight, bias, positions, slot_mapping, k_pool, v_pool,
    *, base=1_000_000.0, block_m=16, block_k=32, num_warps=4,
):
    """Project normalized hidden rows and materialize only rotated Q plus paged K/V."""
    if x.ndim != 3 or x.shape[1] != 1 or x.shape[2] != 1536:
        raise ValueError("fused QKV expects hidden input shaped (batch, 1, 1536)")
    if weight.shape != (2048, 1536) or bias is None or bias.shape != (2048,):
        raise ValueError("fused QKV expects Qwen-1.5B packed weight/bias shapes")
    if not x.is_contiguous() or not weight.is_contiguous() or not bias.is_contiguous():
        raise ValueError("hidden input and packed QKV parameters must be contiguous")
    if positions.shape != (x.shape[0],) or slot_mapping.shape != (x.shape[0],):
        raise ValueError("positions and slots must contain one value per batch row")
    if k_pool.shape != v_pool.shape or k_pool.shape[1:] != (16, 2, 128):
        raise ValueError("expected matching page-16 Qwen K/V pools")
    tensors = (x, weight, bias, positions, slot_mapping, k_pool, v_pool)
    if not all(tensor.is_cuda and tensor.device == x.device for tensor in tensors):
        raise ValueError("all fused QKV tensors must share one CUDA device")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("fused QKV supports FP16 and BF16")
    if any(tensor.dtype != x.dtype for tensor in (weight, bias, k_pool, v_pool)):
        raise ValueError("hidden, weights, bias, and cache must share a dtype")
    if positions.dtype not in (torch.int32, torch.int64):
        raise ValueError("positions must use an integer dtype")
    if slot_mapping.dtype not in (torch.int32, torch.int64):
        raise ValueError("slot mapping must use an integer dtype")
    if block_m not in (8, 16, 32, 64) or block_k not in (16, 32, 64):
        raise ValueError("unsupported fused QKV tile")
    if num_warps not in (4, 8):
        raise ValueError("fused QKV uses four or eight warps")

    rows = x.shape[0]
    hidden = x.view(rows, 1536)
    q = torch.empty(rows, 12, 128, device=x.device, dtype=x.dtype)
    flat_k = k_pool.view(-1, 2, 128)
    flat_v = v_pool.view(-1, 2, 128)
    grid_m = triton.cdiv(rows, block_m)
    common = (hidden, weight, bias, positions, slot_mapping)
    _qk_projection_rope_cache_kernel[(grid_m, 14)](
        *common, q, flat_k,
        hidden.stride(0), hidden.stride(1), weight.stride(0), weight.stride(1),
        q.stride(0), q.stride(1), q.stride(2),
        flat_k.stride(0), flat_k.stride(1), flat_k.stride(2),
        positions.stride(0), slot_mapping.stride(0),
        rows, math.log(base), HIDDEN=1536,
        BLOCK_M=block_m, BLOCK_N=64, BLOCK_K=block_k, num_warps=num_warps,
    )
    _v_projection_cache_kernel[(grid_m, 4)](
        hidden, weight, bias, slot_mapping, flat_v,
        hidden.stride(0), hidden.stride(1), weight.stride(0), weight.stride(1),
        flat_v.stride(0), flat_v.stride(1), flat_v.stride(2),
        slot_mapping.stride(0), rows, HIDDEN=1536,
        BLOCK_M=block_m, BLOCK_N=64, BLOCK_K=block_k, num_warps=num_warps,
    )
    return q
