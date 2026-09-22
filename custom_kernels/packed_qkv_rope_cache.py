"""Post-cuBLAS packed-QKV epilogue: RoPE Q/K and direct paged K/V writes."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _packed_qkv_rope_cache_kernel(
    packed_ptr, positions_ptr, slots_ptr, q_ptr, k_pool_ptr, v_pool_ptr,
    stride_pm, stride_pn, stride_qm, stride_qh, stride_qd,
    stride_km, stride_kh, stride_kd, stride_vm, stride_vh, stride_vd,
    position_stride, slot_stride, log_theta,
    rows, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    segment = tl.program_id(1)  # Q0..11, K0..1, V0..1.
    offsets = tl.arange(0, BLOCK)
    valid = offsets < 64
    source = segment * 128 + offsets
    values = tl.load(
        packed_ptr + row * stride_pm + source * stride_pn,
        mask=(row < rows) & valid,
        other=0.0,
    )
    is_value = segment >= 14
    is_query = segment < 12
    is_kv = ~is_query
    # Packed columns are [Q0..Q11, K0..K1, V0..V1]. K and V each restart
    # their cache-head index at zero; treating V14/V15 as segment-12 writes
    # heads 2/3 past the two-head cache and leaves the real V heads untouched.
    head = tl.where(
        is_query,
        segment,
        tl.where(is_value, segment - 14, segment - 12),
    )
    position = tl.load(positions_ptr + row * position_stride,
                       mask=row < rows, other=0).to(tl.float32)
    half = 64
    first = offsets
    second = offsets + half
    inverse_frequency = tl.exp(
        offsets.to(tl.float32) * (-log_theta / half)
    )
    angle = position * inverse_frequency
    cosine, sine = tl.cos(angle), tl.sin(angle)
    rotated_first = values * cosine - tl.load(
        packed_ptr + row * stride_pm + (segment * 128 + second) * stride_pn,
        mask=(row < rows) & (second < 128), other=0.0,
    ) * sine
    rotated_second = values * sine + tl.load(
        packed_ptr + row * stride_pm + (segment * 128 + second) * stride_pn,
        mask=(row < rows) & (second < 128), other=0.0,
    ) * cosine
    # V segments are copied without RoPE; Q/K use the rotate-half result.
    raw_second = tl.load(
        packed_ptr + row * stride_pm + (segment * 128 + second) * stride_pn,
        mask=(row < rows) & (second < 128), other=0.0,
    )
    out_first = tl.where(is_value, values, rotated_first)
    out_second = tl.where(is_value, raw_second, rotated_second)
    row_mask = (row < rows) & valid
    q_base = row * stride_qm + head * stride_qh
    tl.store(q_ptr + q_base + first * stride_qd, out_first,
             mask=row_mask & is_query)
    tl.store(q_ptr + q_base + second * stride_qd, out_second,
             mask=(row < rows) & (second < 128) & is_query)
    slot = tl.load(slots_ptr + row * slot_stride, mask=row < rows, other=0)
    cache_base = slot * stride_km + head * stride_kh
    tl.store(k_pool_ptr + cache_base + first * stride_kd, out_first,
             mask=row_mask & is_kv & ~is_value)
    tl.store(k_pool_ptr + cache_base + second * stride_kd, out_second,
             mask=(row < rows) & (second < 128) & is_kv & ~is_value)
    v_base = slot * stride_vm + head * stride_vh
    tl.store(v_pool_ptr + v_base + first * stride_vd, out_first,
             mask=row_mask & is_value)
    tl.store(v_pool_ptr + v_base + second * stride_vd, out_second,
             mask=(row < rows) & (second < 128) & is_value)


def packed_qkv_rope_cache(
    packed, positions, slot_mapping, k_pool, v_pool,
    *, base=1_000_000.0, num_warps=4,
):
    """Consume packed QKV GEMM output without split/transpose/materialization."""
    if packed.ndim == 3:
        if packed.shape[1] != 1:
            raise ValueError("packed QKV must have one decode token per row")
        packed = packed[:, 0, :]
    if packed.ndim != 2 or packed.shape[1] != 2048:
        raise ValueError("expected packed QKV shape (batch, 2048)")
    rows = packed.shape[0]
    if positions.shape != (rows,) or slot_mapping.shape != (rows,):
        raise ValueError("positions and slots must contain one value per row")
    if k_pool.shape != v_pool.shape or k_pool.shape[1:] != (16, 2, 128):
        raise ValueError("expected matching page-16 Qwen K/V pools")
    tensors = (packed, positions, slot_mapping, k_pool, v_pool)
    if not all(t.is_cuda and t.device == packed.device for t in tensors):
        raise ValueError("packed QKV epilogue tensors must share one CUDA device")
    if packed.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("packed QKV epilogue supports FP16 and BF16")
    if any(t.dtype != packed.dtype for t in (k_pool, v_pool)):
        raise ValueError("packed QKV and KV pools must share a dtype")
    q = torch.empty(rows, 12, 128, device=packed.device, dtype=packed.dtype)
    flat_k = k_pool.view(-1, 2, 128)
    flat_v = v_pool.view(-1, 2, 128)
    _packed_qkv_rope_cache_kernel[(rows, 16)](
        packed, positions, slot_mapping, q, flat_k, flat_v,
        packed.stride(0), packed.stride(1),
        q.stride(0), q.stride(1), q.stride(2),
        flat_k.stride(0), flat_k.stride(1), flat_k.stride(2),
        flat_v.stride(0), flat_v.stride(1), flat_v.stride(2),
        positions.stride(0), slot_mapping.stride(0), math.log(base), rows,
        BLOCK=64, num_warps=num_warps,
    )
    return q
