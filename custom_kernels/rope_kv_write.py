"""Fuse Q/K RoPE with direct K/V placement into paged cache slots."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kv_write_kernel(
    q_ptr, k_ptr, v_ptr, positions_ptr, slot_mapping_ptr,
    q_out_ptr, k_pool_ptr, v_pool_ptr,
    stride_qh, stride_qt, stride_qd,
    stride_kh, stride_kt, stride_kd,
    stride_vh, stride_vt, stride_vd,
    stride_oqh, stride_oqt, stride_oqd,
    stride_pt, stride_ph, stride_pd,
    position_stride,
    log_theta,
    N_Q_HEADS: tl.constexpr,
    N_KV_HEADS: tl.constexpr,
    HALF: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    program = tl.program_id(0)
    token = program // N_Q_HEADS
    head = program - token * N_Q_HEADS
    offsets = tl.arange(0, BLOCK_SIZE // 2)
    dim_mask = offsets < HALF
    first = offsets
    second = offsets + HALF

    position = tl.load(positions_ptr + token * position_stride).to(tl.float32)
    inv_freq = tl.exp(offsets.to(tl.float32) * (-log_theta / HALF))
    angle = position * inv_freq
    cosine = tl.cos(angle)
    sine = tl.sin(angle)

    q_base = q_ptr + head * stride_qh + token * stride_qt
    q_first = tl.load(q_base + first * stride_qd, mask=dim_mask)
    q_second = tl.load(q_base + second * stride_qd, mask=dim_mask)
    q_out_base = q_out_ptr + head * stride_oqh + token * stride_oqt
    tl.store(
        q_out_base + first * stride_oqd,
        q_first * cosine - q_second * sine,
        mask=dim_mask,
    )
    tl.store(
        q_out_base + second * stride_oqd,
        q_second * cosine + q_first * sine,
        mask=dim_mask,
    )

    kv_mask = head < N_KV_HEADS
    k_base = k_ptr + head * stride_kh + token * stride_kt
    k_first = tl.load(k_base + first * stride_kd, mask=dim_mask & kv_mask, other=0.0)
    k_second = tl.load(k_base + second * stride_kd, mask=dim_mask & kv_mask, other=0.0)
    slot = tl.load(slot_mapping_ptr + token)
    pool_base = slot * stride_pt + head * stride_ph
    tl.store(
        k_pool_ptr + pool_base + first * stride_pd,
        k_first * cosine - k_second * sine,
        mask=dim_mask & kv_mask,
    )
    tl.store(
        k_pool_ptr + pool_base + second * stride_pd,
        k_second * cosine + k_first * sine,
        mask=dim_mask & kv_mask,
    )

    v_base = v_ptr + head * stride_vh + token * stride_vt
    v_first = tl.load(v_base + first * stride_vd, mask=dim_mask & kv_mask, other=0.0)
    v_second = tl.load(v_base + second * stride_vd, mask=dim_mask & kv_mask, other=0.0)
    tl.store(
        v_pool_ptr + pool_base + first * stride_pd,
        v_first,
        mask=dim_mask & kv_mask,
    )
    tl.store(
        v_pool_ptr + pool_base + second * stride_pd,
        v_second,
        mask=dim_mask & kv_mask,
    )


def rope_kv_write(
    q,
    k,
    v,
    positions,
    slot_mapping,
    k_pool,
    v_pool,
    *,
    base=1_000_000.0,
    num_warps=4,
    num_stages=2,
):
    """Rotate Q/K and write rotated K plus raw V directly to paged slots."""
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must be rank-four tensors")
    if q.shape[0] != 1 or k.shape[0] != 1 or v.shape[0] != 1:
        raise ValueError("packed q, k, and v must have batch dimension one")
    if k.shape != v.shape:
        raise ValueError("k and v must have matching shapes")
    if q.shape[2:] != k.shape[2:]:
        raise ValueError("q and k token/head dimensions are incompatible")
    if q.shape[-1] % 2:
        raise ValueError("head dimension must be even")
    if k_pool.ndim != 4 or k_pool.shape != v_pool.shape:
        raise ValueError("K/V pools must have matching rank-four shapes")
    if k_pool.shape[2:] != (k.shape[1], k.shape[-1]):
        raise ValueError("pool KV-head and head dimensions must match k/v")
    if positions.ndim == 2:
        if positions.shape != (1, q.shape[2]):
            raise ValueError("positions must have shape (1, tokens)")
        positions = positions[0]
    if positions.ndim != 1 or positions.numel() != q.shape[2]:
        raise ValueError("positions must contain one entry per token")
    if slot_mapping.ndim != 1 or slot_mapping.numel() != q.shape[2]:
        raise ValueError("slot mapping must contain one entry per token")
    tensors = (q, k, v, positions, slot_mapping, k_pool, v_pool)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("fused RoPE/KV placement requires CUDA tensors")
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("all tensors must share a device")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("q, k, and v must use float16 or bfloat16")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("q, k, and v dtypes must match")
    if k_pool.dtype != q.dtype or v_pool.dtype != q.dtype:
        raise ValueError("Q/K/V and pool dtypes must match")
    if slot_mapping.dtype not in (torch.int32, torch.int64):
        raise ValueError("slot mapping must use an integer dtype")
    if num_warps not in (1, 2, 4, 8) or num_stages < 1:
        raise ValueError("unsupported warp or stage count")

    q_out = torch.empty_like(q)
    flat_k_pool = k_pool.view(-1, k_pool.shape[2], k_pool.shape[3])
    flat_v_pool = v_pool.view(-1, v_pool.shape[2], v_pool.shape[3])
    tokens = q.shape[2]
    dim = q.shape[-1]
    _rope_kv_write_kernel[(tokens * q.shape[1],)](
        q, k, v, positions, slot_mapping, q_out, flat_k_pool, flat_v_pool,
        q.stride(1), q.stride(2), q.stride(3),
        k.stride(1), k.stride(2), k.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        q_out.stride(1), q_out.stride(2), q_out.stride(3),
        flat_k_pool.stride(0), flat_k_pool.stride(1), flat_k_pool.stride(2),
        positions.stride(0),
        math.log(base),
        N_Q_HEADS=q.shape[1],
        N_KV_HEADS=k.shape[1],
        HALF=dim // 2,
        BLOCK_SIZE=triton.next_power_of_2(dim),
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return q_out


@triton.jit
def _native_decode_rope_kv_kernel(
    q_ptr, k_ptr, v_ptr, positions_ptr, slots_ptr,
    q_out_ptr, k_pool_ptr, v_pool_ptr,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kh, stride_kd,
    stride_vb, stride_vh, stride_vd,
    stride_ob, stride_oh, stride_od,
    stride_pt, stride_ph, stride_pd,
    position_stride, slot_stride,
    log_theta,
    N_Q_HEADS: tl.constexpr,
    N_KV_HEADS: tl.constexpr,
    HALF: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    program = tl.program_id(0)
    batch = program // N_Q_HEADS
    head = program - batch * N_Q_HEADS
    offsets = tl.arange(0, BLOCK_SIZE // 2)
    mask = offsets < HALF
    second = offsets + HALF

    position = tl.load(positions_ptr + batch * position_stride).to(tl.float32)
    angle = position * tl.exp(offsets.to(tl.float32) * (-log_theta / HALF))
    cosine, sine = tl.cos(angle), tl.sin(angle)

    q_base = q_ptr + batch * stride_qb + head * stride_qh
    q_first = tl.load(q_base + offsets * stride_qd, mask=mask)
    q_second = tl.load(q_base + second * stride_qd, mask=mask)
    out_base = q_out_ptr + batch * stride_ob + head * stride_oh
    tl.store(out_base + offsets * stride_od,
             q_first * cosine - q_second * sine, mask=mask)
    tl.store(out_base + second * stride_od,
             q_second * cosine + q_first * sine, mask=mask)

    kv_mask = head < N_KV_HEADS
    k_base = k_ptr + batch * stride_kb + head * stride_kh
    v_base = v_ptr + batch * stride_vb + head * stride_vh
    k_first = tl.load(k_base + offsets * stride_kd, mask=mask & kv_mask, other=0.0)
    k_second = tl.load(k_base + second * stride_kd, mask=mask & kv_mask, other=0.0)
    v_first = tl.load(v_base + offsets * stride_vd, mask=mask & kv_mask, other=0.0)
    v_second = tl.load(v_base + second * stride_vd, mask=mask & kv_mask, other=0.0)
    slot = tl.load(slots_ptr + batch * slot_stride)
    pool_base = slot * stride_pt + head * stride_ph
    tl.store(k_pool_ptr + pool_base + offsets * stride_pd,
             k_first * cosine - k_second * sine, mask=mask & kv_mask)
    tl.store(k_pool_ptr + pool_base + second * stride_pd,
             k_second * cosine + k_first * sine, mask=mask & kv_mask)
    tl.store(v_pool_ptr + pool_base + offsets * stride_pd,
             v_first, mask=mask & kv_mask)
    tl.store(v_pool_ptr + pool_base + second * stride_pd,
             v_second, mask=mask & kv_mask)


def native_decode_rope_kv_write(
    q, k, v, positions, slot_mapping, k_pool, v_pool,
    *, base=1_000_000.0, num_warps=4,
):
    """Rotate native-layout decode Q/K and place K/V without layout copies.

    Inputs are (B, H, 1, D), as produced by the model's decode projections.
    This is an experimental kernel; production dispatch does not call it.
    """
    if (q.ndim != 4 or k.ndim != 4 or v.ndim != 4
            or q.shape[2] != 1 or k.shape[2] != 1 or v.shape[2] != 1):
        raise ValueError("decode Q/K/V must have shape (batch, heads, 1, dim)")
    batch, q_heads, _, dim = q.shape
    if k.shape != v.shape or k.shape[0] != batch or k.shape[-1] != dim:
        raise ValueError("decode K/V shapes do not match Q")
    if dim % 2 or q_heads < k.shape[1]:
        raise ValueError("invalid query/KV head dimensions")
    if positions.shape != (batch,) or slot_mapping.shape != (batch,):
        raise ValueError("positions and slots must contain one entry per batch row")
    if k_pool.shape != v_pool.shape or k_pool.ndim != 4:
        raise ValueError("K/V pools must have matching rank-four shapes")
    if k_pool.shape[2:] != (k.shape[1], dim) or not k_pool.is_contiguous() or not v_pool.is_contiguous():
        raise ValueError("K/V pools must be contiguous with matching KV dimensions")
    tensors = (q, k, v, positions, slot_mapping, k_pool, v_pool)
    if not all(t.is_cuda and t.device == q.device for t in tensors):
        raise ValueError("all inputs must share one CUDA device")
    if q.dtype not in (torch.float16, torch.bfloat16) or any(t.dtype != q.dtype for t in (k, v, k_pool, v_pool)):
        raise ValueError("Q/K/V and pools must share FP16 or BF16 dtype")
    if positions.dtype not in (torch.int32, torch.int64, torch.float32):
        raise ValueError("positions must be int32, int64, or float32")
    if slot_mapping.dtype not in (torch.int32, torch.int64):
        raise ValueError("slots must use an integer dtype")
    if num_warps not in (1, 2, 4, 8):
        raise ValueError("num_warps must be 1, 2, 4, or 8")

    out = torch.empty_like(q)
    flat_k = k_pool.view(-1, k.shape[1], dim)
    flat_v = v_pool.view(-1, k.shape[1], dim)
    _native_decode_rope_kv_kernel[(batch * q_heads,)](
        q, k, v, positions, slot_mapping, out, flat_k, flat_v,
        q.stride(0), q.stride(1), q.stride(3),
        k.stride(0), k.stride(1), k.stride(3),
        v.stride(0), v.stride(1), v.stride(3),
        out.stride(0), out.stride(1), out.stride(3),
        flat_k.stride(0), flat_k.stride(1), flat_k.stride(2),
        positions.stride(0), slot_mapping.stride(0),
        math.log(base),
        N_Q_HEADS=q_heads, N_KV_HEADS=k.shape[1],
        HALF=dim // 2, BLOCK_SIZE=triton.next_power_of_2(dim),
        num_warps=num_warps,
    )
    return out
