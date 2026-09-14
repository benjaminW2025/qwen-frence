"""Write packed K/V into paged slots, ignoring padded graph rows."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _masked_kv_write_kernel(
    k_ptr, v_ptr, slots_ptr, valid_tokens_ptr, k_pool_ptr, v_pool_ptr,
    stride_kh, stride_kt, stride_kd,
    stride_vh, stride_vt, stride_vd,
    stride_pt, stride_ph, stride_pd,
    N_KV_HEADS: tl.constexpr,
    D_HEAD: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    program = tl.program_id(0)
    token = program // N_KV_HEADS
    head = program % N_KV_HEADS
    dims = tl.arange(0, BLOCK_D)
    active = token < tl.load(valid_tokens_ptr)
    mask = active & (dims < D_HEAD)
    slot = tl.load(slots_ptr + token, mask=active, other=0)
    k = tl.load(k_ptr + head * stride_kh + token * stride_kt + dims * stride_kd,
                mask=mask, other=0)
    v = tl.load(v_ptr + head * stride_vh + token * stride_vt + dims * stride_vd,
                mask=mask, other=0)
    destination = slot * stride_pt + head * stride_ph + dims * stride_pd
    tl.store(k_pool_ptr + destination, k, mask=mask)
    tl.store(v_pool_ptr + destination, v, mask=mask)


def masked_kv_write(k, v, slots, valid_tokens, k_pool, v_pool):
    """Capture-safe placement; ``valid_tokens`` is a device scalar updated per replay."""
    if k.ndim != 4 or k.shape[0] != 1 or k.shape != v.shape:
        raise ValueError("K/V must have matching shape (1, kv_heads, tokens, head_dim)")
    if slots.shape != (k.shape[2],) or slots.dtype not in (torch.int32, torch.int64):
        raise ValueError("slots must have one integer entry per captured token")
    if valid_tokens.ndim != 0 or valid_tokens.dtype not in (torch.int32, torch.int64):
        raise ValueError("valid_tokens must be an integer device scalar")
    if k_pool.ndim != 4 or k_pool.shape != v_pool.shape or k_pool.shape[2:] != (k.shape[1], k.shape[3]):
        raise ValueError("K/V pools must match the input heads and dimension")
    tensors = (k, v, slots, valid_tokens, k_pool, v_pool)
    if any(not tensor.is_cuda or tensor.device != k.device for tensor in tensors):
        raise ValueError("all inputs must share one CUDA device")
    if k.dtype != v.dtype or k.dtype != k_pool.dtype or k.dtype != v_pool.dtype:
        raise ValueError("K/V and pools must share a dtype")
    flat_k = k_pool.view(-1, k.shape[1], k.shape[3])
    flat_v = v_pool.view(-1, v.shape[1], v.shape[3])
    _masked_kv_write_kernel[(k.shape[2] * k.shape[1],)](
        k, v, slots, valid_tokens, flat_k, flat_v,
        k.stride(1), k.stride(2), k.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        flat_k.stride(0), flat_k.stride(1), flat_k.stride(2),
        N_KV_HEADS=k.shape[1], D_HEAD=k.shape[3],
        BLOCK_D=triton.next_power_of_2(k.shape[3]),
    )
