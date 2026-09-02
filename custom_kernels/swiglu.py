"""Fused SiLU-gated elementwise product used by Qwen's MLP."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _swiglu_kernel(
    gate_ptr,
    up_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    gate = tl.load(gate_ptr + offsets, mask=mask).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask).to(tl.float32)
    result = (gate * tl.sigmoid(gate)) * up
    tl.store(out_ptr + offsets, result, mask=mask)


def swiglu(gate, up, *, block_size=256, num_warps=4, num_stages=2):
    """Return ``silu(gate) * up`` without materializing the SiLU tensor."""
    if gate.shape != up.shape:
        raise ValueError("gate and up tensors must have matching shapes")
    if gate.device != up.device:
        raise ValueError("gate and up tensors must share a device")
    if not gate.is_cuda:
        raise ValueError("fused SwiGLU requires CUDA tensors")
    if gate.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("fused SwiGLU supports float16 and bfloat16")
    if up.dtype != gate.dtype:
        raise ValueError("gate and up tensors must have matching dtypes")
    if not gate.is_contiguous() or not up.is_contiguous():
        raise ValueError("gate and up tensors must be contiguous")
    if block_size not in (128, 256, 512, 1024):
        raise ValueError("block_size must be 128, 256, 512, or 1024")
    if num_warps not in (1, 2, 4, 8):
        raise ValueError("num_warps must be 1, 2, 4, or 8")
    if num_stages < 1:
        raise ValueError("num_stages must be positive")

    output = torch.empty_like(gate)
    n_elements = gate.numel()
    _swiglu_kernel[(triton.cdiv(n_elements, block_size),)](
        gate,
        up,
        output,
        n_elements,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output
