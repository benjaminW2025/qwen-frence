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
    width,
    gate_row_stride,
    up_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    rows = offsets // width
    columns = offsets - rows * width
    gate = tl.load(gate_ptr + rows * gate_row_stride + columns, mask=mask).to(tl.float32)
    up = tl.load(up_ptr + rows * up_row_stride + columns, mask=mask).to(tl.float32)
    result = (gate * tl.sigmoid(gate)) * up
    tl.store(out_ptr + offsets, result, mask=mask)


def _flattened_row_stride(tensor):
    """Return the row stride for a row-major tensor with a strided last split.

    Packed gate/up projection produces two views shaped ``(..., d_ff)`` whose
    rows are separated by ``2 * d_ff`` elements. They are zero-copy, dense
    within each row, and safe for the fused kernel without materialization.
    """
    if tensor.ndim < 1 or tensor.shape[-1] < 1 or tensor.stride(-1) != 1:
        raise ValueError("gate and up tensors must be dense within their last dimension")
    if tensor.ndim == 1:
        return tensor.shape[-1]
    row_stride = tensor.stride(-2)
    expected = row_stride
    for dimension in range(tensor.ndim - 2, -1, -1):
        if tensor.stride(dimension) != expected:
            raise ValueError("gate and up tensors must have a flattenable row-major layout")
        expected *= tensor.shape[dimension]
    return row_stride


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
    gate_row_stride = _flattened_row_stride(gate)
    up_row_stride = _flattened_row_stride(up)
    if block_size not in (128, 256, 512, 1024):
        raise ValueError("block_size must be 128, 256, 512, or 1024")
    if num_warps not in (1, 2, 4, 8):
        raise ValueError("num_warps must be 1, 2, 4, or 8")
    if num_stages < 1:
        raise ValueError("num_stages must be positive")

    output = torch.empty(gate.shape, device=gate.device, dtype=gate.dtype)
    n_elements = gate.numel()
    _swiglu_kernel[(triton.cdiv(n_elements, block_size),)](
        gate,
        up,
        output,
        n_elements,
        gate.shape[-1],
        gate_row_stride,
        up_row_stride,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output
