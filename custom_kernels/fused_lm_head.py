"""Exact greedy output head without materializing the full logits matrix.

The first kernel projects vocabulary tiles and writes one maximum/value index
per tile.  The second kernel reduces those partial winners to one token ID per
row.  Accumulators are rounded to the input dtype before comparison so the
decision boundary matches the FP16/BF16 logits consumed by ``torch.argmax``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _project_partial_argmax(
    hidden_ptr,
    weight_ptr,
    partial_values_ptr,
    partial_indices_ptr,
    rows,
    vocab,
    hidden_size: tl.constexpr,
    vocab_blocks,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row_block = tl.program_id(0)
    vocab_block = tl.program_id(1)
    row_offsets = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
    vocab_offsets = vocab_block * BLOCK_N + tl.arange(0, BLOCK_N)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for start in range(0, hidden_size, BLOCK_K):
        k_offsets = start + tl.arange(0, BLOCK_K)
        hidden = tl.load(
            hidden_ptr + row_offsets[:, None] * hidden_size + k_offsets[None, :],
            mask=(row_offsets[:, None] < rows) & (k_offsets[None, :] < hidden_size),
            other=0.0,
        )
        # The output-head weight is row-major (vocab, hidden), so present its
        # tile directly as KxN for the MxK @ KxN tensor-core operation.
        weight = tl.load(
            weight_ptr + k_offsets[:, None] + vocab_offsets[None, :] * hidden_size,
            mask=(k_offsets[:, None] < hidden_size) & (vocab_offsets[None, :] < vocab),
            other=0.0,
        )
        accumulator += tl.dot(hidden, weight)

    # The reference materializes FP16/BF16 logits before argmax.  Rounding here
    # avoids selecting on extra FP32 accumulator precision.
    logits = accumulator.to(hidden_ptr.dtype.element_ty)
    valid = (row_offsets[:, None] < rows) & (vocab_offsets[None, :] < vocab)
    logits = tl.where(valid, logits, -float("inf"))
    maximum = tl.max(logits, axis=1)
    local = tl.argmax(logits, axis=1)
    index = vocab_block * BLOCK_N + local
    output_offsets = row_offsets * vocab_blocks + vocab_block
    tl.store(partial_values_ptr + output_offsets, maximum,
             mask=row_offsets < rows)
    tl.store(partial_indices_ptr + output_offsets, index,
             mask=row_offsets < rows)


@triton.jit
def _reduce_partial_argmax(
    partial_values_ptr,
    partial_indices_ptr,
    token_ids_ptr,
    maximum_values_ptr,
    vocab_blocks,
    vocab,
    BLOCKS: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCKS)
    mask = offsets < vocab_blocks
    values = tl.load(partial_values_ptr + row * vocab_blocks + offsets,
                     mask=mask, other=-float("inf"))
    indices = tl.load(partial_indices_ptr + row * vocab_blocks + offsets,
                      mask=mask, other=0x7FFFFFFF)
    maximum = tl.max(values, axis=0)
    # Prefer the lowest vocabulary ID on an exact tie, matching torch.argmax.
    winning_indices = tl.where(values == maximum, indices, 0x7FFFFFFF)
    token = tl.min(winning_indices, axis=0)
    # When every comparison is false (most notably for a NaN maximum), do not
    # let the reduction sentinel escape as the next embedding index. Correctness
    # checks still report the fallback token as a mismatch instead of allowing
    # an asynchronous device assertion to poison the whole CUDA process.
    token = tl.where(token < vocab, token, 0)
    tl.store(token_ids_ptr + row, token)
    tl.store(maximum_values_ptr + row, maximum)


def _validate(hidden, weight, block_n, block_k):
    if hidden.ndim < 2:
        raise ValueError("hidden states must have at least two dimensions")
    if weight.ndim != 2 or hidden.shape[-1] != weight.shape[1]:
        raise ValueError("weight must have shape (vocab, hidden_size)")
    if not hidden.is_cuda or not weight.is_cuda or hidden.device != weight.device:
        raise ValueError("hidden states and weight must share one CUDA device")
    if hidden.dtype not in (torch.float16, torch.bfloat16) or weight.dtype != hidden.dtype:
        raise ValueError("hidden states and weight must share FP16 or BF16 dtype")
    if not hidden.is_contiguous() or not weight.is_contiguous():
        raise ValueError("hidden states and weight must be contiguous")
    if block_n not in (64, 128, 256) or block_k not in (32, 64, 128):
        raise ValueError("unsupported vocabulary or reduction tile")
    if hidden.shape[-1] < 16 or weight.shape[0] < 1:
        raise ValueError("hidden size and vocabulary must be nonempty")


def workspace_shape(hidden, weight, *, block_n=128):
    """Return the partial-reduction shape without allocating device memory."""
    return hidden.numel() // hidden.shape[-1], triton.cdiv(weight.shape[0], block_n)


def fused_lm_head_argmax(
    hidden,
    weight,
    *,
    block_m=None,
    block_n=128,
    block_k=64,
    num_warps=8,
    num_stages=3,
    workspace=None,
    output=None,
):
    """Return greedy token IDs while retaining only one winner per vocab tile.

    ``hidden`` may be ``(B, H)`` or ``(B, 1, H)``.  Returned token IDs have the
    leading shape of ``hidden`` and use int64, matching ``torch.argmax``.
    Optional workspaces make repeated eager calls allocation-free; CUDA graph
    capture may omit them because capture owns stable allocations.
    """
    _validate(hidden, weight, block_n, block_k)
    rows = hidden.numel() // hidden.shape[-1]
    if block_m is None:
        block_m = 16 if rows <= 16 else 32 if rows <= 32 else 64
    if block_m not in (16, 32, 64):
        raise ValueError("block_m must be 16, 32, or 64")
    if num_warps not in (4, 8) or num_stages < 1:
        raise ValueError("unsupported warp or stage count")
    leading_shape = hidden.shape[:-1]
    vocab_blocks = triton.cdiv(weight.shape[0], block_n)
    if workspace is None:
        partial_values = torch.empty((rows, vocab_blocks), device=hidden.device,
                                     dtype=hidden.dtype)
        partial_indices = torch.empty((rows, vocab_blocks), device=hidden.device,
                                      dtype=torch.int32)
        maximum_values = torch.empty(rows, device=hidden.device, dtype=hidden.dtype)
    else:
        if (not isinstance(workspace, tuple) or len(workspace) != 3
                or workspace[0].shape != (rows, vocab_blocks)
                or workspace[1].shape != (rows, vocab_blocks)
                or workspace[2].shape != (rows,)
                or workspace[0].dtype != hidden.dtype
                or workspace[1].dtype != torch.int32
                or workspace[2].dtype != hidden.dtype
                or workspace[0].device != hidden.device
                or workspace[1].device != hidden.device
                or workspace[2].device != hidden.device):
            raise ValueError("workspace does not match the required reduction buffers")
        partial_values, partial_indices, maximum_values = workspace
    if output is None:
        tokens = torch.empty(rows, device=hidden.device, dtype=torch.int64)
    else:
        if output.shape != (rows,) or output.dtype != torch.int64 or output.device != hidden.device:
            raise ValueError("output must be an int64 CUDA tensor with one entry per row")
        tokens = output
    _project_partial_argmax[(triton.cdiv(rows, block_m), vocab_blocks)](
        hidden, weight, partial_values, partial_indices,
        rows, weight.shape[0], hidden.shape[-1], vocab_blocks,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        num_warps=num_warps, num_stages=num_stages,
    )
    _reduce_partial_argmax[(rows,)](
        partial_values, partial_indices, tokens, maximum_values,
        vocab_blocks, weight.shape[0],
        BLOCKS=triton.next_power_of_2(vocab_blocks), num_warps=8,
    )
    return tokens.view(leading_shape)


def chunked_lm_head_argmax(hidden, weight, *, chunk_size=8192):
    """Exact low-memory PyTorch control; it is not expected to be fast."""
    if hidden.ndim < 2 or weight.ndim != 2 or hidden.shape[-1] != weight.shape[1]:
        raise ValueError("incompatible hidden states and output-head weight")
    if hidden.device != weight.device or hidden.dtype != weight.dtype:
        raise ValueError("hidden states and weight must share device and dtype")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    import torch.nn.functional as F

    flat = hidden.reshape(-1, hidden.shape[-1])
    best_values = torch.full((flat.shape[0],), -float("inf"), device=hidden.device,
                             dtype=hidden.dtype)
    best_indices = torch.zeros(flat.shape[0], device=hidden.device, dtype=torch.int64)
    for start in range(0, weight.shape[0], chunk_size):
        logits = F.linear(flat, weight[start:start + chunk_size])
        values, indices = logits.max(dim=-1)
        replace = values > best_values
        best_values = torch.where(replace, values, best_values)
        best_indices = torch.where(replace, indices.to(torch.int64) + start, best_indices)
    return best_indices.view(hidden.shape[:-1])
