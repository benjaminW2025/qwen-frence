"""Native CUDA grouped-GQA split-K decode attention.

The partial kernel assigns one CTA to a (request, KV head, split) and one warp
to each query head in the GQA group. The CTA stages K/V once in shared memory;
the existing Triton reduction combines numerically stable FP32 partials.
"""

from __future__ import annotations

import importlib
from pathlib import Path
import sys

import torch


def _extension():
    directory = Path(__file__).resolve().parent / "native_grouped_decode"
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    try:
        return importlib.import_module("_native_grouped_decode")
    except ImportError as exc:
        raise RuntimeError(
            "native grouped decode is not built; run: "
            "python3 custom_kernels/native_grouped_decode/setup.py build_ext --inplace"
        ) from exc


def native_grouped_splitk_decode_attention(
    q,
    k_pool,
    v_pool,
    block_table,
    seq_lens,
    *,
    split_k,
    scale=None,
    partials=None,
):
    """Run the six-warps-per-KV-head native partial and Triton reduction."""
    if q.ndim != 3 or k_pool.ndim != 4 or k_pool.shape != v_pool.shape:
        raise ValueError("expected Q rank three and matching rank-four K/V pools")
    batch, query_heads, head_dim = q.shape
    if head_dim != 128 or k_pool.shape[1] != 16:
        raise ValueError("native grouped decode requires head_dim=128 and page_size=16")
    if query_heads != k_pool.shape[2] * 6:
        raise ValueError("native grouped decode requires six query heads per KV head")
    if not isinstance(split_k, int) or split_k < 2:
        raise ValueError("native grouped decode requires split_k >= 2")
    if scale is None:
        scale = head_dim ** -0.5

    expected_out = (batch, split_k, query_heads, head_dim)
    expected_stats = expected_out[:3]
    if partials is None:
        partial_out = torch.empty(expected_out, dtype=torch.float32, device=q.device)
        partial_max = torch.empty(expected_stats, dtype=torch.float32, device=q.device)
        partial_sum = torch.empty_like(partial_max)
    else:
        partial_out, partial_max, partial_sum = partials
        if tuple(partial_out.shape) != expected_out:
            raise ValueError("preallocated partial output has the wrong shape")
        if tuple(partial_max.shape) != expected_stats or partial_sum.shape != partial_max.shape:
            raise ValueError("preallocated partial statistics have the wrong shape")

    _extension().grouped_gqa_splitk_partial_out(
        q, k_pool, v_pool, block_table, seq_lens,
        partial_out, partial_max, partial_sum, split_k, scale,
    )

    # Reuse the already-tested stable split-K combine. Its cost is around 10 us
    # per layer at the target shapes, while the partial kernel dominates.
    from paged_decode_grouped_splitk import _splitk_reduce_kernel

    out = torch.empty_like(q)
    _splitk_reduce_kernel[(batch, query_heads)](
        partial_out, partial_max, partial_sum, out,
        partial_out.stride(0), partial_out.stride(1),
        partial_out.stride(2), partial_out.stride(3),
        partial_max.stride(0), partial_max.stride(1), partial_max.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        SPLIT_K=split_k,
        D_HEAD=head_dim,
        num_warps=4,
        num_stages=1,
    )
    return out
