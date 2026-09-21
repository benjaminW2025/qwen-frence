#!/usr/bin/env python3
"""Emit one isolated current/native grouped-decode launch for Nsight Compute.

Run this script under ``ncu --set full``. Compilation, allocation, correctness,
and warmup occur before cudaProfilerStart; the report contains exactly one
attention partial and its small split-K reduction inside a named NVTX range.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
for path in (ROOT / "custom_kernels", ROOT / "engine" / "kvcache", SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("current", "native"), required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--split-k", type=int, default=22)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260914)
    return parser


def main():
    args = build_parser().parse_args()
    if min(args.batch_size, args.context_length, args.split_k, args.warmups) < 1:
        raise ValueError("batch, context, split-K, and warmups must be positive")

    import torch
    from grouped_splitk_validation import make_inputs
    from paged_decode_grouped_splitk import grouped_splitk_decode_attention
    from paged_decode_native_grouped import native_grouped_splitk_decode_attention

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    dtype = getattr(torch, args.dtype)
    tensors = make_inputs(
        [args.context_length] * args.batch_size,
        page_size=16,
        dtype=dtype,
        seed=args.seed,
    )
    shape = (args.batch_size, args.split_k, 12, 128)

    def buffers():
        output = torch.empty(shape, dtype=torch.float32, device="cuda")
        maximum = torch.empty(shape[:3], dtype=torch.float32, device="cuda")
        return output, maximum, torch.empty_like(maximum)

    current_partials = buffers()
    native_partials = buffers()

    def current():
        return grouped_splitk_decode_attention(
            *tensors,
            split_k=args.split_k,
            heads_per_program=1,
            num_warps=4,
            num_stages=3,
            partials=current_partials,
        )

    def native():
        return native_grouped_splitk_decode_attention(
            *tensors,
            split_k=args.split_k,
            partials=native_partials,
        )

    # Compile both kernels and prove that the isolated run compares equivalent
    # outputs before enabling the profiler. This work is absent from the report.
    expected = current()
    actual = native()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-3, rtol=2e-3)

    operation = current if args.arm == "current" else native
    for _ in range(args.warmups):
        operation()
    torch.cuda.synchronize()

    label = f"grouped_decode_{args.arm}"
    cudart = torch.cuda.cudart()
    result = cudart.cudaProfilerStart()
    if result != 0:
        raise RuntimeError(f"cudaProfilerStart failed with status {result}")
    torch.cuda.nvtx.range_push(label)
    try:
        operation()
    finally:
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    result = cudart.cudaProfilerStop()
    if result != 0:
        raise RuntimeError(f"cudaProfilerStop failed with status {result}")

    print(
        f"profiled arm={args.arm} B={args.batch_size} "
        f"C={args.context_length} K={args.split_k} range={label}",
        flush=True,
    )


if __name__ == "__main__":
    main()
