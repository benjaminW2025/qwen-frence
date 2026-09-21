#!/usr/bin/env python3
"""Validate and time CTA-shared native GQA decode against current split-K.

This is deliberately a small go/no-go gate for the fixed serving regime. It
checks every measured shape against dense FP32 attention, records tokenwise
error statistics, measures CUDA-graph replay latency, and writes each completed
shape immediately so an interrupted GPU run retains useful results.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import partial
import json
from pathlib import Path
import statistics
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
for path in (ROOT / "custom_kernels", ROOT / "engine" / "kvcache", SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--batch-sizes", default="8,64")
    value.add_argument("--context-lengths", default="2048,4096")
    value.add_argument("--split-k", type=int, default=None,
                       help="Override the production fixed-regime split-K action")
    value.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    value.add_argument("--warmups", type=int, default=5)
    value.add_argument("--repetitions", type=int, default=20)
    value.add_argument("--graph-repeats", type=int, default=20)
    value.add_argument("--seed", type=int, default=20260914)
    value.add_argument("--output-dir", type=Path,
                       default=ROOT / "experiments" / "results" / "native-grouped-decode")
    return value


def measure(torch, operation, warmups, repetitions, graph_repeats):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmups):
            operation()
    side.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        for _ in range(graph_repeats):
            operation()
    torch.cuda.current_stream().wait_stream(side)
    for _ in range(warmups):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repetitions):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end) / graph_repeats)
    return {"median_ms": statistics.median(samples), "samples_ms": samples}


def error_report(torch, actual, expected):
    difference = (actual.float() - expected.float()).abs()
    tolerance = 2e-2 if expected.dtype == torch.bfloat16 else 2e-3
    return {
        "max_abs": difference.max().item(),
        "mean_abs": difference.mean().item(),
        "atol": tolerance,
        "rtol": tolerance,
        "mismatched": int(
            (difference > tolerance + tolerance * expected.float().abs()).sum()
        ),
        "elements": difference.numel(),
        "allclose": bool(torch.allclose(
            actual.float(), expected.float(), atol=tolerance, rtol=tolerance
        )),
    }


def persist(path, args, rows, torch, preflight):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema_version": 1,
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "config": {**vars(args), "output_dir": str(args.output_dir)},
        "correctness_preflight": preflight,
        "rows": rows,
    }, indent=2))


def main():
    args = parser().parse_args()
    batches = [int(item) for item in args.batch_sizes.split(",")]
    contexts = [int(item) for item in args.context_lengths.split(",")]
    if min(batches + contexts + [args.warmups, args.repetitions, args.graph_repeats]) < 1:
        raise ValueError("batches, contexts, warmups, repetitions, and graph repeats must be positive")

    import torch
    from grouped_splitk_validation import attention_reference, make_inputs
    from grouped_splitk_validation import correctness_cases
    from paged_decode_attention import select_splitk_config
    from paged_decode_grouped_splitk import grouped_splitk_decode_attention
    from paged_decode_native_grouped import native_grouped_splitk_decode_attention

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_dir / f"native-grouped-{stamp}.json"
    rows = []

    # Fail before timing on ragged rows, non-contiguous metadata/V, poisoned
    # padding, K greater than the page count, and large-value accumulation.
    preflight = []
    for case, tensors, options in correctness_cases(dtype=dtype, seed=args.seed):
        expected = attention_reference(*tensors, **options)
        actual = native_grouped_splitk_decode_attention(
            *tensors, split_k=8, **options
        )
        report = error_report(torch, actual, expected)
        preflight.append({"case": case, **report})
        if not report["allclose"]:
            raise AssertionError(f"native preflight failed for {case}: {report}")
    print(f"Correctness preflight passed ({len(preflight)} adversarial cases).", flush=True)

    for batch in batches:
        for context in contexts:
            tensors = make_inputs([context] * batch, page_size=16, dtype=dtype, seed=args.seed)
            split_k = args.split_k or select_splitk_config(context)["split_k"]
            partial_shape = (batch, split_k, 12, 128)
            native_partials = (
                torch.empty(partial_shape, dtype=torch.float32, device="cuda"),
                torch.empty(partial_shape[:3], dtype=torch.float32, device="cuda"),
                torch.empty(partial_shape[:3], dtype=torch.float32, device="cuda"),
            )
            triton_partials = tuple(torch.empty_like(item) for item in native_partials)
            native = partial(native_grouped_splitk_decode_attention, *tensors,
                             split_k=split_k, partials=native_partials)
            current = partial(grouped_splitk_decode_attention, *tensors,
                              split_k=split_k, heads_per_program=1,
                              num_warps=4, num_stages=select_splitk_config(context)["num_stages"],
                              partials=triton_partials)

            expected = attention_reference(*tensors)
            native_output = native()
            current_output = current()
            torch.cuda.synchronize()
            native_error = error_report(torch, native_output, expected)
            current_error = error_report(torch, current_output, expected)
            if not native_error["allclose"]:
                raise AssertionError(
                    f"native correctness failed at B={batch} C={context}: {native_error}"
                )

            # Alternate order across shapes to avoid consistently favoring one arm.
            operations = [("native", native), ("current", current)]
            if len(rows) % 2:
                operations.reverse()
            timings = {
                name: measure(torch, operation, args.warmups,
                              args.repetitions, args.graph_repeats)
                for name, operation in operations
            }
            native_ms = timings["native"]["median_ms"]
            current_ms = timings["current"]["median_ms"]
            unique_bytes = batch * 2 * context * 128 * 2 * 2
            row = {
                "batch_size": batch,
                "context_length": context,
                "split_k": split_k,
                "native": timings["native"],
                "current": timings["current"],
                "speedup": current_ms / native_ms,
                "native_unique_kv_gbps": unique_bytes / (native_ms * 1e-3) / 1e9,
                "current_sixfold_kv_gbps": unique_bytes * 6 / (current_ms * 1e-3) / 1e9,
                "native_error": native_error,
                "current_error": current_error,
            }
            rows.append(row)
            persist(output, args, rows, torch, preflight)
            print(
                f"B={batch:2d} C={context:4d} K={split_k:2d}: "
                f"current={current_ms:.4f} ms native={native_ms:.4f} ms "
                f"speedup={row['speedup']:.3f}x correctness={native_error['allclose']}",
                flush=True,
            )

    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
