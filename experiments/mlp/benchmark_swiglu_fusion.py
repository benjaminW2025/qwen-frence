#!/usr/bin/env python3
"""Measure fused SwiGLU across packed-row counts from the regime atlas."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
ROOT = EXPERIMENTS.parent
BENCHMARKS = ROOT / "benchmarks"
for path in (BENCHMARKS, ROOT / "baseline"):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from benchmark_core import parse_int_list
from run_benchmarks import system_metadata


CSV_FIELDS = (
    "rows", "hidden_size", "block_size", "num_warps",
    "baseline_median_ms", "fused_median_ms", "speedup",
    "baseline_effective_gbps", "fused_effective_gbps", "max_abs_error",
    "status", "error",
)

SUMMARY_FIELDS = (
    "rows", "hidden_size", "baseline_median_ms", "best_fused_median_ms",
    "best_speedup", "best_block_size", "best_num_warps",
    "successful_configurations", "failed_configurations",
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows",
        default="1,8,32,64,520,2048,4128,8200,16384",
        help="packed token-row counts; 520/4128/8200 match atlas regimes",
    )
    parser.add_argument("--hidden-size", type=int, default=8960)
    parser.add_argument("--block-sizes", default="128,256,512,1024")
    parser.add_argument("--num-warps", default="4,8")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir", type=Path,
        default=EXPERIMENTS / "results" / "swiglu-fusion",
    )
    return parser


def validate_args(args, rows, block_sizes, warp_counts):
    if any(value < 1 for value in rows):
        raise ValueError("row counts must be positive")
    if args.hidden_size < 1:
        raise ValueError("hidden size must be positive")
    if any(value not in (128, 256, 512, 1024) for value in block_sizes):
        raise ValueError("block sizes must contain 128, 256, 512, or 1024")
    if any(value not in (1, 2, 4, 8) for value in warp_counts):
        raise ValueError("num-warps must contain 1, 2, 4, or 8")
    if args.warmups < 0 or args.repetitions < 1:
        raise ValueError("warmups must be non-negative and repetitions positive")


def _measure(torch, operation, *, warmups, repetitions):
    for _ in range(warmups):
        operation()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return samples


def _correctness_preflight(torch, fused, *, dtype, device, configurations, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    gate = torch.randn(17, 257, generator=generator, device=device, dtype=dtype)
    up = torch.randn(17, 257, generator=generator, device=device, dtype=dtype)
    reference = torch.nn.functional.silu(gate) * up
    records = []
    available = set()
    for block_size, num_warps in configurations:
        try:
            actual = fused(
                gate, up, block_size=block_size, num_warps=num_warps,
            )
            torch.cuda.synchronize()
            max_error = (actual.float() - reference.float()).abs().max().item()
            torch.testing.assert_close(
                actual.float(), reference.float(), atol=2e-2, rtol=2e-2,
            )
            status, error = "ok", None
            available.add((block_size, num_warps))
        except Exception as exc:
            torch.cuda.synchronize()
            max_error = None
            status, error = "failed", f"{type(exc).__name__}: {exc}"
        records.append({
            "shape": [17, 257],
            "block_size": block_size,
            "num_warps": num_warps,
            "max_abs_error": max_error,
            "status": status,
            "error": error,
        })
        suffix = f" max_err={max_error:.6f}" if max_error is not None else f" {error}"
        print(
            f"  preflight block={block_size} warps={num_warps}: "
            f"{status.upper()}{suffix}"
        )
    if not available:
        raise RuntimeError("every fused SwiGLU configuration failed preflight")
    return records, available


def main():
    args = build_parser().parse_args()
    rows_axis = parse_int_list(args.rows)
    block_sizes = parse_int_list(args.block_sizes)
    warp_counts = parse_int_list(args.num_warps)
    validate_args(args, rows_axis, block_sizes, warp_counts)

    import torch
    from kernel_dispatch import swiglu

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("a CUDA device is required")
    dtype = getattr(torch, args.dtype)
    configurations = [
        (block_size, num_warps)
        for block_size in block_sizes
        for num_warps in warp_counts
    ]

    print("fused SwiGLU correctness preflight...")
    preflight, available = _correctness_preflight(
        torch, swiglu, dtype=dtype, device=args.device,
        configurations=configurations, seed=args.seed,
    )

    rows = []
    element_size = torch.empty((), dtype=dtype).element_size()
    for row_count in rows_axis:
        generator = torch.Generator(device=args.device).manual_seed(
            args.seed + row_count * 1009
        )
        gate = torch.randn(
            row_count, args.hidden_size, generator=generator,
            device=args.device, dtype=dtype,
        )
        up = torch.randn(
            row_count, args.hidden_size, generator=generator,
            device=args.device, dtype=dtype,
        )
        baseline = lambda: torch.nn.functional.silu(gate) * up
        reference = baseline()
        baseline_raw = _measure(
            torch, baseline, warmups=args.warmups, repetitions=args.repetitions,
        )
        baseline_ms = statistics.median(baseline_raw)
        elements = row_count * args.hidden_size
        baseline_bytes = 5 * elements * element_size
        fused_bytes = 3 * elements * element_size
        print(f"\nrows={row_count} baseline={baseline_ms:.4f} ms")

        for block_size, num_warps in configurations:
            base = {
                "rows": row_count,
                "hidden_size": args.hidden_size,
                "block_size": block_size,
                "num_warps": num_warps,
                "baseline_median_ms": baseline_ms,
                "baseline_raw_ms": baseline_raw,
                "baseline_effective_gbps": baseline_bytes / (baseline_ms * 1e-3) / 1e9,
                "baseline_estimated_bytes": baseline_bytes,
                "fused_estimated_bytes": fused_bytes,
            }
            if (block_size, num_warps) not in available:
                failure = next(
                    record for record in preflight
                    if record["block_size"] == block_size
                    and record["num_warps"] == num_warps
                )
                rows.append(base | {
                    "fused_median_ms": None,
                    "fused_raw_ms": [],
                    "speedup": None,
                    "fused_effective_gbps": None,
                    "max_abs_error": None,
                    "status": "preflight_failed",
                    "error": failure["error"],
                })
                continue
            try:
                operation = lambda bs=block_size, nw=num_warps: swiglu(
                    gate, up, block_size=bs, num_warps=nw,
                )
                actual = operation()
                torch.cuda.synchronize()
                max_error = (actual.float() - reference.float()).abs().max().item()
                torch.testing.assert_close(
                    actual.float(), reference.float(), atol=2e-2, rtol=2e-2,
                )
                raw = _measure(
                    torch, operation,
                    warmups=args.warmups, repetitions=args.repetitions,
                )
                median_ms = statistics.median(raw)
                row = base | {
                    "fused_median_ms": median_ms,
                    "fused_raw_ms": raw,
                    "speedup": baseline_ms / median_ms,
                    "fused_effective_gbps": fused_bytes / (median_ms * 1e-3) / 1e9,
                    "max_abs_error": max_error,
                    "status": "ok",
                    "error": None,
                }
                print(
                    f"  block={block_size} warps={num_warps}: "
                    f"{median_ms:.4f} ms {row['speedup']:.3f}x"
                )
            except Exception as exc:
                torch.cuda.synchronize()
                row = base | {
                    "fused_median_ms": None,
                    "fused_raw_ms": [],
                    "speedup": None,
                    "fused_effective_gbps": None,
                    "max_abs_error": None,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                print(
                    f"  block={block_size} warps={num_warps}: FAILED {row['error']}"
                )
            rows.append(row)

        del gate, up, reference
        torch.cuda.empty_cache()

    summaries = []
    for row_count in rows_axis:
        shape_rows = [row for row in rows if row["rows"] == row_count]
        successful = [row for row in shape_rows if row["status"] == "ok"]
        best = min(successful, key=lambda row: row["fused_median_ms"])
        summaries.append({
            "rows": row_count,
            "hidden_size": args.hidden_size,
            "baseline_median_ms": shape_rows[0]["baseline_median_ms"],
            "best_fused_median_ms": best["fused_median_ms"],
            "best_speedup": best["speedup"],
            "best_block_size": best["block_size"],
            "best_num_warps": best["num_warps"],
            "successful_configurations": len(successful),
            "failed_configurations": len(shape_rows) - len(successful),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    stem = args.output_dir / f"swiglu-fusion-{stamp}"
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "system": system_metadata(),
        "configuration": {
            "rows": rows_axis,
            "hidden_size": args.hidden_size,
            "block_sizes": block_sizes,
            "num_warps": warp_counts,
            "dtype": args.dtype,
            "device": args.device,
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "seed": args.seed,
        },
        "traffic_model": {
            "baseline_bytes_per_element": 5 * element_size,
            "fused_bytes_per_element": 3 * element_size,
            "note": "baseline materializes SiLU; fused reads gate/up and writes output",
        },
        "correctness_preflight": preflight,
        "rows": rows,
        "shape_summaries": summaries,
    }
    stem.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")
    with stem.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in CSV_FIELDS} for row in rows)
    summary_path = stem.parent / f"{stem.name}-summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summaries)
    print(f"\njson: {stem.with_suffix('.json')}")
    print(f"csv: {stem.with_suffix('.csv')}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
