#!/usr/bin/env python3
"""Separate decode launch tuning, kernel layout, and GQA-head sharing effects."""

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
for path in (BENCHMARKS, ROOT / "baseline", ROOT / "engine" / "kvcache"):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from benchmark_core import parse_int_list
from run_benchmarks import system_metadata


VARIANTS = (
    ("production-default", "production", None, None, None),
    ("production-w4-s2", "production", None, 4, 2),
    ("production-w8-s2", "production", None, 8, 2),
    ("candidate-h1-w4-s2", "candidate", 1, 4, 2),
    ("candidate-h1-w8-s2", "candidate", 1, 8, 2),
    ("candidate-h2-w4-s2", "candidate", 2, 4, 2),
    ("candidate-h3-w4-s2", "candidate", 3, 4, 2),
    ("candidate-h6-w4-s2", "candidate", 6, 4, 2),
)

CSV_FIELDS = (
    "batch_size", "context_length", "variant", "kernel_family",
    "heads_per_program", "num_warps", "num_stages", "median_ms",
    "speedup_vs_production_default", "max_abs_error", "status", "error",
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", default="8,64")
    parser.add_argument("--context-lengths", default="128,2048,8192,16384")
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir", type=Path,
        default=EXPERIMENTS / "results" / "decode-kernel-causality",
    )
    return parser


def _make_inputs(torch, batch, context, *, page_size, dtype, device, seed):
    query_heads, kv_heads, d_head = 12, 2, 128
    pages_per_sequence = (context + page_size - 1) // page_size
    total_pages = batch * pages_per_sequence
    generator = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(
        batch, query_heads, d_head, generator=generator, device=device, dtype=dtype,
    )
    k_pool = torch.randn(
        total_pages, page_size, kv_heads, d_head,
        generator=generator, device=device, dtype=dtype,
    )
    v_pool = torch.randn(
        total_pages, page_size, kv_heads, d_head,
        generator=generator, device=device, dtype=dtype,
    )
    block_table = torch.randperm(
        total_pages, generator=generator, device=device, dtype=torch.int64,
    ).reshape(batch, pages_per_sequence).to(torch.int32)
    seq_lens = torch.full((batch,), context, device=device, dtype=torch.int32)
    return q, k_pool, v_pool, block_table, seq_lens


def _operation(variant, production, candidate, tensors):
    name, family, heads, warps, stages = variant
    if family == "production":
        return lambda: production(
            *tensors, num_warps=warps, num_stages=stages,
        )
    return lambda: candidate(
        *tensors, heads_per_program=heads, num_warps=warps, num_stages=stages,
    )


def _measure(torch, operation, warmups, repetitions):
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


def main():
    args = build_parser().parse_args()
    batches = parse_int_list(args.batch_sizes)
    contexts = parse_int_list(args.context_lengths)
    if any(value < 1 for value in batches + contexts):
        raise ValueError("batch sizes and contexts must be positive")
    if args.page_size < 1 or args.warmups < 0 or args.repetitions < 1:
        raise ValueError("invalid page size, warmup count, or repetition count")

    import torch
    from kernel_dispatch import paged_decode_attention_candidate
    from paged_decode_attention import (
        decode_attention_reference,
        paged_decode_attention,
    )

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("a CUDA device is required")
    dtype = getattr(torch, args.dtype)

    print("ragged causal correctness preflight...")
    preflight_tensors = _make_inputs(
        torch, 7, 70, page_size=args.page_size, dtype=dtype,
        device=args.device, seed=args.seed,
    )
    preflight_tensors = (*preflight_tensors[:-1], torch.tensor(
        [1, 15, 16, 17, 31, 33, 70], device=args.device, dtype=torch.int32,
    ))
    manual = decode_attention_reference(*preflight_tensors)
    preflight = []
    available = set()
    for variant in VARIANTS:
        try:
            operation = _operation(
                variant, paged_decode_attention,
                paged_decode_attention_candidate, preflight_tensors,
            )
            actual = operation()
            torch.cuda.synchronize()
            error = (actual.float() - manual.float()).abs().max().item()
            torch.testing.assert_close(
                actual.float(), manual.float(), atol=5e-2, rtol=2e-2,
            )
            status, message = "ok", None
            available.add(variant[0])
        except Exception as exc:
            torch.cuda.synchronize()
            error = None
            status, message = "failed", f"{type(exc).__name__}: {exc}"
        preflight.append({
            "variant": variant[0], "max_abs_error": error,
            "status": status, "error": message,
        })
        suffix = f" max_err={error:.6f}" if error is not None else f" {message}"
        print(f"  {variant[0]}: {status.upper()}{suffix}")

    rows = []
    for batch in batches:
        for context in contexts:
            tensors = _make_inputs(
                torch, batch, context, page_size=args.page_size, dtype=dtype,
                device=args.device, seed=args.seed + batch * 1009 + context * 9173,
            )
            reference = paged_decode_attention(*tensors)
            shape_rows = []
            print(f"\nB={batch} C={context}")
            for variant in VARIANTS:
                name, family, heads, warps, stages = variant
                base = {
                    "batch_size": batch, "context_length": context,
                    "variant": name, "kernel_family": family,
                    "heads_per_program": heads, "num_warps": warps,
                    "num_stages": stages,
                }
                if name not in available:
                    row = base | {
                        "median_ms": None, "raw_ms": [],
                        "speedup_vs_production_default": None,
                        "max_abs_error": None, "status": "preflight_failed",
                        "error": next(x["error"] for x in preflight if x["variant"] == name),
                    }
                else:
                    try:
                        operation = _operation(
                            variant, paged_decode_attention,
                            paged_decode_attention_candidate, tensors,
                        )
                        actual = operation()
                        torch.cuda.synchronize()
                        error = (actual.float() - reference.float()).abs().max().item()
                        torch.testing.assert_close(
                            actual.float(), reference.float(), atol=5e-2, rtol=2e-2,
                        )
                        raw = _measure(
                            torch, operation, args.warmups, args.repetitions,
                        )
                        row = base | {
                            "median_ms": statistics.median(raw), "raw_ms": raw,
                            "speedup_vs_production_default": None,
                            "max_abs_error": error, "status": "ok", "error": None,
                        }
                    except Exception as exc:
                        torch.cuda.synchronize()
                        row = base | {
                            "median_ms": None, "raw_ms": [],
                            "speedup_vs_production_default": None,
                            "max_abs_error": None, "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                shape_rows.append(row)
            production = next(
                row for row in shape_rows if row["variant"] == "production-default"
            )
            if production["status"] != "ok":
                raise RuntimeError(f"production control failed at B={batch}, C={context}")
            for row in shape_rows:
                if row["status"] == "ok":
                    row["speedup_vs_production_default"] = (
                        production["median_ms"] / row["median_ms"]
                    )
                    print(
                        f"  {row['variant']:<24} {row['median_ms']:.4f} ms "
                        f"{row['speedup_vs_production_default']:.3f}x"
                    )
                else:
                    print(f"  {row['variant']:<24} FAILED {row['error']}")
            rows.extend(shape_rows)
            del tensors, reference
            torch.cuda.empty_cache()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    stem = args.output_dir / f"decode-kernel-causality-{stamp}"
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "system": system_metadata(),
        "configuration": vars(args) | {
            "output_dir": str(args.output_dir), "variants": [list(x) for x in VARIANTS],
        },
        "correctness_preflight": preflight,
        "rows": rows,
    }
    stem.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")
    with stem.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in CSV_FIELDS} for row in rows)
    print(f"\njson: {stem.with_suffix('.json')}")
    print(f"csv: {stem.with_suffix('.csv')}")
    candidate_successes = [
        row for row in rows
        if row["kernel_family"] == "candidate" and row["status"] == "ok"
    ]
    if not candidate_successes:
        raise SystemExit("no decode candidate completed successfully")


if __name__ == "__main__":
    main()
