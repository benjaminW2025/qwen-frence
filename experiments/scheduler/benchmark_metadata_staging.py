#!/usr/bin/env python3
"""Decompose mixed-iteration metadata construction, transfer, and buffer reuse."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
import time


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
ROOT = EXPERIMENTS.parent
BENCHMARKS = ROOT / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

from run_benchmarks import system_metadata


REGIMES = (
    ("launch-bound", 8, 128, 2, 256, 0),
    ("decode-heavy-low", 8, 8192, 2, 256, 0),
    ("balanced-fresh", 32, 2048, 2, 2048, 0),
    ("balanced-resumed", 32, 2048, 2, 2048, 4096),
    ("decode-heavy-high", 64, 8192, 2, 256, 0),
    ("long-fresh-prefill", 8, 128, 2, 8192, 0),
    ("prefill-heavy", 8, 128, 2, 4096, 16384),
    ("dual-heavy", 64, 8192, 2, 4096, 16384),
)

CSV_FIELDS = (
    "regime", "decode_requests", "decode_context", "prefill_requests",
    "prefill_query", "prefill_prefix", "metadata_bytes", "variant",
    "median_wall_ms", "speedup_vs_current", "status",
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument(
        "--output-dir", type=Path,
        default=EXPERIMENTS / "results" / "metadata-staging",
    )
    return parser


def _values(regime, block_size):
    name, decode, context, prefills, query, prefix = regime
    total_tokens = decode + prefills * query
    decode_blocks = (context + block_size - 1) // block_size
    prefill_context = prefix + query
    prefill_blocks = (prefill_context + block_size - 1) // block_size
    offsets = [index * query for index in range(prefills + 1)]
    return (
        list(range(total_tokens)),
        [float(index) for index in range(total_tokens)],
        list(range(total_tokens)),
        [list(range(decode_blocks)) for _ in range(decode)],
        [context] * decode,
        [list(range(prefill_blocks)) for _ in range(prefills)],
        [prefill_context] * prefills,
        offsets,
        [decode + end - 1 for end in offsets[1:]],
    )


def _dtypes(torch):
    return (
        torch.long, torch.float32, torch.long, torch.int32, torch.int32,
        torch.int32, torch.int32, torch.int32, torch.long,
    )


def _measure(torch, operation, warmups, repetitions):
    for _ in range(warmups):
        operation()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        operation()
        torch.cuda.synchronize()
        samples.append((time.perf_counter_ns() - started) / 1e6)
    return samples


def main():
    args = build_parser().parse_args()
    if args.block_size < 1 or args.warmups < 0 or args.repetitions < 1:
        raise ValueError("invalid block size, warmup count, or repetition count")

    import torch

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("a CUDA device is required")
    rows = []
    for regime in REGIMES:
        values = _values(regime, args.block_size)
        dtypes = _dtypes(torch)
        pageable = tuple(
            torch.tensor(value, dtype=dtype)
            for value, dtype in zip(values, dtypes)
        )
        pinned = tuple(tensor.pin_memory() for tensor in pageable)
        buffers = tuple(
            torch.empty_like(tensor, device=args.device)
            for tensor in pageable
        )

        def current():
            return tuple(
                torch.tensor(value, dtype=dtype, device=args.device)
                for value, dtype in zip(values, dtypes)
            )

        def prebuilt_pageable():
            return tuple(tensor.to(args.device) for tensor in pageable)

        def pinned_reuse():
            for output, source in zip(buffers, pinned):
                output.copy_(source, non_blocking=True)
            return buffers

        expected = current()
        actual = pinned_reuse()
        torch.cuda.synchronize()
        for lhs, rhs in zip(expected, actual):
            if not torch.equal(lhs, rhs):
                raise AssertionError(f"metadata mismatch in {regime[0]}")
        metadata_bytes = sum(tensor.numel() * tensor.element_size() for tensor in pageable)
        variants = (
            ("current-python-to-device", current),
            ("prebuilt-pageable-new-device", prebuilt_pageable),
            ("pinned-reused-device", pinned_reuse),
        )
        measured = []
        for name, operation in variants:
            raw = _measure(torch, operation, args.warmups, args.repetitions)
            measured.append((name, raw, statistics.median(raw)))
        baseline = measured[0][2]
        print(f"\n{regime[0]} metadata={metadata_bytes / 1024:.1f} KiB")
        for name, raw, median in measured:
            row = {
                "regime": regime[0], "decode_requests": regime[1],
                "decode_context": regime[2], "prefill_requests": regime[3],
                "prefill_query": regime[4], "prefill_prefix": regime[5],
                "metadata_bytes": metadata_bytes, "variant": name,
                "median_wall_ms": median, "raw_wall_ms": raw,
                "speedup_vs_current": baseline / median, "status": "ok",
            }
            rows.append(row)
            print(
                f"  {name:<30} {median:.4f} ms "
                f"{row['speedup_vs_current']:.2f}x"
            )
        del pageable, pinned, buffers, expected, actual
        torch.cuda.empty_cache()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    stem = args.output_dir / f"metadata-staging-{stamp}"
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "system": system_metadata(),
        "configuration": vars(args) | {"output_dir": str(args.output_dir)},
        "regimes": [list(regime) for regime in REGIMES],
        "rows": rows,
        "interpretation": (
            "pinned reuse is an upper bound because host metadata values are staged "
            "before timing; current includes Python-list conversion and device allocation"
        ),
    }
    stem.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")
    with stem.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in CSV_FIELDS} for row in rows)
    print(f"\njson: {stem.with_suffix('.json')}")
    print(f"csv: {stem.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
