#!/usr/bin/env python3
"""Isolate serial per-row versus batched greedy decode sampling."""

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
for path in (BENCHMARKS,):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from benchmark_core import parse_int_list
from run_benchmarks import system_metadata


CSV_FIELDS = (
    "batch_size", "vocab_size", "serial_median_ms", "batched_median_ms",
    "speedup", "serial_argmax_launches", "batched_argmax_launches", "status",
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", default="1,8,32,64,128")
    parser.add_argument("--vocab-size", type=int, default=151936)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir", type=Path,
        default=EXPERIMENTS / "results" / "batched-sampling",
    )
    return parser


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
    batches = parse_int_list(args.batch_sizes)
    if any(value < 1 for value in batches) or args.vocab_size < 1:
        raise ValueError("batch sizes and vocab size must be positive")
    if args.warmups < 0 or args.repetitions < 1:
        raise ValueError("invalid warmup or repetition count")

    import torch

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("a CUDA device is required")
    dtype = getattr(torch, args.dtype)
    rows = []
    for batch in batches:
        generator = torch.Generator(device=args.device).manual_seed(args.seed + batch)
        logits = torch.randn(
            batch, args.vocab_size, generator=generator,
            device=args.device, dtype=dtype,
        )
        serial = lambda: [int(logits[row].argmax()) for row in range(batch)]
        batched = lambda: logits.argmax(dim=-1).tolist()
        expected = serial()
        actual = batched()
        if actual != expected:
            raise AssertionError(f"batched sampling disagreed at batch {batch}")
        serial_raw = _measure(torch, serial, args.warmups, args.repetitions)
        batched_raw = _measure(torch, batched, args.warmups, args.repetitions)
        serial_ms = statistics.median(serial_raw)
        batched_ms = statistics.median(batched_raw)
        row = {
            "batch_size": batch,
            "vocab_size": args.vocab_size,
            "serial_median_ms": serial_ms,
            "batched_median_ms": batched_ms,
            "speedup": serial_ms / batched_ms,
            "serial_argmax_launches": batch,
            "batched_argmax_launches": 1,
            "serial_raw_ms": serial_raw,
            "batched_raw_ms": batched_raw,
            "status": "ok",
        }
        rows.append(row)
        print(
            f"B={batch:<3} serial={serial_ms:.4f} ms "
            f"batched={batched_ms:.4f} ms speedup={row['speedup']:.2f}x"
        )
        del logits
        torch.cuda.empty_cache()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    stem = args.output_dir / f"batched-sampling-{stamp}"
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "system": system_metadata(),
        "configuration": vars(args) | {"output_dir": str(args.output_dir)},
        "rows": rows,
    }
    stem.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")
    with stem.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in CSV_FIELDS} for row in rows)
    print(f"json: {stem.with_suffix('.json')}")
    print(f"csv: {stem.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
