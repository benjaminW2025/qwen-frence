#!/usr/bin/env python3
"""Test the kernel-level upper bound from overlapping decode and prefill attention."""

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
    "prefill_query", "prefill_prefix", "decode_median_ms", "prefill_median_ms",
    "sequential_median_ms", "overlap_median_ms", "speedup",
    "hidden_fraction", "status", "error",
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir", type=Path,
        default=EXPERIMENTS / "results" / "attention-overlap",
    )
    return parser


def _decode_inputs(torch, batch, context, args, dtype, seed):
    query_heads, kv_heads, head_dim = 12, 2, 128
    pages_per_sequence = (context + args.page_size - 1) // args.page_size
    total_pages = batch * pages_per_sequence
    generator = torch.Generator(device=args.device).manual_seed(seed)
    q = torch.randn(
        batch, query_heads, head_dim, generator=generator,
        device=args.device, dtype=dtype,
    )
    k = torch.randn(
        total_pages, args.page_size, kv_heads, head_dim,
        generator=generator, device=args.device, dtype=dtype,
    )
    v = torch.randn(
        total_pages, args.page_size, kv_heads, head_dim,
        generator=generator, device=args.device, dtype=dtype,
    )
    table = torch.arange(
        total_pages, device=args.device, dtype=torch.int32,
    ).reshape(batch, pages_per_sequence)
    lengths = torch.full((batch,), context, device=args.device, dtype=torch.int32)
    return q, k, v, table, lengths


def _prefill_inputs(torch, batch, query, prefix, args, dtype, seed):
    query_heads, kv_heads, head_dim = 12, 2, 128
    context = prefix + query
    pages_per_sequence = (context + args.page_size - 1) // args.page_size
    total_pages = batch * pages_per_sequence
    generator = torch.Generator(device=args.device).manual_seed(seed)
    q = torch.randn(
        1, query_heads, batch * query, head_dim,
        generator=generator, device=args.device, dtype=dtype,
    )
    k = torch.randn(
        total_pages, args.page_size, kv_heads, head_dim,
        generator=generator, device=args.device, dtype=dtype,
    )
    v = torch.randn(
        total_pages, args.page_size, kv_heads, head_dim,
        generator=generator, device=args.device, dtype=dtype,
    )
    offsets = torch.arange(
        0, (batch + 1) * query, query, device=args.device, dtype=torch.int32,
    )
    table = torch.arange(
        total_pages, device=args.device, dtype=torch.int32,
    ).reshape(batch, pages_per_sequence)
    lengths = torch.full((batch,), context, device=args.device, dtype=torch.int32)
    return q, k, v, offsets, table, lengths


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
    if args.page_size < 1 or args.warmups < 0 or args.repetitions < 1:
        raise ValueError("invalid page size, warmup count, or repetition count")

    import torch
    from kernel_dispatch import packed_paged_prefill_attention
    from paged_decode_attention import paged_decode_attention

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("a CUDA device is required")
    dtype = getattr(torch, args.dtype)
    rows = []
    for index, regime in enumerate(REGIMES):
        name, decode_batch, decode_context, prefill_batch, query, prefix = regime
        decode_tensors = _decode_inputs(
            torch, decode_batch, decode_context, args, dtype,
            args.seed + index * 1009,
        )
        prefill_tensors = _prefill_inputs(
            torch, prefill_batch, query, prefix, args, dtype,
            args.seed + index * 9173 + 1,
        )

        def decode_attention():
            return paged_decode_attention(*decode_tensors)

        def prefill_attention():
            return packed_paged_prefill_attention(
                *prefill_tensors,
                max_query_len=query,
                page_size=args.page_size,
                tile_policy="static",
            )

        def sequential():
            return decode_attention(), prefill_attention()

        decode_stream = torch.cuda.Stream()
        prefill_stream = torch.cuda.Stream()
        ready = torch.cuda.Event()
        decode_done = torch.cuda.Event()
        prefill_done = torch.cuda.Event()

        def overlapped():
            current = torch.cuda.current_stream()
            ready.record(current)
            decode_stream.wait_event(ready)
            prefill_stream.wait_event(ready)
            with torch.cuda.stream(decode_stream):
                decode_output = decode_attention()
                decode_done.record()
            with torch.cuda.stream(prefill_stream):
                prefill_output = prefill_attention()
                prefill_done.record()
            current.wait_event(decode_done)
            current.wait_event(prefill_done)
            return decode_output, prefill_output

        try:
            reference_decode, reference_prefill = sequential()
            actual_decode, actual_prefill = overlapped()
            torch.cuda.synchronize()
            torch.testing.assert_close(actual_decode, reference_decode, atol=0, rtol=0)
            torch.testing.assert_close(actual_prefill, reference_prefill, atol=0, rtol=0)
            decode_raw = _measure(
                torch, decode_attention, args.warmups, args.repetitions,
            )
            prefill_raw = _measure(
                torch, prefill_attention, args.warmups, args.repetitions,
            )
            sequential_raw = _measure(
                torch, sequential, args.warmups, args.repetitions,
            )
            overlap_raw = _measure(
                torch, overlapped, args.warmups, args.repetitions,
            )
            decode_ms = statistics.median(decode_raw)
            prefill_ms = statistics.median(prefill_raw)
            sequential_ms = statistics.median(sequential_raw)
            overlap_ms = statistics.median(overlap_raw)
            smaller = min(decode_ms, prefill_ms)
            hidden_fraction = (
                max(0.0, sequential_ms - overlap_ms) / smaller if smaller else 0.0
            )
            row = {
                "regime": name, "decode_requests": decode_batch,
                "decode_context": decode_context,
                "prefill_requests": prefill_batch, "prefill_query": query,
                "prefill_prefix": prefix, "decode_median_ms": decode_ms,
                "prefill_median_ms": prefill_ms,
                "sequential_median_ms": sequential_ms,
                "overlap_median_ms": overlap_ms,
                "speedup": sequential_ms / overlap_ms,
                "hidden_fraction": hidden_fraction,
                "decode_raw_ms": decode_raw, "prefill_raw_ms": prefill_raw,
                "sequential_raw_ms": sequential_raw, "overlap_raw_ms": overlap_raw,
                "status": "ok", "error": None,
            }
            print(
                f"{name:<18} decode={decode_ms:.3f} prefill={prefill_ms:.3f} "
                f"seq={sequential_ms:.3f} overlap={overlap_ms:.3f} "
                f"speedup={row['speedup']:.3f}x hidden={hidden_fraction:.1%}"
            )
        except Exception as exc:
            torch.cuda.synchronize()
            row = {
                "regime": name, "decode_requests": decode_batch,
                "decode_context": decode_context,
                "prefill_requests": prefill_batch, "prefill_query": query,
                "prefill_prefix": prefix, "decode_median_ms": None,
                "prefill_median_ms": None, "sequential_median_ms": None,
                "overlap_median_ms": None, "speedup": None,
                "hidden_fraction": None, "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"{name:<18} FAILED {row['error']}")
        rows.append(row)
        del decode_tensors, prefill_tensors
        torch.cuda.empty_cache()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    stem = args.output_dir / f"attention-overlap-{stamp}"
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "system": system_metadata(),
        "configuration": vars(args) | {"output_dir": str(args.output_dir)},
        "regimes": [list(regime) for regime in REGIMES],
        "rows": rows,
        "interpretation": (
            "attention-only upper bound; end-to-end overlap also requires layer "
            "dependency events and an output join"
        ),
    }
    stem.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")
    with stem.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in CSV_FIELDS} for row in rows)
    print(f"\njson: {stem.with_suffix('.json')}")
    print(f"csv: {stem.with_suffix('.csv')}")
    failures = [row["regime"] for row in rows if row["status"] != "ok"]
    if failures:
        raise SystemExit("attention overlap failed for: " + ", ".join(failures))


if __name__ == "__main__":
    main()
