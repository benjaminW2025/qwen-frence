#!/usr/bin/env python3
"""Sweep packed-prefill attention tile sizes against per-sequence SDPA."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
EXPERIMENTS_DIR = HERE.parent
ROOT = EXPERIMENTS_DIR.parent
BENCHMARKS_DIR = ROOT / "benchmarks"
for path in (
    BENCHMARKS_DIR,
    ROOT / "baseline",
    ROOT / "engine" / "kvcache",
):
    sys.path.insert(0, str(path))

from benchmark_core import parse_int_list
from benchmark_ragged_prefill import case_lengths, parse_patterns
from run_benchmarks import system_metadata


CSV_FIELDS = (
    "pattern",
    "batch_size",
    "max_prompt_length",
    "total_prompt_tokens",
    "causal_attention_pairs",
    "block_m",
    "block_n",
    "num_warps",
    "num_stages",
    "sdpa_median_ms",
    "triton_median_ms",
    "speedup",
    "triton_pairs_per_second",
    "max_abs_error",
    "status",
    "error",
)


def validate_tile_sizes(values: list[int], label: str) -> None:
    invalid = [value for value in values if value < 16 or value & (value - 1)]
    if not values or invalid:
        raise ValueError(f"{label} must contain powers of two >= 16; invalid={invalid}")


def tile_configurations(block_ms: list[int], block_ns: list[int]) -> list[tuple[int, int]]:
    validate_tile_sizes(block_ms, "block-ms")
    validate_tile_sizes(block_ns, "block-ns")
    return [(block_m, block_n) for block_m in block_ms for block_n in block_ns]


def parse_tile_configurations(value: str) -> list[tuple[int, int]]:
    """Parse an ordered, comma-separated list such as ``32x32,64x32``."""
    configurations = []
    for item in value.split(","):
        parts = item.strip().lower().split("x")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(
                "tile configs must use BLOCK_MxBLOCK_N, for example 64x32"
            )
        try:
            configuration = (int(parts[0]), int(parts[1]))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid tile configuration: {item!r}"
            ) from exc
        configurations.append(configuration)

    if not configurations:
        raise argparse.ArgumentTypeError("at least one tile configuration is required")
    if len(set(configurations)) != len(configurations):
        raise argparse.ArgumentTypeError("tile configurations must be unique")
    try:
        validate_tile_sizes([block_m for block_m, _ in configurations], "block-ms")
        validate_tile_sizes([block_n for _, block_n in configurations], "block-ns")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return configurations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", default="4,8")
    parser.add_argument("--prompt-lengths", default="512,2048,4096")
    parser.add_argument("--patterns", type=parse_patterns, default=["uniform", "ramp"])
    parser.add_argument("--block-ms", default="32,64,128")
    parser.add_argument("--block-ns", default="32,64,128")
    parser.add_argument(
        "--tile-configs",
        type=parse_tile_configurations,
        help=(
            "ordered comma-separated BLOCK_MxBLOCK_N pairs; when provided, this "
            "replaces the Cartesian product from --block-ms and --block-ns"
        ),
    )
    parser.add_argument("--num-warps", type=int, choices=(1, 2, 4, 8), default=4)
    parser.add_argument("--num-stages", type=int, default=2)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXPERIMENTS_DIR / "results" / "packed-attention-tiles",
    )
    return parser


def validate_args(args, batch_sizes, prompt_lengths, configurations) -> None:
    if any(value < 1 for value in batch_sizes + prompt_lengths):
        raise ValueError("batch sizes and prompt lengths must be positive")
    if args.warmups < 0 or args.repetitions < 1:
        raise ValueError("warmups must be >= 0 and repetitions must be >= 1")
    if args.num_stages < 1:
        raise ValueError("num-stages must be positive")
    if not configurations:
        raise ValueError("at least one tile configuration is required")


def main() -> None:
    args = build_parser().parse_args()
    batch_sizes = parse_int_list(args.batch_sizes)
    prompt_lengths = parse_int_list(args.prompt_lengths)
    block_ms = parse_int_list(args.block_ms)
    block_ns = parse_int_list(args.block_ns)
    configurations = args.tile_configs or tile_configurations(block_ms, block_ns)
    validate_args(args, batch_sizes, prompt_lengths, configurations)

    import torch

    from packed_prefill_attention import packed_prefill_attention_sdpa
    from kernel_dispatch import packed_prefill_attention as packed_prefill_attention_triton

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("A CUDA device is required")
    dtype = getattr(torch, args.dtype)
    query_heads, kv_heads, d_head = 12, 2, 128
    group = query_heads // kv_heads

    def measure(operation):
        for _ in range(args.warmups):
            operation()
        torch.cuda.synchronize()
        timings = []
        for _ in range(args.repetitions):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            operation()
            end.record()
            end.synchronize()
            timings.append(float(start.elapsed_time(end)))
        return timings

    rows = []
    for pattern in args.patterns:
        for batch_size in batch_sizes:
            for max_length in prompt_lengths:
                lengths = case_lengths(pattern, batch_size, max_length)
                offsets = [0]
                for length in lengths:
                    offsets.append(offsets[-1] + length)
                total_tokens = offsets[-1]
                attention_pairs = sum(length * (length + 1) // 2 for length in lengths)
                generator = torch.Generator(device=args.device).manual_seed(
                    args.seed + batch_size * 1009 + max_length * 9173 + (pattern == "ramp")
                )
                q = torch.randn(
                    total_tokens,
                    query_heads,
                    d_head,
                    generator=generator,
                    device=args.device,
                    dtype=dtype,
                ).transpose(0, 1).unsqueeze(0)
                k = torch.randn(
                    total_tokens,
                    kv_heads,
                    d_head,
                    generator=generator,
                    device=args.device,
                    dtype=dtype,
                ).transpose(0, 1).unsqueeze(0)
                v = torch.randn(
                    total_tokens,
                    kv_heads,
                    d_head,
                    generator=generator,
                    device=args.device,
                    dtype=dtype,
                ).transpose(0, 1).unsqueeze(0)
                cu_seqlens = torch.tensor(offsets, device=args.device, dtype=torch.int32)

                reference = packed_prefill_attention_sdpa(q, k, v, offsets, group)
                sdpa_raw_ms = measure(
                    lambda: packed_prefill_attention_sdpa(q, k, v, offsets, group)
                )
                sdpa_median = statistics.median(sdpa_raw_ms)

                print(
                    f"\n{pattern} B={batch_size} Lmax={max_length} "
                    f"tokens={total_tokens} sdpa={sdpa_median:.4f} ms"
                )
                for block_m, block_n in configurations:
                    base = {
                        "pattern": pattern,
                        "batch_size": batch_size,
                        "max_prompt_length": max_length,
                        "prompt_lengths": lengths,
                        "total_prompt_tokens": total_tokens,
                        "causal_attention_pairs": attention_pairs,
                        "block_m": block_m,
                        "block_n": block_n,
                        "num_warps": args.num_warps,
                        "num_stages": args.num_stages,
                        "sdpa_median_ms": sdpa_median,
                        "sdpa_raw_ms": sdpa_raw_ms,
                    }
                    try:
                        operation = lambda bm=block_m, bn=block_n: (
                            packed_prefill_attention_triton(
                                q,
                                k,
                                v,
                                cu_seqlens,
                                max(lengths),
                                block_m=bm,
                                block_n=bn,
                                num_warps=args.num_warps,
                                num_stages=args.num_stages,
                            )
                        )
                        actual = operation()  # compile before correctness and timing
                        torch.cuda.synchronize()
                        max_error = (actual.float() - reference.float()).abs().max().item()
                        atol = 6e-2 if dtype == torch.float16 else 1.5e-1
                        rtol = 2e-2 if dtype == torch.float16 else 4e-2
                        torch.testing.assert_close(
                            actual.float(), reference.float(), atol=atol, rtol=rtol
                        )
                        raw_ms = measure(operation)
                        median_ms = statistics.median(raw_ms)
                        row = base | {
                            "triton_median_ms": median_ms,
                            "triton_raw_ms": raw_ms,
                            "speedup": sdpa_median / median_ms,
                            "triton_pairs_per_second": attention_pairs / (median_ms * 1e-3),
                            "max_abs_error": max_error,
                            "status": "ok",
                            "error": None,
                        }
                        print(
                            f"  M={block_m:<3} N={block_n:<3} {median_ms:>8.4f} ms "
                            f"{row['speedup']:>6.3f}x err={max_error:.5f}"
                        )
                    except Exception as exc:
                        torch.cuda.synchronize()
                        row = base | {
                            "triton_median_ms": None,
                            "triton_raw_ms": [],
                            "speedup": None,
                            "triton_pairs_per_second": None,
                            "max_abs_error": None,
                            "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                        print(f"  M={block_m:<3} N={block_n:<3} FAILED {row['error']}")
                    rows.append(row)

                del q, k, v, cu_seqlens, reference
                torch.cuda.empty_cache()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    json_path = args.output_dir / f"packed-attention-tiles-{stamp}.json"
    csv_path = args.output_dir / f"packed-attention-tiles-{stamp}.csv"
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "system": system_metadata(),
        "configuration": {
            "dtype": args.dtype,
            "query_heads": query_heads,
            "kv_heads": kv_heads,
            "d_head": d_head,
            "batch_sizes": batch_sizes,
            "prompt_lengths": prompt_lengths,
            "patterns": args.patterns,
            "block_ms": block_ms,
            "block_ns": block_ns,
            "tile_configurations": [list(configuration) for configuration in configurations],
            "num_warps": args.num_warps,
            "num_stages": args.num_stages,
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "seed": args.seed,
        },
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in CSV_FIELDS} for row in rows)
    print(f"\njson: {json_path}")
    print(f"csv : {csv_path}")

    if not any(row["status"] == "ok" for row in rows):
        raise RuntimeError("every Triton tile configuration failed")


if __name__ == "__main__":
    main()
