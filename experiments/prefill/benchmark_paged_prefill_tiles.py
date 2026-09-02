#!/usr/bin/env python3
"""Sweep paged-prefill attention configs over resumed-prefix shapes."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys


HERE = Path(__file__).resolve().parent
EXPERIMENTS_DIR = HERE.parent
ROOT = EXPERIMENTS_DIR.parent
BENCHMARKS_DIR = ROOT / "benchmarks"
for path in (BENCHMARKS_DIR, ROOT / "baseline"):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from benchmark_core import parse_int_list
from run_benchmarks import system_metadata


CSV_FIELDS = (
    "batch_size",
    "query_length",
    "prefix_length",
    "context_length",
    "attention_pairs",
    "block_m",
    "block_n",
    "num_warps",
    "num_stages",
    "sdpa_median_ms",
    "triton_median_ms",
    "speedup_vs_sdpa",
    "speedup_vs_production",
    "max_abs_error",
    "status",
    "error",
)
PRODUCTION_CONFIG = (64, 32, 4, 2)
DEFAULT_CONFIGS = (
    "32x16x4x2,32x32x2x2,32x32x4x2,64x32x4x2,"
    "64x64x4x2,128x64x8x2"
)


def parse_kernel_configs(value: str) -> list[tuple[int, int, int, int]]:
    """Parse ordered ``BLOCK_MxBLOCK_NxWARPSxSTAGES`` configurations."""
    configs = []
    for item in value.split(","):
        parts = item.strip().lower().split("x")
        if len(parts) != 4:
            raise argparse.ArgumentTypeError(
                "configs must use BLOCK_MxBLOCK_NxWARPSxSTAGES"
            )
        try:
            block_m, block_n, warps, stages = map(int, parts)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid config: {item!r}") from exc
        if (
            block_m < 16
            or block_m & (block_m - 1)
            or block_n < 16
            or block_n & (block_n - 1)
            or warps not in (1, 2, 4, 8)
            or stages < 1
        ):
            raise argparse.ArgumentTypeError(f"unsupported config: {item!r}")
        configs.append((block_m, block_n, warps, stages))
    if not configs or len(configs) != len(set(configs)):
        raise argparse.ArgumentTypeError("configs must be non-empty and unique")
    return configs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", default="1,2,4")
    parser.add_argument("--query-lengths", default="64,128,256,512,1024,2048")
    parser.add_argument("--prefix-lengths", default="0,128,512,2048,8192,16384")
    parser.add_argument("--configs", type=parse_kernel_configs,
                        default=parse_kernel_configs(DEFAULT_CONFIGS))
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXPERIMENTS_DIR / "results" / "paged-prefill-tiles",
    )
    return parser


def validate_args(args, batch_sizes, query_lengths, prefix_lengths) -> None:
    if any(value < 1 for value in batch_sizes + query_lengths):
        raise ValueError("batch sizes and query lengths must be positive")
    if any(value < 0 for value in prefix_lengths):
        raise ValueError("prefix lengths must be non-negative")
    if args.page_size < 1 or args.warmups < 0 or args.repetitions < 1:
        raise ValueError("page size/repetitions must be positive and warmups non-negative")
    if max(prefix_lengths) + max(query_lengths) > 32768:
        raise ValueError("prefix plus query exceeds the model context limit")


def _config_tuple(row):
    return tuple(
        int(row[name])
        for name in ("block_m", "block_n", "num_warps", "num_stages")
    )


def summarize_configs(rows):
    """Compare production, best static, and per-shape oracle configurations."""
    shape_names = ("batch_size", "query_length", "prefix_length")
    shapes = {}
    for row in rows:
        if row["status"] == "ok":
            key = tuple(int(row[name]) for name in shape_names)
            shapes.setdefault(key, []).append(row)
    if not shapes:
        raise ValueError("cannot summarize an empty successful sweep")

    production_total = 0.0
    oracle_total = 0.0
    winner_counts = {}
    for shape_rows in shapes.values():
        production = next(
            (row for row in shape_rows if _config_tuple(row) == PRODUCTION_CONFIG), None
        )
        if production is None:
            raise ValueError("every shape must contain the production control")
        winner = min(shape_rows, key=lambda row: float(row["triton_median_ms"]))
        production_total += float(production["triton_median_ms"])
        oracle_total += float(winner["triton_median_ms"])
        winner_counts[_config_tuple(winner)] = winner_counts.get(_config_tuple(winner), 0) + 1

    all_configs = sorted({_config_tuple(row) for values in shapes.values() for row in values})
    config_rows = []
    for config in all_configs:
        selected = []
        for shape_rows in shapes.values():
            match = next((row for row in shape_rows if _config_tuple(row) == config), None)
            if match is not None:
                selected.append(match)
        total_ms = sum(float(row["triton_median_ms"]) for row in selected)
        speedups = [float(row["speedup_vs_production"]) for row in selected]
        config_rows.append({
            "block_m": config[0],
            "block_n": config[1],
            "num_warps": config[2],
            "num_stages": config[3],
            "supported_shapes": len(selected),
            "total_shapes": len(shapes),
            "winner_count": winner_counts.get(config, 0),
            "aggregate_speedup_vs_production": (
                production_total / total_ms if len(selected) == len(shapes) else None
            ),
            "geomean_speedup_vs_production": (
                math.exp(sum(math.log(value) for value in speedups) / len(speedups))
                if speedups else None
            ),
        })
    eligible = [
        row for row in config_rows if row["supported_shapes"] == row["total_shapes"]
    ]
    if not eligible:
        raise ValueError("no static config succeeded on every shape")
    best_static = max(eligible, key=lambda row: row["aggregate_speedup_vs_production"])
    best_static_total = production_total / best_static["aggregate_speedup_vs_production"]
    best_config = tuple(best_static[name] for name in (
        "block_m", "block_n", "num_warps", "num_stages"
    ))
    for row in config_rows:
        row["best_static"] = _config_tuple(row) == best_config
    return {
        "shape_count": len(shapes),
        "production_config": list(PRODUCTION_CONFIG),
        "best_static_config": list(best_config),
        "best_static_speedup_vs_production": best_static["aggregate_speedup_vs_production"],
        "oracle_speedup_vs_production": production_total / oracle_total,
        "oracle_speedup_vs_best_static": best_static_total / oracle_total,
        "winner_counts": {
            "x".join(map(str, config)): count
            for config, count in sorted(winner_counts.items())
        },
        "configurations": config_rows,
    }


def paged_sdpa_reference(
    q, k_pool, v_pool, offsets, block_table, context_lengths, page_size
):
    import torch

    outputs = []
    group = q.shape[1] // k_pool.shape[2]
    scale = q.shape[-1] ** -0.5
    for row, context_length in enumerate(context_lengths):
        query_start, query_end = offsets[row:row + 2]
        query_length = query_end - query_start
        page_count = (context_length + page_size - 1) // page_size
        pages = block_table[row, :page_count].to(torch.long)
        keys = k_pool.index_select(0, pages).reshape(
            -1, k_pool.shape[2], k_pool.shape[3]
        )[:context_length]
        values = v_pool.index_select(0, pages).reshape(
            -1, v_pool.shape[2], v_pool.shape[3]
        )[:context_length]
        keys = keys.transpose(0, 1).repeat_interleave(group, dim=0).unsqueeze(0)
        values = values.transpose(0, 1).repeat_interleave(group, dim=0).unsqueeze(0)
        query = q[:, :, query_start:query_end]
        prefix_length = context_length - query_length
        query_positions = prefix_length + torch.arange(query_length, device=q.device)
        key_positions = torch.arange(context_length, device=q.device)
        mask = key_positions[None, :] <= query_positions[:, None]
        outputs.append(torch.nn.functional.scaled_dot_product_attention(
            query, keys, values, attn_mask=mask, scale=scale
        ))
    return torch.cat(outputs, dim=2)


def main() -> None:
    args = build_parser().parse_args()
    batch_sizes = parse_int_list(args.batch_sizes)
    query_lengths = parse_int_list(args.query_lengths)
    prefix_lengths = [
        int(part.strip()) for part in args.prefix_lengths.split(",") if part.strip()
    ]
    validate_args(args, batch_sizes, query_lengths, prefix_lengths)

    import torch
    from kernel_dispatch import packed_paged_prefill_attention

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("a CUDA device is required")
    dtype = getattr(torch, args.dtype)
    query_heads, kv_heads, d_head = 12, 2, 128

    def measure(operation):
        for _ in range(args.warmups):
            operation()
        torch.cuda.synchronize()
        samples = []
        for _ in range(args.repetitions):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            operation()
            end.record()
            end.synchronize()
            samples.append(float(start.elapsed_time(end)))
        return samples

    rows = []
    for batch_size in batch_sizes:
        for query_length in query_lengths:
            offsets = [index * query_length for index in range(batch_size + 1)]
            total_queries = offsets[-1]
            for prefix_length in prefix_lengths:
                context_length = prefix_length + query_length
                pages_per_sequence = (
                    context_length + args.page_size - 1
                ) // args.page_size
                total_pages = batch_size * pages_per_sequence
                generator = torch.Generator(device=args.device).manual_seed(
                    args.seed + batch_size * 1009 + query_length * 9173 + prefix_length
                )
                q = torch.randn(
                    1, query_heads, total_queries, d_head,
                    generator=generator, device=args.device, dtype=dtype,
                )
                k_pool = torch.randn(
                    total_pages, args.page_size, kv_heads, d_head,
                    generator=generator, device=args.device, dtype=dtype,
                )
                v_pool = torch.randn(
                    total_pages, args.page_size, kv_heads, d_head,
                    generator=generator, device=args.device, dtype=dtype,
                )
                page_ids = torch.randperm(
                    total_pages, generator=generator, device=args.device,
                    dtype=torch.int64,
                ).reshape(batch_size, pages_per_sequence).to(torch.int32)
                cu_seqlens = torch.tensor(offsets, device=args.device, dtype=torch.int32)
                context_lens = torch.full(
                    (batch_size,), context_length, device=args.device, dtype=torch.int32
                )
                reference = paged_sdpa_reference(
                    q, k_pool, v_pool, offsets, page_ids,
                    [context_length] * batch_size, args.page_size,
                )
                sdpa_samples = measure(lambda: paged_sdpa_reference(
                    q, k_pool, v_pool, offsets, page_ids,
                    [context_length] * batch_size, args.page_size,
                ))
                sdpa_median = statistics.median(sdpa_samples)
                pairs = batch_size * (
                    query_length * prefix_length
                    + query_length * (query_length + 1) // 2
                )
                shape_rows = []
                for block_m, block_n, warps, stages in args.configs:
                    base = {
                        "batch_size": batch_size,
                        "query_length": query_length,
                        "prefix_length": prefix_length,
                        "context_length": context_length,
                        "attention_pairs": pairs,
                        "block_m": block_m,
                        "block_n": block_n,
                        "num_warps": warps,
                        "num_stages": stages,
                        "sdpa_median_ms": sdpa_median,
                        "sdpa_raw_ms": sdpa_samples,
                    }
                    try:
                        operation = lambda bm=block_m, bn=block_n, nw=warps, ns=stages: (
                            packed_paged_prefill_attention(
                                q, k_pool, v_pool, cu_seqlens, page_ids, context_lens,
                                max_query_len=query_length, page_size=args.page_size,
                                block_m=bm, block_n=bn, num_warps=nw, num_stages=ns,
                            )
                        )
                        actual = operation()
                        torch.cuda.synchronize()
                        max_error = (actual.float() - reference.float()).abs().max().item()
                        atol = 8e-2 if dtype == torch.float16 else 1.5e-1
                        rtol = 2e-2 if dtype == torch.float16 else 4e-2
                        torch.testing.assert_close(
                            actual.float(), reference.float(), atol=atol, rtol=rtol
                        )
                        samples = measure(operation)
                        median_ms = statistics.median(samples)
                        row = base | {
                            "triton_median_ms": median_ms,
                            "triton_raw_ms": samples,
                            "speedup_vs_sdpa": sdpa_median / median_ms,
                            "speedup_vs_production": None,
                            "max_abs_error": max_error,
                            "status": "ok",
                            "error": None,
                        }
                    except Exception as exc:
                        torch.cuda.synchronize()
                        row = base | {
                            "triton_median_ms": None,
                            "triton_raw_ms": [],
                            "speedup_vs_sdpa": None,
                            "speedup_vs_production": None,
                            "max_abs_error": None,
                            "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    rows.append(row)
                    shape_rows.append(row)

                production = next(
                    (row for row in shape_rows
                     if tuple(row[name] for name in (
                         "block_m", "block_n", "num_warps", "num_stages"
                     )) == PRODUCTION_CONFIG and row["status"] == "ok"),
                    None,
                )
                if production is None:
                    raise RuntimeError("the production 64x32x4x2 control must succeed")
                for row in shape_rows:
                    if row["status"] == "ok":
                        row["speedup_vs_production"] = (
                            production["triton_median_ms"] / row["triton_median_ms"]
                        )
                winner = min(
                    (row for row in shape_rows if row["status"] == "ok"),
                    key=lambda row: row["triton_median_ms"],
                )
                print(
                    f"B={batch_size} Q={query_length} P={prefix_length} "
                    f"prod={production['triton_median_ms']:.4f}ms "
                    f"best={winner['block_m']}x{winner['block_n']}x"
                    f"{winner['num_warps']}x{winner['num_stages']} "
                    f"{winner['triton_median_ms']:.4f}ms "
                    f"({winner['speedup_vs_production']:.3f}x)"
                )
                del q, k_pool, v_pool, page_ids, cu_seqlens, context_lens, reference
                torch.cuda.empty_cache()

    decision_summary = summarize_configs(rows)
    print(
        "\nstatic/oracle summary: "
        f"best_static={'x'.join(map(str, decision_summary['best_static_config']))} "
        f"static_vs_prod={decision_summary['best_static_speedup_vs_production']:.3f}x "
        f"oracle_vs_prod={decision_summary['oracle_speedup_vs_production']:.3f}x "
        f"oracle_vs_static={decision_summary['oracle_speedup_vs_best_static']:.3f}x"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    json_path = args.output_dir / f"paged-prefill-tiles-{stamp}.json"
    csv_path = args.output_dir / f"paged-prefill-tiles-{stamp}.csv"
    summary_path = args.output_dir / f"paged-prefill-tiles-{stamp}-summary.csv"
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "system": system_metadata(),
        "configuration": {
            "batch_sizes": batch_sizes,
            "query_lengths": query_lengths,
            "prefix_lengths": prefix_lengths,
            "kernel_configs": [list(config) for config in args.configs],
            "production_config": list(PRODUCTION_CONFIG),
            "page_size": args.page_size,
            "dtype": args.dtype,
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "seed": args.seed,
        },
        "rows": rows,
        "decision_summary": decision_summary,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in CSV_FIELDS} for row in rows)
    summary_fields = (
        "block_m", "block_n", "num_warps", "num_stages", "supported_shapes",
        "total_shapes", "winner_count", "aggregate_speedup_vs_production",
        "geomean_speedup_vs_production", "best_static",
    )
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(
            {field: row.get(field) for field in summary_fields}
            for row in decision_summary["configurations"]
        )
    print(f"json: {json_path}")
    print(f"csv : {csv_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
