#!/usr/bin/env python3
"""Sweep GQA head sharing for long-context paged decode attention."""

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
for path in (HERE, BENCHMARKS, ROOT / "baseline", ROOT / "engine" / "kvcache"):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from benchmark_core import parse_int_list
from run_benchmarks import system_metadata


CSV_FIELDS = (
    "batch_size", "context_length", "heads_per_program", "num_warps",
    "production_median_ms", "candidate_median_ms", "speedup",
    "unique_kv_bytes", "estimated_kv_read_bytes", "estimated_read_amplification",
    "unique_kv_gbps", "estimated_kv_gbps", "max_abs_error", "status", "error",
)

SUMMARY_FIELDS = (
    "batch_size", "context_length", "production_median_ms",
    "best_candidate_median_ms", "best_speedup", "best_heads_per_program",
    "best_num_warps", "successful_configurations", "failed_configurations",
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", default="1,8,32,64")
    parser.add_argument("--context-lengths", default="128,2048,8192,16384")
    parser.add_argument("--heads-per-program", default="1,2,3,6")
    parser.add_argument("--num-warps", default="4,8")
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path,
                        default=EXPERIMENTS / "results" / "grouped-gqa-decode")
    return parser


def validate_args(args, batches, contexts, head_groups, warps):
    if any(value < 1 for value in batches + contexts):
        raise ValueError("batch sizes and contexts must be positive")
    if any(value not in (1, 2, 3, 6) for value in head_groups):
        raise ValueError("heads-per-program must contain 1, 2, 3, or 6")
    if any(value not in (1, 2, 4, 8) for value in warps):
        raise ValueError("num-warps must contain 1, 2, 4, or 8")
    if args.page_size < 1 or args.warmups < 0 or args.repetitions < 1:
        raise ValueError("page size/repetitions must be positive and warmups non-negative")
    if max(contexts) > 32768:
        raise ValueError("context exceeds model limit")


def _make_ragged_inputs(torch, lengths, *, page_size, dtype, device, seed):
    """Build isolated, non-contiguous page tables for a small correctness gate."""
    query_heads, kv_heads, d_head = 12, 2, 128
    page_counts = [(length + page_size - 1) // page_size for length in lengths]
    total_pages = sum(page_counts)
    generator = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(
        len(lengths), query_heads, d_head,
        generator=generator, device=device, dtype=dtype,
    )
    k_pool = torch.randn(
        total_pages, page_size, kv_heads, d_head,
        generator=generator, device=device, dtype=dtype,
    )
    v_pool = torch.randn(
        total_pages, page_size, kv_heads, d_head,
        generator=generator, device=device, dtype=dtype,
    )
    physical_pages = torch.randperm(
        total_pages, generator=generator, device=device, dtype=torch.int64,
    ).to(torch.int32)
    block_table = torch.zeros(
        len(lengths), max(page_counts), device=device, dtype=torch.int32,
    )
    cursor = 0
    for row, page_count in enumerate(page_counts):
        block_table[row, :page_count] = physical_pages[cursor:cursor + page_count]
        cursor += page_count
    seq_lens = torch.tensor(lengths, device=device, dtype=torch.int32)
    return q, k_pool, v_pool, block_table, seq_lens


def _correctness_preflight(
    torch,
    grouped_attention,
    production_attention,
    manual_reference,
    *,
    page_size,
    dtype,
    device,
    head_groups,
    warp_counts,
    seed,
):
    """Gate configurations on ragged lengths, page boundaries, and cache isolation."""
    lengths = [1, 15, 16, 17, 31, 33, 70]
    tensors = _make_ragged_inputs(
        torch, lengths, page_size=page_size, dtype=dtype, device=device, seed=seed,
    )
    q, k_pool, v_pool, block_table, seq_lens = tensors
    manual = manual_reference(q, k_pool, v_pool, block_table, seq_lens)
    production = production_attention(q, k_pool, v_pool, block_table, seq_lens)
    torch.testing.assert_close(
        production.float(), manual.float(), atol=5e-2, rtol=2e-2,
    )

    records = []
    available = set()
    for heads_per_program in head_groups:
        for num_warps in warp_counts:
            try:
                actual = grouped_attention(
                    q, k_pool, v_pool, block_table, seq_lens,
                    heads_per_program=heads_per_program,
                    num_warps=num_warps,
                )
                torch.cuda.synchronize()
                max_error = (actual.float() - manual.float()).abs().max().item()
                torch.testing.assert_close(
                    actual.float(), manual.float(), atol=5e-2, rtol=2e-2,
                )
                status, error = "ok", None
                available.add((heads_per_program, num_warps))
            except Exception as exc:
                torch.cuda.synchronize()
                max_error = None
                status, error = "failed", f"{type(exc).__name__}: {exc}"
            records.append({
                "lengths": lengths,
                "heads_per_program": heads_per_program,
                "num_warps": num_warps,
                "max_abs_error": max_error,
                "status": status,
                "error": error,
            })
            suffix = f" max_err={max_error:.6f}" if max_error is not None else f" {error}"
            print(
                f"  preflight heads={heads_per_program} warps={num_warps}: "
                f"{status.upper()}{suffix}"
            )
    del q, k_pool, v_pool, block_table, seq_lens, manual, production
    torch.cuda.empty_cache()
    if not available:
        raise RuntimeError("every grouped-GQA configuration failed correctness preflight")
    return records, available


def main():
    args = build_parser().parse_args()
    batches = parse_int_list(args.batch_sizes)
    contexts = parse_int_list(args.context_lengths)
    head_groups = parse_int_list(args.heads_per_program)
    warp_counts = parse_int_list(args.num_warps)
    validate_args(args, batches, contexts, head_groups, warp_counts)

    import torch
    from kernel_dispatch import paged_decode_attention_candidate
    from paged_decode_attention import decode_attention_reference, paged_decode_attention

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("a CUDA device is required")
    dtype = getattr(torch, args.dtype)
    query_heads, kv_heads, d_head = 12, 2, 128
    element_size = torch.empty((), dtype=dtype).element_size()

    print("ragged manual-reference correctness preflight...")
    preflight, available_configs = _correctness_preflight(
        torch,
        paged_decode_attention_candidate,
        paged_decode_attention,
        decode_attention_reference,
        page_size=args.page_size,
        dtype=dtype,
        device=args.device,
        head_groups=head_groups,
        warp_counts=warp_counts,
        seed=args.seed,
    )

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
    for batch in batches:
        for context in contexts:
            pages_per_sequence = (context + args.page_size - 1) // args.page_size
            total_pages = batch * pages_per_sequence
            generator = torch.Generator(device=args.device).manual_seed(
                args.seed + batch * 1009 + context * 9173
            )
            q = torch.randn(batch, query_heads, d_head, generator=generator,
                            device=args.device, dtype=dtype)
            k_pool = torch.randn(
                total_pages, args.page_size, kv_heads, d_head,
                generator=generator, device=args.device, dtype=dtype,
            )
            v_pool = torch.randn(
                total_pages, args.page_size, kv_heads, d_head,
                generator=generator, device=args.device, dtype=dtype,
            )
            block_table = torch.randperm(
                total_pages, generator=generator, device=args.device,
                dtype=torch.int64,
            ).reshape(batch, pages_per_sequence).to(torch.int32)
            seq_lens = torch.full((batch,), context, device=args.device, dtype=torch.int32)

            reference = paged_decode_attention(
                q, k_pool, v_pool, block_table, seq_lens
            )
            production_raw = measure(lambda: paged_decode_attention(
                q, k_pool, v_pool, block_table, seq_lens
            ))
            production_ms = statistics.median(production_raw)
            print(f"\nB={batch} C={context} production={production_ms:.4f} ms")
            for heads_per_program in head_groups:
                for num_warps in warp_counts:
                    group = query_heads // kv_heads
                    unique_kv_bytes = (
                        batch * kv_heads * context * d_head * 2 * element_size
                    )
                    read_amplification = group // heads_per_program
                    estimated_kv_read_bytes = unique_kv_bytes * read_amplification
                    base = {
                        "batch_size": batch,
                        "context_length": context,
                        "heads_per_program": heads_per_program,
                        "num_warps": num_warps,
                        "production_median_ms": production_ms,
                        "production_raw_ms": production_raw,
                        "unique_kv_bytes": unique_kv_bytes,
                        "estimated_kv_read_bytes": estimated_kv_read_bytes,
                        "estimated_read_amplification": read_amplification,
                    }
                    if (heads_per_program, num_warps) not in available_configs:
                        failure = next(
                            record for record in preflight
                            if record["heads_per_program"] == heads_per_program
                            and record["num_warps"] == num_warps
                        )
                        rows.append(base | {
                            "candidate_median_ms": None,
                            "candidate_raw_ms": [],
                            "speedup": None,
                            "unique_kv_gbps": None,
                            "estimated_kv_gbps": None,
                            "max_abs_error": None,
                            "status": "preflight_failed",
                            "error": failure["error"],
                        })
                        print(
                            f"  heads={heads_per_program} warps={num_warps}: "
                            "SKIPPED preflight failure"
                        )
                        continue
                    try:
                        operation = lambda hp=heads_per_program, nw=num_warps: (
                            paged_decode_attention_candidate(
                                q, k_pool, v_pool, block_table, seq_lens,
                                heads_per_program=hp, num_warps=nw,
                            )
                        )
                        actual = operation()
                        torch.cuda.synchronize()
                        max_error = (actual.float() - reference.float()).abs().max().item()
                        torch.testing.assert_close(
                            actual.float(), reference.float(), atol=5e-2, rtol=2e-2
                        )
                        raw = measure(operation)
                        median_ms = statistics.median(raw)
                        row = base | {
                            "candidate_median_ms": median_ms,
                            "candidate_raw_ms": raw,
                            "speedup": production_ms / median_ms,
                            "unique_kv_gbps": unique_kv_bytes / (median_ms * 1e-3) / 1e9,
                            "estimated_kv_gbps": (
                                estimated_kv_read_bytes / (median_ms * 1e-3) / 1e9
                            ),
                            "max_abs_error": max_error,
                            "status": "ok",
                            "error": None,
                        }
                        print(
                            f"  heads={heads_per_program} warps={num_warps}: "
                            f"{median_ms:.4f} ms {row['speedup']:.3f}x"
                        )
                    except Exception as exc:
                        torch.cuda.synchronize()
                        row = base | {
                            "candidate_median_ms": None,
                            "candidate_raw_ms": [],
                            "speedup": None,
                            "unique_kv_gbps": None,
                            "estimated_kv_gbps": None,
                            "max_abs_error": None,
                            "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                        print(
                            f"  heads={heads_per_program} warps={num_warps}: "
                            f"FAILED {row['error']}"
                        )
                    rows.append(row)

            del q, k_pool, v_pool, block_table, seq_lens, reference
            torch.cuda.empty_cache()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    json_path = args.output_dir / f"grouped-gqa-decode-{stamp}.json"
    csv_path = args.output_dir / f"grouped-gqa-decode-{stamp}.csv"
    summaries = []
    for batch in batches:
        for context in contexts:
            shape_rows = [
                row for row in rows
                if row["batch_size"] == batch and row["context_length"] == context
            ]
            successful = [row for row in shape_rows if row["status"] == "ok"]
            best = (
                min(successful, key=lambda row: row["candidate_median_ms"])
                if successful else None
            )
            summaries.append({
                "batch_size": batch,
                "context_length": context,
                "production_median_ms": shape_rows[0]["production_median_ms"],
                "best_candidate_median_ms": (
                    best["candidate_median_ms"] if best else None
                ),
                "best_speedup": best["speedup"] if best else None,
                "best_heads_per_program": best["heads_per_program"] if best else None,
                "best_num_warps": best["num_warps"] if best else None,
                "successful_configurations": len(successful),
                "failed_configurations": len(shape_rows) - len(successful),
            })

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "system": system_metadata(),
        "configuration": {
            "batch_sizes": batches,
            "context_lengths": contexts,
            "heads_per_program": head_groups,
            "num_warps": warp_counts,
            "page_size": args.page_size,
            "dtype": args.dtype,
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "seed": args.seed,
        },
        "correctness_preflight": preflight,
        "shape_summaries": summaries,
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in CSV_FIELDS} for row in rows)
    summary_path = args.output_dir / f"grouped-gqa-decode-{stamp}-summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summaries)
    print(f"\njson: {json_path}")
    print(f"csv : {csv_path}")
    print(f"summary: {summary_path}")
    if not any(row["status"] == "ok" for row in rows):
        raise RuntimeError("every grouped GQA configuration failed")


if __name__ == "__main__":
    main()
