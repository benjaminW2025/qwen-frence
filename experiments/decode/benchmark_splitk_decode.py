#!/usr/bin/env python3
"""Sweep Split-K decode attention for SM occupancy optimization."""

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
for path in (HERE, BENCHMARKS, ROOT / "custom_kernels", ROOT / "engine" / "kvcache"):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from benchmark_core import parse_int_list
from run_benchmarks import system_metadata


CSV_FIELDS = (
    "batch_size", "context_length", "k_splits", "num_kv_blocks",
    "production_programs", "splitk_programs",
    "production_median_ms", "splitk_median_ms", "speedup",
    "kv_bytes", "production_gbps", "splitk_gbps",
    "max_abs_error", "status", "error",
)

SUMMARY_FIELDS = (
    "batch_size", "context_length", "num_kv_blocks",
    "production_median_ms", "auto_k", "auto_k_median_ms", "auto_k_speedup",
    "best_k", "best_k_median_ms", "best_speedup",
    "successful_configs", "failed_configs",
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", default="1,4,8,16,32,64,128,192,256")
    parser.add_argument("--context-lengths", default="512,2048,4096,8192,16384")
    parser.add_argument("--k-splits", default="1,2,4,8,16,auto",
                        help="Split factors to test. 'auto' uses compute_split_k.")
    parser.add_argument(
        "--auto-k-neighbor-factors",
        default="",
        help=(
            "comma-separated positive factors that add per-shape K candidates "
            "around auto-K (for example: 0.5,0.75,1.25,1.5)"
        ),
    )
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path,
                        default=EXPERIMENTS / "results" / "splitk-decode")
    return parser


def parse_k_splits(value: str) -> list:
    """Parse k-splits argument, handling 'auto' specially."""
    parts = [p.strip() for p in value.split(",")]
    result = []
    for p in parts:
        if p == "auto":
            result.append("auto")
        else:
            result.append(int(p))
    return result


def parse_auto_k_neighbor_factors(value: str) -> list[float]:
    """Parse optional positive multipliers used to bracket a shape's auto-K."""
    if not value.strip():
        return []
    factors = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not factors or any(factor <= 0 for factor in factors):
        raise ValueError("auto-K neighbor factors must be positive")
    return factors


def _rounded_positive(value: float) -> int:
    return max(1, int(value + 0.5))


def materialize_k_candidates(k_values, auto_k, num_kv_blocks, neighbor_factors):
    """Return labeled, executable K values for one benchmark shape.

    The explicitly requested values and the auto-policy row are retained even when
    they use the same K. Neighbor rows only add new K values, avoiding redundant
    measurements while still bracketing the architecture-specific auto choice.
    """
    candidates = []
    present_actual = set()
    for requested in k_values:
        actual = auto_k if requested == "auto" else requested
        label = f"auto({auto_k})" if requested == "auto" else str(requested)
        candidates.append((label, actual, requested == "auto"))
        present_actual.add(actual)

    for factor in neighbor_factors:
        actual = min(_rounded_positive(auto_k * factor), num_kv_blocks)
        if actual in present_actual:
            continue
        candidates.append((f"auto*{factor:g}({actual})", actual, False))
        present_actual.add(actual)

    return candidates


def _attention_operation(production_attention, splitk_attention, tensors, k_actual):
    """Select direct production attention for K=1 without wrapper overhead."""
    if k_actual == 1:
        return lambda: production_attention(*tensors)
    return lambda: splitk_attention(*tensors, k_splits=k_actual)


def _make_test_inputs(torch, batch_size, context_length, *, page_size, dtype, device, seed):
    """Build test inputs with isolated page tables."""
    query_heads, kv_heads, d_head = 12, 2, 128
    pages_per_seq = (context_length + page_size - 1) // page_size
    total_pages = batch_size * pages_per_seq

    generator = torch.Generator(device=device).manual_seed(seed)

    q = torch.randn(batch_size, query_heads, d_head,
                    generator=generator, device=device, dtype=dtype)
    k_pool = torch.randn(total_pages, page_size, kv_heads, d_head,
                         generator=generator, device=device, dtype=dtype)
    v_pool = torch.randn(total_pages, page_size, kv_heads, d_head,
                         generator=generator, device=device, dtype=dtype)

    # Randomize page table to simulate real fragmentation
    perm = torch.randperm(total_pages, generator=generator, device=device, dtype=torch.int64)
    block_table = perm.reshape(batch_size, pages_per_seq).to(torch.int32)
    seq_lens = torch.full((batch_size,), context_length, device=device, dtype=torch.int32)

    return q, k_pool, v_pool, block_table, seq_lens


def _correctness_preflight(torch, splitk_attention, production_attention, *,
                           page_size, dtype, device, k_actuals, seed):
    """Verify Split-K correctness on edge cases before benchmarking."""
    # Test various edge cases
    test_cases = [
        (2, 17),    # small, non-aligned
        (4, 128),   # aligned
        (8, 1000),  # medium
        (4, 4096),  # longer
    ]

    records = []
    available = set()

    for batch, ctx in test_cases:
        q, k_pool, v_pool, block_table, seq_lens = _make_test_inputs(
            torch, batch, ctx, page_size=page_size, dtype=dtype, device=device, seed=seed
        )
        reference = production_attention(q, k_pool, v_pool, block_table, seq_lens)
        num_kv_blocks = (ctx + page_size - 1) // page_size
        candidates = [
            (str(k_actual), k_actual, False)
            for k_actual in sorted(k_actuals)
            if k_actual <= num_kv_blocks
        ]

        for label, k_actual, _ in candidates:
            try:
                operation = _attention_operation(
                    production_attention, splitk_attention,
                    (q, k_pool, v_pool, block_table, seq_lens), k_actual,
                )
                actual = operation()
                torch.cuda.synchronize()

                max_error = (actual.float() - reference.float()).abs().max().item()
                # Allow slightly looser tolerance due to reduction
                if max_error > 1e-2:
                    raise AssertionError(f"max error {max_error:.6f} exceeds threshold")

                status, error = "ok", None
                available.add(k_actual)
            except Exception as exc:
                torch.cuda.synchronize()
                max_error = None
                status, error = "failed", f"{type(exc).__name__}: {exc}"

            records.append({
                "batch_size": batch,
                "context_length": ctx,
                "k_splits": label,
                "k_actual": k_actual,
                "max_abs_error": max_error,
                "status": status,
                "error": error,
            })

            suffix = f" max_err={max_error:.6f}" if max_error is not None else f" {error}"
            print(f"  preflight B={batch} C={ctx} K={label}: {status.upper()}{suffix}")

        del q, k_pool, v_pool, block_table, seq_lens, reference
        torch.cuda.empty_cache()

    if not available:
        raise RuntimeError("all Split-K configurations failed correctness preflight")

    return records, available


def main():
    args = build_parser().parse_args()
    batches = parse_int_list(args.batch_sizes)
    contexts = parse_int_list(args.context_lengths)
    k_values = parse_k_splits(args.k_splits)
    neighbor_factors = parse_auto_k_neighbor_factors(args.auto_k_neighbor_factors)

    import torch
    from paged_decode_splitk import paged_decode_attention_splitk, compute_split_k, get_num_sms
    from paged_decode_attention import paged_decode_attention

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("CUDA device required")

    dtype = getattr(torch, args.dtype)
    query_heads, kv_heads, d_head = 12, 2, 128
    element_size = torch.empty((), dtype=dtype).element_size()

    num_sms = get_num_sms()
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"SMs: {num_sms}")
    print()

    print("Correctness preflight...")
    preflight_k_actuals = {
        k_actual
        for batch in batches
        for context in contexts
        for _, k_actual, _ in materialize_k_candidates(
            k_values,
            compute_split_k(
                batch, query_heads,
                (context + args.page_size - 1) // args.page_size,
            ),
            (context + args.page_size - 1) // args.page_size,
            neighbor_factors,
        )
    }
    preflight, available_k = _correctness_preflight(
        torch, paged_decode_attention_splitk, paged_decode_attention,
        page_size=args.page_size, dtype=dtype, device=args.device,
        k_actuals=preflight_k_actuals, seed=args.seed,
    )
    print()

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
            num_kv_blocks = (context + args.page_size - 1) // args.page_size
            production_programs = batch * query_heads
            auto_k = compute_split_k(batch, query_heads, num_kv_blocks)

            q, k_pool, v_pool, block_table, seq_lens = _make_test_inputs(
                torch, batch, context,
                page_size=args.page_size, dtype=dtype, device=args.device,
                seed=args.seed + batch * 1009 + context * 9173,
            )

            # Production baseline
            reference = paged_decode_attention(q, k_pool, v_pool, block_table, seq_lens)
            production_raw = measure(
                lambda: paged_decode_attention(q, k_pool, v_pool, block_table, seq_lens)
            )
            production_ms = statistics.median(production_raw)

            kv_bytes = batch * kv_heads * context * d_head * 2 * element_size
            production_gbps = kv_bytes / (production_ms * 1e-3) / 1e9

            print(f"\nB={batch} C={context} blocks={num_kv_blocks} "
                  f"auto_k={auto_k} prod={production_ms:.3f}ms ({production_gbps:.1f} GB/s)")

            candidates = materialize_k_candidates(
                k_values, auto_k, num_kv_blocks, neighbor_factors,
            )
            for label, k_actual, _ in candidates:
                if k_actual not in available_k:
                    print(f"  K={label}: SKIPPED (preflight failed)")
                    continue

                splitk_programs = production_programs * k_actual

                base = {
                    "batch_size": batch,
                    "context_length": context,
                    "k_splits": label,
                    "k_actual": k_actual,
                    "num_kv_blocks": num_kv_blocks,
                    "production_programs": production_programs,
                    "splitk_programs": splitk_programs,
                    "production_median_ms": production_ms,
                    "kv_bytes": kv_bytes,
                    "production_gbps": production_gbps,
                }

                try:
                    if k_actual == 1:
                        raw = production_raw
                        median_ms = production_ms
                        max_error = 0.0
                    else:
                        operation = _attention_operation(
                            paged_decode_attention, paged_decode_attention_splitk,
                            (q, k_pool, v_pool, block_table, seq_lens), k_actual,
                        )
                        actual = operation()
                        torch.cuda.synchronize()
                        max_error = (actual.float() - reference.float()).abs().max().item()
                        raw = measure(operation)
                        median_ms = statistics.median(raw)

                    speedup = production_ms / median_ms
                    splitk_gbps = kv_bytes / (median_ms * 1e-3) / 1e9

                    row = base | {
                        "splitk_median_ms": median_ms,
                        "splitk_raw_ms": raw,
                        "speedup": speedup,
                        "splitk_gbps": splitk_gbps,
                        "max_abs_error": max_error,
                        "status": "ok",
                        "error": None,
                    }

                    marker = "**" if speedup > 1.1 else ""
                    print(f"  K={label}: {median_ms:.3f}ms {speedup:.2f}x "
                          f"({splitk_gbps:.1f} GB/s) {marker}")

                except Exception as exc:
                    torch.cuda.synchronize()
                    row = base | {
                        "splitk_median_ms": None,
                        "splitk_raw_ms": [],
                        "speedup": None,
                        "splitk_gbps": None,
                        "max_abs_error": None,
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    print(f"  K={label}: FAILED {row['error']}")

                rows.append(row)

            del q, k_pool, v_pool, block_table, seq_lens, reference
            torch.cuda.empty_cache()

    # Build summaries
    summaries = []
    for batch in batches:
        for context in contexts:
            shape_rows = [
                r for r in rows
                if r["batch_size"] == batch and r["context_length"] == context
            ]
            successful = [r for r in shape_rows if r["status"] == "ok"]

            auto_row = next(
                (r for r in successful if str(r["k_splits"]).startswith("auto")), None
            )
            best_row = min(successful, key=lambda r: r["splitk_median_ms"]) if successful else None

            num_kv_blocks = (context + args.page_size - 1) // args.page_size

            summaries.append({
                "batch_size": batch,
                "context_length": context,
                "num_kv_blocks": num_kv_blocks,
                "production_median_ms": shape_rows[0]["production_median_ms"] if shape_rows else None,
                "auto_k": auto_row["k_actual"] if auto_row else None,
                "auto_k_median_ms": auto_row["splitk_median_ms"] if auto_row else None,
                "auto_k_speedup": auto_row["speedup"] if auto_row else None,
                "best_k": best_row["k_actual"] if best_row else None,
                "best_k_median_ms": best_row["splitk_median_ms"] if best_row else None,
                "best_speedup": best_row["speedup"] if best_row else None,
                "successful_configs": len(successful),
                "failed_configs": len(shape_rows) - len(successful),
            })

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "system": system_metadata(),
        "device": {
            "name": torch.cuda.get_device_name(),
            "num_sms": num_sms,
        },
        "configuration": {
            "batch_sizes": batches,
            "context_lengths": contexts,
            "k_splits": k_values,
            "auto_k_neighbor_factors": neighbor_factors,
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

    json_path = args.output_dir / f"splitk-decode-{stamp}.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")

    csv_path = args.output_dir / f"splitk-decode-{stamp}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in CSV_FIELDS})

    summary_path = args.output_dir / f"splitk-decode-{stamp}-summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summaries)

    print(f"\njson: {json_path}")
    print(f"csv: {csv_path}")
    print(f"summary: {summary_path}")

    # Print summary table
    print("\n=== Summary ===")
    print(f"{'B':>4} {'C':>6} {'prod_ms':>8} {'auto_K':>6} {'auto_ms':>8} {'speedup':>7} {'best_K':>6} {'best_sp':>7}")
    print("-" * 70)
    for s in summaries:
        auto_k = s['auto_k'] if s['auto_k'] else '-'
        auto_ms = f"{s['auto_k_median_ms']:.3f}" if s['auto_k_median_ms'] else '-'
        auto_sp = f"{s['auto_k_speedup']:.2f}x" if s['auto_k_speedup'] else '-'
        best_k = s['best_k'] if s['best_k'] else '-'
        best_sp = f"{s['best_speedup']:.2f}x" if s['best_speedup'] else '-'
        print(f"{s['batch_size']:>4} {s['context_length']:>6} {s['production_median_ms']:>8.3f} "
              f"{auto_k:>6} {auto_ms:>8} {auto_sp:>7} {best_k:>6} {best_sp:>7}")


if __name__ == "__main__":
    main()
