#!/usr/bin/env python3
"""Sweep KV head grouping for decode attention.

Goal: Find regimes where grouping query heads (sharing KV reads) helps,
independent of Split-K. This informs the combined dispatch strategy.

Key insight: Grouping reduces memory reads by GROUP/heads_per_program,
but also reduces program count by the same factor.
"""

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
    "batch_size", "context_length", "heads_per_program", "num_warps",
    "num_programs", "kv_read_factor",
    "production_median_ms", "grouped_median_ms", "speedup",
    "production_gbps", "grouped_gbps",
    "max_abs_error", "status", "error",
)

SUMMARY_FIELDS = (
    "batch_size", "context_length",
    "production_median_ms", "production_gbps",
    "best_heads_per_program", "best_num_warps",
    "best_grouped_median_ms", "best_speedup", "best_gbps",
    "successful_configs", "failed_configs",
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    # Dense sweep for thorough analysis
    parser.add_argument("--batch-sizes", default="1,2,4,8,12,16,24,32,48,64,96,128,192,256")
    parser.add_argument("--context-lengths", default="256,512,1024,2048,4096,8192,12288,16384")
    parser.add_argument("--heads-per-program", default="1,2,3,6",
                        help="1=no grouping (baseline), 6=full grouping (max KV sharing)")
    parser.add_argument("--num-warps", default="4,8")
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path,
                        default=EXPERIMENTS / "results" / "grouped-attention")
    return parser


def _make_test_inputs(torch, batch_size, context_length, *, page_size, dtype, device, seed):
    """Build test inputs with randomized page tables."""
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

    perm = torch.randperm(total_pages, generator=generator, device=device, dtype=torch.int64)
    block_table = perm.reshape(batch_size, pages_per_seq).to(torch.int32)
    seq_lens = torch.full((batch_size,), context_length, device=device, dtype=torch.int32)

    return q, k_pool, v_pool, block_table, seq_lens


def _correctness_check(torch, grouped_attention, production_attention, tensors,
                       heads_per_program, num_warps):
    """Verify grouped matches production."""
    q, k_pool, v_pool, block_table, seq_lens = tensors

    reference = production_attention(q, k_pool, v_pool, block_table, seq_lens)
    actual = grouped_attention(
        q, k_pool, v_pool, block_table, seq_lens,
        heads_per_program=heads_per_program,
        num_warps=num_warps,
    )
    torch.cuda.synchronize()

    max_error = (actual.float() - reference.float()).abs().max().item()
    if max_error > 5e-2:
        raise AssertionError(f"max error {max_error:.6f} exceeds threshold")

    return max_error


def main():
    args = build_parser().parse_args()
    batches = parse_int_list(args.batch_sizes)
    contexts = parse_int_list(args.context_lengths)
    heads_per_program_values = parse_int_list(args.heads_per_program)
    warp_values = parse_int_list(args.num_warps)

    import torch
    from paged_decode_candidate import paged_decode_attention_candidate
    from paged_decode_attention import paged_decode_attention

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    dtype = getattr(torch, args.dtype)
    query_heads, kv_heads, d_head = 12, 2, 128
    group = query_heads // kv_heads  # 6
    element_size = torch.empty((), dtype=dtype).element_size()

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    num_sms = props.multi_processor_count

    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"SMs: {num_sms}")
    print(f"GROUP (query heads per KV head): {group}")
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

            # KV data size (what we'd read with no redundancy)
            kv_bytes = batch * kv_heads * context * d_head * 2 * element_size

            tensors = _make_test_inputs(
                torch, batch, context,
                page_size=args.page_size, dtype=dtype, device=args.device,
                seed=args.seed + batch * 1009 + context * 9173,
            )
            q, k_pool, v_pool, block_table, seq_lens = tensors

            # Production baseline (heads_per_program=1 implicitly)
            production_raw = measure(
                lambda: paged_decode_attention(q, k_pool, v_pool, block_table, seq_lens)
            )
            production_ms = statistics.median(production_raw)
            production_programs = batch * query_heads
            production_gbps = kv_bytes * group / (production_ms * 1e-3) / 1e9  # includes read amplification

            print(f"\nB={batch:3d} C={context:5d} blocks={num_kv_blocks:4d} "
                  f"prod={production_ms:.3f}ms ({production_gbps:.1f} GB/s) "
                  f"programs={production_programs}")

            for hpp in heads_per_program_values:
                for nw in warp_values:
                    # Calculate metrics
                    # With hpp heads per program, we launch fewer programs
                    # but each program handles hpp heads with shared KV
                    num_programs = batch * kv_heads * (group // hpp)
                    kv_read_factor = group // hpp  # how many times KV is read (1 = optimal)

                    base = {
                        "batch_size": batch,
                        "context_length": context,
                        "heads_per_program": hpp,
                        "num_warps": nw,
                        "num_programs": num_programs,
                        "kv_read_factor": kv_read_factor,
                        "production_median_ms": production_ms,
                        "production_gbps": production_gbps,
                    }

                    try:
                        # Correctness check
                        max_error = _correctness_check(
                            torch, paged_decode_attention_candidate,
                            paged_decode_attention, tensors, hpp, nw
                        )

                        # Benchmark
                        operation = lambda h=hpp, w=nw: paged_decode_attention_candidate(
                            q, k_pool, v_pool, block_table, seq_lens,
                            heads_per_program=h, num_warps=w,
                        )
                        raw = measure(operation)
                        median_ms = statistics.median(raw)
                        speedup = production_ms / median_ms

                        # Effective bandwidth (actual KV reads)
                        actual_kv_reads = kv_bytes * kv_read_factor
                        grouped_gbps = actual_kv_reads / (median_ms * 1e-3) / 1e9

                        row = base | {
                            "grouped_median_ms": median_ms,
                            "speedup": speedup,
                            "grouped_gbps": grouped_gbps,
                            "max_abs_error": max_error,
                            "status": "ok",
                            "error": None,
                        }

                        marker = "**" if speedup > 1.05 else ("--" if speedup < 0.95 else "")
                        print(f"  hpp={hpp} warps={nw}: {median_ms:.3f}ms "
                              f"{speedup:.2f}x progs={num_programs:4d} "
                              f"kv_factor={kv_read_factor} {marker}")

                    except Exception as exc:
                        torch.cuda.synchronize()
                        row = base | {
                            "grouped_median_ms": None,
                            "speedup": None,
                            "grouped_gbps": None,
                            "max_abs_error": None,
                            "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                        print(f"  hpp={hpp} warps={nw}: FAILED {row['error']}")

                    rows.append(row)

            del q, k_pool, v_pool, block_table, seq_lens
            torch.cuda.empty_cache()

    # Build summaries
    summaries = []
    for batch in batches:
        for context in contexts:
            shape_rows = [r for r in rows
                          if r["batch_size"] == batch and r["context_length"] == context]
            successful = [r for r in shape_rows if r["status"] == "ok"]

            if not successful:
                continue

            best_row = max(successful, key=lambda r: r["speedup"])
            prod_row = shape_rows[0]

            summaries.append({
                "batch_size": batch,
                "context_length": context,
                "production_median_ms": prod_row["production_median_ms"],
                "production_gbps": prod_row["production_gbps"],
                "best_heads_per_program": best_row["heads_per_program"],
                "best_num_warps": best_row["num_warps"],
                "best_grouped_median_ms": best_row["grouped_median_ms"],
                "best_speedup": best_row["speedup"],
                "best_gbps": best_row["grouped_gbps"],
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
        "model_config": {
            "query_heads": query_heads,
            "kv_heads": kv_heads,
            "group": group,
            "d_head": d_head,
        },
        "configuration": {
            "batch_sizes": batches,
            "context_lengths": contexts,
            "heads_per_program": heads_per_program_values,
            "num_warps": warp_values,
            "page_size": args.page_size,
            "dtype": args.dtype,
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "seed": args.seed,
        },
        "shape_summaries": summaries,
        "rows": rows,
    }

    json_path = args.output_dir / f"grouped-attention-{stamp}.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")

    csv_path = args.output_dir / f"grouped-attention-{stamp}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in CSV_FIELDS})

    summary_path = args.output_dir / f"grouped-attention-{stamp}-summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summaries)

    print(f"\njson: {json_path}")
    print(f"csv: {csv_path}")
    print(f"summary: {summary_path}")

    # Analysis table
    print("\n" + "=" * 80)
    print("ANALYSIS: Where does grouping help?")
    print("=" * 80)

    print(f"\n{'B':>4} {'C':>6} {'prod_ms':>8} {'best_hpp':>8} {'best_ms':>8} {'speedup':>8} {'programs':>8}")
    print("-" * 70)

    for s in summaries:
        hpp = s['best_heads_per_program']
        num_progs = s['batch_size'] * 2 * (6 // hpp)
        marker = "✓" if s['best_speedup'] > 1.05 else ("✗" if s['best_speedup'] < 0.95 else "")
        print(f"{s['batch_size']:>4} {s['context_length']:>6} {s['production_median_ms']:>8.3f} "
              f"{hpp:>8} {s['best_grouped_median_ms']:>8.3f} "
              f"{s['best_speedup']:>7.2f}x {num_progs:>8} {marker}")

    # Summary statistics
    wins = [s for s in summaries if s['best_speedup'] > 1.05]
    losses = [s for s in summaries if s['best_speedup'] < 0.95]
    neutral = [s for s in summaries if 0.95 <= s['best_speedup'] <= 1.05]

    print(f"\nWins (>1.05x):   {len(wins)}")
    print(f"Neutral:         {len(neutral)}")
    print(f"Losses (<0.95x): {len(losses)}")

    if wins:
        print(f"\nWinning regimes (grouping helps):")
        for s in wins:
            print(f"  B={s['batch_size']:3d} C={s['context_length']:5d} → "
                  f"{s['best_speedup']:.2f}x with hpp={s['best_heads_per_program']}")


if __name__ == "__main__":
    main()
