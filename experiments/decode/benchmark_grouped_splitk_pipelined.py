#!/usr/bin/env python3
"""Ablation study: Grouped + Split-K + Pipelined decode attention.

Tests four configurations to isolate each optimization's effect:
  A: hpp=1, K=1, no pipeline  (baseline)
  B: hpp=6, K=1, no pipeline  (+grouping only)
  C: hpp=6, K=auto, no pipeline  (+grouping +split-k)
  D: hpp=6, K=auto, pipeline  (+grouping +split-k +pipeline)

Measures:
  - B vs A = grouping effect (fewer programs, fewer KV reads)
  - C vs B = split-k effect (more programs, same KV reads)
  - D vs C = pipelining effect (latency hiding)
  - D vs production = total improvement
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

# Add paths
SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
ROOT = EXPERIMENTS_DIR.parent
for path in (SCRIPT_DIR, ROOT / "custom_kernels", ROOT / "engine" / "kvcache"):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from paged_decode_grouped_splitk_pipelined import (
    grouped_splitk_attention,
    compute_split_k,
    get_num_sms,
)
from paged_decode_attention import paged_decode_attention


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ablation: Group + Split-K + Pipeline")
    parser.add_argument(
        "--batch-sizes",
        default="1,4,8,16,32,64,96,128,256",
        help="Comma-separated batch sizes",
    )
    parser.add_argument(
        "--context-lengths",
        default="512,2048,8192,16384",
        help="Comma-separated context lengths",
    )
    parser.add_argument(
        "--target-occupancy",
        type=int, default=4,
        help="Target programs per SM for split-k",
    )
    parser.add_argument(
        "--warmup", type=int, default=5, help="Warmup iterations"
    )
    parser.add_argument(
        "--repetitions", type=int, default=20, help="Timed iterations"
    )
    parser.add_argument(
        "--output-dir",
        default=str(EXPERIMENTS_DIR / "results" / "ablation-grouped-splitk-pipelined"),
        help="Output directory",
    )
    parser.add_argument(
        "--page-size", type=int, default=16, help="KV cache page size"
    )
    return parser


def create_test_inputs(
    batch_size: int,
    context_length: int,
    page_size: int,
    query_heads: int = 12,
    kv_heads: int = 2,
    head_dim: int = 128,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
):
    """Create test inputs for decode attention."""
    num_pages_per_seq = (context_length + page_size - 1) // page_size
    total_pages = batch_size * num_pages_per_seq

    q = torch.randn(batch_size, query_heads, head_dim, dtype=dtype, device=device)
    k_pool = torch.randn(total_pages, page_size, kv_heads, head_dim, dtype=dtype, device=device)
    v_pool = torch.randn(total_pages, page_size, kv_heads, head_dim, dtype=dtype, device=device)

    block_table = torch.arange(total_pages, device=device, dtype=torch.int32)
    block_table = block_table.view(batch_size, num_pages_per_seq)

    seq_lens = torch.full((batch_size,), context_length, device=device, dtype=torch.int32)

    return q, k_pool, v_pool, block_table, seq_lens


def benchmark_kernel(fn, warmup: int, reps: int) -> float:
    """Benchmark a kernel and return median time in ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    return times[len(times) // 2]


def main():
    parser = build_parser()
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    context_lengths = [int(x) for x in args.context_lengths.split(",")]

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    base_name = f"ablation-{timestamp}"

    csv_path = Path(args.output_dir) / f"{base_name}.csv"
    json_path = Path(args.output_dir) / f"{base_name}.json"

    num_sms = get_num_sms()
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"SMs: {num_sms}")
    print(f"Target occupancy: {args.target_occupancy} programs/SM")
    print()
    print("Configs:")
    print("  A: hpp=1, K=1, no pipeline  (baseline)")
    print("  B: hpp=6, K=1, no pipeline  (+grouping)")
    print("  C: hpp=6, K=auto, no pipeline  (+split-k)")
    print("  D: hpp=6, K=auto, pipeline  (+pipeline)")
    print()

    all_results = []

    for B in batch_sizes:
        for C in context_lengths:
            q, k_pool, v_pool, block_table, seq_lens = create_test_inputs(
                B, C, args.page_size
            )
            num_kv_blocks = (C + args.page_size - 1) // args.page_size

            # Compute K for this config
            # hpp=6 means head_tiles=1 (6 heads per program, GROUP=6)
            actual_k = compute_split_k(B, 2, 1, num_kv_blocks, args.target_occupancy)

            # Programs for each config
            progs_A = B * 2 * 6  # hpp=1, K=1: batch * kv_heads * GROUP
            progs_B = B * 2 * 1  # hpp=6, K=1: batch * kv_heads * 1
            progs_C = B * 2 * 1 * actual_k  # hpp=6, K=auto
            progs_D = progs_C  # same as C

            print(f"B={B:3d} C={C:5d} blocks={num_kv_blocks:3d} K={actual_k:2d}")
            print(f"  programs: A={progs_A:4d} B={progs_B:4d} C={progs_C:4d} D={progs_D:4d}")

            # Production baseline
            try:
                prod_time = benchmark_kernel(
                    lambda: paged_decode_attention(q, k_pool, v_pool, block_table, seq_lens),
                    args.warmup, args.repetitions
                )
            except Exception as e:
                print(f"  PROD ERROR: {e}")
                continue

            # Config A: hpp=1, K=1, no pipeline (our baseline)
            try:
                time_A = benchmark_kernel(
                    lambda: grouped_splitk_attention(
                        q, k_pool, v_pool, block_table, seq_lens,
                        heads_per_program=1, split_k=1, pipelined=False
                    ),
                    args.warmup, args.repetitions
                )
            except Exception as e:
                print(f"  A ERROR: {e}")
                continue

            # Config B: hpp=6, K=1, no pipeline (+grouping)
            try:
                time_B = benchmark_kernel(
                    lambda: grouped_splitk_attention(
                        q, k_pool, v_pool, block_table, seq_lens,
                        heads_per_program=6, split_k=1, pipelined=False
                    ),
                    args.warmup, args.repetitions
                )
            except Exception as e:
                print(f"  B ERROR: {e}")
                continue

            # Config C: hpp=6, K=auto, no pipeline (+split-k)
            try:
                time_C = benchmark_kernel(
                    lambda: grouped_splitk_attention(
                        q, k_pool, v_pool, block_table, seq_lens,
                        heads_per_program=6, split_k=None,
                        target_occupancy=args.target_occupancy, pipelined=False
                    ),
                    args.warmup, args.repetitions
                )
            except Exception as e:
                print(f"  C ERROR: {e}")
                continue

            # Config D: hpp=6, K=auto, pipeline (+pipeline)
            try:
                time_D = benchmark_kernel(
                    lambda: grouped_splitk_attention(
                        q, k_pool, v_pool, block_table, seq_lens,
                        heads_per_program=6, split_k=None,
                        target_occupancy=args.target_occupancy, pipelined=True
                    ),
                    args.warmup, args.repetitions
                )
            except Exception as e:
                print(f"  D ERROR: {e}")
                continue

            # Calculate speedups
            speedup_B_vs_A = time_A / time_B  # grouping effect
            speedup_C_vs_B = time_B / time_C  # split-k effect
            speedup_D_vs_C = time_C / time_D  # pipeline effect
            speedup_D_vs_prod = prod_time / time_D  # total vs production

            # Markers
            def marker(speedup):
                if speedup > 1.05:
                    return "**"
                elif speedup < 0.95:
                    return "--"
                return "  "

            print(f"  prod={prod_time:.3f}ms  A={time_A:.3f}ms  B={time_B:.3f}ms  C={time_C:.3f}ms  D={time_D:.3f}ms")
            print(f"  B/A={speedup_B_vs_A:.2f}x{marker(speedup_B_vs_A)} (grouping)  "
                  f"C/B={speedup_C_vs_B:.2f}x{marker(speedup_C_vs_B)} (split-k)  "
                  f"D/C={speedup_D_vs_C:.2f}x{marker(speedup_D_vs_C)} (pipeline)  "
                  f"D/prod={speedup_D_vs_prod:.2f}x{marker(speedup_D_vs_prod)} (total)")
            print()

            result = {
                "batch_size": B,
                "context_length": C,
                "num_kv_blocks": num_kv_blocks,
                "actual_k": actual_k,
                "programs_A": progs_A,
                "programs_B": progs_B,
                "programs_C": progs_C,
                "programs_D": progs_D,
                "prod_ms": prod_time,
                "time_A_ms": time_A,
                "time_B_ms": time_B,
                "time_C_ms": time_C,
                "time_D_ms": time_D,
                "speedup_B_vs_A": speedup_B_vs_A,
                "speedup_C_vs_B": speedup_C_vs_B,
                "speedup_D_vs_C": speedup_D_vs_C,
                "speedup_D_vs_prod": speedup_D_vs_prod,
            }
            all_results.append(result)

    # Write results
    if all_results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)

        with open(json_path, "w") as f:
            json.dump({"config": vars(args), "results": all_results}, f, indent=2)

        print(f"\nWrote {len(all_results)} results to {csv_path}")

        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)

        # Find where each optimization helps most
        best_grouping = max(all_results, key=lambda r: r["speedup_B_vs_A"])
        best_splitk = max(all_results, key=lambda r: r["speedup_C_vs_B"])
        best_pipeline = max(all_results, key=lambda r: r["speedup_D_vs_C"])
        best_total = max(all_results, key=lambda r: r["speedup_D_vs_prod"])

        print(f"\nBest GROUPING effect (B/A):")
        print(f"  B={best_grouping['batch_size']} C={best_grouping['context_length']}: "
              f"{best_grouping['speedup_B_vs_A']:.2f}x")

        print(f"\nBest SPLIT-K effect (C/B):")
        print(f"  B={best_splitk['batch_size']} C={best_splitk['context_length']}: "
              f"{best_splitk['speedup_C_vs_B']:.2f}x (K={best_splitk['actual_k']})")

        print(f"\nBest PIPELINE effect (D/C):")
        print(f"  B={best_pipeline['batch_size']} C={best_pipeline['context_length']}: "
              f"{best_pipeline['speedup_D_vs_C']:.2f}x")

        print(f"\nBest TOTAL vs production (D/prod):")
        print(f"  B={best_total['batch_size']} C={best_total['context_length']}: "
              f"{best_total['speedup_D_vs_prod']:.2f}x")


if __name__ == "__main__":
    main()
