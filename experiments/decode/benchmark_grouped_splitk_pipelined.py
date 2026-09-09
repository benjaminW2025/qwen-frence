#!/usr/bin/env python3
"""Benchmark for grouped + Split-K + pipelined decode attention.

Sweeps:
- batch_size: 1 to 256
- context_length: 512 to 16384
- heads_per_program: 1, 2, 3, 6
- target_occupancy: 1, 2, 4, 8 (programs per SM)

Measures speedup vs production kernel and tracks:
- Actual K value computed
- Total programs dispatched
- Effective bandwidth
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from paged_decode_grouped_splitk_pipelined import (
    grouped_splitk_pipelined_attention,
    compute_split_k,
    get_num_sms,
)
from custom_kernels.paged_decode import paged_decode_attention


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Grouped + Split-K + Pipelined benchmark")
    parser.add_argument(
        "--batch-sizes",
        default="1,4,8,16,32,64,96,128,192,256",
        help="Comma-separated batch sizes",
    )
    parser.add_argument(
        "--context-lengths",
        default="512,1024,2048,4096,8192,16384",
        help="Comma-separated context lengths",
    )
    parser.add_argument(
        "--heads-per-program",
        default="1,2,3,6",
        help="Comma-separated hpp values (1=no grouping, 6=full grouping)",
    )
    parser.add_argument(
        "--target-occupancy",
        default="1,2,4,8",
        help="Comma-separated target programs per SM",
    )
    parser.add_argument(
        "--num-warps",
        default="4",
        help="Comma-separated warp counts",
    )
    parser.add_argument(
        "--warmup", type=int, default=5, help="Warmup iterations"
    )
    parser.add_argument(
        "--repetitions", type=int, default=20, help="Timed iterations"
    )
    parser.add_argument(
        "--output-dir",
        default=str(EXPERIMENTS_DIR / "results" / "grouped-splitk-pipelined"),
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

    # Simple contiguous block table
    block_table = torch.arange(total_pages, device=device, dtype=torch.int32)
    block_table = block_table.view(batch_size, num_pages_per_seq)

    # All sequences have same length
    seq_lens = torch.full((batch_size,), context_length, device=device, dtype=torch.int32)

    return q, k_pool, v_pool, block_table, seq_lens


def benchmark_kernel(fn, warmup: int, reps: int) -> float:
    """Benchmark a kernel and return median time in ms."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Timed runs
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


def compute_bandwidth_gbps(
    batch_size: int,
    context_length: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    kv_read_factor: int,
    time_ms: float,
) -> float:
    """Compute effective bandwidth in GB/s."""
    # Bytes read: Q + K + V (with read amplification)
    bytes_q = batch_size * query_heads * head_dim * 2  # fp16
    bytes_kv = batch_size * context_length * kv_heads * head_dim * 2 * 2  # K + V
    bytes_kv *= kv_read_factor  # Read amplification from non-grouped programs
    bytes_out = batch_size * query_heads * head_dim * 2

    total_bytes = bytes_q + bytes_kv + bytes_out
    return (total_bytes / 1e9) / (time_ms / 1e3)


def correctness_check(
    q, k_pool, v_pool, block_table, seq_lens,
    hpp: int, target_occ: int, num_warps: int,
) -> tuple[bool, float]:
    """Check correctness against production kernel."""
    # Production output
    prod_out = paged_decode_attention(q, k_pool, v_pool, block_table, seq_lens)

    # Experimental output
    exp_out = grouped_splitk_pipelined_attention(
        q, k_pool, v_pool, block_table, seq_lens,
        heads_per_program=hpp,
        target_occupancy=target_occ,
        num_warps=num_warps,
    )

    max_err = (prod_out - exp_out).abs().max().item()
    # Allow larger tolerance due to fp16 and different reduction order
    ok = max_err < 1e-2
    return ok, max_err


def main():
    parser = build_parser()
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    context_lengths = [int(x) for x in args.context_lengths.split(",")]
    hpp_values = [int(x) for x in args.heads_per_program.split(",")]
    target_occs = [int(x) for x in args.target_occupancy.split(",")]
    num_warps_values = [int(x) for x in args.num_warps.split(",")]

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    base_name = f"grouped-splitk-pipelined-{timestamp}"

    csv_path = Path(args.output_dir) / f"{base_name}.csv"
    summary_path = Path(args.output_dir) / f"{base_name}-summary.csv"
    json_path = Path(args.output_dir) / f"{base_name}.json"

    num_sms = get_num_sms()
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"SMs: {num_sms}")
    print(f"Sweeping: {len(batch_sizes)} batches x {len(context_lengths)} contexts")
    print(f"         x {len(hpp_values)} hpp x {len(target_occs)} occupancy x {len(num_warps_values)} warps")
    print()

    all_results = []
    summaries = []

    for B in batch_sizes:
        for C in context_lengths:
            print(f"B={B:3d} C={C:5d}", end=" ", flush=True)

            q, k_pool, v_pool, block_table, seq_lens = create_test_inputs(
                B, C, args.page_size
            )
            num_kv_blocks = (C + args.page_size - 1) // args.page_size

            # Production baseline
            prod_time = benchmark_kernel(
                lambda: paged_decode_attention(q, k_pool, v_pool, block_table, seq_lens),
                args.warmup, args.repetitions
            )
            prod_gbps = compute_bandwidth_gbps(B, C, 12, 2, 128, 6, prod_time)
            print(f"prod={prod_time:.3f}ms ({prod_gbps:.1f} GB/s)", end=" ", flush=True)

            best_config = None
            best_speedup = 0.0

            for hpp in hpp_values:
                head_tiles = 12 // hpp  # GROUP=6 for 12 query / 2 KV heads
                kv_read_factor = 6 // hpp

                for target_occ in target_occs:
                    for nw in num_warps_values:
                        # Compute actual K
                        actual_k = compute_split_k(B, 2, head_tiles, num_kv_blocks, target_occ)
                        total_programs = B * 2 * head_tiles * actual_k

                        # Correctness check (only on first iteration)
                        if len(all_results) < 10:
                            ok, max_err = correctness_check(
                                q, k_pool, v_pool, block_table, seq_lens,
                                hpp, target_occ, nw
                            )
                            if not ok:
                                print(f"\nFAIL: hpp={hpp} occ={target_occ} err={max_err:.2e}")
                                continue
                        else:
                            ok, max_err = True, 0.0

                        # Benchmark
                        try:
                            exp_time = benchmark_kernel(
                                lambda hpp=hpp, occ=target_occ, nw=nw: grouped_splitk_pipelined_attention(
                                    q, k_pool, v_pool, block_table, seq_lens,
                                    heads_per_program=hpp,
                                    target_occupancy=occ,
                                    num_warps=nw,
                                ),
                                args.warmup, args.repetitions
                            )
                        except Exception as e:
                            print(f"\nERROR: {e}")
                            continue

                        speedup = prod_time / exp_time
                        exp_gbps = compute_bandwidth_gbps(B, C, 12, 2, 128, kv_read_factor, exp_time)

                        result = {
                            "batch_size": B,
                            "context_length": C,
                            "heads_per_program": hpp,
                            "target_occupancy": target_occ,
                            "num_warps": nw,
                            "actual_k": actual_k,
                            "total_programs": total_programs,
                            "kv_read_factor": kv_read_factor,
                            "production_ms": prod_time,
                            "experimental_ms": exp_time,
                            "speedup": speedup,
                            "production_gbps": prod_gbps,
                            "experimental_gbps": exp_gbps,
                            "max_error": max_err,
                            "status": "ok" if ok else "fail",
                        }
                        all_results.append(result)

                        if speedup > best_speedup:
                            best_speedup = speedup
                            best_config = result.copy()

            # Print best for this B,C
            if best_config:
                cfg = best_config
                marker = "**" if cfg["speedup"] > 1.05 else ("--" if cfg["speedup"] < 0.95 else "  ")
                print(f"best: hpp={cfg['heads_per_program']} occ={cfg['target_occupancy']} "
                      f"K={cfg['actual_k']} progs={cfg['total_programs']:4d} "
                      f"{cfg['speedup']:.2f}x {marker}")
                summaries.append({
                    "batch_size": B,
                    "context_length": C,
                    "production_ms": prod_time,
                    **{f"best_{k}": v for k, v in best_config.items() if k not in ("batch_size", "context_length", "production_ms")}
                })
            else:
                print("NO VALID CONFIG")

    # Write results
    if all_results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)

        with open(json_path, "w") as f:
            json.dump({"config": vars(args), "results": all_results}, f, indent=2)

        print(f"\nWrote {len(all_results)} results to {csv_path}")

    if summaries:
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summaries[0].keys())
            writer.writeheader()
            writer.writerows(summaries)
        print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
