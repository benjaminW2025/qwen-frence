#!/usr/bin/env python3
"""Compare separate Q/K RoPE plus cache scatters with one fused placement kernel."""

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
for path in (BENCHMARKS, ROOT / "baseline"):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from benchmark_core import parse_int_list
from run_benchmarks import system_metadata


CSV_FIELDS = (
    "tokens", "num_warps", "baseline_median_ms", "fused_median_ms", "speedup",
    "q_max_abs_error", "k_max_abs_error", "v_max_abs_error", "status", "error",
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-counts", default="64,520,2048,4128,8200,16384")
    parser.add_argument("--num-warps", default="1,2,4,8")
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--rope-theta", type=float, default=1_000_000.0)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir", type=Path,
        default=EXPERIMENTS / "results" / "rope-kv-fusion",
    )
    return parser


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
    token_counts = parse_int_list(args.token_counts)
    warp_counts = parse_int_list(args.num_warps)
    if any(value < 1 for value in token_counts):
        raise ValueError("token counts must be positive")
    if any(value not in (1, 2, 4, 8) for value in warp_counts):
        raise ValueError("num-warps must contain 1, 2, 4, or 8")
    if args.page_size < 1 or args.warmups < 0 or args.repetitions < 1:
        raise ValueError("invalid page size, warmup count, or repetition count")

    import torch
    from kernel_dispatch import rope, rope_kv_write

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("a CUDA device is required")
    dtype = getattr(torch, args.dtype)
    query_heads, kv_heads, head_dim = 12, 2, 128
    rows = []
    for tokens in token_counts:
        generator = torch.Generator(device=args.device).manual_seed(
            args.seed + tokens * 1009
        )
        q = torch.randn(
            tokens, query_heads, head_dim, generator=generator,
            device=args.device, dtype=dtype,
        ).transpose(0, 1).unsqueeze(0)
        k = torch.randn(
            tokens, kv_heads, head_dim, generator=generator,
            device=args.device, dtype=dtype,
        ).transpose(0, 1).unsqueeze(0)
        v = torch.randn(
            tokens, kv_heads, head_dim, generator=generator,
            device=args.device, dtype=dtype,
        ).transpose(0, 1).unsqueeze(0)
        positions = torch.arange(tokens, device=args.device, dtype=torch.float32)[None]
        total_slots = tokens + args.page_size
        pages = (total_slots + args.page_size - 1) // args.page_size
        slot_mapping = torch.randperm(
            pages * args.page_size, generator=generator,
            device=args.device, dtype=torch.int64,
        )[:tokens]
        baseline_k_pool = torch.zeros(
            pages, args.page_size, kv_heads, head_dim,
            device=args.device, dtype=dtype,
        )
        baseline_v_pool = torch.zeros_like(baseline_k_pool)
        fused_k_pool = torch.zeros_like(baseline_k_pool)
        fused_v_pool = torch.zeros_like(baseline_k_pool)

        def baseline():
            q_rotated = rope(q, positions=positions, theta=args.rope_theta)
            k_rotated = rope(k, positions=positions, theta=args.rope_theta)
            baseline_k_pool.view(-1, kv_heads, head_dim).index_copy_(
                0, slot_mapping, k_rotated[0].transpose(0, 1).contiguous(),
            )
            baseline_v_pool.view(-1, kv_heads, head_dim).index_copy_(
                0, slot_mapping, v[0].transpose(0, 1).contiguous(),
            )
            return q_rotated

        reference_q = baseline()
        torch.cuda.synchronize()
        baseline_raw = _measure(
            torch, baseline, args.warmups, args.repetitions,
        )
        baseline_ms = statistics.median(baseline_raw)
        print(f"\nT={tokens} baseline={baseline_ms:.4f} ms")
        for num_warps in warp_counts:
            try:
                operation = lambda nw=num_warps: rope_kv_write(
                    q, k, v, positions, slot_mapping, fused_k_pool, fused_v_pool,
                    base=args.rope_theta, num_warps=nw,
                )
                actual_q = operation()
                torch.cuda.synchronize()
                q_error = (actual_q.float() - reference_q.float()).abs().max().item()
                selected_k = fused_k_pool.view(-1, kv_heads, head_dim).index_select(
                    0, slot_mapping
                )
                selected_v = fused_v_pool.view(-1, kv_heads, head_dim).index_select(
                    0, slot_mapping
                )
                reference_k = baseline_k_pool.view(-1, kv_heads, head_dim).index_select(
                    0, slot_mapping
                )
                reference_v = baseline_v_pool.view(-1, kv_heads, head_dim).index_select(
                    0, slot_mapping
                )
                k_error = (selected_k.float() - reference_k.float()).abs().max().item()
                v_error = (selected_v.float() - reference_v.float()).abs().max().item()
                torch.testing.assert_close(
                    actual_q.float(), reference_q.float(), atol=2e-2, rtol=2e-2,
                )
                torch.testing.assert_close(
                    selected_k.float(), reference_k.float(), atol=2e-2, rtol=2e-2,
                )
                torch.testing.assert_close(selected_v, reference_v, atol=0, rtol=0)
                raw = _measure(
                    torch, operation, args.warmups, args.repetitions,
                )
                median_ms = statistics.median(raw)
                row = {
                    "tokens": tokens, "num_warps": num_warps,
                    "baseline_median_ms": baseline_ms,
                    "baseline_raw_ms": baseline_raw,
                    "fused_median_ms": median_ms, "fused_raw_ms": raw,
                    "speedup": baseline_ms / median_ms,
                    "q_max_abs_error": q_error, "k_max_abs_error": k_error,
                    "v_max_abs_error": v_error, "status": "ok", "error": None,
                }
                print(
                    f"  warps={num_warps}: {median_ms:.4f} ms "
                    f"{row['speedup']:.2f}x"
                )
            except Exception as exc:
                torch.cuda.synchronize()
                row = {
                    "tokens": tokens, "num_warps": num_warps,
                    "baseline_median_ms": baseline_ms,
                    "baseline_raw_ms": baseline_raw,
                    "fused_median_ms": None, "fused_raw_ms": [], "speedup": None,
                    "q_max_abs_error": None, "k_max_abs_error": None,
                    "v_max_abs_error": None, "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                print(f"  warps={num_warps}: FAILED {row['error']}")
            rows.append(row)

        del q, k, v, positions, slot_mapping
        del baseline_k_pool, baseline_v_pool, fused_k_pool, fused_v_pool, reference_q
        torch.cuda.empty_cache()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    stem = args.output_dir / f"rope-kv-fusion-{stamp}"
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
    print(f"\njson: {stem.with_suffix('.json')}")
    print(f"csv: {stem.with_suffix('.csv')}")
    if not any(row["status"] == "ok" for row in rows):
        raise SystemExit("no fused RoPE/KV configuration completed successfully")


if __name__ == "__main__":
    main()
