#!/usr/bin/env python3
"""Correctness and timing gate for the two packed-prefill fusion candidates.

This measures both the isolated post-GEMM work and the QKV GEMM boundary. The
full piecewise-engine gate lives in ``benchmark_integrated_graph.py
--prefill-fusions`` so a fast microkernel cannot be accepted on its own.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (ROOT / "custom_kernels", ROOT / "baseline"):
    sys.path.insert(0, str(path))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--token-buckets", type=int, nargs="+",
                        default=[256, 2048, 4096, 8192])
    result.add_argument("--padding-rows", type=int, default=17,
                        help="make this many tail rows inactive to test bucket masking")
    result.add_argument("--warmups", type=int, default=5)
    result.add_argument("--repetitions", type=int, default=30)
    result.add_argument("--seed", type=int, default=20260914)
    result.add_argument("--output-dir", type=Path,
                        default=ROOT / "experiments/results/prefill-fusion-kernels")
    return result


def measure(torch, operation, warmups, repetitions):
    for _ in range(warmups):
        operation()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repetitions):
        begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        begin.record()
        operation()
        end.record()
        end.synchronize()
        samples.append(float(begin.elapsed_time(end)))
    return {"median_ms": statistics.median(samples), "samples_ms": samples}


def compare(torch, actual, expected, *, atol=2e-2, rtol=2e-2):
    difference = (actual.float() - expected.float()).abs()
    outside = difference > atol + rtol * expected.float().abs()
    return {"max_abs": float(difference.max()), "mean_abs": float(difference.mean()),
            "mismatched": int(outside.sum()), "elements": outside.numel(),
            "atol": atol, "rtol": rtol, "allclose": not bool(outside.any())}


def run_bucket(torch, rows, padding_rows, warmups, repetitions, seed):
    import torch.nn.functional as F
    from fused_rms import residual_add_rms_norm, rms_norm
    from masked_kv_write import masked_kv_write
    from packed_qkv_rope_cache import packed_qkv_rope_cache
    from rope import rope

    if padding_rows >= rows:
        raise ValueError("padding rows must be smaller than every token bucket")
    live = rows - padding_rows
    generator = torch.Generator(device="cuda").manual_seed(seed + rows)
    dtype, device = torch.float16, "cuda"
    packed = torch.randn(rows, 2048, device=device, dtype=dtype, generator=generator)
    positions = torch.zeros(rows, device=device, dtype=torch.long)
    positions[:live] = torch.arange(live, device=device)
    slots = torch.zeros(rows, device=device, dtype=torch.long)
    slots[:live] = torch.arange(16, 16 + live, device=device)
    valid_tokens = torch.tensor(live, device=device, dtype=torch.int32)
    blocks = (live + 31) // 16 + 1
    baseline_k = torch.full((blocks, 16, 2, 128), float("nan"), device=device, dtype=dtype)
    baseline_v = torch.full_like(baseline_k, float("nan"))
    fused_k = torch.full_like(baseline_k, float("nan"))
    fused_v = torch.full_like(baseline_v, float("nan"))

    def qkv_baseline(packed_output=packed):
        q_raw, k_raw, v_raw = packed_output.split((1536, 256, 256), dim=-1)
        q = q_raw.view(rows, 12, 128).transpose(0, 1).unsqueeze(0)
        k = k_raw.view(rows, 2, 128).transpose(0, 1).unsqueeze(0)
        v = v_raw.view(rows, 2, 128).transpose(0, 1).unsqueeze(0)
        q = rope(q, positions[None, :], 1_000_000.0)
        k = rope(k, positions[None, :], 1_000_000.0)
        masked_kv_write(k, v, slots, valid_tokens, baseline_k, baseline_v)
        return q

    def qkv_fused(packed_output=packed):
        return packed_qkv_rope_cache(
            packed_output, positions, slots, fused_k, fused_v,
            valid_tokens=valid_tokens,
        ).transpose(0, 1).unsqueeze(0)

    expected_q = qkv_baseline()
    actual_q = qkv_fused()
    torch.cuda.synchronize()
    qkv_correctness = {
        "q_live": compare(torch, actual_q[:, :, :live], expected_q[:, :, :live]),
        "q_padding_is_zero": bool((actual_q[:, :, live:] == 0).all()),
        "k_cache": compare(torch, fused_k.view(-1, 2, 128)[slots[:live]],
                           baseline_k.view(-1, 2, 128)[slots[:live]]),
        "v_cache": compare(torch, fused_v.view(-1, 2, 128)[slots[:live]],
                           baseline_v.view(-1, 2, 128)[slots[:live]], atol=0, rtol=0),
        "padding_slot_untouched": bool(torch.isnan(fused_k.view(-1, 2, 128)[0]).all()
                                        and torch.isnan(fused_v.view(-1, 2, 128)[0]).all()),
    }
    baseline_epilogue = measure(torch, qkv_baseline, warmups, repetitions)
    fused_epilogue = measure(torch, qkv_fused, warmups, repetitions)

    # Include the identical cuBLAS projection on both sides. This is the honest
    # boundary-level effect expected inside the captured prefill graph.
    hidden = torch.randn(rows, 1536, device=device, dtype=dtype, generator=generator)
    weight = torch.randn(2048, 1536, device=device, dtype=dtype,
                         generator=generator).mul_(0.02)
    bias = torch.randn(2048, device=device, dtype=dtype, generator=generator).mul_(0.02)

    def qkv_boundary_baseline():
        return qkv_baseline(F.linear(hidden, weight, bias))

    def qkv_boundary_fused():
        return qkv_fused(F.linear(hidden, weight, bias))

    baseline_boundary = measure(torch, qkv_boundary_baseline, warmups, repetitions)
    fused_boundary = measure(torch, qkv_boundary_fused, warmups, repetitions)

    residual = torch.randn(rows, 1536, device=device, dtype=dtype, generator=generator)
    branch = torch.randn_like(residual)
    norm_weight = torch.randn(1536, device=device, dtype=dtype, generator=generator)

    def residual_baseline():
        summed = residual + branch
        return summed, rms_norm(summed, norm_weight)

    expected_sum, expected_norm = residual_baseline()
    actual_sum, actual_norm = residual_add_rms_norm(residual, branch, norm_weight)
    residual_correctness = {
        "sum": compare(torch, actual_sum, expected_sum, atol=0, rtol=0),
        "norm": compare(torch, actual_norm, expected_norm, atol=2e-3, rtol=2e-3),
    }
    baseline_residual = measure(torch, residual_baseline, warmups, repetitions)
    fused_residual = measure(
        torch, lambda: residual_add_rms_norm(residual, branch, norm_weight),
        warmups, repetitions,
    )
    return {
        "bucket_rows": rows, "live_rows": live,
        "packed_qkv_rope_kv": {
            "correctness": qkv_correctness,
            "epilogue_only": {"baseline": baseline_epilogue, "fused": fused_epilogue,
                              "speedup": baseline_epilogue["median_ms"] /
                                         fused_epilogue["median_ms"]},
            "with_cublas_qkv": {"baseline": baseline_boundary, "fused": fused_boundary,
                                "speedup": baseline_boundary["median_ms"] /
                                           fused_boundary["median_ms"]},
        },
        "residual_add_rmsnorm": {
            "correctness": residual_correctness,
            "baseline": baseline_residual, "fused": fused_residual,
            "speedup": baseline_residual["median_ms"] / fused_residual["median_ms"],
        },
    }


def row_passed(row):
    qkv = row["packed_qkv_rope_kv"]["correctness"]
    residual = row["residual_add_rmsnorm"]["correctness"]
    return (qkv["q_live"]["allclose"] and qkv["q_padding_is_zero"] and
            qkv["k_cache"]["allclose"] and qkv["v_cache"]["allclose"] and
            qkv["padding_slot_untouched"] and residual["sum"]["allclose"] and
            residual["norm"]["allclose"])


def main():
    args = parser().parse_args()
    if (not args.token_buckets or min(args.token_buckets) < 1 or
            args.padding_rows < 0 or args.warmups < 0 or args.repetitions < 1):
        raise ValueError("invalid bucket, padding, warmup, or repetition count")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    rows = [run_bucket(torch, value, args.padding_rows, args.warmups,
                       args.repetitions, args.seed) for value in args.token_buckets]
    passed = all(row_passed(row) for row in rows)
    payload = {"schema_version": 1,
               "created_at": datetime.now(timezone.utc).isoformat(),
               "status": "passed" if passed else "failed",
               "scope": "packed-prefill fusion microbenchmark; not an engine acceptance gate",
               "rows": rows}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporary = args.output_dir / "report.tmp"
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output_dir / "report.json")
    for row in rows:
        qkv = row["packed_qkv_rope_kv"]
        residual = row["residual_add_rmsnorm"]
        print(f"T={row['bucket_rows']} live={row['live_rows']}: "
              f"QKV epilogue={qkv['epilogue_only']['speedup']:.3f}x, "
              f"QKV+cuBLAS={qkv['with_cublas_qkv']['speedup']:.3f}x, "
              f"residual+RMSNorm={residual['speedup']:.3f}x, "
              f"correctness={row_passed(row)}")
    if not passed:
        raise SystemExit("prefill fusion correctness failed; inspect report.json")


if __name__ == "__main__":
    main()
