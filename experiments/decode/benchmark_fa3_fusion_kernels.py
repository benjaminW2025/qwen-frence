#!/usr/bin/env python3
"""Cheap correctness and microbenchmark gate for FA3-era decode fusions."""

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
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[8, 64])
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "experiments/results/fa3-fusion-kernels")
    return parser


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


def check(torch, actual, expected, *, atol=2e-3, rtol=2e-3):
    difference = (actual.float() - expected.float()).abs()
    outside = difference > atol + rtol * expected.float().abs()
    return {"max_abs": float(difference.max()), "mean_abs": float(difference.mean()),
            "mismatched": int(outside.sum()), "elements": outside.numel(),
            "atol": atol, "rtol": rtol, "allclose": not bool(outside.any())}


def run_batch(torch, batch, warmups, repetitions, seed):
    import torch.nn.functional as F
    from fused_rms import residual_add_rms_norm, rms_norm
    from fused_qkv_rope_cache import fused_qkv_rope_cache
    from rope import rope
    from rope_kv_write import native_decode_rope_kv_write

    generator = torch.Generator(device="cuda").manual_seed(seed + batch)
    residual = torch.randn(batch, 1, 1536, device="cuda", dtype=torch.float16,
                           generator=generator)
    branch = torch.randn(residual.shape, device="cuda", dtype=residual.dtype,
                         generator=generator)
    weight = torch.randn(1536, device="cuda", dtype=residual.dtype,
                         generator=generator)

    def residual_baseline():
        summed = residual + branch
        return summed, rms_norm(summed.view(-1, 1536), weight).view_as(summed)

    expected_sum, expected_norm = residual_baseline()
    actual_sum, actual_norm = residual_add_rms_norm(residual, branch, weight)
    residual_check = {"sum": check(torch, actual_sum, expected_sum, atol=0, rtol=0),
                      "norm": check(torch, actual_norm, expected_norm)}
    residual_baseline_timing = measure(torch, residual_baseline, warmups, repetitions)
    residual_fused_timing = measure(
        torch, lambda: residual_add_rms_norm(residual, branch, weight),
        warmups, repetitions,
    )

    q = torch.randn(batch, 12, 1, 128, device="cuda", dtype=torch.float16,
                    generator=generator)
    k = torch.randn(batch, 2, 1, 128, device="cuda", dtype=torch.float16,
                    generator=generator)
    v = torch.randn(k.shape, device="cuda", dtype=k.dtype, generator=generator)
    positions = torch.arange(4096, 4096 + batch, device="cuda", dtype=torch.int32)
    slots = torch.arange(batch, device="cuda", dtype=torch.long)
    baseline_k = torch.empty(batch, 16, 2, 128, device="cuda", dtype=torch.float16)
    baseline_v = torch.empty_like(baseline_k)
    fused_k = torch.empty_like(baseline_k)
    fused_v = torch.empty_like(baseline_v)

    def qkv_baseline():
        q_out = rope(q, positions[:, None], 1_000_000.0)
        k_out = rope(k, positions[:, None], 1_000_000.0)
        baseline_k.view(-1, 2, 128).index_copy_(
            0, slots, k_out[:, :, 0, :].contiguous()
        )
        baseline_v.view(-1, 2, 128).index_copy_(
            0, slots, v[:, :, 0, :].contiguous()
        )
        return q_out

    expected_q = qkv_baseline()
    actual_q = native_decode_rope_kv_write(
        q, k, v, positions, slots, fused_k, fused_v,
        base=1_000_000.0, rotate_q=True,
    )
    torch.cuda.synchronize()
    qkv_check = {
        "q": check(torch, actual_q, expected_q),
        "k_cache": check(torch, fused_k.view(-1, 2, 128)[slots],
                         baseline_k.view(-1, 2, 128)[slots]),
        "v_cache": check(torch, fused_v.view(-1, 2, 128)[slots],
                         baseline_v.view(-1, 2, 128)[slots], atol=0, rtol=0),
    }
    qkv_baseline_timing = measure(torch, qkv_baseline, warmups, repetitions)
    qkv_fused_timing = measure(
        torch,
        lambda: native_decode_rope_kv_write(
            q, k, v, positions, slots, fused_k, fused_v,
            base=1_000_000.0, rotate_q=True,
        ),
        warmups, repetitions,
    )

    hidden = torch.randn(batch, 1, 1536, device="cuda", dtype=torch.float16,
                         generator=generator)
    packed_weight = torch.randn(2048, 1536, device="cuda", dtype=torch.float16,
                                generator=generator).mul_(0.02)
    packed_bias = torch.randn(2048, device="cuda", dtype=torch.float16,
                              generator=generator).mul_(0.02)
    full_baseline_k = torch.empty_like(baseline_k)
    full_baseline_v = torch.empty_like(baseline_v)
    full_fused_k = torch.empty_like(fused_k)
    full_fused_v = torch.empty_like(fused_v)

    def full_qkv_baseline():
        packed = F.linear(hidden, packed_weight, packed_bias)
        q_raw, k_raw, v_raw = packed.split((1536, 256, 256), dim=-1)
        q_heads = q_raw.view(batch, 1, 12, 128).transpose(1, 2)
        k_heads = k_raw.view(batch, 1, 2, 128).transpose(1, 2)
        v_heads = v_raw.view(batch, 1, 2, 128).transpose(1, 2)
        q_out = rope(q_heads, positions[:, None], 1_000_000.0)
        k_out = rope(k_heads, positions[:, None], 1_000_000.0)
        full_baseline_k.view(-1, 2, 128).index_copy_(
            0, slots, k_out[:, :, 0, :].contiguous()
        )
        full_baseline_v.view(-1, 2, 128).index_copy_(
            0, slots, v_heads[:, :, 0, :].contiguous()
        )
        return q_out[:, :, 0, :]

    expected_full_q = full_qkv_baseline()
    actual_full_q = fused_qkv_rope_cache(
        hidden, packed_weight, packed_bias, positions, slots,
        full_fused_k, full_fused_v,
    )
    torch.cuda.synchronize()
    full_qkv_check = {
        # Tensor-core accumulation routes need not be bitwise identical to cuBLAS.
        "q": check(torch, actual_full_q, expected_full_q, atol=2e-2, rtol=2e-2),
        "k_cache": check(torch, full_fused_k.view(-1, 2, 128)[slots],
                         full_baseline_k.view(-1, 2, 128)[slots],
                         atol=2e-2, rtol=2e-2),
        "v_cache": check(torch, full_fused_v.view(-1, 2, 128)[slots],
                         full_baseline_v.view(-1, 2, 128)[slots],
                         atol=2e-2, rtol=2e-2),
    }
    full_baseline_timing = measure(torch, full_qkv_baseline, warmups, repetitions)
    full_fused_timing = measure(
        torch,
        lambda: fused_qkv_rope_cache(
            hidden, packed_weight, packed_bias, positions, slots,
            full_fused_k, full_fused_v,
        ),
        warmups, repetitions,
    )
    return {
        "batch": batch,
        "residual_rmsnorm": {
            "correctness": residual_check,
            "baseline": residual_baseline_timing,
            "fused": residual_fused_timing,
            "speedup": (residual_baseline_timing["median_ms"] /
                        residual_fused_timing["median_ms"]),
        },
        "qkv_rope_kv_postprocess": {
            "correctness": qkv_check,
            "baseline": qkv_baseline_timing,
            "fused": qkv_fused_timing,
            "speedup": qkv_baseline_timing["median_ms"] / qkv_fused_timing["median_ms"],
        },
        "full_qkv_rope_kv": {
            "correctness": full_qkv_check,
            "baseline": full_baseline_timing,
            "fused": full_fused_timing,
            "speedup": full_baseline_timing["median_ms"] / full_fused_timing["median_ms"],
        },
    }


def main():
    args = build_parser().parse_args()
    if not args.batch_sizes or min(args.batch_sizes) < 1:
        raise ValueError("batch sizes must be positive")
    if args.warmups < 0 or args.repetitions < 1:
        raise ValueError("warmups must be nonnegative and repetitions positive")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    rows = [run_batch(torch, batch, args.warmups, args.repetitions, args.seed)
            for batch in args.batch_sizes]
    passed = all(
        check_row["allclose"]
        for row in rows
        for candidate in ("residual_rmsnorm", "qkv_rope_kv_postprocess",
                          "full_qkv_rope_kv")
        for check_row in row[candidate]["correctness"].values()
    )
    report = {"schema_version": 1,
              "created_at": datetime.now(timezone.utc).isoformat(),
              "device": torch.cuda.get_device_name(), "torch_version": torch.__version__,
              "configuration": {**vars(args), "output_dir": str(args.output_dir)},
              "rows": rows, "all_correct": passed}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_dir / f"fa3-fusion-kernels-{stamp}.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for row in rows:
        print(f"B={row['batch']}: residual+RMSNorm "
              f"{row['residual_rmsnorm']['speedup']:.3f}x; QKV postprocess "
              f"{row['qkv_rope_kv_postprocess']['speedup']:.3f}x; full QKV "
              f"{row['full_qkv_rope_kv']['speedup']:.3f}x")
    print(f"all_correct={passed}; wrote {output}")
    if not passed:
        raise SystemExit("fusion kernel correctness gate failed")


if __name__ == "__main__":
    main()
