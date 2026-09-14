#!/usr/bin/env python3
"""Test native-layout decode RoPE/KV fusion against the current four-op path."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "baseline"))
sys.path.insert(0, str(ROOT / "benchmarks"))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=("smoke", "full"), default="full")
    p.add_argument("--repetitions", type=int, default=30)
    p.add_argument("--warmups", type=int, default=5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", type=Path,
                   default=ROOT / "experiments/results/native-decode-rope-kv")
    return p


def measure_pair(torch, baseline, candidate, warmups, repetitions):
    for _ in range(warmups):
        baseline()
        candidate()
    torch.cuda.synchronize()
    measurements = {"baseline": {"gpu_ms": [], "wall_ms": []},
                    "fused": {"gpu_ms": [], "wall_ms": []}}
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for repetition in range(repetitions):
        order = (("baseline", baseline), ("fused", candidate))
        if repetition % 2:
            order = order[::-1]
        for name, fn in order:
            wall_start = time.perf_counter()
            start.record()
            fn()
            end.record()
            end.synchronize()
            measurements[name]["wall_ms"].append((time.perf_counter() - wall_start) * 1000)
            measurements[name]["gpu_ms"].append(float(start.elapsed_time(end)))
    for item in measurements.values():
        item["gpu_median_ms"] = statistics.median(item["gpu_ms"])
        item["wall_median_ms"] = statistics.median(item["wall_ms"])
    return measurements


def case(torch, rope, fused, *, batch, position, packed, warp, repetitions, warmups, device):
    q_heads, kv_heads, dim, page_size = 12, 2, 128, 16
    generator = torch.Generator(device=device).manual_seed(
        117 + batch * 1009 + position * 7 + int(packed)
    )

    def projection(width):
        return torch.randn(batch, 1, width * dim, generator=generator,
                           device=device, dtype=torch.float16)

    if packed:
        q_flat, k_flat, v_flat = projection(q_heads + 2 * kv_heads).split(
            (q_heads * dim, kv_heads * dim, kv_heads * dim), dim=-1
        )
    else:
        q_flat, k_flat, v_flat = projection(q_heads), projection(kv_heads), projection(kv_heads)
    q = q_flat.view(batch, 1, q_heads, dim).transpose(1, 2)
    k = k_flat.view(batch, 1, kv_heads, dim).transpose(1, 2)
    v = v_flat.view(batch, 1, kv_heads, dim).transpose(1, 2)
    positions = torch.full((batch,), position, device=device, dtype=torch.int32)
    blocks = batch * 2
    slots = torch.randperm(blocks * page_size, generator=generator, device=device)[:batch]
    pool_shape = (blocks, page_size, kv_heads, dim)
    old_k = torch.zeros(pool_shape, device=device, dtype=torch.float16)
    old_v = torch.zeros_like(old_k)
    new_k = torch.zeros_like(old_k)
    new_v = torch.zeros_like(old_k)

    def baseline():
        rotated_q = rope(q, positions[:, None], 1_000_000.0)
        rotated_k = rope(k, positions[:, None], 1_000_000.0)
        old_k.view(-1, kv_heads, dim).index_copy_(0, slots, rotated_k[:, :, 0, :].contiguous())
        old_v.view(-1, kv_heads, dim).index_copy_(0, slots, v[:, :, 0, :].contiguous())
        return rotated_q

    def candidate():
        return fused(q, k, v, positions, slots, new_k, new_v,
                     base=1_000_000.0, num_warps=warp)

    expected_q = baseline()
    actual_q = candidate()
    torch.cuda.synchronize()
    expected_k = old_k.view(-1, kv_heads, dim).index_select(0, slots)
    actual_k = new_k.view(-1, kv_heads, dim).index_select(0, slots)
    expected_v = old_v.view(-1, kv_heads, dim).index_select(0, slots)
    actual_v = new_v.view(-1, kv_heads, dim).index_select(0, slots)
    torch.testing.assert_close(actual_q, expected_q, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(actual_k, expected_k, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(actual_v, expected_v, atol=0, rtol=0)

    measurements = measure_pair(torch, baseline, candidate, warmups, repetitions)
    baseline_time, fused_time = measurements["baseline"], measurements["fused"]
    return {
        "batch": batch, "position": position, "packed_projection": packed,
        "num_warps": warp,
        "baseline": baseline_time, "fused": fused_time,
        "gpu_speedup": baseline_time["gpu_median_ms"] / fused_time["gpu_median_ms"],
        "wall_speedup": baseline_time["wall_median_ms"] / fused_time["wall_median_ms"],
        "q_max_abs_error": float((actual_q.float() - expected_q.float()).abs().max()),
        "k_max_abs_error": float((actual_k.float() - expected_k.float()).abs().max()),
    }


def main():
    args = parser().parse_args()
    if args.repetitions < 1 or args.warmups < 0:
        raise ValueError("invalid repetition or warmup count")
    import torch
    from kernel_dispatch import rope
    from run_benchmarks import system_metadata
    from importlib.util import module_from_spec, spec_from_file_location
    path = ROOT / "custom_kernels/rope_kv_write.py"
    spec = spec_from_file_location("native_rope_kv_experiment", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    fused = module.native_decode_rope_kv_write
    if not torch.cuda.is_available() or torch.device(args.device).type != "cuda":
        raise RuntimeError("requires CUDA")

    batches = (1, 8) if args.preset == "smoke" else (1, 8, 32, 64, 128, 256)
    positions = (512,) if args.preset == "smoke" else (512, 8192)
    warps = (2,) if args.preset == "smoke" else (1, 2, 4)
    rows = []
    for batch in batches:
        for position in positions:
            for packed in (False, True):
                for warp in warps:
                    row = case(torch, rope, fused, batch=batch, position=position,
                               packed=packed, warp=warp, repetitions=args.repetitions,
                               warmups=args.warmups, device=args.device)
                    rows.append(row)
                    print(f"B={batch} pos={position} packed={packed} warps={warp}: "
                          f"GPU {row['gpu_speedup']:.2f}x wall {row['wall_speedup']:.2f}x",
                          flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / (
        "native-decode-rope-kv-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json"
    )
    payload = {
        "scope": "native-layout Q/K RoPE plus paged K/V placement only; full model excluded",
        "system": system_metadata(),
        "configuration": {**vars(args), "output_dir": str(args.output_dir)},
        "rows": rows,
    }
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(f"results: {path}")


if __name__ == "__main__":
    main()
