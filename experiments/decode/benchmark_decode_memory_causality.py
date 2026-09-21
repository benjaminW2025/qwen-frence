#!/usr/bin/env python3
"""Causal decode study without privileged GPU performance counters.

The duplicated-MHA control stores the same K/V values at six distinct physical
addresses. It performs identical attention math with identical launch count but
removes cross-query-head address reuse. Warm and explicitly L2-evicted timings
therefore isolate how much the original H=1 kernel already benefits from cache
reuse. The optional FA3 arm runs vLLM's bundled Hopper kernel directly on the
same page pool and metadata, separating kernel quality from executor overhead.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import partial
import json
from pathlib import Path
import statistics
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (HERE, ROOT / "custom_kernels", ROOT / "engine" / "kvcache"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--split-k", type=int, default=22)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--l2-evict-mib", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--skip-fa3", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "experiments" / "results" / "decode-memory-causality",
    )
    return parser


def error_report(torch, actual, expected):
    difference = (actual.float() - expected.float()).abs()
    tolerance = 2e-2 if expected.dtype == torch.bfloat16 else 2e-3
    mismatch = difference > tolerance + tolerance * expected.float().abs()
    return {
        "max_abs": difference.max().item(),
        "mean_abs": difference.mean().item(),
        "mismatched": int(mismatch.sum()),
        "elements": mismatch.numel(),
        "atol": tolerance,
        "rtol": tolerance,
        "allclose": bool(not mismatch.any().item()),
    }


def measure(torch, operation, *, warmups, repetitions, evict=None):
    for _ in range(warmups):
        operation()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repetitions):
        if evict is not None:
            evict.add_(1)
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        operation()
        end.record()
        end.synchronize()
        samples.append(float(begin.elapsed_time(end)))
    return {"median_ms": statistics.median(samples), "samples_ms": samples}


def main():
    args = build_parser().parse_args()
    if min(args.batch_size, args.context_length, args.split_k,
           args.warmups, args.repetitions, args.l2_evict_mib) < 1:
        raise ValueError("all numeric arguments must be positive")

    import torch
    from grouped_splitk_validation import make_inputs
    from paged_decode_grouped_splitk import grouped_splitk_decode_attention
    from paged_decode_grouped_splitk_pipelined import grouped_splitk_attention
    from paged_decode_native_grouped import (
        native_grouped_decode_diagnostics,
        native_grouped_stage_cycles,
        native_grouped_splitk_decode_attention,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    dtype = getattr(torch, args.dtype)
    tensors = make_inputs(
        [args.context_length] * args.batch_size,
        page_size=16,
        dtype=dtype,
        seed=args.seed,
    )
    q, k_pool, v_pool, block_table, seq_lens = tensors
    native_diagnostics = native_grouped_decode_diagnostics(q, block_table)
    print(f"native compiled resources: {native_diagnostics}", flush=True)
    stage_names = ("global_stage_and_barrier", "qk", "softmax", "probability_times_v")
    cycle_tensor = native_grouped_stage_cycles(
        q, k_pool, v_pool, block_table, seq_lens, split_k=args.split_k
    ).double().reshape(-1, 4)
    cycle_totals = cycle_tensor.sum(dim=1).clamp_min(1)
    shares = cycle_tensor / cycle_totals[:, None]
    stage_cycles = {
        name: {
            "median_cycles_per_cta": cycle_tensor[:, index].median().item(),
            "mean_cycles_per_cta": cycle_tensor[:, index].mean().item(),
            "median_fraction_per_cta": shares[:, index].median().item(),
            "mean_fraction_per_cta": shares[:, index].mean().item(),
        }
        for index, name in enumerate(stage_names)
    }
    print(f"native instrumented stage cycles: {stage_cycles}", flush=True)

    # Same numerical K/V for every Q head, but stored at distinct addresses.
    # This changes cache reuse only: output values and CTA count are unchanged.
    k_duplicated = k_pool.repeat_interleave(6, dim=2)
    v_duplicated = v_pool.repeat_interleave(6, dim=2)
    duplicated_tensors = q, k_duplicated, v_duplicated, block_table, seq_lens

    def triton_resources(candidate_tensors):
        compiled = {}
        grouped_splitk_attention(
            *candidate_tensors, heads_per_program=1, split_k=args.split_k,
            pipelined=True, num_warps=4, num_stages=3, diagnostics=compiled,
        )
        torch.cuda.synchronize()
        return {
            role: {
                "registers_per_thread": kernel.n_regs,
                "spills": kernel.n_spills,
                "shared_bytes": kernel.metadata.shared,
            }
            for role, kernel in compiled.items()
        }

    triton_diagnostics = {
        "shared_gqa": triton_resources(tensors),
        "duplicated_mha": triton_resources(duplicated_tensors),
    }
    print(f"triton compiled resources: {triton_diagnostics}", flush=True)

    shape = (args.batch_size, args.split_k, 12, 128)
    def partials():
        output = torch.empty(shape, dtype=torch.float32, device=q.device)
        maximum = torch.empty(shape[:3], dtype=torch.float32, device=q.device)
        return output, maximum, torch.empty_like(maximum)

    operations = {
        "current_shared_gqa": partial(
            grouped_splitk_decode_attention, *tensors,
            split_k=args.split_k, heads_per_program=1, num_warps=4,
            num_stages=3, partials=partials(),
        ),
        "current_duplicated_mha": partial(
            grouped_splitk_decode_attention, *duplicated_tensors,
            split_k=args.split_k, heads_per_program=1, num_warps=4,
            num_stages=3, partials=partials(),
        ),
        "native_shared_gqa": partial(
            native_grouped_splitk_decode_attention, *tensors,
            split_k=args.split_k, partials=partials(),
        ),
    }
    fa3_error = None
    if not args.skip_fa3:
        try:
            from paged_decode_fa3 import fa3_paged_decode_attention
            # Use the same explicit split count first; tuning can follow only if
            # the exact-layout oracle is correct and competitive.
            operations["fa3_auto_shared_gqa"] = partial(
                fa3_paged_decode_attention, *tensors, num_splits=0,
            )
            operations["fa3_k22_shared_gqa"] = partial(
                fa3_paged_decode_attention, *tensors, num_splits=args.split_k,
            )
        except Exception as exc:
            fa3_error = f"{type(exc).__name__}: {exc}"

    expected = operations["current_shared_gqa"]()
    checks, runnable = {}, {}
    for name, operation in operations.items():
        try:
            actual = operation()
            torch.cuda.synchronize()
            check = error_report(torch, actual, expected)
            checks[name] = check
            if check["allclose"]:
                runnable[name] = operation
            else:
                checks[name]["error"] = "strict attention-output mismatch"
        except Exception as exc:
            torch.cuda.synchronize()
            checks[name] = {"allclose": False, "error": f"{type(exc).__name__}: {exc}"}
            if name.startswith("fa3_"):
                fa3_error = checks[name]["error"]

    required = ("current_shared_gqa", "current_duplicated_mha", "native_shared_gqa")
    failed = [name for name in required if name not in runnable]
    if failed:
        raise AssertionError(f"required correctness arms failed: {failed}; checks={checks}")

    evict_bytes = args.l2_evict_mib * 1024 * 1024
    eviction = torch.empty(evict_bytes, dtype=torch.uint8, device=q.device)
    rows = {}
    for name, operation in runnable.items():
        warm = measure(torch, operation, warmups=args.warmups,
                       repetitions=args.repetitions)
        cold = measure(torch, operation, warmups=args.warmups,
                       repetitions=args.repetitions, evict=eviction)
        rows[name] = {"warm": warm, "l2_evicted": cold, "correctness": checks[name]}
        print(
            f"{name:<24} warm={warm['median_ms']:.4f} ms "
            f"evicted={cold['median_ms']:.4f} ms",
            flush=True,
        )

    shared = rows["current_shared_gqa"]["warm"]["median_ms"]
    duplicate = rows["current_duplicated_mha"]["warm"]["median_ms"]
    native = rows["native_shared_gqa"]["warm"]["median_ms"]
    effects = {
        "physical_duplication_penalty_warm": duplicate / shared,
        "native_speedup_vs_current_warm": shared / native,
        "current_cold_penalty": (
            rows["current_shared_gqa"]["l2_evicted"]["median_ms"] / shared
        ),
        "duplicated_cold_penalty": (
            rows["current_duplicated_mha"]["l2_evicted"]["median_ms"] / duplicate
        ),
        "native_cold_penalty": (
            rows["native_shared_gqa"]["l2_evicted"]["median_ms"] / native
        ),
    }
    if "fa3_auto_shared_gqa" in rows:
        fa3 = rows["fa3_auto_shared_gqa"]["warm"]["median_ms"]
        effects.update({
            "fa3_auto_speedup_vs_current_warm": shared / fa3,
            "fa3_auto_speedup_vs_native_warm": native / fa3,
            "fa3_auto_cold_penalty": (
                rows["fa3_auto_shared_gqa"]["l2_evicted"]["median_ms"] / fa3
            ),
        })
    if "fa3_k22_shared_gqa" in rows:
        fa3_k = rows["fa3_k22_shared_gqa"]["warm"]["median_ms"]
        effects["fa3_k22_speedup_vs_native_warm"] = native / fa3_k

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_dir / f"memory-causality-{stamp}.json"
    output.write_text(json.dumps({
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "configuration": {**vars(args), "output_dir": str(args.output_dir)},
        "logical_bytes": {
            "unique_kv": args.batch_size * 2 * args.context_length * 128 * 2 * 2,
            "duplicated_kv": args.batch_size * 12 * args.context_length * 128 * 2 * 2,
        },
        "native_compiled_resources": native_diagnostics,
        "native_instrumented_stage_cycles": stage_cycles,
        "triton_compiled_resources": triton_diagnostics,
        "checks": checks,
        "fa3_error": fa3_error,
        "rows": rows,
        "effects": effects,
    }, indent=2))
    print(json.dumps(effects, indent=2))
    if fa3_error:
        print(f"FA3 unavailable: {fa3_error}")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
