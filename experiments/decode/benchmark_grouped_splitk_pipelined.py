#!/usr/bin/env python3
"""Matched decode ablation: query-head grouping, split-K, then loop pipelining.

A/B differ only in heads per program. B/C add partitioning and reduction.
C/D use identical K and warps and differ only in loop pipeline stages.
CUDA graph replay measures device execution, including the split-K reduction;
allocation, host dispatch, and auto-K selection are excluded. Repeated inputs
make this a warm-cache microbenchmark, not an end-to-end serving measurement.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from functools import partial
import json
from pathlib import Path
import random
import statistics
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
ROOT = EXPERIMENTS_DIR.parent
for path in (SCRIPT_DIR, ROOT / "engine" / "kvcache"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from grouped_splitk_validation import attention_reference, check_output, correctness_cases, make_inputs


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", default="1,4,8,16,32,64,96,128,256")
    parser.add_argument("--context-lengths", default="512,2048,8192,16384")
    parser.add_argument("--target-occupancy", type=int, default=4,
                        help="Target launched programs per SM; not measured occupancy")
    parser.add_argument("--split-k", type=int, default=None,
                        help="Override auto-K for C/D, e.g. sweep 1, 2, 4, 8 in separate runs")
    parser.add_argument("--heads-per-program", type=int, choices=(2, 3, 6), default=6)
    parser.add_argument("--pipeline-stages", type=int, default=2)
    parser.add_argument("--num-warps", type=int, choices=(1, 2, 4, 8), default=4)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--graph-repeats", type=int, default=16,
                        help="Attention calls per captured graph; amortizes replay overhead")
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dump-ir", action="store_true", help="Save TTGIR/PTX for C/D inspection")
    parser.add_argument("--output-dir", type=Path,
                        default=EXPERIMENTS_DIR / "results" / "ablation-grouped-splitk-pipelined")
    return parser


def ablation_configs(heads_per_program, split_k, num_warps=4, pipeline_stages=2):
    common = {"num_warps": num_warps}
    return {
        "A": dict(common, heads_per_program=1, split_k=1, pipelined=False, num_stages=1),
        "B": dict(common, heads_per_program=heads_per_program, split_k=1, pipelined=False, num_stages=1),
        "C": dict(common, heads_per_program=heads_per_program, split_k=split_k, pipelined=False, num_stages=1),
        "D": dict(common, heads_per_program=heads_per_program, split_k=split_k, pipelined=True,
                  num_stages=pipeline_stages),
    }


def correctness_preflight(attention, configs, *, page_size, dtype, seed):
    records = []
    # Include K > pages regardless of the benchmark's actual K to cover empty partitions.
    checks = dict(configs)
    checks["empty_C"] = dict(configs["C"], split_k=7)
    checks["empty_D"] = dict(configs["D"], split_k=7)
    checks["overflow_C"] = dict(configs["C"], split_k=2)
    checks["overflow_D"] = dict(configs["D"], split_k=2)
    for case, tensors, options in correctness_cases(page_size=page_size, dtype=dtype, seed=seed):
        expected = attention_reference(*tensors, **options)
        for label, config in checks.items():
            actual = attention(*tensors, **config, **options)
            error = check_output(actual, expected)
            records.append({"case": case, "config": label, "max_abs_error": error})
    return records


def benchmark_kernel(fn, warmup, reps, graph_repeats=16):
    import torch

    # Warm up and capture on a side stream, with explicit dependency on input creation.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(max(1, warmup)):
            fn()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(graph_repeats):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    for _ in range(max(1, warmup)):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / graph_repeats)
    return {"median_ms": statistics.median(samples), "samples_ms": samples}


def kernel_diagnostics(kernels, dump_prefix=None):
    records = {}
    for role, kernel in kernels.items():
        records[role] = {
            "registers_per_thread": kernel.n_regs,
            "spills": kernel.n_spills,
            "shared_bytes": kernel.metadata.shared,
        }
        if dump_prefix is not None:
            for kind in ("ttgir", "ptx"):
                if kind in kernel.asm:
                    path = Path(f"{dump_prefix}-{role}.{kind}")
                    path.write_text(kernel.asm[kind])
                    records[role][kind] = str(path)
    return records


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        batches = [int(x) for x in args.batch_sizes.split(",")]
        contexts = [int(x) for x in args.context_lengths.split(",")]
        if min(batches + contexts) < 1:
            raise ValueError("batch sizes and contexts must be positive")
        if args.page_size < 1 or args.page_size & (args.page_size - 1):
            raise ValueError("page size must be a power of two")
        if min(args.target_occupancy, args.repetitions, args.graph_repeats) < 1 or args.warmup < 0:
            raise ValueError("target/repetitions/graph-repeats must be positive; warmup non-negative")
        if args.pipeline_stages < 2 or (args.split_k is not None and args.split_k < 1):
            raise ValueError("pipeline-stages must be >= 2 and split-k positive")
    except ValueError as exc:
        parser.error(str(exc))

    import torch
    import triton
    from paged_decode_grouped_splitk_pipelined import grouped_splitk_attention, compute_split_k, get_num_sms
    from paged_decode_attention import paged_decode_attention

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA and Triton")
    torch.backends.cuda.matmul.allow_tf32 = False
    dtype = getattr(torch, args.dtype)
    num_sms = get_num_sms()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    prefix = args.output_dir / f"ablation-{timestamp}"
    print(f"Device: {torch.cuda.get_device_name()} ({num_sms} SMs)")
    print("Timing: warm-cache CUDA graph device latency, including split-K reduction")
    print("A→B: grouping; B→C: split-K; C→D: loop pipeline stages")
    print("Checking ragged/strided inputs, empty splits, and FP16 accumulator overflow...", flush=True)
    preflight = correctness_preflight(
        grouped_splitk_attention,
        ablation_configs(args.heads_per_program, args.split_k or 2, args.num_warps, args.pipeline_stages),
        page_size=args.page_size, dtype=dtype, seed=args.seed,
    )
    # Production is only used on contiguous, positive-length inputs.
    prod_inputs = make_inputs([1, args.page_size, args.page_size + 1],
                              page_size=args.page_size, dtype=dtype, seed=args.seed)
    check_output(paged_decode_attention(*prod_inputs), attention_reference(*prod_inputs))
    print(f"Correctness preflight passed ({len(preflight)} candidate checks).", flush=True)

    results, details = [], []
    rng = random.Random(args.seed)
    for batch in batches:
        for context in contexts:
            tensors = make_inputs([context] * batch, page_size=args.page_size, dtype=dtype, seed=args.seed)
            blocks = (context + args.page_size - 1) // args.page_size
            head_tiles = 6 // args.heads_per_program
            actual_k = args.split_k if args.split_k is not None else compute_split_k(
                batch, 2, head_tiles, blocks, args.target_occupancy, num_sms=num_sms,
            )
            configs = ablation_configs(args.heads_per_program, actual_k, args.num_warps, args.pipeline_stages)
            operations = {"prod": partial(paged_decode_attention, *tensors)}
            operations.update({label: partial(grouped_splitk_attention, *tensors, **config)
                               for label, config in configs.items()})
            # Check every measured shape before timing. Production was independently
            # checked above; the adversarial preflight checks against dense FP32 math.
            expected = operations["prod"]()
            errors, resources = {}, {}
            for label, config in configs.items():
                compiled = {}
                actual = grouped_splitk_attention(*tensors, **config, diagnostics=compiled)
                errors[label] = check_output(actual, expected)
                dump_prefix = f"{prefix}-B{batch}-L{context}-{label}" if args.dump_ir else None
                resources[label] = kernel_diagnostics(compiled, dump_prefix)
            order = list(operations)
            rng.shuffle(order)
            timings = {label: benchmark_kernel(operations[label], args.warmup, args.repetitions, args.graph_repeats)
                       for label in order}
            ms = {label: record["median_ms"] for label, record in timings.items()}
            programs = {label: batch * 2 * (6 // config["heads_per_program"]) * config["split_k"]
                        for label, config in configs.items()}
            row = {
                "batch_size": batch, "context_length": context, "num_kv_blocks": blocks,
                "actual_k": actual_k, "heads_per_program": args.heads_per_program,
                "pipeline_stages": args.pipeline_stages,
                "min_pages_per_split": blocks // actual_k,
                "max_pages_per_split": (blocks + actual_k - 1) // actual_k,
                "active_programs_C": batch * 2 * head_tiles * min(actual_k, blocks),
                "launched_programs_per_sm_C": programs["C"] / num_sms,
                **{f"programs_{label}": count for label, count in programs.items()},
                "prod_ms": ms["prod"],
                **{f"time_{label}_ms": ms[label] for label in configs},
                "speedup_B_vs_A": ms["A"] / ms["B"],
                "speedup_C_vs_B": ms["B"] / ms["C"],
                "speedup_D_vs_C": ms["C"] / ms["D"],
                "speedup_D_vs_prod": ms["prod"] / ms["D"],
            }
            results.append(row)
            details.append({"batch_size": batch, "context_length": context, "configs": configs,
                            "timing_order": order, "timings": timings, "max_abs_errors": errors,
                            "kernel_resources": resources})
            print(f"B={batch:3d} L={context:5d} K={actual_k:3d} "
                  f"group={row['speedup_B_vs_A']:.2f}x split={row['speedup_C_vs_B']:.2f}x "
                  f"pipeline={row['speedup_D_vs_C']:.2f}x", flush=True)
            # Persist each completed shape; failures abort instead of being hidden.
            with prefix.with_suffix(".csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
            with prefix.with_suffix(".json").open("w") as f:
                json.dump({"config": {**vars(args), "output_dir": str(args.output_dir)},
                           "device": torch.cuda.get_device_name(), "num_sms": num_sms,
                           "torch_version": torch.__version__, "triton_version": triton.__version__,
                           "timing_mode": "warm_cache_cuda_graph_device_latency",
                           "correctness_preflight": preflight, "results": results, "details": details}, f, indent=2)
    print(f"Wrote {len(results)} shapes to {prefix}.csv and .json")


if __name__ == "__main__":
    main()
