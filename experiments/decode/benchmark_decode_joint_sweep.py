#!/usr/bin/env python3
"""Joint sweep of decode query-head grouping, split-K, and loop pipeline stages.

Uses the existing grouped split-K kernel and warm-cache CUDA graph timing.
Mandatory hpp=1, K=1, and stage=1 controls enable matched one-axis comparisons.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from functools import partial
import json
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from benchmark_grouped_splitk_pipelined import benchmark_kernel, kernel_diagnostics
from grouped_splitk_validation import attention_reference, check_output, correctness_cases, make_inputs


def parse_axis(value, *, allowed=None, allow_auto=False):
    result = []
    for part in value.split(","):
        item = "auto" if allow_auto and part.strip() == "auto" else int(part)
        if item != "auto" and (item < 1 or (allowed is not None and item not in allowed)):
            raise ValueError(f"invalid axis value: {part}")
        if item not in result:
            result.append(item)
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", default="1,4,8,16,32,64,96,128,256")
    parser.add_argument("--context-lengths", default="512,2048,8192,16384")
    parser.add_argument("--heads-per-program", default="1,2,3,6")
    parser.add_argument("--k-splits", default="1,2,4,8,16,32,auto")
    parser.add_argument("--pipeline-stages", default="1,2,3", help="Comma-separated stages; 1 disables pipelining")
    parser.add_argument("--target-occupancy", type=int, default=4, help="Target launched programs per SM")
    parser.add_argument("--num-warps", type=int, choices=(1, 2, 4, 8), default=4)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--graph-repeats", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dump-ir", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved configurations without CUDA")
    parser.add_argument("--num-sms", type=int, default=None, help="SM count for --dry-run with auto-K only")
    parser.add_argument("--output-dir", type=Path, default=HERE.parent / "results" / "decode-joint-sweep")
    return parser


def resolve_plan(args, num_sms):
    batches = parse_axis(args.batch_sizes)
    contexts = parse_axis(args.context_lengths)
    heads = sorted(set(parse_axis(args.heads_per_program, allowed=(1, 2, 3, 6))) | {1})
    stages = sorted(set(parse_axis(args.pipeline_stages)) | {1})
    requested_k = parse_axis(args.k_splits, allow_auto=True)
    if args.page_size < 1 or args.page_size & (args.page_size - 1):
        raise ValueError("page-size must be a power of two")
    if min(args.target_occupancy, args.repetitions, args.graph_repeats) < 1 or args.warmup < 0:
        raise ValueError("target/repetitions/graph-repeats must be positive; warmup non-negative")
    if "auto" in requested_k and (num_sms is None or num_sms < 1):
        raise ValueError("auto-K dry runs require --num-sms with the target GPU's SM count")
    plan = []
    for batch in batches:
        for context in contexts:
            pages = (context + args.page_size - 1) // args.page_size
            auto = {}
            if "auto" in requested_k:
                for h in heads:
                    programs = batch * 2 * (6 // h)
                    target_k = (num_sms * args.target_occupancy + programs - 1) // programs
                    auto[h] = min(max(1, target_k), max(1, pages // 2))
            # Union auto choices across head sizes so every K has matched grouping controls.
            ks = sorted({1} | {k for k in requested_k if k != "auto"} | set(auto.values()))
            configs = [{"heads_per_program": h, "split_k": k, "num_stages": s,
                        "pipelined": s > 1, "num_warps": args.num_warps}
                       for h in heads for k in ks for s in stages]
            plan.append({"batch_size": batch, "context_length": context,
                         "num_kv_blocks": pages, "auto_k_by_head_size": auto, "configs": configs})
    return plan


def config_key(config):
    return (config["heads_per_program"], config["split_k"], config["num_stages"])


def summarize_shape(rows, production_ms):
    """Ratios use measured controls from this shape and differ in exactly one axis."""
    lookup = {config_key(row): row["median_ms"] for row in rows}
    baseline = lookup[(1, 1, 1)]
    for row in rows:
        h, k, s = config_key(row)
        time = row["median_ms"]
        row.update({
            "speedup_vs_production": production_ms / time,
            "speedup_vs_ungrouped_k1_stage1": baseline / time,
            "grouping_speedup_at_fixed_k_stages": lookup[(1, k, s)] / time,
            "splitk_speedup_at_fixed_heads_stages": lookup[(h, 1, s)] / time,
            "pipeline_speedup_at_fixed_heads_k": lookup[(h, k, 1)] / time,
        })
    best = min(rows, key=lambda row: row["median_ms"])
    ungrouped = min((r for r in rows if r["heads_per_program"] == 1), key=lambda r: r["median_ms"])
    return {
        "batch_size": best["batch_size"], "context_length": best["context_length"],
        "configurations": len(rows), "production_ms": production_ms,
        "ungrouped_k1_stage1_ms": baseline,
        "best_heads_per_program": best["heads_per_program"], "best_k": best["split_k"],
        "best_stages": best["num_stages"], "best_ms": best["median_ms"],
        "best_speedup_vs_production": production_ms / best["median_ms"],
        "best_speedup_vs_ungrouped_k1_stage1": baseline / best["median_ms"],
        "best_ungrouped_k": ungrouped["split_k"], "best_ungrouped_stages": ungrouped["num_stages"],
        "best_ungrouped_ms": ungrouped["median_ms"],
        "best_speedup_vs_best_ungrouped": ungrouped["median_ms"] / best["median_ms"],
    }


def write_results(prefix, payload):
    prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")
    for suffix, records in ((".csv", payload["rows"]), ("-summary.csv", payload["shape_summaries"])):
        if not records:
            continue
        flat = [{k: v for k, v in row.items() if k not in ("samples_ms", "kernel_resources")}
                for row in records]
        with Path(f"{prefix}{suffix}").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=flat[0].keys())
            writer.writeheader()
            writer.writerows(flat)


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        # Validate arguments before importing CUDA dependencies.
        resolve_plan(args, args.num_sms or 1)
        if args.dry_run:
            plan = resolve_plan(args, args.num_sms)
            print(json.dumps({"shapes": len(plan), "candidate_measurements": sum(len(p["configs"]) for p in plan),
                              "plan": plan}, indent=2))
            return
        if args.num_sms is not None:
            raise ValueError("--num-sms is only for --dry-run; real runs query the GPU")
    except ValueError as exc:
        parser.error(str(exc))

    import torch
    import triton
    from paged_decode_grouped_splitk_pipelined import grouped_splitk_attention, get_num_sms
    from paged_decode_attention import paged_decode_attention

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA and Triton are required")
    torch.backends.cuda.matmul.allow_tf32 = False
    num_sms = get_num_sms()
    plan = resolve_plan(args, num_sms)
    dtype = getattr(torch, args.dtype)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / ("joint-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    payload = {"schema_version": 1, "status": "running", "configuration": {**vars(args), "output_dir": str(args.output_dir)},
               "device": torch.cuda.get_device_name(), "num_sms": num_sms,
               "torch_version": torch.__version__, "triton_version": triton.__version__,
               "timing_mode": "warm_cache_cuda_graph_device_latency", "plan": plan,
               "correctness_preflight": [], "production": [], "rows": [], "shape_summaries": []}
    print(f"{len(plan)} shapes, {sum(len(p['configs']) for p in plan)} candidate measurements; "
          "includes matched hpp=1 / K=1 / stage=1 controls", flush=True)
    # Reuse independent references, and gate each unique configuration once before timing.
    cases = [(name, tensors, options, attention_reference(*tensors, **options))
             for name, tensors, options in correctness_cases(page_size=args.page_size, dtype=dtype, seed=args.seed)]
    checked = set()
    rng = random.Random(args.seed)
    current = {}
    try:
        prod_inputs = make_inputs([1, args.page_size, args.page_size + 1], page_size=args.page_size,
                                  dtype=dtype, seed=args.seed)
        check_output(paged_decode_attention(*prod_inputs), attention_reference(*prod_inputs))
        for shape in plan:
            batch, context = shape["batch_size"], shape["context_length"]
            current = {"batch_size": batch, "context_length": context, "phase": "preflight"}
            for config in shape["configs"]:
                key = config_key(config)
                if key in checked:
                    continue
                current["config"] = config
                for name, tensors, options, reference in cases:
                    error = check_output(grouped_splitk_attention(*tensors, **config, **options), reference)
                    payload["correctness_preflight"].append({"config": config, "case": name, "max_abs_error": error})
                checked.add(key)
                print(f"Preflight passed: H={key[0]} K={key[1]} stages={key[2]}", flush=True)
            tensors = make_inputs([context] * batch, page_size=args.page_size, dtype=dtype, seed=args.seed)
            expected = paged_decode_attention(*tensors)
            order = [None, *shape["configs"]]
            rng.shuffle(order)
            rows = []
            payload["incomplete_shape"] = {"batch_size": batch, "context_length": context, "rows": rows}
            production_ms = None
            for index, config in enumerate(order):
                current = {"batch_size": batch, "context_length": context, "phase": "measurement", "config": config}
                if config is None:
                    timing = benchmark_kernel(partial(paged_decode_attention, *tensors), args.warmup,
                                              args.repetitions, args.graph_repeats)
                    production_ms = timing["median_ms"]
                    payload["production"].append({"batch_size": batch, "context_length": context,
                                                  "measurement_order": index, **timing})
                    continue
                compiled = {}
                actual = grouped_splitk_attention(*tensors, **config, diagnostics=compiled)
                error = check_output(actual, expected)
                h, k, s = config_key(config)
                resources = kernel_diagnostics(compiled, f"{prefix}-B{batch}-L{context}-H{h}-K{k}-S{s}" if args.dump_ir else None)
                timing = benchmark_kernel(partial(grouped_splitk_attention, *tensors, **config), args.warmup,
                                          args.repetitions, args.graph_repeats)
                programs = batch * 2 * (6 // h)
                row = {"batch_size": batch, "context_length": context, **config, **timing,
                       "measurement_order": index, "is_auto_k": shape["auto_k_by_head_size"].get(h) == k,
                       "num_kv_blocks": shape["num_kv_blocks"], "launched_programs": programs * k,
                       "active_programs": programs * min(k, shape["num_kv_blocks"]),
                       "launched_programs_per_sm": programs * k / num_sms,
                       "min_pages_per_split": shape["num_kv_blocks"] // k,
                       "max_pages_per_split": (shape["num_kv_blocks"] + k - 1) // k,
                       "max_abs_error": error, "kernel_resources": resources}
                rows.append(row)
                print(f"B={batch:3d} L={context:5d} H={h} K={k:3d} stages={s} "
                      f"{timing['median_ms']:.6f} ms [{index + 1}/{len(order)}]", flush=True)
            summary = summarize_shape(rows, production_ms)
            payload["rows"].extend(rows)
            payload["shape_summaries"].append(summary)
            payload.pop("incomplete_shape")
            write_results(prefix, payload)
            print(f"Best: H={summary['best_heads_per_program']} K={summary['best_k']} "
                  f"stages={summary['best_stages']}: {summary['best_speedup_vs_production']:.2f}x production; "
                  f"{summary['best_speedup_vs_best_ungrouped']:.2f}x best ungrouped", flush=True)
        payload["status"] = "complete"
    except Exception as exc:
        payload["status"] = "failed"
        payload["failure"] = {**current, "error": f"{type(exc).__name__}: {exc}"}
        raise
    finally:
        write_results(prefix, payload)
        print(f"Results: {prefix}.json, .csv, -summary.csv", flush=True)


if __name__ == "__main__":
    main()
