#!/usr/bin/env python3
"""Focused CPU/GPU trace of one warmed C++ scheduler + model step.

An uninstrumented preflight chooses a deterministic decode, prefill, or mixed
step. Separate unprofiled full-workload runs measure its wall time. One final
run profiles only that step; the rest of the workload is outside the trace.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for directory in (HERE, ROOT / "benchmarks", ROOT / "baseline", ROOT / "engine/kvcache",
                  ROOT / "engine/model_runner", ROOT / "engine/cpp/build"):
    sys.path.insert(0, str(directory))

from benchmark_scheduler_decode import execute, make_config
from benchmark_integrated_graph import resolve_requests, verify_actual_work
from benchmark_latest_vs_vllm import resolve_model_source
from design import make_plan, make_requests
from fixed_regime import PREFILL_TOKENS_PER_STEP, verify_fixed_result
from model_adapter import GraphModelAdapter, PiecewiseGraphModelAdapter, allocate_pool
from model_setup import check_startup, load_model_only


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("smoke", "full", "longctx", "fixed"), default="smoke")
    parser.add_argument("--case-id", default="uniform-b4-l513")
    parser.add_argument("--kind", choices=("decode", "prefill", "mixed"), default="decode")
    parser.add_argument("--occurrence", type=int, default=0,
                        help="zero-based occurrence; fixed decode/prefill selects only "
                             "full-batch/full-budget steps")
    parser.add_argument("--adapter", choices=("eager-prefill", "piecewise-prefill"),
                        default="piecewise-prefill")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--decode-attention-policy", choices=("production", "splitk", "fa3"),
                        default="production")
    parser.add_argument("--qkv-mode",
                        choices=("none", "native-k", "native", "packed", "full"),
                        default="none", help="decode QKV/RoPE/KV-write fusion arm")
    parser.add_argument("--enable-residual-rmsnorm", action="store_true",
                        help="enable the fused residual-add plus RMSNorm decode path")
    parser.add_argument("--prefill-budget", type=int,
                        help="override the case budget and matching piecewise graph bucket")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--workload-in", type=Path,
                        help="use exact saved prompt IDs/output lengths from the fixed checkpoint")
    parser.add_argument("--with-stack", action="store_true")
    parser.add_argument("--check-setup", action="store_true",
                        help="check runtime dependencies before loading weights or "
                             "allocating model GPU memory")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "experiments/results/cpp-control-profile")
    return parser


def validate_args(args):
    if args.warmups < 0 or args.occurrence < 0 or args.repetitions < 1:
        raise ValueError("warmups/occurrence must be nonnegative and repetitions positive")
    if args.case_id not in {case["id"] for case in make_plan(args.preset)}:
        raise ValueError("case ID is not in the selected preset")
    if args.prefill_budget is not None and args.prefill_budget < 1:
        raise ValueError("prefill-budget must be positive")


def select_target(steps, kind, occurrence):
    matches = [i for i, step in enumerate(steps) if step["kind"] == kind]
    if occurrence >= len(matches):
        raise ValueError(f"case has {len(matches)} {kind} steps, not occurrence {occurrence}")
    return matches[occurrence]


def select_fixed_target(steps, case, kind, occurrence,
                        prefill_tokens=PREFILL_TOKENS_PER_STEP):
    if kind == "decode":
        matches = [i for i, step in enumerate(steps)
                   if step["kind"] == "decode" and
                   any(call[0] and call[1] == case["max_running"] for call in step["calls"])]
    elif kind == "prefill":
        matches = [i for i, step in enumerate(steps)
                   if step["kind"] == "prefill" and
                   any(not call[0] and call[1] == prefill_tokens
                       for call in step["calls"])]
    else:
        return select_target(steps, kind, occurrence)
    if occurrence >= len(matches):
        raise ValueError(f"fixed case has {len(matches)} eligible {kind} steps, "
                         f"not occurrence {occurrence}")
    return matches[occurrence]


class ProfileCallback:
    def __init__(self, adapter):
        self.adapter = adapter
        self.calls = []

    def __call__(self, *args):
        from torch.profiler import record_function
        # Keep a reference and materialize it only after profiling has stopped;
        # a device-to-host copy here would contaminate the target step trace.
        self.calls.append({
            "decode": bool(args[-1]),
            "tokens": int(args[0].numel()),
            "context": args[4],
        })
        name = "python/model_callback_decode" if args[-1] else "python/model_callback_prefill"
        with record_function(name):
            return self.adapter(*args)

    def materialize(self):
        return [{"decode": row["decode"], "tokens": row["tokens"],
                 "context_lengths": [int(value) for value in
                                     row["context"].cpu().tolist()]}
                for row in self.calls]


def drive(torch, cpp, config, requests, adapter, target_index, *, profile_target=False,
          with_stack=False, collect_target_metadata=False):
    """Drain the same arrival-index workload, profiling only one chosen step."""
    from torch.profiler import ProfilerActivity, profile, record_function

    loop = cpp.IterationLoop(config, torch.device(adapter.pool.k_pool[0].device))
    pending = sorted(requests, key=lambda request: (request["arrival"], request["id"]))
    cursor = iteration = 0
    mapping, outputs, steps = {}, {}, []
    iteration_limit = (sum(len(r["prompt"]) + r["output"] for r in requests)
                       + max(r["arrival"] for r in requests) + 1)
    target_ms = None
    target_calls = None
    prof = None
    torch.cuda.synchronize()
    while cursor < len(pending) or loop.num_pending() or loop.num_running():
        if not loop.num_pending() and not loop.num_running() and cursor < len(pending):
            iteration = max(iteration, pending[cursor]["arrival"])
        while cursor < len(pending) and pending[cursor]["arrival"] <= iteration:
            request = pending[cursor]
            mapping[loop.submit_request(request["prompt"], request["output"])] = request["id"]
            cursor += 1
        adapter.step_calls = []
        is_target = len(steps) == target_index
        if is_target:
            start = time.perf_counter()
            if profile_target:
                callback = ProfileCallback(adapter)
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                             record_shapes=False, profile_memory=False,
                             with_stack=with_stack) as prof:
                    with record_function("experiment/target_step"):
                        completed = loop.step(callback)
                    prof.step()
                # Record latency before copying metadata back to the host. The
                # C++ loop reuses this device buffer on the following step.
                target_ms = (time.perf_counter() - start) * 1000
                target_calls = callback.materialize()
            else:
                callback = ProfileCallback(adapter) if collect_target_metadata else adapter
                completed = loop.step(callback)
                target_ms = (time.perf_counter() - start) * 1000
                if collect_target_metadata:
                    target_calls = callback.materialize()
        else:
            completed = loop.step(adapter)
        if not adapter.step_calls:
            raise RuntimeError("scheduler made no progress")
        kinds = {call[0] for call in adapter.step_calls}
        kind = "mixed" if len(kinds) == 2 else "decode" if True in kinds else "prefill"
        steps.append((kind, list(adapter.step_calls), completed))
        for request_id, tokens in loop.pop_completed():
            outputs[mapping[request_id]] = tokens
        iteration += 1
        if iteration > iteration_limit:
            raise RuntimeError("iteration limit exceeded")
    torch.cuda.synchronize()
    if target_ms is None:
        raise AssertionError("selected target step was not reached")
    return {"target_wall_ms": target_ms, "steps": steps, "outputs": outputs,
            "profiler": prof, "target_calls": target_calls}


def stage_summary(prof):
    rows = []
    for event in prof.key_averages():
        if event.key.startswith(("cpp/", "python/", "experiment/")):
            rows.append({"name": event.key, "calls": event.count,
                         "cpu_total_us": event.cpu_time_total,
                         "cpu_self_us": event.self_cpu_time_total})
    return sorted(rows, key=lambda row: row["cpu_total_us"], reverse=True)


def cuda_kernel_category(name):
    """Classify one raw CUDA activity without double-counting parent ranges."""
    lower = name.lower()
    if "memcpy" in lower or "memset" in lower:
        return "memory_copy_or_set"
    if ("index_elementwise" in lower or "scatter" in lower
            or "reshape_and_cache" in lower or "concat_and_cache" in lower):
        return "kv_write"
    if any(token in lower for token in
           ("attention", "grouped_gqa", "grouped_splitk", "splitk_reduce",
            "flashattn", "flash_attn", "fmha", "paged_attention")):
        return "attention"
    if "rope" in lower:
        return "rope"
    if "rms" in lower or "norm" in lower:
        return "normalization"
    if any(token in lower for token in ("swiglu", "silu", "sigmoid")):
        return "activation"
    if any(token in lower for token in
           ("nvjet", "gemm", "cublas", "cutlass", "matmul")):
        return "gemm"
    if "argmax" in lower or "reduce_kernel" in lower:
        return "sampling"
    if "elementwise" in lower:
        return "elementwise"
    return "other"


def cuda_activity_summary(events):
    """Aggregate leaf CUDA activities by semantic category and exact kernel name."""
    kernels = {}
    for event in events:
        if "cuda" not in str(getattr(event, "device_type", "")).lower():
            continue
        duration = float(getattr(event, "self_device_time_total", 0.0))
        if duration <= 0:
            duration = float(getattr(event, "device_time_total", 0.0))
        if duration <= 0:
            continue
        name = str(event.name)
        row = kernels.setdefault(name, {"name": name, "category": cuda_kernel_category(name),
                                        "calls": 0, "total_us": 0.0})
        row["calls"] += 1
        row["total_us"] += duration
    kernel_rows = sorted(kernels.values(), key=lambda row: row["total_us"], reverse=True)
    total = sum(row["total_us"] for row in kernel_rows)
    categories = {}
    for row in kernel_rows:
        row["mean_us"] = row["total_us"] / row["calls"]
        row["percent_of_cuda_activity"] = 100 * row["total_us"] / total if total else 0.0
        category = categories.setdefault(row["category"], {
            "category": row["category"], "calls": 0, "total_us": 0.0})
        category["calls"] += row["calls"]
        category["total_us"] += row["total_us"]
    category_rows = sorted(categories.values(), key=lambda row: row["total_us"], reverse=True)
    for row in category_rows:
        row["percent_of_cuda_activity"] = 100 * row["total_us"] / total if total else 0.0
    return {"summed_cuda_activity_us": total,
            "activity_count": sum(row["calls"] for row in kernel_rows),
            "categories": category_rows, "kernels": kernel_rows}


def main():
    args = build_parser().parse_args()
    validate_args(args)
    if args.check_setup:
        print(json.dumps(check_startup(args.device), indent=2))
        return
    startup = check_startup(args.device)
    import torch
    import inference_engine_cpp as cpp
    from run_benchmarks import system_metadata

    case = next(case for case in make_plan(args.preset) if case["id"] == args.case_id)
    if args.prefill_budget is not None:
        case = {**case, "prefill_budget": args.prefill_budget}
    config = make_config(cpp, case)
    blocks = config.max_batch_size * (((config.max_context_length + 15) // 16) + 1)
    model_source = resolve_model_source(args)
    engine, load_seconds, hub_transfer = load_model_only(
        model_source, args.device, "float16", hub_transfer=startup["hub_transfer"])
    pool = allocate_pool(engine.cfg, blocks, engine.device)
    adapter_cls = (PiecewiseGraphModelAdapter if args.adapter == "piecewise-prefill"
                   else GraphModelAdapter)
    adapter_options = dict(max_running=config.max_batch_size,
                           max_context_length=config.max_context_length,
                           decode_attention_policy=args.decode_attention_policy,
                           enable_residual_rmsnorm=args.enable_residual_rmsnorm,
                           enable_native_decode_rope_kv=args.qkv_mode == "native-k",
                           enable_native_decode_qkv_postprocess=args.qkv_mode == "native",
                           enable_packed_qkv_rope_cache=args.qkv_mode == "packed",
                           enable_fused_qkv_rope_cache=args.qkv_mode == "full")
    if adapter_cls is PiecewiseGraphModelAdapter and args.prefill_budget is not None:
        adapter_options.update(max_capture_tokens=args.prefill_budget,
                               max_prefill_shapes=1,
                               prefill_buckets=[args.prefill_budget])
    adapter = adapter_cls(engine.model, pool, None, **adapter_options)
    requests = (resolve_requests(args, case) if args.workload_in is not None
                else make_requests(case, args.seed, engine.cfg.vocab))

    def poison():
        for tensor in pool.k_pool + pool.v_pool:
            tensor.fill_(float("nan"))

    poison()
    preflight = execute(torch, cpp.IterationLoop(config, torch.device(args.device)),
                        adapter, requests)
    expected_steps = [(step["kind"], step["calls"], step["completed"])
                      for step in preflight["steps"]]
    if args.preset == "fixed":
        if args.prefill_budget is None:
            verify_fixed_result(preflight, args.case_id)
        else:
            verify_actual_work(preflight, case["max_running"], args.prefill_budget, 1)
        target_index = select_fixed_target(preflight["steps"], case, args.kind,
                                           args.occurrence,
                                           args.prefill_budget or PREFILL_TOKENS_PER_STEP)
    else:
        target_index = select_target(preflight["steps"], args.kind, args.occurrence)

    def checked_run(*, trace=False):
        poison()
        result = drive(torch, cpp, config, requests, adapter, target_index,
                       profile_target=trace, with_stack=args.with_stack)
        if result["outputs"] != preflight["outputs"] or result["steps"] != expected_steps:
            raise AssertionError("profiled workload changed outputs or scheduled work")
        return result

    for _ in range(args.warmups):
        checked_run()
    baseline_ms = [checked_run()["target_wall_ms"] for _ in range(args.repetitions)]
    traced = checked_run(trace=True)
    prof = traced["profiler"]
    if not any(row["name"] == "cpp/step" for row in stage_summary(prof)):
        raise RuntimeError("C++ profiler ranges missing; rebuild the extension")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    stem = f"{case['id']}-{args.kind}-{args.adapter}-{stamp}"
    trace_path = args.output_dir / f"{stem}-trace.json"
    report_path = args.output_dir / f"{stem}-report.json"
    table_path = args.output_dir / f"{stem}-operators.txt"
    prof.export_chrome_trace(str(trace_path))
    table_path.write_text(prof.key_averages().table(sort_by="self_cuda_time_total",
                                                     row_limit=80) + "\n")
    cuda = cuda_activity_summary(prof.events())
    report = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "case": case, "kind": args.kind, "occurrence": args.occurrence,
        "adapter": args.adapter, "model": model_source,
        "decode_attention_policy": args.decode_attention_policy,
        "decode_attention_action": adapter.action,
        "qkv_mode": args.qkv_mode,
        "enable_residual_rmsnorm": args.enable_residual_rmsnorm,
        "prefill_budget": case["prefill_budget"],
        "target_step_index": target_index,
        "target_step_calls": expected_steps[target_index][1],
        "target_callback_calls": traced["target_calls"],
        "unprofiled_target_wall_ms": baseline_ms,
        "unprofiled_median_wall_ms": statistics.median(baseline_ms),
        "profiled_target_wall_ms": traced["target_wall_ms"],
        "cpu_ranges": stage_summary(prof),
        "cuda_activity": cuda,
        "system": system_metadata(), "model_load_seconds": load_seconds,
        "hub_transfer": hub_transfer,
        "trace": str(trace_path), "operators": str(table_path),
    }
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"{args.kind} step {target_index}: unprofiled median "
          f"{report['unprofiled_median_wall_ms']:.3f} ms; trace {trace_path}")
    print("CUDA activity by category:")
    for row in cuda["categories"]:
        print(f"  {row['category']:<20} {row['total_us']:>10.1f} us "
              f"{row['percent_of_cuda_activity']:>6.2f}%  ({row['calls']} activities)")


if __name__ == "__main__":
    main()
