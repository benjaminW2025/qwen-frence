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
from design import make_plan, make_requests
from model_adapter import GraphModelAdapter, PiecewiseGraphModelAdapter, allocate_pool


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("smoke", "full", "longctx"), default="smoke")
    parser.add_argument("--case-id", default="uniform-b4-l513")
    parser.add_argument("--kind", choices=("decode", "prefill", "mixed"), default="decode")
    parser.add_argument("--occurrence", type=int, default=0,
                        help="zero-based occurrence of the selected step kind")
    parser.add_argument("--adapter", choices=("eager-prefill", "piecewise-prefill"),
                        default="piecewise-prefill")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--with-stack", action="store_true")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "experiments/results/cpp-control-profile")
    return parser


def validate_args(args):
    if args.warmups < 0 or args.occurrence < 0 or args.repetitions < 1:
        raise ValueError("warmups/occurrence must be nonnegative and repetitions positive")
    if args.case_id not in {case["id"] for case in make_plan(args.preset)}:
        raise ValueError("case ID is not in the selected preset")


def select_target(steps, kind, occurrence):
    matches = [i for i, step in enumerate(steps) if step["kind"] == kind]
    if occurrence >= len(matches):
        raise ValueError(f"case has {len(matches)} {kind} steps, not occurrence {occurrence}")
    return matches[occurrence]


class ProfileCallback:
    def __init__(self, adapter):
        self.adapter = adapter

    def __call__(self, *args):
        from torch.profiler import record_function
        name = "python/model_callback_decode" if args[-1] else "python/model_callback_prefill"
        with record_function(name):
            return self.adapter(*args)


def drive(torch, cpp, config, requests, adapter, target_index, *, profile_target=False,
          with_stack=False):
    """Drain the same arrival-index workload, profiling only one chosen step."""
    from torch.profiler import ProfilerActivity, profile, record_function

    loop = cpp.IterationLoop(config, torch.device(adapter.pool.k_pool[0].device))
    pending = sorted(requests, key=lambda request: (request["arrival"], request["id"]))
    cursor = iteration = 0
    mapping, outputs, steps = {}, {}, []
    iteration_limit = (sum(len(r["prompt"]) + r["output"] for r in requests)
                       + max(r["arrival"] for r in requests) + 1)
    target_ms = None
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
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                             record_shapes=False, profile_memory=False,
                             with_stack=with_stack) as prof:
                    with record_function("experiment/target_step"):
                        completed = loop.step(ProfileCallback(adapter))
                    prof.step()
            else:
                completed = loop.step(adapter)
            target_ms = (time.perf_counter() - start) * 1000
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
    return {"target_wall_ms": target_ms, "steps": steps, "outputs": outputs, "profiler": prof}


def stage_summary(prof):
    rows = []
    for event in prof.key_averages():
        if event.key.startswith(("cpp/", "python/", "experiment/")):
            rows.append({"name": event.key, "calls": event.count,
                         "cpu_total_us": event.cpu_time_total,
                         "cpu_self_us": event.self_cpu_time_total})
    return sorted(rows, key=lambda row: row["cpu_total_us"], reverse=True)


def main():
    args = build_parser().parse_args()
    validate_args(args)
    import torch
    if not torch.cuda.is_available() or torch.device(args.device).type != "cuda":
        raise RuntimeError("this experiment requires CUDA")
    import inference_engine_cpp as cpp
    from run_benchmarks import system_metadata
    from run_phase_sweep import make_engine

    extension = Path(cpp.__file__)
    sources = list((ROOT / "engine/cpp/src").glob("*.cpp"))
    if any(source.stat().st_mtime_ns > extension.stat().st_mtime_ns for source in sources):
        raise RuntimeError("C++ extension is older than source; run make cpp-scheduler-build")
    case = next(case for case in make_plan(args.preset) if case["id"] == args.case_id)
    config = make_config(cpp, case)
    blocks = config.max_batch_size * ((config.max_context_length + 15) // 16)
    engine, load_seconds = make_engine(args.model, "custom-kernels", args.device,
                                       "float16", 16)
    pool = allocate_pool(engine.cfg, blocks, engine.device)
    adapter_cls = (PiecewiseGraphModelAdapter if args.adapter == "piecewise-prefill"
                   else GraphModelAdapter)
    adapter = adapter_cls(engine.model, pool, None, max_running=config.max_batch_size,
                          max_context_length=config.max_context_length)
    requests = make_requests(case, args.seed, engine.cfg.vocab)

    def poison():
        for tensor in pool.k_pool + pool.v_pool:
            tensor.fill_(float("nan"))

    poison()
    preflight = execute(torch, cpp.IterationLoop(config, torch.device(args.device)),
                        adapter, requests)
    expected_steps = [(step["kind"], step["calls"], step["completed"])
                      for step in preflight["steps"]]
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
    table_path.write_text(prof.key_averages().table(sort_by="self_cpu_time_total",
                                                     row_limit=80) + "\n")
    report = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "case": case, "kind": args.kind, "adapter": args.adapter,
        "target_step_index": target_index,
        "target_step_calls": expected_steps[target_index][1],
        "unprofiled_target_wall_ms": baseline_ms,
        "unprofiled_median_wall_ms": statistics.median(baseline_ms),
        "profiled_target_wall_ms": traced["target_wall_ms"],
        "cpu_ranges": stage_summary(prof),
        "system": system_metadata(), "model_load_seconds": load_seconds,
        "trace": str(trace_path), "operators": str(table_path),
    }
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"{args.kind} step {target_index}: unprofiled median "
          f"{report['unprofiled_median_wall_ms']:.3f} ms; trace {trace_path}")


if __name__ == "__main__":
    main()
