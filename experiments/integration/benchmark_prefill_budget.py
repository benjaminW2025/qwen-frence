#!/usr/bin/env python3
"""Sweep packed-prefill token budgets with one model load and matched execution.

This isolates the effect of doing fewer, larger packed prefill calls.  Every
budget uses the same requests, weights, production decode graph, and packed
prefill implementation.  The piecewise prefill graph is replaced between
budgets so only one potentially large token bucket remains live at a time.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for directory in (HERE, ROOT / "benchmarks", ROOT / "baseline", ROOT / "engine/kvcache",
                  ROOT / "engine/model_runner", ROOT / "engine/cpp/build"):
    sys.path.insert(0, str(directory))

from benchmark_integrated_graph import (SameHistoryCheck, atomic_json, requests_fingerprint,
                                        schedule_of, verify_actual_work)
from benchmark_latest_vs_vllm import MODEL, resolve_model_source
from benchmark_scheduler_decode import TraceCheck, execute, independent_check, make_config
from design import make_requests
from fixed_regime import FIXED_SHAPES, get_fixed_case
from model_adapter import ModelAdapter, PiecewiseGraphModelAdapter, allocate_pool
from model_setup import check_startup, load_model_only


DEFAULT_BUDGETS = (2048, 4096, 8192)
PHASES = ("wall_ms", "prefill_wall_ms", "mixed_wall_ms", "prefill_plus_mixed_ms",
          "decode_wall_ms", "output_tokens_per_s")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape-id", choices=tuple(row["id"] for row in FIXED_SHAPES),
                        default="fixed-b8-l2048-o128")
    parser.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS),
                        help="packed tokens per prefill iteration; use 16384 explicitly "
                             "for the full B8 cohort")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--logit-atol", type=float, default=.05)
    parser.add_argument("--workload-in", type=Path,
                        help="optional benchmark_core workload.json with the exact "
                             "table-row requests")
    parser.add_argument("--plan", action="store_true", help="print the validated GPU work plan")
    parser.add_argument("--dry-schedule", action="store_true",
                        help="run every budget through the C++ scheduler on CPU")
    parser.add_argument("--check-setup", action="store_true",
                        help="check CUDA/Triton/C++ setup without loading the model")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "experiments/results/prefill-budget")
    return parser


def validate_args(args):
    if len(set(args.budgets)) != len(args.budgets) or any(value < 1 for value in args.budgets):
        raise ValueError("budgets must be distinct positive integers")
    if args.budgets != sorted(args.budgets):
        raise ValueError("budgets must be in increasing order")
    if min(args.trials, args.samples) < 1 or args.warmups < 0:
        raise ValueError("trials and samples must be positive; warmups must be nonnegative")
    if not 0 <= args.logit_atol < float("inf"):
        raise ValueError("logit-atol must be finite and nonnegative")
    base = get_fixed_case(args.shape_id)
    total_prompt_tokens = sum(base["lengths"])
    if args.budgets[-1] > total_prompt_tokens:
        raise ValueError(f"largest budget exceeds the cohort's {total_prompt_tokens} prompt tokens")
    if any(total_prompt_tokens % value for value in args.budgets):
        raise ValueError("every budget must divide total prompt tokens so the sweep "
                         "has no tail bucket")
    return base


def case_for_budget(base, budget):
    return {**base, "id": f"{base['id']}-budget{budget}", "prefill_budget": budget}


def resolve_requests(args, base):
    if args.workload_in is None:
        return make_requests(base, args.seed, 151936)
    from benchmark_core import Workload

    workload = Workload.from_dict(json.loads(args.workload_in.read_text()))
    if workload.seed != args.seed or workload.has_staggered_arrivals:
        raise ValueError("workload seed/arrival pattern differs from the budget sweep")
    if len(workload.requests) != len(base["lengths"]):
        raise ValueError("workload request count differs from the selected shape")
    requests = []
    for index, (request, length, output) in enumerate(
            zip(workload.requests, base["lengths"], base["outputs"])):
        if (request.request_id != str(index) or len(request.prompt_ids) != length
                or request.max_tokens != output):
            raise ValueError(f"workload request {index} differs from the selected shape")
        requests.append({"id": index, "prompt": request.prompt_ids,
                         "output": request.max_tokens, "arrival": 0})
    return requests


def call_summary(result, budget):
    calls = [call for step in result["steps"] for call in step["calls"] if not call[0]]
    tokens = [call[1] for call in calls]
    sequences = [call[2] for call in calls]
    if not calls:
        raise AssertionError("no prefill calls were observed")
    return {
        "prefill_calls": len(calls),
        "packed_token_counts": tokens,
        "sequences_per_call": sequences,
        "mean_packed_tokens": statistics.mean(tokens),
        "mean_sequences_per_call": statistics.mean(sequences),
        "token_budget_utilization": sum(tokens) / (len(tokens) * budget),
    }


def measured_metrics(result):
    return {
        "wall_ms": result["wall_ms"],
        "prefill_wall_ms": result["prefill_wall_ms"],
        "mixed_wall_ms": result["mixed_wall_ms"],
        "prefill_plus_mixed_ms": result["prefill_wall_ms"] + result["mixed_wall_ms"],
        "decode_wall_ms": result["decode_wall_ms"],
        "output_tokens_per_s": result["output_tokens_per_s"],
    }


def aggregate(rows):
    baseline = rows[0]
    base_medians = baseline["medians"]
    for row in rows:
        medians = row["medians"]
        row["relative_to_smallest_budget"] = {
            "end_to_end_speedup": base_medians["wall_ms"] / medians["wall_ms"],
            "prefill_plus_mixed_speedup": (
                base_medians["prefill_plus_mixed_ms"] / medians["prefill_plus_mixed_ms"]),
            "output_throughput_ratio": (
                medians["output_tokens_per_s"] / base_medians["output_tokens_per_s"]),
            "prefill_call_reduction": (
                baseline["work"]["prefill_calls"] / row["work"]["prefill_calls"]),
        }
    return rows


def plan_payload(args, base, requests):
    prompt_tokens = sum(len(row["prompt"]) for row in requests)
    output_tokens = sum(row["output"] for row in requests)
    return {
        "shape_id": args.shape_id,
        "batch": base["max_running"],
        "prompt_tokens_per_request": base["lengths"][0],
        "output_tokens_per_request": base["outputs"][0],
        "cohort_prompt_tokens": prompt_tokens,
        "cohort_output_tokens": output_tokens,
        "budgets": args.budgets,
        "expected_prefill_calls": {str(value): prompt_tokens // value for value in args.budgets},
        "graphs_per_budget": 29,
        "timed_workloads_per_budget": args.trials * args.samples,
        "warmup_workloads_per_budget": args.warmups,
        "correctness_workloads_per_budget": 2,
        "requests_sha256": requests_fingerprint(requests),
    }


def run_budget(torch, cpp, engine, base, requests, args, budget, candidate, eager, pool,
               reference_pool):
    from piecewise_prefill import PiecewisePrefill

    case = case_for_budget(base, budget)
    config = make_config(cpp, case)
    if hasattr(candidate, "piecewise_prefill"):
        del candidate.piecewise_prefill
        gc.collect()
        torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(engine.device)
    candidate.piecewise_prefill = PiecewisePrefill(
        engine.model, pool, max_capture_tokens=budget, max_shapes=1,
        token_buckets=[budget])

    def execute_arm(adapter, target_pool, observer=None):
        for tensor in target_pool.k_pool + target_pool.v_pool:
            tensor.fill_(float("nan"))
        adapter.observer = observer
        loop = cpp.IterationLoop(config, torch.device(engine.device))
        try:
            return execute(torch, loop, adapter, requests)
        finally:
            adapter.observer = None

    print(f"budget={budget}: eager reference", flush=True)
    trace = TraceCheck(torch)
    eager_result = execute_arm(eager, reference_pool, trace)
    trace.finish()
    eager_work = verify_actual_work(eager_result, base["max_running"], budget, 1)

    print(f"budget={budget}: piecewise same-history correctness", flush=True)
    for tensor in reference_pool.k_pool + reference_pool.v_pool:
        tensor.fill_(float("nan"))
    checker = SameHistoryCheck(torch, eager, trace.rows, f"budget-{budget}",
                               atol=args.logit_atol, report_only=True)
    checked = execute_arm(candidate, pool, checker)
    checker.finish()
    work = verify_actual_work(checked, base["max_running"], budget, 1)
    if work != eager_work or schedule_of(checked) != schedule_of(eager_result):
        raise AssertionError(f"budget={budget}: eager and piecewise schedules differ")
    if checked["outputs"] != eager_result["outputs"]:
        raise AssertionError(f"budget={budget}: same-history output tokens differ")

    piece = candidate.piecewise_prefill
    if sorted(piece.shapes) != [budget]:
        raise AssertionError(f"budget={budget}: expected graph bucket was not captured")
    free = execute_arm(candidate, pool)
    expected_schedule = schedule_of(free)
    expected_outputs = free["outputs"]
    if expected_schedule != schedule_of(eager_result):
        raise AssertionError(f"budget={budget}: free-generation schedule changed")
    for _ in range(args.warmups):
        warm = execute_arm(candidate, pool)
        if schedule_of(warm) != expected_schedule or warm["outputs"] != expected_outputs:
            raise AssertionError(f"budget={budget}: warmup changed tokens or schedule")

    before = (piece.captured_calls, piece.eager_calls, piece.graph_replays)
    measurements = []
    for trial in range(args.trials):
        samples = []
        for _ in range(args.samples):
            result = execute_arm(candidate, pool)
            verify_actual_work(result, base["max_running"], budget, 1)
            if schedule_of(result) != expected_schedule or result["outputs"] != expected_outputs:
                raise AssertionError(f"budget={budget}: timed run changed tokens or schedule")
            samples.append(measured_metrics(result))
        measurements.append(samples)

    medians = {phase: statistics.median(
        statistics.median(sample[phase] for sample in trial) for trial in measurements)
        for phase in PHASES}
    timed_calls = piece.captured_calls - before[0]
    eager_fallbacks = piece.eager_calls - before[1]
    graph_replays = piece.graph_replays - before[2]
    if eager_fallbacks or timed_calls == 0:
        raise AssertionError(f"budget={budget}: timing did not stay on captured prefill")
    return {
        "budget": budget,
        "work": call_summary(free, budget),
        "actual_work": work,
        "correctness": {
            "max_logit_error": checker.max_logit_error,
            "logits_compared": checker.logits_compared,
            "logits_outside_tolerance": checker.logits_outside_tolerance,
            "callbacks_outside_tolerance": checker.callbacks_outside_tolerance,
            "argmax_differences_on_reference_history": checker.argmax_differences,
            "first_tolerance_failure": checker.first_tolerance_failure,
            "first_argmax_difference": checker.first_argmax_difference,
        },
        "capture": {
            "captured_token_bucket": budget,
            "graphs": len(engine.model.layers) + 1,
            "timed_prefill_calls": timed_calls,
            "timed_eager_fallback_calls": eager_fallbacks,
            "timed_graph_replays": graph_replays,
        },
        "gpu_memory": {
            "allocated_gib": torch.cuda.memory_allocated(engine.device) / 2**30,
            "reserved_gib": torch.cuda.memory_reserved(engine.device) / 2**30,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(engine.device) / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(engine.device) / 2**30,
        },
        "measurements": measurements,
        "medians": medians,
    }


def main():
    args = build_parser().parse_args()
    try:
        base = validate_args(args)
        requests = resolve_requests(args, base)
    except (OSError, ValueError) as error:
        raise SystemExit(f"invalid experiment plan: {error}") from error
    plan = plan_payload(args, base, requests)
    if args.plan:
        print(json.dumps(plan, indent=2))
        return
    if args.check_setup:
        print(json.dumps(check_startup(args.device), indent=2))
        return

    import torch
    import inference_engine_cpp as cpp
    from benchmark_integrated_graph import dry_schedule

    dry = {}
    for budget in args.budgets:
        case = case_for_budget(base, budget)
        result = dry_schedule(torch, cpp, case, args.seed, requests)
        verify_actual_work(result, base["max_running"], budget, 1)
        dry[str(budget)] = call_summary(result, budget)
    print("CPU scheduler budget gates passed: " + json.dumps(dry), flush=True)
    if args.dry_schedule:
        return
    if any((args.output_dir / name).exists() for name in ("manifest.json", "report.json")):
        raise SystemExit(f"refusing to overwrite results in {args.output_dir}")

    startup = check_startup(args.device)
    model_source = resolve_model_source(args)
    from run_benchmarks import system_metadata

    engine, load_seconds, hub_transfer = load_model_only(
        model_source, args.device, "float16", hub_transfer=startup["hub_transfer"])
    independent_check(torch, engine.model, args.device)
    max_context = max(n + output for n, output in zip(base["lengths"], base["outputs"]))
    blocks = base["max_running"] * (((max_context + 15) // 16) + 1)
    pool = allocate_pool(engine.cfg, blocks, engine.device)
    reference_pool = allocate_pool(engine.cfg, blocks, engine.device)
    common = {"max_running": base["max_running"], "max_context_length": max_context}
    candidate = PiecewiseGraphModelAdapter(
        engine.model, pool, None, **common, max_capture_tokens=args.budgets[0],
        max_prefill_shapes=1, prefill_buckets=[args.budgets[0]])
    eager = ModelAdapter(engine.model, reference_pool, None)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model_source,
        "device": args.device,
        "plan": plan,
        "cpu_dry_schedule": dry,
        "model_load_seconds": load_seconds,
        "hub_transfer": hub_transfer,
        "system": system_metadata(),
        "scope": ("piecewise production path; one model/decode graph; one prefill "
                  "bucket live at a time"),
    }
    atomic_json(args.output_dir / "manifest.json", manifest)
    rows = []
    try:
        for budget in args.budgets:
            rows.append(run_budget(torch, cpp, engine, base, requests, args, budget,
                                   candidate, eager, pool, reference_pool))
            atomic_json(args.output_dir / "report.json", {
                "status": "running", "shape_id": args.shape_id, "rows": aggregate(rows)})
    except Exception as error:
        atomic_json(args.output_dir / "report.json", {
            "status": "error", "shape_id": args.shape_id, "rows": aggregate(rows),
            "error": repr(error)})
        raise
    rows = aggregate(rows)
    atomic_json(args.output_dir / "report.json", {
        "status": "ok", "created_at": datetime.now(timezone.utc).isoformat(),
        "shape_id": args.shape_id, "rows": rows})
    print("\nbudget  calls  seq/call  prefill+mixed ms  wall ms  output tok/s  vs base")
    for row in rows:
        print(f"{row['budget']:>6} {row['work']['prefill_calls']:>6} "
              f"{row['work']['mean_sequences_per_call']:>9.2f} "
              f"{row['medians']['prefill_plus_mixed_ms']:>16.2f} "
              f"{row['medians']['wall_ms']:>8.2f} "
              f"{row['medians']['output_tokens_per_s']:>13.1f} "
              f"{row['relative_to_smallest_budget']['end_to_end_speedup']:>8.3f}x")
    print(f"wrote {args.output_dir / 'report.json'}")


if __name__ == "__main__":
    main()
