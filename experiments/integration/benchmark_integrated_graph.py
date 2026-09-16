#!/usr/bin/env python3
"""One paired C++-scheduled experiment: eager, decode graph, prefill pieces, split-K.

This is an implementation comparison, not a vLLM or production-Python comparison.
All arms share requests, weights, scheduler configuration, and the physical KV pool.
Correctness and *actual* work-shape gates run before any timed workload.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for directory in (HERE, ROOT / "benchmarks", ROOT / "baseline", ROOT / "engine/kvcache",
                  ROOT / "engine/model_runner", ROOT / "engine/cpp/build"):
    sys.path.insert(0, str(directory))

from benchmark_scheduler_decode import TraceCheck, execute, independent_check, make_config
from design import make_plan, make_requests
from fixed_regime import verify_fixed_result
from model_adapter import (GraphModelAdapter, ModelAdapter, PiecewiseGraphModelAdapter,
                           allocate_pool)
from model_setup import check_startup, load_model_only

ARMS = ("eager", "decode_graph", "piecewise", "piecewise_splitk")
PHASES = ("wall_ms", "decode_wall_ms", "prefill_wall_ms", "mixed_wall_ms",
          "prefill_plus_mixed_ms")
COMPARISONS = (("decode_graph", "eager"),
               ("piecewise", "decode_graph"),
               ("piecewise_splitk", "piecewise"),
               ("piecewise", "eager"),
               ("piecewise_splitk", "eager"))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("fixed", "smoke", "full", "longctx"), default="fixed")
    parser.add_argument("--case-id", default="fixed-b8-l256-o128")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--expected-max-decode-batch", type=int, default=8,
                        help="hard gate on observed maximum decode batch; set for the chosen case")
    parser.add_argument("--min-full-decode-steps", type=int, default=64,
                        help="hard gate on how many pure/full decode steps reach the target batch")
    parser.add_argument("--expected-prefill-tokens", type=int, default=2048,
                        help="hard gate: at least one prefill call must have this packed-token count")
    parser.add_argument("--max-capture-tokens", type=int, default=2048)
    parser.add_argument("--max-prefill-shapes", type=int, default=8)
    parser.add_argument("--prefill-buckets", type=int, nargs="+", default=[2048],
                        help="explicit buckets; default captures only the 2048-token regime")
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--workload-in", type=Path,
                        help="frozen benchmark_core workload.json with the exact same burst requests")
    parser.add_argument("--plan", action="store_true", help="validate and print the plan without CUDA")
    parser.add_argument("--dry-schedule", action="store_true",
                        help="check actual C++ schedule shapes on CPU, then exit")
    parser.add_argument("--check-setup", action="store_true",
                        help="check dependencies without model loading or capture")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "experiments/results/integrated-graph")
    return parser


def validate_args(args):
    if min(args.trials, args.samples, args.expected_max_decode_batch,
           args.min_full_decode_steps, args.expected_prefill_tokens, args.max_capture_tokens,
           args.max_prefill_shapes) < 1 or args.warmups < 0:
        raise ValueError("trials, samples, shape targets, and capture limits must be positive; "
                         "warmups cannot be negative")
    if not args.prefill_buckets or any(b < 1 or b > args.max_capture_tokens
                                      for b in args.prefill_buckets):
        raise ValueError("prefill buckets must be positive and within the capture limit")
    if len(set(args.prefill_buckets)) != len(args.prefill_buckets):
        raise ValueError("prefill buckets must be distinct")
    if not any(b >= args.expected_prefill_tokens for b in args.prefill_buckets):
        raise ValueError("no configured bucket can capture the expected prefill token count")
    cases = [case for case in make_plan(args.preset) if case["id"] == args.case_id]
    if len(cases) != 1:
        raise ValueError("case ID is not in the selected preset")
    case = cases[0]
    if args.expected_max_decode_batch > case["max_running"]:
        raise ValueError("expected decode batch exceeds max_running")
    if args.expected_prefill_tokens > case["prefill_budget"]:
        raise ValueError("expected prefill tokens exceed the per-iteration budget")
    return case


def resolve_requests(args, case):
    if args.workload_in is None:
        return make_requests(case, args.seed, 151936)
    from benchmark_core import Workload

    workload = Workload.from_dict(json.loads(args.workload_in.read_text()))
    if workload.has_staggered_arrivals or any(case["arrivals"]):
        raise ValueError("frozen workload comparison supports burst cases only")
    if workload.seed != args.seed:
        raise ValueError("frozen workload seed differs from --seed")
    if len(workload.requests) != len(case["lengths"]):
        raise ValueError("frozen workload request count differs from the case")
    requests = []
    for i, request in enumerate(workload.requests):
        if request.request_id != str(i):
            raise ValueError("frozen workload request IDs must be 0,1,... in case order")
        if len(request.prompt_ids) != case["lengths"][i] or request.max_tokens != case["outputs"][i]:
            raise ValueError(f"frozen workload request {i} length/output differs from the case")
        if any(token < 0 or token >= 151936 for token in request.prompt_ids):
            raise ValueError(f"frozen workload request {i} contains an invalid Qwen token ID")
        requests.append({"id": i, "prompt": request.prompt_ids,
                         "output": request.max_tokens, "arrival": 0})
    return requests


def requests_fingerprint(requests):
    payload = json.dumps(requests, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def verify_actual_work(result, expected_batch, expected_tokens, min_full_decode_steps=1):
    batches = [call[1] for step in result["steps"] for call in step["calls"] if call[0]]
    prefill = [call[1] for step in result["steps"] for call in step["calls"] if not call[0]]
    if not batches or max(batches) != expected_batch:
        raise AssertionError(f"actual max decode batch {max(batches, default=0)} "
                             f"!= expected {expected_batch}; observed {sorted(set(batches))}")
    if expected_tokens not in prefill:
        raise AssertionError(f"no packed-prefill call with {expected_tokens} tokens; "
                             f"observed {sorted(set(prefill))}")
    full_steps = sum(step.get("kind") == "decode" and
                     any(call[0] and call[1] == expected_batch for call in step["calls"])
                     for step in result["steps"])
    if full_steps < min_full_decode_steps:
        raise AssertionError(f"only {full_steps} pure decode steps reached B={expected_batch}; "
                             f"required at least {min_full_decode_steps}")
    return {"decode_batch_histogram": result["decode_batch_histogram"],
            "prefill_token_counts": sorted(set(prefill)),
            "max_actual_decode_batch": max(batches),
            "pure_full_decode_steps_at_target_batch": full_steps}


def dry_schedule(torch, cpp, case, seed, requests=None):
    """Exercise real C++ scheduling on CPU before loading a model or CUDA graphs.

    The EOS ID is disabled and output lengths are fixed, so replacing sampled
    tokens with zero changes token values but not admission or iteration shapes.
    """
    config = make_config(cpp, case)
    if requests is None:
        requests = make_requests(case, seed, 1024)
    pending = sorted(requests, key=lambda row: (row["arrival"], row["id"]))
    loop = cpp.IterationLoop(config, torch.device("cpu"))
    cursor = iteration = 0
    steps = []
    limit = (sum(len(row["prompt"]) + row["output"] for row in requests)
             + max(row["arrival"] for row in requests) + 1)
    while cursor < len(pending) or loop.num_pending() or loop.num_running():
        if not loop.num_pending() and not loop.num_running() and cursor < len(pending):
            iteration = max(iteration, pending[cursor]["arrival"])
        while cursor < len(pending) and pending[cursor]["arrival"] <= iteration:
            request = pending[cursor]
            loop.submit_request(request["prompt"], request["output"])
            cursor += 1
        calls = []

        def dummy_forward(ids, positions, slots, cu, context, table, max_query, decode):
            calls.append((bool(decode), ids.numel(), context.numel(), max_query))
            return torch.zeros((context.numel(), 1))

        loop.step(dummy_forward)
        loop.pop_completed()
        if not calls:
            raise RuntimeError("CPU scheduler dry run made no progress")
        kinds = {call[0] for call in calls}
        steps.append({"kind": "mixed" if len(kinds) == 2 else
                               "decode" if True in kinds else "prefill",
                      "calls": calls})
        iteration += 1
        if iteration > limit:
            raise RuntimeError("CPU scheduler dry run exceeded iteration limit")
    batches = [call[1] for step in steps for call in step["calls"] if call[0]]
    return {"steps": steps,
            "decode_batch_histogram": {str(b): batches.count(b) for b in sorted(set(batches))}}


def schedule_of(result):
    return [(step["kind"], step["calls"], step["completed"])
            for step in result["steps"]]


def metrics(result):
    return {**{name: result[name] for name in PHASES if name != "prefill_plus_mixed_ms"},
            "prefill_plus_mixed_ms": result["prefill_wall_ms"] + result["mixed_wall_ms"],
            "max_actual_decode_batch": result["max_actual_decode_batch"]}


def paired_summary(measurements):
    medians = {phase: {arm: [statistics.median(row[phase] for row in trial)
                             for trial in measurements[arm]] for arm in ARMS}
               for phase in PHASES}
    effects = {}
    for candidate, baseline in COMPARISONS:
        phase_effects = {}
        for phase in PHASES:
            ratios = [a / b for a, b in zip(medians[phase][baseline],
                                            medians[phase][candidate]) if b > 0]
            if ratios:
                phase_effects[phase] = {"median_speedup": statistics.median(ratios),
                                        "trial_speedups": ratios,
                                        "range": [min(ratios), max(ratios)]}
        effects[f"{candidate}_vs_{baseline}"] = phase_effects
    return medians, effects


def splitk_decision(effects, minimum_speedup=1.02, *, executed=True):
    """Conservative fixed-regime gate, not a statistical significance claim."""
    if not executed:
        return {"choice": "inactive_short_context", "criterion":
                "production policy is selected below the split-K context threshold"}
    ratios = effects["piecewise_splitk_vs_piecewise"]["wall_ms"]["trial_speedups"]
    if len(ratios) < 3:
        return {"choice": "insufficient_trials", "minimum_speedup": minimum_speedup,
                "criterion": "at least 3 paired trials, each above the threshold"}
    return {"choice": "splitk_candidate" if all(ratio >= minimum_speedup for ratio in ratios)
            else "retain_production", "minimum_speedup": minimum_speedup,
            "criterion": "at least 3 paired trials, each above the threshold"}


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run_case(torch, cpp, engine, case, args, requests):
    config = make_config(cpp, case)
    # Include the same one-block-per-sequence headroom as the existing benchmark
    # driver and its matched vLLM KV-cache reservation.
    blocks = config.max_batch_size * (((config.max_context_length + 15) // 16) + 1)
    pool = allocate_pool(engine.cfg, blocks, engine.device)
    common = dict(max_running=config.max_batch_size,
                  max_context_length=config.max_context_length)
    adapters = {
        "eager": ModelAdapter(engine.model, pool, None),
        "decode_graph": GraphModelAdapter(engine.model, pool, None, **common),
        "piecewise": PiecewiseGraphModelAdapter(
            engine.model, pool, None, **common,
            max_capture_tokens=args.max_capture_tokens,
            max_prefill_shapes=args.max_prefill_shapes,
            prefill_buckets=args.prefill_buckets),
        "piecewise_splitk": PiecewiseGraphModelAdapter(
            engine.model, pool, None, **common, decode_attention_policy="splitk",
            max_capture_tokens=args.max_capture_tokens,
            max_prefill_shapes=args.max_prefill_shapes,
            prefill_buckets=args.prefill_buckets),
    }
    torch.cuda.synchronize()
    if any(token >= engine.cfg.vocab for row in requests for token in row["prompt"]):
        raise ValueError("workload token ID exceeds the loaded model vocabulary")

    def execute_arm(arm, *, observer=None):
        for tensor in pool.k_pool + pool.v_pool:
            tensor.fill_(float("nan"))
        adapter = adapters[arm]
        adapter.observer = observer
        loop = cpp.IterationLoop(config, torch.device(engine.device))
        try:
            return execute(torch, loop, adapter, requests)
        finally:
            adapter.observer = None

    reference = expected_outputs = expected_schedule = actual_work = None
    checks = {}
    for arm in ARMS:
        checker = TraceCheck(torch, reference)
        result = execute_arm(arm, observer=checker)
        checker.finish()
        work = verify_actual_work(result, args.expected_max_decode_batch,
                                  args.expected_prefill_tokens, args.min_full_decode_steps)
        if args.preset == "fixed":
            verify_fixed_result(result, case["id"])
        schedule = schedule_of(result)
        if reference is None:
            reference, expected_outputs, expected_schedule, actual_work = (
                checker.rows, result["outputs"], schedule, work)
        elif result["outputs"] != expected_outputs or schedule != expected_schedule:
            raise AssertionError(f"{arm}: tokens or scheduled work differ from eager reference")
        checks[arm] = {"max_logit_error": checker.max_logit_error,
                       "decisions": result["decisions"], "actual_work": work}

    splitk_executed = any(key != "production" for key in
                          checks["piecewise_splitk"]["decisions"])
    if adapters["piecewise_splitk"].action != "production" and not splitk_executed:
        raise AssertionError("split-K arm did not use split-K; choose a supported context")
    pieces = {arm: adapters[arm].piecewise_prefill for arm in ARMS[2:]}
    for arm, piece in pieces.items():
        if piece.captured_calls == 0:
            raise AssertionError(f"{arm}: piecewise capture was never used")
        if args.expected_prefill_tokens not in piece.shapes:
            # A larger bucket is allowed to capture the exact packed-token count.
            bucket = min(b for b in piece.buckets if b >= args.expected_prefill_tokens)
            if bucket not in piece.shapes:
                raise AssertionError(f"{arm}: expected packed-prefill bucket was not captured")

    for _ in range(args.warmups):
        for arm in ARMS:
            execute_arm(arm)
    capture_before = {arm: {"calls": piece.captured_calls,
                            "fallbacks": piece.eager_calls,
                            "replays": piece.graph_replays,
                            "shapes": sorted(piece.shapes)}
                      for arm, piece in pieces.items()}

    measurements = {arm: [] for arm in ARMS}
    for trial in range(args.trials):
        trial_samples = {arm: [] for arm in ARMS}
        for sample in range(args.samples):
            order = list(ARMS)
            random.Random(args.seed + trial * 1009 + sample).shuffle(order)
            for arm in order:
                result = execute_arm(arm)
                verify_actual_work(result, args.expected_max_decode_batch,
                                   args.expected_prefill_tokens, args.min_full_decode_steps)
                if args.preset == "fixed":
                    verify_fixed_result(result, case["id"])
                if result["outputs"] != expected_outputs or schedule_of(result) != expected_schedule:
                    raise AssertionError(f"{arm}: timed tokens or schedule changed")
                trial_samples[arm].append(metrics(result))
        for arm in ARMS:
            measurements[arm].append(trial_samples[arm])

    capture = {}
    for arm, piece in pieces.items():
        before = capture_before[arm]
        if sorted(piece.shapes) != before["shapes"]:
            raise AssertionError(f"{arm}: a new prefill graph was captured during timing")
        capture[arm] = {"captured_token_buckets": before["shapes"],
                        "timed_prefill_capture_calls": piece.captured_calls - before["calls"],
                        "timed_prefill_eager_fallback_calls": piece.eager_calls - before["fallbacks"],
                        "timed_prefill_graph_replays": piece.graph_replays - before["replays"]}
        if capture[arm]["timed_prefill_capture_calls"] == 0:
            raise AssertionError(f"{arm}: no timed prefill used a captured bucket")
    medians, effects = paired_summary(measurements)
    return {"status": "ok", "case_id": case["id"], "actual_work": actual_work,
            "checks": checks, "capture": capture, "measurements": measurements,
            "trial_medians_ms": medians, "effects": effects,
            "output_ids": expected_outputs,
            "splitk_executed": splitk_executed,
            "splitk_decision": splitk_decision(effects, executed=splitk_executed)}


def main():
    args = build_parser().parse_args()
    try:
        case = validate_args(args)
        requests = resolve_requests(args, case)
    except ValueError as error:
        raise SystemExit(f"invalid experiment plan: {error}") from error
    plan = {"case": case, "arms": ARMS,
            "expected_max_decode_batch": args.expected_max_decode_batch,
            "min_full_decode_steps": args.min_full_decode_steps,
            "expected_prefill_tokens": args.expected_prefill_tokens,
            "prefill_buckets": args.prefill_buckets, "trials": args.trials,
            "samples": args.samples, "warmups": args.warmups,
            "workload_in": str(args.workload_in) if args.workload_in else None,
            "requests_sha256": requests_fingerprint(requests)}
    if args.plan:
        print(json.dumps(plan, indent=2))
        return
    if args.check_setup:
        print(json.dumps(check_startup(args.device), indent=2))
        return
    if not args.dry_schedule:
        for name in ("manifest.json", "report.json"):
            if (args.output_dir / name).exists():
                raise SystemExit(f"refusing to overwrite existing {args.output_dir / name}; "
                                 "choose a fresh --output-dir")

    import torch
    import inference_engine_cpp as cpp

    cpu_work = verify_actual_work(dry_schedule(torch, cpp, case, args.seed, requests),
                                  args.expected_max_decode_batch,
                                  args.expected_prefill_tokens, args.min_full_decode_steps)
    print("CPU scheduler shape gate passed: " + json.dumps(cpu_work), flush=True)
    if args.dry_schedule:
        return
    startup = check_startup(args.device)
    from run_benchmarks import system_metadata

    engine, load_seconds, hub_transfer = load_model_only(
        args.model, args.device, "float16", hub_transfer=startup["hub_transfer"])
    independent_check(torch, engine.model, args.device)
    atomic_json(args.output_dir / "manifest.json", {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model, "device": args.device, "plan": plan,
        "cpu_dry_schedule_work": cpu_work,
        "model_load_seconds": load_seconds, "hub_transfer": hub_transfer,
        "system": system_metadata(),
        "scope": "matched C++ scheduler and model; staged eager/graph/piecewise/split-K arms; no vLLM"})
    try:
        row = run_case(torch, cpp, engine, case, args, requests)
    except Exception as error:
        atomic_json(args.output_dir / "report.json", {
            "status": "error", "case_id": case["id"], "error": repr(error)})
        raise
    atomic_json(args.output_dir / "report.json", row)
    for comparison, effects in row["effects"].items():
        print(f"{comparison}: {effects['wall_ms']['median_speedup']:.3f}x wall, "
              f"{effects['prefill_plus_mixed_ms']['median_speedup']:.3f}x prefill+mixed")


if __name__ == "__main__":
    main()
