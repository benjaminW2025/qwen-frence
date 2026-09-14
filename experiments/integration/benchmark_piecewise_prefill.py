#!/usr/bin/env python3
"""Paired C++-scheduled eager vs piecewise-captured packed prefill.

Both arms use the same captured production decode graph. Capture occurs during
correctness preflight, before warmup and timing; eager attention remains identical.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
from model_adapter import GraphModelAdapter, PiecewiseGraphModelAdapter, allocate_pool
from model_setup import check_startup, load_model_only


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run_case(torch, cpp, engine, case, *, trials, samples, warmups, seed,
             max_capture_tokens, max_prefill_shapes, prefill_buckets):
    config = make_config(cpp, case)
    blocks = config.max_batch_size * ((config.max_context_length + 15) // 16)
    pool = allocate_pool(engine.cfg, blocks, engine.device)
    common = dict(max_running=config.max_batch_size,
                  max_context_length=config.max_context_length)
    adapters = {
        "eager": GraphModelAdapter(engine.model, pool, None, **common),
        "piecewise": PiecewiseGraphModelAdapter(
            engine.model, pool, None, **common,
            max_capture_tokens=max_capture_tokens,
            max_prefill_shapes=max_prefill_shapes,
            prefill_buckets=prefill_buckets),
    }
    torch.cuda.synchronize()
    requests = make_requests(case, seed, engine.cfg.vocab)

    def execute_arm(name, observer=None):
        for tensor in pool.k_pool + pool.v_pool:
            tensor.fill_(float("nan"))
        adapter = adapters[name]
        adapter.observer = observer
        loop = cpp.IterationLoop(config, torch.device(engine.device))
        try:
            return execute(torch, loop, adapter, requests)
        finally:
            adapter.observer = None

    reference = None
    expected_outputs = expected_schedule = None
    checks = {}
    for name in ("eager", "piecewise"):
        checker = TraceCheck(torch, reference)
        result = execute_arm(name, checker)
        checker.finish()
        schedule = [(step["kind"], step["calls"], step["completed"])
                    for step in result["steps"]]
        if reference is None:
            reference, expected_outputs, expected_schedule = checker.rows, result["outputs"], schedule
        elif result["outputs"] != expected_outputs or schedule != expected_schedule:
            raise AssertionError("piecewise prefill changed tokens or scheduled work")
        checks[name] = {"max_logit_error": checker.max_logit_error}

    piecewise = adapters["piecewise"].piecewise_prefill
    capture_shapes = sorted(piecewise.shapes)
    if not piecewise.captured_calls:
        raise AssertionError("no prefill call used piecewise capture")
    for _ in range(warmups):
        for name in ("eager", "piecewise"):
            execute_arm(name)
    calls_before_timing = (piecewise.captured_calls, piecewise.eager_calls,
                           piecewise.graph_replays)

    measurements = {name: [] for name in adapters}
    for trial in range(trials):
        trial_samples = {name: [] for name in adapters}
        for sample in range(samples):
            order = list(adapters)
            random.Random(seed + trial * 1009 + sample).shuffle(order)
            for name in order:
                result = execute_arm(name)
                schedule = [(step["kind"], step["calls"], step["completed"])
                            for step in result["steps"]]
                if result["outputs"] != expected_outputs or schedule != expected_schedule:
                    raise AssertionError(f"{name}: timed generation changed")
                trial_samples[name].append({
                    "wall_ms": result["wall_ms"],
                    "prefill_wall_ms": result["prefill_wall_ms"],
                    "mixed_wall_ms": result["mixed_wall_ms"],
                    "prefill_plus_mixed_ms": result["prefill_wall_ms"] + result["mixed_wall_ms"],
                    "decode_wall_ms": result["decode_wall_ms"],
                })
        for name in adapters:
            measurements[name].append(trial_samples[name])
    if sorted(piecewise.shapes) != capture_shapes:
        raise AssertionError("a new graph shape was captured during timing")
    medians = {phase: {name: [statistics.median(row[phase] for row in trial)
                              for trial in measurements[name]] for name in adapters}
               for phase in ("wall_ms", "prefill_plus_mixed_ms")}
    ratios = {phase: [a / b for a, b in zip(values["eager"], values["piecewise"])]
              for phase, values in medians.items()}
    return {"status": "ok", "case_id": case["id"], "checks": checks,
            "captured_token_shapes": capture_shapes,
            "configured_token_buckets": piecewise.buckets,
            "graphs_per_captured_shape": len(engine.model.layers) + 1,
            "timed_prefill_capture_calls": piecewise.captured_calls - calls_before_timing[0],
            "timed_prefill_eager_fallback_calls": piecewise.eager_calls - calls_before_timing[1],
            "timed_prefill_graph_replays": piecewise.graph_replays - calls_before_timing[2],
            "measurements": measurements, "trial_medians_ms": medians,
            "speedup": {phase: statistics.median(values) for phase, values in ratios.items()},
            "speedup_range": {phase: [min(values), max(values)]
                              for phase, values in ratios.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("smoke", "full", "longctx"), default="smoke")
    parser.add_argument("--case-id", default="ragged-b4-l769")
    parser.add_argument("--all-cases", action="store_true")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--max-capture-tokens", type=int, default=2048)
    parser.add_argument("--max-prefill-shapes", type=int, default=8)
    parser.add_argument("--prefill-buckets", type=int, nargs="+",
                        help="explicit packed-token buckets, e.g. 128 512 2048")
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--check-setup", action="store_true",
                        help="check dependencies before model loading or graph capture")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "experiments/results/piecewise-prefill")
    args = parser.parse_args()
    if min(args.trials, args.samples, args.max_capture_tokens, args.max_prefill_shapes) < 1 or args.warmups < 0:
        parser.error("trials, samples, and capture limits must be positive; warmups cannot be negative")
    if args.prefill_buckets and (any(b < 1 or b > args.max_capture_tokens
                                     for b in args.prefill_buckets) or
                                 len(set(args.prefill_buckets)) != len(args.prefill_buckets)):
        parser.error("prefill buckets must be distinct positive sizes within the capture limit")

    if args.check_setup:
        print(json.dumps(check_startup(args.device), indent=2))
        return

    startup = check_startup(args.device)
    import torch
    import inference_engine_cpp as cpp
    from run_benchmarks import system_metadata
    plan = make_plan(args.preset)
    selected = plan if args.all_cases else [c for c in plan if c["id"] == args.case_id]
    if not selected:
        raise ValueError("case ID is not in the selected preset")

    engine, load_seconds, hub_transfer = load_model_only(
        args.model, args.device, "float16", hub_transfer=startup["hub_transfer"])
    independent_check(torch, engine.model, args.device)
    atomic_json(args.output_dir / "manifest.json", {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model, "preset": args.preset, "cases": selected,
        "trials": args.trials, "samples": args.samples, "warmups": args.warmups,
        "seed": args.seed, "max_capture_tokens": args.max_capture_tokens,
        "max_prefill_shapes": args.max_prefill_shapes,
        "prefill_buckets": args.prefill_buckets,
        "model_load_seconds": load_seconds, "hub_transfer": hub_transfer,
        "system": system_metadata(),
        "execution": "C++ scheduler; captured production decode in both arms; eager vs piecewise packed prefill",
    })
    rows = []
    for case in selected:
        try:
            row = run_case(torch, cpp, engine, case, trials=args.trials,
                           samples=args.samples, warmups=args.warmups, seed=args.seed,
                           max_capture_tokens=args.max_capture_tokens,
                           max_prefill_shapes=args.max_prefill_shapes,
                           prefill_buckets=args.prefill_buckets)
        except Exception as error:
            row = {"status": "error", "case_id": case["id"], "error": repr(error)}
        rows.append(row)
        atomic_json(args.output_dir / "report.json", {"cases": rows})
        if row["status"] == "ok":
            print(f"{case['id']}: piecewise {row['speedup']['prefill_plus_mixed_ms']:.3f}x "
                  f"prefill+mixed, {row['speedup']['wall_ms']:.3f}x end to end", flush=True)
        else:
            print(f"{case['id']}: {row['error']}", flush=True)
    failed = [row for row in rows if row["status"] != "ok"]
    atomic_json(args.output_dir / "report.json",
                {"status": "complete" if not failed else "incomplete", "cases": rows})
    if failed:
        raise SystemExit(f"{len(failed)} prefill cases failed; inspect report.json")


if __name__ == "__main__":
    main()
