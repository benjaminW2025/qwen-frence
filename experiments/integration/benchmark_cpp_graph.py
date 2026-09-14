#!/usr/bin/env python3
"""Paired real-model C++ scheduler + captured decode, production vs split-K.

Both arms use the same C++ iteration loop, packed eager prefill, model weights,
physical KV pool, requests, and graph buckets. Only captured decode attention
differs. Capture and pool allocation are excluded from timed workloads.
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
from model_adapter import GraphModelAdapter, allocate_pool
from model_setup import check_startup, load_model_only

ARMS = ("production", "splitk")


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run_case(torch, cpp, engine, case, *, trials, samples, warmups, seed):
    config = make_config(cpp, case)
    blocks = config.max_batch_size * ((config.max_context_length + 15) // 16)
    pool = allocate_pool(engine.cfg, blocks, engine.device)
    adapters = {}
    for arm in ARMS:
        adapters[arm] = GraphModelAdapter(
            engine.model, pool, None,
            max_running=config.max_batch_size,
            max_context_length=config.max_context_length,
            decode_attention_policy=arm,
        )
    torch.cuda.synchronize()

    requests = make_requests(case, seed, engine.cfg.vocab)

    def execute_arm(arm, *, poison=False, observer=None):
        if poison:
            for tensor in pool.k_pool + pool.v_pool:
                tensor.fill_(float("nan"))
        adapter = adapters[arm]
        adapter.observer = observer
        loop = cpp.IterationLoop(config, torch.device(engine.device))
        try:
            return execute(torch, loop, adapter, requests)
        finally:
            adapter.observer = None

    reference = None
    expected_outputs = None
    expected_schedule = None
    checks = {}
    for arm in ARMS:
        checker = TraceCheck(torch, reference)
        result = execute_arm(arm, poison=True, observer=checker)
        checker.finish()
        schedule = [(step["kind"], step["calls"], step["completed"])
                    for step in result["steps"]]
        if reference is None:
            reference = checker.rows
            expected_outputs = result["outputs"]
            expected_schedule = schedule
        elif result["outputs"] != expected_outputs or schedule != expected_schedule:
            raise AssertionError("captured arms produced different tokens or scheduled work")
        checks[arm] = {"max_logit_error": checker.max_logit_error,
                       "decisions": result["decisions"],
                       "decode_batches": result["decode_batch_histogram"]}

    if not any(key != "production" for key in checks["splitk"]["decisions"]):
        raise AssertionError("split-K arm fell back to production for every decode step")

    for _ in range(warmups):
        for arm in ARMS:
            execute_arm(arm, poison=True)

    measurements = {arm: [] for arm in ARMS}
    for trial in range(trials):
        trial_samples = {arm: [] for arm in ARMS}
        for sample in range(samples):
            order = list(ARMS)
            random.Random(seed + trial * 1009 + sample).shuffle(order)
            for arm in order:
                result = execute_arm(arm, poison=True)
                schedule = [(step["kind"], step["calls"], step["completed"])
                            for step in result["steps"]]
                if result["outputs"] != expected_outputs or schedule != expected_schedule:
                    raise AssertionError(f"{arm}: timed generation changed from preflight")
                trial_samples[arm].append({
                    "wall_ms": result["wall_ms"],
                    "decode_wall_ms": result["decode_wall_ms"],
                    "prefill_wall_ms": result["prefill_wall_ms"],
                    "mixed_wall_ms": result["mixed_wall_ms"],
                })
        for arm in ARMS:
            measurements[arm].append(trial_samples[arm])

    medians = {arm: [statistics.median(sample["wall_ms"] for sample in trial)
                     for trial in measurements[arm]] for arm in ARMS}
    ratios = [a / c for a, c in zip(medians["production"], medians["splitk"])]
    return {"status": "ok", "case_id": case["id"], "checks": checks,
            "measurements": measurements, "trial_medians_ms": medians,
            "speedup": statistics.median(ratios),
            "speedup_range": [min(ratios), max(ratios)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("smoke", "full", "longctx"), default="longctx")
    parser.add_argument("--case-id", default="uniform-b4-l2048")
    parser.add_argument("--all-cases", action="store_true")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--check-setup", action="store_true",
                        help="check dependencies before model loading or graph capture")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "experiments/results/cpp-graph-splitk")
    args = parser.parse_args()
    if min(args.trials, args.samples) < 1 or args.warmups < 0:
        parser.error("trials and samples must be positive; warmups cannot be negative")

    if args.check_setup:
        print(json.dumps(check_startup(args.device), indent=2))
        return

    startup = check_startup(args.device)
    import torch
    import inference_engine_cpp as cpp
    from run_benchmarks import system_metadata
    plan = make_plan(args.preset)
    selected = plan if args.all_cases else [case for case in plan if case["id"] == args.case_id]
    if not selected:
        raise ValueError("case ID is not in the selected preset")

    engine, load_seconds, hub_transfer = load_model_only(
        args.model, args.device, "float16", hub_transfer=startup["hub_transfer"])
    independent_check(torch, engine.model, args.device)
    manifest = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
                "model": args.model, "dtype": "float16", "block_size": 16,
                "preset": args.preset, "cases": selected, "arms": list(ARMS),
                "trials": args.trials, "samples": args.samples, "warmups": args.warmups,
                "seed": args.seed, "model_load_seconds": load_seconds,
                "hub_transfer": hub_transfer,
                "system": system_metadata(),
                "execution": "C++ scheduler; eager packed prefill; bucketed captured decode"}
    atomic_json(args.output_dir / "manifest.json", manifest)

    rows = []
    for case in selected:
        try:
            row = run_case(torch, cpp, engine, case, trials=args.trials,
                           samples=args.samples, warmups=args.warmups, seed=args.seed)
        except Exception as error:
            row = {"status": "error", "case_id": case["id"], "error": repr(error)}
        rows.append(row)
        atomic_json(args.output_dir / "report.json", {"cases": rows})
        if row["status"] == "ok":
            print(f"{case['id']}: split-K {row['speedup']:.3f}x vs production graph", flush=True)
        else:
            print(f"{case['id']}: {row['error']}", flush=True)

    failed = [row for row in rows if row["status"] != "ok"]
    atomic_json(args.output_dir / "report.json",
                {"status": "complete" if not failed else "incomplete", "cases": rows})
    if failed:
        raise SystemExit(f"{len(failed)} graph cases failed; inspect report.json")


if __name__ == "__main__":
    main()
