#!/usr/bin/env python3
"""Resumable pages/program experiment and held-out (K, stages) policy study.

Commands: plan, run, analyze, profile. See experiments/decode/STAGE_POLICY.md.
H=1, page size=16, head dimension=128, four warps, FP16 are fixed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from functools import partial
import json
import math
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from decode_stage_policy import (aggregate_cases, auto_k, evaluate, make_plan, paired_interval,
                                 predict, select_policy, stable_hash, work_features)
from grouped_splitk_validation import attention_reference, check_output, correctness_cases, make_inputs
from benchmark_grouped_splitk_pipelined import kernel_diagnostics

SUITES = ("mechanism", "train", "validation", "test", "ragged")
SOURCES = ("benchmark_decode_stage_policy.py", "decode_stage_policy.py", "grouped_splitk_validation.py",
           "paged_decode_grouped_splitk_pipelined.py", "benchmark_grouped_splitk_pipelined.py")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("plan", "run", "analyze", "profile"))
    p.add_argument("--preset", choices=("smoke", "full"), default="full")
    p.add_argument("--output-dir", type=Path, default=HERE.parent / "results" / "decode-stage-policy")
    p.add_argument("--suite", choices=(*SUITES, "all"), default="mechanism")
    p.add_argument("--num-sms", type=int, default=132, help="Plan only; runs query hardware")
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--samples", type=int, default=9)
    p.add_argument("--seed", type=int, default=20260911)
    p.add_argument("--cache-modes", default="warm,evict", help="warm, evict, or both")
    p.add_argument("--evict-mib", type=int, default=256, help="L2 eviction stress buffer; outside timed interval")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--case-id", help="Run/profile one ID from the plan")
    p.add_argument("--k", type=int, default=8, help="Profile only")
    p.add_argument("--stages", type=int, choices=(1, 2, 3, 4), default=2, help="Profile only")
    p.add_argument("--role", choices=("partial", "reduce", "full"), default="partial", help="Profile only")
    p.add_argument("--fit-cache", choices=("warm", "evict"), default="warm", help="Analyze only")
    return p


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_records(directory):
    records = []
    for path in sorted((directory / "trials").glob("*.json")):
        trial = json.loads(path.read_text())
        if trial.get("status") != "complete":
            raise ValueError(f"invalid checkpoint {path}")
        records.extend(trial["records"])
    return records


def fingerprint(metadata):
    # Exclude suite/case selectors: separate invocations populate one frozen design.
    return stable_hash(metadata)


def ensure_manifest(directory, spec):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "manifest.json"
    identity = fingerprint(spec)
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest["fingerprint"] != identity:
            raise ValueError("Resume refused: source, hardware, or protocol changed; use a new output directory")
        return manifest
    manifest = {"fingerprint": identity, "created_at": datetime.now(timezone.utc).isoformat(), **spec}
    atomic_json(path, manifest)
    return manifest


def telemetry():
    try:
        return {"nvidia_smi": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu",
             "--format=csv,noheader"], text=True, timeout=5).strip()}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"unavailable": str(exc)}


def driver_version():
    try:
        return subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version",
                                        "--format=csv,noheader"], text=True, timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def capture(torch, launch):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            launch()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        launch()
    torch.cuda.current_stream().wait_stream(stream)
    return graph


def prepare(torch, attention, tensors, action):
    k, stages = action
    launches, compiled = {}, {}
    output = attention(*tensors, heads_per_program=1, split_k=k, num_stages=stages,
                       pipelined=stages > 1, num_warps=4, diagnostics=compiled, launches=launches)
    return output, launches, compiled


def do_profile(args, torch, attention, case, cache_modes):
    # One launch under a stable NVTX range; compilation/warmup excluded from ncu.
    if [args.k, args.stages] not in case["actions"]:
        raise ValueError("profile action is not in this case's planned actions")
    if len(cache_modes) != 1:
        raise ValueError("profile requires exactly one --cache-modes value")
    tensors = make_inputs(case["lengths"], seed=args.seed)
    output, launches, compiled = prepare(torch, attention, tensors, [args.k, args.stages])
    check_output(output, attention_reference(*tensors))
    if args.role not in launches:
        raise ValueError("K=1 has no separate reduction kernel")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    resources = kernel_diagnostics(compiled, str(args.output_dir / f"profile-{case['id']}-k{args.k}-s{args.stages}"))
    flush = torch.empty(args.evict_mib * 1024 * 1024, device="cuda", dtype=torch.uint8)
    for _ in range(3):
        launches["full"]()
    if cache_modes[0] == "evict":
        flush.zero_()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("decode_stage_profile")
    try:
        launches[args.role]()
    finally:
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    print(json.dumps({"case": case, "role": args.role, "cache": cache_modes[0], "resources": resources}, indent=2))


def run(args, cache_modes):
    import torch
    import triton
    from paged_decode_grouped_splitk_pipelined import grouped_splitk_attention
    from paged_decode_attention import paged_decode_attention

    torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    props = torch.cuda.get_device_properties(args.device)
    sms = props.multi_processor_count
    plan = make_plan(args.preset, sms)
    selected = [c for c in plan if (args.case_id == c["id"] if args.case_id else
                                   args.suite == "all" or c["suite"] == args.suite)]
    if not selected:
        raise ValueError("no matching cases")
    if args.command == "profile":
        if len(selected) != 1:
            raise ValueError("profile requires --case-id")
        return do_profile(args, torch, grouped_splitk_attention, selected[0], cache_modes)
    hardware = {"name": props.name, "sms": sms, "memory": props.total_memory,
                "capability": list(torch.cuda.get_device_capability()), "uuid": str(getattr(props, "uuid", "unknown")),
                "torch": torch.__version__, "triton": triton.__version__, "cuda": torch.version.cuda,
                "python": platform.python_version(), "driver": driver_version()}
    spec = {"schema_version": 1, "preset": args.preset, "trials": args.trials, "samples": args.samples,
            "seed": args.seed, "cache_modes": cache_modes, "evict_mib": args.evict_mib, "hardware": hardware,
            "source_hashes": {name: stable_hash((HERE / name).read_text()) for name in SOURCES}, "plan": plan,
            "timing": "one attention invocation per CUDA graph replay; cache conditioning outside events"}
    manifest = ensure_manifest(args.output_dir, spec)
    trial_dir = args.output_dir / "trials"
    trial_dir.mkdir(exist_ok=True)
    flush = torch.empty(args.evict_mib * 1024 * 1024, device="cuda", dtype=torch.uint8) if "evict" in cache_modes else None
    # Independent math checks for every kernel specialization, including one-page,
    # strided, masked-tail, empty-partition, negative-score, and overflow cases.
    adversarial = [(name, tensors, options, attention_reference(*tensors, **options))
                   for name, tensors, options in correctness_cases(seed=args.seed)]
    preflight_path = args.output_dir / "preflight.json"
    checked = json.loads(preflight_path.read_text()) if preflight_path.exists() else {}
    all_actions = sorted({tuple(a) for case in selected for a in case["actions"]})
    for k, stages in all_actions:
        key = f"{k}:{stages}"
        if key in checked:
            continue
        errors = []
        for name, tensors, options, expected in adversarial:
            result = grouped_splitk_attention(*tensors, heads_per_program=1, split_k=k, num_stages=stages,
                                              pipelined=stages > 1, **options)
            errors.append({"case": name, "max_abs_error": check_output(result, expected)})
        checked[key] = errors
        atomic_json(preflight_path, checked)
        print(f"Preflight K={k} stages={stages} passed", flush=True)
    # Trial order and case order are randomized; trials use independent tensor seeds.
    work = [(case, trial) for case in selected for trial in range(args.trials)]
    random.Random(args.seed).shuffle(work)
    for case, trial in work:
        checkpoint = trial_dir / f"{case['id']}-t{trial}.json"
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text())
            if saved.get("fingerprint") != manifest["fingerprint"] or saved.get("status") != "complete":
                raise ValueError(f"invalid checkpoint {checkpoint}")
            continue
        print(f"Running {case['id']} trial {trial + 1}/{args.trials}", flush=True)
        started = datetime.now(timezone.utc).isoformat()
        wall_start = time.perf_counter()
        before = telemetry()
        seed = args.seed + trial * 1000003 + int(stable_hash(case["id"])[:7], 16)
        tensors = make_inputs(case["lengths"], seed=seed, poison_padding=True)
        # Dense reference checks every measured row; includes ragged and tail shapes.
        expected = attention_reference(*tensors)
        prepared = {}
        for action in case["actions"]:
            output, launches, compiled = prepare(torch, grouped_splitk_attention, tensors, action)
            error = check_output(output, expected)
            prepared[tuple(action)] = {"output": output, "launches": launches,
                "graphs": {role: capture(torch, launch) for role, launch in launches.items()},
                "error": error, "resources": kernel_diagnostics(compiled)}
        prod = partial(paged_decode_attention, *tensors)
        prod_error = check_output(prod(), expected)
        prod_graph = capture(torch, prod)
        tasks = [(action, role, cache) for action, item in prepared.items() for role in item["graphs"] for cache in cache_modes]
        tasks += [("production", "full", cache) for cache in cache_modes]
        samples = {task: [] for task in tasks}
        orders = []
        rng = random.Random(seed)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for sample in range(args.samples):
            rng.shuffle(tasks)
            orders.append([[list(a) if a != "production" else a, role, cache] for a, role, cache in tasks])
            for action, role, cache in tasks:
                graph = prod_graph if action == "production" else prepared[action]["graphs"][role]
                # For reduce-only, partial inputs were computed during preparation.
                # Warm the exact operation; eviction stress writes > H100 L2 capacity.
                if cache == "warm":
                    graph.replay()
                else:
                    flush.zero_()
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                elapsed = start.elapsed_time(end)
                if not math.isfinite(elapsed) or elapsed <= 0:
                    raise ValueError("nonpositive/nonfinite device timing")
                samples[(action, role, cache)].append(elapsed)
            print(f"  paired sample round {sample + 1}/{args.samples}", flush=True)
        # Verify actual graph replay outputs, not only eager launch outputs.
        for item in prepared.values():
            item["graphs"]["full"].replay()
            check_output(item["output"], expected)
        rows = []
        for (action, role, cache), values in samples.items():
            row = {"case_id": case["id"], "suite": case["suite"], "trial": trial,
                   "action": list(action) if action != "production" else action, "role": role, "cache": cache,
                   "samples_ms": values, "median_ms": statistics.median(values), "seed": seed}
            if action == "production":
                row["max_abs_error"] = prod_error
            else:
                row.update({"max_abs_error": prepared[action]["error"],
                            "resources": prepared[action]["resources"], **work_features(case, action[0], sms)})
            rows.append(row)
        atomic_json(checkpoint, {"status": "complete", "fingerprint": manifest["fingerprint"],
                                "case_id": case["id"], "trial": trial, "before": before, "after": telemetry(),
                                "started_at": started, "wall_seconds": time.perf_counter() - wall_start,
                                "orders": orders, "records": rows})
        # Graphs/closures retain their own buffers; release before constructing another case.
        del prepared, tensors, expected, prod, prod_graph, item, graph, output, launches, compiled
    print(f"Completed selected work; resumable checkpoints: {trial_dir}")


def analyze(args):
    directory = args.output_dir
    manifest = json.loads((directory / "manifest.json").read_text())
    records = load_records(directory)
    for path in (directory / "trials").glob("*.json"):
        if json.loads(path.read_text())["fingerprint"] != manifest["fingerprint"]:
            raise ValueError("mixed protocol checkpoints")
    trials = manifest["trials"]
    plan = manifest["plan"]
    cases_by_id = {case["id"]: case for case in plan}
    seen = set()
    for record in records:
        action = record["action"] if record["action"] == "production" else tuple(record["action"])
        key = (record["case_id"], action, record["role"], record["cache"], record["trial"])
        case = cases_by_id.get(record["case_id"])
        samples = record["samples_ms"]
        if (key in seen or case is None or not 0 <= record["trial"] < trials
                or record["cache"] not in manifest["cache_modes"]
                or not samples or any(not math.isfinite(x) or x <= 0 for x in samples)
                or len(samples) != manifest.get("samples", len(samples))
                or not math.isclose(record["median_ms"], statistics.median(samples), rel_tol=1e-10)):
            raise ValueError(f"invalid/duplicate timing record: {key}")
        if action != "production" and list(action) not in case["actions"]:
            raise ValueError(f"unplanned action: {key}")
        seen.add(key)
    # Always emit matched stage-effect estimates, even for a mechanism-only run.
    index = {(r["case_id"], tuple(r["action"]), r["role"], r["cache"], r["trial"]): r["median_ms"]
             for r in records if r["action"] != "production"}
    effects = []
    for case in plan:
        for k, s in case["actions"]:
            if s == 1:
                continue
            for role in ("partial", "full"):
                for cache in manifest["cache_modes"]:
                    pairs = [(index.get((case["id"], (k, 1), role, cache, t)),
                              index.get((case["id"], (k, s), role, cache, t))) for t in range(trials)]
                    if any(a is None or b is None for a, b in pairs):
                        continue
                    effects.append({"case_id": case["id"], "suite": case["suite"], "k": k, "stages": s,
                                    "role": role, "cache": cache, **work_features(case, k, manifest["hardware"]["sms"]),
                                    **paired_interval([a / b for a, b in pairs], seed=manifest["seed"])})
    atomic_json(directory / "stage-effects.json", effects)
    dispatch_plan = [c for c in plan if c["suite"] != "mechanism"]
    try:
        data = aggregate_cases(dispatch_plan, records, trials, args.fit_cache)
    except ValueError as exc:
        atomic_json(directory / "policy-report.json", {"status": "incomplete_data", "production_ready": False,
                                                       "fingerprint": manifest["fingerprint"], "reason": str(exc)})
        print(f"Stage effects written ({len(effects)} comparisons). Policy fitting pending: {exc}")
        return
    policy = select_policy(data)
    # The selected tree depends only on train+validation, never on test/ragged costs.
    chosen = policy["selected"]["tree"]
    reports = {}
    for cache in manifest["cache_modes"]:
        dataset = aggregate_cases(dispatch_plan, records, trials, cache)
        for suite in ("test", "ragged"):
            heldout = [r for r in dataset if r["suite"] == suite]
            selectors = {
                "policy": lambda r: predict(chosen, r["features"]),
                "fixed_train_best": lambda r: predict(policy["fixed_baseline"], r["features"]),
                "auto_k_stage1": lambda r: (auto_k(r["features"]["batch"], r["features"]["max_pages"], manifest["hardware"]["sms"]), 1),
                "k1_stage1": lambda r: (1, 1),
            }
            result = {name: evaluate(heldout, select) for name, select in selectors.items()}
            policy_rows = {row["case_id"]: tuple(row["action"]) for row in result["policy"]["rows"]}
            # Paired trial uncertainty for the selected policy against production.
            production = {(r["case_id"], r["trial"]): r["median_ms"] for r in records
                          if r["action"] == "production" and r["cache"] == cache}
            result["policy_vs_production"] = [
                {"case_id": row["id"], **paired_interval([
                    production[(row["id"], t)] / index[(row["id"], policy_rows[row["id"]], "full", cache, t)]
                    for t in range(trials)], seed=manifest["seed"])} for row in heldout]
            reports[f"{suite}:{cache}"] = result
    validation = policy["selected"]["validation"]
    test = reports[f"test:{args.fit_cache}"]["policy"]
    gate = (manifest.get("preset") == "full" and trials >= 5 and validation["p95_regret"] <= 1.10
            and all(r["policy"]["p95_regret"] <= 1.10 and r["policy"]["max_regret"] <= 1.20
                    for r in reports.values())
            and all(row["ci95"] is not None and row["ci95"][0] >= .95
                    for r in reports.values() for row in r["policy_vs_production"]))
    boundary_hits = []
    for row in data:
        best = min(row["costs"], key=row["costs"].get)
        if best[0] == max(a[0] for a in row["costs"]) or best[1] == max(a[1] for a in row["costs"]):
            boundary_hits.append({"case_id": row["id"], "best_measured_action": list(best)})
    artifact = {"status": "microbenchmark_candidate" if gate else "needs_more_work",
                "production_ready": False, "fingerprint": manifest["fingerprint"], "fit_cache": args.fit_cache,
                "scope": "H=1 FP16 page16 head128 four warps; GPU/software fixed by manifest",
                "gate": "full preset and at least 5 trials; validation and every held-out cache/suite p95 regret <=1.10; "
                        "held-out max regret <=1.20; per-shape policy/production speedup CI lower bound >=0.95",
                "boundary_hits": boundary_hits, "policy": policy, "heldout": reports}
    atomic_json(directory / "policy-report.json", artifact)
    print(json.dumps({"status": artifact["status"], "tree": chosen, "test": {k: v for k, v in test.items() if k != "rows"}}, indent=2))


def main():
    args = parser().parse_args()
    caches = args.cache_modes.split(",")
    if len(set(caches)) != len(caches) or not caches or any(c not in ("warm", "evict") for c in caches):
        raise ValueError("cache-modes must contain warm and/or evict without duplicates")
    if args.trials < 1 or args.samples < 3 or args.evict_mib < 1:
        raise ValueError("trials/evict-mib must be positive and samples at least 3")
    if args.command == "analyze":
        return analyze(args)
    if args.command == "plan":
        plan = make_plan(args.preset, args.num_sms)
        selected = [c for c in plan if (c["id"] == args.case_id if args.case_id else args.suite == "all" or c["suite"] == args.suite)]
        counts = {suite: sum(c["suite"] == suite for c in plan) for suite in SUITES}
        observations = sum(sum(2 if k == 1 else 3 for k, s in c["actions"]) + 1 for c in selected)
        print(json.dumps({"cases_by_suite": counts, "selected_cases": len(selected),
                          "timed_replays": observations * args.trials * args.samples * len(caches),
                          "negative_control": "one page/program offers no cross-page overlap",
                          "plan": selected}, indent=2))
    else:
        run(args, caches)


if __name__ == "__main__":
    main()
