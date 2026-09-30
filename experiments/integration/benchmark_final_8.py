#!/usr/bin/env python3
"""Final accepted-engine comparison only: no candidate gates or new budget sweep.

Runs burst, staggered mixed, and synchronized prefill/decode/mixed diagnostics
through the existing validated harness. FA3 is external; GEMM epilogues, shared
graph pools, boundary reuse, resident evolving metadata and fused heads are off.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(HERE)]
from fixed_regime import FACTORIAL_SHAPES
from benchmark_current_8_vs_vllm import atomic_json

SHAPES = tuple(row["id"] for row in FACTORIAL_SHAPES)
BENCHMARK = HERE / "benchmark_current_8_vs_vllm.py"
MAX_BUDGET = 8192


def read_sweep(path):
    if path.is_dir():
        path = next((path / name for name in ("summary.final.json", "summary.json")
                     if (path / name).is_file()), path / "summary.json")
    content = path.read_bytes()
    report = json.loads(content)
    if report.get("status") != "complete" or report.get("attention") != "fa3":
        raise ValueError("budget summary must be a completed FA3 sweep")
    if report.get("gemm_epilogues", "off") != "off":
        raise ValueError("budget calibration must not use rejected GEMM epilogues")
    return report, {"path": str(path.resolve()),
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "configuration": {key: report.get(key) for key in (
                        "prefill_graph_pool", "boundary_buffers",
                        "stable_decode_metadata", "fused_greedy_output")}}


def candidates(report, shape, fixed_budget):
    if report is None:
        return {fixed_budget: 1.0}
    raw = report["cells"][shape]["tokens_per_s"]
    rates = {int(budget): float(rate) for budget, rate in raw.items()
             if 2048 <= int(budget) <= MAX_BUDGET
             and math.isfinite(float(rate)) and float(rate) > 0}
    if not rates:
        raise ValueError(f"{shape}: no usable measured budgets between 2048 and {MAX_BUDGET}")
    return rates


def choose_budget(rates, tolerance):
    best = max(rates.values())
    return min(budget for budget, rate in rates.items() if rate >= best * (1 - tolerance))


def command(args, action, shape, budget):
    return [sys.executable, str(BENCHMARK), action,
            "--suite-dir", str(args.suite_dir.resolve()),
            "--output-dir", str(args.output_dir.resolve()), "--shape-id", shape,
            "--model", args.model, "--attention", "fa3", "--device", "cuda:0",
            "--seed", str(args.seed), "--warmups", str(args.warmups),
            "--repetitions", str(args.repetitions), "--vllm-budget", "default",
            "--vllm-python", sys.executable,
            "--prefill-budget", str(budget), "--prefill-graph-pool", "private",
            "--gemm-epilogues", "off"]


def supported_plan(args, shape, budget):
    result = subprocess.run(command(args, "plan", shape, budget), cwd=ROOT,
                            check=True, capture_output=True, text=True)
    row = json.loads(result.stdout)["shapes"][shape]
    return None if "unsupported" in row["staggered_mixed"] else row


def stream_run(cmd, log):
    """Keep the complete child traceback on disk even when engine startup fails."""
    print(" ".join(cmd), flush=True)
    with log.open("a") as handle:
        handle.write("\nCOMMAND: " + " ".join(cmd) + "\n")
        handle.flush()
        with subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as child:
            for line in child.stdout:
                print(line, end="", flush=True)
                handle.write(line)
                handle.flush()
            code = child.wait()
    if code:
        raise subprocess.CalledProcessError(code, cmd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite-dir", type=Path,
                        default=ROOT / "experiments/results/full-checkpoint-20260916T033540Z")
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument("--budget-summary", type=Path,
                        help="completed sweep JSON or its directory; selects budgets per cell")
    budget.add_argument("--prefill-budget", type=int,
                        help="explicit fixed budget instead of sweep selection; default 8192")
    parser.add_argument("--budget-tolerance", type=float, default=.01)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--plan", action="store_true", help="CPU schedule validation only")
    args = parser.parse_args()
    fixed = args.prefill_budget if args.prefill_budget is not None else MAX_BUDGET
    if not 2048 <= fixed <= MAX_BUDGET or not 0 <= args.budget_tolerance < 1:
        parser.error("budget must be 2048..8192; tolerance must be in [0,1)")
    if args.warmups < 1 or args.repetitions < 1:
        parser.error("warmups and repetitions must be positive")
    sweep, source = read_sweep(args.budget_summary) if args.budget_summary else (None, None)
    plans, budgets = {}, {}
    for shape in SHAPES:
        rates = candidates(sweep, shape, fixed)
        valid = {}
        for value in sorted(rates):
            plan = supported_plan(args, shape, value)
            if plan is not None:
                valid[value] = (rates[value], plan)
        if not valid:
            raise ValueError(f"{shape}: no budget supports both burst and staggered mixed")
        selected = choose_budget({b: row[0] for b, row in valid.items()}, args.budget_tolerance)
        budgets[shape], plans[shape] = selected, valid[selected][1]
    manifest = {"preset": "accepted-fa3-no-candidates-v1", "model": args.model,
                "suite_dir": str(args.suite_dir.resolve()), "seed": args.seed,
                "warmups": args.warmups, "repetitions": args.repetitions,
                "vllm_version": "0.30.0", "vllm_budget": "default",
                "local_graph_pool": "private", "budgets": budgets,
                "budget_tolerance": args.budget_tolerance,
                "budget_calibration": source,
                "note": "Sweep rates choose budgets only; all final timings are measured anew."}
    print(json.dumps({"manifest": manifest, "plans": plans}, indent=2), flush=True)
    if args.plan:
        for shape in SHAPES:
            print(" ".join(command(args, "run-cell", shape, budgets[shape])))
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "final-config.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("final-config.json differs; keep results and use a new output directory")
    atomic_json(manifest_path, manifest)
    atomic_json(args.output_dir / "plans.json", plans)
    rows = []
    for index, shape in enumerate(SHAPES, 1):
        print(f"[final {index}/8] {shape}, budget={budgets[shape]}", flush=True)
        try:
            stream_run(command(args, "run-cell", shape, budgets[shape]),
                       args.output_dir / f"{shape}.log")
        except subprocess.CalledProcessError as error:
            atomic_json(args.output_dir / "failure.json",
                        {"shape_id": shape, "exit_code": error.returncode,
                         "log": str(args.output_dir / f"{shape}.log"),
                         "completed_cells": [row["shape_id"] for row in rows]})
            raise SystemExit(f"Stopped at {shape}; preceding results retained. See {shape}.log")
        reports = {kind: json.loads(path.read_text()) for kind, path in (
            ("burst", args.output_dir / shape / "comparison.json"),
            ("mixed", args.output_dir / "mixed" / shape / "comparison.json"),
            ("phases", args.output_dir / "phases" / shape / "comparison.json"))}
        rows.append({"shape_id": shape, "prefill_budget": budgets[shape], **reports})
        atomic_json(args.output_dir / "summary.json",
                    {"status": "complete" if index == 8 else "partial",
                     "configuration": manifest, "rows": rows})
    failure = args.output_dir / "failure.json"
    if failure.exists():
        atomic_json(failure, {"status": "resolved", "completed_cells": list(SHAPES)})
    print(f"Final results: {args.output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
