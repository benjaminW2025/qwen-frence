#!/usr/bin/env python3
"""Pick the local engine's prefill token budget across the eight frozen cells.

Runs benchmark_current_8_vs_vllm.py's local arm (the exact engine of the
accepted FA3 comparison: C++ packed mixed step, CUDA graphs, accepted fusions,
--attention fa3 by default) for each (cell, budget). Only the budget changes.
Off the frozen 2048 budget, each GPU run must reproduce the CPU dry schedule step
for step before it is timed. No vLLM runs: this picks our budget; a vLLM
comparison at a new budget is a separate, matched experiment.

Run from the vLLM environment when --attention fa3 (FA3 lives there). Budgets at
or above a cell's whole prompt cohort behave identically, so they are measured
once and marked equivalent. Every subprocess, plan and run, uses the same
interpreter and repository state.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE)]
from fixed_regime import FACTORIAL_SHAPES, PREFILL_TOKENS_PER_STEP  # noqa: E402

BENCHMARK = HERE / "benchmark_current_8_vs_vllm.py"
DEFAULT_BUDGETS = (1024, 2048, 4096, 8192, 16384)   # 16384 is vLLM 0.30.0's H100 offline default
# With private per-segment graph pools every captured bucket keeps ~29 segments of
# scratch for this many tokens; the shared pool keeps roughly one segment's worth.
PRIVATE_POOL_BUDGET_LIMIT = 8192


def cohort_tokens(shape):
    return shape["batch"] * shape["prompt_length"]


def plan_runs(shapes, budgets):
    """(shape_id, budget, equivalent_to) with budgets past a cohort folded onto one run."""
    runs = []
    for shape in shapes:
        seen = {}
        for budget in budgets:
            effective = min(budget, cohort_tokens(shape))
            runs.append((shape["id"], budget, seen.get(effective)))
            seen.setdefault(effective, budget)
    return runs


def budget_dir(root, budget):
    return root / f"budget-{budget}"


def common(args):
    return ["--suite-dir", str(args.suite_dir), "--model", args.model, "--device", args.device,
            "--seed", str(args.seed), "--warmups", str(args.warmups),
            "--repetitions", str(args.repetitions), "--attention", args.attention,
            "--prefill-graph-pool", args.prefill_graph_pool,
            "--gemm-epilogues", args.gemm_epilogues,
            *(["--boundary-buffers"] if args.boundary_buffers else [])]


def push(args, message):
    commands = [["git", "add", "-f", str(args.output_dir)],
                ["git", "commit", "-qm", message],
                ["git", "push", "-q"]]
    for command in commands:
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode != 0:
            # Results stay on disk; a failed push must not stop the sweep.
            print(f"warning: {' '.join(command)} failed ({result.returncode})", flush=True)
            return


def summarize(args, runs):
    table = {}
    for shape_id, budget, equivalent in runs:
        source = budget_dir(args.output_dir, equivalent or budget) / shape_id / "local.json"
        if not source.is_file():
            continue
        local = json.loads(source.read_text())
        table.setdefault(shape_id, {})[budget] = dict(
            tokens_per_s=local["median_output_tokens_per_s"], measured_as=equivalent or budget,
            outputs=local["runs"][0]["outputs"],
            peak_reserved_gib=(local["peak_reserved_bytes"] / 2**30
                               if "peak_reserved_bytes" in local else None),
            graph_reserved_gib={bucket: value / 2**30 for bucket, value in
                                local.get("prefill_graph_reserved_bytes", {}).items()})
    cells, geomean = {}, {}
    for shape_id, row in table.items():
        control = row.get(PREFILL_TOKENS_PER_STEP)
        best = max(row, key=lambda b: row[b]["tokens_per_s"])
        cells[shape_id] = dict(
            best_budget=best,
            tokens_per_s={b: r["tokens_per_s"] for b, r in row.items()},
            measured_as={b: r["measured_as"] for b, r in row.items() if r["measured_as"] != b},
            peak_reserved_gib={b: r["peak_reserved_gib"] for b, r in row.items()},
            prefill_graph_reserved_gib={b: r["graph_reserved_gib"] for b, r in row.items()},
            speedup_vs_2048={b: r["tokens_per_s"] / control["tokens_per_s"] for b, r in row.items()}
            if control else None,
            # Batching changes FP16 accumulation order, so tokens may legitimately differ.
            outputs_equal_to_2048={b: r["outputs"] == control["outputs"] for b, r in row.items()}
            if control else None)
    # A shared graph pool moves where segment scratch lives, not the math: at the
    # frozen budget its tokens must equal a private-pool run of the same engine.
    reference = getattr(args, "reference_dir", None)
    if getattr(args, "gemm_epilogues", "off") != "off":
        # Fused epilogues change rounding, so exact tokens against the unfused
        # reference are not expected; divergence_report.py judges them instead.
        reference = None
    if reference is not None:
        for shape_id, cell in cells.items():
            path = reference / shape_id / "local.json"
            control = table[shape_id].get(PREFILL_TOKENS_PER_STEP)
            cell["outputs_equal_to_reference_2048"] = (
                json.loads(path.read_text())["runs"][0]["outputs"] == control["outputs"]
                if path.is_file() and control else None)
    complete = [c for c in cells.values() if c["speedup_vs_2048"]]
    for budget in args.budgets:
        ratios = [c["speedup_vs_2048"][budget] for c in complete if budget in c["speedup_vs_2048"]]
        if len(ratios) == len(complete) and ratios:
            geomean[budget] = math.exp(sum(map(math.log, ratios)) / len(ratios))
    summary = dict(status="complete" if len(complete) == len(args.shape_ids) else "partial",
                   attention=args.attention, prefill_graph_pool=args.prefill_graph_pool,
                   gemm_epilogues=getattr(args, "gemm_epilogues", "off"),
                   boundary_buffers=bool(getattr(args, "boundary_buffers", False)),
                   budgets=args.budgets, cells=cells,
                   geomean_speedup_vs_2048=geomean,
                   best_single_budget=max(geomean, key=geomean.get) if geomean else None,
                   note="tokens_per_s is the median local burst throughput; speedups are relative to "
                        "the frozen 2048 budget on the same engine and pod")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    header = "cell".ljust(22) + "".join(f"{b:>10}" for b in args.budgets) + "   best"
    print(header)
    for shape_id in args.shape_ids:
        cell = cells.get(shape_id)
        if cell is None:
            print(shape_id.ljust(22) + "  (not run)")
            continue
        values = "".join(f"{cell['tokens_per_s'].get(b, float('nan')):>10.0f}" for b in args.budgets)
        print(shape_id.ljust(22) + values + f"   {cell['best_budget']}")
    mismatched = [shape for shape, cell in cells.items() if cell.get("outputs_equal_to_reference_2048") is False]
    if mismatched:
        print(f"WARNING: 2048-budget tokens differ from the reference run for {mismatched}; "
              "do not trust this sweep's timings until explained", flush=True)
    if geomean:
        print("geomean vs 2048: " + ", ".join(f"{b}: {v:.3f}x" for b, v in geomean.items())
              + f"  -> best single budget {summary['best_single_budget']}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite-dir", type=Path,
                        default=ROOT / "experiments/results/full-checkpoint-20260916T033540Z")
    parser.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS))
    parser.add_argument("--shape-ids", nargs="+", default=[row["id"] for row in FACTORIAL_SHAPES],
                        choices=[row["id"] for row in FACTORIAL_SHAPES])
    parser.add_argument("--attention", choices=("fa3", "project"), default="fa3")
    parser.add_argument("--prefill-graph-pool", choices=("shared", "private"), default="shared",
                        help="shared (default) lets large budgets fit; every budget in one sweep "
                             "uses the same setting, including the 2048 reference")
    parser.add_argument("--gemm-epilogues", choices=("off", "prefill", "decode", "all"), default="off",
                        help="tune the budget on the engine that will run: the fused GEMM "
                             "epilogues change per-token cost, so the best budget can move")
    parser.add_argument("--boundary-buffers", action="store_true")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--reference-dir", type=Path,
                        default=ROOT / "experiments/results/attention-rerun-4XR8Oi/eight-fa3",
                        help="accepted private-pool 2048 run; its tokens must equal this sweep's 2048 tokens")
    parser.add_argument("--push", action="store_true",
                        help="git add/commit/push the output directory after every cell")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if PREFILL_TOKENS_PER_STEP not in args.budgets:
        parser.error(f"include {PREFILL_TOKENS_PER_STEP}: it is the reference every budget is compared to")
    if args.budgets != sorted(set(args.budgets)) or min(args.budgets) < 1:
        parser.error("budgets must be distinct, positive and increasing")
    if args.prefill_graph_pool == "private" and max(args.budgets) > PRIVATE_POOL_BUDGET_LIMIT:
        parser.error(f"budgets above {PRIVATE_POOL_BUDGET_LIMIT} need --prefill-graph-pool shared: "
                     "private per-segment pools would hold ~29 segments of scratch per bucket")
    shapes = [row for row in FACTORIAL_SHAPES if row["id"] in args.shape_ids]
    runs = plan_runs(shapes, args.budgets)
    if args.summarize_only:
        summarize(args, runs)
        return

    # CPU plans for every budget first: a budget the scheduler cannot honor fails here, not after GPU time.
    for budget in args.budgets:
        command = [sys.executable, str(BENCHMARK), "plan", "--output-dir", str(budget_dir(args.output_dir, budget)),
                   "--prefill-budget", str(budget), *common(args)]
        for shape_id in args.shape_ids:
            subprocess.run([*command, "--shape-id", shape_id], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    print(f"CPU plans pass for {len(args.budgets)} budgets x {len(shapes)} cells", flush=True)

    measured = [(s, b) for s, b, equivalent in runs if equivalent is None]
    for index, (shape_id, budget) in enumerate(measured, 1):
        print(f"[{index}/{len(measured)}] {shape_id} budget={budget}", flush=True)
        subprocess.run([sys.executable, str(BENCHMARK), "run-local", "--shape-id", shape_id,
                        "--output-dir", str(budget_dir(args.output_dir, budget)),
                        "--prefill-budget", str(budget), *common(args)], cwd=ROOT, check=True)
        summarize(args, runs)
        if args.push:
            push(args, f"prefill budget sweep: {shape_id} budget {budget}")
    summarize(args, runs)
    if args.push:
        push(args, "prefill budget sweep: summary")


if __name__ == "__main__":
    main()
