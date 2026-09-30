#!/usr/bin/env python3
"""Final accepted-engine comparison only: no candidate gates or new budget sweep.

Runs burst, staggered mixed, and synchronized prefill/decode/mixed diagnostics
through the existing validated harness. FA3 is external. Fused-head is an
explicit candidate; K-step is deliberately excluded from this comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
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
    result = [sys.executable, str(BENCHMARK), action,
            "--suite-dir", str(args.suite_dir.resolve()),
            "--output-dir", str(args.output_dir.resolve()), "--shape-id", shape,
            "--model", args.model, "--attention", "fa3", "--device", "cuda:0",
            "--seed", str(args.seed), "--warmups", str(args.warmups),
            "--repetitions", str(args.repetitions), "--vllm-budget", "default",
            "--vllm-python", sys.executable,
            "--prefill-budget", str(budget), "--prefill-graph-pool", "private",
            "--gemm-epilogues", "off"]
    if getattr(args, "fused_greedy_output", False):
        result.append("--fused-greedy-output")
    for commit in getattr(args, "reference_commits", ()):
        result.extend(["--resume-commit", commit])
    return result


def phase_command(args, module, action, shape, budget):
    result = command(args, action, shape, budget)
    result[1] = str(HERE / module)
    return result


def cell_commands(args, shape, budget):
    if args.backend == "both" and args.reference_dir is None:
        return [command(args, "run-cell", shape, budget)]
    action = "run-vllm" if args.backend == "vllm" else "run-local"
    commands = [phase_command(args, module, action, shape, budget) for module in (
        "benchmark_current_8_vs_vllm.py", "benchmark_current_mixed_8_vs_vllm.py",
        "benchmark_current_phases_vs_vllm.py")]
    if action == "run-local":
        commands.insert(0, command(args, "check", shape, budget))
        commands.append(command(args, "analyze", shape, budget))
    return commands


def validate_reference(args, budgets, *, check_hardware=True):
    """Validate every reference before any local model load; never relabel a result."""
    import benchmark_current_8_vs_vllm as burst
    import benchmark_current_mixed_8_vs_vllm as mixed
    import benchmark_current_phases_vs_vllm as phases
    from benchmark_latest_vs_vllm import resolve_model_source
    import torch

    staged, commits = [], set()
    model = None
    for shape in SHAPES:
        files = sorted((args.reference_dir / shape / "vllm").glob("*.json"))
        if len(files) != 1:
            raise ValueError(f"{shape}: reference needs exactly one burst vLLM JSON")
        mixed_path = args.reference_dir / "mixed" / shape / "vllm.json"
        phase_path = args.reference_dir / "phases" / shape / "vllm.json"
        payload = json.loads(files[0].read_text())
        mixed_row = json.loads(mixed_path.read_text())
        phase_row = json.loads(phase_path.read_text())
        source_commits = {payload.get("system", {}).get("repository", {}).get("commit"),
                          mixed_row.get("repository_commit"), phase_row.get("repository_commit")}
        if any(not value or len(value) < 7 or any(c not in "0123456789abcdef" for c in value)
               for value in source_commits):
            raise ValueError(f"{shape}: reference source commits are missing or invalid")
        commits.update(source_commits)
        contract = burst.parser().parse_args(command(args, "run-local", shape, budgets[shape])[2:])
        contract.resume_commit = sorted(source_commits)
        model = resolve_model_source(contract) if model is None else model
        case, _, workload, digest, blocks = burst.input_contract(contract, shape)
        burst.vllm_result(files[0].parent, workload, digest, case, blocks, model,
                          args.warmups, args.repetitions, vllm_budget="default")
        _, _, first, arrival, _, fingerprint, _ = mixed.mixed_plan(
            contract, shape, validate_schedule=False)
        mixed.validate_saved(mixed_path, shape_id=shape, fingerprint=fingerprint,
                             model=model, args=contract)
        phases.validate_saved(phase_path, contract, shape, fingerprint, model)
        for row in (mixed_row, phase_row):
            if (row.get("num_blocks") != blocks or row.get("arrival_step") != arrival
                    or not row.get("vllm_step_mode")
                    or len(row.get("runs", [])) != args.repetitions):
                raise ValueError(f"{shape}: reference KV capacity, arrival or repetitions differ")
        if mixed_row.get("first_wave") != first:
            raise ValueError(f"{shape}: reference first wave differs")
        if check_hardware:
            gpu = payload.get("system", {}).get("gpu", {})
            props = torch.cuda.get_device_properties("cuda:0")
            expected = {"name": props.name, "total_memory_bytes": props.total_memory,
                        "compute_capability": f"{props.major}.{props.minor}"}
            if any(gpu.get(key) != value for key, value in expected.items()):
                raise ValueError(f"{shape}: reference GPU hardware differs")
            recorded_torch = payload.get("system", {}).get("packages", {}).get("torch")
            if recorded_torch is not None and recorded_torch != torch.__version__:
                raise ValueError(f"{shape}: reference PyTorch version differs")
        for source, relative in ((files[0], Path(shape) / "vllm" / files[0].name),
                                 (mixed_path, Path("mixed") / shape / "vllm.json"),
                                 (phase_path, Path("phases") / shape / "vllm.json")):
            staged.append({"source": source, "relative": relative,
                           "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    args.reference_commits = sorted(commits)
    return staged


def stage_reference(args, staged):
    provenance = []
    for item in staged:
        destination = args.output_dir / item["relative"]
        if destination.exists():
            if hashlib.sha256(destination.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError(f"existing reference differs: {destination}")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item["source"], destination)
        provenance.append({"source": str(item["source"].resolve()),
                           "destination": str(item["relative"]), "sha256": item["sha256"]})
    atomic_json(args.output_dir / "reference-provenance.json",
                {"source_commits": args.reference_commits, "files": provenance})


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
    parser.add_argument("--backend", choices=("both", "local", "vllm"), default="both",
                        help="local never launches vLLM; vllm measures only the reusable reference")
    parser.add_argument("--reference-dir", type=Path,
                        help="completed reference tree from a prior final or eight-cell run")
    parser.add_argument("--fused-greedy-output", action="store_true",
                        help="opt-in full-engine fused LM-head/argmax candidate; K-step is NOT implied")
    parser.add_argument("--baseline-dir", type=Path,
                        help="after a fused-head local run, compare against this completed accepted run")
    parser.add_argument("--plan", action="store_true", help="CPU schedule validation only")
    args = parser.parse_args()
    fixed = args.prefill_budget if args.prefill_budget is not None else MAX_BUDGET
    if not 2048 <= fixed <= MAX_BUDGET or not 0 <= args.budget_tolerance < 1:
        parser.error("budget must be 2048..8192; tolerance must be in [0,1)")
    if args.warmups < 1 or args.repetitions < 1:
        parser.error("warmups and repetitions must be positive")
    if args.backend == "local" and args.reference_dir is None:
        parser.error("--backend local requires --reference-dir; no vLLM fallback is allowed")
    if args.backend == "vllm" and (args.reference_dir or args.fused_greedy_output or args.baseline_dir):
        parser.error("reference-only runs cannot use local candidate or reuse flags")
    if args.baseline_dir and not args.fused_greedy_output:
        parser.error("--baseline-dir is for the explicitly enabled fused-head candidate")
    if args.reference_dir and args.reference_dir.resolve() == args.output_dir.resolve():
        parser.error("reference and local output directories must differ")
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
    staged = validate_reference(args, budgets, check_hardware=not args.plan) if args.reference_dir else []
    manifest = {"preset": "accepted-fa3-no-candidates-v1", "model": args.model,
                "suite_dir": str(args.suite_dir.resolve()), "seed": args.seed,
                "warmups": args.warmups, "repetitions": args.repetitions,
                "vllm_version": "0.30.0", "vllm_budget": "default",
                "local_graph_pool": "private", "budgets": budgets,
                "budget_tolerance": args.budget_tolerance,
                "budget_calibration": source,
                "note": "Sweep rates choose budgets only; all final timings are measured anew."}
    if args.backend != "both" or args.reference_dir or args.fused_greedy_output:
        manifest.update(backend=args.backend,
                        fused_greedy_output=args.fused_greedy_output,
                        k_step_production=False,
                        reference_dir=str(args.reference_dir.resolve()) if args.reference_dir else None)
    if args.fused_greedy_output:
        manifest["preset"] = "accepted-fa3-fused-head-candidate-v1"
    if args.baseline_dir:
        from compare_fused_head import validate_baseline
        validate_baseline(args.baseline_dir, manifest)
    print(json.dumps({"manifest": manifest, "plans": plans}, indent=2), flush=True)
    if args.plan:
        for shape in SHAPES:
            for cmd in cell_commands(args, shape, budgets[shape]):
                print(" ".join(cmd))
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "final-config.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("final-config.json differs; keep results and use a new output directory")
    atomic_json(manifest_path, manifest)
    atomic_json(args.output_dir / "plans.json", plans)
    if staged:
        stage_reference(args, staged)
    rows = []
    for index, shape in enumerate(SHAPES, 1):
        print(f"[final {index}/8] {shape}, budget={budgets[shape]}", flush=True)
        try:
            for cmd in cell_commands(args, shape, budgets[shape]):
                stream_run(cmd, args.output_dir / f"{shape}.log")
        except subprocess.CalledProcessError as error:
            atomic_json(args.output_dir / "failure.json",
                        {"shape_id": shape, "exit_code": error.returncode,
                         "log": str(args.output_dir / f"{shape}.log"),
                         "completed_cells": [row["shape_id"] for row in rows]})
            raise SystemExit(f"Stopped at {shape}; preceding results retained. See {shape}.log")
        if args.backend == "vllm":
            rows.append({"shape_id": shape, "reference_complete": True})
            atomic_json(args.output_dir / "reference-summary.json",
                        {"status": "complete" if index == 8 else "partial",
                         "configuration": manifest, "rows": rows})
            continue
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
    if args.baseline_dir:
        from compare_fused_head import compare
        result = compare(args.baseline_dir, args.output_dir)
        atomic_json(args.output_dir / "head-ablation.json", result)
        print(f"Head A/B: burst {result['geomean_burst_speedup']:.3f}x, "
              f"mixed {result['geomean_mixed_speedup']:.3f}x; "
              f"exact baseline tokens={result['exact_baseline_tokens']}; "
              f"selected={result['selected']}", flush=True)
    print(f"Final results: {args.output_dir / ('reference-summary.json' if args.backend == 'vllm' else 'summary.json')}", flush=True)


if __name__ == "__main__":
    main()
