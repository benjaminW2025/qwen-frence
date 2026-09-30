#!/usr/bin/env python3
"""Qualify individual epilogues in full prefill/decode graphs against the accepted engine.

Each intervention runs in its own process. Original RMSNorm is retained for
all individual candidates and their combination; the legacy all-fused path
also folds normalization and is reported separately. Reports persist on
failure, and selection requires finite results and wins in every phase case.
"""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from ab_gemm_epilogues import INTERVENTIONS


def select(reports):
    """Choose one intervention and its qualified phases; never infer a win from a crash."""
    options = []
    for intervention, report in reports.items():
        if report.get("status") != "pass":
            continue
        phases = []
        rates = []
        for phase in ("prefill", "decode"):
            rows = report.get(phase, [])
            if rows and all(row["speedup"] > 1 and math.isfinite(row["speedup"])
                            and row["numerics"].get("finite", True)
                            and not row["numerics"]["non_tie_disagreements"]
                            and row.get("candidate_kv_finite", True) for row in rows):
                phases.append(phase)
                rates.extend(row["speedup"] for row in rows)
            else:
                rates.extend(1.0 for _ in rows)
        if phases:
            score = math.exp(sum(map(math.log, rates)) / len(rates))
            mode = "all" if len(phases) == 2 else phases[0]
            options.append((score, intervention, mode))
    if not options:
        return {"intervention": "all", "mode": "off", "score": 1.0}
    score, intervention, mode = max(options)
    return {"intervention": intervention, "mode": mode, "score": score}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--all-report", type=Path)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--attention", choices=("fa3", "project"), default="fa3")
    parser.add_argument("--interventions", nargs="+", choices=INTERVENTIONS, default=list(INTERVENTIONS))
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--decode-steps", type=int, default=32)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = {}
    for intervention in args.interventions:
        directory = args.output_dir / intervention
        report = directory / "report.json"
        if intervention == "all" and args.all_report and args.all_report.is_file():
            reports[intervention] = json.loads(args.all_report.read_text())
        else:
            if not report.exists():
                command = [sys.executable, str(HERE / "ab_gemm_epilogues.py"),
                           "--output-dir", str(directory), "--model", args.model,
                           "--attention", args.attention, "--intervention", intervention,
                           "--rounds", str(args.rounds), "--decode-steps", str(args.decode_steps)]
                print(f"[epilogue {intervention}] {' '.join(command)}", flush=True)
                result = subprocess.run(command, cwd=ROOT)
                if not report.exists():
                    directory.mkdir(parents=True, exist_ok=True)
                    report.write_text(json.dumps({"status": "crashed",
                        "intervention": intervention, "exit_code": result.returncode,
                        "prefill": [], "decode": []}, indent=2) + "\n")
            reports[intervention] = json.loads(report.read_text())
        summary = {"status": "running", "reports": reports, "selection": select(reports)}
        (args.output_dir / "progress.json").write_text(json.dumps(summary, indent=2) + "\n")
    summary = {"status": "complete", "reports": reports, "selection": select(reports)}
    (args.output_dir / "report.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"selected epilogues: {summary['selection']}", flush=True)


if __name__ == "__main__":
    main()
