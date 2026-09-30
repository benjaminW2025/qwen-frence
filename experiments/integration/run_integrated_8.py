#!/usr/bin/env python3
"""One GPU session: verify each improvement, integrate what passed, run the eight cells.

Stages, each resumable (a finished stage's report is reused) and pushed when
--push is given:

  preflight          harness check with every variant flag on: FA3 in-place output,
                     the CUTLASS extension (ABI, source hash, fragment proof), one GEMM
  micro/mlp          SwiGLU epilogue and CUTLASS-vs-cuBLAS mainloop parity
  micro/layer        one decoder layer: 8-kernel chain vs 4 fused GEMMs, vs an FP32 chain
  micro/model        whole model, fused vs accepted: all-step logits, K/V, timing
  micro/graph-pool   private vs shared graph pool: bitwise outputs, capture memory
  micro/boundary/*   boundary-buffer reuse at B8-short and B64-long
  micro/metadata/*   resident decode metadata at B8-short and B64-long
  micro/output/*     universal final-norm/LM-head/argmax at B8-short and B64-long
  divergence/accepted  why the accepted baseline's tokens differ from vLLM's (FP32)
  decisions          admitted epilogues, pool, buffers, metadata and greedy output
  micro/kstep/*      advisory K=2/4/8 graphs and fused head at B8/B64
  budget-sweep       engine-only prefill budgets on the admitted engine
  eight              all eight cells (burst, staggered mixed, phases) vs vLLM at its default budget
  divergence/final   the new run vs vLLM, and vs the accepted baseline's local tokens

A failed gate disables that feature and is recorded; a required stage that
crashes without writing its report stops the session. K-step capture is
advisory and cannot stop the eight-cell comparison. Run from the vLLM
environment (FA3 and vLLM live there) with the CUTLASS extension built in it.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE)]
from fixed_regime import FACTORIAL_SHAPES, PREFILL_TOKENS_PER_STEP  # noqa: E402

SHAPES = tuple(row["id"] for row in FACTORIAL_SHAPES)
SUITE = ROOT / "experiments/results/full-checkpoint-20260916T033540Z"
ACCEPTED = ROOT / "experiments/results/attention-rerun-4XR8Oi/eight-fa3"
BENCHMARK = HERE / "benchmark_current_8_vs_vllm.py"
PRIVATE_POOL_BUDGET_LIMIT = 8192
PUSHABLE_RESULT_SUFFIXES = frozenset({".csv", ".json", ".log", ".md", ".txt"})
MAX_PUSHABLE_RESULT_BYTES = 10 * 1024 * 1024


def choose_budget(tokens_per_s, tolerance):
    """The smallest budget within `tolerance` of the fastest: gains inside run-to-run noise
    do not justify a larger budget (more graph memory, longer prefill steps)."""
    best = max(tokens_per_s.values())
    return min(int(budget) for budget, rate in tokens_per_s.items() if rate >= best * (1 - tolerance))


def mixed_supported(python, shape, budget, graph_pool, suite_dir, output_dir):
    """Whether the staggered-mixed workload's fixed arrival step is mixed at this budget (CPU plan)."""
    result = subprocess.run([python, str(BENCHMARK), "plan", "--shape-id", shape, "--suite-dir", str(suite_dir),
                             "--output-dir", str(output_dir), "--prefill-budget", str(budget),
                             "--prefill-graph-pool", graph_pool],
                            cwd=ROOT, check=True, capture_output=True, text=True)
    return "unsupported" not in json.loads(result.stdout)["shapes"][shape]["staggered_mixed"]


def pushable_result_artifacts(root, max_bytes=MAX_PUSHABLE_RESULT_BYTES):
    """Return compact, reviewable evidence and exclude binary/raw profiling state."""
    if not root.is_dir():
        return []
    return [path for path in sorted(root.rglob("*"))
            if path.is_file()
            and path.suffix.lower() in PUSHABLE_RESULT_SUFFIXES
            and path.stat().st_size <= max_bytes]


def decide(root, args):
    """Admit each feature only on its microbenchmark evidence; explicit CLI choices override."""
    evidence, decision = {}, {}

    def load(path):
        return json.loads(path.read_text()) if path.is_file() else None

    layer = load(root / "micro/layer/report.json")
    model = load(root / "micro/model/report.json")
    layer_ok = bool(layer) and all(row["correct"] for row in layer["rows"])
    numerics_ok = bool(model) and model["status"] == "pass"
    prefill_fast = bool(model) and all(row["speedup"] > 1 for row in model["prefill"])
    decode_fast = bool(model) and all(row["speedup"] > 1 for row in model["decode"])
    evidence["gemm_epilogues"] = {
        "layer_correct": layer_ok, "model_numerics_pass": numerics_ok,
        "prefill_speedups": [row["speedup"] for row in model["prefill"]] if model else None,
        "decode_speedups": [row["speedup"] for row in model["decode"]] if model else None}
    admitted = [phase for phase, fast in (("prefill", prefill_fast), ("decode", decode_fast))
                if fast and layer_ok and numerics_ok]
    auto = {(): "off", ("prefill",): "prefill", ("decode",): "decode",
            ("prefill", "decode"): "all"}[tuple(admitted)]
    decision["gemm_epilogues"] = auto if args.gemm_epilogues == "auto" else args.gemm_epilogues

    pool = load(root / "micro/graph-pool/report.json")
    evidence["graph_pool"] = pool and {"status": pool["status"], "rows": pool["rows"]}
    auto = "shared" if pool and pool["status"] == "pass" else "private"
    decision["graph_pool"] = auto if args.graph_pool == "auto" else args.graph_pool

    boundary = [load(root / f"micro/boundary/{name}/report.json")
                for name in ("b8-short", "b64-long")]
    evidence["boundary_buffers"] = boundary
    auto = bool(all(row and row["status"] == "complete"
                    and row["boundary_reuse_executed"]
                    and row["speedup_prefill_mixed"] > 1 for row in boundary))
    decision["boundary_buffers"] = auto if args.boundary_buffers == "auto" else args.boundary_buffers == "on"
    metadata = [load(root / f"micro/metadata/{name}/report.json")
                for name in ("b8-short", "b64-long")]
    evidence["stable_decode_metadata"] = metadata
    auto = bool(all(row and row["status"] == "complete"
                    and row["metadata_reuse_executed"]
                    and row["speedup_decode"] > 1 for row in metadata))
    decision["stable_decode_metadata"] = (
        auto if args.stable_decode_metadata == "auto"
        else args.stable_decode_metadata == "on")
    return decision, evidence


def variant_cli(decision):
    return ["--prefill-graph-pool", decision["graph_pool"],
            "--gemm-epilogues", decision["gemm_epilogues"],
            *(["--boundary-buffers"] if decision["boundary_buffers"] else []),
            *(["--stable-decode-metadata"] if decision["stable_decode_metadata"] else []),
            *(["--fused-greedy-output"] if decision["fused_greedy_output"] else [])]


class Session:
    def __init__(self, args):
        self.args, self.root = args, args.output_dir
        self.python = sys.executable

    def push(self, message):
        if not self.args.push:
            return
        artifacts = pushable_result_artifacts(self.root)
        if not artifacts:
            print(f"warning: no compact artifacts to push under {self.root}", flush=True)
            return
        commands = []
        for offset in range(0, len(artifacts), 100):
            commands.append(["git", "add", "-f", "--",
                             *map(str, artifacts[offset:offset + 100])])
        commands.extend((["git", "diff", "--cached", "--quiet", "--", str(self.root)],))
        for command in commands:
            result = subprocess.run(command, cwd=ROOT)
            if command[1:4] == ["diff", "--cached", "--quiet"]:
                if result.returncode == 0:
                    return
                break
            if result.returncode != 0:
                print(f"warning: {' '.join(command[:4])} failed", flush=True)
                return
        for command in (["git", "commit", "-qm", message, "--", str(self.root)],
                        ["git", "push", "-q"]):
            if subprocess.run(command, cwd=ROOT).returncode != 0:
                # Results stay on disk; a failed push must not stop the session.
                print(f"warning: {' '.join(command)} failed", flush=True)
                return

    def stage(self, name, report, command, *, gate=False):
        """Run command unless report exists. With gate, a nonzero exit that still wrote
        its report is a recorded failure, not a crash."""
        if report.exists():
            print(f"[{name}] complete; reusing {report.relative_to(ROOT)}", flush=True)
            return True
        stale = report.parent if report.suffix == ".json" and report.name == "report.json" else None
        if stale is not None and stale.exists():
            # Crashed earlier run: keep it for inspection, never delete results.
            aside = stale.with_name(f"{stale.name}.incomplete-{int(time.time())}")
            stale.rename(aside)
            print(f"[{name}] moved incomplete output to {aside.name}", flush=True)
        print(f"[{name}] {' '.join(map(str, command))}", flush=True)
        if self.args.plan:
            return True
        code = subprocess.run([str(part) for part in command], cwd=ROOT).returncode
        if code and not (gate and report.exists()):
            raise SystemExit(f"[{name}] failed with exit code {code} before writing {report}")
        self.push(f"integrated eight-cell session: {name}")
        return code == 0

    def advisory(self, name, report, command):
        """Run an optional experiment without risking the required session.

        A crashed subprocess is isolated from the subsequent budget/eight-cell
        processes.  Its partial directory is retained and automatically moved
        aside on the next invocation so the experiment remains resumable.
        """
        if report.exists():
            print(f"[{name}] complete; reusing {report.relative_to(ROOT)}", flush=True)
            return True
        if report.parent.exists():
            aside = report.parent.with_name(
                f"{report.parent.name}.incomplete-{int(time.time())}")
            report.parent.rename(aside)
            print(f"[{name}] moved incomplete output to {aside.name}", flush=True)
        print(f"[{name} advisory] {' '.join(map(str, command))}", flush=True)
        if self.args.plan:
            return True
        code = subprocess.run([str(part) for part in command], cwd=ROOT).returncode
        if code:
            report.parent.mkdir(parents=True, exist_ok=True)
            (report.parent / "failure.json").write_text(json.dumps({
                "status": "crashed", "exit_code": code,
                "command": [str(part) for part in command],
                "note": "Advisory failure; required integrated run continued."
            }, indent=2) + "\n")
            print(f"warning: [{name}] advisory failed with exit code {code}; continuing",
                  flush=True)
        self.push(f"integrated eight-cell session: {name}")
        return code == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite-dir", type=Path, default=SUITE)
    parser.add_argument("--attention", choices=("fa3", "project"), default="fa3")
    parser.add_argument("--vllm-budget", choices=("default", "matched"), default="default")
    parser.add_argument("--gemm-epilogues", choices=("auto", "off", "prefill", "decode", "all"), default="auto")
    parser.add_argument("--graph-pool", choices=("auto", "shared", "private"), default="auto")
    parser.add_argument("--boundary-buffers", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--stable-decode-metadata", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--fused-greedy-output", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--budgets", type=int, nargs="+", default=[1024, 2048, 4096, 8192, 16384])
    parser.add_argument("--budget-choice", choices=("per-cell", "single"), default="per-cell",
                        help="per-cell: each cell's own best budget; single: the best geomean budget")
    parser.add_argument("--budget-tolerance", type=float, default=.01,
                        help="prefer the smallest budget within this fraction of a cell's best")
    parser.add_argument("--shape-ids", nargs="+", choices=SHAPES, default=list(SHAPES))
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--push", action="store_true", help="git add/commit/push after every stage and cell")
    parser.add_argument("--plan", action="store_true", help="print every stage's command; run nothing")
    args = parser.parse_args()
    root = args.output_dir
    # The preflight command validates kernels without writing into the session
    # directory.  Create that directory before recording its successful
    # sentinel; a brand-new --output-dir must work without manual preparation.
    if not args.plan:
        root.mkdir(parents=True, exist_ok=True)
    session = Session(args)
    python = sys.executable
    common = ["--model", args.model, "--attention", args.attention]
    run = ["--suite-dir", str(args.suite_dir), "--warmups", str(args.warmups),
           "--repetitions", str(args.repetitions), *common]

    session.stage("preflight", root / "preflight.json",
                  [python, BENCHMARK, "check", "--shape-id", args.shape_ids[0],
                   "--output-dir", root / "eight", *run, "--gemm-epilogues", "all",
                   "--stable-decode-metadata", "--fused-greedy-output",
                   *(["--boundary-buffers"] if args.attention == "fa3" else [])])
    if not args.plan and not (root / "preflight.json").exists():
        (root / "preflight.json").write_text(json.dumps({"status": "pass"}) + "\n")

    micro = root / "micro"
    session.stage("micro/mlp", micro / "mlp/report.json",
                  [python, ROOT / "experiments/prefill/benchmark_mlp_epilogue.py",
                   "--output-dir", micro / "mlp"])
    session.stage("micro/layer", micro / "layer/report.json",
                  [python, ROOT / "experiments/prefill/benchmark_layer_epilogues.py",
                   "--output-dir", micro / "layer",
                   "--rows", "16384", "8192", "4096", "2048", "1024", "256", "64", "8"], gate=True)
    session.stage("micro/model", micro / "model/report.json",
                  [python, HERE / "ab_gemm_epilogues.py", "--output-dir", micro / "model", *common], gate=True)
    session.stage("micro/graph-pool", micro / "graph-pool/report.json",
                  [python, ROOT / "experiments/prefill/benchmark_graph_pool.py",
                   "--output", micro / "graph-pool/report.json", *common,
                   "--cases", "8x256", "1x2048", "4x2048", "8x2048"], gate=True)
    if args.attention == "fa3":
        probes = (("b8-short", "fixed-b8-l256-o128"),
                  ("b64-long", "fixed-b64-l2048-o128"))
        for label, shape in probes:
            session.stage(f"micro/boundary/{label}", micro / f"boundary/{label}/report.json",
                          [python, HERE / "benchmark_cpp_packed_mixed.py", "run",
                           "--intervention", "boundary-buffers", "--shape-id", shape,
                           "--suite-dir", args.suite_dir, "--model", args.model,
                           "--repetitions", str(args.repetitions),
                           "--output-dir", micro / f"boundary/{label}"], gate=True)
            session.stage(f"micro/metadata/{label}", micro / f"metadata/{label}/report.json",
                          [python, HERE / "benchmark_cpp_packed_mixed.py", "run",
                           "--intervention", "resident-metadata", "--shape-id", shape,
                           "--suite-dir", args.suite_dir, "--model", args.model,
                           "--repetitions", str(args.repetitions),
                           "--output-dir", micro / f"metadata/{label}"], gate=True)
    session.stage("divergence/accepted", root / "divergence/accepted-vs-vllm.json",
                  [python, HERE / "divergence_report.py", "--results-dir", ACCEPTED,
                   "--suite-dir", args.suite_dir, "--model", args.model,
                   "--output", root / "divergence/accepted-vs-vllm.json"])
    if args.plan:
        decision = {"gemm_epilogues": "<from micro/model>", "graph_pool": "<from micro/graph-pool>",
                    "boundary_buffers": "<from micro/boundary>",
                    "stable_decode_metadata": "<from micro/metadata>",
                    "fused_greedy_output": "<from micro/output>"}
    else:
        decision, evidence = decide(root, args)
        if decision["graph_pool"] == "private":
            args.budgets = [b for b in args.budgets if b <= PRIVATE_POOL_BUDGET_LIMIT]

    # The head kernel is row-generic. Qualify one universal scheduler contract
    # after the other feature choices are known, holding those choices constant
    # in control and candidate.
    output_reports = []
    for label, shape in (("b8-short", "fixed-b8-l256-o128"),
                         ("b64-long", "fixed-b64-l2048-o128")):
        command = [python, HERE / "benchmark_cpp_packed_mixed.py", "run",
                   "--intervention", "fused-greedy-output", "--shape-id", shape,
                   "--suite-dir", args.suite_dir, "--model", args.model,
                   "--repetitions", str(args.repetitions),
                   "--output-dir", micro / f"output/{label}"]
        if not args.plan:
            command.extend(["--gemm-epilogues", decision["gemm_epilogues"]])
            if decision["boundary_buffers"]:
                command.append("--baseline-boundary-buffers")
            if decision["stable_decode_metadata"]:
                command.append("--baseline-stable-decode-metadata")
        report = micro / f"output/{label}/report.json"
        session.stage(f"micro/output/{label}", report, command, gate=True)
        output_reports.append(report)

    if not args.plan:
        rows = [json.loads(path.read_text()) if path.is_file() else None
                for path in output_reports]
        evidence["fused_greedy_output"] = rows
        auto = bool(all(row and row["status"] == "complete"
                        and row["fused_greedy_output_executed"]
                        and row["speedup_decode"] > 1
                        and row["speedup_prefill_mixed"] > 1
                        for row in rows))
        decision["fused_greedy_output"] = (
            auto if args.fused_greedy_output == "auto"
            else args.fused_greedy_output == "on")
        (root / "decisions.json").write_text(json.dumps(
            {"decision": decision, "evidence": evidence, "budgets": args.budgets},
            indent=2) + "\n")
        print(f"[decisions] {decision}", flush=True)
        session.push("integrated eight-cell session: decisions")

    # Advisory only until the C++ scheduler has chunk commit, first-EOS
    # truncation and K1 fallback. Run against the decode GEMM path selected
    # above so the measured K-step effect reflects the candidate engine.
    for label, batch in (("b8", 8), ("b64", 64)):
        command = [python, HERE / "benchmark_decode_control_plane.py", "run",
                   "--batch", str(batch), "--context", "4096",
                   "--warmups", str(max(1, args.warmups)),
                   "--repetitions", str(max(10, args.repetitions)),
                   "--output-dir", micro / f"kstep/{label}"]
        if not args.plan and decision["gemm_epilogues"] in ("decode", "all"):
            command.append("--fused-gemm-epilogues")
        session.advisory(f"micro/kstep/{label}",
                         micro / f"kstep/{label}/report.json", command)

    sweep_dir = root / "budget-sweep"
    session.stage("budget-sweep", sweep_dir / "summary.final.json",
                  [python, HERE / "sweep_prefill_budget_8.py", "--output-dir", sweep_dir,
                   "--suite-dir", args.suite_dir, "--budgets", *map(str, args.budgets),
                   "--shape-ids", *args.shape_ids, *run, *(["--push"] if args.push else []),
                   *(variant_cli(decision) if not args.plan else ["<variant>"])])
    if args.plan:
        print(f"[eight] per cell: {BENCHMARK.name} run-cell --prefill-budget <chosen> "
              f"--vllm-budget {args.vllm_budget} <variant> ...")
        return
    if not (sweep_dir / "summary.final.json").exists():
        summary = json.loads((sweep_dir / "summary.json").read_text())
        if summary["status"] != "complete":
            raise SystemExit("budget sweep incomplete; rerun to resume it")
        shutil.copyfile(sweep_dir / "summary.json", sweep_dir / "summary.final.json")
    sweep = json.loads((sweep_dir / "summary.final.json").read_text())
    budgets = {}
    for shape in args.shape_ids:
        rates = {int(budget): rate for budget, rate in sweep["cells"][shape]["tokens_per_s"].items()
                 if mixed_supported(python, shape, int(budget), decision["graph_pool"],
                                    args.suite_dir, root / "eight")}
        if args.budget_choice == "single":
            budgets[shape] = int(sweep["best_single_budget"])
            if budgets[shape] not in rates:
                raise SystemExit(f"{shape}: the single best budget {budgets[shape]} cannot run the mixed stage")
        else:
            budgets[shape] = choose_budget(rates, args.budget_tolerance)
    (root / "budgets.json").write_text(json.dumps(
        {"choice": args.budget_choice, "tolerance": args.budget_tolerance, "budgets": budgets}, indent=2) + "\n")
    print(f"[budgets] {budgets}", flush=True)

    eight = root / "eight"
    for index, shape in enumerate(args.shape_ids, 1):
        session.stage(f"eight {index}/{len(args.shape_ids)} {shape}", eight / "phases" / shape / "comparison.json",
                      [python, BENCHMARK, "run-cell", "--shape-id", shape, "--output-dir", eight, *run,
                       "--vllm-python", python, "--vllm-budget", args.vllm_budget,
                       "--prefill-budget", str(budgets[shape]), *variant_cli(decision)])

    rows = []
    for shape in args.shape_ids:
        burst = json.loads((eight / shape / "comparison.json").read_text())
        mixed = json.loads((eight / "mixed" / shape / "comparison.json").read_text())
        accepted = {kind: json.loads(path.read_text())["local_over_vllm"] if path.is_file() else None
                    for kind, path in (("burst", ACCEPTED / shape / "comparison.json"),
                                       ("mixed", ACCEPTED / "mixed" / shape / "comparison.json"))}
        rows.append({"shape_id": shape, "prefill_budget": budgets[shape],
                     "burst_local_over_vllm": burst["local_over_vllm"],
                     "mixed_local_over_vllm": mixed["local_over_vllm"],
                     "accepted_burst_local_over_vllm": accepted["burst"],
                     "accepted_mixed_local_over_vllm": accepted["mixed"],
                     "burst_tokens_per_s": {"local": burst["local_output_tokens_per_s"],
                                            "vllm": burst["vllm_output_tokens_per_s"]},
                     "mixed_tokens_per_s": {"local": mixed["local_output_tokens_per_s"],
                                            "vllm": mixed["vllm_output_tokens_per_s"]},
                     "exact_outputs": {"burst": [burst["exact_output_requests"], burst["total_requests"]],
                                       "mixed": [mixed["exact_output_requests"], mixed["total_requests"]]}})

    def geomean(values):
        return math.exp(sum(map(math.log, values)) / len(values))

    summary = {"status": "complete", "attention": args.attention, "vllm_budget": args.vllm_budget,
               "decision": decision, "budgets": budgets, "rows": rows,
               "geomean_burst": geomean([r["burst_local_over_vllm"] for r in rows]),
               "geomean_mixed": geomean([r["mixed_local_over_vllm"] for r in rows])}
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{'cell':24}{'budget':>7}{'burst':>8}{'(was)':>8}{'mixed':>8}{'(was)':>8}")
    for row in rows:
        was = [row[f"accepted_{kind}_local_over_vllm"] for kind in ("burst", "mixed")]
        print(f"{row['shape_id']:24}{row['prefill_budget']:>7}{row['burst_local_over_vllm']:>8.3f}"
              f"{was[0] or float('nan'):>8.3f}{row['mixed_local_over_vllm']:>8.3f}{was[1] or float('nan'):>8.3f}")
    print(f"geomean burst {summary['geomean_burst']:.3f}x, mixed {summary['geomean_mixed']:.3f}x "
          f"(local/vLLM; vLLM at its {args.vllm_budget} budget; '(was)' = accepted baseline at matched 2048)")
    session.push("integrated eight-cell session: summary")

    session.stage("divergence/final-vs-vllm", root / "divergence/final-vs-vllm.json",
                  [python, HERE / "divergence_report.py", "--results-dir", eight, "--suite-dir", args.suite_dir,
                   "--model", args.model, "--output", root / "divergence/final-vs-vllm.json"])
    session.stage("divergence/final-vs-accepted", root / "divergence/final-vs-accepted.json",
                  [python, HERE / "divergence_report.py", "--results-dir", eight, "--against", ACCEPTED,
                   "--suite-dir", args.suite_dir, "--model", args.model,
                   "--output", root / "divergence/final-vs-accepted.json"])


if __name__ == "__main__":
    main()
