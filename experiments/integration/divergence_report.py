#!/usr/bin/env python3
"""Explain greedy-token differences between two eight-cell output sets.

For every request whose tokens differ, find the first diverging position k. The
two engines agree on the prompt and on tokens [0, k), so at k they saw the same
context and chose different tokens. With --reference fp32 an independent FP32
Transformers forward over that shared context scores both choices:

    margin = logit_fp32[left token] - logit_fp32[right token]

|margin| <= --tie-margin is a near tie: FP16 accumulation order legitimately
decides it and neither engine is wrong. A clearly negative margin means the
left engine picked a worse token than the right one did (look for a bug on the
left); clearly positive, the reverse. Without a reference only the CPU part
runs: counts and first-divergence positions.

Left is the local engine of --results-dir. Right is vLLM from the same
directory, or with --against the local engine of another results directory
(e.g. fused epilogues vs the unfused baseline).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for directory in (HERE, ROOT / "baseline", ROOT / "benchmarks"):
    sys.path.insert(0, str(directory))

from fixed_regime import FACTORIAL_SHAPES  # noqa: E402

SHAPES = tuple(row["id"] for row in FACTORIAL_SHAPES)
SCENARIOS = ("burst", "mixed")


def local_outputs(directory, scenario, shape):
    path = directory / shape / "local.json" if scenario == "burst" else directory / "mixed" / shape / "local.json"
    if not path.is_file():
        return None
    return {str(key): value for key, value in json.loads(path.read_text())["runs"][0]["outputs"].items()}


def vllm_outputs(directory, scenario, shape):
    if scenario == "burst":
        files = sorted((directory / shape / "vllm").glob("*.json"))
        if len(files) != 1:
            return None
        runs = json.loads(files[0].read_text())["backends"]["vllm"]["runs"]
        return {str(row["request_id"]): row["output_ids"] for row in runs[-1]["requests"]}
    path = directory / "mixed" / shape / "vllm.json"
    if not path.is_file():
        return None
    return {str(key): value for key, value in json.loads(path.read_text())["runs"][0]["outputs"].items()}


def prompts(suite_dir, shape):
    workload = json.loads((suite_dir / shape / "workload.json").read_text())
    return {str(row["request_id"]): row["prompt_ids"] for row in workload["requests"]}


def first_divergence(left, right):
    return next((index for index, (a, b) in enumerate(zip(left, right)) if a != b),
                None if len(left) == len(right) else min(len(left), len(right)))


def divergences(args):
    """[(scenario, shape, request, k, prompt, left, right)] plus per-cell counts; CPU only."""
    rows, cells = [], {}
    for scenario in args.scenarios:
        for shape in args.shape_ids:
            left = local_outputs(args.results_dir, scenario, shape)
            right = (local_outputs(args.against, scenario, shape) if args.against
                     else vllm_outputs(args.results_dir, scenario, shape))
            if left is None or right is None:
                continue
            if set(left) != set(right):
                raise ValueError(f"{scenario} {shape}: the two output sets cover different requests")
            prompt = prompts(args.suite_dir, shape)
            positions = []
            for request in sorted(left, key=int):
                k = first_divergence(left[request], right[request])
                if k is None:
                    continue
                positions.append(k)
                rows.append((scenario, shape, request, k, prompt[request], left[request], right[request]))
            cells[f"{scenario}/{shape}"] = {
                "requests": len(left), "divergent": len(positions),
                "first_divergence_min": min(positions, default=None),
                "first_divergence_median": statistics.median(positions) if positions else None,
                "output_length": len(next(iter(left.values())))}
    return rows, cells


def score(rows, args):
    """FP32 Transformers logits at each first divergence; TF32 off so FP32 is FP32."""
    import torch
    from transformers import AutoModelForCausalLM
    from benchmark_latest_vs_vllm import resolve_model_source

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(
        resolve_model_source(args), torch_dtype=torch.float32).to(args.device).eval()
    scored, cache = [], {}
    with torch.no_grad():
        for index, (scenario, shape, request, k, prompt, left, right) in enumerate(rows, 1):
            # o128/o256 and burst/mixed share prompts, so one context often recurs.
            key = tuple(prompt + left[:k])
            if key not in cache:
                context = torch.tensor([list(key)], device=args.device)
                hidden = model.model(input_ids=context).last_hidden_state[:, -1]
                cache[key] = model.lm_head(hidden)[0].float()
            logits = cache[key]
            ours, theirs = left[k], right[k]
            ranks = logits.argsort(descending=True)
            top2 = logits.topk(2).values
            margin = float(logits[ours] - logits[theirs])
            scored.append({"scenario": scenario, "shape": shape, "request": request, "position": k,
                           "left_token": ours, "right_token": theirs, "fp32_margin": margin,
                           "fp32_top_token": int(ranks[0]), "fp32_top2_gap": float(top2[0] - top2[1]),
                           "left_rank": int((ranks == ours).nonzero()[0]),
                           "right_rank": int((ranks == theirs).nonzero()[0]),
                           "class": ("near_tie" if abs(margin) <= args.tie_margin else
                                     "left_worse" if margin < 0 else "right_worse")})
            if index % 25 == 0:
                print(f"scored {index}/{len(rows)}", flush=True)
    return scored


def summarize(cells, scored, args):
    by_cell = {}
    for row in scored:
        by_cell.setdefault(f"{row['scenario']}/{row['shape']}", []).append(row)
    for key, cell in cells.items():
        rows = by_cell.get(key, [])
        if rows:
            cell.update({kind: sum(row["class"] == kind for row in rows)
                         for kind in ("near_tie", "left_worse", "right_worse")})
            cell["max_abs_fp32_margin"] = max(abs(row["fp32_margin"]) for row in rows)
            cell["left_is_fp32_top"] = sum(row["left_token"] == row["fp32_top_token"] for row in rows)
            cell["right_is_fp32_top"] = sum(row["right_token"] == row["fp32_top_token"] for row in rows)
    verdict = None
    if scored:
        worse = sum(row["class"] == "left_worse" for row in scored)
        verdict = ("all divergences are near ties" if all(row["class"] == "near_tie" for row in scored)
                   else f"{worse} divergences where the left engine chose a clearly worse token"
                   if worse else "no left-side regressions; some right-side non-ties")
    return {"left": f"{args.results_dir} local",
            "right": f"{args.against} local" if args.against else f"{args.results_dir} vllm",
            "reference": args.reference, "tie_margin": args.tie_margin,
            "verdict": verdict, "cells": cells, "divergences": scored}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="an eight-cell output directory (e.g. .../attention-rerun-4XR8Oi/eight-fa3)")
    parser.add_argument("--against", type=Path,
                        help="compare against this directory's local engine instead of vLLM")
    parser.add_argument("--suite-dir", type=Path,
                        default=ROOT / "experiments/results/full-checkpoint-20260916T033540Z")
    parser.add_argument("--shape-ids", nargs="+", choices=SHAPES, default=list(SHAPES))
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--reference", choices=("none", "fp32"), default="fp32")
    parser.add_argument("--tie-margin", type=float, default=0.25)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, help="report JSON (default: <results-dir>/divergence.json)")
    args = parser.parse_args()
    rows, cells = divergences(args)
    if not cells:
        parser.error(f"no comparable output pairs under {args.results_dir}")
    scored = score(rows, args) if args.reference == "fp32" and rows else []
    report = summarize(cells, scored, args)
    output = args.output or args.results_dir / ("divergence.json" if args.reference == "fp32"
                                                else "divergence-positions.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{'cell':34}{'diverge':>9}{'first k':>9}" + ("" if not scored else
          f"{'tie':>6}{'L worse':>9}{'R worse':>9}{'max|m|':>8}"))
    for key, cell in cells.items():
        line = (f"{key:34}{cell['divergent']:>4}/{cell['requests']:<4}"
                f"{'-' if cell['first_divergence_min'] is None else cell['first_divergence_min']:>9}")
        if "near_tie" in cell:
            line += (f"{cell['near_tie']:>6}{cell['left_worse']:>9}{cell['right_worse']:>9}"
                     f"{cell['max_abs_fp32_margin']:>8.3f}")
        print(line)
    if report["verdict"]:
        print(f"verdict: {report['verdict']}")
    print(f"report={output}")


if __name__ == "__main__":
    main()
