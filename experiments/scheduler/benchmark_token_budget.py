#!/usr/bin/env python3
"""Sweep the static mixed prefill/decode token-budget frontier on one loaded model."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

_EXPERIMENTS = Path(__file__).resolve().parents[1]
_ROOT = _EXPERIMENTS.parent
_BENCHMARKS = _ROOT / "benchmarks"
for path in (_BENCHMARKS, _ROOT / "baseline", _ROOT / "engine" / "kvcache",
             _ROOT / "engine" / "scheduler", _ROOT / "engine" / "graph"):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from benchmark_backends import SchedulerBackend
from benchmark_core import make_synthetic_workload, median_aggregate, parse_int_list
from run_benchmarks import system_metadata, validate_run


def parse_float_list(value):
    values = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not values or any(item < 0 for item in values):
        raise ValueError("expected a comma-separated list of non-negative numbers")
    return values


def parse_optional_int_list(value):
    values = []
    for part in value.split(","):
        item = part.strip().lower()
        if not item:
            continue
        values.append(None if item in ("none", "off") else int(item))
    if not values or any(item is not None and item < 1 for item in values):
        raise ValueError("expected positive integers or 'none'")
    return values


def matched_prefix(left, right):
    for index, (actual, expected) in enumerate(zip(left, right)):
        if actual != expected:
            return index
    return min(len(left), len(right))


def metric(summary, name, statistic="median"):
    value = summary.get(name)
    return value.get(statistic) if isinstance(value, dict) else value


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--implementation", choices=("continuous-batching", "custom-kernels"),
                        default="custom-kernels")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--prompt-lengths", default="128,512,2048,8192")
    parser.add_argument("--request-counts", default="1,4,16")
    parser.add_argument("--request-rates", default="0,20",
                        help="zero is burst; positive values are Poisson requests/second")
    parser.add_argument("--max-num-batched-tokens", default="512,1024,2048,4096,8192,16384")
    parser.add_argument(
        "--max-prefill-attention-pairs",
        default="none",
        help=(
            "comma-separated prefill attention-work ceilings in causal query-key "
            "pairs; use 'none' for the token-only baseline"
        ),
    )
    parser.add_argument("--output-length", type=int, default=32)
    parser.add_argument("--max-running", type=int, default=16)
    parser.add_argument("--num-blocks", type=int)
    parser.add_argument("--max-prefill-chunk-size", type=int)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path,
                        default=_EXPERIMENTS / "results" / "token-budget")
    return parser


def main():
    args = build_parser().parse_args()
    prompt_lengths = parse_int_list(args.prompt_lengths)
    request_counts = parse_int_list(args.request_counts)
    request_rates = parse_float_list(args.request_rates)
    budgets = parse_int_list(args.max_num_batched_tokens)
    pair_budgets = parse_optional_int_list(args.max_prefill_attention_pairs)
    if args.output_length < 1 or args.max_running < max(request_counts):
        raise ValueError("output-length must be positive and max-running must fit request-counts")
    if min(budgets) < args.max_running:
        raise ValueError("every token budget must be at least max-running")
    if args.warmups < 0 or args.repetitions < 1:
        raise ValueError("warmups must be non-negative and repetitions must be positive")

    # Size the shared model/cache once for the largest case, then mutate only the
    # scheduler's scalar budget between matched runs.
    capacity_workload = make_synthetic_workload(
        name="token-budget-capacity",
        num_requests=max(request_counts),
        prompt_lengths=[max(prompt_lengths)],
        output_lengths=[args.output_length],
        vocab_size=151936,
        seed=args.seed,
    )
    backend = SchedulerBackend(
        name=args.implementation,
        workload=capacity_workload,
        model_id=args.model,
        device=args.device,
        dtype=args.dtype,
        block_size=args.block_size,
        max_running=args.max_running,
        num_blocks=args.num_blocks,
        max_num_batched_tokens=max(budgets),
        max_prefill_chunk_size=args.max_prefill_chunk_size,
        max_prefill_attention_pairs=None,
    )
    if backend.scheduler.use_graph:
        raise AssertionError("the token-budget frontier must use the fully eager executor")

    rows = []
    detailed = []
    try:
        for prompt_length in prompt_lengths:
            for request_count in request_counts:
                for request_rate in request_rates:
                    workload = make_synthetic_workload(
                        name=(f"L{prompt_length}-R{request_count}-rate{request_rate:g}"),
                        num_requests=request_count,
                        prompt_lengths=[prompt_length],
                        output_lengths=[args.output_length],
                        vocab_size=151936,
                        seed=args.seed + prompt_length * 1009 + request_count,
                        request_rate=request_rate,
                    )
                    reference = None
                    for budget in budgets:
                      for pair_budget in pair_budgets:
                        print(
                            f"L={prompt_length} requests={request_count} "
                            f"rate={request_rate:g} token_budget={budget} "
                            f"attention_pairs={pair_budget}"
                        )
                        backend.scheduler.max_num_batched_tokens = budget
                        backend.scheduler.max_prefill_attention_pairs = pair_budget
                        for _ in range(args.warmups):
                            validate_run(backend.run(workload), workload)
                        runs = [backend.run(workload) for _ in range(args.repetitions)]
                        for run in runs:
                            validate_run(run, workload)
                            assert run.metadata["execution_mode"] == "fully-eager"
                            assert max(run.metadata["iteration_token_counts"], default=0) <= budget
                            if pair_budget is not None:
                                assert all(
                                    pairs <= pair_budget
                                    for kind, pairs in zip(
                                        run.metadata["iteration_kinds"],
                                        run.metadata["iteration_prefill_attention_pairs"],
                                    )
                                    if kind == "mixed"
                                )
                        summary = median_aggregate(runs)
                        outputs = {
                            trace.request_id: trace.output_ids for trace in runs[-1].traces
                        }
                        if reference is None:
                            reference = outputs
                        prefixes = [
                            matched_prefix(outputs[request_id], reference[request_id])
                            for request_id in outputs
                        ]
                        exact = sum(
                            outputs[request_id] == reference[request_id]
                            for request_id in outputs
                        )
                        iteration_counts = runs[-1].metadata["iteration_type_counts"]
                        scheduled_counts = runs[-1].metadata["iteration_token_counts"]
                        row = {
                            "execution_mode": "fully-eager",
                            "prompt_length": prompt_length,
                            "request_count": request_count,
                            "request_rate": request_rate,
                            "arrival_pattern": workload.arrival_pattern,
                            "max_num_batched_tokens": budget,
                            "max_prefill_attention_pairs": pair_budget,
                            "request_throughput_rps": summary["request_throughput_rps"],
                            "output_throughput_tok_s": summary["output_throughput_tok_s"],
                            "total_throughput_tok_s": summary["total_throughput_tok_s"],
                            "ttft_p50_ms": metric(summary, "ttft_ms"),
                            "ttft_p95_ms": metric(summary, "ttft_ms", "p95"),
                            "ttft_p99_ms": metric(summary, "ttft_ms", "p99"),
                            "tpot_p50_ms": metric(summary, "tpot_ms"),
                            "tpot_p95_ms": metric(summary, "tpot_ms", "p95"),
                            "tpot_p99_ms": metric(summary, "tpot_ms", "p99"),
                            "itl_p95_ms": metric(summary, "itl_ms", "p95"),
                            "itl_p99_ms": metric(summary, "itl_ms", "p99"),
                            "e2e_p95_ms": metric(summary, "e2e_ms", "p95"),
                            "exact_fraction_vs_first_budget": exact / request_count,
                            "min_matched_prefix_vs_first_budget": min(prefixes),
                            "decode_only_iterations": iteration_counts["decode_only"],
                            "prefill_only_iterations": iteration_counts["prefill_only"],
                            "mixed_iterations": iteration_counts["mixed"],
                            "mean_token_budget_utilization": (
                                sum(scheduled_counts) / (len(scheduled_counts) * budget)
                                if scheduled_counts else None
                            ),
                            "mean_mixed_attention_pair_budget_utilization": None,
                        }
                        mixed_pair_counts = [
                            pairs
                            for kind, pairs in zip(
                                runs[-1].metadata["iteration_kinds"],
                                runs[-1].metadata["iteration_prefill_attention_pairs"],
                            )
                            if kind == "mixed"
                        ]
                        if pair_budget is not None and mixed_pair_counts:
                            row["mean_mixed_attention_pair_budget_utilization"] = (
                                sum(mixed_pair_counts)
                                / (len(mixed_pair_counts) * pair_budget)
                            )
                        rows.append(row)
                        detailed.append({
                            "case": row,
                            "workload": workload.to_dict(),
                            "runs": [run.to_dict() for run in runs],
                        })
    finally:
        backend.close()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    json_path = args.output_dir / f"token-budget-{stamp}.json"
    csv_path = args.output_dir / f"token-budget-{stamp}.csv"
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "configuration": vars(args) | {
            "output_dir": str(args.output_dir),
            "prompt_lengths": prompt_lengths,
            "request_counts": request_counts,
            "request_rates": request_rates,
            "max_num_batched_tokens": budgets,
            "max_prefill_attention_pairs": pair_budgets,
        },
        "system": system_metadata(),
        "cases": detailed,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
