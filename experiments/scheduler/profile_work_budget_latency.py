#!/usr/bin/env python3
"""Profile decode-bearing iteration latency across prefill attention-work budgets.

The raw CSV contains one row per scheduler iteration, aligned with the exact decode
and prefill metadata used to plan that iteration. Decode-latency distributions weight
each decode-bearing iteration by its active decode count, matching the number of
output-token intervals exposed to that latency.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
ROOT = EXPERIMENTS.parent
BENCHMARKS = ROOT / "benchmarks"
for path in (
    BENCHMARKS,
    HERE,
    ROOT / "baseline",
    ROOT / "engine" / "kvcache",
    ROOT / "engine" / "scheduler",
    ROOT / "engine" / "graph",
):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from benchmark_backends import SchedulerBackend
from benchmark_core import make_synthetic_workload, median_aggregate, percentile
from benchmark_token_budget import matched_prefix, parse_optional_int_list
from run_benchmarks import system_metadata, validate_run


def parse_positive_float_list(value: str) -> list[float]:
    values = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not values or any(item <= 0 for item in values):
        raise ValueError("expected a comma-separated list of positive numbers")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument(
        "--implementation",
        choices=("continuous-batching", "custom-kernels"),
        default="custom-kernels",
    )
    parser.add_argument("--prompt-length", type=int, default=8192)
    parser.add_argument("--request-count", type=int, default=16)
    parser.add_argument("--request-rate", type=float, default=20.0)
    parser.add_argument("--output-length", type=int, default=64)
    parser.add_argument("--max-running", type=int, default=16)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--max-prefill-chunk-size", type=int, default=2048)
    parser.add_argument(
        "--max-prefill-attention-pairs",
        default=(
            "none,67108864,50331648,33554432,25165824,16777216,"
            "12582912,8388608,6291456,4194304,3145728,2097152,1048576"
        ),
    )
    parser.add_argument(
        "--latency-thresholds-ms",
        default="40,50,75,100,150",
        help="thresholds used for weighted decode-interval violation rates",
    )
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-blocks", type=int)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXPERIMENTS / "results" / "work-budget-latency",
    )
    return parser


def _mean(values):
    return statistics.fmean(values) if values else None


def _serialized(values):
    return json.dumps(values, separators=(",", ":"))


def _iteration_rows(run, cap, repetition):
    metadata = run.metadata
    fields = (
        metadata["iteration_wall_ms"],
        metadata["iteration_kinds"],
        metadata["iteration_decode_token_counts"],
        metadata["iteration_decode_context_lengths"],
        metadata["iteration_prefill_token_counts"],
        metadata["iteration_prefill_attention_pairs"],
        metadata["iteration_prefill_prefix_lengths"],
        metadata["iteration_token_counts"],
    )
    if len({len(values) for values in fields}) != 1:
        raise AssertionError("iteration profiler metadata is not aligned")

    rows = []
    for index, values in enumerate(zip(*fields)):
        (
            latency_ms,
            kind,
            decode_count,
            decode_contexts,
            prefill_tokens,
            prefill_pairs,
            prefill_prefixes,
            total_tokens,
        ) = values
        rows.append({
            "max_prefill_attention_pairs": cap,
            "repetition": repetition,
            "iteration": index,
            "iteration_type": kind,
            "iteration_latency_ms": latency_ms,
            "decode_count": decode_count,
            "decode_context_lengths": _serialized(decode_contexts),
            "decode_context_min": min(decode_contexts, default=None),
            "decode_context_mean": _mean(decode_contexts),
            "decode_context_max": max(decode_contexts, default=None),
            "prefill_tokens": prefill_tokens,
            "prefill_attention_pairs": prefill_pairs,
            "prefill_prefix_lengths": _serialized(prefill_prefixes),
            "prefill_prefix_min": min(prefill_prefixes, default=None),
            "prefill_prefix_mean": _mean(prefill_prefixes),
            "prefill_prefix_max": max(prefill_prefixes, default=None),
            "total_scheduled_tokens": total_tokens,
        })
    return rows


def _metric(summary, name, statistic):
    return summary[name][statistic]


def _summary_row(cap, runs, iteration_rows, thresholds, reference):
    summary = median_aggregate(runs)
    decode_rows = [row for row in iteration_rows if row["decode_count"]]
    weighted_latencies = [
        row["iteration_latency_ms"]
        for row in decode_rows
        for _ in range(row["decode_count"])
    ]
    outputs = {
        trace.request_id: trace.output_ids for trace in runs[-1].traces
    }
    prefixes = [
        matched_prefix(outputs[request_id], reference[request_id])
        for request_id in outputs
    ]
    row = {
        "max_prefill_attention_pairs": cap,
        "wall_time_s": summary["wall_time_s"],
        "request_throughput_rps": summary["request_throughput_rps"],
        "input_throughput_tok_s": (
            sum(trace.prompt_tokens for trace in runs[-1].traces)
            / summary["wall_time_s"]
        ),
        "output_throughput_tok_s": summary["output_throughput_tok_s"],
        "total_throughput_tok_s": summary["total_throughput_tok_s"],
        "ttft_p95_ms": _metric(summary, "ttft_ms", "p95"),
        "ttft_p99_ms": _metric(summary, "ttft_ms", "p99"),
        "tpot_p95_ms": _metric(summary, "tpot_ms", "p95"),
        "tpot_p99_ms": _metric(summary, "tpot_ms", "p99"),
        "trace_itl_p95_ms": _metric(summary, "itl_ms", "p95"),
        "trace_itl_p99_ms": _metric(summary, "itl_ms", "p99"),
        "decode_interval_count": len(weighted_latencies),
        "decode_iteration_count": len(decode_rows),
        "decode_iteration_latency_p50_ms": percentile(weighted_latencies, 50),
        "decode_iteration_latency_p90_ms": percentile(weighted_latencies, 90),
        "decode_iteration_latency_p95_ms": percentile(weighted_latencies, 95),
        "decode_iteration_latency_p99_ms": percentile(weighted_latencies, 99),
        "decode_iteration_latency_max_ms": max(weighted_latencies, default=None),
        "exact_fraction_vs_uncapped": sum(
            outputs[request_id] == reference[request_id] for request_id in outputs
        ) / len(outputs),
        "min_matched_prefix_vs_uncapped": min(prefixes),
    }
    for threshold in thresholds:
        label = f"decode_interval_fraction_over_{threshold:g}ms"
        row[label] = (
            sum(value > threshold for value in weighted_latencies)
            / len(weighted_latencies)
            if weighted_latencies else None
        )
        row[f"decode_slo_goodput_tok_s_at_{threshold:g}ms"] = (
            summary["output_throughput_tok_s"] * (1 - row[label])
            if row[label] is not None else None
        )
    return row


def main() -> None:
    args = build_parser().parse_args()
    caps = parse_optional_int_list(args.max_prefill_attention_pairs)
    thresholds = parse_positive_float_list(args.latency_thresholds_ms)
    if args.prompt_length < 1 or args.request_count < 1 or args.output_length < 2:
        raise ValueError("prompt/request counts must be positive and output-length >= 2")
    if args.request_rate < 0 or args.max_running < args.request_count:
        raise ValueError("request-rate must be non-negative and max-running fit requests")
    if args.warmups < 0 or args.repetitions < 1:
        raise ValueError("warmups must be non-negative and repetitions positive")

    workload = make_synthetic_workload(
        name="work-budget-latency",
        num_requests=args.request_count,
        prompt_lengths=[args.prompt_length],
        output_lengths=[args.output_length],
        vocab_size=151936,
        seed=args.seed,
        request_rate=args.request_rate,
    )
    backend = SchedulerBackend(
        name=args.implementation,
        workload=workload,
        model_id=args.model,
        device=args.device,
        dtype=args.dtype,
        block_size=args.block_size,
        max_running=args.max_running,
        num_blocks=args.num_blocks,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_prefill_chunk_size=args.max_prefill_chunk_size,
        max_prefill_attention_pairs=None,
    )
    if backend.scheduler.use_graph:
        raise AssertionError("work-budget latency profiling requires eager execution")

    all_iterations = []
    summaries = []
    detailed = []
    reference = None
    try:
        for cap in caps:
            print(f"attention_pairs={cap}")
            backend.scheduler.max_prefill_attention_pairs = cap
            for _ in range(args.warmups):
                validate_run(backend.run(workload), workload)
            runs = [backend.run(workload) for _ in range(args.repetitions)]
            for run in runs:
                validate_run(run, workload)
                if cap is not None:
                    assert all(
                        pairs <= cap
                        for kind, pairs in zip(
                            run.metadata["iteration_kinds"],
                            run.metadata["iteration_prefill_attention_pairs"],
                        )
                        if kind == "mixed"
                    )
            outputs = {
                trace.request_id: trace.output_ids for trace in runs[-1].traces
            }
            if reference is None:
                reference = outputs
            cap_iterations = []
            for repetition, run in enumerate(runs):
                cap_iterations.extend(_iteration_rows(run, cap, repetition))
            all_iterations.extend(cap_iterations)
            summary = _summary_row(
                cap, runs, cap_iterations, thresholds, reference
            )
            summaries.append(summary)
            detailed.append({
                "summary": summary,
                "runs": [run.to_dict() for run in runs],
            })
    finally:
        backend.close()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    stem = f"work-budget-latency-{stamp}"
    summary_csv = args.output_dir / f"{stem}-summary.csv"
    iterations_csv = args.output_dir / f"{stem}-iterations.csv"
    json_path = args.output_dir / f"{stem}.json"
    with summary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    with iterations_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_iterations[0]))
        writer.writeheader()
        writer.writerows(all_iterations)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "configuration": vars(args) | {
            "output_dir": str(args.output_dir),
            "max_prefill_attention_pairs": caps,
            "latency_thresholds_ms": thresholds,
        },
        "system": system_metadata(),
        "workload": workload.to_dict(),
        "cases": detailed,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {summary_csv}")
    print(f"wrote {iterations_csv}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
