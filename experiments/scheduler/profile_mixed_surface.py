#!/usr/bin/env python3
"""Measure the decode/prefill execution surface with matched eager controls.

For every mixed point ``T(D, P)``, this profiler measures the corresponding
decode-only ``T(D, 0)`` and prefill-only ``T(0, P)`` iterations on the same loaded
model. Decode histories and prefill prefixes are staged directly in paged KV outside
the timed region; the target iteration executes the real scheduler/model kernels.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
import time


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
ROOT = EXPERIMENTS.parent
BENCHMARKS = ROOT / "benchmarks"
for path in (
    BENCHMARKS,
    ROOT / "baseline",
    ROOT / "engine" / "kvcache",
    ROOT / "engine" / "model_runner",
    ROOT / "engine" / "scheduler",
    ROOT / "engine" / "graph",
):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from benchmark_core import parse_int_list
from run_benchmarks import system_metadata
from run_phase_sweep import make_engine


CSV_FIELDS = (
    "implementation",
    "decode_requests",
    "decode_context_length",
    "prefill_requests",
    "prefill_tokens",
    "prefill_chunk_size",
    "prefill_prefix_length",
    "decode_only_median_ms",
    "prefill_only_median_ms",
    "mixed_median_ms",
    "decode_only_median_wall_ms",
    "prefill_only_median_wall_ms",
    "mixed_median_wall_ms",
    "incremental_prefill_ms",
    "incremental_decode_ms",
    "decode_stretch",
    "separate_sum_ms",
    "packing_benefit_ms",
    "packing_speedup",
    "wall_incremental_prefill_ms",
    "wall_decode_stretch",
    "wall_packing_benefit_ms",
    "wall_packing_speedup",
    "mixed_scheduled_tokens_per_second",
    "mixed_peak_gpu_memory_bytes",
    "status",
    "error",
)


def parse_nonnegative_int_list(value: str) -> list[int]:
    parsed = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not parsed or any(item < 0 for item in parsed):
        raise ValueError("expected a comma-separated list of non-negative integers")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument(
        "--implementation",
        choices=("continuous-batching", "custom-kernels"),
        default="custom-kernels",
    )
    parser.add_argument("--decode-requests", default="1,8,32,64")
    parser.add_argument("--decode-context-lengths", default="128,2048,8192")
    parser.add_argument("--prefill-tokens", default="512,2048,4096,8192")
    parser.add_argument("--prefill-prefix-lengths", default="0,4096,16384")
    parser.add_argument("--prefill-requests", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXPERIMENTS / "results" / "mixed-surface",
    )
    return parser


def validate_surface(
    *,
    decode_requests: list[int],
    decode_context_lengths: list[int],
    prefill_tokens: list[int],
    prefill_prefix_lengths: list[int],
    prefill_requests: int,
    block_size: int,
    warmups: int,
    repetitions: int,
    max_seq_len: int = 32768,
) -> None:
    named = {
        "decode requests": decode_requests,
        "decode context lengths": decode_context_lengths,
        "prefill tokens": prefill_tokens,
    }
    for label, values in named.items():
        if not values or any(value < 1 for value in values):
            raise ValueError(f"{label} must be positive")
    if not prefill_prefix_lengths or any(value < 0 for value in prefill_prefix_lengths):
        raise ValueError("prefill prefix lengths must be non-negative")
    if prefill_requests < 1 or block_size < 1:
        raise ValueError("prefill-requests and block-size must be positive")
    indivisible = [value for value in prefill_tokens if value % prefill_requests]
    if indivisible:
        raise ValueError(
            f"prefill token totals {indivisible} are not divisible by "
            f"prefill_requests={prefill_requests}"
        )
    if max(decode_context_lengths) + 1 > max_seq_len:
        raise ValueError("decode context plus one target token exceeds max_seq_len")
    invalid = [
        (prefix, total)
        for prefix in prefill_prefix_lengths
        for total in prefill_tokens
        if prefix + total // prefill_requests > max_seq_len
    ]
    if invalid:
        raise ValueError(f"prefill prefix/chunk combinations exceed max_seq_len: {invalid}")
    if warmups < 0 or repetitions < 1:
        raise ValueError("warmups must be non-negative and repetitions must be positive")


def planned_mixed_cases(
    decode_requests: list[int],
    decode_context_lengths: list[int],
    prefill_tokens: list[int],
    prefill_prefix_lengths: list[int],
) -> list[tuple[int, int, int, int]]:
    return [
        (requests, context, tokens, prefix)
        for requests in decode_requests
        for context in decode_context_lengths
        for tokens in prefill_tokens
        for prefix in prefill_prefix_lengths
    ]


def derive_tradeoff(mixed_ms: float, decode_ms: float, prefill_ms: float) -> dict:
    if min(mixed_ms, decode_ms, prefill_ms) <= 0:
        raise ValueError("latencies must be positive")
    separate = decode_ms + prefill_ms
    return {
        "incremental_prefill_ms": mixed_ms - decode_ms,
        "incremental_decode_ms": mixed_ms - prefill_ms,
        "decode_stretch": mixed_ms / decode_ms,
        "separate_sum_ms": separate,
        "packing_benefit_ms": separate - mixed_ms,
        "packing_speedup": separate / mixed_ms,
    }


def _blocks(length: int, block_size: int) -> int:
    return (length + block_size - 1) // block_size


def _summary(raw_cuda_ms, raw_wall_ms, scheduled_tokens, peak_memory):
    median_cuda = statistics.median(raw_cuda_ms)
    return {
        "raw_cuda_ms": raw_cuda_ms,
        "raw_wall_ms": raw_wall_ms,
        "median_ms": median_cuda,
        "median_wall_ms": statistics.median(raw_wall_ms),
        "min_ms": min(raw_cuda_ms),
        "max_ms": max(raw_cuda_ms),
        "scheduled_tokens": scheduled_tokens,
        "scheduled_tokens_per_second": scheduled_tokens / (median_cuda * 1e-3),
        "peak_gpu_memory_bytes": peak_memory,
        "status": "ok",
        "error": None,
    }


def main() -> None:
    args = build_parser().parse_args()
    decode_requests = parse_int_list(args.decode_requests)
    decode_context_lengths = parse_int_list(args.decode_context_lengths)
    prefill_tokens = parse_int_list(args.prefill_tokens)
    prefill_prefix_lengths = parse_nonnegative_int_list(args.prefill_prefix_lengths)
    validate_surface(
        decode_requests=decode_requests,
        decode_context_lengths=decode_context_lengths,
        prefill_tokens=prefill_tokens,
        prefill_prefix_lengths=prefill_prefix_lengths,
        prefill_requests=args.prefill_requests,
        block_size=args.block_size,
        warmups=args.warmups,
        repetitions=args.repetitions,
    )

    import torch
    from scheduler import Scheduler, Status

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("a CUDA device is required")

    engine, model_load_seconds = make_engine(
        args.model, args.implementation, args.device, args.dtype, args.block_size
    )
    validate_surface(
        decode_requests=decode_requests,
        decode_context_lengths=decode_context_lengths,
        prefill_tokens=prefill_tokens,
        prefill_prefix_lengths=prefill_prefix_lengths,
        prefill_requests=args.prefill_requests,
        block_size=args.block_size,
        warmups=args.warmups,
        repetitions=args.repetitions,
        max_seq_len=engine.cfg.max_seq_len,
    )
    max_decode = max(decode_requests)
    max_context = max(decode_context_lengths)
    max_prefill_total = max(prefill_tokens)
    max_chunk = max_prefill_total // args.prefill_requests
    max_prefix = max(prefill_prefix_lengths)
    max_running = max_decode + args.prefill_requests
    num_blocks = (
        max_decode * _blocks(max_context + 1, args.block_size)
        + args.prefill_requests * _blocks(max_prefix + max_chunk, args.block_size)
        + max_running
    )
    scheduler = Scheduler(
        engine.model,
        engine.cfg,
        max_running=max_running,
        num_blocks=num_blocks,
        block_size=args.block_size,
        eos_ids=set(),
        device=args.device,
        dtype=engine.dtype,
        use_graph=False,
        profile_prefill=False,
        max_num_batched_tokens=max_decode + max_prefill_total,
    )
    if scheduler.use_graph:
        raise AssertionError("mixed execution-surface profiling must be fully eager")

    token = args.seed % (engine.cfg.vocab - 1) + 1
    prompt_cache: dict[int, list[int]] = {}

    def prompt(length: int) -> list[int]:
        if length not in prompt_cache:
            prompt_cache[length] = [token] * length
        return prompt_cache[length]

    def stage(kind, requests, context, total_prefill, prefix):
        scheduler.profile_prefill = False
        scheduler.reset()
        chunk = total_prefill // args.prefill_requests if total_prefill else 0
        decode_slots = []
        prefill_slots = []

        if kind in ("decode_only", "mixed"):
            for _ in range(requests):
                scheduler.add_request(prompt(context), max_tokens=2)
            decode_slots = scheduler._admit_waiting(requests)

        if kind in ("prefill_only", "mixed"):
            for _ in range(args.prefill_requests):
                scheduler.add_request(prompt(prefix + chunk), max_tokens=1)
            prefill_slots = scheduler._admit_waiting(args.prefill_requests)

        n_news = [0] * scheduler.max_running
        for slot in decode_slots:
            n_news[slot] = context
        for slot in prefill_slots:
            n_news[slot] = prefix
        scheduler.cache.allocate_block(n_news)

        for slot in decode_slots:
            request = scheduler.prefilling.pop(slot)
            request.num_prompt_tokens_computed = context
            request.output_ids.append(token)
            request.status = Status.RUNNING
            scheduler.running[slot] = request
        for slot in prefill_slots:
            scheduler.prefilling[slot].num_prompt_tokens_computed = prefix

        expected_decode = requests if kind in ("decode_only", "mixed") else 0
        expected_prefill = total_prefill if kind in ("prefill_only", "mixed") else 0
        expected_kind = kind
        scheduler.max_num_batched_tokens = expected_decode + expected_prefill
        scheduler.iteration_decode_token_counts.clear()
        scheduler.iteration_prefill_token_counts.clear()
        scheduler.iteration_token_counts.clear()
        scheduler.iteration_kinds.clear()
        torch.cuda.synchronize()
        return expected_kind, expected_decode, expected_prefill

    def target(expected):
        scheduler.step()
        observed = (
            scheduler.iteration_kinds[-1],
            scheduler.iteration_decode_token_counts[-1],
            scheduler.iteration_prefill_token_counts[-1],
        )
        if observed != expected:
            raise AssertionError(f"scheduled {observed}, expected {expected}")

    def measure(kind, requests=0, context=0, total_prefill=0, prefix=0):
        expected = stage(kind, requests, context, total_prefill, prefix)
        target(expected)
        for _ in range(args.warmups):
            expected = stage(kind, requests, context, total_prefill, prefix)
            target(expected)
        torch.cuda.reset_peak_memory_stats()
        raw_cuda_ms = []
        raw_wall_ms = []
        for _ in range(args.repetitions):
            expected = stage(kind, requests, context, total_prefill, prefix)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            started = time.perf_counter()
            start_event.record()
            target(expected)
            end_event.record()
            end_event.synchronize()
            raw_wall_ms.append((time.perf_counter() - started) * 1e3)
            raw_cuda_ms.append(float(start_event.elapsed_time(end_event)))
        scheduled = (
            (requests if kind in ("decode_only", "mixed") else 0)
            + (total_prefill if kind in ("prefill_only", "mixed") else 0)
        )
        return _summary(
            raw_cuda_ms,
            raw_wall_ms,
            scheduled,
            int(torch.cuda.max_memory_allocated()),
        )

    decode_controls = {}
    prefill_controls = {}
    detailed = []
    rows = []

    with torch.no_grad():
        for requests in decode_requests:
            for context in decode_context_lengths:
                print(f"decode-only D={requests} C={context}", flush=True)
                decode_controls[(requests, context)] = measure(
                    "decode_only", requests=requests, context=context
                )

        for total in prefill_tokens:
            for prefix in prefill_prefix_lengths:
                print(
                    f"prefill-only P={total} prefix={prefix} "
                    f"requests={args.prefill_requests}",
                    flush=True,
                )
                prefill_controls[(total, prefix)] = measure(
                    "prefill_only", total_prefill=total, prefix=prefix
                )

        cases = planned_mixed_cases(
            decode_requests,
            decode_context_lengths,
            prefill_tokens,
            prefill_prefix_lengths,
        )
        for index, (requests, context, total, prefix) in enumerate(cases, 1):
            print(
                f"[{index}/{len(cases)}] mixed D={requests} C={context} "
                f"P={total} prefix={prefix}",
                flush=True,
            )
            try:
                mixed = measure(
                    "mixed",
                    requests=requests,
                    context=context,
                    total_prefill=total,
                    prefix=prefix,
                )
                decode = decode_controls[(requests, context)]
                prefill = prefill_controls[(total, prefix)]
                derived = derive_tradeoff(
                    mixed["median_ms"], decode["median_ms"], prefill["median_ms"]
                )
                wall_derived = derive_tradeoff(
                    mixed["median_wall_ms"],
                    decode["median_wall_ms"],
                    prefill["median_wall_ms"],
                )
                row = {
                    "implementation": args.implementation,
                    "decode_requests": requests,
                    "decode_context_length": context,
                    "prefill_requests": args.prefill_requests,
                    "prefill_tokens": total,
                    "prefill_chunk_size": total // args.prefill_requests,
                    "prefill_prefix_length": prefix,
                    "decode_only_median_ms": decode["median_ms"],
                    "prefill_only_median_ms": prefill["median_ms"],
                    "mixed_median_ms": mixed["median_ms"],
                    "decode_only_median_wall_ms": decode["median_wall_ms"],
                    "prefill_only_median_wall_ms": prefill["median_wall_ms"],
                    "mixed_median_wall_ms": mixed["median_wall_ms"],
                    **derived,
                    "wall_incremental_prefill_ms": wall_derived[
                        "incremental_prefill_ms"
                    ],
                    "wall_decode_stretch": wall_derived["decode_stretch"],
                    "wall_packing_benefit_ms": wall_derived["packing_benefit_ms"],
                    "wall_packing_speedup": wall_derived["packing_speedup"],
                    "mixed_scheduled_tokens_per_second": mixed[
                        "scheduled_tokens_per_second"
                    ],
                    "mixed_peak_gpu_memory_bytes": mixed["peak_gpu_memory_bytes"],
                    "status": "ok",
                    "error": None,
                }
                detailed.append(
                    {"case": row, "decode_only": decode, "prefill_only": prefill,
                     "mixed": mixed}
                )
            except Exception as exc:
                if args.fail_fast:
                    raise
                row = {
                    "implementation": args.implementation,
                    "decode_requests": requests,
                    "decode_context_length": context,
                    "prefill_requests": args.prefill_requests,
                    "prefill_tokens": total,
                    "prefill_chunk_size": total // args.prefill_requests,
                    "prefill_prefix_length": prefix,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                detailed.append({"case": row})
            rows.append(row)
            if row["status"] == "ok":
                print(
                    f"  mixed={row['mixed_median_ms']:.3f} ms "
                    f"prefill-interference={row['incremental_prefill_ms']:.3f} ms "
                    f"packing={row['packing_speedup']:.3f}x",
                    flush=True,
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    json_path = args.output_dir / f"mixed-surface-{stamp}.json"
    csv_path = args.output_dir / f"mixed-surface-{stamp}.csv"
    configuration = vars(args) | {
        "output_dir": str(args.output_dir),
        "decode_requests": decode_requests,
        "decode_context_lengths": decode_context_lengths,
        "prefill_tokens": prefill_tokens,
        "prefill_prefix_lengths": prefill_prefix_lengths,
    }
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "system": system_metadata(),
        "configuration": configuration,
        "model_load_seconds": model_load_seconds,
        "execution_mode": "fully-eager",
        "staging_protocol": "synthetic-resident-paged-kv-outside-timing",
        "decode_controls": [
            {"decode_requests": key[0], "decode_context_length": key[1], **value}
            for key, value in decode_controls.items()
        ],
        "prefill_controls": [
            {"prefill_tokens": key[0], "prefill_prefix_length": key[1], **value}
            for key, value in prefill_controls.items()
        ],
        "cases": detailed,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in CSV_FIELDS} for row in rows)
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
