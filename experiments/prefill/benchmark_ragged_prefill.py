#!/usr/bin/env python3
"""Compare serial scheduler admission with packed ragged prefill on one model."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
EXPERIMENTS_DIR = HERE.parent
ROOT = EXPERIMENTS_DIR.parent
BENCHMARKS_DIR = ROOT / "benchmarks"
for path in (BENCHMARKS_DIR, ROOT / "engine" / "kvcache"):
    sys.path.insert(0, str(path))

from benchmark_core import parse_int_list
from run_benchmarks import system_metadata
from run_phase_sweep import make_engine


MODEL_ID = "Qwen/Qwen2.5-1.5B"
PATTERNS = ("uniform", "ramp")
CSV_FIELDS = (
    "pattern",
    "batch_size",
    "max_prompt_length",
    "total_prompt_tokens",
    "serial_median_ms",
    "packed_sdpa_median_ms",
    "packed_median_ms",
    "serial_prompt_tokens_per_second",
    "packed_sdpa_prompt_tokens_per_second",
    "packed_prompt_tokens_per_second",
    "speedup",
    "packed_attention_speedup",
    "tokens_match",
    "admission_shape_match",
)


def parse_patterns(value: str) -> list[str]:
    patterns = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [pattern for pattern in patterns if pattern not in PATTERNS]
    if not patterns or unknown:
        raise argparse.ArgumentTypeError(
            f"unknown pattern(s) {unknown}; choose from {', '.join(PATTERNS)}"
        )
    return patterns


def case_lengths(pattern: str, batch_size: int, max_length: int) -> list[int]:
    if pattern == "uniform":
        return [max_length] * batch_size
    if pattern == "ramp":
        return [max(1, max_length * (index + 1) // batch_size) for index in range(batch_size)]
    raise ValueError(pattern)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--batch-sizes", default="1,2,4,8")
    parser.add_argument("--prompt-lengths", default="128,512,2048,4096")
    parser.add_argument("--patterns", type=parse_patterns, default=list(PATTERNS))
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXPERIMENTS_DIR / "results" / "ragged-prefill",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    batch_sizes = parse_int_list(args.batch_sizes)
    prompt_lengths = parse_int_list(args.prompt_lengths)
    if any(value < 1 for value in batch_sizes + prompt_lengths):
        raise ValueError("batch sizes and prompt lengths must be positive")
    if max(prompt_lengths) > 32768:
        raise ValueError("prompt length exceeds the model context limit")
    if args.warmups < 0 or args.repetitions < 1:
        raise ValueError("warmups must be >= 0 and repetitions must be >= 1")

    import torch
    from scheduler import Scheduler

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("A CUDA device is required")

    engine, model_load_seconds = make_engine(
        args.model,
        "custom-kernels",
        args.device,
        args.dtype,
        args.block_size,
    )
    rows = []
    for pattern in args.patterns:
        for batch_size in batch_sizes:
            for max_length in prompt_lengths:
                lengths = case_lengths(pattern, batch_size, max_length)
                generator = torch.Generator(device=args.device).manual_seed(
                    args.seed + batch_size * 1009 + max_length * 9173 + (pattern == "ramp")
                )
                prompts = [
                    torch.randint(
                        1,
                        engine.cfg.vocab,
                        (length,),
                        generator=generator,
                        device=args.device,
                    ).tolist()
                    for length in lengths
                ]
                prompt_blocks = sum(
                    (length + args.block_size - 1) // args.block_size for length in lengths
                )

                def measure(max_prefill_batch_size, attention_backend):
                    scheduler = Scheduler(
                        engine.model,
                        engine.cfg,
                        batch_size,
                        prompt_blocks + batch_size,
                        args.block_size,
                        set(),
                        args.device,
                        engine.dtype,
                        max_prefill_batch_size=max_prefill_batch_size,
                        prefill_attention_backend=attention_backend,
                        # This ablation measures one-shot packed prefill rather than
                        # the global scheduler-budget policy.
                        max_num_batched_tokens=sum(lengths),
                    )

                    def one_run():
                        scheduler.reset()
                        request_ids = [
                            scheduler.add_request(prompt, max_tokens=1) for prompt in prompts
                        ]
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        while scheduler.waiting or scheduler.prefilling or scheduler.running:
                            scheduler.step()
                        end.record()
                        end.synchronize()
                        tokens = [scheduler.finished[request_id][0] for request_id in request_ids]
                        return float(start.elapsed_time(end)), tokens

                    for _ in range(args.warmups):
                        one_run()
                    timings = []
                    tokens = None
                    for _ in range(args.repetitions):
                        elapsed_ms, tokens = one_run()
                        timings.append(elapsed_ms)
                    return timings, tokens, list(scheduler.prefill_batch_sizes)

                serial_ms, serial_tokens, serial_batches = measure(1, "sdpa")
                packed_sdpa_ms, packed_sdpa_tokens, packed_sdpa_batches = measure(
                    batch_size, "sdpa"
                )
                packed_ms, packed_tokens, packed_batches = measure(batch_size, "triton")
                serial_median = statistics.median(serial_ms)
                packed_sdpa_median = statistics.median(packed_sdpa_ms)
                packed_median = statistics.median(packed_ms)
                total_tokens = sum(lengths)
                admission_shape_match = (
                    serial_batches == [1] * batch_size
                    and packed_sdpa_batches == ([batch_size] if batch_size > 1 else [1])
                    and packed_batches == ([batch_size] if batch_size > 1 else [1])
                )
                row = {
                    "pattern": pattern,
                    "batch_size": batch_size,
                    "max_prompt_length": max_length,
                    "prompt_lengths": lengths,
                    "total_prompt_tokens": total_tokens,
                    "serial_median_ms": serial_median,
                    "packed_sdpa_median_ms": packed_sdpa_median,
                    "packed_median_ms": packed_median,
                    "serial_prompt_tokens_per_second": total_tokens / (serial_median * 1e-3),
                    "packed_sdpa_prompt_tokens_per_second": (
                        total_tokens / (packed_sdpa_median * 1e-3)
                    ),
                    "packed_prompt_tokens_per_second": total_tokens / (packed_median * 1e-3),
                    "speedup": serial_median / packed_median,
                    "packed_attention_speedup": packed_sdpa_median / packed_median,
                    "tokens_match": serial_tokens == packed_sdpa_tokens == packed_tokens,
                    "admission_shape_match": admission_shape_match,
                    "serial_prefill_batch_sizes": serial_batches,
                    "packed_sdpa_prefill_batch_sizes": packed_sdpa_batches,
                    "packed_prefill_batch_sizes": packed_batches,
                    "serial_raw_ms": serial_ms,
                    "packed_sdpa_raw_ms": packed_sdpa_ms,
                    "packed_raw_ms": packed_ms,
                }
                rows.append(row)
                print(
                    f"{pattern:>7} B={batch_size:<2} Lmax={max_length:<5} "
                    f"serial={serial_median:>8.3f} ms "
                    f"ragged-sdpa={packed_sdpa_median:>8.3f} ms "
                    f"varlen={packed_median:>8.3f} ms "
                    f"total={row['speedup']:.2f}x attention={row['packed_attention_speedup']:.2f}x "
                    f"tokens={'PASS' if row['tokens_match'] else 'FAIL'} "
                    f"admission={'PASS' if admission_shape_match else 'FAIL'}"
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    json_path = args.output_dir / f"ragged-prefill-{stamp}.json"
    csv_path = args.output_dir / f"ragged-prefill-{stamp}.csv"
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "system": system_metadata(),
        "configuration": {
            "model": args.model,
            "dtype": args.dtype,
            "block_size": args.block_size,
            "batch_sizes": batch_sizes,
            "prompt_lengths": prompt_lengths,
            "patterns": args.patterns,
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "seed": args.seed,
        },
        "model_load_seconds": model_load_seconds,
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows({field: row[field] for field in CSV_FIELDS} for row in rows)
    print(f"json: {json_path}")
    print(f"csv : {csv_path}")

    if not all(row["tokens_match"] for row in rows):
        raise RuntimeError("packed prefill changed at least one greedy first token")
    if not all(row["admission_shape_match"] for row in rows):
        raise RuntimeError("scheduler did not execute the requested serial/packed admission shapes")


if __name__ == "__main__":
    main()
