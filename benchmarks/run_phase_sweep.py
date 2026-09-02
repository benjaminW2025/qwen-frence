#!/usr/bin/env python3
"""Phase-separated prefill and long-context decode benchmark.

Prefill runs the real model over identical token-ID prompts. Decode seeds a paged KV
cache to the requested context length without doing quadratic prompt prefill, then
measures teacher-forced single-token steps. The synthetic history isolates the O(L)
decode path; cache allocation, zero initialization, and CUDA-graph capture are setup
costs and are not included in steady-state timing.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any

from benchmark_core import parse_int_list
from run_benchmarks import system_metadata


MODEL_ID = "Qwen/Qwen2.5-1.5B"
IMPLEMENTATIONS = (
    "continuous-batching",
    "bucketed-cuda-graphs",
    "custom-kernels",
    "regime-dispatched",
)
DEFAULT_BATCH_SIZES = "1,2,4,8"
DEFAULT_PREFILL_LENGTHS = "128,512,1024,2048,4096"
DEFAULT_CONTEXT_LENGTHS = "128,512,1024,2048,4096,8192,16384,32736"
CSV_FIELDS = (
    "phase",
    "implementation",
    "batch_size",
    "sequence_length",
    "decode_steps",
    "median_ms",
    "min_ms",
    "max_ms",
    "tokens_per_second",
    "per_sequence_tokens_per_second",
    "peak_gpu_memory_bytes",
    "setup_ms",
    "correct",
    "max_logit_error",
    "status",
    "error",
)


def parse_names(value: str, allowed: tuple[str, ...], label: str) -> list[str]:
    names = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [name for name in names if name not in allowed]
    if not names or unknown:
        raise argparse.ArgumentTypeError(
            f"unknown {label} {unknown}; choose from {', '.join(allowed)}"
        )
    return names


def parse_phases(value: str) -> list[str]:
    return parse_names(value, ("prefill", "decode"), "phase(s)")


def parse_implementations(value: str) -> list[str]:
    return parse_names(value, IMPLEMENTATIONS, "implementation(s)")


def validate_sweep(
    *,
    batch_sizes: list[int],
    prefill_lengths: list[int],
    context_lengths: list[int],
    decode_steps: int,
    max_seq_len: int,
) -> None:
    if not batch_sizes or any(value < 1 for value in batch_sizes):
        raise ValueError("batch sizes must be positive")
    if not prefill_lengths or any(value < 1 for value in prefill_lengths):
        raise ValueError("prefill lengths must be positive")
    if not context_lengths or any(value < 1 for value in context_lengths):
        raise ValueError("context lengths must be positive")
    if decode_steps < 1:
        raise ValueError("decode-steps must be positive")
    if max(prefill_lengths) > max_seq_len:
        raise ValueError(
            f"prefill length {max(prefill_lengths)} exceeds max_seq_len={max_seq_len}"
        )
    invalid = [length for length in context_lengths if length + decode_steps > max_seq_len]
    if invalid:
        raise ValueError(
            f"context length(s) {invalid} plus decode_steps={decode_steps} exceed "
            f"max_seq_len={max_seq_len}"
        )


def planned_cases(
    phases: list[str],
    implementations: list[str],
    batch_sizes: list[int],
    prefill_lengths: list[int],
    context_lengths: list[int],
) -> list[tuple[str, str, int, int]]:
    cases = []
    for implementation in implementations:
        if "prefill" in phases:
            cases.extend(
                ("prefill", implementation, batch, length)
                for batch in batch_sizes
                for length in prefill_lengths
            )
        if "decode" in phases:
            cases.extend(
                ("decode", implementation, batch, length)
                for batch in batch_sizes
                for length in context_lengths
            )
    return cases


def summarize_times(milliseconds: list[float], tokens: int, batch_size: int) -> dict[str, float]:
    if not milliseconds or any(value <= 0 for value in milliseconds):
        raise ValueError("timings must contain positive values")
    median_ms = statistics.median(milliseconds)
    tokens_per_second = tokens / (median_ms * 1e-3)
    return {
        "median_ms": median_ms,
        "min_ms": min(milliseconds),
        "max_ms": max(milliseconds),
        "tokens_per_second": tokens_per_second,
        "per_sequence_tokens_per_second": tokens_per_second / batch_size,
    }


def implementation_flags(name: str) -> tuple[bool, bool]:
    if name == "continuous-batching":
        return False, False
    if name == "bucketed-cuda-graphs":
        return True, False
    if name in ("custom-kernels", "regime-dispatched"):
        return False, True
    raise ValueError(name)


def make_engine(model_id: str, implementation: str, device: str, dtype_name: str, block_size: int):
    import torch
    from naive_forward import Qwen2Config
    from paged_engine import PagedEngine

    _, use_custom_kernels = implementation_flags(implementation)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_name]
    config = Qwen2Config(use_custom_kernels=use_custom_kernels)
    started = time.perf_counter()
    engine = PagedEngine(
        model_id=model_id,
        cfg=config,
        block_size=block_size,
        device=device,
        dtype=dtype,
    )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return engine, time.perf_counter() - started


def _new_uniform_cache(engine, batch_size: int, total_tokens: int):
    from paged_kv_cache import PagedKVCache

    blocks_per_sequence = (
        total_tokens + engine.block_size - 1
    ) // engine.block_size
    cache = PagedKVCache(
        engine.cfg,
        batch_size,
        batch_size * blocks_per_sequence,
        engine.block_size,
        engine.device,
        engine.dtype,
    )
    return cache, blocks_per_sequence


def _reset_peak(torch, device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def _peak_memory(torch, device: str) -> int | None:
    if not device.startswith("cuda"):
        return None
    torch.cuda.synchronize()
    return int(torch.cuda.max_memory_allocated())


def run_prefill_case(
    engine,
    *,
    implementation: str,
    batch_size: int,
    prompt_length: int,
    warmups: int,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    import torch
    from scheduler import Scheduler

    setup_started = time.perf_counter()
    blocks_per_sequence = (
        prompt_length + engine.block_size - 1
    ) // engine.block_size
    scheduler = Scheduler(
        engine.model,
        engine.cfg,
        max_running=batch_size,
        num_blocks=batch_size * blocks_per_sequence + batch_size,
        block_size=engine.block_size,
        eos_ids=set(),
        device=engine.device,
        dtype=engine.dtype,
        use_graph=False,
        # Phase isolation requires the complete uniform prefill in one iteration.
        max_num_batched_tokens=batch_size * prompt_length,
        prefill_tile_policy=(
            "adaptive" if implementation == "regime-dispatched" else "static"
        ),
        decode_attention_policy=(
            "adaptive" if implementation == "regime-dispatched" else "production"
        ),
        enable_regime_fusions=implementation == "regime-dispatched",
    )
    generator = torch.Generator(device=engine.device).manual_seed(
        seed + batch_size * 100_003 + prompt_length
    )
    base = torch.randint(
        1,
        engine.cfg.vocab,
        (1, prompt_length),
        generator=generator,
        device=engine.device,
    )
    prompt_ids = base[0].tolist()
    torch.cuda.synchronize()
    setup_ms = (time.perf_counter() - setup_started) * 1e3

    def one_run() -> tuple[float, Any]:
        scheduler.reset()
        request_ids = [
            scheduler.add_request(prompt_ids, max_tokens=1)
            for _ in range(batch_size)
        ]
        torch.cuda.synchronize()
        started = time.perf_counter()
        while scheduler.waiting or scheduler.prefilling or scheduler.running:
            scheduler.step()
        torch.cuda.synchronize()
        token_ids = [scheduler.finished[request_id][0] for request_id in request_ids]
        return (time.perf_counter() - started) * 1e3, token_ids

    for _ in range(warmups):
        one_run()
    _reset_peak(torch, engine.device)
    raw_ms = []
    final_tokens = None
    for _ in range(repetitions):
        elapsed_ms, final_tokens = one_run()
        raw_ms.append(elapsed_ms)

    identical = all(token == final_tokens[0] for token in final_tokens)
    summary = summarize_times(
        raw_ms,
        tokens=batch_size * prompt_length,
        batch_size=batch_size,
    )
    return {
        "phase": "prefill",
        "implementation": implementation,
        "batch_size": batch_size,
        "sequence_length": prompt_length,
        "decode_steps": 0,
        **summary,
        "peak_gpu_memory_bytes": _peak_memory(torch, engine.device),
        "setup_ms": setup_ms,
        "correct": identical,
        "max_logit_error": None,
        "status": "ok" if identical else "failed_correctness",
        "error": None if identical else "identical prompt rows produced different top-1 tokens",
        "raw_ms": raw_ms,
        "prefill_batch_sizes": list(scheduler.prefill_batch_sizes),
        "protocol": "real-identical-token-id-ragged-batched-admission-prefill-plus-greedy-first-token",
    }


def _zero_cache(cache) -> None:
    for key, value in zip(cache.k_pool, cache.v_pool):
        key.zero_()
        value.zero_()


def run_decode_case(
    engine,
    *,
    implementation: str,
    batch_size: int,
    context_length: int,
    decode_steps: int,
    warmups: int,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    import torch
    from paged_graph_decoder import (
        CUDAGraphDecoder,
        build_decode_step_inputs,
        graph_decode_forward,
    )

    use_graph, _ = implementation_flags(implementation)
    setup_started = time.perf_counter()
    cache, max_blocks = _new_uniform_cache(
        engine, batch_size, context_length + decode_steps
    )
    decoder = None
    capture_ms = 0.0
    if use_graph:
        started = time.perf_counter()
        decoder = CUDAGraphDecoder(
            engine.model,
            cache,
            batch_size=batch_size,
            max_blocks=max_blocks,
            device=engine.device,
            dtype=engine.dtype,
        )
        decoder.capture()
        torch.cuda.synchronize()
        capture_ms = (time.perf_counter() - started) * 1e3
        # Capture uses zero-valued static metadata and may write non-finite values to
        # pool slot zero. Restore the synthetic history before validation/timing.
        _zero_cache(cache)
        cache.reset()
    torch.cuda.synchronize()
    setup_ms = (time.perf_counter() - setup_started) * 1e3

    token = (seed + context_length + batch_size) % (engine.cfg.vocab - 1) + 1
    tokens = [token] * batch_size

    def prepare() -> None:
        cache.reset()
        cache.allocate_block([context_length] * batch_size)

    def step(step_tokens):
        cache.allocate_block([1] * batch_size)
        inputs = build_decode_step_inputs(cache, step_tokens, max_blocks, engine.device)
        if decoder is not None:
            return decoder.decode(*inputs), inputs
        return graph_decode_forward(
            engine.model,
            cache,
            *inputs,
            decode_attention_policy=(
                "adaptive" if implementation == "regime-dispatched" else "production"
            ),
            max_decode_context_length=max(cache.cur_lens),
            enable_regime_fusions=implementation == "regime-dispatched",
        ), inputs

    # One teacher-forced contract check at this exact batch/context shape.
    prepare()
    graph_or_eager, inputs = step(tokens)
    output = graph_or_eager.clone()
    eager = graph_decode_forward(engine.model, cache, *inputs)
    torch.cuda.synchronize()
    max_error = (output.float() - eager.float()).abs().max().item()
    top1_match = bool(torch.equal(
        output[:, -1].argmax(dim=-1), eager[:, -1].argmax(dim=-1)
    ))
    identical_rows = bool(torch.all(
        output[:, -1].argmax(dim=-1) == output[0, -1].argmax(dim=-1)
    ).item())
    finite = bool(torch.isfinite(output).all().item())
    correct = finite and top1_match and identical_rows and max_error <= 1e-2

    def timed_run() -> float:
        prepare()
        torch.cuda.synchronize()
        started = time.perf_counter()
        for step_index in range(decode_steps):
            # Teacher forcing removes the per-step device-to-host argmax sync and
            # keeps the benchmark focused on cache traversal plus model execution.
            forced = [((token + step_index) % (engine.cfg.vocab - 1)) + 1] * batch_size
            step(forced)
        torch.cuda.synchronize()
        return (time.perf_counter() - started) * 1e3

    for _ in range(warmups):
        timed_run()
    _reset_peak(torch, engine.device)
    raw_ms = [timed_run() for _ in range(repetitions)]
    summary = summarize_times(
        raw_ms,
        tokens=batch_size * decode_steps,
        batch_size=batch_size,
    )
    return {
        "phase": "decode",
        "implementation": implementation,
        "batch_size": batch_size,
        "sequence_length": context_length,
        "decode_steps": decode_steps,
        **summary,
        "peak_gpu_memory_bytes": _peak_memory(torch, engine.device),
        "setup_ms": setup_ms,
        "capture_ms": capture_ms,
        "correct": correct,
        "max_logit_error": max_error,
        "status": "ok" if correct else "failed_correctness",
        "error": None if correct else (
            f"finite={finite}, top1_match={top1_match}, "
            f"identical_rows={identical_rows}, max_error={max_error:.6f}"
        ),
        "raw_ms": raw_ms,
        "protocol": "synthetic-zero-kv-history-teacher-forced-steady-state-decode",
    }


def error_row(
    *,
    phase: str,
    implementation: str,
    batch_size: int,
    sequence_length: int,
    decode_steps: int,
    status: str,
    error: str,
) -> dict[str, Any]:
    return {
        "phase": phase,
        "implementation": implementation,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "decode_steps": decode_steps if phase == "decode" else 0,
        "median_ms": None,
        "min_ms": None,
        "max_ms": None,
        "tokens_per_second": None,
        "per_sequence_tokens_per_second": None,
        "peak_gpu_memory_bytes": None,
        "setup_ms": None,
        "correct": False,
        "max_logit_error": None,
        "status": status,
        "error": error,
        "raw_ms": [],
    }


def format_number(value, digits=2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def print_row(row: dict[str, Any]) -> None:
    print(
        f"{row['phase']:>7} | {row['implementation']:>23} | "
        f"B={row['batch_size']:<2} | L={row['sequence_length']:<5} | "
        f"{format_number(row['median_ms']):>10} ms | "
        f"{format_number(row['tokens_per_second']):>10} tok/s | "
        f"{row['status']}"
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in CSV_FIELDS})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--phases", type=parse_phases, default=["prefill", "decode"])
    parser.add_argument(
        "--implementations",
        type=parse_implementations,
        default=["custom-kernels"],
    )
    parser.add_argument("--batch-sizes", default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--prefill-lengths", default=DEFAULT_PREFILL_LENGTHS)
    parser.add_argument("--context-lengths", default=DEFAULT_CONTEXT_LENGTHS)
    parser.add_argument("--decode-steps", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "phase-sweeps",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.warmups < 0 or args.repetitions < 1:
        raise ValueError("warmups must be >= 0 and repetitions must be >= 1")
    if args.block_size < 1:
        raise ValueError("block-size must be positive")

    batch_sizes = parse_int_list(args.batch_sizes)
    prefill_lengths = parse_int_list(args.prefill_lengths)
    context_lengths = parse_int_list(args.context_lengths)
    max_seq_len = 32768
    validate_sweep(
        batch_sizes=batch_sizes,
        prefill_lengths=prefill_lengths,
        context_lengths=context_lengths,
        decode_steps=args.decode_steps,
        max_seq_len=max_seq_len,
    )
    cases = planned_cases(
        args.phases,
        args.implementations,
        batch_sizes,
        prefill_lengths,
        context_lengths,
    )

    import torch

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("A CUDA device is required for the phase sweep")

    rows: list[dict[str, Any]] = []
    model_load_times = {}
    print(
        f"phase sweep: {len(cases)} cases, warmups={args.warmups}, "
        f"repetitions={args.repetitions}"
    )
    for implementation in args.implementations:
        print(f"\nloading {implementation}...")
        engine, load_seconds = make_engine(
            args.model,
            implementation,
            args.device,
            args.dtype,
            args.block_size,
        )
        model_load_times[implementation] = load_seconds
        implementation_cases = [case for case in cases if case[1] == implementation]
        for phase, _, batch_size, sequence_length in implementation_cases:
            try:
                with torch.no_grad():
                    if phase == "prefill":
                        row = run_prefill_case(
                            engine,
                            implementation=implementation,
                            batch_size=batch_size,
                            prompt_length=sequence_length,
                            warmups=args.warmups,
                            repetitions=args.repetitions,
                            seed=args.seed,
                        )
                    else:
                        row = run_decode_case(
                            engine,
                            implementation=implementation,
                            batch_size=batch_size,
                            context_length=sequence_length,
                            decode_steps=args.decode_steps,
                            warmups=args.warmups,
                            repetitions=args.repetitions,
                            seed=args.seed,
                        )
            except torch.OutOfMemoryError as exc:
                row = error_row(
                    phase=phase,
                    implementation=implementation,
                    batch_size=batch_size,
                    sequence_length=sequence_length,
                    decode_steps=args.decode_steps,
                    status="oom",
                    error=str(exc),
                )
                torch.cuda.empty_cache()
            rows.append(row)
            print_row(row)
            if args.fail_fast and row["status"] != "ok":
                raise RuntimeError(f"phase sweep failed: {row}")
            gc.collect()
            torch.cuda.empty_cache()

        del engine
        gc.collect()
        torch.cuda.empty_cache()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    json_path = args.output_dir / f"phase-sweep-{stamp}.json"
    csv_path = args.output_dir / f"phase-sweep-{stamp}.csv"
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "system": system_metadata(),
        "configuration": {
            "model": args.model,
            "device": args.device,
            "dtype": args.dtype,
            "block_size": args.block_size,
            "phases": args.phases,
            "implementations": args.implementations,
            "batch_sizes": batch_sizes,
            "prefill_lengths": prefill_lengths,
            "context_lengths": context_lengths,
            "decode_steps": args.decode_steps,
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "seed": args.seed,
        },
        "model_load_time_s": model_load_times,
        "protocol": {
            "prefill": (
                "real identical token-ID prompts admitted through the current packed "
                "ragged scheduler path; cache setup excluded; timing includes model "
                "prefill and first-token argmax"
            ),
            "decode": (
                "synthetic zero-valued paged KV history; graph capture excluded; "
                "teacher-forced steps exclude device-to-host sampling sync"
            ),
        },
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    write_csv(csv_path, rows)
    failures = [row for row in rows if row["status"] not in ("ok", "oom")]
    successful = sum(row["status"] == "ok" for row in rows)
    oom = sum(row["status"] == "oom" for row in rows)
    print(f"\nraw results: {json_path}")
    print(f"summary CSV: {csv_path}")
    print(
        f"successful={successful}/{len(rows)} oom={oom} "
        f"correctness_failures={len(failures)}"
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
