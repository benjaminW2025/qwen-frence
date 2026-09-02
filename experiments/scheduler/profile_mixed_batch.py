#!/usr/bin/env python3
"""Profile one real mixed decode/prefill scheduler iteration on CUDA."""

from __future__ import annotations

import argparse
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
    ROOT / "baseline",
    ROOT / "engine" / "kvcache",
    ROOT / "engine" / "scheduler",
    ROOT / "engine" / "graph",
):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from run_benchmarks import system_metadata
from run_phase_sweep import IMPLEMENTATIONS, make_engine


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--implementation", choices=IMPLEMENTATIONS,
                        default="custom-kernels")
    parser.add_argument("--decode-requests", type=int, default=32)
    parser.add_argument("--decode-context-length", type=int, default=2048)
    parser.add_argument("--prefill-requests", type=int, default=2)
    parser.add_argument("--prefill-prefix-length", type=int, default=0)
    parser.add_argument("--prefill-chunk-size", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--timing-repetitions", type=int, default=5)
    parser.add_argument("--profile-repetitions", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prefill-tile-policy", choices=("static", "adaptive"),
                        default="static")
    parser.add_argument("--decode-attention-policy", choices=("production", "adaptive"),
                        default="production")
    parser.add_argument("--with-stack", action="store_true")
    parser.add_argument("--nvtx-only", action="store_true")
    parser.add_argument(
        "--cuda-profiler-range",
        action="store_true",
        help=(
            "bracket only the NVTX target iteration with cudaProfilerStart/Stop; "
            "use with nsys --capture-range=cudaProfilerApi"
        ),
    )
    parser.add_argument("--output-dir", type=Path,
                        default=EXPERIMENTS / "results" / "profiles")
    return parser


def validate_args(args):
    positive = (
        "decode_requests", "decode_context_length", "prefill_requests",
        "prefill_chunk_size", "block_size", "timing_repetitions",
        "profile_repetitions",
    )
    if any(getattr(args, name) < 1 for name in positive):
        raise ValueError("request counts, lengths, and repetition counts must be positive")
    if args.prefill_prefix_length < 0 or args.warmups < 0:
        raise ValueError("prefill-prefix-length and warmups must be non-negative")
    if not args.nvtx_only and args.profile_repetitions != 1:
        raise ValueError(
            "torch-profiler mode supports one target iteration so staging stays excluded"
        )
    if args.cuda_profiler_range and not args.nvtx_only:
        raise ValueError("--cuda-profiler-range requires --nvtx-only")
    if max(
        args.decode_context_length,
        args.prefill_prefix_length + args.prefill_chunk_size,
    ) > 32768:
        raise ValueError("configured sequence exceeds the model context limit")


def main():
    args = build_parser().parse_args()
    validate_args(args)

    import torch
    from torch.profiler import ProfilerActivity, profile, record_function
    from scheduler import Scheduler, Status

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("a CUDA device is required")

    engine, model_load_seconds = make_engine(
        args.model, args.implementation, args.device, args.dtype, args.block_size
    )
    max_running = args.decode_requests + args.prefill_requests
    decode_blocks = (
        args.decode_context_length + 4 + args.block_size - 1
    ) // args.block_size
    prefill_length = args.prefill_prefix_length + args.prefill_chunk_size
    prefill_blocks = (prefill_length + args.block_size - 1) // args.block_size
    num_blocks = (
        args.decode_requests * decode_blocks
        + args.prefill_requests * prefill_blocks
        + max_running
    )
    target_budget = args.decode_requests + args.prefill_requests * args.prefill_chunk_size
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
        max_num_batched_tokens=max(target_budget, max_running),
        prefill_tile_policy=args.prefill_tile_policy,
        decode_attention_policy=args.decode_attention_policy,
    )
    if scheduler.use_graph:
        raise AssertionError("mixed profiling must use the fully eager executor")

    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    decode_prompt = torch.randint(
        1, engine.cfg.vocab, (args.decode_context_length,),
        generator=generator, device=args.device,
    ).tolist()
    prefill_prompt = torch.randint(
        1, engine.cfg.vocab, (prefill_length,),
        generator=generator, device=args.device,
    ).tolist()

    def prepare():
        scheduler.profile_prefill = False
        scheduler.reset()
        for _ in range(args.decode_requests):
            scheduler.add_request(decode_prompt, max_tokens=4)
        decode_slots = scheduler._admit_waiting(args.decode_requests)

        for _ in range(args.prefill_requests):
            scheduler.add_request(prefill_prompt, max_tokens=4)
        admission_count = scheduler._planned_prefill_count()
        if admission_count != args.prefill_requests:
            raise AssertionError(
                f"could admit only {admission_count}/{args.prefill_requests} prefills"
            )
        slots = scheduler._admit_waiting(admission_count)
        n_news = [0] * scheduler.max_running
        for slot in decode_slots:
            n_news[slot] = args.decode_context_length
        for slot in slots:
            n_news[slot] = args.prefill_prefix_length
        scheduler.cache.allocate_block(n_news)

        # Stage resident paged histories without paying quadratic model prefill.
        # K/V values do not change target launch geometry or attention work.
        for slot in decode_slots:
            request = scheduler.prefilling.pop(slot)
            request.num_prompt_tokens_computed = args.decode_context_length
            request.output_ids.append(decode_prompt[-1])
            request.status = Status.RUNNING
            scheduler.running[slot] = request
        for slot in slots:
            scheduler.prefilling[slot].num_prompt_tokens_computed = (
                args.prefill_prefix_length
            )
        if any(scheduler.prefilling[slot].status is not Status.PREFILLING for slot in slots):
            raise AssertionError("prefill staging completed unexpectedly")
        scheduler.iteration_decode_token_counts.clear()
        scheduler.iteration_prefill_token_counts.clear()
        scheduler.iteration_token_counts.clear()
        scheduler.iteration_kinds.clear()
        scheduler.max_num_batched_tokens = max(target_budget, max_running)
        torch.cuda.synchronize()

    def target(profile_labels):
        scheduler.profile_prefill = profile_labels
        outer = record_function("mixed/profiled_iteration") if profile_labels else None
        if outer is not None:
            outer.__enter__()
        try:
            scheduler.step()
        finally:
            if outer is not None:
                outer.__exit__(*sys.exc_info())
        observed = (
            scheduler.iteration_decode_token_counts[-1],
            scheduler.iteration_prefill_token_counts[-1],
            scheduler.iteration_token_counts[-1],
        )
        expected = (args.decode_requests, args.prefill_requests * args.prefill_chunk_size,
                    target_budget)
        if observed != expected:
            raise AssertionError(f"scheduled {observed}, expected {expected}")

    with torch.no_grad():
        for _ in range(args.warmups):
            prepare()
            target(False)
        torch.cuda.synchronize()

        raw_ms = []
        for _ in range(args.timing_repetitions):
            prepare()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            target(False)
            end.record()
            end.synchronize()
            raw_ms.append(float(start.elapsed_time(end)))

        torch.cuda.reset_peak_memory_stats()
        prof = None
        if args.nvtx_only:
            if args.cuda_profiler_range:
                torch.cuda.cudart().cudaProfilerStart()
            try:
                for _ in range(args.profile_repetitions):
                    prepare()
                    target(True)
                torch.cuda.synchronize()
            finally:
                if args.cuda_profiler_range:
                    torch.cuda.cudart().cudaProfilerStop()
        else:
            prepare()
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                with_flops=True,
                with_stack=args.with_stack,
            ) as prof:
                target(True)
                prof.step()
        torch.cuda.synchronize()
        peak_memory = int(torch.cuda.max_memory_allocated())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    stem = (
        f"mixed-D{args.decode_requests}-C{args.decode_context_length}"
        f"-P{args.prefill_requests}-Q{args.prefill_chunk_size}-{stamp}"
    )
    trace_path = args.output_dir / f"{stem}.json"
    table_path = args.output_dir / f"{stem}.txt"
    metadata_path = args.output_dir / f"{stem}-metadata.json"
    if prof is not None:
        prof.export_chrome_trace(str(trace_path))
        tables = []
        for title, sort_by, limit in (
            ("CUDA total time", "cuda_time_total", 100),
            ("CUDA self-time", "self_cuda_time_total", 100),
            ("CPU self-time", "self_cpu_time_total", 60),
        ):
            tables.append(
                f"{title}\n{'=' * len(title)}\n"
                + prof.key_averages(group_by_input_shape=False).table(
                    sort_by=sort_by, row_limit=limit
                )
            )
        table_path.write_text("\n\n".join(tables) + "\n")

    median_ms = statistics.median(raw_ms)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "system": system_metadata(),
        "configuration": vars(args) | {"output_dir": str(args.output_dir)},
        "model_load_seconds": model_load_seconds,
        "execution_mode": "fully-eager",
        "staging_protocol": "synthetic-resident-paged-kv-outside-timing",
        "target_iteration_kind": "mixed",
        "decode_tokens": args.decode_requests,
        "prefill_tokens": args.prefill_requests * args.prefill_chunk_size,
        "target_budget": target_budget,
        "raw_latency_ms": raw_ms,
        "median_latency_ms": median_ms,
        "scheduled_tokens_per_second": target_budget / (median_ms * 1e-3),
        "peak_gpu_memory_bytes": peak_memory,
        "trace": str(trace_path) if prof is not None else None,
        "operator_tables": str(table_path) if prof is not None else None,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"median mixed latency : {median_ms:.3f} ms")
    print(f"scheduled throughput : {metadata['scheduled_tokens_per_second']:.2f} tok/s")
    print(f"peak PyTorch memory  : {peak_memory / 2**30:.2f} GiB")
    if prof is not None:
        print(f"trace                : {trace_path}")
        print(f"operator tables      : {table_path}")
    print(f"metadata             : {metadata_path}")


if __name__ == "__main__":
    main()
