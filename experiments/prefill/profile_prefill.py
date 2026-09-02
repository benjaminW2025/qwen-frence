#!/usr/bin/env python3
"""Profile the engine's real scheduler prefill path on CUDA.

The script first collects low-overhead CUDA-event latency, then runs a separate
instrumented pass that exports a Chrome trace and operator tables. Scheduler NVTX
ranges make the same command useful when wrapped by Nsight Systems.
"""

from __future__ import annotations

import argparse
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

from run_benchmarks import system_metadata
from run_phase_sweep import IMPLEMENTATIONS, make_engine


MODEL_ID = "Qwen/Qwen2.5-1.5B"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument(
        "--implementation",
        choices=IMPLEMENTATIONS,
        default="custom-kernels",
    )
    parser.add_argument("--prompt-length", type=int, default=2048)
    parser.add_argument(
        "--requests",
        type=int,
        default=1,
        help="number of identical requests packed into each profiled prefill iteration",
    )
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument(
        "--attention-backend",
        choices=("triton", "sdpa"),
        default="triton",
        help="packed attention backend used when --requests is greater than one",
    )
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--timing-repetitions", type=int, default=5)
    parser.add_argument("--profile-repetitions", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--with-stack", action="store_true")
    parser.add_argument(
        "--nvtx-only",
        action="store_true",
        help="emit NVTX ranges without starting torch.profiler (for Nsight Systems)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXPERIMENTS_DIR / "results" / "profiles",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "prompt_length",
        "requests",
        "block_size",
        "timing_repetitions",
        "profile_repetitions",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.warmups < 0:
        raise ValueError("warmups must be non-negative")
    if args.prompt_length > 32768:
        raise ValueError("prompt-length exceeds the model's 32768-token limit")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)

    import torch
    from torch.profiler import ProfilerActivity, profile, record_function
    from scheduler import Request, Scheduler

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("A CUDA device is required for prefill profiling")

    engine, model_load_seconds = make_engine(
        args.model,
        args.implementation,
        args.device,
        args.dtype,
        args.block_size,
    )
    blocks_per_request = (args.prompt_length + args.block_size - 1) // args.block_size
    scheduler = Scheduler(
        engine.model,
        engine.cfg,
        max_running=args.requests,
        num_blocks=args.requests * blocks_per_request,
        block_size=args.block_size,
        eos_ids=set(),
        device=args.device,
        dtype=engine.dtype,
        use_graph=False,
        profile_prefill=True,
        prefill_attention_backend=args.attention_backend,
        # Preserve the profiler's one packed-prefill launch measurement contract.
        max_num_batched_tokens=args.requests * args.prompt_length,
    )

    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    prompt_ids = torch.randint(
        1,
        engine.cfg.vocab,
        (args.prompt_length,),
        generator=generator,
        device=args.device,
    ).tolist()

    def iteration(profile_labels: bool) -> list[int]:
        scheduler.profile_prefill = profile_labels
        scheduler.reset()
        requests = [
            Request(index, list(prompt_ids), max_tokens=1)
            for index in range(args.requests)
        ]
        slots = list(reversed(range(args.requests)))
        outer = record_function("prefill/profiled_iteration") if profile_labels else None
        if outer is not None:
            outer.__enter__()
        try:
            if args.requests == 1:
                tokens = [scheduler._prefill(slots[0], requests[0])]
            else:
                tokens = scheduler._prefill_batch(slots, requests)
        finally:
            if outer is not None:
                outer.__exit__(*sys.exc_info())
        return tokens

    with torch.no_grad():
        for _ in range(args.warmups):
            iteration(False)
        torch.cuda.synchronize()

        raw_ms = []
        for _ in range(args.timing_repetitions):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            tokens = iteration(False)
            end.record()
            end.synchronize()
            raw_ms.append(float(start.elapsed_time(end)))

        torch.cuda.reset_peak_memory_stats()
        prof = None
        if args.nvtx_only:
            for _ in range(args.profile_repetitions):
                iteration(True)
        else:
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                with_flops=True,
                with_stack=args.with_stack,
            ) as prof:
                for _ in range(args.profile_repetitions):
                    iteration(True)
                    prof.step()
        torch.cuda.synchronize()
        peak_memory = int(torch.cuda.max_memory_allocated())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    stem = f"prefill-L{args.prompt_length}-R{args.requests}-{stamp}"
    trace_path = args.output_dir / f"{stem}.json"
    table_path = args.output_dir / f"{stem}.txt"
    metadata_path = args.output_dir / f"{stem}-metadata.json"

    if prof is not None:
        prof.export_chrome_trace(str(trace_path))
        region_table = prof.key_averages(group_by_input_shape=False).table(
            sort_by="cuda_time_total",
            row_limit=80,
        )
        cuda_table = prof.key_averages(group_by_input_shape=False).table(
            sort_by="self_cuda_time_total",
            row_limit=80,
        )
        cpu_table = prof.key_averages(group_by_input_shape=False).table(
            sort_by="self_cpu_time_total",
            row_limit=40,
        )
        table_path.write_text(
            "CUDA total time (use this to rank nested prefill regions)\n"
            "========================================================\n"
            + region_table
            + "\n\nCUDA self-time (use this to rank individual operators/kernels)\n"
            "==========================================================\n"
            + cuda_table
            + "\n\nCPU self-time\n=============\n"
            + cpu_table
            + "\n"
        )

    median_ms = statistics.median(raw_ms)
    total_tokens = args.requests * args.prompt_length
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "system": system_metadata(),
        "configuration": vars(args) | {"output_dir": str(args.output_dir)},
        "model_load_seconds": model_load_seconds,
        "raw_latency_ms": raw_ms,
        "median_latency_ms": median_ms,
        "aggregate_prompt_tokens_per_second": total_tokens / (median_ms * 1e-3),
        "peak_gpu_memory_bytes": peak_memory,
        "identical_prompt_tokens_match": len(set(tokens)) == 1,
        "trace": str(trace_path) if prof is not None else None,
        "operator_tables": str(table_path) if prof is not None else None,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"median prefill latency : {median_ms:.3f} ms")
    print(f"aggregate throughput  : {metadata['aggregate_prompt_tokens_per_second']:.2f} tok/s")
    print(f"peak PyTorch memory   : {peak_memory / 2**30:.2f} GiB")
    if prof is not None:
        print(f"trace                 : {trace_path}")
        print(f"operator tables       : {table_path}")
    else:
        print("trace                 : captured by the external Nsight process")
    print(f"metadata              : {metadata_path}")


if __name__ == "__main__":
    main()
