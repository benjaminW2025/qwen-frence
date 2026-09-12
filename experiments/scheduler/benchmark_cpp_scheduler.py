#!/usr/bin/env python3
"""Compare C++ and Python scheduler overhead with identical synthetic logits.

This is deliberately a scheduler/control-plane benchmark, not a model throughput
benchmark. Both implementations receive deterministic logits from lightweight
fake model runners, and a correctness preflight checks every logit tensor and
final token sequence before timing begins.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace
import types
from unittest import mock

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CPP_DIR = ROOT / "engine" / "cpp"
BENCHMARKS = ROOT / "benchmarks"
for path in (CPP_DIR / "build", ROOT / "engine" / "scheduler", BENCHMARKS):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

try:
    import inference_engine_cpp as cpp
except ImportError as error:
    raise RuntimeError(
        "build the extension first with `make cpp-scheduler-build`"
    ) from error

# Avoid importing GPU model-runner modules merely to benchmark the Python
# scheduler control plane. The functions are replaced below before execution.
for module_name, function_name in (
    ("ragged_prefill", "ragged_prefill"),
    ("mixed_batch", "mixed_batch_forward"),
    ("paged_graph_decoder", "graph_decode_forward"),
):
    if module_name not in sys.modules:
        module = types.ModuleType(module_name)
        setattr(module, function_name, lambda *args, **kwargs: None)
        sys.modules[module_name] = module

import scheduler as python_scheduler
from run_benchmarks import system_metadata


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-requests", type=int, default=8)
    parser.add_argument("--prompt-length", type=int, default=128)
    parser.add_argument("--output-length", type=int, default=16)
    parser.add_argument("--max-running", type=int, default=8)
    parser.add_argument("--max-prefill-tokens", type=int, default=4096)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments" / "results" / "cpp-scheduler.json",
    )
    return parser


def deterministic_logits(source_tokens, vocab_size, trace=None, phase=None):
    source_tokens = source_tokens.reshape(-1).to(torch.long)
    targets = (source_tokens + 1) % vocab_size
    logits = torch.full(
        (source_tokens.numel(), vocab_size),
        -8.0,
        dtype=torch.float32,
        device=source_tokens.device,
    )
    logits.scatter_(1, targets.unsqueeze(1), 8.0)
    if trace is not None:
        trace.append((phase, logits.detach().cpu().clone()))
    return logits


def make_workload(args):
    return [
        [(request + position) % args.vocab_size for position in range(args.prompt_length)]
        for request in range(args.num_requests)
    ]


def make_cpp_runner(args, prompts, device, trace=None):
    config = cpp.SchedulerConfig()
    config.max_batch_size = args.max_running
    config.max_prefill_tokens_per_iter = args.max_prefill_tokens
    config.max_context_length = args.prompt_length + args.output_length
    config.block_size = args.block_size
    config.num_kv_heads = 1
    config.head_dim = 8
    config.eos_token_id = -1
    loop = cpp.IterationLoop(config, torch.device(device))

    def forward(ids, positions, slots, cu, context, blocks, max_query, is_decode):
        del positions, slots, context, blocks, max_query
        if is_decode:
            source = ids
            phase = "decode"
        else:
            source = ids.index_select(0, cu[1:].to(torch.long) - 1)
            phase = "prefill"
        return deterministic_logits(source, args.vocab_size, trace, phase)

    def run_once():
        for prompt in prompts:
            loop.submit_request(prompt, args.output_length)
        while loop.num_pending() or loop.num_running():
            loop.step(forward)
        return [tokens for _, tokens in sorted(loop.pop_completed())]

    return run_once


def make_python_runner(args, prompts, device, trace=None):
    config = SimpleNamespace(
        n_layers=0,
        n_kv_heads=1,
        d_head=8,
        max_seq_len=args.prompt_length + args.output_length,
    )
    max_blocks_per_request = (
        config.max_seq_len + args.block_size - 1
    ) // args.block_size
    scheduler = python_scheduler.Scheduler(
        model=None,
        cfg=config,
        max_running=args.max_running,
        num_blocks=args.max_running * max_blocks_per_request,
        block_size=args.block_size,
        eos_ids=set(),
        device=device,
        dtype=torch.float32,
        max_num_batched_tokens=args.max_prefill_tokens,
    )

    def record(phase, source):
        return deterministic_logits(source, args.vocab_size, trace, phase)

    def fake_prefill(model, cache, slots, chunks, **kwargs):
        del model, kwargs
        new_tokens = [0] * cache.batch_size
        for slot, chunk in zip(slots, chunks):
            new_tokens[slot] = len(chunk)
        cache.allocate_block(new_tokens)
        source = torch.tensor(
            [chunk[-1] for chunk in chunks], device=device, dtype=torch.long
        )
        return record("prefill", source)

    def fake_decode(model, cache, input_ids, *unused_args, **unused_kwargs):
        del model, cache, unused_args, unused_kwargs
        return record("decode", input_ids.reshape(-1)).unsqueeze(1)

    def fake_mixed(
        model,
        cache,
        *,
        decode_slots,
        decode_tokens,
        prefill_slots,
        prefill_chunks,
        **kwargs,
    ):
        del model, kwargs
        new_tokens = [0] * cache.batch_size
        for slot in decode_slots:
            new_tokens[slot] = 1
        for slot, chunk in zip(prefill_slots, prefill_chunks):
            new_tokens[slot] = len(chunk)
        cache.allocate_block(new_tokens)
        decode = record(
            "decode", torch.tensor(decode_tokens, device=device, dtype=torch.long)
        )
        prefill = record(
            "prefill",
            torch.tensor(
                [chunk[-1] for chunk in prefill_chunks],
                device=device,
                dtype=torch.long,
            ),
        )
        return decode, prefill

    patches = (
        mock.patch.object(python_scheduler, "ragged_prefill", fake_prefill),
        mock.patch.object(python_scheduler, "graph_decode_forward", fake_decode),
        mock.patch.object(python_scheduler, "mixed_batch_forward", fake_mixed),
    )

    def run_once():
        if scheduler.waiting or scheduler.prefilling or scheduler.running:
            raise RuntimeError("Python scheduler was not drained by the previous run")
        scheduler.finished.clear()
        for prompt in prompts:
            scheduler.add_request(prompt, args.output_length)
        with ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            outputs = scheduler.run()
        return [tokens for _, tokens in sorted(outputs.items())]

    return run_once


def synchronize(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))


def measure(operation, warmups, repetitions, device):
    for _ in range(warmups):
        operation()
    synchronize(device)
    samples = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        operation()
        synchronize(device)
        samples.append((time.perf_counter_ns() - started) / 1e6)
    return samples


def main():
    args = build_parser().parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    positive = (
        args.num_requests,
        args.prompt_length,
        args.output_length,
        args.max_running,
        args.max_prefill_tokens,
        args.block_size,
        args.vocab_size,
        args.repetitions,
    )
    if any(value < 1 for value in positive) or args.warmups < 0:
        raise ValueError("sizes and repetitions must be positive; warmups cannot be negative")
    if args.num_requests > args.max_running:
        raise ValueError("this matched benchmark requires num_requests <= max_running")
    if args.num_requests * args.prompt_length > args.max_prefill_tokens:
        raise ValueError(
            "this matched launch-bound benchmark requires all prompts to fit in "
            "the first prefill iteration"
        )

    prompts = make_workload(args)
    cpp_trace = []
    python_trace = []
    cpp_expected = make_cpp_runner(args, prompts, args.device, cpp_trace)()
    python_expected = make_python_runner(args, prompts, args.device, python_trace)()
    if cpp_expected != python_expected:
        raise AssertionError("C++ and Python schedulers produced different token outputs")
    if len(cpp_trace) != len(python_trace):
        raise AssertionError("C++ and Python schedulers produced different logit traces")
    max_logit_error = 0.0
    for (cpp_phase, cpp_logits), (python_phase, python_logits) in zip(
        cpp_trace, python_trace
    ):
        if cpp_phase != python_phase:
            raise AssertionError("C++ and Python scheduler phases differ")
        error = float((cpp_logits - python_logits).abs().max())
        max_logit_error = max(max_logit_error, error)
    if max_logit_error != 0.0:
        raise AssertionError(f"logit traces differ by {max_logit_error}")

    cpp_samples = measure(
        make_cpp_runner(args, prompts, args.device),
        args.warmups,
        args.repetitions,
        args.device,
    )
    python_samples = measure(
        make_python_runner(args, prompts, args.device),
        args.warmups,
        args.repetitions,
        args.device,
    )
    cpp_median = statistics.median(cpp_samples)
    python_median = statistics.median(python_samples)
    result = {
        "schema_version": 1,
        "system": system_metadata(),
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "correctness": {
            "outputs_equal": True,
            "logit_events": len(cpp_trace),
            "max_abs_logit_error": max_logit_error,
        },
        "cpp": {"median_ms": cpp_median, "raw_ms": cpp_samples},
        "python": {"median_ms": python_median, "raw_ms": python_samples},
        "speedup_python_over_cpp": python_median / cpp_median,
        "scope": "synthetic scheduler/control-plane overhead; no transformer layers",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("correctness: exact logits and output tokens")
    print(f"C++ median:    {cpp_median:.4f} ms")
    print(f"Python median: {python_median:.4f} ms")
    print(f"speedup:       {result['speedup_python_over_cpp']:.2f}x")
    print(f"result:        {args.output}")


if __name__ == "__main__":
    main()
