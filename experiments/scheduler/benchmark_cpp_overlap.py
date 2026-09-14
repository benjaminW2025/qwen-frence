#!/usr/bin/env python3
"""Paired C++ scheduler A/B for mixed-iteration CPU/GPU metadata overlap.

The forward callback uses a small synthetic GPU workload so this isolates the
scheduler change. It is not a full-model throughput benchmark.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import time

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "engine/cpp/build"))
import inference_engine_cpp as cpp


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda")
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--long-requests", type=int, default=7)
    p.add_argument("--prompt-length", type=int, default=4096)
    p.add_argument("--prefill-chunk", type=int, default=512)
    p.add_argument("--matrix-size", type=int, default=2048)
    p.add_argument("--decode-matmuls", type=int, default=8)
    p.add_argument("--output", type=Path)
    return p


def run_case(args, overlap, a, b, workspace):
    config = cpp.SchedulerConfig()
    config.max_batch_size = args.long_requests + 1
    config.max_prefill_tokens_per_iter = args.prefill_chunk
    config.max_context_length = args.prompt_length + 128
    config.overlap_prefill_build = overlap
    loop = cpp.IterationLoop(config, torch.device(args.device))

    def forward(ids, positions, slots, cu, context, blocks, max_query, is_decode):
        del positions, slots, context, blocks, max_query
        if is_decode:
            for _ in range(args.decode_matmuls):
                torch.mm(a, b, out=workspace)
            source = ids
        else:
            source = ids.index_select(0, cu[1:].long() - 1)
        return torch.nn.functional.one_hot((source + 1) % 64, 64).float()

    loop.submit_request([1], 128)
    loop.step(forward)  # Admit the short request so later steps mix decode/prefill.
    for index in range(args.long_requests):
        loop.submit_request([index + 2] * args.prompt_length, 2)

    mixed_ms = []
    outputs = dict(loop.pop_completed())
    steps = 0
    while loop.num_pending() or loop.num_running():
        phases = []

        def traced_forward(*inputs):
            phases.append(inputs[-1])
            return forward(*inputs)

        start = time.perf_counter()
        loop.step(traced_forward)
        elapsed_ms = (time.perf_counter() - start) * 1000
        if phases == [True, False]:
            mixed_ms.append(elapsed_ms)
        outputs.update(loop.pop_completed())
        steps += 1
        if steps > args.prompt_length * args.long_requests + 256:
            raise RuntimeError("scheduler failed to drain")
    if not mixed_ms:
        raise RuntimeError("workload produced no mixed iterations")
    return {"mixed_ms": mixed_ms, "median_mixed_ms": statistics.median(mixed_ms),
            "mixed_steps": len(mixed_ms), "outputs": outputs}


def main():
    args = parser().parse_args()
    if not torch.cuda.is_available() or torch.device(args.device).type != "cuda":
        raise RuntimeError("this overlap benchmark requires CUDA")
    if min(args.trials, args.long_requests, args.prompt_length, args.prefill_chunk,
           args.matrix_size, args.decode_matmuls) < 1:
        raise ValueError("counts and sizes must be positive")
    device = torch.device(args.device)
    a = torch.randn(args.matrix_size, args.matrix_size, device=device, dtype=torch.float16)
    b = torch.randn_like(a)
    workspace = torch.empty_like(a)
    # Initialize matmul kernels before either measured variant.
    torch.mm(a, b, out=workspace)
    torch.cuda.synchronize(device)

    trials = []
    for trial in range(args.trials):
        order = (False, True) if trial % 2 == 0 else (True, False)
        results = {overlap: run_case(args, overlap, a, b, workspace)
                   for overlap in order}
        if results[False]["outputs"] != results[True]["outputs"]:
            raise AssertionError(f"outputs differ in trial {trial}")
        if results[False]["mixed_steps"] != results[True]["mixed_steps"]:
            raise AssertionError(f"mixed schedules differ in trial {trial}")
        ratio = results[False]["median_mixed_ms"] / results[True]["median_mixed_ms"]
        row = {"trial": trial, "baseline_mixed_ms": results[False]["median_mixed_ms"],
               "overlap_mixed_ms": results[True]["median_mixed_ms"],
               "mixed_steps": results[True]["mixed_steps"], "speedup": ratio}
        trials.append(row)
        print(f"trial {trial + 1}/{args.trials}: {ratio:.3f}x, "
              f"mixed steps={row['mixed_steps']}", flush=True)

    report = {"scope": "synthetic mixed-iteration C++ scheduling, not full-model throughput",
              "configuration": {k: str(v) if isinstance(v, Path) else v
                                for k, v in vars(args).items()},
              "trials": trials,
              "geomean_speedup": math.exp(statistics.mean(math.log(r["speedup"]) for r in trials))}
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
