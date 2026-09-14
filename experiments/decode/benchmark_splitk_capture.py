#!/usr/bin/env python3
"""Captured production decode attention (A) against captured split-K decode (C).

The integration experiment measured split-K on an eager adapter, where each call
pays one extra kernel launch plus three partial-buffer allocations. That overhead
is fixed per call while split-K's device-time saving scales with page count, so
below a break-even near 1200 tokens the eager arm loses -- which is what
`e2e-scheduler-splitk-smoke-v2` recorded.

That comparison does not decide adoption, because the engine does not ship the
eager path for pure decode: it captures a CUDA graph per batch bucket. Under
capture the wrapper's allocations are served from the graph's private pool and the
host-side launch never runs again, so the overhead is paid once at capture instead
of every step. Both arms here are captured, so the only difference is the
attention kernel, and the ratio is the number the adoption decision needs.

Neither arm is the eager reference, and neither number is comparable to the
integration experiment's. Timing covers steady-state replay only; capture and
cache seeding are setup.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (ROOT / "benchmarks", ROOT / "baseline", ROOT / "engine/graph",
             ROOT / "engine/kvcache", ROOT / "engine/scheduler"):
    sys.path.insert(0, str(path))

MODEL_ID = "Qwen/Qwen2.5-1.5B"
ARMS = ("production", "splitk")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def build_decoder(engine, cache, batch_size, max_blocks, arm, context_bound):
    """Capture one graph for this arm. The action is frozen by the context bound."""
    from paged_graph_decoder import CUDAGraphDecoder

    decoder = CUDAGraphDecoder(
        engine.model, cache, batch_size=batch_size, max_blocks=max_blocks,
        device=engine.device, dtype=engine.dtype,
        decode_attention_policy=arm,
        max_decode_context_length=context_bound,
    )
    decoder.capture()
    return decoder


def resolved_action(arm, context_bound, block_size):
    """What the policy actually baked in, so a silent production fallback is visible."""
    from paged_decode_attention import (resolve_decode_attention_policy,
                                        select_splitk_config)
    effective = resolve_decode_attention_policy(arm, context_bound)
    if effective != "splitk":
        return {"effective_policy": effective, "action": None}
    config = select_splitk_config(context_bound, page_size=block_size)
    return {"effective_policy": effective,
            "action": f"K{config['split_k']}-S{config['num_stages']}"}


def seed_cache_history(cache, seed):
    """Fill the synthetic prompt history once, outside validation and timing.

    A private generator gives each layer distinct, repeatable K/V values without
    perturbing model or sampling RNG state. Decode writes replace only positions
    after the prompt, so resetting cache metadata preserves this history across
    both arms and every trial.
    """
    import torch

    generator = torch.Generator(device=cache.device)
    generator.manual_seed(seed)
    with torch.no_grad():
        for key, value in zip(cache.k_pool, cache.v_pool):
            key.normal_(generator=generator)
            value.normal_(generator=generator)


def run_case(engine, *, batch_size, context_length, decode_steps, trials, samples,
             warmups, seed):
    import torch
    from run_phase_sweep import _new_uniform_cache, _peak_memory, _reset_peak, _zero_cache
    from paged_graph_decoder import build_decode_step_inputs, graph_decode_forward

    cache, max_blocks = _new_uniform_cache(engine, batch_size, context_length + decode_steps)
    # The graph's block table can address this much context, so it is the longest
    # step any replay can present and the bound the baked action must cover.
    context_bound = max_blocks * engine.block_size
    actions = {arm: resolved_action(arm, context_bound, engine.block_size) for arm in ARMS}

    setup_started = time.perf_counter()
    decoders = {}
    for arm in ARMS:
        decoders[arm] = build_decoder(engine, cache, batch_size, max_blocks, arm, context_bound)
        torch.cuda.synchronize()
        # Capture replays zero-valued metadata into pool slot zero; clear it before
        # any validation or timing reads the cache.
        _zero_cache(cache)
        cache.reset()
    seed_cache_history(cache, seed + context_length * 1009 + batch_size)
    torch.cuda.synchronize()
    setup_ms = (time.perf_counter() - setup_started) * 1e3

    token = (seed + context_length + batch_size) % (engine.cfg.vocab - 1) + 1

    def prepare():
        cache.reset()
        cache.allocate_block([context_length] * batch_size)

    def step(decoder, step_tokens):
        cache.allocate_block([1] * batch_size)
        inputs = build_decode_step_inputs(cache, step_tokens, max_blocks, engine.device)
        return decoder.decode(*inputs), inputs

    # Both captured arms must agree with the eager production kernel at this shape,
    # before any timing is accepted. Split-K reassociates the softmax reduction, so
    # it is not bit-identical; the gate is top-1 agreement plus a bounded max error.
    checks = {}
    for arm in ARMS:
        prepare()
        captured, inputs = step(decoders[arm], [token] * batch_size)
        captured = captured.clone()
        eager = graph_decode_forward(engine.model, cache, *inputs)
        torch.cuda.synchronize()
        max_error = (captured.float() - eager.float()).abs().max().item()
        top1 = bool(torch.equal(captured[:, -1].argmax(dim=-1), eager[:, -1].argmax(dim=-1)))
        finite = bool(torch.isfinite(captured).all().item())
        checks[arm] = {"max_logit_error": max_error, "top1_match": top1,
                       "finite": finite, "correct": finite and top1 and max_error <= 1e-2}
    if not all(c["correct"] for c in checks.values()):
        return {"status": "incorrect", "batch_size": batch_size,
                "context_length": context_length, "decode_steps": decode_steps,
                "checks": checks, "actions": actions}

    def timed(decoder):
        prepare()
        torch.cuda.synchronize()
        started = time.perf_counter()
        for index in range(decode_steps):
            # Teacher forcing keeps a device-to-host argmax out of the loop.
            forced = [((token + index) % (engine.cfg.vocab - 1)) + 1] * batch_size
            step(decoder, forced)
        torch.cuda.synchronize()
        return (time.perf_counter() - started) * 1e3

    for arm in ARMS:
        for _ in range(warmups):
            timed(decoders[arm])
    _reset_peak(torch, engine.device)

    # Shuffle arm order within every sample so drift cannot load onto one arm.
    medians = {arm: [] for arm in ARMS}
    for trial in range(trials):
        raw = {arm: [] for arm in ARMS}
        for sample in range(samples):
            order = list(ARMS)
            random.Random(seed + trial * 1009 + sample).shuffle(order)
            for arm in order:
                raw[arm].append(timed(decoders[arm]))
        for arm in ARMS:
            medians[arm].append(statistics.median(raw[arm]))

    ratios = [a / c for a, c in zip(medians["production"], medians["splitk"])]
    return {"status": "ok", "batch_size": batch_size, "context_length": context_length,
            "decode_steps": decode_steps, "context_bound": context_bound,
            "actions": actions, "checks": checks,
            "trial_medians_ms": medians,
            # >1 means split-K is faster. Range over trials, not a calibrated interval.
            "speedup": statistics.median(ratios),
            "speedup_range": [min(ratios), max(ratios)],
            "tokens": batch_size * decode_steps,
            "setup_ms": setup_ms, "peak_gpu_memory_bytes": _peak_memory(torch, engine.device)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-sizes", default="1,4,32")
    parser.add_argument("--context-lengths", default="1024,2048,4096,8192,16384")
    parser.add_argument("--decode-steps", type=int, default=64)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16", choices=("float16", "bfloat16"))
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "experiments/results/splitk-capture-ab")
    parser.add_argument("--splitk-min-context", type=int,
                        help="Override the split-K context floor. The shipped floor was "
                             "fitted on the eager path, where the per-call overhead is "
                             "real; lower it to locate the captured break-even instead "
                             "of inheriting the eager one.")
    args = parser.parse_args()
    if min(args.trials, args.samples, args.decode_steps) < 1 or args.warmups < 0:
        parser.error("trials, samples and decode-steps must be positive")

    from benchmark_core import parse_int_list
    from run_benchmarks import system_metadata
    from run_phase_sweep import make_engine
    import paged_decode_attention

    floor = paged_decode_attention.MIN_SPLITK_DECODE_CONTEXT_LENGTH
    if args.splitk_min_context is not None:
        # Deliberate measurement override, recorded in the manifest.
        paged_decode_attention.MIN_SPLITK_DECODE_CONTEXT_LENGTH = args.splitk_min_context
        floor = args.splitk_min_context

    batch_sizes = parse_int_list(args.batch_sizes)
    context_lengths = parse_int_list(args.context_lengths)
    engine, load_seconds = make_engine(
        args.model, "custom-kernels", args.device, args.dtype, args.block_size
    )

    manifest = {"schema_version": 1, "model": args.model, "arms": list(ARMS),
                "both_arms_captured": True, "dtype": args.dtype,
                "block_size": args.block_size, "decode_steps": args.decode_steps,
                "trials": args.trials, "samples": args.samples, "warmups": args.warmups,
                "seed": args.seed, "splitk_min_context": floor,
                "model_load_seconds": load_seconds,
                "cache_history": "seeded independent normal K/V per layer, shared by both arms",
                "batch_sizes": batch_sizes, "context_lengths": context_lengths,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "system": system_metadata(),
                "scope": "captured-vs-captured decode attention; teacher-forced steps; "
                         "synthetic cache history; not comparable to the eager "
                         "integration experiment"}
    atomic_json(args.output_dir / "manifest.json", manifest)

    rows = []
    for batch_size in batch_sizes:
        for context_length in context_lengths:
            try:
                row = run_case(engine, batch_size=batch_size, context_length=context_length,
                               decode_steps=args.decode_steps, trials=args.trials,
                               samples=args.samples, warmups=args.warmups, seed=args.seed)
            except Exception as error:  # keep the sweep going; record the failure
                row = {"status": "error", "batch_size": batch_size,
                       "context_length": context_length, "error": repr(error)}
            rows.append(row)
            atomic_json(args.output_dir / "report.json", {"manifest_seed": args.seed,
                                                          "cases": rows})
            if row["status"] == "ok":
                act = row["actions"]["splitk"]["action"] or "production-fallback"
                print(f"B={batch_size:<4} L={context_length:<6} {act:<18} "
                      f"A={statistics.median(row['trial_medians_ms']['production']):8.2f}ms "
                      f"C={statistics.median(row['trial_medians_ms']['splitk']):8.2f}ms "
                      f"speedup={row['speedup']:.3f}x", flush=True)
            else:
                print(f"B={batch_size:<4} L={context_length:<6} {row['status']}", flush=True)

    ok = [r for r in rows if r["status"] == "ok"]
    failed = [r for r in rows if r["status"] != "ok"]
    measured_splitk = [r for r in ok if r["actions"]["splitk"]["action"] is not None]
    fell_back = [r["context_length"] for r in ok if r["actions"]["splitk"]["action"] is None]
    print(f"\n{len(ok)}/{len(rows)} cases measured")
    if fell_back:
        print(f"split-K fell back to production at contexts {sorted(set(fell_back))}; "
              f"lower --splitk-min-context to measure them")
    atomic_json(args.output_dir / "report.json",
                {"manifest_seed": args.seed, "cases": rows,
                 "status": "complete" if not failed and measured_splitk else "incomplete",
                 "production_ready": False,
                 "production_fallback_contexts": sorted(set(fell_back))})
    if failed or not measured_splitk:
        raise SystemExit(f"capture experiment incomplete: {len(failed)} failed cases, "
                         f"{len(measured_splitk)} measured split-K cases")


if __name__ == "__main__":
    main()
