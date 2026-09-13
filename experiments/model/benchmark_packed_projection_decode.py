#!/usr/bin/env python3
"""Paired real-Qwen decode ablation for separate and packed projections.

This measures eager production decode compute only: scheduler work, model load,
weight packing, CUDA compilation, and paged-KV staging are outside timing.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import statistics
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (ROOT / "baseline", ROOT / "engine" / "graph", ROOT / "engine" / "kvcache", ROOT / "benchmarks"):
    sys.path.insert(0, str(path))

LAYOUTS = (("separate", False, False), ("qkv_packed", True, False),
           ("qkv_gate_up_packed", True, True))


def int_list(value):
    try:
        result = tuple(int(x) for x in value.split(",") if x)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated positive integers") from exc
    if not result or any(x < 1 for x in result) or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("values must be unique positive integers")
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=("smoke", "full"), default="full")
    p.add_argument("--batches", type=int_list)
    p.add_argument("--contexts", type=int_list)
    p.add_argument("--cache-modes", default="warm,evict")
    p.add_argument("--evict-mib", type=int, default=256)
    p.add_argument("--warmups", type=int, default=3)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--samples", type=int, default=7)
    p.add_argument("--seed", type=int, default=20260913)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", type=Path, default=ROOT / "experiments/results/packed-projection-decode")
    return p


def validate(args):
    if args.warmups < 0 or args.trials < 1 or args.samples < 1 or args.evict_mib < 1:
        raise ValueError("invalid warmup, trial, sample, or eviction count")
    modes = tuple(x for x in args.cache_modes.split(",") if x)
    if not modes or len(set(modes)) != len(modes) or set(modes) - {"warm", "evict"}:
        raise ValueError("--cache-modes must be warm, evict, or both")
    return modes


def timed(torch, fn):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record(); fn(); end.record(); end.synchronize()
    return float(start.elapsed_time(end))


def stage_case(torch, cfg, batch, context, dtype, seed):
    from paged_kv_cache import PagedKVCache
    block = 16
    pages = (context + 1 + block - 1) // block
    cache = PagedKVCache(cfg, batch, batch * pages, block, "cuda", dtype)
    cache.block_tables = [list(range(i * pages, (i + 1) * pages)) for i in range(batch)]
    cache.cur_lens = [context + 1] * batch
    gen = torch.Generator(device="cuda").manual_seed(seed)
    for pool in (*cache.k_pool, *cache.v_pool):
        pool.normal_(generator=gen)
    ids = torch.randint(1, cfg.vocab, (batch, 1), device="cuda", generator=gen)
    positions = torch.full((batch,), context, device="cuda", dtype=torch.int32)
    lengths = torch.full((batch,), context + 1, device="cuda", dtype=torch.int32)
    table = torch.arange(batch * pages, device="cuda", dtype=torch.int32).view(batch, pages)
    slots = torch.arange(batch, device="cuda", dtype=torch.long) * pages * block + context % block + (pages - 1) * block
    return cache, (ids, positions, lengths, table, slots)


def load_models(args, torch):
    from transformers import AutoModelForCausalLM
    from naive_forward import Qwen2Config
    from weight_loader import QwenWeightLoader
    hf = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, attn_implementation="sdpa")
    models = {}
    for name, pack_qkv, pack_gate_up in LAYOUTS:
        cfg = Qwen2Config(use_custom_kernels=True, pack_qkv=pack_qkv, pack_gate_up=pack_gate_up)
        models[name] = QwenWeightLoader(cfg).convert(hf, args.device, torch.float16)
    del hf
    torch.cuda.empty_cache()
    return models


def main():
    args = parser().parse_args(); modes = validate(args)
    import torch
    from paged_graph_decoder import graph_decode_forward
    from run_benchmarks import system_metadata
    if not torch.cuda.is_available(): raise RuntimeError("requires CUDA")
    batches = args.batches or ((1, 8) if args.preset == "smoke" else (1, 2, 4, 8, 16, 32, 64, 96, 128, 256))
    contexts = args.contexts or ((512, 2048) if args.preset == "smoke" else (512, 2048, 8192, 16384))
    models = load_models(args, torch)
    flush = torch.empty(args.evict_mib * 1024 * 1024, device="cuda", dtype=torch.uint8)
    records = []
    with torch.no_grad():
        for batch in batches:
            for context in contexts:
                cache, tensors = stage_case(torch, models["separate"].cfg, batch, context, torch.float16,
                                            args.seed + batch * 100000 + context)
                launch = {name: (lambda model=model: graph_decode_forward(
                    model, cache, *tensors, max_decode_context_length=context + 1)) for name, model in models.items()}
                reference = launch["separate"]()
                for name in tuple(launch)[1:]:
                    candidate = launch[name]()
                    torch.testing.assert_close(candidate, reference, atol=.05, rtol=.01)
                    if not torch.equal(candidate.argmax(-1), reference.argmax(-1)):
                        raise AssertionError(f"{name}: sampled tokens differ at B={batch}, L={context}")
                for fn in launch.values():
                    for _ in range(args.warmups): fn()
                torch.cuda.synchronize()
                for trial in range(args.trials):
                    for mode in modes:
                        order = list(launch); random.Random(args.seed + batch + context + trial).shuffle(order)
                        samples = {name: [] for name in launch}
                        for _ in range(args.samples):
                            for name in order:
                                if mode == "evict": flush.zero_()
                                samples[name].append(timed(torch, launch[name]))
                        medians = {name: statistics.median(values) for name, values in samples.items()}
                        for name, values in samples.items():
                            records.append({"batch": batch, "context": context, "trial": trial, "cache": mode,
                                            "layout": name, "samples_ms": values, "median_ms": medians[name],
                                            "speedup_vs_separate": medians["separate"] / medians[name]})
                print(f"complete: B={batch} L={context}", flush=True)
                del cache, tensors
                torch.cuda.empty_cache()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"decode-ablation-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    out.write_text(json.dumps({"created_at": datetime.now(timezone.utc).isoformat(), "system": system_metadata(),
        "configuration": vars(args) | {"batches": list(batches), "contexts": list(contexts)},
        "layouts": [x[0] for x in LAYOUTS], "scope": "eager production decode forward; scheduler and staging excluded",
        "records": records}, indent=2) + "\n")
    print(f"results: {out}")

if __name__ == "__main__": main()
