#!/usr/bin/env python3
"""Paired real-Qwen decode ablation for separate and packed projections.

This measures eager production decode compute only: scheduler work, model load,
weight packing, CUDA compilation, and paged-KV staging are outside timing.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
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
    if not args.device.startswith("cuda"):
        raise ValueError("--device must select a CUDA device")
    return modes


def design(args, batches, contexts, modes):
    """Freeze the workload and measurement protocol for safe resumption."""
    sources = (
        Path(__file__),
        ROOT / "baseline" / "naive_forward.py",
        ROOT / "baseline" / "weight_loader.py",
        ROOT / "engine" / "graph" / "paged_graph_decoder.py",
        ROOT / "engine" / "kvcache" / "paged_decode_attention.py",
        ROOT / "baseline" / "kernel_dispatch.py",
    )
    return {
        "preset": args.preset,
        "batches": list(batches),
        "contexts": list(contexts),
        "cache_modes": list(modes),
        "evict_mib": args.evict_mib,
        "warmups": args.warmups,
        "trials": args.trials,
        "samples": args.samples,
        "seed": args.seed,
        "model": args.model,
        "device": args.device,
        "output_dir": str(args.output_dir),
        "sources_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
    }


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def checkpoint_path(output_dir, batch, context):
    return output_dir / "trials" / f"b{batch}-l{context}.json"


def validate_checkpoint(payload, *, fingerprint, batch, context, modes, trials, samples):
    if (payload.get("status") != "complete" or payload.get("fingerprint") != fingerprint
            or payload.get("batch") != batch or payload.get("context") != context):
        raise ValueError(f"invalid checkpoint identity for B={batch}, L={context}")
    expected = {(trial, mode, name) for trial in range(trials) for mode in modes
                for name, _, _ in LAYOUTS}
    seen = set()
    for row in payload.get("records", []):
        key = (row.get("trial"), row.get("cache"), row.get("layout"))
        values = row.get("samples_ms")
        if (key not in expected or key in seen or row.get("batch") != batch
                or row.get("context") != context or not isinstance(values, list)
                or len(values) != samples
                or any(not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0
                       for x in values)
                or row.get("median_ms") != statistics.median(values)
                or not isinstance(row.get("speedup_vs_separate"), (int, float))
                or not math.isfinite(row["speedup_vs_separate"])
                or row["speedup_vs_separate"] <= 0):
            raise ValueError(f"invalid checkpoint observation for B={batch}, L={context}")
        seen.add(key)
    if seen != expected:
        raise ValueError(f"incomplete checkpoint for B={batch}, L={context}")
    index = {(row["trial"], row["cache"], row["layout"]): row for row in payload["records"]}
    for trial in range(trials):
        for mode in modes:
            baseline_ms = index[(trial, mode, "separate")]["median_ms"]
            for name, _, _ in LAYOUTS:
                row = index[(trial, mode, name)]
                if not math.isclose(row["speedup_vs_separate"], baseline_ms / row["median_ms"], rel_tol=1e-9):
                    raise ValueError(f"inconsistent speedup for B={batch}, L={context}")
    return payload["records"]


def timed(torch, fn):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def stage_case(torch, cfg, batch, context, dtype, seed, device):
    from paged_kv_cache import PagedKVCache
    block = 16
    pages = (context + 1 + block - 1) // block
    cache = PagedKVCache(cfg, batch, batch * pages, block, device, dtype)
    cache.block_tables = [list(range(i * pages, (i + 1) * pages)) for i in range(batch)]
    cache.cur_lens = [context + 1] * batch
    gen = torch.Generator(device=device).manual_seed(seed)
    for pool in (*cache.k_pool, *cache.v_pool):
        pool.normal_(generator=gen)
    ids = torch.randint(1, cfg.vocab, (batch, 1), device=device, generator=gen)
    positions = torch.full((batch,), context, device=device, dtype=torch.int32)
    lengths = torch.full((batch,), context + 1, device=device, dtype=torch.int32)
    table = torch.arange(batch * pages, device=device, dtype=torch.int32).view(batch, pages)
    slots = torch.arange(batch, device=device, dtype=torch.long) * pages * block + context % block + (pages - 1) * block
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


def report_payload(configuration, fingerprint, system, records):
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fingerprint": fingerprint,
        "system": system,
        "configuration": configuration,
        "layouts": [name for name, _, _ in LAYOUTS],
        "scope": "eager production decode forward; scheduler and staging excluded",
        "records": records,
    }


def main():
    args = parser().parse_args()
    modes = validate(args)
    import torch
    from paged_graph_decoder import graph_decode_forward
    from run_benchmarks import system_metadata
    if not torch.cuda.is_available():
        raise RuntimeError("requires CUDA")
    torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    batches = args.batches or ((1, 8) if args.preset == "smoke" else (1, 2, 4, 8, 16, 32, 64, 96, 128, 256))
    contexts = args.contexts or ((512, 2048) if args.preset == "smoke" else (512, 2048, 8192, 16384))
    configuration = design(args, batches, contexts, modes)
    fingerprint = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing.get("fingerprint") != fingerprint:
            raise ValueError("output directory has a different protocol or source; use a new directory")
    else:
        atomic_json(manifest_path, {"fingerprint": fingerprint, "configuration": configuration,
                                    "system": system_metadata(),
                                    "created_at": datetime.now(timezone.utc).isoformat()})
    pending = [(batch, context) for batch in batches for context in contexts
               if not checkpoint_path(args.output_dir, batch, context).exists()]
    for batch in batches:
        for context in contexts:
            checkpoint = checkpoint_path(args.output_dir, batch, context)
            if checkpoint.exists():
                validate_checkpoint(json.loads(checkpoint.read_text()), fingerprint=fingerprint,
                                    batch=batch, context=context, modes=modes,
                                    trials=args.trials, samples=args.samples)
    if not pending:
        print("all case checkpoints already complete", flush=True)
    else:
        print(f"pending cases: {len(pending)}/{len(batches) * len(contexts)}", flush=True)
    models = load_models(args, torch) if pending else None
    flush = (torch.empty(args.evict_mib * 1024 * 1024, device=args.device, dtype=torch.uint8)
             if pending and "evict" in modes else None)
    with torch.no_grad():
        for batch, context in pending:
            case_records = []
            cache, tensors = stage_case(
                torch, models["separate"].cfg, batch, context, torch.float16,
                args.seed + batch * 100000 + context, args.device,
            )
            launch = {
                name: (lambda model=model: graph_decode_forward(
                    model, cache, *tensors, max_decode_context_length=context + 1))
                for name, model in models.items()
            }
            reference = launch["separate"]()
            for name in tuple(launch)[1:]:
                candidate = launch[name]()
                torch.testing.assert_close(candidate, reference, atol=.05, rtol=.01)
                if not torch.equal(candidate.argmax(-1), reference.argmax(-1)):
                    raise AssertionError(f"{name}: sampled tokens differ at B={batch}, L={context}")
            for fn in launch.values():
                for _ in range(args.warmups):
                    fn()
            torch.cuda.synchronize()
            for trial in range(args.trials):
                for mode in modes:
                    order = list(launch)
                    samples = {name: [] for name in launch}
                    for sample in range(args.samples):
                        random.Random(
                            args.seed + batch * 100003 + context * 101 + trial * 1009 + sample
                        ).shuffle(order)
                        for name in order:
                            if mode == "evict":
                                flush.zero_()
                            samples[name].append(timed(torch, launch[name]))
                    medians = {name: statistics.median(values) for name, values in samples.items()}
                    for name, values in samples.items():
                        case_records.append({
                            "batch": batch, "context": context, "trial": trial, "cache": mode,
                            "layout": name, "samples_ms": values, "median_ms": medians[name],
                            "speedup_vs_separate": medians["separate"] / medians[name],
                        })
            checkpoint = checkpoint_path(args.output_dir, batch, context)
            atomic_json(checkpoint, {
                "status": "complete", "fingerprint": fingerprint,
                "batch": batch, "context": context, "records": case_records,
            })
            print(f"complete: B={batch} L={context}; saved {checkpoint}", flush=True)
            del launch, cache, tensors, reference, candidate
            torch.cuda.empty_cache()
    records = []
    for batch in batches:
        for context in contexts:
            records.extend(validate_checkpoint(
                json.loads(checkpoint_path(args.output_dir, batch, context).read_text()),
                fingerprint=fingerprint, batch=batch, context=context, modes=modes,
                trials=args.trials, samples=args.samples))
    out = args.output_dir / "decode-ablation-results.json"
    atomic_json(out, report_payload(configuration, fingerprint, system_metadata(), records))
    print(f"results: {out}")

if __name__ == "__main__":
    main()
