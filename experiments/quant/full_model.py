#!/usr/bin/env python3
"""One-forward FP8 accuracy interventions using real FA3/graph engine paths.

Example (H100, final vLLM/Triton environment):
  python experiments/quant/full_model.py --output artifacts/fp8.json

This measures warm fixed-shape forward latency, not serving throughput or
long-horizon quality. Candidates start from the same FP16-generated history;
KV-storage interventions additionally quantize that history before attention.
Thresholds are optional; without them results are explicitly unassessed.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
import math
import random
import statistics
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "baseline", ROOT / "engine/kvcache",
                  ROOT / "engine/model_runner", ROOT / "engine/graph",
                  ROOT / "benchmarks", ROOT / "experiments/integration"):
    sys.path.insert(0, str(directory))

import torch
from experiments.quant.projections import (parse_groups, replace_projections,
                                          strict_accumulation)
from experiments.quant.staged_cache import StagedFP8KVCache
from experiments.quant.native_fa3 import NativeFP8Cache, smoke_native


def make_case(kind, batch, prompt, prefix, vocab, seed, device):
    """Allocate disjoint, noncontiguous physical pages per sequence."""
    if kind not in ("prefill", "decode", "mixed"):
        raise ValueError("invalid case kind")
    if min(batch, prompt, prefix) <= 0 or (kind == "mixed" and batch < 2):
        raise ValueError("positive dimensions and at least two mixed requests required")
    decode_rows = batch if kind == "decode" else batch // 2 if kind == "mixed" else 0
    chunks = [1 if row < decode_rows else prompt for row in range(batch)]
    prefixes = [prefix if row < decode_rows else 0 for row in range(batch)]
    contexts = [p + q for p, q in zip(prefixes, chunks)]
    pages = (max(contexts) + 15) // 16
    # Transposition interleaves ownership; reversed pages avoid identity mappings.
    # .contiguous() can preserve a non-unit last stride when pages == 1,
    # because PyTorch ignores singleton strides. FA3 explicitly requires 1.
    table = torch.empty((batch, pages), dtype=torch.int32)
    table.copy_(torch.arange(batch * pages, dtype=torch.int32).flip(0).reshape(pages, batch).T)
    ids, positions, slots, cu = [], [], [], [0]
    generator = torch.Generator().manual_seed(seed)
    history, history_slots = [], []
    for row, (start, chunk) in enumerate(zip(prefixes, chunks)):
        history.append(torch.randint(vocab, (start,), generator=generator).tolist())
        history_slots.extend(int(table[row, pos // 16]) * 16 + pos % 16 for pos in range(start))
        ids.extend(torch.randint(vocab, (chunk,), generator=generator).tolist())
        for pos in range(start, start + chunk):
            positions.append(pos)
            slots.append(int(table[row, pos // 16]) * 16 + pos % 16)
        cu.append(len(ids))
    tensor = lambda values, dtype: torch.tensor(values, dtype=dtype, device=device)
    return {"kind": kind, "table": table.to(device), "blocks": batch * pages,
            "ids": tensor(ids, torch.int64), "positions": tensor(positions, torch.int64),
            "slots": tensor(slots, torch.int64), "cu": tensor(cu, torch.int32),
            "contexts": tensor(contexts, torch.int32), "max_query": max(chunks),
            "max_context": max(contexts), "history": history,
            "history_slots": tensor(history_slots, torch.int64),
            "decode_rows": decode_rows, "batch": batch, "chunks": chunks}


def error_metrics(reference, actual):
    r, a = reference.float(), actual.float()
    if r.shape != a.shape:
        raise ValueError("comparison shape mismatch")
    finite = bool(torch.isfinite(r).all() and torch.isfinite(a).all())
    if not finite:
        return {"finite": False, "max_abs": None, "mean_abs": None,
                "relative_l2": None}
    diff = a - r
    return {"finite": True, "max_abs": float(diff.abs().max()),
            "mean_abs": float(diff.abs().mean()),
            "relative_l2": float(diff.norm() / r.norm().clamp_min(1e-12))}


def compare_logits(reference, actual, atol=None, rtol=None, min_agreement=None):
    metrics = error_metrics(reference, actual)
    if not metrics["finite"]:
        return {**metrics, "status": "fail", "top1_agreement": None}
    r, a = reference.float(), actual.float()
    same = r.argmax(-1) == a.argmax(-1)
    top = r.topk(2, dim=-1).values
    metrics.update(top1_agreement=float(same.float().mean()),
                   reference_top2_margins=(top[:, 0] - top[:, 1]).tolist(),
                   reference_tokens=r.argmax(-1).tolist(),
                   candidate_tokens=a.argmax(-1).tolist(),
                   divergent_rows=(~same).nonzero().flatten().tolist())
    checks = []
    if atol is not None:
        metrics["elementwise_within_tolerance"] = bool(
            ((a - r).abs() <= atol + rtol * r.abs()).all())
        checks.append(metrics["elementwise_within_tolerance"])
    if min_agreement is not None:
        checks.append(metrics["top1_agreement"] >= min_agreement)
    metrics["status"] = "unassessed" if not checks else "pass" if all(checks) else "fail"
    return metrics


def prepare_prefix(model, pool, case):
    from model_adapter import ModelAdapter
    rows = case["decode_rows"]
    if not rows:
        return
    ids, positions, slots, cu, contexts = [], [], [], [0], []
    table_cpu = case["table"].cpu()
    for row, history in enumerate(case["history"][:rows]):
        ids.extend(history)
        positions.extend(range(len(history)))
        slots.extend(int(table_cpu[row, p // 16]) * 16 + p % 16 for p in range(len(history)))
        cu.append(len(ids))
        contexts.append(len(history))
    device = case["ids"].device
    t = lambda data, dtype: torch.tensor(data, device=device, dtype=dtype)
    adapter = ModelAdapter(model, pool, None)
    adapter.prefill_attention_policy = "fa3_varlen"
    adapter(t(ids, torch.int64), t(positions, torch.int64), t(slots, torch.int64),
            t(cu, torch.int32), t(contexts, torch.int32), case["table"][:rows],
            max(contexts), False)


def build_forward(model, pool, case):
    from engine.accepted_config import retained_fusion_options
    from model_adapter import PiecewiseGraphModelAdapter
    adapter = PiecewiseGraphModelAdapter(
        model, pool, None, max_running=case["batch"],
        max_context_length=case["max_context"], decode_attention_policy="fa3",
        decode_buckets=[case["batch"]], prefill_attention_policy="fa3_varlen",
        max_capture_tokens=case["ids"].numel(), max_prefill_shapes=1,
        prefill_buckets=[case["ids"].numel()], **retained_fusion_options())

    def run():
        if case["kind"] == "decode":
            return adapter(case["ids"], case["positions"], case["slots"], case["cu"][:0],
                           case["contexts"], case["table"], 1, True)
        # Same piecewise FA3 varlen path used by the C++ packed-mixed adapter.
        return adapter(case["ids"], case["positions"], case["slots"], case["cu"],
                       case["contexts"], case["table"], case["max_query"], False)
    return adapter, run


def cache_state(pool):
    return [tensor for name in ("k_pool", "v_pool") for tensor in getattr(pool, name)]


def restore_cache(pool, seed):
    for destination, source in zip(cache_state(pool), seed):
        destination.copy_(source)


def capture_writes(pool, case, seed):
    slots = case["slots"]
    writes, unchanged = [], True
    for tensor, previous in zip(cache_state(pool), seed):
        flat = tensor.flatten(0, 1)
        writes.append(flat.index_select(0, slots).clone())
        outside = torch.ones(flat.shape[0], device=flat.device, dtype=torch.bool)
        outside[slots] = False
        unchanged = unchanged and torch.equal(flat[outside], previous.flatten(0, 1)[outside])
    return torch.stack(writes), unchanged


def latency_summary(samples):
    if not samples or any(not math.isfinite(value) or value <= 0 for value in samples):
        raise ValueError("latency samples must be finite and positive")
    ordered = sorted(samples)
    return {"median_ms": statistics.median(samples), "min_ms": ordered[0],
            "max_ms": ordered[-1], "samples_ms": samples}


def measure_forward(run, reset, device, warmup, repeats):
    """Reset fixed-history state outside each timing window; include input copies.

    Event span includes GPU work and any host-submission gaps between events.
    Synchronized wall latency additionally includes CPU dispatch and final wait.
    No accuracy diagnostics, cache reset, capture or JIT compilation are timed.
    """
    for _ in range(warmup):
        reset()
        run()
    torch.cuda.synchronize(device)
    gpu, wall = [], []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        reset()
        torch.cuda.synchronize(device)
        before = time.perf_counter()
        start.record()
        result = run()
        end.record()
        end.synchronize()
        wall.append((time.perf_counter() - before) * 1000.)
        gpu.append(start.elapsed_time(end))
        del result
    return {"cuda_event": latency_summary(gpu), "synchronized_wall": latency_summary(wall)}


def kernel_gate(device):
    """Validate real FP8 math against dequantized operands before loading model."""
    from experiments.quant.projections import FP8Linear, quantize_rows_reference
    from experiments.quant.kernels.projections import quantize_activations
    g = torch.Generator(device=device).manual_seed(317)
    x = torch.randn(7, 160, generator=g, device=device, dtype=torch.float16) * .2
    layer = torch.nn.Linear(160, 96, device=device, dtype=torch.float16)
    layer.weight.copy_(torch.randn(layer.weight.shape, generator=g, device=device,
                                   dtype=torch.float16) * .1)
    layer.bias.zero_()
    quantized = FP8Linear(layer)
    q, scale = quantize_activations(x)
    expected_q, expected_scale = quantize_rows_reference(x)
    torch.testing.assert_close(q.float(), expected_q.float(), atol=0, rtol=0)
    torch.testing.assert_close(scale, expected_scale, atol=0, rtol=0)
    expected = ((q.float() * scale[:, None]) @
                (quantized.weight_fp8.float() * quantized.weight_scales[:, None]).T).half()
    actual = quantized(x)
    torch.testing.assert_close(actual, expected, atol=.002, rtol=.003)
    return {"passed": True, "comparison": error_metrics(expected, actual)}


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--prompt", type=int, default=256)
    prefix_options = parser.add_mutually_exclusive_group()
    prefix_options.add_argument("--prefix", type=int, help="single cached-prefix length (legacy option)")
    prefix_options.add_argument("--prefixes", nargs="+", type=int,
                                help="cached-prefix sweep; default: 256 2048 8192")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--samples", type=int, default=3,
                        help="independent synthetic histories per case; timing repeats are separate")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--kv-modes", nargs="+", choices=("off", "k", "v", "kv"),
                        default=["k", "v", "kv"],
                        help="decode-only staged FP8 cache interventions; off disables them")
    parser.add_argument("--combine-kv-with-projections", action="store_true",
                        help="also combine each selected projection variant with K+V FP8 storage")
    parser.add_argument("--native-fa3", action="store_true",
                        help="also test native FP8 Q/K/V attention in pure decode; no FP16 cache staging")
    parser.add_argument("--native-scale-headroom", type=float, default=1.25,
                        help="fixed KV scales calibrated from FP16 prefix only, with this range multiplier")
    parser.add_argument("--scenarios", nargs="+", choices=("prefill", "decode", "mixed"),
                        default=["prefill", "decode", "mixed"])
    parser.add_argument("--variants", nargs="*", type=parse_groups,
                        default=[("gate-up",), ("down",), ("output",), ("qkv",),
                                 ("gate-up", "down", "output", "qkv")])
    parser.add_argument("--atol", type=float)
    parser.add_argument("--rtol", type=float)
    parser.add_argument("--min-top1-agreement", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.prefixes = ([args.prefix] if args.prefix is not None else
                     list(dict.fromkeys(args.prefixes or [256, 2048, 8192])))
    if args.samples < 1:
        parser.error("samples must be positive")
    if args.warmup < 0 or args.repeats < 1:
        parser.error("warmup must be nonnegative and repeats positive")
    if not math.isfinite(args.native_scale_headroom) or args.native_scale_headroom < 1:
        parser.error("native scale headroom must be finite and >= 1")
    if "off" in args.kv_modes and len(args.kv_modes) != 1:
        parser.error("off cannot be combined with other cache modes")
    if min(args.batch, args.prompt, *args.prefixes) <= 0 or ("mixed" in args.scenarios and args.batch < 2):
        parser.error("positive dimensions and batch >= 2 for mixed are required")
    if max(args.prompt, max(args.prefixes) + 1) > 32768:
        parser.error("context exceeds model configuration")
    if (args.atol is None) != (args.rtol is None):
        parser.error("supply both --atol and --rtol")
    for value in (args.atol, args.rtol, args.min_top1_agreement):
        if value is not None and (not math.isfinite(value) or value < 0):
            parser.error("thresholds must be finite and nonnegative")
    if args.min_top1_agreement is not None and args.min_top1_agreement > 1:
        parser.error("agreement must be <= 1")
    if args.output.exists():
        parser.error("output already exists; use a fresh report path")
    return args


def case_plan(args):
    """Prefill has no cached prefix: don't duplicate it for every prefix length."""
    return [{"kind": kind, "prefix": prefix, "sample": sample, "seed": args.seed + sample}
            for kind in args.scenarios
            for prefix in ([args.prefixes[0]] if kind == "prefill" else args.prefixes)
            for sample in range(args.samples)]


def aggregate_accuracy(rows):
    """Weight top-1 agreement by request count, keeping contexts separate."""
    grouped = {}
    for row in rows:
        if row["variant"].startswith("fp16-control"):
            continue
        key = (row["scenario"], row["cached_prefix_tokens"], row["variant"])
        grouped.setdefault(key, []).append(row)
    summary = []
    for (scenario, prefix, variant), samples in sorted(grouped.items()):
        logits = [row["logits"] for row in samples]
        finite = all(metrics["finite"] for metrics in logits)
        total = sum(len(metrics.get("reference_tokens", [])) for metrics in logits)
        divergent = sum(len(metrics.get("divergent_rows", [])) for metrics in logits)
        summary.append({"scenario": scenario, "cached_prefix_tokens": prefix, "variant": variant,
                        "independent_histories": len(samples), "compared_requests": total,
                        "finite": finite,
                        "top1_agreement": 1. - divergent / total if finite and total else None,
                        "worst_max_abs_logit_error": max(metrics["max_abs"] for metrics in logits) if finite else None,
                        "median_relative_l2_logit_error": statistics.median(metrics["relative_l2"] for metrics in logits) if finite else None,
                        "median_wall_speedup": statistics.median(row["speedup_vs_opening_control"]["synchronized_wall"] for row in samples)})
    return summary


@torch.no_grad()
def main(argv=None):
    args = arguments(argv)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("H100/CUDA required; CPU results cannot validate FP8 kernels")
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) != (9, 0):
        raise RuntimeError("this experiment is scoped to Hopper SM90")
    import triton
    from benchmark_latest_vs_vllm import resolve_model_source
    from model_setup import load_model_only
    from model_adapter import allocate_pool
    from custom_kernels.paged_varlen_fa3 import smoke_varlen_fa3
    with strict_accumulation() as settings:
        gate = kernel_gate(device)
        smoke_varlen_fa3(device)
        native_gate = smoke_native(device) if args.native_fa3 else None
        source = resolve_model_source(args)
        loaded, _, _ = load_model_only(source, str(device), "float16")
        model = loaded.model
        report = {"status": "running", "model_source": str(source), "seed": args.seed,
                  "hardware": torch.cuda.get_device_name(device), "torch": torch.__version__,
                  "cuda": torch.version.cuda, "triton": triton.__version__,
                  "accumulation_settings": settings, "kernel_gate": gate,
                  "native_fa3_gate": native_gate,
                  "native_fa3": {"enabled": args.native_fa3, "scope": "pure decode",
                                 "kv_scales": "fixed per-sequence/KV-head from FP16 prefix only",
                                 "scale_headroom": args.native_scale_headroom,
                                 "q_scales": "dynamic per sequence/GQA group",
                                 "arithmetic": "external FA3 FP8 path; PyTorch matmul flags do not control its internal arithmetic",
                                 "output": "FA3 BF16 output cast to FP16",
                                 "workspace": "FP16 producer pool retained for new KV writes; no full-pool dequantization"},
                  "sweep": {"prefixes": args.prefixes, "samples": args.samples,
                            "prompt_tokens": args.prompt, "cases": case_plan(args),
                            "note": "decode context is cached prefix + 1; independent histories, not autoregressive rollouts"},
                  "quantization": {"format": "e4m3fn", "weights": "per-output-channel",
                                   "activations": "dynamic-per-token", "accumulator": "fp32",
                                   "max_num_imprecise_acc": 0, "output": "fp16"},
                  "thresholds": {"atol": args.atol, "rtol": args.rtol,
                                 "min_top1_agreement": args.min_top1_agreement},
                  "scope": "fixed-history full forwards; synthetic tokens; warm forward latency, not serving throughput",
                  "timing": {"warmup": args.warmup, "repeats": args.repeats,
                             "excludes": ["model/weight setup", "capture/JIT", "cache reset", "scheduler", "sampling", "accuracy diagnostics"],
                             "includes": ["adapter dispatch", "graph-input preparation", "activation quantization", "forward", "KV staging when enabled"],
                             "order": "control first, seeded shuffled candidates, control again"},
                  "kv_storage": {"format": "e4m3fn", "scale": "FP32 per token per KV head",
                                 "attention": "FP16 FA3 with full-pool dequantization staging",
                                 "native_fp8_attention": False,
                                 "limitation": "extra FP16 workspace retained; timings include staging and do not establish native FP8 cache bandwidth gains"},
                  "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in (Path(__file__).resolve(),
                                                 ROOT / "experiments/quant/projections.py",
                                                 ROOT / "experiments/quant/kernels/projections.py",
                                                 ROOT / "experiments/quant/staged_cache.py",
                                                 ROOT / "experiments/quant/kernels/staged_cache.py",
                                                 ROOT / "experiments/quant/native_fa3.py")},
                  "accumulation_note": "FP32 accumulators and accuracy-oriented FP8 promotion requested; hardware instructions not profiled",
                  "cases": []}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        for spec in case_plan(args):
            kind = spec["kind"]
            case = make_case(kind, args.batch, args.prompt, spec["prefix"],
                             model.cfg.vocab, spec["seed"], device)
            pool = allocate_pool(model.cfg, case["blocks"], device)
            for tensor in cache_state(pool):
                tensor.zero_()
            prepare_prefix(model, pool, case)
            seed = [tensor.clone() for tensor in cache_state(pool)]
            baseline_logits = baseline_kv = None
            candidates = [(groups, None) for groups in args.variants]
            if kind == "decode":
                candidates.extend(((), mode) for mode in args.kv_modes if mode != "off")
                if args.combine_kv_with_projections:
                    candidates.extend((groups, "kv") for groups in args.variants)
                if args.native_fa3:
                    candidates.append(((), "native"))
                    if args.combine_kv_with_projections:
                        candidates.extend((groups, "native") for groups in args.variants)
            random.Random(spec["seed"]).shuffle(candidates)
            baseline_timing = None
            for variant_index, (groups, kv_mode) in enumerate([((), None), *candidates, ((), None)]):
                label = ",".join(groups) if groups else "fp16-control"
                if kv_mode:
                    label = (label + "+" if groups else "") + "fp8-cache-" + kv_mode
                closing_control = variant_index == len(candidates) + 1
                if closing_control:
                    label = "fp16-control-closing"
                context = replace_projections(model, groups) if groups else nullcontext()
                native = kv_mode == "native"
                shadow = (NativeFP8Cache(pool, seed, case, args.native_scale_headroom) if native else
                          StagedFP8KVCache(pool, seed, kv_mode) if kv_mode else None)
                hook = shadow.attention_hook() if shadow else nullcontext()

                def reset():
                    restore_cache(pool, seed)
                    if shadow:
                        shadow.reset()

                with context, hook:
                    reset()
                    adapter, run = build_forward(model, pool, case)
                    run()  # Compile/capture first: warmup is allowed to mutate cache.
                    torch.cuda.synchronize(device)
                    reset()  # Identical history for the accuracy forward.
                    logits = run().clone()
                    expected_seed = shadow.expected_seed if shadow and not native else seed
                    writes, unchanged = capture_writes(pool, case, expected_seed)
                    if native:
                        writes = shadow.stored_writes(pool)
                        unchanged = unchanged and shadow.unchanged_prefix()
                    row = {"scenario": kind, "variant": label, "batch": args.batch,
                           "sample": spec["sample"], "input_seed": spec["seed"],
                           "cached_prefix_tokens": 0 if kind == "prefill" else spec["prefix"],
                           "tokens": case["ids"].numel(), "query_lengths": case["chunks"],
                           "contexts": case["contexts"].tolist(),
                           "untouched_cache_preserved": unchanged}
                    if variant_index == 0:
                        baseline_logits, baseline_kv = logits.clone(), writes.clone()
                        row["logits"] = compare_logits(logits, logits)
                        if not row["logits"]["finite"]:
                            raise RuntimeError("nonfinite FP16 control")
                    else:
                        row["logits"] = compare_logits(baseline_logits, logits,
                                                       args.atol, args.rtol, args.min_top1_agreement)
                    row["kv_writes"] = error_metrics(baseline_kv, writes)
                    layers = model.cfg.n_layers
                    row["kv_writes_by_kind"] = {
                        "k": error_metrics(baseline_kv[:layers], writes[:layers]),
                        "v": error_metrics(baseline_kv[layers:], writes[layers:])}
                    if shadow and not native:
                        row["cache_mode"] = kv_mode
                        row["cache_backend"] = "fp8-storage-to-fp16-fa3-staging"
                        # Only valid historical vectors; reserved zero slots must
                        # not dilute the average, and avoid copying entire pools.
                        row["prefix_quantization_by_layer"] = [
                            {"kind": "k" if index < layers else "v", "layer": index % layers,
                             **error_metrics(original.flatten(0, 1).index_select(0, case["history_slots"]),
                                             staged.flatten(0, 1).index_select(0, case["history_slots"]))}
                            for index, (original, staged) in enumerate(zip(seed, shadow.expected_seed))]
                        row["fp8_selected_storage_bytes_including_scales"] = shadow.storage_bytes()
                        row["extra_fp16_workspace_retained"] = True
                    if native:
                        row["cache_mode"] = "native-fp8-qkv"
                        row["cache_backend"] = "native-fa3-fp8-no-dequantization-staging"
                        row["new_kv_saturation"] = shadow.saturation(pool)
                        row["fp8_selected_storage_bytes_including_scales"] = shadow.storage_bytes()
                        row["fp16_producer_pool_retained"] = True
                    timing = measure_forward(run, reset, device, args.warmup, args.repeats)
                    row["latency"] = timing
                    if baseline_timing is None:
                        baseline_timing = timing
                    row["speedup_vs_opening_control"] = {
                        name: baseline_timing[name]["median_ms"] / timing[name]["median_ms"]
                        for name in ("cuda_event", "synchronized_wall")}
                    row["status"] = ("fail" if not unchanged or not row["kv_writes"]["finite"]
                                     else row["logits"]["status"])
                    report["cases"].append(row)
                    print(f"{kind}/prefix={row['cached_prefix_tokens']}/sample={spec['sample']}/{label}: {row['status']}; "
                          f"top1={row['logits']['top1_agreement']}; "
                          f"wall={timing['synchronized_wall']['median_ms']:.3f} ms", flush=True)
                    del adapter, run, logits, writes
                del shadow
                gc.collect()
                torch.cuda.empty_cache()
                report["accuracy_by_context"] = aggregate_accuracy(report["cases"])
                args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            del pool, seed, baseline_logits, baseline_kv
        report["status"] = "complete"
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 1 if any(row["status"] == "fail" for row in report["cases"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
