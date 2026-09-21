#!/usr/bin/env python3
"""One-command, two-model-load diagnosis of steady-state decode bottlenecks.

The local worker loads Qwen once and measures exact current graph steps, raw
attention scaling/page locality, and isolated fusion opportunities.  A separate
vLLM worker loads once for the entire shape surface.  Only one selected shape is
traced in each worker; all other cells use low-overhead CUDA-event/wall timing.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for directory in (HERE, ROOT / "benchmarks", ROOT / "baseline", ROOT / "engine/kvcache",
                  ROOT / "engine/model_runner", ROOT / "engine/graph", ROOT / "engine/cpp/build",
                  ROOT / "experiments/model", ROOT / "experiments/decode"):
    sys.path.insert(0, str(directory))

from benchmark_latest_vs_vllm import (MODEL, contract, load_frozen,
                                      resolve_model_source)
from fixed_regime import PREFILL_TOKENS_PER_STEP, get_fixed_case, get_fixed_shape, shape_summary
from profile_cpp_control import (cuda_activity_summary, drive,
                                 select_fixed_target, stage_summary)
from profile_latest_vs_vllm import (PINNED_VLLM, discover_vllm_target,
                                    compare_categories, measure_vllm_target,
                                    summarize_chrome_trace, verify_external_setup)

SLOPE_SHAPES = (
    "fixed-b8-l256-o128", "fixed-b8-l2048-o128", "probe-b8-l4096-o256",
    "fixed-b64-l256-o128", "fixed-b64-l2048-o128", "probe-b64-l4096-o256",
)
POLICIES = ("production", "splitk")
DECODE_BATCHES = (8, 64)
OUTPUT_HEAD_CONFIGS = {
    "n64_k64_w4_s3": {"block_n": 64, "block_k": 64, "num_warps": 4, "num_stages": 3},
    "n128_k64_w8_s3": {"block_n": 128, "block_k": 64, "num_warps": 8, "num_stages": 3},
    "n256_k64_w8_s3": {"block_n": 256, "block_k": 64, "num_warps": 8, "num_stages": 3},
}

# Exact source hashes from c54bfe3.  That run could safely complete every
# component preceding unrolled_decode, but an invalid fused token could poison
# CUDA while capturing the first K-step component.  Keep this one migration
# explicit and narrow: it is not a general bypass for protocol mismatches.
PRE_UNROLLED_HOTFIX_SOURCE_HASHES = {
    "experiments/integration/profile_decode_comprehensive.py":
        "21ff5ce722f62cc89b343014c9667c0f25badb5d770964e2bbffcfd022822c17",
    "custom_kernels/fused_lm_head.py":
        "01559568ed90dff57dc6326e9ded938260a53492c6a182bf8cab98faa7f377b4",
    "experiments/decode/benchmark_unrolled_graph.py":
        "e4fb53f4ffea259e4f1bd25844b9f997718c60d8733d2c71741df78b7acca259",
}
PRE_UNROLLED_COMPONENTS = {
    "attention", "fusion", "gemm", "output_head", "output_head_full_model",
}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "check-setup", "run", "run-local",
                                           "run-vllm", "analyze"))
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "experiments/results/comprehensive-decode-profile")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--occurrence", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--kernel-repetitions", type=int, default=20)
    parser.add_argument("--l2-evict-mib", type=int, default=128,
                        help="Scratch bytes touched before cache-evicted GEMM samples")
    parser.add_argument("--eos-token-id", type=int, default=151645,
                        help="EOS ID scanned across every K-step validation trajectory")
    parser.add_argument("--trace-shape", choices=SLOPE_SHAPES,
                        default="probe-b64-l4096-o256")
    parser.add_argument("--retry-failed", action="store_true")
    return parser


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def protocol_fingerprint(args):
    sources = (
        Path(__file__), HERE / "profile_cpp_control.py",
        HERE / "profile_latest_vs_vllm.py", HERE / "model_adapter.py",
        ROOT / "engine/graph/paged_graph_decoder.py",
        ROOT / "engine/graph/unrolled_graph_decoder.py",
        ROOT / "engine/kvcache/paged_decode_attention.py",
        ROOT / "custom_kernels/rope_kv_write.py",
        ROOT / "custom_kernels/swiglu.py", ROOT / "custom_kernels/fused_lm_head.py",
        ROOT / "baseline/naive_forward.py",
        ROOT / "experiments/decode/benchmark_unrolled_graph.py",
    )
    protocol = {
        "schema_version": 1, "shapes": list(SLOPE_SHAPES),
        "policies": list(POLICIES), "model": args.model, "device": args.device,
        "seed": args.seed, "occurrence": args.occurrence, "warmups": args.warmups,
        "repetitions": args.repetitions,
        "kernel_repetitions": args.kernel_repetitions,
        "l2_evict_mib": args.l2_evict_mib,
        "eos_token_id": args.eos_token_id,
        "trace_shape": args.trace_shape,
        "local_budgets": {shape: local_budget(shape) for shape in SLOPE_SHAPES},
        "vllm_prefill_budget": PREFILL_TOKENS_PER_STEP,
        "sources_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
    }
    encoded = json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), protocol


def pre_unrolled_hotfix_fingerprint(args):
    """Fingerprint of the sole checkpoint version accepted by this hotfix."""
    _, protocol = protocol_fingerprint(args)
    protocol = {**protocol, "sources_sha256": dict(protocol["sources_sha256"])}
    protocol["sources_sha256"].update(PRE_UNROLLED_HOTFIX_SOURCE_HASHES)
    encoded = json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def migrate_pre_unrolled_checkpoint(path, saved, current, predecessor):
    """Upgrade a verified c54bfe3 pre-unrolled checkpoint in place."""
    if saved.get("fingerprint") == current:
        return saved
    if saved.get("fingerprint") != predecessor:
        raise ValueError(f"checkpoint uses an incompatible protocol: {path}")
    saved["fingerprint"] = current
    atomic_json(path, saved)
    print(f"migrated compatible pre-unrolled checkpoint: {path}", flush=True)
    return saved


def local_budget(shape_id):
    return 4096 if get_fixed_shape(shape_id)["batch"] == 8 else 8192


def checkpoint_args(args, shape_id):
    return argparse.Namespace(
        shape_id=shape_id, output_dir=args.suite_dir / shape_id,
        seed=args.seed, model=args.model, device=args.device, logit_atol=.05,
        trials=1, samples=1, repetitions=1, warmups=0, profile_occurrence=0,
    )


def validate_args(args):
    if args.device != "cuda:0":
        raise ValueError("comprehensive profile is pinned to cuda:0")
    if (args.occurrence < 0 or args.warmups < 0 or args.repetitions < 1
            or args.kernel_repetitions < 1 or args.l2_evict_mib < 1
            or args.eos_token_id < -1):
        raise ValueError("invalid occurrence, warmup, or repetition count")
    args.suite_dir = args.suite_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    contracts = {}
    for shape_id in SLOPE_SHAPES:
        child = checkpoint_args(args, shape_id)
        case, blocks = contract(child)
        load_frozen(child, case)
        contracts[shape_id] = (case, blocks)
    return contracts


def verify_setup(args):
    from model_setup import check_startup
    return {"external": verify_external_setup(args),
            "local": check_startup(args.device)}


def timed_cuda(torch, operation, repetitions, warmups=1, before_each=None):
    for _ in range(warmups):
        operation()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repetitions):
        if before_each is not None:
            before_each()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return {"median_ms": statistics.median(samples), "samples_ms": samples}


def capture_operation(torch, operation, warmups):
    """Capture one exact-shape operation after warming its CUDA library state."""
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        for _ in range(max(1, warmups)):
            operation()
    torch.cuda.current_stream().wait_stream(side_stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = operation()
    return graph, output


def available_blas_routes(torch):
    """Return supported PyTorch BLAS routing choices without assuming an API version."""
    selector = getattr(torch.backends.cuda, "preferred_blas_library", None)
    if selector is None:
        return [("current", None)], None
    original = selector()
    routes = [("current", None)]
    for route in ("cublas", "cublaslt"):
        try:
            selector(route)
        except (RuntimeError, ValueError):
            continue
        routes.append((route, route))
    selector(original)
    return routes, original


def gemm_operations(torch, engine, batch, generator):
    """Use production packed weights at the exact fixed-regime decode batch."""
    layer = engine.model.layers[0]
    hidden = torch.randn(batch, engine.cfg.d_model, device=engine.device,
                         dtype=torch.float16, generator=generator)
    activated = torch.randn(batch, engine.cfg.d_ff, device=engine.device,
                            dtype=torch.float16, generator=generator)
    operations = {
        "qkv_packed": (lambda: layer.qkv_proj(hidden), layer.qkv_proj),
        "attention_output": (lambda: layer.o_proj(hidden), layer.o_proj),
        "gate_up_packed": (lambda: layer.gate_up_proj(hidden), layer.gate_up_proj),
        "mlp_down": (lambda: layer.down_proj(activated), layer.down_proj),
        "lm_head_logits": (lambda: engine.model.lm_head(hidden), engine.model.lm_head),
        "lm_head_logits_argmax": (
            lambda: engine.model.lm_head(hidden).argmax(dim=-1), engine.model.lm_head),
    }
    details = {}
    for name, (_, linear) in operations.items():
        details[name] = {
            "input_shape": list(hidden.shape if name != "mlp_down" else activated.shape),
            "weight_shape": list(linear.weight.shape),
            "bias": linear.bias is not None,
        }
    return {name: operation for name, (operation, _) in operations.items()}, details


def gemm_sweep(torch, engine, args):
    """Screen exact production GEMMs under warm/cold cache and eager/graph execution."""
    routes, original_route = available_blas_routes(torch)
    selector = getattr(torch.backends.cuda, "preferred_blas_library", None)
    eviction_elements = args.l2_evict_mib * 1024 * 1024 // 2
    eviction = torch.empty(eviction_elements, device=args.device, dtype=torch.float16)

    def evict_l2():
        # The event is recorded after this same-stream write, so eviction work is
        # excluded while the following GEMM sees cold(er) weights and inputs.
        eviction.add_(1)

    rows = []
    try:
        for batch in DECODE_BATCHES:
            generator = torch.Generator(device=args.device).manual_seed(
                args.seed + batch * 104729)
            operations, details = gemm_operations(torch, engine, batch, generator)
            if selector is not None:
                selector(original_route)
            references = {name: operation().detach().clone()
                          for name, operation in operations.items()}
            measured = {}
            for route_name, route in routes:
                if selector is not None:
                    selector(original_route if route is None else route)
                route_rows = {}
                for name, operation in operations.items():
                    try:
                        reference = operation().detach().clone()
                        baseline = references[name]
                        route_error = float(
                            (reference.float() - baseline.float()).abs().max())
                        try:
                            torch.testing.assert_close(
                                reference.float(), baseline.float(), atol=.05, rtol=.01)
                            route_correctness = {"status": "pass", "error": None}
                        except AssertionError as exc:
                            route_correctness = {"status": "fail", "error": str(exc)}
                        eager_warm = timed_cuda(
                            torch, operation, args.kernel_repetitions, args.warmups)
                        eager_evicted = timed_cuda(
                            torch, operation, args.kernel_repetitions, args.warmups,
                            before_each=evict_l2)
                        graph, graph_output = capture_operation(
                            torch, operation, args.warmups)
                        graph.replay()
                        torch.cuda.synchronize()
                        graph_error = float(
                            (graph_output.float() - reference.float()).abs().max())
                        try:
                            torch.testing.assert_close(
                                graph_output.float(), reference.float(), atol=.05, rtol=.01)
                            correctness = {"status": "pass", "error": None}
                        except AssertionError as exc:
                            correctness = {"status": "fail", "error": str(exc)}
                        graph_warm = timed_cuda(
                            torch, graph.replay, args.kernel_repetitions, args.warmups)
                        graph_evicted = timed_cuda(
                            torch, graph.replay, args.kernel_repetitions, args.warmups,
                            before_each=evict_l2)
                        route_rows[name] = {
                            "status": "complete", "eager_warm": eager_warm,
                            "eager_cache_evicted": eager_evicted,
                            "graph_warm": graph_warm,
                            "graph_cache_evicted": graph_evicted,
                            "route_correctness": route_correctness,
                            "max_route_error": route_error,
                            "graph_correctness": correctness,
                            "max_graph_error": graph_error,
                        }
                        del graph, graph_output, reference
                    except (RuntimeError, ValueError) as exc:
                        # Backend routing is a candidate screen. An unsupported
                        # route is data, and must not discard other route timings.
                        route_rows[name] = {"status": "unsupported", "error": str(exc)}
                    torch.cuda.empty_cache()
                measured[route_name] = route_rows

            layer_names = ("qkv_packed", "attention_output", "gate_up_packed", "mlp_down")
            route_estimates = {}
            for route_name, route_rows in measured.items():
                complete = {name: row for name, row in route_rows.items()
                            if row["status"] == "complete"}
                estimates = {}
                for mode in ("eager_warm", "eager_cache_evicted",
                             "graph_warm", "graph_cache_evicted"):
                    if all(name in complete for name in (*layer_names, "lm_head_logits")):
                        layer_ms = sum(complete[name][mode]["median_ms"]
                                       for name in layer_names)
                        total_ms = engine.cfg.n_layers * layer_ms + complete[
                            "lm_head_logits"][mode]["median_ms"]
                        estimates[mode] = {
                            "one_layer_projection_ms": layer_ms,
                            "projected_28_layer_plus_lm_head_ms": total_ms,
                            "lm_head_fraction": (
                                complete["lm_head_logits"][mode]["median_ms"] / total_ms),
                        }
                route_estimates[route_name] = estimates
            best_routes = {}
            for mode in ("eager_warm", "eager_cache_evicted",
                         "graph_warm", "graph_cache_evicted"):
                candidates = []
                for route_name, estimates in route_estimates.items():
                    if mode not in estimates:
                        continue
                    required = (*layer_names, "lm_head_logits")
                    rows_for_route = measured[route_name]
                    if not all(rows_for_route[name]["route_correctness"]["status"] == "pass"
                               for name in required):
                        continue
                    if mode.startswith("graph") and not all(
                            rows_for_route[name]["graph_correctness"]["status"] == "pass"
                            for name in required):
                        continue
                    candidates.append((
                        estimates[mode]["projected_28_layer_plus_lm_head_ms"], route_name))
                if candidates:
                    best_ms, best_route = min(candidates)
                    current_ms = route_estimates["current"].get(mode, {}).get(
                        "projected_28_layer_plus_lm_head_ms")
                    best_routes[mode] = {
                        "route": best_route, "projected_ms": best_ms,
                        "speedup_over_current": (current_ms / best_ms
                                                   if current_ms is not None else None),
                    }
            rows.append({
                "batch": batch, "capture_batch": batch,
                "capture_policy": "exact fixed-regime batch; no power-of-two ladder",
                "l2_evict_mib": args.l2_evict_mib,
                "operation_shapes": details, "routes": measured,
                "route_estimates": route_estimates,
                "current_route_estimates": route_estimates["current"],
                "best_correct_route_by_mode": best_routes,
            })
            del references
    finally:
        if selector is not None:
            selector(original_route)
        del eviction
        torch.cuda.empty_cache()
    return rows


def output_head_sweep(torch, engine, args):
    """Measure every exact-greedy output-head intervention at B8 and B64."""
    from kernel_dispatch import chunked_lm_head_argmax, fused_lm_head_argmax

    weight = engine.model.lm_head.weight
    vocab, hidden_size = weight.shape
    eviction = torch.empty(args.l2_evict_mib * 1024 * 1024 // 2,
                           device=args.device, dtype=torch.float16)

    def evict_l2():
        eviction.add_(1)

    rows = []
    for batch in DECODE_BATCHES:
        generator = torch.Generator(device=args.device).manual_seed(
            args.seed + batch * 130363)
        hidden = torch.randn(batch, hidden_size, device=args.device,
                             dtype=torch.float16, generator=generator)
        fused_resources = {}
        for config_name, config in OUTPUT_HEAD_CONFIGS.items():
            blocks = (vocab + config["block_n"] - 1) // config["block_n"]
            fused_resources[config_name] = {
                "workspace": (
                    torch.empty((batch, blocks), device=args.device, dtype=torch.float16),
                    torch.empty((batch, blocks), device=args.device, dtype=torch.int32),
                    torch.empty(batch, device=args.device, dtype=torch.float16),
                ),
                "output": torch.empty(batch, device=args.device, dtype=torch.int64),
            }
        operations = {
            "materialized_logits": lambda: engine.model.lm_head(hidden),
            "materialized_logits_argmax": lambda: engine.model.lm_head(hidden).argmax(-1),
            "chunked_logits_argmax": lambda: chunked_lm_head_argmax(
                hidden, weight, chunk_size=8192),
        }
        for config_name, config in OUTPUT_HEAD_CONFIGS.items():
            resources = fused_resources[config_name]
            operations[f"fused_projection_argmax_{config_name}"] = (
                lambda config=config, resources=resources: fused_lm_head_argmax(
                    hidden, weight, block_m=(16 if batch == 8 else 64),
                    workspace=resources["workspace"], output=resources["output"],
                    **config))
        reference_logits = operations["materialized_logits"]()
        reference_tokens = reference_logits.argmax(-1)
        top2 = reference_logits.float().topk(2, dim=-1).values
        margins = top2[:, 0] - top2[:, 1]
        measurements = {}
        for name, operation in operations.items():
            try:
                observed = operation()
                if name == "materialized_logits":
                    correctness = {"status": "reference", "matching_tokens": batch,
                                   "total_tokens": batch, "error": None}
                else:
                    observed_tokens = observed.reshape(-1)
                    different = observed_tokens != reference_tokens
                    mismatch_rows = different.nonzero().flatten()
                    correctness = {
                        "status": "pass" if not bool(different.any()) else "fail",
                        "matching_tokens": int((~different).sum()),
                        "total_tokens": batch,
                        "error": None,
                        "mismatch_rows": [int(value) for value in mismatch_rows.cpu().tolist()],
                        "mismatch_reference_margins": [
                            float(margins[index]) for index in mismatch_rows.cpu().tolist()
                        ],
                    }
                eager_warm = timed_cuda(
                    torch, operation, args.kernel_repetitions, args.warmups)
                eager_evicted = timed_cuda(
                    torch, operation, args.kernel_repetitions, args.warmups,
                    before_each=evict_l2)
                graph, graph_output = capture_operation(torch, operation, args.warmups)
                graph.replay()
                torch.cuda.synchronize()
                if name == "materialized_logits":
                    graph_tokens = graph_output.argmax(-1)
                else:
                    graph_tokens = graph_output.reshape(-1)
                graph_matches = int((graph_tokens == reference_tokens).sum())
                graph_correctness = {
                    "status": "pass" if graph_matches == batch else "fail",
                    "matching_tokens": graph_matches, "total_tokens": batch,
                }
                graph_warm = timed_cuda(
                    torch, graph.replay, args.kernel_repetitions, args.warmups)
                graph_evicted = timed_cuda(
                    torch, graph.replay, args.kernel_repetitions, args.warmups,
                    before_each=evict_l2)
                measurements[name] = {
                    "status": "complete", "correctness": correctness,
                    "graph_correctness": graph_correctness,
                    "eager_warm": eager_warm, "eager_cache_evicted": eager_evicted,
                    "graph_warm": graph_warm, "graph_cache_evicted": graph_evicted,
                }
                del graph, graph_output, observed
            except (RuntimeError, ValueError) as exc:
                measurements[name] = {"status": "unsupported", "error": str(exc)}
            torch.cuda.empty_cache()
        weight_bytes = weight.numel() * weight.element_size()
        logits_bytes = batch * vocab * hidden.element_size()
        baseline = measurements.get("materialized_logits_argmax", {})
        derived = {
            "weight_bytes": weight_bytes,
            "materialized_logits_bytes": logits_bytes,
            "fused_partial_workspace_bytes": {},
            "estimated_materialized_argmax_traffic_bytes": weight_bytes + 2 * logits_bytes,
            "estimated_fused_traffic_bytes": {},
        }
        for config_name, config in OUTPUT_HEAD_CONFIGS.items():
            blocks = (vocab + config["block_n"] - 1) // config["block_n"]
            derived["fused_partial_workspace_bytes"][config_name] = (
                batch * blocks * (hidden.element_size() + 4) + batch * hidden.element_size())
            derived["estimated_fused_traffic_bytes"][config_name] = (
                weight_bytes + 2 * derived["fused_partial_workspace_bytes"][config_name])
        for measurement in measurements.values():
            if measurement.get("status") != "complete":
                continue
            for mode in ("eager_warm", "eager_cache_evicted",
                         "graph_warm", "graph_cache_evicted"):
                measurement[mode]["effective_weight_gbps"] = (
                    weight_bytes / (measurement[mode]["median_ms"] * 1e-3) / 1e9)
        best = {}
        if baseline.get("status") == "complete":
            for mode in ("eager_warm", "eager_cache_evicted",
                         "graph_warm", "graph_cache_evicted"):
                candidates = []
                for config_name in OUTPUT_HEAD_CONFIGS:
                    name = f"fused_projection_argmax_{config_name}"
                    candidate = measurements.get(name, {})
                    if (candidate.get("status") == "complete"
                            and candidate["correctness"]["status"] == "pass"
                            and candidate["graph_correctness"]["status"] == "pass"):
                        candidates.append((candidate[mode]["median_ms"], config_name))
                if candidates:
                    candidate_ms, config_name = min(candidates)
                    best[mode] = {
                        "config": config_name, "median_ms": candidate_ms,
                        "speedup": baseline[mode]["median_ms"] / candidate_ms,
                    }
        derived["best_correct_fused_by_mode"] = best
        rows.append({
            "batch": batch, "hidden_size": hidden_size, "vocab": vocab,
            "capture_batch": batch,
            "capture_policy": "exact fixed-regime batch; no power-of-two ladder",
            "measurements": measurements, "derived": derived,
            "reference_top1_margin": {
                "minimum": float(margins.min()),
                "median": float(margins.median()),
                "maximum": float(margins.max()),
            },
        })
        del hidden, fused_resources, reference_logits, reference_tokens, top2, margins
        torch.cuda.empty_cache()
    del eviction
    torch.cuda.empty_cache()
    return rows


def output_head_full_model_sweep(torch, engine, args):
    """A/B token-producing output heads inside the complete split-K decode path."""
    from benchmark_packed_projection_decode import stage_case
    from paged_graph_decoder import graph_decode_forward

    rows = []
    context = 2048
    for batch in DECODE_BATCHES:
        cache, tensors = stage_case(
            torch, engine.cfg, batch, context, torch.float16,
            args.seed + batch * 15485863 + context, args.device,
        )

        def materialized():
            return graph_decode_forward(
                engine.model, cache, *tensors, decode_attention_policy="splitk",
                max_decode_context_length=context + 1,
            ).argmax(-1).reshape(-1)

        def fused(config):
            return graph_decode_forward(
                engine.model, cache, *tensors, decode_attention_policy="splitk",
                max_decode_context_length=context + 1,
                output_head_policy="fused_argmax",
                output_head_config={
                    "block_m": 16 if batch == 8 else 64,
                    **config,
                },
            ).reshape(-1)

        reference = materialized().detach().clone()
        eager_materialized = timed_cuda(
            torch, materialized, args.repetitions, args.warmups)
        base_graph, base_output = capture_operation(torch, materialized, args.warmups)
        base_graph.replay()
        torch.cuda.synchronize()
        base_graph_matches = int((base_output == reference).sum())
        graph_materialized = timed_cuda(
            torch, base_graph.replay, args.repetitions, args.warmups)
        row = {
            "batch": batch, "context": context,
            "materialized_logits_argmax": {
                "eager": eager_materialized, "graph": graph_materialized},
            "scope": "complete 28-layer split-K decode; token output only",
        }
        candidates = {}
        for config_name, config in OUTPUT_HEAD_CONFIGS.items():
            operation = lambda config=config: fused(config)
            try:
                candidate = operation().detach().clone()
                matching = int((candidate == reference).sum())
                eager_fused = timed_cuda(
                    torch, operation, args.repetitions, args.warmups)
                fused_graph, fused_output = capture_operation(
                    torch, operation, args.warmups)
                fused_graph.replay()
                torch.cuda.synchronize()
                fused_graph_matches = int((fused_output == reference).sum())
                graph_fused = timed_cuda(
                    torch, fused_graph.replay, args.repetitions, args.warmups)
                status = ("pass" if matching == fused_graph_matches == batch else "fail")
                candidates[config_name] = {
                    "status": "complete",
                    "correctness": {
                        "eager_matching_tokens": matching,
                        "fused_graph_matching_tokens": fused_graph_matches,
                        "total_tokens": batch, "status": status,
                    },
                    "eager": eager_fused, "graph": graph_fused,
                    "speedup": {
                        "eager": eager_materialized["median_ms"] / eager_fused["median_ms"],
                        "graph": graph_materialized["median_ms"] / graph_fused["median_ms"],
                    },
                }
                del candidate, fused_graph, fused_output
            except (RuntimeError, ValueError) as exc:
                candidates[config_name] = {"status": "unsupported", "error": str(exc)}
        valid = [(candidate["graph"]["median_ms"], name)
                 for name, candidate in candidates.items()
                 if (candidate.get("status") == "complete"
                     and candidate["correctness"]["status"] == "pass"
                     and base_graph_matches == batch)]
        row["candidates"] = candidates
        row["correctness"] = {
            "materialized_graph_matching_tokens": base_graph_matches,
            "total_tokens": batch,
        }
        if valid:
            _, best_name = min(valid)
            row.update({
                "status": "complete", "best_config": best_name,
                "speedup": candidates[best_name]["speedup"],
                "correctness": candidates[best_name]["correctness"] | {
                    "materialized_graph_matching_tokens": base_graph_matches},
            })
        else:
            row["status"] = ("no_correct_candidate"
                             if any(candidate.get("status") == "complete"
                                    for candidate in candidates.values())
                             else "unsupported")
        rows.append(row)
        del cache, tensors, reference, base_graph, base_output
        torch.cuda.empty_cache()
    return rows


def capture_with_memory(torch, factory):
    """Capture one graph and record startup plus incremental allocator pressure."""
    torch.cuda.synchronize()
    before_allocated = torch.cuda.memory_allocated()
    before_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    value = factory()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return value, {
        "capture_seconds": elapsed,
        "allocated_delta_bytes": torch.cuda.memory_allocated() - before_allocated,
        "reserved_delta_bytes": torch.cuda.memory_reserved() - before_reserved,
        "peak_increment_bytes": max(
            0, torch.cuda.max_memory_allocated() - before_allocated),
    }


def unrolled_decode_sweep(torch, engine, args, output_head_rows):
    """Compare repeated K1 replay with exact K2/K4/K8 autoregressive graphs."""
    from benchmark_unrolled_graph import (
        UNROLL_STEPS, eager_trajectory, future_slots,
        initialize_cache, kv_accuracy, restore_slots, snapshot_slots,
        stage_unrolled_case, timed_cuda_wall, trajectory_accuracy,
    )
    from paged_graph_decoder import CUDAGraphDecoder
    from unrolled_graph_decoder import UnrolledCUDAGraphDecoder, first_eos_positions

    if args.eos_token_id >= engine.cfg.vocab:
        raise ValueError("EOS token ID exceeds the model vocabulary")
    best_head_config = {}
    for row in output_head_rows:
        best = row["derived"]["best_correct_fused_by_mode"].get("graph_warm")
        if best is not None:
            best_head_config[row["batch"]] = {
                "block_m": 16 if row["batch"] == 8 else 64,
                **OUTPUT_HEAD_CONFIGS[best["config"]],
            }

    results = []
    max_steps = max(UNROLL_STEPS)
    for batch in DECODE_BATCHES:
        for context in (256, 2048, 4096):
            cache, metadata, first_ids, generator = stage_unrolled_case(
                torch, engine.cfg, batch, context, max_steps, torch.float16,
                args.seed + batch * 32452843 + context, args.device,
            )
            max_blocks = metadata[0][2].shape[1]
            single, single_capture = capture_with_memory(
                torch,
                lambda: CUDAGraphDecoder(
                    engine.model, cache, batch, max_blocks, engine.device,
                    torch.float16, decode_attention_policy="splitk",
                    max_decode_context_length=context + max_steps,
                ).capture(),
            )
            decoders = {}
            captures = {"repeated_k1": single_capture}
            for steps in UNROLL_STEPS:
                selected = metadata[:steps]
                print(f"capturing unrolled validation B={batch} C={context} K={steps}",
                      flush=True)
                validation, validation_capture = capture_with_memory(
                    torch,
                    lambda selected=selected: UnrolledCUDAGraphDecoder(
                        engine.model, cache, selected,
                        decode_attention_policy="splitk",
                        max_decode_context_length=context + steps,
                        retain_logits=True, output_head_policy="logits",
                    ).capture(),
                )
                print(f"capturing unrolled production B={batch} C={context} K={steps}",
                      flush=True)
                production, production_capture = capture_with_memory(
                    torch,
                    lambda selected=selected: UnrolledCUDAGraphDecoder(
                        engine.model, cache, selected,
                        decode_attention_policy="splitk",
                        max_decode_context_length=context + steps,
                        retain_logits=False, output_head_policy="logits",
                    ).capture(),
                )
                fused = fused_capture = None
                if batch in best_head_config:
                    try:
                        print(f"capturing unrolled fused-head B={batch} "
                              f"C={context} K={steps}", flush=True)
                        fused, fused_capture = capture_with_memory(
                            torch,
                            lambda selected=selected: UnrolledCUDAGraphDecoder(
                                engine.model, cache, selected,
                                decode_attention_policy="splitk",
                                max_decode_context_length=context + steps,
                                retain_logits=False, output_head_policy="fused_argmax",
                                output_head_config=best_head_config[batch],
                            ).capture(),
                        )
                    except (RuntimeError, ValueError) as exc:
                        fused_capture = {"status": "unsupported", "error": str(exc)}
                decoders[steps] = {"validation": validation,
                                   "production": production, "fused": fused}
                captures[f"k{steps}_validation"] = validation_capture
                captures[f"k{steps}_production"] = production_capture
                captures[f"k{steps}_fused"] = fused_capture

            # Capture warmups intentionally use zero-filled sacrificial state.
            # Install the deterministic staged context only after every graph is frozen.
            initialize_cache(torch, cache, generator)
            case_rows = []
            for steps in UNROLL_STEPS:
                selected = metadata[:steps]
                slots = future_slots(selected)
                initial_kv = snapshot_slots(cache, slots)
                reference_tokens, reference_logits = eager_trajectory(
                    engine.model, cache, first_ids, selected)
                torch.cuda.synchronize()
                reference_kv = snapshot_slots(cache, slots)
                restore_slots(cache, slots, initial_kv)

                validation = decoders[steps]["validation"]
                candidate_tokens, candidate_logits = validation.replay(first_ids)
                torch.cuda.synchronize()
                candidate_kv = snapshot_slots(cache, slots)
                logit_accuracy = trajectory_accuracy(
                    torch, candidate_tokens, candidate_logits,
                    reference_tokens, reference_logits, atol=.05)
                validation_kv = kv_accuracy(torch, candidate_kv, reference_kv, atol=.05)
                restore_slots(cache, slots, initial_kv)

                production = decoders[steps]["production"]
                production_tokens, production_logits = production.replay(first_ids)
                torch.cuda.synchronize()
                if production_logits:
                    raise AssertionError("production K-step graph retained validation logits")
                production_matching = int((production_tokens == reference_tokens).sum())
                production_kv = kv_accuracy(
                    torch, snapshot_slots(cache, slots), reference_kv, atol=.05)
                restore_slots(cache, slots, initial_kv)

                def repeated_k1():
                    token = first_ids
                    for positions, lengths, table, step_slots in selected:
                        logits = single.decode(
                            token, positions, lengths, table, step_slots)
                        token = logits.argmax(-1).reshape(batch, 1)
                    return token

                repeated_timing = timed_cuda_wall(
                    torch, repeated_k1, args.repetitions, args.warmups)
                validation_timing = timed_cuda_wall(
                    torch, lambda: validation.replay(first_ids),
                    args.repetitions, args.warmups)
                production_timing = timed_cuda_wall(
                    torch, lambda: production.replay(first_ids),
                    args.repetitions, args.warmups)
                token_d2h_timing = timed_cuda_wall(
                    torch, lambda: production.s_tokens.cpu(),
                    args.repetitions, args.warmups)
                eos_scan_timing = timed_cuda_wall(
                    torch,
                    lambda: first_eos_positions(
                        production.s_tokens, args.eos_token_id).cpu(),
                    args.repetitions, args.warmups,
                )

                fused_result = None
                fused = decoders[steps]["fused"]
                if fused is not None:
                    restore_slots(cache, slots, initial_kv)
                    fused_tokens, fused_logits = fused.replay(first_ids)
                    torch.cuda.synchronize()
                    if fused_logits:
                        raise AssertionError("fused K-step graph retained validation logits")
                    fused_matching = int((fused_tokens == reference_tokens).sum())
                    first_difference = next(
                        (index for index in range(steps)
                         if not torch.equal(fused_tokens[index], reference_tokens[index])), None)
                    fused_result = {
                        "matching_tokens": fused_matching,
                        "total_tokens": reference_tokens.numel(),
                        "first_divergence_step": first_difference,
                        "kv": kv_accuracy(
                            torch, snapshot_slots(cache, slots), reference_kv, atol=.05),
                        "timing": timed_cuda_wall(
                            torch, lambda: fused.replay(first_ids),
                            args.repetitions, args.warmups),
                    }
                eos_positions = first_eos_positions(reference_tokens, args.eos_token_id)
                case_rows.append({
                    "steps": steps,
                    "validation": {"logits": logit_accuracy, "kv": validation_kv,
                                   "timing": validation_timing,
                                   "retained_logits_bytes": sum(
                                       tensor.numel() * tensor.element_size()
                                       for tensor in candidate_logits)},
                    "production_materialized_head": {
                        "matching_tokens": production_matching,
                        "total_tokens": reference_tokens.numel(), "kv": production_kv,
                        "timing": production_timing, "retained_logits_bytes": 0,
                    },
                    "production_fused_head": fused_result,
                    "repeated_k1": {"timing": repeated_timing},
                    "speedup": {
                        "validation_wall": (repeated_timing["wall_median_ms"]
                                            / validation_timing["wall_median_ms"]),
                        "production_wall": (repeated_timing["wall_median_ms"]
                                            / production_timing["wall_median_ms"]),
                        "production_per_token_ms": (
                            production_timing["wall_median_ms"] / steps),
                    },
                    "eos": {
                        "token_id": args.eos_token_id,
                        "first_positions": [int(value) for value in eos_positions.cpu().tolist()],
                        "rows_with_eos": int((eos_positions < steps).sum()),
                        "token_d2h_timing": token_d2h_timing,
                        "gpu_scan_and_d2h_timing": eos_scan_timing,
                        "contract": "commit through first EOS; discard later generated tokens",
                    },
                    "capture": {
                        "validation": captures[f"k{steps}_validation"],
                        "production": captures[f"k{steps}_production"],
                        "fused": captures[f"k{steps}_fused"],
                    },
                })
                restore_slots(cache, slots, initial_kv)
                del (initial_kv, reference_kv, candidate_kv,
                     reference_tokens, reference_logits)
                torch.cuda.empty_cache()
            results.append({
                "batch": batch, "context": context,
                "capture_policy": "exact B/context with K=2,4,8",
                "single_graph_capture": single_capture,
                "rows": case_rows,
            })
            del cache, metadata, first_ids, single, decoders
            torch.cuda.empty_cache()
    return results


def summarize_context(call):
    values = sorted(call["context_lengths"])
    return {"minimum": values[0], "median": statistics.median(values),
            "maximum": values[-1], "values": values}


def poison_pool(torch, pool):
    # Only poison once per shape/policy correctness run. This catches missing KV
    # writes without adding work to measured target steps.
    for tensor in pool.k_pool + pool.v_pool:
        tensor.fill_(float("nan"))
    torch.cuda.synchronize()


def measure_local_arm(torch, cpp, engine, pool, case, requests, args, policy, trace):
    from benchmark_scheduler_decode import execute, make_config
    from model_adapter import PiecewiseGraphModelAdapter

    budget = case["prefill_budget"]
    config = make_config(cpp, case)
    adapter = PiecewiseGraphModelAdapter(
        engine.model, pool, None, max_running=config.max_batch_size,
        max_context_length=config.max_context_length,
        decode_attention_policy=policy, max_capture_tokens=budget,
        max_prefill_shapes=1, prefill_buckets=[budget],
        # Only the full-cohort graph is measured. It is byte-for-byte the same
        # graph as the production bucket while avoiding six unused captures.
        decode_buckets=[config.max_batch_size],
    )
    poison_pool(torch, pool)
    preflight = execute(torch, cpp.IterationLoop(config, torch.device(args.device)),
                        adapter, requests)
    target = select_fixed_target(preflight["steps"], case, "decode", args.occurrence, budget)
    expected_steps = [(step["kind"], step["calls"], step["completed"])
                      for step in preflight["steps"]]
    expected_outputs = preflight["outputs"]

    def checked(profile=False, metadata=False):
        result = drive(torch, cpp, config, requests, adapter, target,
                       profile_target=profile, collect_target_metadata=metadata)
        if result["outputs"] != expected_outputs:
            raise AssertionError("local slope replay changed generated tokens")
        if result["steps"] != expected_steps:
            raise AssertionError("local slope replay changed scheduled work")
        return result

    metadata_run = checked(metadata=True)
    for _ in range(max(0, args.warmups - 1)):
        checked()
    baselines = [checked()["target_wall_ms"] for _ in range(args.repetitions)]
    traced = checked(profile=True) if trace else None
    row = {
        "policy": policy, "action": adapter.action, "target_step_index": target,
        "target_step_calls": expected_steps[target][1],
        "capture_configuration": {
            "batch": config.max_batch_size,
            "max_context_length": config.max_context_length,
            "block_table_width": adapter.max_blocks,
            "decode_buckets": [config.max_batch_size],
            "prefill_token_buckets": [budget],
            "generic_power_of_two_decode_ladder": False,
        },
        "unprofiled_wall_ms": baselines,
        "median_wall_ms": statistics.median(baselines),
        "trace": None, "cuda_activity": None,
        "context": summarize_context(metadata_run["target_calls"][0]),
    }
    if traced is not None:
        profiler = traced["profiler"]
        if not any(item["name"] == "cpp/step" for item in stage_summary(profiler)):
            raise RuntimeError("C++ profiler ranges missing; rebuild the extension")
        trace_dir = args.output_dir / "local" / "traces"
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = trace_dir / f"{case['id']}-{policy}.json"
        profiler.export_chrome_trace(str(trace_path))
        traced_context = summarize_context(traced["target_calls"][0])
        if traced_context != row["context"]:
            raise AssertionError("local target context changed between metadata and trace runs")
        row.update(trace=str(trace_path), cuda_activity=cuda_activity_summary(profiler.events()))
    del adapter
    torch.cuda.empty_cache()
    return row


def attention_sweep(torch, args):
    from benchmark_decode_kernel_causality import _make_inputs
    from paged_decode_attention import paged_decode_attention, paged_decode_attention_dispatch

    rows = []
    for batch in (8, 64):
        for context in (256, 2048, 4096):
            tensors = _make_inputs(torch, batch, context, page_size=16,
                                   dtype=torch.float16, device=args.device,
                                   seed=args.seed + batch * 1009 + context)
            q, k, v, random_table, lengths = tensors
            pages = random_table.shape[1]
            contiguous_table = torch.arange(batch * pages, device=args.device,
                                            dtype=torch.int32).view(batch, pages)
            operations = {
                "production_contiguous_pages": lambda: paged_decode_attention(
                    q, k, v, contiguous_table, lengths),
                "production_random_pages": lambda: paged_decode_attention(
                    q, k, v, random_table, lengths),
                "splitk_random_pages": lambda: paged_decode_attention_dispatch(
                    q, k, v, random_table, lengths, policy="splitk",
                    max_context_length=context),
            }
            reference = operations["production_random_pages"]()
            candidate = operations["splitk_random_pages"]()
            error = float((reference.float() - candidate.float()).abs().max())
            try:
                torch.testing.assert_close(candidate.float(), reference.float(),
                                           atol=.05, rtol=.02)
                correctness = {"status": "pass", "error": None}
            except AssertionError as exc:
                # A diagnostic must retain the timing even when a candidate is
                # numerically rejected; correctness remains explicit in output.
                correctness = {"status": "fail", "error": str(exc)}
            kv_bytes = batch * context * 2 * 2 * 128 * 2
            measurements = {}
            for name, operation in operations.items():
                timed = timed_cuda(torch, operation, args.kernel_repetitions, args.warmups)
                timed["effective_kv_gbps"] = kv_bytes / (timed["median_ms"] * 1e-3) / 1e9
                measurements[name] = timed
            contiguous_ms = measurements["production_contiguous_pages"]["median_ms"]
            random_ms = measurements["production_random_pages"]["median_ms"]
            splitk_ms = measurements["splitk_random_pages"]["median_ms"]
            rows.append({"batch": batch, "context": context,
                         "estimated_kv_bytes": kv_bytes, "max_splitk_error": error,
                         "splitk_correctness": correctness,
                         "measurements": measurements,
                         "derived": {
                             "random_page_locality_penalty": random_ms / contiguous_ms,
                             "splitk_speedup_on_random_pages": random_ms / splitk_ms,
                         }})
            del tensors, q, k, v, random_table, contiguous_table, lengths
            torch.cuda.empty_cache()
    return rows


def fusion_sweep(torch, engine, args):
    import torch.nn.functional as F
    from kernel_dispatch import rms_norm, swiglu

    rows = []
    layer = engine.model.layers[0]
    for batch in (8, 64):
        generator = torch.Generator(device=args.device).manual_seed(args.seed + batch)
        hidden = torch.randn(batch, engine.cfg.d_model, device=args.device,
                             dtype=torch.float16, generator=generator)
        q_width = engine.cfg.n_heads * engine.cfg.d_head
        kv_width = engine.cfg.n_kv_heads * engine.cfg.d_head
        qkv_widths = (q_width, kv_width, kv_width)
        qkv_offsets = (0, q_width, q_width + kv_width, q_width + 2 * kv_width)
        gate_width = engine.cfg.d_ff

        def qkv_separate():
            return tuple(F.linear(
                hidden,
                layer.qkv_proj.weight[qkv_offsets[index]:qkv_offsets[index + 1]],
                layer.qkv_proj.bias[qkv_offsets[index]:qkv_offsets[index + 1]],
            ) for index in range(3))

        def qkv_packed():
            return layer.qkv_proj(hidden).split(qkv_widths, dim=-1)

        def gate_up_separate():
            return (F.linear(hidden, layer.gate_up_proj.weight[:gate_width]),
                    F.linear(hidden, layer.gate_up_proj.weight[gate_width:]))

        def gate_up_packed():
            return layer.project_gate_up(hidden)

        separate = (*qkv_separate(), *gate_up_separate())
        packed = (*qkv_packed(), *gate_up_packed())
        projection_errors = []
        for expected, actual in zip(separate, packed):
            projection_errors.append(float((actual.float() - expected.float()).abs().max()))
        gate, up = gate_up_packed()
        residual = torch.randn(batch, 1, engine.cfg.d_model, device=args.device,
                               dtype=torch.float16, generator=generator)
        update = torch.randn(residual.shape, device=args.device, dtype=torch.float16,
                             generator=generator)
        candidates = {
            "qkv_separate": qkv_separate,
            "qkv_packed": qkv_packed,
            "gate_up_separate": gate_up_separate,
            "gate_up_packed": gate_up_packed,
            "swiglu_torch": lambda: F.silu(gate) * up,
            "swiglu_triton": lambda: swiglu(gate.contiguous(), up.contiguous(),
                                             block_size=512, num_warps=4, num_stages=2),
            "lm_head": lambda: engine.model.lm_head(hidden),
            "lm_head_argmax": lambda: engine.model.lm_head(hidden).argmax(dim=-1),
            "residual_add": lambda: residual + update,
            "rmsnorm_only": lambda: rms_norm(
                residual.view(batch, -1), layer.input_norm.weight,
                engine.cfg.rms_norm_eps),
            "residual_add_rmsnorm": lambda: rms_norm(
                (residual + update).view(batch, -1), layer.input_norm.weight,
                engine.cfg.rms_norm_eps),
        }
        measured = {name: timed_cuda(torch, operation, args.kernel_repetitions,
                                     args.warmups)
                    for name, operation in candidates.items()}
        rows.append({
            "batch": batch,
            "shape_details": {"hidden": list(hidden.shape),
                              "qkv_output_features": list(qkv_widths),
                              "gate_up_output_features": [gate_width, gate_width]},
            "measurements": measured,
            "max_projection_error": max(projection_errors),
            "derived": {
                "qkv_packing_speedup": (measured["qkv_separate"]["median_ms"]
                                        / measured["qkv_packed"]["median_ms"]),
                "gate_up_packing_speedup": (measured["gate_up_separate"]["median_ms"]
                                            / measured["gate_up_packed"]["median_ms"]),
                "swiglu_fusion_speedup": (measured["swiglu_torch"]["median_ms"]
                                          / measured["swiglu_triton"]["median_ms"]),
                "sampling_increment_ms": (measured["lm_head_argmax"]["median_ms"]
                                          - measured["lm_head"]["median_ms"]),
                "residual_rmsnorm_candidate_available": False,
                "residual_add_rmsnorm_boundary_ms": measured[
                    "residual_add_rmsnorm"]["median_ms"],
                "residual_add_ms": measured["residual_add"]["median_ms"],
                "rmsnorm_only_ms": measured["rmsnorm_only"]["median_ms"],
            },
        })
    return rows


def rope_kv_fusion_sweep(torch, engine, args):
    """Full-model A/B for the existing native K-RoPE plus direct KV-write candidate."""
    from benchmark_packed_projection_decode import stage_case
    from paged_graph_decoder import graph_decode_forward

    rows = []
    context = 2048
    for batch in (8, 64):
        cache, tensors = stage_case(
            torch, engine.cfg, batch, context, torch.float16,
            args.seed + batch * 100000 + context, args.device,
        )
        baseline = lambda: graph_decode_forward(
            engine.model, cache, *tensors, decode_attention_policy="splitk",
            max_decode_context_length=context + 1,
        )
        candidate = lambda: graph_decode_forward(
            engine.model, cache, *tensors, decode_attention_policy="splitk",
            max_decode_context_length=context + 1,
            enable_native_decode_rope_kv=True,
        )
        reference_logits = baseline()
        candidate_logits = candidate()
        error = (reference_logits.float() - candidate_logits.float()).abs()
        token_matches = int((reference_logits.argmax(-1)
                             == candidate_logits.argmax(-1)).sum())
        total_tokens = reference_logits.shape[0]
        try:
            torch.testing.assert_close(candidate_logits.float(), reference_logits.float(),
                                       atol=.1, rtol=.01)
            close_status, close_error = "pass", None
        except AssertionError as exc:
            close_status, close_error = "fail", str(exc)
        base_time = timed_cuda(torch, baseline, args.kernel_repetitions, args.warmups)
        fused_time = timed_cuda(torch, candidate, args.kernel_repetitions, args.warmups)
        rows.append({
            "batch": batch, "context": context,
            "baseline": base_time, "native_k_rope_kv_write": fused_time,
            "speedup": base_time["median_ms"] / fused_time["median_ms"],
            "max_logit_error": float(error.max()),
            "mean_logit_error": float(error.mean()),
            "matching_tokens": token_matches, "total_tokens": total_tokens,
            "correctness": {"logits": close_status, "error": close_error,
                            "sampled_tokens_exact": token_matches == total_tokens},
            "scope": "full eager model; production Q-RoPE retained in both arms",
        })
        del cache, tensors, reference_logits, candidate_logits
        torch.cuda.empty_cache()
    return rows


def run_local(args, contracts):
    fingerprint, protocol = protocol_fingerprint(args)
    predecessor_fingerprint = pre_unrolled_hotfix_fingerprint(args)
    output = args.output_dir / "local/report.json"
    if output.is_file():
        if json.loads(output.read_text()).get("fingerprint") != fingerprint:
            raise ValueError(f"local report uses a different protocol: {output}")
        print(f"local report already complete: {output}", flush=True)
        return
    if output.parent.exists() and any(output.parent.iterdir()):
        if not args.retry_failed:
            raise ValueError(f"partial local output at {output.parent}; use --retry-failed")
        print(f"resuming validated local checkpoints in {output.parent}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    from benchmark_integrated_graph import resolve_requests
    from model_adapter import allocate_pool
    from model_setup import check_startup, load_model_only
    from run_benchmarks import system_metadata
    import torch
    import inference_engine_cpp as cpp

    startup = check_startup(args.device)
    model = resolve_model_source(args)
    engine, load_seconds, transfer = load_model_only(
        model, args.device, "float16", hub_transfer=startup["hub_transfer"])
    max_blocks = max(blocks for _, blocks in contracts.values())
    pool = allocate_pool(engine.cfg, max_blocks, engine.device)
    slope = []
    for shape_id in SLOPE_SHAPES:
        cell_path = output.parent / "cells" / f"{shape_id}.json"
        if cell_path.is_file():
            cell = json.loads(cell_path.read_text())
            if cell.get("status") != "complete" or cell.get("shape_id") != shape_id:
                raise ValueError(f"invalid local cell checkpoint: {cell_path}")
            cell = migrate_pre_unrolled_checkpoint(
                cell_path, cell, fingerprint, predecessor_fingerprint)
            slope.append(cell["row"])
            print(f"local slope resumed: {shape_id}", flush=True)
            continue
        original, _ = contracts[shape_id]
        case = {**original, "prefill_budget": local_budget(shape_id)}
        child = argparse.Namespace(workload_in=args.suite_dir / shape_id / "workload.json",
                                   seed=args.seed)
        requests = resolve_requests(child, case)
        arms = []
        for policy in POLICIES:
            if policy == "splitk" or get_fixed_shape(shape_id)["prompt_length"] >= 1024:
                arms.append(measure_local_arm(
                    torch, cpp, engine, pool, case, requests, args, policy,
                    trace=shape_id == args.trace_shape and policy == "splitk"))
        row = {"shape_id": shape_id, "batch": case["max_running"],
               "prompt": get_fixed_shape(shape_id)["prompt_length"],
               "prefill_budget": case["prefill_budget"], "arms": arms}
        slope.append(row)
        atomic_json(cell_path, {"status": "complete", "fingerprint": fingerprint,
                                "shape_id": shape_id, "row": row})
        print(f"local slope complete: {shape_id}", flush=True)

    component_paths = {
        "attention": output.parent / "attention.json",
        "fusion": output.parent / "fusion.json",
        "gemm": output.parent / "gemm.json",
        "output_head": output.parent / "output-head.json",
        "output_head_full_model": output.parent / "output-head-full-model.json",
        "unrolled_decode": output.parent / "unrolled-decode.json",
        "rope_kv_fusion": output.parent / "rope-kv-fusion.json",
    }
    components = {}
    builders = {"attention": lambda: attention_sweep(torch, args),
                "fusion": lambda: fusion_sweep(torch, engine, args),
                "gemm": lambda: gemm_sweep(torch, engine, args),
                "output_head": lambda: output_head_sweep(torch, engine, args),
                "output_head_full_model": lambda: output_head_full_model_sweep(
                    torch, engine, args),
                "unrolled_decode": lambda: unrolled_decode_sweep(
                    torch, engine, args, components["output_head"]),
                "rope_kv_fusion": lambda: rope_kv_fusion_sweep(torch, engine, args)}
    for name, path in component_paths.items():
        if path.is_file():
            saved = json.loads(path.read_text())
            if saved.get("status") != "complete":
                raise ValueError(f"invalid local component checkpoint: {path}")
            if name not in PRE_UNROLLED_COMPONENTS:
                if saved.get("fingerprint") != fingerprint:
                    raise ValueError(f"invalid local component checkpoint: {path}")
            else:
                saved = migrate_pre_unrolled_checkpoint(
                    path, saved, fingerprint, predecessor_fingerprint)
            components[name] = saved["rows"]
        else:
            components[name] = builders[name]()
            atomic_json(path, {"status": "complete", "fingerprint": fingerprint,
                               "rows": components[name]})
        print(f"local component ready: {name}", flush=True)
    atomic_json(output, {
        "schema_version": 1, "status": "complete",
        "fingerprint": fingerprint, "protocol": protocol,
        "created_at": datetime.now(timezone.utc).isoformat(), "model": model,
        "model_load_seconds": load_seconds, "hub_transfer": transfer,
        "system": system_metadata(), "slope": slope,
        "attention": components["attention"], "fusion": components["fusion"],
        "gemm": components["gemm"],
        "output_head": components["output_head"],
        "output_head_full_model": components["output_head_full_model"],
        "unrolled_decode": components["unrolled_decode"],
        "rope_kv_fusion": components["rope_kv_fusion"],
    })
    print(f"local comprehensive report: {output}", flush=True)


def run_vllm(args, contracts):
    fingerprint, protocol = protocol_fingerprint(args)
    output = args.output_dir / "vllm/report.json"
    if output.is_file():
        if json.loads(output.read_text()).get("fingerprint") != fingerprint:
            raise ValueError(f"vLLM report uses a different protocol: {output}")
        print(f"vLLM report already complete: {output}", flush=True)
        return
    if output.parent.exists() and any(output.parent.iterdir()):
        if not args.retry_failed:
            raise ValueError(f"partial vLLM output at {output.parent}; use --retry-failed")
        print(f"resuming validated vLLM checkpoints in {output.parent}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    trace_dir = output.parent / "trace"
    trace_dir.mkdir(exist_ok=True)
    os.environ["VLLM_TORCH_PROFILER_DIR"] = str(trace_dir)
    os.environ["VLLM_TORCH_PROFILER_WITH_STACK"] = "0"

    version = importlib.metadata.version("vllm")
    if version != PINNED_VLLM:
        raise ValueError(f"requires vLLM {PINNED_VLLM}, found {version}")
    model = resolve_model_source(args)
    from benchmark_backends import _matched_kv_cache_bytes
    from run_benchmarks import system_metadata
    from vllm import LLM

    max_blocks = max(blocks for _, blocks in contracts.values())
    max_context = max(shape_summary(get_fixed_shape(shape))["max_context_tokens_per_request"]
                      for shape in SLOPE_SHAPES)
    kv_bytes = _matched_kv_cache_bytes(model, dtype="float16", block_size=16,
                                       num_blocks=max_blocks)
    started = time.perf_counter()
    llm = LLM(model=model, dtype="float16", seed=args.seed, max_num_seqs=64,
              max_num_batched_tokens=PREFILL_TOKENS_PER_STEP,
              max_model_len=max_context, block_size=16, enable_prefix_caching=False,
              enable_chunked_prefill=True, generation_config="vllm",
              enforce_eager=False, kv_cache_memory_bytes=kv_bytes)
    load_seconds = time.perf_counter() - started
    rows = []
    profiled = False
    for shape_id in SLOPE_SHAPES:
        cell_path = output.parent / "cells" / f"{shape_id}.json"
        if cell_path.is_file():
            cell = json.loads(cell_path.read_text())
            if (cell.get("status") != "complete" or cell.get("shape_id") != shape_id
                    or cell.get("fingerprint") != fingerprint):
                raise ValueError(f"invalid vLLM cell checkpoint: {cell_path}")
            rows.append(cell["row"])
            profiled = profiled or shape_id == args.trace_shape
            print(f"vLLM slope resumed: {shape_id}", flush=True)
            continue
        case, _ = contracts[shape_id]
        _, requests = load_frozen(checkpoint_args(args, shape_id), case)
        discovery = discover_vllm_target(llm, requests, args.occurrence,
                                         f"{shape_id}-discovery")
        target = discovery["target_engine_step_index"]
        for index in range(args.warmups):
            warm = measure_vllm_target(llm, requests, target,
                                       f"{shape_id}-warmup-{index}")
            if warm["total_engine_steps"] != discovery["total_engine_steps"]:
                raise AssertionError("vLLM schedule changed during warmup")
        measured = [measure_vllm_target(llm, requests, target,
                                        f"{shape_id}-sample-{index}")
                    for index in range(args.repetitions)]
        trace = cuda = None
        if shape_id == args.trace_shape:
            if profiled:
                raise AssertionError("vLLM profiler is intentionally one-shot")
            before = set(trace_dir.rglob("*"))
            traced = measure_vllm_target(llm, requests, target,
                                         f"{shape_id}-profile", profile=True)
            if traced["total_engine_steps"] != discovery["total_engine_steps"]:
                raise AssertionError("vLLM schedule changed during trace")
            candidates = [path for path in trace_dir.rglob("*") if path.is_file()
                          and path not in before
                          and (path.name.endswith(".json") or path.name.endswith(".json.gz"))]
            if len(candidates) != 1:
                raise ValueError(f"expected one vLLM trace, found {candidates}")
            trace = str(candidates[0])
            cuda = summarize_chrome_trace(candidates[0])
            profiled = True
        context_values = sorted(discovery["target_context_lengths"].values())
        row = {
            "shape_id": shape_id, "batch": get_fixed_shape(shape_id)["batch"],
            "prompt": get_fixed_shape(shape_id)["prompt_length"],
            "target_engine_step_index": target,
            "wall_ms": [row["target_wall_ms"] for row in measured],
            "median_wall_ms": statistics.median(row["target_wall_ms"] for row in measured),
            "context": {"minimum": context_values[0],
                        "median": statistics.median(context_values),
                        "maximum": context_values[-1], "values": context_values},
            "trace": trace, "cuda_activity": cuda,
        }
        rows.append(row)
        atomic_json(cell_path, {"status": "complete", "fingerprint": fingerprint,
                                "shape_id": shape_id, "row": row})
        print(f"vLLM slope complete: {shape_id}", flush=True)
    atomic_json(output, {
        "schema_version": 1, "status": "complete",
        "fingerprint": fingerprint, "protocol": protocol,
        "created_at": datetime.now(timezone.utc).isoformat(), "model": model,
        "vllm_version": version, "engine_max_num_seqs": 64,
        "engine_max_model_len": max_context, "engine_matched_num_blocks": max_blocks,
        "kv_cache_memory_bytes": kv_bytes, "model_load_seconds": load_seconds,
        "timed_output_kind": "FINAL_ONLY", "system": system_metadata(), "slope": rows,
    })
    print(f"vLLM comprehensive report: {output}", flush=True)


def linear_slope(points):
    if len(points) < 2:
        return None
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    x_mean, y_mean = statistics.mean(xs), statistics.mean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator


def analyze(args):
    fingerprint, protocol = protocol_fingerprint(args)
    local_path = args.output_dir / "local/report.json"
    vllm_path = args.output_dir / "vllm/report.json"
    if not local_path.is_file() or not vllm_path.is_file():
        raise ValueError("local and vLLM comprehensive reports must both exist")
    local, vllm = json.loads(local_path.read_text()), json.loads(vllm_path.read_text())
    if (local.get("status") != "complete" or vllm.get("status") != "complete"
            or local.get("fingerprint") != fingerprint
            or vllm.get("fingerprint") != fingerprint):
        raise ValueError("an input report is incomplete")
    if local["model"] != vllm["model"] or vllm["vllm_version"] != PINNED_VLLM:
        raise ValueError("model or pinned vLLM identity differs")
    local_rows = {row["shape_id"]: row for row in local["slope"]}
    vllm_rows = {row["shape_id"]: row for row in vllm["slope"]}
    if set(local_rows) != set(SLOPE_SHAPES) or set(vllm_rows) != set(SLOPE_SHAPES):
        raise ValueError("slope surface is incomplete")
    rows = []
    for shape_id in SLOPE_SHAPES:
        arms = {row["policy"]: row for row in local_rows[shape_id]["arms"]}
        current = arms["splitk"]
        reference = vllm_rows[shape_id]
        rows.append({
            "shape_id": shape_id, "batch": local_rows[shape_id]["batch"],
            "prompt": local_rows[shape_id]["prompt"],
            "local_splitk_ms": current["median_wall_ms"],
            "local_production_ms": arms.get("production", current)["median_wall_ms"],
            "vllm_ms": reference["median_wall_ms"],
            "local_splitk_over_vllm": current["median_wall_ms"] / reference["median_wall_ms"],
            "splitk_speedup": (arms.get("production", current)["median_wall_ms"]
                               / current["median_wall_ms"]),
            "local_context_median": (current["context"] or {}).get("median"),
            "vllm_context_median": reference["context"]["median"],
        })
    slopes = []
    for batch in (8, 64):
        selected = [row for row in rows if row["batch"] == batch]
        slopes.append({
            "batch": batch,
            "local_us_per_context_token": 1000 * linear_slope(
                [(row["local_context_median"], row["local_splitk_ms"])
                 for row in selected]),
            "vllm_us_per_context_token": 1000 * linear_slope(
                [(row["vllm_context_median"], row["vllm_ms"])
                 for row in selected]),
        })
    local_trace = next(row for row in local_rows[args.trace_shape]["arms"]
                       if row["policy"] == "splitk")
    vllm_trace = vllm_rows[args.trace_shape]
    if local_trace["cuda_activity"] is None or vllm_trace["cuda_activity"] is None:
        raise ValueError("selected detailed traces are missing CUDA activity")
    trace_comparison = {
        "categories": compare_categories(local_trace["cuda_activity"],
                                          vllm_trace["cuda_activity"]),
        "summed_cuda_activity_local_over_vllm": (
            local_trace["cuda_activity"]["summed_cuda_activity_us"]
            / vllm_trace["cuda_activity"]["summed_cuda_activity_us"]),
        "activity_counts": {
            "local": local_trace["cuda_activity"]["activity_count"],
            "vllm": vllm_trace["cuda_activity"]["activity_count"],
        },
        "wall_minus_summed_activity_ms": {
            "local": (local_trace["median_wall_ms"]
                      - local_trace["cuda_activity"]["summed_cuda_activity_us"] / 1000),
            "vllm": (vllm_trace["median_wall_ms"]
                     - vllm_trace["cuda_activity"]["summed_cuda_activity_us"] / 1000),
            "note": "Diagnostic residual, not a direct GPU-idle measurement; CUDA activities can overlap.",
        },
    }
    gemm_rows = []
    for saved in local["gemm"]:
        row = dict(saved)
        batch = row["batch"]
        shortest = min((candidate for candidate in rows if candidate["batch"] == batch),
                       key=lambda candidate: candidate["local_context_median"])
        graph_estimate = row["current_route_estimates"].get(
            "graph_cache_evicted", {}).get("projected_28_layer_plus_lm_head_ms")
        row["derived"] = {
            "comparison_shape": shortest["shape_id"],
            "projected_cache_evicted_gemm_fraction_of_local_step": (
                graph_estimate / shortest["local_splitk_ms"]
                if graph_estimate is not None else None),
            "note": ("Sum of isolated projection CUDA times divided by the shortest-context "
                     "full decode graph; use as a prioritization estimate, not attribution."),
        }
        gemm_rows.append(row)
    report = {
        "schema_version": 1, "status": "complete",
        "fingerprint": fingerprint, "protocol": protocol,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rows": rows, "decode_context_slopes": slopes,
        "attention": local["attention"], "fusion": local["fusion"],
        "gemm": gemm_rows,
        "output_head": local["output_head"],
        "output_head_full_model": local["output_head_full_model"],
        "unrolled_decode": local["unrolled_decode"],
        "rope_kv_fusion": local["rope_kv_fusion"],
        "trace_shape": args.trace_shape,
        "local_trace": local_trace, "vllm_trace": vllm_trace,
        "trace_comparison": trace_comparison,
        "scope_note": ("Local graph captures use each shape's exact current configuration. "
                       "vLLM uses one max-B64/max-context diagnostic engine so its weights and "
                       "graphs are loaded only once across the slope surface."),
    }
    atomic_json(args.output_dir / "summary.json", report)
    print("\nshape                              local ms    vLLM ms   local/vLLM  splitK", flush=True)
    for row in rows:
        print(f"{row['shape_id']:<34} {row['local_splitk_ms']:>9.3f} "
              f"{row['vllm_ms']:>10.3f} {row['local_splitk_over_vllm']:>11.3f}x "
              f"{row['splitk_speedup']:>7.3f}x", flush=True)
    print("\ncontext slope:", flush=True)
    for row in slopes:
        print(f"  B={row['batch']}: local {row['local_us_per_context_token']:.4f} us/token, "
              f"vLLM {row['vllm_us_per_context_token']:.4f} us/token", flush=True)
    print("\n4096-token attention controls:", flush=True)
    for row in (row for row in local["attention"] if row["context"] == 4096):
        print(f"  B={row['batch']}: random-page penalty "
              f"{row['derived']['random_page_locality_penalty']:.3f}x, split-K speedup "
              f"{row['derived']['splitk_speedup_on_random_pages']:.3f}x", flush=True)
    print("\nfusion controls:", flush=True)
    for row in local["fusion"]:
        derived = row["derived"]
        print(f"  B={row['batch']}: QKV {derived['qkv_packing_speedup']:.3f}x, "
              f"gate/up {derived['gate_up_packing_speedup']:.3f}x, "
              f"SwiGLU {derived['swiglu_fusion_speedup']:.3f}x", flush=True)
    print("\nexact-shape GEMM graph controls:", flush=True)
    for row in gemm_rows:
        estimates = row["current_route_estimates"]
        eager = estimates.get("eager_warm", {}).get("projected_28_layer_plus_lm_head_ms")
        graph = estimates.get("graph_warm", {}).get("projected_28_layer_plus_lm_head_ms")
        fraction = row["derived"]["projected_cache_evicted_gemm_fraction_of_local_step"]
        if eager is not None and graph is not None:
            fraction_text = f"{fraction:.1%}" if fraction is not None else "unavailable"
            best = row["best_correct_route_by_mode"].get("graph_cache_evicted", {})
            best_speedup = best.get("speedup_over_current")
            best_text = (f"{best.get('route')} {best_speedup:.3f}x"
                         if best and best_speedup is not None else "unavailable")
            print(f"  B={row['batch']}: projected GEMMs eager={eager:.3f} ms, "
                  f"exact graph={graph:.3f} ms ({eager / graph:.3f}x), "
                  f"cold projection share~{fraction_text}, best route={best_text}", flush=True)
    print("\noutput-head interventions:", flush=True)
    for row in local["output_head"]:
        best = row["derived"]["best_correct_fused_by_mode"].get(
            "graph_cache_evicted", {})
        speedup = best.get("speedup")
        print(f"  B={row['batch']}: fused/cold-graph speedup="
              f"{speedup:.3f}x ({best.get('config')})" if speedup is not None else
              f"  B={row['batch']}: fused candidate unavailable", flush=True)
    for row in local["output_head_full_model"]:
        if row.get("status") == "complete":
            print(f"  B={row['batch']} full-model: {row['speedup']['graph']:.3f}x, "
                  f"correctness={row['correctness']['status']}", flush=True)
        else:
            print(f"  B={row['batch']} full-model: fused candidate unavailable", flush=True)
    print("\nK-step exact-regime capture:", flush=True)
    for case in local["unrolled_decode"]:
        summary = ", ".join(
            f"K{row['steps']}={row['speedup']['production_wall']:.3f}x"
            for row in case["rows"])
        correctness = all(
            row["validation"]["logits"]["status"] == "pass"
            and row["validation"]["kv"]["status"] == "pass"
            for row in case["rows"])
        print(f"  B={case['batch']} C={case['context']}: {summary}; "
              f"full-logit/KV correctness={correctness}", flush=True)
    for row in local["rope_kv_fusion"]:
        print(f"  B={row['batch']}: native K-RoPE/KV {row['speedup']:.3f}x, "
              f"tokens exact={row['correctness']['sampled_tokens_exact']}", flush=True)
    print(f"summary: {args.output_dir / 'summary.json'}", flush=True)


def plan(args):
    short_shapes = sum(get_fixed_shape(shape)["prompt_length"] < 1024
                       for shape in SLOPE_SHAPES)
    local_arms = short_shapes + 2 * (len(SLOPE_SHAPES) - short_shapes)
    per_arm = 1 + max(1, args.warmups) + args.repetitions
    per_vllm_shape = 1 + args.warmups + args.repetitions
    per_output_head_full_model_arm = (
        3 + 2 * args.warmups + 2 * args.repetitions + max(1, args.warmups))
    print(json.dumps({
        "shape_ids": list(SLOPE_SHAPES), "policies": list(POLICIES),
        "trace_shape": args.trace_shape, "local_model_loads": 1, "vllm_model_loads": 1,
        "local_measured_arms": local_arms,
        "local_full_workloads_per_arm": per_arm,
        "local_scheduler_workloads_total": local_arms * per_arm + 1,
        # Backward-compatible name: these are scheduler workloads, not the
        # short staged full-model kernel controls reported separately below.
        "local_full_workloads_total": local_arms * per_arm + 1,
        "vllm_full_workloads_per_shape": per_vllm_shape,
        "vllm_full_workloads_total": len(SLOPE_SHAPES) * per_vllm_shape + 1,
        "heavy_traces": {"local": 1, "vllm": 1},
        "decode_graph_captures": {
            "full_model": ("one exact target batch and exact per-cell maximum context "
                           "width per local arm"),
            "gemm_controls": "one exact B8 and B64 graph per operation and BLAS route",
            "power_of_two_ladder": False,
        },
        "output_head_interventions": {
            "materialized_logits": True,
            "materialized_logits_argmax": True,
            "chunked_exact_argmax": True,
            "fused_tiled_projection_argmax": list(OUTPUT_HEAD_CONFIGS),
            "full_model_splitk_ab": True,
            "isolated_arms": len(DECODE_BATCHES) * (3 + len(OUTPUT_HEAD_CONFIGS)),
            "staged_full_model_arms": len(DECODE_BATCHES) * (1 + len(OUTPUT_HEAD_CONFIGS)),
            "staged_full_model_forward_passes": (
                len(DECODE_BATCHES) * (1 + len(OUTPUT_HEAD_CONFIGS))
                * per_output_head_full_model_arm),
        },
        "unrolled_decode": {
            "steps": [2, 4, 8], "batches": list(DECODE_BATCHES),
            "contexts": [256, 2048, 4096],
            "cases": len(DECODE_BATCHES) * 3 * 3,
            "graph_captures_without_optional_fused": (
                len(DECODE_BATCHES) * 3 * (1 + 2 * 3)),
            "optional_fused_graph_captures": len(DECODE_BATCHES) * 3 * 3,
            "largest_retained_validation_logits_bytes": 8 * 64 * 151936 * 2,
            "validation_retains_every_logit": True,
            "production_retains_only_tokens": True,
            "eos_scanned_across_every_generated_position": True,
        },
        "kernel_repetitions": args.kernel_repetitions,
        "output_dir": str(args.output_dir),
    }, indent=2))


def run_all(args):
    verify_setup(args)
    common = ["--suite-dir", str(args.suite_dir), "--output-dir", str(args.output_dir),
              "--model", args.model, "--device", args.device, "--seed", str(args.seed),
              "--occurrence", str(args.occurrence), "--warmups", str(args.warmups),
              "--repetitions", str(args.repetitions),
              "--kernel-repetitions", str(args.kernel_repetitions),
              "--l2-evict-mib", str(args.l2_evict_mib),
              "--eos-token-id", str(args.eos_token_id),
              "--trace-shape", args.trace_shape]
    if args.retry_failed:
        common.append("--retry-failed")
    for action in ("run-local", "run-vllm", "analyze"):
        subprocess.run([sys.executable, str(Path(__file__)), action, *common],
                       cwd=ROOT, check=True)


def main():
    args = build_parser().parse_args()
    contracts = validate_args(args)
    if args.action == "plan":
        plan(args)
    elif args.action == "check-setup":
        setup = verify_setup(args)
        print(json.dumps({"status": "ready", **setup, "shapes": list(SLOPE_SHAPES),
                          "trace_shape": args.trace_shape}, indent=2))
    elif args.action == "run-local":
        run_local(args, contracts)
    elif args.action == "run-vllm":
        run_vllm(args, contracts)
    elif args.action == "analyze":
        analyze(args)
    else:
        run_all(args)


if __name__ == "__main__":
    main()
