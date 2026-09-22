#!/usr/bin/env python3
"""Measure decode CPU/GPU handoffs and fixed-regime GPU-resident execution.

The cheap microbenchmarks isolate sampling, token copies, state updates, and
metadata staging. The model arm compares materialized-logit and fused-argmax
FA3 graphs at K=1/2/4/8 using the current accepted decode fusions. Validation
graphs retain every logit. Unsupported CUDA graph control-flow features are
reported explicitly instead of being skipped.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
import time


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for directory in (HERE, ROOT / "baseline", ROOT / "engine/kvcache",
                  ROOT / "engine/model_runner", ROOT / "engine/graph",
                  ROOT / "engine/cpp/build", ROOT / "experiments/decode"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


STEPS = (2, 4, 8)
GRAPH_STEPS = (1, *STEPS)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "check-setup", "run-micro",
                                           "run-model", "run", "analyze"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--eos-token-id", type=int, default=151645)
    parser.add_argument("--qkv-mode", choices=("none", "native-k", "native", "packed"),
                        default="native")
    parser.add_argument("--residual-rmsnorm", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "experiments/results/decode-control-plane")
    return parser


def validate(args):
    if min(args.batch, args.context, args.repetitions) < 1 or args.warmups < 0:
        raise ValueError("batch, context, and repetitions must be positive")
    if args.eos_token_id < -1:
        raise ValueError("EOS token ID must be -1 or nonnegative")


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def measure(torch, operation, warmups, repetitions):
    for _ in range(warmups):
        operation()
    torch.cuda.synchronize()
    gpu, wall = [], []
    for _ in range(repetitions):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        started = time.perf_counter()
        begin.record()
        operation()
        end.record()
        end.synchronize()
        wall.append((time.perf_counter() - started) * 1000)
        gpu.append(float(begin.elapsed_time(end)))
    return {"gpu_median_ms": statistics.median(gpu),
            "wall_median_ms": statistics.median(wall),
            "gpu_samples_ms": gpu, "wall_samples_ms": wall}


def capabilities(torch):
    graph = torch.cuda.CUDAGraph
    return {
        "torch_cuda_graph": True,
        "torch_graph_debug_mode": hasattr(graph, "enable_debug_mode"),
        "torch_device_graph_launch": any(
            hasattr(graph, name) for name in ("device_launch", "launch_from_device")
        ),
        "torch_conditional_graph_nodes": any(
            hasattr(graph, name) for name in ("conditional", "add_conditional_node")
        ),
        "decision": (
            "benchmark device launch/conditional nodes"
            if any(hasattr(graph, name) for name in
                   ("device_launch", "launch_from_device", "conditional",
                    "add_conditional_node"))
            else "unsupported by the installed PyTorch API; requires a CUDA C++ prototype"
        ),
    }


def run_micro(args):
    import torch

    device = torch.device(args.device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    logits = torch.randn(args.batch, 151936, device=device, dtype=torch.float16,
                         generator=generator)
    tokens = torch.randint(0, 151936, (max(STEPS), args.batch), device=device,
                           dtype=torch.int64, generator=generator)
    pinned_tokens = torch.empty(tokens.shape, dtype=tokens.dtype,
                                device="cpu", pin_memory=True)
    cpu_positions = torch.full((args.batch,), args.context, dtype=torch.int32,
                               pin_memory=True)
    cpu_lengths = cpu_positions + 1
    cpu_ids = torch.zeros(args.batch, dtype=torch.int64, pin_memory=True)
    gpu_positions = cpu_positions.to(device, non_blocking=True)
    gpu_lengths = cpu_lengths.to(device, non_blocking=True)
    gpu_ids = cpu_ids.to(device, non_blocking=True)
    active = torch.ones(args.batch, dtype=torch.bool, device=device)
    torch.cuda.synchronize()

    def per_step_blocking_copy():
        for step in range(max(STEPS)):
            tokens[step].cpu()

    def chunk_blocking_copy():
        tokens.cpu()

    def chunk_pinned_copy():
        pinned_tokens.copy_(tokens, non_blocking=True)

    def gpu_state_update():
        gpu_ids.copy_(tokens[0])
        gpu_positions.add_(1)
        gpu_lengths.add_(1)
        active.logical_and_(tokens[0] != args.eos_token_id)

    def metadata_h2d():
        gpu_ids.copy_(cpu_ids, non_blocking=True)
        gpu_positions.copy_(cpu_positions, non_blocking=True)
        gpu_lengths.copy_(cpu_lengths, non_blocking=True)

    rows = {
        "gpu_argmax": measure(torch, lambda: logits.argmax(-1),
                              args.warmups, args.repetitions),
        "per_step_blocking_token_d2h_k8": measure(
            torch, per_step_blocking_copy, args.warmups, args.repetitions),
        "chunk_blocking_token_d2h_k8": measure(
            torch, chunk_blocking_copy, args.warmups, args.repetitions),
        "chunk_pinned_async_token_d2h_k8": measure(
            torch, chunk_pinned_copy, args.warmups, args.repetitions),
        "gpu_state_update": measure(
            torch, gpu_state_update, args.warmups, args.repetitions),
        "metadata_h2d_three_tensors": measure(
            torch, metadata_h2d, args.warmups, args.repetitions),
    }
    rows["derived"] = {
        "chunked_vs_per_step_d2h": (
            rows["per_step_blocking_token_d2h_k8"]["wall_median_ms"] /
            rows["chunk_blocking_token_d2h_k8"]["wall_median_ms"]),
        "pinned_async_vs_per_step_d2h": (
            rows["per_step_blocking_token_d2h_k8"]["wall_median_ms"] /
            rows["chunk_pinned_async_token_d2h_k8"]["wall_median_ms"]),
    }
    report = {
        "schema_version": 1, "status": "complete", "kind": "micro",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "configuration": {"batch": args.batch, "context": args.context,
                          "warmups": args.warmups,
                          "repetitions": args.repetitions,
                          "eos_token_id": args.eos_token_id},
        "device": torch.cuda.get_device_name(device),
        "measurements": rows, "cuda_graph_capabilities": capabilities(torch),
    }
    atomic_json(args.output_dir / "micro.json", report)
    return report


def fusion_options(args):
    return {
        "enable_residual_rmsnorm": args.residual_rmsnorm,
        "enable_native_decode_rope_kv": args.qkv_mode == "native-k",
        "enable_native_decode_qkv_postprocess": args.qkv_mode == "native",
        "enable_packed_qkv_rope_cache": args.qkv_mode == "packed",
    }


def output_head_config(batch):
    """Reuse the best correctness-passing fixed-regime configurations.

    The B8/B64 choices come from the checked-in isolated output-head sweep.  The
    fallback keeps ad-hoc smoke batches runnable without pretending they were tuned.
    """
    if batch < 1:
        raise ValueError("output-head batch must be positive")
    block_m = 16 if batch <= 16 else 32 if batch <= 32 else 64
    if batch <= 16:
        return {"block_m": block_m, "block_n": 256, "block_k": 64,
                "num_warps": 8, "num_stages": 3,
                "source": "checked-in B8 graph-warm winner"}
    return {"block_m": block_m, "block_n": 64, "block_k": 64,
            "num_warps": 4, "num_stages": 3,
            "source": ("checked-in B64 graph-warm winner" if batch == 64
                       else "untuned smoke fallback")}


def check_experiment_setup(args):
    """Fail before model allocation if any selected experimental path is unavailable."""
    from kernel_dispatch import _load
    from model_setup import check_startup

    startup = check_startup(args.device)
    _load("paged_decode_fa3")._load_fa3()
    _load("fused_lm_head")
    components = ["vllm_bundled_fa3", "fused_lm_head"]
    qkv_module = {
        "native-k": "rope_kv_write", "native": "rope_kv_write",
        "packed": "packed_qkv_rope_cache",
    }.get(args.qkv_mode)
    if qkv_module is not None:
        _load(qkv_module)
        components.append(qkv_module)
    startup["experiment_components"] = components
    startup["configuration"] = {
        "attention": "fa3", "qkv_mode": args.qkv_mode,
        "residual_rmsnorm": args.residual_rmsnorm,
        "output_heads": ["materialized_logits_argmax", "fused_projection_argmax"],
        "steps": list(GRAPH_STEPS),
    }
    return startup


def run_model(args):
    import torch
    from benchmark_unrolled_graph import (eager_trajectory, future_slots,
                                           initialize_cache, kv_accuracy,
                                           restore_slots, snapshot_slots,
                                           stage_unrolled_case,
                                           trajectory_accuracy)
    from model_setup import load_model_only, prepare_hub_transfer
    from paged_graph_decoder import CUDAGraphDecoder
    from unrolled_graph_decoder import UnrolledCUDAGraphDecoder, first_eos_positions

    torch.cuda.set_device(torch.device(args.device))
    engine, load_seconds, hub_transfer = load_model_only(
        args.model, args.device, "float16", hub_transfer=prepare_hub_transfer())
    if args.eos_token_id >= engine.cfg.vocab:
        raise ValueError("EOS token ID exceeds model vocabulary")
    cache, metadata, first_ids, generator = stage_unrolled_case(
        torch, engine.cfg, args.batch, args.context, max(STEPS), torch.float16,
        args.seed, args.device,
    )
    options = fusion_options(args)
    head_config = output_head_config(args.batch)
    kernel_head_config = {key: value for key, value in head_config.items()
                          if key != "source"}
    single = CUDAGraphDecoder(
        engine.model, cache, args.batch, metadata[0][2].shape[1], args.device,
        torch.float16, decode_attention_policy="fa3",
        max_decode_context_length=args.context + max(STEPS), **options,
    ).capture()
    fused_single = CUDAGraphDecoder(
        engine.model, cache, args.batch, metadata[0][2].shape[1], args.device,
        torch.float16, decode_attention_policy="fa3",
        max_decode_context_length=args.context + max(STEPS),
        output_head_policy="fused_argmax",
        output_head_config=kernel_head_config, **options,
    ).capture()
    validation_chunks = {
        steps: UnrolledCUDAGraphDecoder(
            engine.model, cache, metadata[:steps], decode_attention_policy="fa3",
            max_decode_context_length=args.context + steps,
            retain_logits=True, output_head_policy="logits", **options,
        ).capture()
        for steps in GRAPH_STEPS
    }
    materialized_chunks = {
        steps: UnrolledCUDAGraphDecoder(
            engine.model, cache, metadata[:steps], decode_attention_policy="fa3",
            max_decode_context_length=args.context + steps,
            retain_logits=False, output_head_policy="logits", **options,
        ).capture()
        for steps in GRAPH_STEPS
    }
    fused_head_chunks = {
        steps: UnrolledCUDAGraphDecoder(
            engine.model, cache, metadata[:steps], decode_attention_policy="fa3",
            max_decode_context_length=args.context + steps,
            retain_logits=False, output_head_policy="fused_argmax",
            output_head_config=kernel_head_config, **options,
        ).capture()
        for steps in GRAPH_STEPS
    }
    initialize_cache(torch, cache, generator)
    pinned = {steps: torch.empty((steps, args.batch), dtype=torch.int64,
                                 device="cpu", pin_memory=True)
              for steps in GRAPH_STEPS}
    rows = []
    for steps in GRAPH_STEPS:
        selected = metadata[:steps]
        slots = future_slots(selected)
        initial = snapshot_slots(cache, slots)

        def repeated_k1(copy_each_step=False):
            token = first_ids
            emitted = []
            for positions, lengths, table, step_slots in selected:
                logits = single.decode(token, positions, lengths, table, step_slots)
                token = logits.argmax(-1).reshape(args.batch, 1)
                emitted.append(token.reshape(-1))
                if copy_each_step:
                    # This is the synchronization performed by the current C++
                    # scheduler before it can update request state.
                    token.cpu()
            return torch.stack(emitted)

        def repeated_fused_k1(copy_each_step=False):
            token = first_ids
            emitted = []
            for positions, lengths, table, step_slots in selected:
                token = fused_single.decode(
                    token, positions, lengths, table, step_slots,
                ).reshape(args.batch, 1)
                emitted.append(token.reshape(-1))
                if copy_each_step:
                    token.cpu()
            return torch.stack(emitted)

        reference, reference_logits = eager_trajectory(
            engine.model, cache, first_ids, selected, attention_policy="fa3",
            forward_options=options)
        torch.cuda.synchronize()
        reference_kv = snapshot_slots(cache, slots)
        restore_slots(cache, slots, initial)

        validation_tokens, validation_logits = validation_chunks[steps].replay(first_ids)
        torch.cuda.synchronize()
        validation_kv = snapshot_slots(cache, slots)
        logit_correctness = trajectory_accuracy(
            torch, validation_tokens, validation_logits,
            reference, reference_logits, atol=.05,
        )
        restore_slots(cache, slots, initial)

        materialized_tokens, retained = materialized_chunks[steps].replay(first_ids)
        torch.cuda.synchronize()
        if retained:
            raise AssertionError("materialized production chunk retained validation logits")
        materialized_kv = snapshot_slots(cache, slots)
        restore_slots(cache, slots, initial)

        fused_tokens, fused_logits = fused_head_chunks[steps].replay(first_ids)
        torch.cuda.synchronize()
        if fused_logits:
            raise AssertionError("fused-head production chunk retained validation logits")
        fused_kv = snapshot_slots(cache, slots)
        correctness = {
            "validation_logits": logit_correctness,
            "retained_validation_logits_bytes": sum(
                tensor.numel() * tensor.element_size()
                for tensor in validation_logits
            ),
            "validation_kv": kv_accuracy(torch, validation_kv, reference_kv, atol=.05),
            "materialized_head": {
                "matching_tokens": int((materialized_tokens == reference).sum()),
                "total_tokens": reference.numel(),
                "kv": kv_accuracy(torch, materialized_kv, reference_kv, atol=.05),
            },
            "fused_head": {
                "matching_tokens": int((fused_tokens == reference).sum()),
                "total_tokens": reference.numel(),
                "first_divergence_step": next(
                    (index for index in range(steps)
                     if not torch.equal(fused_tokens[index], reference[index])), None),
                "kv": kv_accuracy(torch, fused_kv, reference_kv, atol=.05),
            },
            "matching_tokens": int((materialized_tokens == reference).sum()),
            "total_tokens": reference.numel(),
            "kv": kv_accuracy(torch, materialized_kv, reference_kv, atol=.05),
            "first_eos_positions": [int(x) for x in
                                    first_eos_positions(
                                        materialized_tokens, args.eos_token_id).cpu()],
        }
        restore_slots(cache, slots, initial)

        def chunk_blocking():
            materialized_chunks[steps].replay(first_ids)
            return materialized_chunks[steps].s_tokens.cpu()

        def chunk_pinned():
            materialized_chunks[steps].replay(first_ids)
            pinned[steps].copy_(materialized_chunks[steps].s_tokens, non_blocking=True)

        def fused_chunk_blocking():
            fused_head_chunks[steps].replay(first_ids)
            return fused_head_chunks[steps].s_tokens.cpu()

        def fused_chunk_pinned():
            fused_head_chunks[steps].replay(first_ids)
            pinned[steps].copy_(fused_head_chunks[steps].s_tokens, non_blocking=True)

        timings = {
            "repeated_k1_gpu_chain": measure(
                torch, repeated_k1, args.warmups, args.repetitions),
            "repeated_k1_cpu_sync_each_step": measure(
                torch, lambda: repeated_k1(True), args.warmups, args.repetitions),
            "repeated_fused_k1_gpu_chain": measure(
                torch, repeated_fused_k1, args.warmups, args.repetitions),
            "repeated_fused_k1_cpu_sync_each_step": measure(
                torch, lambda: repeated_fused_k1(True),
                args.warmups, args.repetitions),
            "chunk_gpu_resident": measure(
                torch, lambda: materialized_chunks[steps].replay(first_ids),
                args.warmups, args.repetitions),
            "chunk_one_blocking_d2h": measure(
                torch, chunk_blocking, args.warmups, args.repetitions),
            "chunk_one_pinned_async_d2h": measure(
                torch, chunk_pinned, args.warmups, args.repetitions),
            "fused_head_chunk_gpu_resident": measure(
                torch, lambda: fused_head_chunks[steps].replay(first_ids),
                args.warmups, args.repetitions),
            "fused_head_chunk_one_blocking_d2h": measure(
                torch, fused_chunk_blocking, args.warmups, args.repetitions),
            "fused_head_chunk_one_pinned_async_d2h": measure(
                torch, fused_chunk_pinned, args.warmups, args.repetitions),
        }
        baseline = timings["repeated_k1_cpu_sync_each_step"]["wall_median_ms"]
        rows.append({
            "steps": steps, "correctness": correctness, "timings": timings,
            "fused_head_speedup_vs_materialized_chunk": {
                suffix: (timings[f"chunk_{suffix}"]["wall_median_ms"] /
                         timings[f"fused_head_chunk_{suffix}"]["wall_median_ms"])
                for suffix in ("gpu_resident", "one_blocking_d2h",
                               "one_pinned_async_d2h")
            },
            "speedup_vs_current_cpu_sync": {
                name: baseline / value["wall_median_ms"]
                for name, value in timings.items()
            },
        })
        restore_slots(cache, slots, initial)

    report = {
        "schema_version": 2, "status": "complete", "kind": "full_model",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model, "model_load_seconds": load_seconds,
        "hub_transfer": hub_transfer,
        "configuration": {"batch": args.batch, "context": args.context,
                          "steps": list(GRAPH_STEPS), "attention": "fa3",
                          "qkv_mode": args.qkv_mode,
                          "residual_rmsnorm": args.residual_rmsnorm,
                          "output_heads": ["materialized_logits_argmax",
                                           "fused_projection_argmax"],
                          "fused_output_head_config": head_config,
                          "warmups": args.warmups,
                          "repetitions": args.repetitions,
                          "eos_token_id": args.eos_token_id},
        "rows": rows,
        "scope": (
            "full 28-layer fixed-regime FA3 decode with real weights and KV cache; "
            "validation graphs retain and compare every logit while production "
            "graphs compare materialized and fused output heads; flexible scheduler "
            "chunk commit is not enabled by this experiment"
        ),
    }
    atomic_json(args.output_dir / "full-model.json", report)
    return report


def analyze(args):
    micro_path = args.output_dir / "micro.json"
    model_path = args.output_dir / "full-model.json"
    if not micro_path.is_file() or not model_path.is_file():
        raise ValueError("run-micro and run-model must both complete before analyze")
    micro = json.loads(micro_path.read_text())
    model = json.loads(model_path.read_text())
    rows = []
    for row in model["rows"]:
        rows.append({"steps": row["steps"],
                     "logit_status": row["correctness"]["validation_logits"]["status"],
                     "max_logit_error": row["correctness"]["validation_logits"][
                         "max_absolute_error"],
                     "materialized_matching_tokens": row["correctness"][
                         "materialized_head"]["matching_tokens"],
                     "fused_matching_tokens": row["correctness"]["fused_head"][
                         "matching_tokens"],
                     "total_tokens": row["correctness"]["total_tokens"],
                     "fused_head_speedup_vs_materialized_chunk": row[
                         "fused_head_speedup_vs_materialized_chunk"],
                     **row["speedup_vs_current_cpu_sync"]})
    report = {
        "schema_version": 2, "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "micro_d2h_speedups": micro["measurements"]["derived"],
        "full_model_speedups": rows,
        "cuda_graph_capabilities": micro["cuda_graph_capabilities"],
        "production_decision": (
            "Report every measured fused-head and K-step effect, including sub-percent "
            "wins. Promote only correctness-passing arms whose paired samples establish "
            "that the effect exceeds run-to-run noise; then add C++ chunk commit and "
            "EOS truncation."
        ),
    }
    atomic_json(args.output_dir / "report.json", report)
    print(json.dumps(report, indent=2))
    return report


def main():
    args = build_parser().parse_args()
    validate(args)
    setup = (check_experiment_setup(args)
             if args.action in ("check-setup", "run-model", "run") else None)
    if args.action == "plan":
        print(json.dumps({"batch": args.batch, "context": args.context,
                          "steps": list(GRAPH_STEPS), "attention": "fa3",
                          "qkv_mode": args.qkv_mode,
                          "residual_rmsnorm": args.residual_rmsnorm,
                          "output_heads": ["materialized_logits_argmax",
                                           "fused_projection_argmax"],
                          "fused_output_head_config": output_head_config(args.batch),
                          "experiments": [
                              "gpu sampling", "per-step vs chunked token D2H",
                              "GPU state update", "K-step graph", "pinned output ring",
                              "metadata H2D vs resident state",
                              "device/conditional graph capability gate",
                          ]}, indent=2))
        return
    if args.action == "check-setup":
        print(json.dumps(setup, indent=2))
        return
    if args.action in ("run-micro", "run"):
        run_micro(args)
    if args.action in ("run-model", "run"):
        run_model(args)
    if args.action in ("analyze", "run"):
        analyze(args)


if __name__ == "__main__":
    main()
