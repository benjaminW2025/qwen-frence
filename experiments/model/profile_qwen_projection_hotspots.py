#!/usr/bin/env python3
"""Measure Qwen2.5-1.5B decode projection hotspots and trace them with NVTX.

``bench`` times isolated, real-weight GEMMs with CUDA events.  ``trace`` runs
the same operations under named NVTX ranges for Nsight Systems.  The packed
QKV and gate/up variants only test GEMM launch/layout potential: their packed
weights are built once, outside every timed or traced region.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for directory in (ROOT / "baseline", ROOT / "benchmarks"):
    value = str(directory)
    if value not in sys.path:
        sys.path.insert(0, value)


def batch_sizes(value: str) -> tuple[int, ...]:
    try:
        sizes = tuple(int(item) for item in value.split(",") if item)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--batches must be comma-separated integers") from exc
    if not sizes or any(size < 1 for size in sizes):
        raise argparse.ArgumentTypeError("--batches must contain positive integers")
    if len(set(sizes)) != len(sizes):
        raise argparse.ArgumentTypeError("--batches must not contain duplicates")
    return sizes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("bench", "trace"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--batches", type=batch_sizes,
                        default=(1, 2, 4, 8, 16, 32, 64, 96, 128, 256),
                        help="Decode batch sizes, comma-separated (default: 1,2,4,8,16,32,64,96,128,256)")
    parser.add_argument("--trace-batch", type=int, default=32,
                        help="One batch size from --batches, used by trace")
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--trace-repetitions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--cuda-profiler-range", action="store_true",
                        help="Use cudaProfilerStart/Stop around trace; pair with nsys capture-range=cudaProfilerApi")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "experiments" / "results" / "model-projections")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.warmups < 0:
        raise ValueError("--warmups must be non-negative")
    if args.repetitions < 1 or args.trace_repetitions < 1:
        raise ValueError("repetition counts must be positive")
    if args.trace_batch < 1:
        raise ValueError("--trace-batch must be positive")
    if args.command == "trace" and args.trace_batch not in args.batches:
        raise ValueError("--trace-batch must be included in --batches")
    if args.cuda_profiler_range and args.command != "trace":
        raise ValueError("--cuda-profiler-range requires the trace command")


@contextmanager
def nvtx_range(torch, label: str):
    """Mark one logical projection region for Nsight Systems and torch profiler."""
    from torch.profiler import record_function

    torch.cuda.nvtx.range_push(label)
    with record_function(label):
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lo, hi = int(position), min(int(position) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def time_cuda(torch, fn, repetitions: int) -> dict[str, float | list[float]]:
    samples = []
    for _ in range(repetitions):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return {
        "median_ms": statistics.median(samples),
        "p10_ms": percentile(samples, 0.10),
        "p90_ms": percentile(samples, 0.90),
        "samples_ms": samples,
    }


def make_operations(torch, model, batch: int, dtype, seed: int):
    """Return independently callable real-weight operations for one decode batch."""
    import torch.nn.functional as F

    layer = model.layers[0]
    generator = torch.Generator(device=model.embed.weight.device).manual_seed(seed + batch)
    hidden = torch.randn(batch, model.cfg.d_model, device=model.embed.weight.device,
                         dtype=dtype, generator=generator)
    qkv_weight = torch.cat((layer.q_proj.weight, layer.k_proj.weight, layer.v_proj.weight), dim=0)
    qkv_bias = torch.cat((layer.q_proj.bias, layer.k_proj.bias, layer.v_proj.bias), dim=0)
    gate_up_weight = torch.cat((layer.gate_proj.weight, layer.up_proj.weight), dim=0)
    q_width = layer.q_proj.out_features
    k_width = layer.k_proj.out_features
    v_width = layer.v_proj.out_features
    gate_width = layer.gate_proj.out_features

    def qkv_separate():
        return layer.q_proj(hidden), layer.k_proj(hidden), layer.v_proj(hidden)

    def qkv_packed():
        return F.linear(hidden, qkv_weight, qkv_bias).split((q_width, k_width, v_width), dim=-1)

    def gate_up_separate():
        return layer.gate_proj(hidden), layer.up_proj(hidden)

    def gate_up_packed():
        return F.linear(hidden, gate_up_weight).split((gate_width, gate_width), dim=-1)

    # Keep this separate: it says whether the two input projections are worth
    # optimizing when the activation and down projection remain unchanged.
    gate, up = gate_up_separate()

    def swiglu():
        return F.silu(gate) * up

    activated = swiglu()

    def mlp_down():
        return layer.down_proj(activated)

    def lm_head_logits():
        return model.lm_head(hidden)

    def lm_head_logits_argmax():
        return model.lm_head(hidden).argmax(dim=-1)

    operations = {
        "qkv_separate": qkv_separate,
        "qkv_packed": qkv_packed,
        "mlp_gate_up_separate": gate_up_separate,
        "mlp_gate_up_packed": gate_up_packed,
        "mlp_swiglu": swiglu,
        "mlp_down": mlp_down,
        "lm_head_logits": lm_head_logits,
        "lm_head_logits_argmax": lm_head_logits_argmax,
    }
    details = {
        "hidden_shape": list(hidden.shape),
        "qkv_output_features": [q_width, k_width, v_width],
        "gate_up_output_features": [gate_width, gate_width],
        "packed_weight_bytes": {
            "qkv": qkv_weight.numel() * qkv_weight.element_size() + qkv_bias.numel() * qkv_bias.element_size(),
            "gate_up": gate_up_weight.numel() * gate_up_weight.element_size(),
        },
    }
    return operations, details


def validate_packed(torch, operations) -> None:
    separate_qkv, packed_qkv = operations["qkv_separate"](), operations["qkv_packed"]()
    separate_gate, packed_gate = operations["mlp_gate_up_separate"](), operations["mlp_gate_up_packed"]()
    for expected, observed in zip((*separate_qkv, *separate_gate), (*packed_qkv, *packed_gate)):
        torch.testing.assert_close(observed, expected, rtol=2e-3, atol=2e-3)


def load_model(args, torch):
    from naive_forward import Qwen2Config
    from weight_loader import QwenWeightLoader

    dtype = getattr(torch, args.dtype)
    # Keep this benchmark in its reference layout so it can compare it with an
    # ephemeral packed candidate. Production loading packs these weights.
    model = QwenWeightLoader(Qwen2Config(pack_qkv=False, pack_gate_up=False)).load_pretrained(
        args.model, args.device, dtype
    )
    return model, dtype


def bench(args, torch, model, dtype) -> dict:
    rows = []
    shapes = {}
    summaries = []
    for batch in args.batches:
        operations, details = make_operations(torch, model, batch, dtype, args.seed)
        shapes[str(batch)] = details
        validate_packed(torch, operations)
        for fn in operations.values():
            for _ in range(args.warmups):
                fn()
        torch.cuda.synchronize()
        for name, fn in operations.items():
            rows.append({"batch": batch, "operation": name, **time_cuda(torch, fn, args.repetitions)})
        measured = {row["operation"]: row["median_ms"] for row in rows if row["batch"] == batch}
        separate_layer = sum(measured[name] for name in (
            "qkv_separate", "mlp_gate_up_separate", "mlp_swiglu", "mlp_down",
        ))
        packed_layer = sum(measured[name] for name in (
            "qkv_packed", "mlp_gate_up_packed", "mlp_swiglu", "mlp_down",
        ))
        summaries.append({
            "batch": batch,
            "one_layer_projection_path_ms": separate_layer,
            "projected_28_layer_path_plus_sampling_ms": (
                model.cfg.n_layers * separate_layer + measured["lm_head_logits_argmax"]
            ),
            "projected_packed_28_layer_path_plus_sampling_ms": (
                model.cfg.n_layers * packed_layer + measured["lm_head_logits_argmax"]
            ),
            "packed_projection_path_savings_ms_per_layer": separate_layer - packed_layer,
            "lm_head_logits_argmax_ms": measured["lm_head_logits_argmax"],
        })
    return {"measurements": rows, "shape_details": shapes, "decode_path_estimates": summaries}


def trace(args, torch, model, dtype) -> dict:
    operations, details = make_operations(torch, model, args.trace_batch, dtype, args.seed)
    validate_packed(torch, operations)
    for fn in operations.values():
        for _ in range(args.warmups):
            fn()
    torch.cuda.synchronize()
    if args.cuda_profiler_range:
        torch.cuda.cudart().cudaProfilerStart()
    try:
        with nvtx_range(torch, "qwen_projection_trace"):
            for _ in range(args.trace_repetitions):
                for name, fn in operations.items():
                    with nvtx_range(torch, f"qwen_projection/{name}"):
                        fn()
        torch.cuda.synchronize()
    finally:
        if args.cuda_profiler_range:
            torch.cuda.cudart().cudaProfilerStop()
    return {"trace_batch": args.trace_batch, "trace_repetitions": args.trace_repetitions,
            "shape_details": details}


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    import torch
    from run_benchmarks import system_metadata

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("this experiment requires a CUDA device")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_grad_enabled(False)
    model, dtype = load_model(args, torch)
    payload = bench(args, torch, model, dtype) if args.command == "bench" else trace(args, torch, model, dtype)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output_dir / f"{args.command}-{stamp}.json"
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": args.command,
        "configuration": vars(args) | {"batches": list(args.batches), "output_dir": str(args.output_dir)},
        "system": system_metadata(),
        "model_config": vars(model.cfg),
        "tf32": False,
        "notes": [
            "All measurements are eager CUDA-event timings of isolated decode-shaped operations.",
            "Packed weights are materialized once before measurement; this is an implementation candidate, not the current production layout.",
            "lm_head_logits_argmax includes both vocabulary projection and greedy sampling.",
        ],
        **payload,
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    if args.command == "bench":
        for row in result["measurements"]:
            print(f"B={row['batch']:>3} {row['operation']:<24} {row['median_ms']:.4f} ms "
                  f"(p10-p90 {row['p10_ms']:.4f}-{row['p90_ms']:.4f})")
    print(f"results: {output}")


if __name__ == "__main__":
    main()
