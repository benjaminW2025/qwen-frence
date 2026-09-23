#!/usr/bin/env python3
"""Screen production-weight packed-prefill GEMM routes at exact graph bucket sizes.

This is a candidate screen, not an end-to-end speedup claim. A winning route
still needs a full piecewise-engine correctness and throughput comparison.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import statistics
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for directory in (ROOT / "experiments/integration", ROOT / "benchmarks",
                  ROOT / "baseline", ROOT / "engine/model_runner",
                  ROOT / "engine/cpp/build"):
    sys.path.insert(0, str(directory))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "check-setup", "run"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rows", nargs="+", type=int, default=[2048, 8192],
                        help="exact packed prefill token buckets to capture")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "experiments/results/prefill-gemm")
    return parser


def validate(args):
    if not args.rows or any(row < 1 for row in args.rows):
        raise ValueError("rows must contain positive token counts")
    if args.warmups < 1 or args.repetitions < 1:
        raise ValueError("warmups and repetitions must be positive")


def plan(args):
    # These dimensions are the pinned Qwen2.5-1.5B configuration. The run
    # checks the actual loaded model and records its real weight shapes.
    return {"model": args.model, "device": args.device, "rows": args.rows,
            "expected_model_dimensions": {"hidden": 1536, "intermediate": 8960},
            "operations": ["gate_up_packed", "mlp_down", "mlp_with_swiglu"],
            "routes": ["current", "cublas", "cublaslt"],
            "execution": "one exact-shape CUDA graph replay per operation",
            "caveat": "candidate screen only; production acceptance requires full-model validation"}


def available_routes(torch):
    selector = getattr(torch.backends.cuda, "preferred_blas_library", None)
    if selector is None:
        return [("current", None)], None, None
    original = selector()
    routes = [("current", original)]
    try:
        for name in ("cublas", "cublaslt"):
            try:
                selector(name)
                routes.append((name, name))
            except (RuntimeError, ValueError):
                pass
    finally:
        selector(original)
    return routes, selector, original


def capture(torch, operation, warmups):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmups):
            operation()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = operation()
    return graph, output


def time_graph(torch, graph, warmups, repetitions):
    for _ in range(warmups):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return {"median_ms": statistics.median(samples), "samples_ms": samples}


def compare(torch, actual, reference):
    difference = (actual.float() - reference.float()).abs()
    threshold = .05 + .01 * reference.float().abs()
    outside = difference > threshold
    return {"allclose": not bool(outside.any()),
            "outside_tolerance": int(outside.sum()),
            "elements": outside.numel(),
            "max_abs": float(difference.max()),
            "mean_abs": float(difference.mean()),
            "atol": .05, "rtol": .01}


def check_setup(device):
    import torch
    from model_setup import prepare_hub_transfer

    transfer = prepare_hub_transfer()
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("a CUDA device is required")
    if importlib.util.find_spec("triton") is None:
        raise RuntimeError("Triton is required for the production SwiGLU kernel")
    import triton  # noqa: F401 - verify the installed package actually imports
    return {"device": device, "gpu": torch.cuda.get_device_name(device),
            "triton_available": True, "hub_transfer": transfer,
            "note": "No model loaded and no CUDA graph captured"}


def run(args):
    import torch
    from benchmark_latest_vs_vllm import resolve_model_source
    from model_setup import load_model_only
    from naive_forward import apply_swiglu

    startup = check_setup(args.device)
    source = resolve_model_source(args)
    engine, load_seconds, transfer = load_model_only(
        source, args.device, "float16", hub_transfer=startup["hub_transfer"])
    cfg = engine.cfg
    if (cfg.d_model, cfg.d_ff) != (1536, 8960):
        raise ValueError(f"expected Qwen2.5-1.5B dimensions, got {(cfg.d_model, cfg.d_ff)}")
    layer = engine.model.layers[0]
    if layer.gate_up_proj is None:
        raise ValueError("model weights were not packed into gate/up projection")
    routes, selector, original = available_routes(torch)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "model_source": str(source), "model_load_seconds": load_seconds,
              "hub_transfer": transfer, "startup": startup,
              "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(args.device), "dtype": "float16",
              "warmups": args.warmups, "repetitions": args.repetitions,
              "weights": {"gate_up": list(layer.gate_up_proj.weight.shape),
                          "down": list(layer.down_proj.weight.shape)},
              "rows": {}}

    try:
        for row_count in args.rows:
            generator = torch.Generator(device=args.device).manual_seed(args.seed + row_count)
            hidden = torch.randn(row_count, cfg.d_model, device=args.device,
                                 dtype=torch.float16, generator=generator)
            activated = torch.randn(row_count, cfg.d_ff, device=args.device,
                                    dtype=torch.float16, generator=generator)

            def full_mlp():
                gate, up = layer.project_gate_up(hidden)
                return layer.down_proj(apply_swiglu(
                    gate, up, cfg, enable_regime_fusions=True))

            operations = {"gate_up_packed": lambda: layer.gate_up_proj(hidden),
                          "mlp_down": lambda: layer.down_proj(activated),
                          "mlp_with_swiglu": full_mlp}
            row_report = {}
            report["rows"][str(row_count)] = row_report
            # Keep only one large reference and one graph alive at a time.
            for name, operation in operations.items():
                if selector is not None:
                    selector(original)
                reference = operation().detach().clone()
                measurements = {}
                row_report[name] = measurements
                for route_name, route in routes:
                    try:
                        if selector is not None:
                            selector(route)
                        graph, output = capture(torch, operation, args.warmups)
                        graph.replay()
                        torch.cuda.synchronize()
                        correctness = compare(torch, output, reference)
                        timing = time_graph(torch, graph, args.warmups,
                                            args.repetitions)
                        measurements[route_name] = {
                            "status": "measured", "correctness": correctness,
                            "graph": timing}
                        del graph, output
                    except (RuntimeError, ValueError) as error:
                        measurements[route_name] = {
                            "status": "unsupported", "error": str(error)}
                    finally:
                        if selector is not None:
                            selector(original)
                    # Make progress recoverable if one later shape fails.
                    args.output_dir.mkdir(parents=True, exist_ok=True)
                    (args.output_dir / "report.json").write_text(
                        json.dumps(report, indent=2) + "\n")
                baseline = measurements.get("current", {})
                if baseline.get("status") == "measured":
                    base_ms = baseline["graph"]["median_ms"]
                    for measurement in measurements.values():
                        if measurement["status"] == "measured":
                            measurement["speedup_vs_current"] = (
                                base_ms / measurement["graph"]["median_ms"])
                (args.output_dir / "report.json").write_text(
                    json.dumps(report, indent=2) + "\n")
                del reference
                torch.cuda.empty_cache()
    finally:
        if selector is not None:
            selector(original)

    report["interpretation"] = (
        "Route speedups are only single-layer graph timings. Accept a candidate "
        "only if numerical checks pass and a full production piecewise run improves.")
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    args = build_parser().parse_args()
    validate(args)
    if args.action == "plan":
        print(json.dumps(plan(args), indent=2))
    elif args.action == "check-setup":
        print(json.dumps(check_setup(args.device), indent=2))
    else:
        report = run(args)
        for row_count, operations in report["rows"].items():
            for name, routes in operations.items():
                print(f"rows={row_count} {name}")
                for route, result in routes.items():
                    if result["status"] == "measured":
                        speedup = result.get("speedup_vs_current")
                        speedup_text = (f"{speedup:.3f}x" if speedup is not None
                                        else "n/a")
                        print(f"  {route}: {result['graph']['median_ms']:.4f} ms, "
                              f"{speedup_text}, "
                              f"correct={result['correctness']['allclose']}")
                    else:
                        print(f"  {route}: unsupported ({result['error']})")


if __name__ == "__main__":
    main()
