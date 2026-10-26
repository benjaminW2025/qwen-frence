#!/usr/bin/env python3
"""Local FP8 KV sensitivity and tile-local dequantization experiments.

CUDA: python experiments/quant/attention.py --output artifacts/kv-tiles.json
CPU:  ... --device cpu --reference-only --output artifacts/kv-reference.json

Optional --input contains a weights_only-loadable list of dicts with `q`, `k`,
`v`, and optional `layer`: Q [B,HQ,Q,D], K/V [B,HK,L,D], after RoPE, with the
queries at the final Q positions of L. It must contain only valid, unpadded
positions. Real activations are needed for model sensitivity conclusions;
default synthetic tensors test implementation behavior only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def quantize(x, scaling="token"):
    if x.ndim != 4 or not x.numel() or not torch.isfinite(x).all():
        raise ValueError("expected finite, nonempty B,H,L,D values")
    if scaling not in ("token", "head"):
        raise ValueError("scaling must be token or head")
    values = x.float()
    maximum = values.abs().amax(-1)
    if scaling == "head":
        maximum = maximum.amax(-1, keepdim=True)
    scale = (maximum / 448.).clamp_min(1e-12)
    # Keep E4M3 as bytes: reading it never requires native FP8 instructions.
    packed = (values / scale[..., None]).clamp(-448., 448.).to(torch.float8_e4m3fn)
    return packed.view(torch.uint8).contiguous(), scale.contiguous()


def dequantize(bits, scales):
    """Portable reference matching tile-local decoding, including E4M3 subnormals."""
    values = bits.int()
    exponent, mantissa = (values >> 3) & 15, values & 7
    magnitude = torch.where(exponent == 0, mantissa.float() * 2. ** -9,
                            (1. + mantissa.float() / 8.) * torch.exp2(exponent.float() - 7.))
    magnitude = torch.where((exponent == 15) & (mantissa == 7), float("nan"), magnitude)
    signed = torch.where((values & 128) != 0, -magnitude, magnitude)
    return (signed * scales[..., None]).half()


def effective_kv(k, v, k8, v8, ks, vs, mask, mode):
    selected = mask.bool()[None, :, None, None]
    key = torch.where(selected, dequantize(k8, ks), k) if mode in ("k", "kv") else k
    value = torch.where(selected, dequantize(v8, vs), v) if mode in ("v", "kv") else v
    return key, value


def oracle(q, k, v):
    """Explicit FP32 GQA with bottom-right causal alignment, for small probes."""
    group = q.shape[1] // k.shape[1]
    keys, values = k.float().repeat_interleave(group, 1), v.float().repeat_interleave(group, 1)
    scores = q.float() @ keys.transpose(-1, -2) * q.shape[-1] ** -.5
    qi = torch.arange(q.shape[2], device=q.device)[:, None]
    ki = torch.arange(k.shape[2], device=q.device)[None, :]
    visible = ki <= k.shape[2] - q.shape[2] + qi
    return (scores.masked_fill(~visible, -float("inf")).softmax(-1) @ values).half()


def metrics(reference, actual):
    r, a = reference.float(), actual.float()
    if not torch.isfinite(r).all() or not torch.isfinite(a).all():
        return {"finite": False}
    diff = a - r
    return {"finite": True, "max_abs": float(diff.abs().max()),
            "relative_l2": float(diff.norm() / r.norm().clamp_min(1e-12)),
            "per_query_head_relative_l2": [float(diff[:, h].norm() / r[:, h].norm().clamp_min(1e-12))
                                          for h in range(r.shape[1])]}


def variants(kv_heads, sweep):
    selections = [tuple(range(kv_heads))]
    if sweep and kv_heads > 1:
        selections += [(h,) for h in range(kv_heads)]
    return [(mode, heads) for heads in selections for mode in ("k", "v", "kv")]


def time_call(call, warmup, repeats):
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    samples = []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        start.record()
        result = call()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
        del result
    return {"median_ms": statistics.median(samples), "samples_ms": samples}


def args_for(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--reference-only", action="store_true")
    p.add_argument("--input", type=Path)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--query-heads", type=int, default=12)
    p.add_argument("--kv-heads", type=int, default=2)
    p.add_argument("--head-dim", type=int, choices=(32, 64, 128), default=128)
    p.add_argument("--contexts", nargs="+", type=int, default=[256, 2048])
    p.add_argument("--queries", nargs="+", type=int, default=[1, 32])
    p.add_argument("--scaling", choices=("token", "head"), default="token")
    p.add_argument("--sweep-heads", action="store_true")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--seed", type=int, default=31)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args(argv)
    if min(args.batch, args.query_heads, args.kv_heads, *args.contexts, *args.queries) < 1:
        p.error("dimensions must be positive")
    if args.query_heads % args.kv_heads or args.warmup < 0 or args.repeats < 1:
        p.error("invalid GQA grouping or timing counts")
    if args.output.exists():
        p.error("use a new report path")
    if torch.device(args.device).type != "cuda" and not args.reference_only:
        p.error("non-CUDA execution requires --reference-only")
    return args


def cases(args, device):
    if args.input:
        captured = torch.load(args.input, map_location="cpu", weights_only=True)
        if not isinstance(captured, list) or not captured:
            raise ValueError("input must contain a nonempty list of Q/K/V records")
        for row in captured:
            q, k, v = [row[name].to(device=device, dtype=torch.float16).contiguous() for name in ("q", "k", "v")]
            yield str(row.get("layer", "unspecified")), q, k, v
    else:
        generator = torch.Generator(device=device).manual_seed(args.seed)
        for length in args.contexts:
            for queries in args.queries:
                if queries > length:
                    continue
                q = torch.randn(args.batch, args.query_heads, queries, args.head_dim,
                                device=device, dtype=torch.float16, generator=generator)
                k = torch.randn(args.batch, args.kv_heads, length, args.head_dim,
                                device=device, dtype=torch.float16, generator=generator)
                v = torch.randn(k.shape, device=device, dtype=torch.float16, generator=generator)
                yield "synthetic", q, k, v


@torch.no_grad()
def main(argv=None):
    args = args_for(argv)
    device = torch.device(args.device)
    kernel = None
    if not args.reference_only:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; use --device cpu --reference-only for sensitivity math")
        torch.cuda.set_device(device)
        if torch.cuda.get_device_capability(device)[0] < 8:
            raise RuntimeError("tile-local kernel requires Ampere-or-newer CUDA hardware")
        from experiments.quant.kernels.attention import attention
        kernel = attention
    from experiments.quant.projections import strict_accumulation
    report = {"status": "running", "torch": torch.__version__, "device": str(device),
              "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
              "data": "captured QKV" if args.input else "synthetic; not a model-quality result",
              "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest() if args.input else None,
              "scaling": args.scaling, "seed": args.seed,
              "kernel": "tile-local KV dequantization + FP16 dot; NOT FA3; contiguous KV, not paged",
              "precision": "FP16 Q, FP8 selected KV, FP16 dot operands, FP32 accumulators",
              "scope": "isolated attention; excludes cache writes/quantization setup; no model or serving speed claims",
              "cases": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with strict_accumulation():
        for label, q, k, v in cases(args, device):
            if (q.ndim != 4 or k.ndim != 4 or k.shape != v.shape or q.shape[0] != k.shape[0]
                    or q.shape[-1] != k.shape[-1] or q.shape[1] % k.shape[1]
                    or not 0 < q.shape[2] <= k.shape[2]):
                raise ValueError("incompatible Q/K/V capture shapes")
            k8, ks = quantize(k, args.scaling)
            v8, vs = quantize(v, args.scaling)
            baseline = oracle(q, k, v)
            control_time = None
            specs = variants(k.shape[1], args.sweep_heads)
            random.Random(args.seed).shuffle(specs)
            for mode, heads in [("off", tuple(range(k.shape[1]))), *specs]:
                mask = torch.zeros(k.shape[1], device=device, dtype=torch.int32)
                mask[list(heads)] = 1
                ek, ev = effective_kv(k, v, k8, v8, ks, vs, mask, mode)
                reference = oracle(q, ek, ev)
                row = {"layer": label, "q_shape": list(q.shape), "kv_shape": list(k.shape),
                       "mode": mode, "kv_heads": heads,
                       "storage_error": metrics(baseline, reference)}
                if kernel:
                    run = lambda: kernel(q, k, v, k8, v8, ks, vs, mask, mode)
                    actual = run()
                    torch.testing.assert_close(actual, reference, atol=.01, rtol=.02)
                    row["kernel_vs_quantized_reference"] = metrics(reference, actual)
                    row["total_error"] = metrics(baseline, actual)
                    timing = time_call(run, args.warmup, args.repeats)
                    row["tile_local_latency"] = timing
                    if mode == "off":
                        control_time = timing["median_ms"]
                    row["speedup_vs_same_kernel_fp16"] = control_time / timing["median_ms"]
                    if mode != "off":
                        def staged():
                            staged_k, staged_v = effective_kv(k, v, k8, v8, ks, vs, mask, mode)
                            return kernel(q, staged_k, staged_v, k8, v8, ks, vs, mask, "off")
                        torch.testing.assert_close(staged(), actual, atol=.002, rtol=.003)
                        row["full_staging_latency"] = time_call(staged, args.warmup, args.repeats)
                report["cases"].append(row)
                print(f"{label} Q={q.shape[2]} L={k.shape[2]} {mode} heads={heads}: "
                      f"relative_l2={row['storage_error'].get('relative_l2')}", flush=True)
                args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    report["status"] = "complete"
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0 if all(row["storage_error"]["finite"] for row in report["cases"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
