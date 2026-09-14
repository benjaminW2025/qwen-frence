#!/usr/bin/env python3
"""Locate full-model divergence from the experimental native RoPE/KV fusion."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "baseline", ROOT / "engine/graph", ROOT / "engine/kvcache",
             ROOT / "experiments/model"):
    sys.path.insert(0, str(path))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--context", type=int, default=512)
    p.add_argument("--device", default="cuda")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--output", type=Path,
                   default=ROOT / "experiments/results/native-decode-rope-kv-diagnostic.json")
    return p


def difference(torch, actual, expected):
    error = (actual.float() - expected.float()).abs()
    return {"max_abs": float(error.max()), "mean_abs": float(error.mean()),
            "nonzero": int(torch.count_nonzero(error)), "elements": error.numel()}


def main():
    args = parser().parse_args()
    if args.batch < 1 or args.context < 1:
        raise ValueError("batch and context must be positive")
    import torch
    from transformers import AutoModelForCausalLM
    from naive_forward import Qwen2Config
    from weight_loader import QwenWeightLoader
    from paged_graph_decoder import graph_decode_forward
    from benchmark_packed_projection_decode import select_device, stage_case

    if not torch.cuda.is_available():
        raise RuntimeError("requires CUDA")
    select_device(torch, args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = Qwen2Config(use_custom_kernels=True)
    hf = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, attn_implementation="sdpa"
    )
    model = QwenWeightLoader(cfg).convert(hf, args.device, torch.float16)
    del hf
    torch.cuda.empty_cache()
    cache, tensors = stage_case(
        torch, cfg, args.batch, args.context, torch.float16,
        20260914 + args.batch * 100000 + args.context, args.device,
    )

    def run(fused):
        snapshots = {}

        def observe(layer, stage, *values):
            names = ("q", "k", "v") if stage == "rope_kv" else ("x",)
            for name, value in zip(names, values):
                snapshots[(layer, name)] = value.detach().clone()

        logits = graph_decode_forward(
            model, cache, *tensors, max_decode_context_length=args.context + 1,
            enable_native_decode_rope_kv=fused, layer_observer=observe,
        ).detach().clone()
        return logits, snapshots

    with torch.no_grad():
        reference, baseline = run(False)
        candidate, fused = run(True)
        repeat, baseline_repeat = run(False)

    rows = []
    for layer in range(cfg.n_layers):
        row = {"layer": layer}
        for name in ("q", "k", "v", "x"):
            key = (layer, name)
            row[name] = difference(torch, fused[key], baseline[key])
            row[name + "_baseline_repeat"] = difference(
                torch, baseline_repeat[key], baseline[key]
            )
        rows.append(row)
        print(f"layer {layer:02d}: q={row['q']['max_abs']:.6g} "
              f"k={row['k']['max_abs']:.6g} v={row['v']['max_abs']:.6g} "
              f"x={row['x']['max_abs']:.6g} "
              f"repeat_x={row['x_baseline_repeat']['max_abs']:.6g}", flush=True)

    result = {
        "batch": args.batch, "context": args.context, "model": args.model,
        "logits": difference(torch, candidate, reference),
        "logits_baseline_repeat": difference(torch, repeat, reference),
        "matching_tokens": int((candidate.argmax(-1) == reference.argmax(-1)).sum()),
        "total_tokens": reference.shape[0] * reference.shape[1],
        "layers": rows,
    }
    print("logits:", result["logits"])
    print("baseline repeat:", result["logits_baseline_repeat"])
    print(f"matching tokens: {result['matching_tokens']}/{result['total_tokens']}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"results: {args.output}")


if __name__ == "__main__":
    main()
