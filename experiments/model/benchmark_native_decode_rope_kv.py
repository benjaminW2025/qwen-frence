#!/usr/bin/env python3
"""Real-Qwen eager decode A/B for native RoPE plus paged KV-write fusion.

The two variants share weights, input tensors, resident KV, and dispatch. Model
loading, KV staging, and scheduler work are outside the paired timing.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LOGIT_ATOL = .1
LOGIT_RTOL = .01
for path in (HERE, ROOT / "baseline", ROOT / "engine/graph", ROOT / "engine/kvcache", ROOT / "benchmarks"):
    sys.path.insert(0, str(path))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=("smoke", "full"), default="full")
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--samples", type=int, default=7)
    p.add_argument("--warmups", type=int, default=3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--output-dir", type=Path,
                   default=ROOT / "experiments/results/native-decode-rope-kv-model")
    return p


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def timed(torch, fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def main():
    args = parser().parse_args()
    if min(args.trials, args.samples) < 1 or args.warmups < 0:
        raise ValueError("trials/samples must be positive and warmups nonnegative")
    import torch
    from transformers import AutoModelForCausalLM
    from naive_forward import Qwen2Config
    from weight_loader import QwenWeightLoader
    from paged_graph_decoder import graph_decode_forward
    from benchmark_packed_projection_decode import select_device, stage_case
    from run_benchmarks import system_metadata

    if not torch.cuda.is_available():
        raise RuntimeError("requires CUDA")
    select_device(torch, args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    cases = ([(1, 512), (8, 2048)] if args.preset == "smoke"
             else [(batch, context) for batch in (1, 8, 32, 64, 128)
                   for context in (512, 8192)])
    sources = (
        Path(__file__), HERE / "benchmark_packed_projection_decode.py",
        ROOT / "engine/graph/paged_graph_decoder.py",
        ROOT / "custom_kernels/rope_kv_write.py",
        ROOT / "baseline/kernel_dispatch.py",
    )
    config = {
        "preset": args.preset, "trials": args.trials, "samples": args.samples,
        "warmups": args.warmups, "device": args.device, "model": args.model,
        "cases": cases, "packed_qkv": True, "packed_gate_up": True,
        "sources_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sources},
    }
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()).get("fingerprint") != fingerprint:
            raise ValueError("output directory contains a different protocol; choose another")
    else:
        atomic_json(manifest_path, {"fingerprint": fingerprint, "configuration": config,
                                    "system": system_metadata(),
                                    "created_at": datetime.now(timezone.utc).isoformat()})

    pending = [(batch, context) for batch, context in cases
               if not (args.output_dir / "trials" / f"b{batch}-l{context}.json").exists()]
    model = None
    if pending:
        cfg = Qwen2Config(use_custom_kernels=True)
        hf = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16, attn_implementation="sdpa"
        )
        model = QwenWeightLoader(cfg).convert(hf, args.device, torch.float16)
        del hf
        torch.cuda.empty_cache()

    with torch.no_grad():
        for batch, context in pending:
            cache, tensors = stage_case(
                torch, model.cfg, batch, context, torch.float16,
                20260914 + batch * 100000 + context, args.device,
            )
            launch = {
                "baseline": lambda: graph_decode_forward(
                    model, cache, *tensors, max_decode_context_length=context + 1
                ),
                "fused": lambda: graph_decode_forward(
                    model, cache, *tensors, max_decode_context_length=context + 1,
                    enable_native_decode_rope_kv=True,
                ),
            }
            reference = launch["baseline"]()
            candidate = launch["fused"]()
            reference_float = reference.float()
            candidate_float = candidate.float()
            error = (candidate_float - reference_float).abs()
            max_logit_error = float(error.max())
            mean_logit_error = float(error.mean())
            reference_tokens = reference.argmax(-1)
            candidate_tokens = candidate.argmax(-1)
            matching_tokens = int((candidate_tokens == reference_tokens).sum())
            total_tokens = reference_tokens.numel()
            if matching_tokens != total_tokens:
                raise AssertionError(
                    f"sampled tokens differ at B={batch}, L={context}: "
                    f"{matching_tokens}/{total_tokens} agree, max logit error={max_logit_error:.6f}"
                )
            try:
                torch.testing.assert_close(
                    candidate_float, reference_float, atol=LOGIT_ATOL, rtol=LOGIT_RTOL
                )
            except AssertionError as exc:
                raise AssertionError(
                    f"logits differ at B={batch}, L={context}; "
                    f"max error={max_logit_error:.6f}, mean error={mean_logit_error:.6f}"
                ) from exc
            for fn in launch.values():
                for _ in range(args.warmups):
                    fn()
            torch.cuda.synchronize()

            trials = []
            for trial in range(args.trials):
                samples = {name: [] for name in launch}
                for sample in range(args.samples):
                    order = list(launch)
                    random.Random(20260914 + batch * 100003 + context * 101
                                  + trial * 1009 + sample).shuffle(order)
                    for name in order:
                        samples[name].append(timed(torch, launch[name]))
                baseline_ms = statistics.median(samples["baseline"])
                fused_ms = statistics.median(samples["fused"])
                trials.append({"trial": trial, "samples_ms": samples,
                               "baseline_median_ms": baseline_ms,
                               "fused_median_ms": fused_ms,
                               "speedup": baseline_ms / fused_ms})
            checkpoint = args.output_dir / "trials" / f"b{batch}-l{context}.json"
            atomic_json(checkpoint, {"status": "complete", "fingerprint": fingerprint,
                                     "batch": batch, "context": context,
                                     "max_logit_error": max_logit_error,
                                     "mean_logit_error": mean_logit_error,
                                     "matching_tokens": matching_tokens,
                                     "total_tokens": total_tokens, "trials": trials})
            print(f"B={batch} L={context}: "
                  f"{statistics.median(t['speedup'] for t in trials):.3f}x", flush=True)
            del launch, cache, tensors, reference, candidate
            torch.cuda.empty_cache()

    results = []
    for batch, context in cases:
        checkpoint = args.output_dir / "trials" / f"b{batch}-l{context}.json"
        payload = json.loads(checkpoint.read_text())
        if (payload.get("status") != "complete" or payload.get("fingerprint") != fingerprint
                or payload.get("batch") != batch or payload.get("context") != context
                or len(payload.get("trials", [])) != args.trials):
            raise ValueError(f"invalid case checkpoint: {checkpoint}")
        results.append(payload)
    output = args.output_dir / "report.json"
    atomic_json(output, {"configuration": config, "fingerprint": fingerprint,
                         "system": system_metadata(), "cases": results})
    print(f"results: {output}")


if __name__ == "__main__":
    main()
