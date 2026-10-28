"""Isolate FP8 key sensitivity in saved, post-RoPE model Q/K/V captures.

All attention arithmetic is the same FP32 reference: only selected stored K
values change. No FA3 or custom attention kernel is involved. Queries occupy
final positions of each unpadded capture. This is a local sensitivity probe,
not an accumulated-error or final-logit experiment.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from experiments.quant.attention import quantize, dequantize
from experiments.quant.projections import strict_accumulation


def relative_error(reference, actual):
    return float((actual.float() - reference.float()).norm()
                 / reference.float().norm().clamp_min(1e-12))


def attention(q, k, v):
    group = q.shape[1] // k.shape[1]
    scores = q.float() @ k.float().repeat_interleave(group, 1).transpose(-1, -2)
    scores *= q.shape[-1] ** -.5
    visible = torch.arange(k.shape[2], device=q.device)[None, :] <= (
        k.shape[2] - q.shape[2] + torch.arange(q.shape[2], device=q.device)[:, None])
    probabilities = scores.masked_fill(~visible, -torch.inf).softmax(-1)
    output = probabilities @ v.float().repeat_interleave(group, 1)
    return scores, probabilities, output, visible


def compare(q, reference_k, candidate_k, v, baseline=None):
    scores, p, output, visible = baseline or attention(q, reference_k, v)
    changed_scores, changed_p, changed_output, _ = attention(q, candidate_k, v)
    delta = (changed_scores - scores)[..., visible]
    # clamp avoids log(0); mathematically masked positions contribute zero.
    kl = (p * (p.clamp_min(1e-30).log() - changed_p.clamp_min(1e-30).log())).sum(-1)
    return {
        "key_relative_l2": relative_error(reference_k, candidate_k),
        "score_rmse": float(delta.square().mean().sqrt()),
        "score_max_abs": float(delta.abs().max()),
        "probability_total_variation_mean": float((p - changed_p).abs().sum(-1).mean() / 2),
        "probability_kl_reference_to_candidate_mean": float(kl.mean().clamp_min(0)),
        "attention_argmax_agreement": float((p.argmax(-1) == changed_p.argmax(-1)).float().mean()),
        "output_relative_l2": relative_error(output, changed_output),
        "output_max_abs": float((output - changed_output).abs().max()),
    }


def diagnose(probe, scaling="token", rescue_counts=(1, 4, 16), top=8):
    q, k, v = (probe[name] for name in ("q", "k", "v"))
    if (any(x.ndim != 4 or not x.numel() or not torch.isfinite(x).all() for x in (q, k, v))
            or k.shape != v.shape or q.shape[0] != k.shape[0]
            or q.shape[-1] != k.shape[-1] or q.shape[1] % k.shape[1]
            or q.shape[2] > k.shape[2]):
        raise ValueError("expected finite Q[B,HQ,Q,D], K/V[B,HK,L,D] with valid GQA and Q<=L")
    bits, scales = quantize(k, scaling)
    restored = dequantize(bits, scales)
    native = (bits.view(torch.float8_e4m3fn).float() * scales[..., None]).half()
    if not torch.equal(restored, native):
        raise RuntimeError("portable FP8 decoding disagrees with PyTorch")
    baseline = attention(q, k, v)
    group = q.shape[1] // k.shape[1]
    heads = []
    for head in range(k.shape[1]):
        query = q[:, head * group:(head + 1) * group]
        key, value = k[:, head:head+1], v[:, head:head+1]
        candidate = restored[:, head:head+1]
        local = attention(query, key, value)
        # Rank dimensions by squared contribution to score perturbation, before
        # summing across D. This is an attribution heuristic, not causal proof.
        error = candidate.float() - key.float()
        visible = local[3]
        q_energy = query.float().square()
        k_error_energy = error[:, 0].square()
        contribution = torch.einsum('bhqd,bld,ql->d', q_energy, k_error_energy, visible.float())
        dimensions = contribution.argsort(descending=True)
        # Score perturbation per physical token in the valid contiguous capture.
        score_delta = query.float() @ error[:, 0].transpose(-1, -2)[:, None]
        score_delta *= query.shape[-1] ** -.5
        token_energy = (score_delta.square() * visible).sum((0, 1, 2))
        positions = token_energy.argsort(descending=True)
        row = {"kv_head": head, "query_heads": list(range(head*group, (head+1)*group)),
               "metrics": compare(query, key, candidate, value, local),
               "top_score_error_dimensions": dimensions[:top].tolist(),
               "top_score_error_positions": positions[:top].tolist(),
               "key_absmax_by_dimension": key.float().abs().amax((0, 1, 2)).tolist(),
               "rescues": []}
        for count in rescue_counts:
            for kind, ranking, axis in (("dimensions", dimensions, -1), ("positions", positions, -2)):
                actual_count = min(count, key.shape[axis])
                fixed = candidate.clone()
                index = ranking[:actual_count]
                if axis == -1:
                    fixed[..., index] = key[..., index]
                else:
                    fixed[:, :, index] = key[:, :, index]
                row["rescues"].append({"kind": kind, "count": actual_count,
                                       **compare(query, key, fixed, value, local)})
        heads.append(row)
    return {"layer": probe.get("layer"), "q_shape": list(q.shape), "kv_shape": list(k.shape),
            "scaling": scaling, "decoder_matches_torch": True,
            "identity_control": compare(q, k, k, v, baseline),
            "all_keys_quantized": compare(q, k, restored, v, baseline), "heads": heads}


@torch.no_grad()
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--scaling", nargs="+", choices=("token", "head"), default=["token", "head"])
    parser.add_argument("--rescue-counts", nargs="+", type=int, default=[1, 4, 16])
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("use a new output path")
    if min(args.rescue_counts) < 1 or args.top < 1:
        parser.error("rescue counts and top must be positive")
    probes = torch.load(args.input, weights_only=True, map_location="cpu")
    if not isinstance(probes, list) or not probes:
        parser.error("input must contain a nonempty capture list")
    report = {"status": "running", "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
              "scope": "local key-only sensitivity; fixed FP16 Q/V, FP32 reference attention; no speed or final-logit claims",
              "rescue_note": "oracle diagnostic restores dimensions/positions chosen using reference error; not a deployable quantization policy",
              "cases": []}
    with strict_accumulation() as settings:
        report["accumulation_settings"] = settings
        for probe in probes:
            values = {name: probe[name].to(args.device) for name in ("q", "k", "v")}
            values["layer"] = probe.get("layer")
            for scaling in args.scaling:
                case = diagnose(values, scaling, args.rescue_counts, args.top)
                report["cases"].append(case)
                print(f"layer={case['layer']} Q={case['q_shape'][2]} L={case['kv_shape'][2]} "
                      f"scaling={scaling} output_L2={case['all_keys_quantized']['output_relative_l2']:.6f}")
    report["ranking"] = sorted([
        {"case": index, "layer": case["layer"], "scaling": case["scaling"],
         "context": case["kv_shape"][2], "queries": case["q_shape"][2],
         "kv_head": head["kv_head"], **head["metrics"]}
        for index, case in enumerate(report["cases"]) for head in case["heads"]],
        key=lambda row: row["output_relative_l2"], reverse=True)
    report["status"] = "complete"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f"Saved key diagnostics to {args.output}")
