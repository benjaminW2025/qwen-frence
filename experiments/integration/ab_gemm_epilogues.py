#!/usr/bin/env python3
"""Model-level A/B of the fused GEMM epilogues: the integration gate before the eight cells.

Control is the accepted engine path (cuBLAS + fused residual/RMSNorm + SwiGLU
fusion, native decode QKV postprocess). Candidate is the same model with the
CUTLASS fused-epilogue GEMMs (custom_kernels/fused_gemm.py). Both arms share
weights and attention; each owns its KV pool.

Prefill: one packed forward per case through each arm's piecewise graphs, then
logits and every layer's live K/V are compared, then alternating timed replays.
Decode: both pools are filled from the control prefill, then both captured
decode graphs run teacher-forced steps (the control's greedy token feeds both),
comparing logits at every step; graph replays are timed alternately.

Numerics are expected to differ slightly (norm and activations applied to FP32
accumulators, gamma folded into weights). The gate is that every greedy-token
disagreement is a near tie in the control's own logits. Run in the interpreter
that owns the attention (the vLLM env for --attention fa3) and the built
extension.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import statistics
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for directory in (HERE, ROOT / "baseline", ROOT / "benchmarks", ROOT / "engine/graph",
                  ROOT / "engine/model_runner", ROOT / "engine/kvcache", ROOT / "engine/cpp/build"):
    sys.path.insert(0, str(directory))

POLICIES = {"fa3": ("fa3_varlen", "fa3"), "project": ("flash_varlen", "flash")}
PRIVATE_POOL_LIMIT = 8192


def parse_case(text):
    sequences, length = (int(part) for part in text.lower().split("x"))
    if sequences < 1 or length < 1:
        raise argparse.ArgumentTypeError(f"bad case {text!r}; use SEQUENCESxLENGTH")
    return sequences, length


def compare_logits(control, candidate, tie_margin):
    """Greedy agreement, and for each disagreement the control's own margin to the candidate's pick."""
    import torch
    control, candidate = control.float(), candidate.float()
    control_top, candidate_top = control.argmax(-1), candidate.argmax(-1)
    differ = (control_top != candidate_top).nonzero().flatten().tolist()
    margins = [float(control[row, control_top[row]] - control[row, candidate_top[row]])
               for row in differ]
    return {"rows": control.shape[0], "greedy_agree": control.shape[0] - len(differ),
            "max_abs_logit_diff": float((control - candidate).abs().max()),
            "mean_abs_logit_diff": float((control - candidate).abs().mean()),
            "disagreement_margins": margins,
            "non_tie_disagreements": sum(margin > tie_margin for margin in margins),
            "_next": control_top}


def merge(rows):
    rows = [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]
    margins = [m for row in rows for m in row["disagreement_margins"]]
    return {"rows": sum(r["rows"] for r in rows), "greedy_agree": sum(r["greedy_agree"] for r in rows),
            "max_abs_logit_diff": max(r["max_abs_logit_diff"] for r in rows),
            "mean_abs_logit_diff": statistics.fmean(r["mean_abs_logit_diff"] for r in rows),
            "max_disagreement_margin": max(margins, default=0.0),
            "non_tie_disagreements": sum(r["non_tie_disagreements"] for r in rows)}


def event_ms(torch, run, repetitions):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repetitions):
        run()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repetitions


def alternate(torch, arms, repetitions, rounds):
    """Median per-call ms per arm over alternating rounds (A B B A ...), after one warm round."""
    samples = {name: [] for name in arms}
    order = list(arms)
    for index in range(rounds + 1):
        for name in (order if index % 2 == 0 else order[::-1]):
            value = event_ms(torch, arms[name], repetitions)
            if index:
                samples[name].append(value)
    return {name: statistics.median(values) for name, values in samples.items()}, samples


class Layout:
    """Sequences on disjoint pages; page 0 is left for graph-capture scratch writes."""

    def __init__(self, torch, device, sequences, capacity):
        self.pages = math.ceil(capacity / 16)
        self.table = (1 + torch.arange(sequences * self.pages, device=device, dtype=torch.int32)
                      ).view(sequences, self.pages)

    def slots(self, torch, rows, positions):
        pages = self.table[rows, positions // 16].to(torch.long)
        return pages * 16 + positions % 16


def prefill_inputs(torch, device, prompts, layout, rows):
    lengths = [prompts[row].numel() for row in rows]
    ids = torch.cat([prompts[row] for row in rows]).to(device)
    positions = torch.cat([torch.arange(n, device=device) for n in lengths])
    seq = torch.cat([torch.full((n,), row, device=device, dtype=torch.long) for row, n in zip(rows, lengths)])
    slots = layout.slots(torch, seq, positions)
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device=device, dtype=torch.int32)
    context = torch.tensor(lengths, device=device, dtype=torch.int32)
    table = layout.table[torch.tensor(rows, device=device)]
    return ids, positions, slots, cu, context, table, max(lengths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attention", choices=tuple(POLICIES), default="fa3")
    parser.add_argument("--prefill-cases", type=parse_case, nargs="+",
                        default=[(8, 256), (1, 2048), (4, 2048), (8, 2048)],
                        help="SEQUENCESxLENGTH packed prefill forwards; 2048, 8192 and 16384 "
                             "tokens are the eight-cell buckets at the candidate budgets")
    parser.add_argument("--decode-cases", type=parse_case, nargs="+",
                        default=[(8, 256), (8, 2048), (64, 256), (64, 2048)],
                        help="BATCHxCONTEXT decode graphs (the eight cells' batch and prompt sizes)")
    parser.add_argument("--decode-steps", type=int, default=32)
    parser.add_argument("--tie-margin", type=float, default=0.25,
                        help="a greedy disagreement passes only if the control's own logit gap "
                             "between the two tokens is at most this")
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f"refusing to overwrite existing results: {args.output_dir}")
    if any(length > 2048 for _, length in args.decode_cases):
        parser.error("decode contexts are prefilled through the 2048-token bucket; keep them <= 2048")

    from benchmark_latest_vs_vllm import resolve_model_source
    from model_setup import check_startup, load_model_only
    import torch
    torch.set_grad_enabled(False)
    from model_adapter import allocate_pool
    from piecewise_prefill import PiecewisePrefill
    from paged_graph_decoder import CUDAGraphDecoder
    from kernel_dispatch import _load

    setup = check_startup(args.device)
    fused = _load("fused_gemm")
    extension = fused._extension()  # fail before loading weights if unbuilt or stale
    engine, _, _ = load_model_only(resolve_model_source(args), args.device, "float16",
                                   hub_transfer=setup["hub_transfer"])
    model, cfg, device = engine.model, engine.cfg, engine.device
    prepare_started = time.perf_counter()
    fused.prepare_model(model)
    torch.cuda.synchronize()
    prepare_seconds = time.perf_counter() - prepare_started
    prefill_policy, decode_policy = POLICIES[args.attention]
    generator = torch.Generator().manual_seed(args.seed)

    capacity = max([n * length for n, length in args.prefill_cases]
                   + [b * math.ceil((length + args.decode_steps) / 16) * 16
                      for b, length in args.decode_cases])
    blocks = capacity // 16 + max(n for n, _ in args.prefill_cases + args.decode_cases) + 1
    pools = {"control": allocate_pool(cfg, blocks, device), "candidate": allocate_pool(cfg, blocks, device)}
    report = {"status": "running", "attention": args.attention, "model": args.model,
              "gpu": torch.cuda.get_device_name(device), "tie_margin": args.tie_margin,
              "fused_configs": extension.configs(), "prepare_model_seconds": prepare_seconds,
              "fused_weight_bytes": sum(t.numel() * t.element_size() for layer in model.fused_gemm_layers
                                        for t in (layer.qkv, layer.gate_up)),
              "prefill": [], "decode": []}

    def prefill_arm(name, bucket):
        return PiecewisePrefill(model, pools[name], max_capture_tokens=bucket, max_shapes=1,
                                token_buckets=[bucket], enable_residual_rmsnorm=True,
                                enable_swiglu_fusion=True,
                                share_graph_pool=bucket > PRIVATE_POOL_LIMIT,
                                enable_fused_gemm_epilogues=name == "candidate")

    for sequences, length in args.prefill_cases:
        tokens = sequences * length
        print(f"prefill {sequences}x{length} ({tokens} tokens)", flush=True)
        prompts = [torch.randint(0, cfg.vocab, (length,), generator=generator) for _ in range(sequences)]
        layout = Layout(torch, device, sequences, length)
        inputs = prefill_inputs(torch, device, prompts, layout, list(range(sequences)))
        arms, logits = {}, {}
        for name in ("control", "candidate"):
            for tensor in pools[name].k_pool + pools[name].v_pool:
                tensor.fill_(float("nan"))
            arms[name] = prefill_arm(name, tokens)
            logits[name] = arms[name].forward(*inputs, mixed_attention_policy=prefill_policy)
            if logits[name] is None or arms[name].eager_calls:
                raise AssertionError(f"{name} prefill missed its captured bucket")
        live = inputs[2]
        kv = max(float((a.view(-1, 2, 128)[live].float() - b.view(-1, 2, 128)[live].float()).abs().max())
                 for a, b in zip(pools["control"].k_pool + pools["control"].v_pool,
                                 pools["candidate"].k_pool + pools["candidate"].v_pool))
        kv_finite = all(bool(torch.isfinite(t.view(-1, 2, 128)[live]).all())
                        for t in pools["candidate"].k_pool + pools["candidate"].v_pool)
        numerics = merge([compare_logits(logits["control"], logits["candidate"], args.tie_margin)])
        timed, samples = alternate(torch, {name: (lambda arm=arm: arm.forward(
            *inputs, mixed_attention_policy=prefill_policy)) for name, arm in arms.items()},
            repetitions=5, rounds=args.rounds)
        row = {"sequences": sequences, "length": length, "tokens": tokens,
               "graph_pool": "shared" if tokens > PRIVATE_POOL_LIMIT else "private",
               "numerics": numerics, "max_abs_kv_diff": kv, "candidate_kv_finite": kv_finite,
               "median_forward_ms": timed, "samples_ms": samples,
               "speedup": timed["control"] / timed["candidate"],
               "capture_reserved_bytes": {name: arm.capture_memory.get(tokens) for name, arm in arms.items()}}
        report["prefill"].append(row)
        print(f"  control {timed['control']:.3f} ms, fused {timed['candidate']:.3f} ms "
              f"({row['speedup']:.3f}x); greedy {numerics['greedy_agree']}/{numerics['rows']}, "
              f"max |dlogit| {numerics['max_abs_logit_diff']:.3f}, max |dKV| {kv:.4f}", flush=True)
        del arms, logits
        gc.collect()
        torch.cuda.empty_cache()

    fill = PiecewisePrefill(model, pools["control"], max_capture_tokens=2048, max_shapes=1,
                            token_buckets=[2048], enable_residual_rmsnorm=True,
                            enable_swiglu_fusion=True)
    for batch, length in args.decode_cases:
        print(f"decode B={batch} context={length}", flush=True)
        capacity = length + args.decode_steps
        layout = Layout(torch, device, batch, capacity)
        decoders = {name: CUDAGraphDecoder(
            model, pools[name], batch_size=batch, max_blocks=layout.pages, device=device,
            dtype=torch.float16, decode_attention_policy=decode_policy,
            max_decode_context_length=capacity, enable_residual_rmsnorm=True,
            enable_native_decode_qkv_postprocess=True,
            enable_fused_gemm_epilogues=name == "candidate").capture()
            for name in ("control", "candidate")}
        prompts = [torch.randint(0, cfg.vocab, (length,), generator=generator) for _ in range(batch)]
        group = max(1, 2048 // length)
        last = []
        for first in range(0, batch, group):
            rows = list(range(first, min(batch, first + group)))
            last.append(fill.forward(*prefill_inputs(torch, device, prompts, layout, rows),
                                     mixed_attention_policy=prefill_policy))
        for a, b in zip(pools["candidate"].k_pool + pools["candidate"].v_pool,
                        pools["control"].k_pool + pools["control"].v_pool):
            a.copy_(b)  # isolate decode numerics: both arms start from one history
        tokens = torch.cat(last).argmax(-1)
        rows = torch.arange(batch, device=device)
        steps = []
        for step in range(args.decode_steps):
            position = torch.full((batch,), length + step, device=device, dtype=torch.int32)
            slots = layout.slots(torch, rows, position.to(torch.long))
            step_logits = {name: decoder.decode(tokens.view(batch, 1), position, position + 1,
                                                layout.table, slots).squeeze(1).clone()
                           for name, decoder in decoders.items()}
            steps.append(compare_logits(step_logits["control"], step_logits["candidate"], args.tie_margin))
            tokens = steps[-1]["_next"]  # teacher-forced on the control's greedy token
        numerics = merge(steps)
        timed, samples = alternate(torch, {name: decoder.graph.replay for name, decoder in decoders.items()},
                                   repetitions=50, rounds=args.rounds)
        row = {"batch": batch, "context": length, "steps": args.decode_steps, "numerics": numerics,
               "first_disagreement_step": next((i for i, s in enumerate(steps)
                                                if s["greedy_agree"] < s["rows"]), None),
               "median_graph_replay_ms": timed, "samples_ms": samples,
               "speedup": timed["control"] / timed["candidate"]}
        report["decode"].append(row)
        print(f"  control {timed['control']:.4f} ms, fused {timed['candidate']:.4f} ms "
              f"({row['speedup']:.3f}x); greedy {numerics['greedy_agree']}/{numerics['rows']}, "
              f"max |dlogit| {numerics['max_abs_logit_diff']:.3f}", flush=True)
        del decoders
        gc.collect()
        torch.cuda.empty_cache()

    failures = [f"prefill {r['sequences']}x{r['length']}" for r in report["prefill"]
                if r["numerics"]["non_tie_disagreements"] or not r["candidate_kv_finite"]]
    failures += [f"decode B={r['batch']} L={r['context']}" for r in report["decode"]
                 if r["numerics"]["non_tie_disagreements"]]
    report["status"] = "pass" if not failures else "fail"
    report["failures"] = failures
    report["note"] = ("speedups are per forward (prefill: graphs plus eager attention, host included) "
                      "or per decode graph replay; numerics pass when every greedy disagreement is a "
                      "near tie in the control's logits")
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"status={report['status']}" + (f"; non-tie disagreements in {failures}" if failures else "")
          + f"; report={args.output_dir / 'report.json'}")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
