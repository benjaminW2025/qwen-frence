#!/usr/bin/env python3
"""One decoder layer's non-attention work: fused-epilogue GEMMs vs the engine's current kernels.

Qwen2.5-1.5B shapes (hidden 1536, FFN 8960, packed QKV 2048), FP16, H100, random
weights. The layer segment between two attentions:

  baseline  o_proj (cuBLAS) -> residual_add_rms_norm -> gate_up (cuBLAS) -> SwiGLU
            (engine threshold rule) -> down (cuBLAS) -> residual_add_rms_norm ->
            QKV + bias (cuBLAS) -> packed_qkv_rope_cache            (8 kernels)
  fused     residual_gemm -> gate_up_swiglu (norm folded) -> residual_gemm ->
            qkv_rope_cache (norm folded)                             (4 kernels)

Both are checked against an FP32 chain with no intermediate rounding; the fused
path must be no less accurate than the baseline (within 1.5x + 1e-3) on every
output, padded rows must produce zero Q and leave the KV cache untouched. Only
then are the CUDA-graph replay timings reported.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'custom_kernels'), str(ROOT / 'baseline')]

HIDDEN, FFN, QKV, EPS, THETA, LAYERS = 1536, 8960, 2048, 1e-6, 1_000_000.0, 28
SWIGLU_FUSION_ROW_THRESHOLD = 1408   # baseline/naive_forward.py
SENTINEL = 7.0                       # marks KV-cache slots nothing may write


def graph_median_ms(torch, operation, repetitions):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            operation()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = operation()
    torch.cuda.current_stream().wait_stream(stream)
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repetitions):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end))
    return statistics.median(samples), samples, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--rows', type=int, nargs='+', default=[2048, 1024, 512, 256, 128, 64, 32, 8])
    parser.add_argument('--repetitions', type=int, default=30)
    parser.add_argument('--seed', type=int, default=20260927)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f'refusing to overwrite {args.output_dir}')

    import torch
    import torch.nn.functional as functional
    import fused_gemm as fused
    from kernel_dispatch import packed_qkv_rope_cache, residual_add_rms_norm, swiglu
    props = torch.cuda.get_device_properties(0)
    if (props.major, props.minor) != (9, 0):
        raise RuntimeError('requires SM90 Hopper')

    g = torch.Generator(device='cuda').manual_seed(args.seed)
    def randn(*shape, scale=1.0):
        return torch.randn(*shape, device='cuda', generator=g) * scale
    w_o = randn(HIDDEN, HIDDEN, scale=.02).half()
    w_gate, w_up = randn(FFN, HIDDEN, scale=.02).half(), randn(FFN, HIDDEN, scale=.02).half()
    w_down = randn(HIDDEN, FFN, scale=.02).half()
    w_qkv, b_qkv = randn(QKV, HIDDEN, scale=.02).half(), randn(QKV, scale=.1).half()
    gamma_post = (torch.rand(HIDDEN, device='cuda', generator=g) + .5).half()
    gamma_in = (torch.rand(HIDDEN, device='cuda', generator=g) + .5).half()
    w_gate_up = torch.cat((w_gate, w_up)).contiguous()                    # engine's packed layout
    w_gate_up_fused = fused.prepare_gate_up(w_gate, w_up, gamma_post)     # folded + interleaved
    w_qkv_fused = fused.prepare_qkv(w_qkv, gamma_in)

    def rmsnorm(x, gamma):
        return gamma.float() * x / torch.sqrt(x.square().mean(-1, keepdim=True) + EPS)

    def rope(heads, positions):
        d = torch.arange(64, device='cuda', dtype=torch.float32)
        angle = positions.float()[:, None, None] * torch.exp(d * (-math.log(THETA) / 64))
        a, b = heads[..., :64], heads[..., 64:]
        return torch.cat((a * angle.cos() - b * angle.sin(), a * angle.sin() + b * angle.cos()), -1)

    report = []
    for rows in args.rows:
        x0, attn = randn(rows, HIDDEN).half(), randn(rows, HIDDEN).half()
        pages = rows // 16 + 8
        slots = torch.randperm(pages * 16, device='cuda', generator=g)[:rows].long()
        positions = (torch.arange(rows, device='cuda') % 1024 + 37).long()
        live = max(rows - 3, 1)                     # exercise padded bucket rows
        valid = torch.tensor(live, device='cuda', dtype=torch.int32)
        pools = {arm: [torch.full((pages, 16, 2, 128), SENTINEL, device='cuda', dtype=torch.float16)
                       for _ in range(2)] for arm in ('baseline', 'fused')}

        def baseline():
            k_pool, v_pool = pools['baseline']
            x1, h1 = residual_add_rms_norm(x0, functional.linear(attn, w_o), gamma_post, EPS)
            gate, up = functional.linear(h1, w_gate_up).split((FFN, FFN), dim=-1)
            act = (swiglu(gate, up, block_size=512, num_warps=4, num_stages=2)
                   if rows > SWIGLU_FUSION_ROW_THRESHOLD else functional.silu(gate) * up)
            x2, h2 = residual_add_rms_norm(x1, functional.linear(act, w_down), gamma_in, EPS)
            q = packed_qkv_rope_cache(functional.linear(h2, w_qkv, b_qkv), positions, slots,
                                      k_pool, v_pool, base=THETA, valid_tokens=valid)
            return x1, x2, q

        def fused_chain():
            k_pool, v_pool = pools['fused']
            x1, p1 = fused.residual_gemm(attn, w_o, x0)
            act = fused.gate_up_swiglu(x1, w_gate_up_fused, p1, hidden=HIDDEN, eps=EPS)
            x2, p2 = fused.residual_gemm(act, w_down, x1)
            q = fused.qkv_rope_cache(x2, w_qkv_fused, b_qkv, p2, positions=positions, slots=slots,
                                     k_pool=k_pool, v_pool=v_pool, valid_tokens=valid,
                                     hidden=HIDDEN, eps=EPS, theta=THETA)
            return x1, x2, q

        timings, outputs = {}, {}
        for arm, operation in (('baseline', baseline), ('fused', fused_chain)):
            median, samples, output = graph_median_ms(torch, operation, args.repetitions)
            timings[arm] = dict(median_ms=median, samples_ms=samples)
            outputs[arm] = output

        # FP32 reference chain, no intermediate rounding.
        x1r = x0.float() + attn.float() @ w_o.float().T
        h1r = rmsnorm(x1r, gamma_post)
        actr = functional.silu(h1r @ w_gate.float().T) * (h1r @ w_up.float().T)
        x2r = x1r + actr @ w_down.float().T
        heads = (rmsnorm(x2r, gamma_in) @ w_qkv.float().T + b_qkv.float()).reshape(rows, 16, 128)
        reference = dict(x1=x1r[:live], x2=x2r[:live], q=rope(heads[:live, :12], positions[:live]),
                         k=rope(heads[:live, 12:14], positions[:live]), v=heads[:live, 14:])

        errors, checks = {}, {}
        live_slots = slots[:live]
        for arm in ('baseline', 'fused'):
            x1, x2, q = outputs[arm]
            k_pool, v_pool = pools[arm]
            got = dict(x1=x1[:live], x2=x2[:live], q=q[:live],
                       k=k_pool.view(-1, 2, 128)[live_slots], v=v_pool.view(-1, 2, 128)[live_slots])
            errors[arm] = {name: (got[name].float() - reference[name]).abs().max().item() for name in reference}
            untouched = torch.ones(pages * 16, dtype=torch.bool, device='cuda')
            untouched[live_slots] = False
            checks[arm] = dict(
                padded_q_zero=bool((q[live:] == 0).all()),
                cache_outside_live_rows_untouched=bool(
                    (k_pool.view(-1, 2, 128)[untouched] == SENTINEL).all() and
                    (v_pool.view(-1, 2, 128)[untouched] == SENTINEL).all()))
        accurate = {name: errors['fused'][name] <= 1.5 * errors['baseline'][name] + 1e-3 for name in reference}
        correct = all(accurate.values()) and all(all(c.values()) for c in checks.values())
        speedup = timings['baseline']['median_ms'] / timings['fused']['median_ms']
        saved = (timings['baseline']['median_ms'] - timings['fused']['median_ms']) * LAYERS
        report.append(dict(rows=rows, live_rows=live, correct=correct, accuracy_not_worse=accurate,
                           checks=checks, max_abs_error_vs_fp32=errors, timings=timings,
                           speedup=speedup, estimated_ms_saved_per_step=saved))
        print(f"M={rows:5d}: baseline {timings['baseline']['median_ms']:.4f} ms, fused "
              f"{timings['fused']['median_ms']:.4f} ms -> {speedup:.3f}x "
              f"(~{saved:.2f} ms/step over {LAYERS} layers) | correct={correct}", flush=True)
        if not correct:
            print(f"  errors {errors} checks {checks}", flush=True)
        del outputs, pools
        torch.cuda.empty_cache()

    args.output_dir.mkdir(parents=True)
    (args.output_dir / 'report.json').write_text(json.dumps(dict(
        device=props.name, torch=torch.__version__, cuda=torch.version.cuda, seed=args.seed,
        repetitions=args.repetitions, rows=report,
        note='speedup > 1 means the fused chain is faster; timings are only meaningful where correct'),
        indent=2) + '\n')
    if not all(row['correct'] for row in report):
        raise SystemExit('fused epilogues failed a correctness check; timings retained but not qualified')


if __name__ == '__main__':
    main()
