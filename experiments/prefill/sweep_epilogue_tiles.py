#!/usr/bin/env python3
"""Sweep compiled tile/cluster/scheduler candidates for the fused-epilogue GEMMs.

Requires a GEMM_EPILOGUE_SWEEP=1 build of custom_kernels/gemm_epilogue (every
candidate compiled in). For each kernel at its Qwen2.5-1.5B shape and each row
count, every candidate is checked against an FP32 reference, then timed by CUDA
graph replay. The report names the fastest correct candidate per (kernel, rows)
and its speedup over the current default. Nothing is changed in the dispatch:
bake the winners into resolve_config only after the layer benchmark agrees.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'custom_kernels')]

HIDDEN, FFN, QKV, EPS, THETA = 1536, 8960, 2048, 1e-6, 1_000_000.0
DEFAULT_ROWS = (8, 32, 64, 128, 256, 512, 1024, 2048)
TOLERANCE = dict(atol=2e-2, rtol=2e-2)


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
    return statistics.median(samples), output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--rows', type=int, nargs='+', default=list(DEFAULT_ROWS))
    parser.add_argument('--kernels', nargs='+', default=['residual_o', 'residual_down', 'swiglu', 'qkv'],
                        choices=['residual_o', 'residual_down', 'swiglu', 'qkv'])
    parser.add_argument('--repetitions', type=int, default=30)
    parser.add_argument('--seed', type=int, default=20260927)
    parser.add_argument('--allow-default-build', action='store_true',
                        help='sweep only the two default candidates (no GEMM_EPILOGUE_SWEEP build)')
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f'refusing to overwrite {args.output_dir}')

    import torch
    import torch.nn.functional as functional
    import fused_gemm as fused
    extension = fused._extension()
    if not extension.sweep_build and not args.allow_default_build:
        parser.error('rebuild with GEMM_EPILOGUE_SWEEP=1 to compile every candidate')
    candidates = [dict(row) for row in fused.configs()]
    print('candidates:', ', '.join(f"{c['index']}: {c['tile'][1]}N {c['cluster'][0]}x{c['cluster'][1]} "
                                   f"{c['scheduler']}" for c in candidates), flush=True)

    g = torch.Generator(device='cuda').manual_seed(args.seed)
    def randn(*shape, scale=1.0):
        return torch.randn(*shape, device='cuda', generator=g) * scale
    w_o = randn(HIDDEN, HIDDEN, scale=.02).half()
    w_down = randn(HIDDEN, FFN, scale=.02).half()
    w_gate, w_up = randn(FFN, HIDDEN, scale=.02).half(), randn(FFN, HIDDEN, scale=.02).half()
    gamma = (torch.rand(HIDDEN, device='cuda', generator=g) + .5).half()
    w_gate_up = fused.prepare_gate_up(w_gate, w_up, gamma)
    w_qkv, b_qkv = randn(QKV, HIDDEN, scale=.02).half(), randn(QKV, scale=.1).half()
    w_qkv_fused = fused.prepare_qkv(w_qkv, gamma)

    def rmsnorm(x):
        return gamma.float() * x / torch.sqrt(x.square().mean(-1, keepdim=True) + EPS)

    def rope(heads, positions):
        d = torch.arange(64, device='cuda', dtype=torch.float32)
        angle = positions.float()[:, None, None] * torch.exp(d * (-math.log(THETA) / 64))
        a, b = heads[..., :64], heads[..., 64:]
        return torch.cat((a * angle.cos() - b * angle.sin(), a * angle.sin() + b * angle.cos()), -1)

    results = []
    for rows in args.rows:
        x = randn(rows, HIDDEN).half()
        act = randn(rows, FFN).half()
        residual = randn(rows, HIDDEN).half()
        partials = fused.row_square_partials(x)
        pages = rows // 16 + 8
        slots = torch.randperm(pages * 16, device='cuda', generator=g)[:rows].long()
        positions = (torch.arange(rows, device='cuda') % 1024 + 37).long()
        valid = torch.tensor(rows, device='cuda', dtype=torch.int32)
        normalized = rmsnorm(x.float())
        references = {
            'residual_o': residual.float() + x.float() @ w_o.float().T,
            'residual_down': residual.float() + act.float() @ w_down.float().T,
            'swiglu': functional.silu(normalized @ w_gate.float().T) * (normalized @ w_up.float().T),
        }
        heads = (normalized @ w_qkv.float().T + b_qkv.float()).reshape(rows, 16, 128)
        references['qkv'] = (rope(heads[:, :12], positions), rope(heads[:, 12:14], positions), heads[:, 14:])

        for kernel in args.kernels:
            timings = {}
            for candidate in candidates:
                index = candidate['index']
                pools = [torch.zeros(pages, 16, 2, 128, device='cuda', dtype=torch.float16) for _ in range(2)]
                operation = {
                    'residual_o': lambda: fused.residual_gemm(x, w_o, residual, config=index)[0],
                    'residual_down': lambda: fused.residual_gemm(act, w_down, residual, config=index)[0],
                    'swiglu': lambda: fused.gate_up_swiglu(x, w_gate_up, partials, hidden=HIDDEN, eps=EPS,
                                                           config=index),
                    'qkv': lambda: fused.qkv_rope_cache(
                        x, w_qkv_fused, b_qkv, partials, positions=positions, slots=slots, k_pool=pools[0],
                        v_pool=pools[1], valid_tokens=valid, hidden=HIDDEN, eps=EPS, theta=THETA, config=index),
                }[kernel]
                try:
                    median, output = graph_median_ms(torch, operation, args.repetitions)
                    if kernel == 'qkv':
                        q_ref, k_ref, v_ref = references['qkv']
                        torch.testing.assert_close(output.float(), q_ref, **TOLERANCE)
                        torch.testing.assert_close(pools[0].view(-1, 2, 128)[slots].float(), k_ref, **TOLERANCE)
                        torch.testing.assert_close(pools[1].view(-1, 2, 128)[slots].float(), v_ref, **TOLERANCE)
                    else:
                        torch.testing.assert_close(output.float(), references[kernel], **TOLERANCE)
                    timings[index] = dict(median_ms=median, correct=True)
                except Exception as error:  # a candidate that fails to run or to match is recorded, not fatal
                    timings[index] = dict(median_ms=None, correct=False, error=f'{type(error).__name__}: {error}'[:300])
                del pools
            correct = {i: t['median_ms'] for i, t in timings.items() if t['correct']}
            default = 1 if rows <= extension.decode_rows else 0
            best = min(correct, key=correct.get) if correct else None
            row = dict(kernel=kernel, rows=rows, default_config=default, best_config=best, candidates=timings,
                       best_over_default=(correct[default] / correct[best]
                                          if best is not None and default in correct else None))
            results.append(row)
            speed = f"{row['best_over_default']:.3f}x" if row['best_over_default'] else 'n/a'
            print(f"{kernel:13} M={rows:5d}: best config {best} vs default {default} -> {speed}; "
                  f"failed: {[i for i, t in timings.items() if not t['correct']]}", flush=True)
        torch.cuda.empty_cache()

    args.output_dir.mkdir(parents=True)
    selection = {kernel: {row['rows']: row['best_config'] for row in results if row['kernel'] == kernel}
                 for kernel in args.kernels}
    (args.output_dir / 'report.json').write_text(json.dumps(dict(
        device=torch.cuda.get_device_properties(0).name, torch=torch.__version__, seed=args.seed,
        repetitions=args.repetitions, candidates=candidates, tolerance=TOLERANCE, results=results,
        selection=selection, production_enabled=False,
        note='best_over_default > 1 means a non-default candidate is faster; bake into resolve_config '
             'only after benchmark_layer_epilogues confirms at the layer level'), indent=2) + '\n')


if __name__ == '__main__':
    main()
