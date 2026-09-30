#!/usr/bin/env python3
"""Gate/up projection + SwiGLU: fused CUTLASS epilogue vs cuBLAS + separate SwiGLU.

Qwen2.5-1.5B shapes (K = 1536, F = 8960), FP16, H100. Four arms per row count:
  cublas_gemm      packed [gate; up] projection only (the engine's cuBLAS GEMM)
  baseline         cublas_gemm + the faster of the Triton SwiGLU kernel and silu*mul
  fused            one CUTLASS GEMM with SwiGLU in the epilogue (interleaved weight)
  cutlass_plain    the same CUTLASS mainloop, plain output: mainloop parity with cuBLAS
Every arm is captured in a CUDA graph and timed by replay. Outputs are checked
against an FP32 reference before any timing is reported.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'custom_kernels'), str(ROOT / 'baseline')]

K, F = 1536, 8960
DEFAULT_ROWS = (2048, 1024, 512, 256, 128, 64, 8)
TOLERANCE = dict(atol=2e-3, rtol=2e-3)


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
    parser.add_argument('--rows', type=int, nargs='+', default=list(DEFAULT_ROWS))
    parser.add_argument('--repetitions', type=int, default=50)
    parser.add_argument('--seed', type=int, default=20260927)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f'refusing to overwrite {args.output_dir}')

    import torch
    import torch.nn.functional as functional
    from fused_gemm import gate_up_swiglu, gemm_plain, interleave_gate_up, split_interleaved
    from kernel_dispatch import swiglu
    props = torch.cuda.get_device_properties(0)
    if (props.major, props.minor) != (9, 0):
        raise RuntimeError('requires SM90 Hopper')

    generator = torch.Generator(device='cuda').manual_seed(args.seed)
    gate_w = (torch.randn(F, K, device='cuda', generator=generator) * .02).half()
    up_w = (torch.randn(F, K, device='cuda', generator=generator) * .02).half()
    packed = torch.cat((gate_w, up_w)).contiguous()          # the engine's gate_up_proj weight
    interleaved = interleave_gate_up(gate_w, up_w)

    rows_report = []
    for rows in args.rows:
        x = torch.randn(rows, K, device='cuda', generator=generator).half()
        gate_ref = x.float() @ gate_w.float().T
        up_ref = x.float() @ up_w.float().T
        reference = functional.silu(gate_ref) * up_ref

        def cublas_gemm():
            return functional.linear(x, packed)

        def baseline_triton():
            gate, up = functional.linear(x, packed).split((F, F), dim=-1)
            return swiglu(gate, up, block_size=512, num_warps=4, num_stages=2)

        def baseline_torch():
            gate, up = functional.linear(x, packed).split((F, F), dim=-1)
            return functional.silu(gate) * up

        arms = {}
        for name, operation in (('cublas_gemm', cublas_gemm), ('baseline_triton', baseline_triton),
                                ('baseline_torch', baseline_torch),
                                ('fused', lambda: gate_up_swiglu(x, interleaved)),
                                ('fused_config0', lambda: gate_up_swiglu(x, interleaved, config=0)),
                                ('cutlass_plain', lambda: gemm_plain(x, interleaved))):
            median, samples, output = graph_median_ms(torch, operation, args.repetitions)
            arms[name] = dict(median_ms=median, samples_ms=samples, output=output)

        # Correctness before timing is believed.
        errors = {}
        for name in ('baseline_triton', 'baseline_torch', 'fused', 'fused_config0'):
            out = arms[name]['output'].float()
            errors[name] = (out - reference).abs().max().item()
            torch.testing.assert_close(out, reference, **TOLERANCE)
        plain_gate, plain_up = split_interleaved(arms['cutlass_plain']['output'])
        cublas_gate, cublas_up = arms['cublas_gemm']['output'].split((F, F), dim=-1)
        torch.testing.assert_close(plain_gate, cublas_gate, **TOLERANCE)
        torch.testing.assert_close(plain_up, cublas_up, **TOLERANCE)
        errors['cutlass_plain_vs_cublas'] = max((plain_gate.float() - cublas_gate.float()).abs().max().item(),
                                                (plain_up.float() - cublas_up.float()).abs().max().item())

        best = min(('baseline_triton', 'baseline_torch'), key=lambda name: arms[name]['median_ms'])
        flops = 2 * rows * K * 2 * F
        row = dict(rows=rows, baseline_arm=best,
                   median_ms={name: arm['median_ms'] for name, arm in arms.items()},
                   samples_ms={name: arm['samples_ms'] for name, arm in arms.items()},
                   fused_speedup_vs_baseline=arms[best]['median_ms'] / arms['fused']['median_ms'],
                   cutlass_mainloop_vs_cublas=arms['cublas_gemm']['median_ms'] / arms['cutlass_plain']['median_ms'],
                   parity_config=0,
                   parity_matches_default_fused=rows > 64,
                   matched_config_fusion_speedup=arms[best]['median_ms'] / arms['fused_config0']['median_ms'],
                   default_vs_config0=arms['fused_config0']['median_ms'] / arms['fused']['median_ms'],
                   gemm_tflops={name: flops / arm['median_ms'] / 1e9
                                for name, arm in arms.items() if name in ('cublas_gemm', 'cutlass_plain', 'fused')},
                   max_abs_error_vs_fp32=errors)
        rows_report.append(row)
        print(f"M={rows:5d}: baseline ({best}) {arms[best]['median_ms']:.4f} ms, fused {arms['fused']['median_ms']:.4f} ms "
              f"-> {row['fused_speedup_vs_baseline']:.3f}x | mainloop cutlass/cublas "
              f"{row['cutlass_mainloop_vs_cublas']:.3f}x | err fused {errors['fused']:.2e} "
              f"baseline {errors[best]:.2e}", flush=True)
        del arms, x, gate_ref, up_ref, reference
        torch.cuda.empty_cache()

    args.output_dir.mkdir(parents=True)
    (args.output_dir / 'report.json').write_text(json.dumps(dict(
        device=props.name, torch=torch.__version__, cuda=torch.version.cuda, K=K, F=F,
        seed=args.seed, repetitions=args.repetitions, tolerance=TOLERANCE,
        note='mainloop parity uses config 0 and is paired with fused_config0. At M<=64 the default '
             'fused kernel uses config 1 (Stream-K); its mainloop parity is not measured by this binary. '
             'default_vs_config0 isolates the observed schedule/tile difference; speedup > 1 is faster.',
        rows=rows_report), indent=2) + '\n')


if __name__ == '__main__':
    main()
