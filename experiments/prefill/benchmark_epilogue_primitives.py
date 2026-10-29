#!/usr/bin/env python3
"""Qualify and graph-benchmark FP16 GEMM epilogues against existing operations.

No normalization folding, quantization, or attention change. Numerical checks
precede timing and include graph replay with changed inputs. QKV probes also
change positions, slots and live counts, and guard every untouched cache slot.
"""
from contextlib import nullcontext
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'custom_kernels'), str(ROOT/'baseline')]
import torch
import torch.nn.functional as F


def compare(reference, candidate, atol, rtol):
    finite = bool(torch.isfinite(reference).all() and torch.isfinite(candidate).all())
    if not finite:
        raise AssertionError('nonfinite epilogue output')
    error = candidate.float() - reference.float()
    result = {'max_abs': float(error.abs().max()),
              'relative_l2': float(error.norm()/reference.float().norm().clamp_min(1e-12))}
    torch.testing.assert_close(candidate, reference, atol=atol, rtol=rtol)
    return result


def capture(operation):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            operation()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = operation()
    torch.cuda.current_stream().wait_stream(stream)
    return graph, output


def paired_timing(graphs, rounds, repetitions):
    samples = {name: [] for name in graphs}
    names = list(graphs)
    for round_index in range(rounds+1):
        for name in (names if round_index % 2 == 0 else names[::-1]):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repetitions):
                graphs[name].replay()
            end.record()
            end.synchronize()
            if round_index:
                samples[name].append(start.elapsed_time(end)/repetitions)
    return {name: {'median_ms': statistics.median(values), 'samples_ms': values}
            for name, values in samples.items()}


@torch.no_grad()
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--gemm-tuning',type=Path)
    parser.add_argument('--rows', nargs='+', type=int, default=[1, 8, 64, 65, 128, 1408, 1409, 2048])
    parser.add_argument('--rounds', type=int, default=6)
    parser.add_argument('--repetitions', type=int, default=30)
    parser.add_argument('--atol', type=float, default=.003)
    parser.add_argument('--rtol', type=float, default=.005)
    args = parser.parse_args(argv)
    if args.output_dir.exists() or min(*args.rows, args.rounds, args.repetitions) < 1:
        parser.error('use a fresh output directory and positive dimensions/repetitions')
    args.output_dir.mkdir(parents=True)
    report = {'status': 'running', 'scope': 'FP16 primitive graph replay; no normalization folding',
              'atol': args.atol, 'rtol': args.rtol, 'cases': []}
    def save():
        (args.output_dir/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    try:
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
            raise RuntimeError('requires Hopper SM90; CPU cannot qualify these kernels')
        import fused_gemm as fused
        from kernel_dispatch import packed_qkv_rope_cache, swiglu
        from naive_forward import SWIGLU_FUSION_ROW_THRESHOLD
        from experiments.integration.ab_gemm_epilogues import strict_accumulation
        extension = fused._extension()
        report.update(gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
                      source_sha256=hashlib.sha256((ROOT/'custom_kernels/gemm_epilogue/fused_gemm.cu').read_bytes()).hexdigest(),
                      abi=extension.abi_version, configs=extension.configs())
        torch.manual_seed(971)
        with strict_accumulation() as settings, (fused.use_gemm_tuning(args.gemm_tuning) if args.gemm_tuning else nullcontext({})) as usage:
            report["tuning_usage"] = usage
            report["gemm_tuning"] = str(args.gemm_tuning) if args.gemm_tuning else None
            report['accumulation_settings'] = settings
            # Exact cancellation distinguishes preserving FP16 GEMM rounding
            # from adding the residual directly to an FP32 accumulator.
            x_edge = torch.zeros(3,1536,device='cuda',dtype=torch.float16)
            x_edge[:,0], x_edge[:,1] = 1., 2.**-12
            w_edge = torch.zeros(1536,1536,device='cuda',dtype=torch.float16)
            w_edge[:,0:2] = 1.
            residual_edge = torch.full((3,1536),-1.,device='cuda',dtype=torch.float16)
            expected_edge = F.linear(x_edge,w_edge)+residual_edge
            for config in (0,1):
                actual_edge,_ = fused.residual_gemm(x_edge,w_edge,residual_edge,partials=False,config=config)
                torch.testing.assert_close(actual_edge,expected_edge,atol=0,rtol=0)
                aliased = residual_edge.clone()
                actual_edge,_ = fused.residual_gemm(x_edge,w_edge,aliased,partials=False,out=aliased,config=config)
                torch.testing.assert_close(actual_edge,expected_edge,atol=0,rtol=0)
            report['residual_rounding_and_alias_gate'] = 'pass'
            for rows in args.rows:
                # Both compiled tile/scheduler defaults must pass at each row count.
                for config in ((-1,) if args.gemm_tuning else (0, 1)):
                    for op in ('residual-o', 'residual-down', 'gate-up', 'qkv'):
                        report['active_case'] = {'rows': rows, 'config': config, 'operation': op}
                        save()
                        k = 8960 if op == 'residual-down' else 1536
                        n = 17920 if op == 'gate-up' else 2048 if op == 'qkv' else 1536
                        x = (torch.randn(rows, k, device='cuda')*.1).half()
                        w = (torch.randn(n, k, device='cuda')*.02).half()
                        checks = []
                        guard = lambda: None
                        if op.startswith('residual'):
                            residual = torch.randn(rows, n, device='cuda').half()
                            baseline = lambda: F.linear(x, w) + residual
                            separate = lambda: fused.gemm_plain(x,w,config=config)+residual
                            candidate = lambda: fused.residual_gemm(x, w, residual, partials=False, config=config)[0]
                        elif op == 'gate-up':
                            gate_w, up_w = w.chunk(2)
                            interleaved = fused.interleave_gate_up(gate_w, up_w)
                            def baseline():
                                gate, up = F.linear(x, w).chunk(2, -1)
                                return swiglu(gate, up, block_size=512) if rows > SWIGLU_FUSION_ROW_THRESHOLD else F.silu(gate)*up
                            def separate():
                                gate, up = fused.gemm_plain(x,w,config=config).chunk(2,-1)
                                return swiglu(gate,up,block_size=512) if rows > SWIGLU_FUSION_ROW_THRESHOLD else F.silu(gate)*up
                            candidate = lambda: fused.gate_up_swiglu(x, interleaved, config=config, round_silu=rows <= SWIGLU_FUSION_ROW_THRESHOLD)
                        else:
                            bias = (torch.randn(2048, device='cuda')*.1).half()
                            positions = torch.arange(rows, device='cuda', dtype=torch.long) * 17
                            slots = torch.arange(rows, device='cuda', dtype=torch.long)*2+1
                            valid = torch.tensor(rows, dtype=torch.int32, device='cuda')
                            pages = (2*rows+31)//16
                            pools = {name: [torch.full((pages,16,2,128),7.,device='cuda',dtype=torch.float16)
                                            for _ in range(2)] for name in ('control','candidate','cutlass-separate')}
                            baseline = lambda: packed_qkv_rope_cache(F.linear(x,w,bias), positions, slots,
                                *pools['control'], valid_tokens=valid)
                            separate = lambda: packed_qkv_rope_cache(fused.gemm_plain(x,w,bias=bias,config=config),
                                positions,slots,*pools['cutlass-separate'],valid_tokens=valid)
                            candidate = lambda: fused.qkv_rope_cache(x,w,bias,None,positions=positions,slots=slots,
                                k_pool=pools['candidate'][0],v_pool=pools['candidate'][1],valid_tokens=valid,config=config)
                            def guard():
                                live = int(valid)
                                untouched = torch.ones(pages*16, dtype=torch.bool, device='cuda')
                                untouched[slots[:live]] = False
                                for name in ('candidate','cutlass-separate'):
                                    for a,b in zip(pools['control'],pools[name]):
                                        checks.append(compare(a.flatten(0,1)[slots[:live]],b.flatten(0,1)[slots[:live]],args.atol,args.rtol))
                                        if not torch.equal(b.flatten(0,1)[untouched], torch.full_like(b.flatten(0,1)[untouched],7.)):
                                            raise AssertionError('QKV wrote outside live slots')
                        graphs, outputs = {}, {}
                        for name, operation in [('control',baseline),('cutlass-separate',separate),('candidate',candidate)]:
                            graphs[name], outputs[name] = capture(operation)
                        for factor in (0., -1., 2.):
                            x.copy_((torch.randn_like(x)*factor*.1))
                            if op == 'qkv':
                                positions.add_(311)
                                slots.copy_(slots.flip(0))
                                valid.fill_(max(1,rows-1))
                                for tensors in pools.values():
                                    for tensor in tensors:
                                        tensor.fill_(7.)
                            for graph in graphs.values():
                                graph.replay()
                            for name in ('candidate','cutlass-separate'):
                                checks.append(compare(outputs['control'],outputs[name],args.atol,args.rtol))
                            guard()
                        timing = paired_timing(graphs, args.rounds, args.repetitions)
                        report['cases'].append({**report['active_case'], 'status':'pass', 'checks':checks,
                            'timing':timing, 'speedup':timing['control']['median_ms']/timing['candidate']['median_ms'],
                            'fusion_speedup_vs_cutlass_separate':timing['cutlass-separate']['median_ms']/timing['candidate']['median_ms']})
                        print(f'{op} rows={rows} config={config}: {report["cases"][-1]["speedup"]:.3f}x',flush=True)
                        save()
                        del graphs, outputs
            report.pop('active_case', None)
            report['status'] = 'pass'
    except Exception:
        report.update(status='fail', error=traceback.format_exc())
        save()
        raise
    save()


if __name__ == '__main__':
    main()
