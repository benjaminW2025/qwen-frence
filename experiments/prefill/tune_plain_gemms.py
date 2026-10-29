#!/usr/bin/env python3
"""Tune the five Qwen FP16 GEMM shapes before applying epilogue fusion.

--plan writes the exact shape inventory without CUDA. Actual tuning requires
an H100 and GEMM_EPILOGUE_SWEEP=1 extension build. cuBLAS and every CUTLASS
candidate use graph replay, the same weights/layout/bias, and precision flags.
No engine configuration is changed. A selected configuration is remeasured
against cuBLAS with changed inputs in a separate confirmation round.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'custom_kernels'), str(ROOT/'baseline')]
import torch
import torch.nn.functional as F
from experiments.prefill.benchmark_epilogue_primitives import capture, compare

SHAPES = {'qkv': (2048,1536,True), 'output': (1536,1536,False),
          'gate-up': (17920,1536,False), 'down': (1536,8960,False),
          'lm-head': (151936,1536,False)}
DEFAULT_ROWS = (1,8,32,64,65,128,256,512,1024,1028,1408,1409,2048,4096,8192,8196)


def inventory(rows, head_rows, groups):
    return [{'group':group,'m':m,'n':SHAPES[group][0],'k':SHAPES[group][1],
             'bias':SHAPES[group][2]} for group in groups
            for m in sorted(set(head_rows if group == 'lm-head' else rows))]


def best_candidate(candidates):
    valid = [(row['training_median_ms'],row['config']) for row in candidates
             if row.get('status') == 'pass' and math.isfinite(row.get('training_median_ms',float('nan')))
             and row['training_median_ms'] > 0]
    return min(valid)[1] if valid else None


def measure(graphs, rounds, repetitions, flush, seed):
    samples = {name:[] for name in graphs}
    rng = random.Random(seed)
    start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    for round_index in range(rounds+1):
        order = list(graphs)
        rng.shuffle(order)
        for name in order:
            values = []
            for _ in range(repetitions):
                if flush is not None:
                    flush.zero_()  # outside timing; reduce hot-L2 bias from isolated GEMMs
                start.record()
                graphs[name].replay()
                end.record()
                end.synchronize()
                values.append(start.elapsed_time(end))
            if round_index:
                samples[name].append(statistics.median(values))
    if any(not math.isfinite(value) or value <= 0 for values in samples.values() for value in values):
        raise RuntimeError('invalid CUDA timing; refusing to select a configuration')
    return {name:{'median_ms':statistics.median(values),'round_medians_ms':values}
            for name,values in samples.items()}


def reference(x,w,bias):
    original = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        return F.linear(x.float(),w.float(),None if bias is None else bias.float()).half()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original


@torch.no_grad()
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--plan',action='store_true')
    parser.add_argument('--rows',type=int,nargs='+',default=list(DEFAULT_ROWS))
    parser.add_argument('--head-rows',type=int,nargs='+',default=[1,8,64])
    parser.add_argument('--groups',choices=tuple(SHAPES),nargs='+',default=list(SHAPES))
    parser.add_argument('--rounds',type=int,default=5)
    parser.add_argument('--repetitions',type=int,default=20)
    parser.add_argument('--confirmation-rounds',type=int,default=3)
    parser.add_argument('--cache-mode',choices=('streaming','warm'),default='streaming')
    parser.add_argument('--allow-default-build',action='store_true')
    parser.add_argument('--seed',type=int,default=2197)
    args = parser.parse_args(argv)
    if args.output_dir.exists() or min(*args.rows,*args.head_rows,args.rounds,args.repetitions,args.confirmation_rounds)<1:
        parser.error('use a fresh output directory and positive dimensions/repetitions')
    args.output_dir.mkdir(parents=True)
    plan = inventory(args.rows,args.head_rows,args.groups)
    report = {'status':'plan' if args.plan else 'running','dtype':'float16',
              'layout':'row-major activations; row-major stored weights, transposed without copying',
              'cache_mode':args.cache_mode,'scope':'plain GEMMs, not epilogues or full-model speed',
              'problems':plan,'cases':[], 'seed':args.seed,
              'rounds':args.rounds,'confirmation_rounds':args.confirmation_rounds,'repetitions':args.repetitions,
              'atol':.003,'rtol':.005,
              'sources': ['https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/profiler.md',
                          'https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/heuristics.md']}
    def save():
        (args.output_dir/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save()
    if args.plan:
        print(f'{len(plan)} GEMM problems; plan saved to {args.output_dir}/report.json')
        return
    try:
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9,0):
            raise RuntimeError('Hopper SM90 required; --plan is available on CPU')
        import fused_gemm as fused
        extension = fused._extension()
        if not extension.sweep_build and not args.allow_default_build:
            raise RuntimeError('rebuild with GEMM_EPILOGUE_SWEEP=1 for the full candidate set')
        configs = [dict(row) for row in fused.configs()]
        report.update(gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
                      capability=list(torch.cuda.get_device_capability()),configs=configs,
                      source_sha256=hashlib.sha256((ROOT/'custom_kernels/gemm_epilogue/fused_gemm.cu').read_bytes()).hexdigest(),
                      abi=extension.abi_version,
                      accumulation_settings={name:getattr(torch.backends.cuda.matmul,name)
                        for name in ('allow_fp16_reduced_precision_reduction','allow_fp16_accumulation','allow_tf32')
                        if hasattr(torch.backends.cuda.matmul,name)})
        props = torch.cuda.get_device_properties(0)
        report['multiprocessors'] = props.multi_processor_count
        report['total_memory'] = props.total_memory
        flush = torch.empty(256*1024*1024,dtype=torch.uint8,device='cuda') if args.cache_mode == 'streaming' else None
        report['cache_flush_bytes'] = 0 if flush is None else flush.numel()
        torch.manual_seed(args.seed)
        for problem in plan:
            report['active_problem'] = problem
            save()
            m,n,k = (problem[key] for key in ('m','n','k'))
            x = (torch.randn(m,k,device='cuda')*.1).half()
            w = (torch.randn(n,k,device='cuda')*.02).half()
            bias = (torch.randn(n,device='cuda')*.1).half() if problem['bias'] else None
            ref = reference(x,w,bias)
            graphs, outputs = {},{}
            graphs['cublas'],outputs['cublas'] = capture(lambda:F.linear(x,w,bias))
            compare(ref,outputs['cublas'],.003,.005)
            candidates = []
            for config in configs:
                index = config['index']
                row = {'config':index,'status':'fail'}
                try:
                    graph,output = capture(lambda:fused.gemm_plain(x,w,bias=bias,config=index))
                    row['numerics'] = compare(ref,output,.003,.005)
                    row['status'] = 'pass'
                    graphs[str(index)],outputs[str(index)] = graph,output
                except Exception as error:
                    row['error'] = f'{type(error).__name__}: {error}'
                    # A device fault can invalidate the context; never continue timing it.
                    if 'illegal memory' in str(error).lower() or 'device-side assert' in str(error).lower():
                        raise
                candidates.append(row)
            x.copy_(torch.randn_like(x)*.2)
            ref = reference(x,w,bias)
            for name,graph in list(graphs.items()):
                graph.replay()
                try:
                    compare(ref,outputs[name],.003,.005)
                except AssertionError as error:
                    if name == 'cublas':
                        raise
                    next(row for row in candidates if str(row['config']) == name).update(status='fail',error=str(error))
                    del graphs[name]
            timing = measure(graphs,args.rounds,args.repetitions,flush,args.seed+len(report['cases']))
            for row in candidates:
                if row['status'] == 'pass':
                    row['training_median_ms'] = timing[str(row['config'])]['median_ms']
            best = best_candidate(candidates)
            if best is None:
                raise RuntimeError(f'no correct CUTLASS configuration for {problem}')
            x.copy_(torch.randn_like(x)*.15)
            ref = reference(x,w,bias)
            for name in ('cublas',str(best)):
                graphs[name].replay()
                compare(ref,outputs[name],.003,.005)
            confirmation = measure({name:graphs[name] for name in ('cublas',str(best))},
                                   args.confirmation_rounds,args.repetitions,flush,args.seed+10000+len(report['cases']))
            speedup = confirmation['cublas']['median_ms']/confirmation[str(best)]['median_ms']
            result = {**problem,'status':'pass','candidates':candidates,'training':timing,
                      'best_cutlass_config':best,'confirmation':confirmation,'confirmed_speedup_vs_cublas':speedup,
                      'recommended_plain_backend':'cutlass' if speedup>=1.02 else 'cublas'}
            report['cases'].append(result)
            print(f"{problem['group']} M={m} N={n} K={k}: config={best}, confirmed {speedup:.3f}x",flush=True)
            save()
            del graphs,outputs,ref,graph,output,x,w,bias
            torch.cuda.empty_cache()
        report.pop('active_problem',None)
        report['status'] = 'pass'
    except Exception:
        report.update(status='fail',error=traceback.format_exc())
        save()
        raise
    save()


if __name__ == '__main__':
    main()
