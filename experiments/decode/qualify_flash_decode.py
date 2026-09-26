#!/usr/bin/env python3
"""Measure the independent kernel against current-vLLM FA3 in isolated environments.

Exactly shared tensor fixtures, warm/cold replay timing, independent FP32 oracle,
and an explicit parity gate. This never labels a candidate FA3-equivalent merely
because it passed a tensor check. Full-model qualification is a separate stage.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'custom_kernels'), str(ROOT / 'benchmarks'), str(Path(__file__).parent)]
from reference_version import require_vllm_version

SHAPES = [(kind, b, c) for kind in ('decode', 'mixed') for b in (8, 64) for c in (256, 2048, 4096)]
SHAPES += [('prefill', b, q) for b, queries in ((8, (256, 1024)), (64, (32, 128))) for q in queries]
SHAPES += [('resumed_prefill', b, c) for b in (8, 64) for c in (256, 2048, 4096)]


def query_lengths_for(kind, batch, context):
    if kind == 'decode':
        return [1] * batch
    if kind == 'prefill':
        return [context] * batch
    if kind == 'resumed_prefill':
        # Leave a nonempty prefix, bounded by the real 8192-token budget.
        return [min(context // 2, 8192 // batch)] * batch
    if kind == 'mixed':
        return [1] * (batch // 2) + [min(256, 4096 // batch)] * (batch // 2)
    raise ValueError(f'unknown attention regime: {kind}')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def measure(torch, fn, repetitions, eviction=None):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repetitions):
        if eviction is not None:
            eviction.add_(1)
        begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        begin.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end))
    return dict(median_ms=statistics.median(samples), samples_ms=samples)


def run_arm(args, reference=False):
    import torch
    from grouped_splitk_validation import attention_reference, check_output, make_inputs
    from correctness.checks.check_flash_decode import varlen_reference
    if not torch.cuda.is_available():
        raise RuntimeError('H100 CUDA execution required')
    props = torch.cuda.get_device_properties(0)
    if props.major != 9:
        raise RuntimeError('this qualification targets Hopper (SM90)')
    if reference:
        version = require_vllm_version(importlib.metadata.version('vllm'))
        from vllm.vllm_flash_attn import flash_attn_varlen_func
    else:
        version = None
        from paged_flash_decode import flash_decode, flash_varlen
        from correctness.checks.check_flash_decode import run_checks
        run_checks()
    rows = []
    eviction = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device='cuda')
    for kind, batch, context in SHAPES:
        fixture = args.output_dir / f'{kind}-b{batch}-c{context}.pt'
        query_lengths = query_lengths_for(kind, batch, context)
        offsets = [0]
        for count in query_lengths:
            offsets.append(offsets[-1] + count)
        max_query = max(query_lengths)
        if not reference:
            if fixture.exists():
                raise ValueError(f'fixture already exists; use a fresh directory: {fixture}')
            args.output_dir.mkdir(parents=True, exist_ok=True)
            _, k_cpu, v_cpu, table_cpu, lengths_cpu = make_inputs(
                [context] * batch, device='cpu', seed=args.seed + batch + context)
            generator = torch.Generator(device='cpu').manual_seed(args.seed + batch + context + 1)
            q_cpu = torch.randn((offsets[-1], 12, 128), dtype=torch.float16, generator=generator)
            torch.save((q_cpu, k_cpu, v_cpu, table_cpu, lengths_cpu,
                        torch.tensor(offsets, dtype=torch.int32)), fixture)
        cpu = torch.load(fixture, map_location='cpu', weights_only=True)
        digest = hashlib.sha256(fixture.read_bytes()).hexdigest()
        q, k, v, table, lengths, cu = [t.to('cuda') for t in cpu]
        if reference:
            def operation():
                return flash_attn_varlen_func(q=q, k=k, v=v, cu_seqlens_q=cu,
                    seqused_k=lengths, max_seqlen_q=max_query, max_seqlen_k=context,
                    block_table=table, causal=kind != 'decode', softmax_scale=128 ** -.5,
                    fa_version=3, num_splits=0)
        else:
            def operation():
                if kind == 'decode':
                    return flash_decode(q, k, v, table, lengths)
                return flash_varlen(q, k, v, cu, table, lengths, max_query_len=max_query)
        expected = (attention_reference(q, k, v, table, lengths) if kind == 'decode'
                    else varlen_reference(q, k, v, cu, table, lengths))
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                operation()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = operation()
        torch.cuda.current_stream().wait_stream(side)
        graph.replay()
        error = None
        try:
            max_abs = check_output(output, expected)
        except AssertionError as exc:
            max_abs = None
            error = str(exc)
        row = dict(kind=kind, batch=batch, context=context, fixture_sha256=digest,
                   correct=error is None, correctness_error=error, max_abs=max_abs,
                   warm=measure(torch, graph.replay, args.repetitions),
                   cold=measure(torch, graph.replay, args.repetitions, eviction))
        rows.append(row)
        print(f'{kind} B={batch} C={context}: {row["warm"]["median_ms"]:.6f} ms; correct={row["correct"]}', flush=True)
        write_json(args.output_dir / ('reference.json' if reference else 'local.json'),
                   dict(status='complete' if len(rows) == len(SHAPES) else 'partial',
                        implementation='external_vllm_FA3' if reference else 'project_owned_SM90a_TMA_WGMMA_candidate',
                        vllm_version=version, torch_version=torch.__version__,
                        gpu=dict(name=props.name, memory=props.total_memory, sm=props.major * 10 + props.minor),
                        repetitions=args.repetitions, seed=args.seed, rows=rows))
        del graph, output, expected, operation, q, k, v, table, lengths, cu, cpu
        torch.cuda.empty_cache()


def analyze(directory, max_ratio):
    local, ref = [json.loads((directory / f'{name}.json').read_text()) for name in ('local', 'reference')]
    require_vllm_version(ref['vllm_version'])
    if any(item['status'] != 'complete' for item in (local, ref)) or local['gpu'] != ref['gpu']:
        raise ValueError('both complete arms on matched hardware are required')
    if local['seed'] != ref['seed'] or local['repetitions'] != ref['repetitions']:
        raise ValueError('measurement settings differ')
    expected = set(SHAPES)
    if any({(r['kind'], r['batch'], r['context']) for r in arm['rows']} != expected or len(arm['rows']) != len(expected)
           for arm in (local, ref)):
        raise ValueError('missing or duplicated qualification shapes')
    ref_rows = {(r['kind'], r['batch'], r['context']): r for r in ref['rows']}
    rows = []
    for ours in local['rows']:
        theirs = ref_rows[ours['kind'], ours['batch'], ours['context']]
        if ours['fixture_sha256'] != theirs['fixture_sha256']:
            raise ValueError('attention tensor fixtures differ')
        ratios = {mode: ours[mode]['median_ms'] / theirs[mode]['median_ms'] for mode in ('warm', 'cold')}
        passed = ours['correct'] and theirs['correct'] and all(v <= max_ratio for v in ratios.values())
        rows.append(dict(kind=ours['kind'], batch=ours['batch'], context=ours['context'], local_over_fa3_ms=ratios, passed=passed))
    result = dict(status='complete', kernel_parity_passed=all(r['passed'] for r in rows),
                  max_local_over_fa3_ms=max_ratio, full_model_qualified=False, rows=rows)
    write_json(directory / 'comparison.json', result)
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('run', 'run-local', 'run-reference', 'analyze'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--vllm-python')
    parser.add_argument('--seed', type=int, default=20260914)
    parser.add_argument('--repetitions', type=int, default=30)
    parser.add_argument('--max-ratio', type=float, default=1.05,
                        help='kernel parity requires local median <= this * reference in every warm/cold shape')
    args = parser.parse_args()
    if args.repetitions < 3 or args.max_ratio <= 0:
        parser.error('requires >=3 repetitions and a positive ratio')
    if args.action == 'run':
        if not args.vllm_python:
            parser.error('--vllm-python is required for the isolated reference')
        common = ['--output-dir', str(args.output_dir), '--seed', str(args.seed),
                  '--repetitions', str(args.repetitions)]
        for executable, action in ((sys.executable, 'run-local'), (args.vllm_python, 'run-reference')):
            subprocess.run([executable, str(Path(__file__)), action, *common], cwd=ROOT, check=True)
    elif args.action in ('run-local', 'run-reference'):
        run_arm(args, args.action == 'run-reference')
        return
    result = analyze(args.output_dir, args.max_ratio)
    if not result['kernel_parity_passed']:
        raise SystemExit('FA3 performance/correctness target not met; detailed timings retained')


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT))
    main()
