#!/usr/bin/env python3
"""Bounded, resumable attention-only tuning from actual eight-cell CPU schedules.

No model downloads. No installs. No production policy updates. Independent local
candidate and vLLM FA3 run in separate interpreters. GPU execution is unqualified
until this and full-model checks have actually passed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import signal
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'custom_kernels'), str(ROOT / 'benchmarks'),
               str(ROOT / 'experiments/integration'), str(Path(__file__).parent)]
from qualify_flash_decode import measure, write_json
from reference_version import require_vllm_version, VLLM_VERSION


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def source_digest():
    files = [Path(__file__), ROOT / 'custom_kernels/paged_flash_decode.py',
             ROOT / 'custom_kernels/hopper_attention/attention.cu',
             ROOT / 'custom_kernels/hopper_attention/setup.py',
             ROOT / 'correctness/checks/check_flash_decode.py',
             ROOT / 'experiments/decode/qualify_flash_decode.py']
    return digest({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})


def trace_case(case, requests):
    import torch
    sys.path.insert(0, str(ROOT / 'engine/cpp/build'))
    import inference_engine_cpp as cpp
    from benchmark_scheduler_decode import make_config
    config = make_config(cpp, case)
    config.packed_mixed_step = True
    loop = cpp.IterationLoop(config, torch.device('cpu'))
    pending = sorted(requests, key=lambda row: (row['arrival'], row['id']))
    cursor = step = 0
    rows = []
    limit = sum(len(r['prompt']) + r['output'] for r in requests) + max(r['arrival'] for r in requests) + 1
    while cursor < len(pending) or loop.num_pending() or loop.num_running():
        if not loop.num_pending() and not loop.num_running() and cursor < len(pending):
            step = max(step, pending[cursor]['arrival'])
        while cursor < len(pending) and pending[cursor]['arrival'] <= step:
            row = pending[cursor]
            loop.submit_request(row['prompt'], row['output'])
            cursor += 1
        def forward(ids, positions, slots, cu, context, table, max_query, decode):
            queries = [1] * context.numel() if decode else (cu[1:] - cu[:-1]).tolist()
            kind = 'decode' if decode else 'mixed' if loop.current_step_is_mixed() else 'prefill'
            rows.append(dict(kind=kind, queries=queries, lengths=context.tolist()))
            return torch.zeros((context.numel(), 1))
        loop.step(forward)
        loop.pop_completed()
        step += 1
        if step > limit:
            raise RuntimeError('CPU schedule exceeded progress bound')
    return rows


def make_plan(args):
    from benchmark_current_8_vs_vllm import SHAPES, input_contract
    from benchmark_current_mixed_8_vs_vllm import mixed_plan
    selected = {}
    coverage = {}
    for shape in SHAPES:
        case, requests, *_ = input_contract(args, shape)
        mixed_case, mixed_requests, *_ = mixed_plan(args, shape, validate_schedule=False)
        for mode, cfg, reqs in (('burst', case, requests), ('mixed', mixed_case, mixed_requests)):
            rows = trace_case(cfg, reqs)
            for kind in sorted({r['kind'] for r in rows}):
                candidates = [r for r in rows if r['kind'] == kind]
                # Decode endpoints capture context growth. Prefill/mixed choose
                # highest dot-product work, not a hand-invented cohort.
                fullest = max(len(r['queries']) for r in candidates)
                last_full = next(r for r in reversed(candidates) if len(r['queries']) == fullest)
                picks = [candidates[0], last_full, candidates[-1]] if kind == 'decode' else [
                    max(candidates, key=lambda r: sum(q * c for q, c in zip(r['queries'], r['lengths'])))]
                tag = f'{shape}/{mode}/{kind}'
                coverage[tag] = []
                for row in picks:
                    key = digest(row)[:16]
                    selected.setdefault(key, dict(id=key, **row, workloads=[]))
                    if tag not in selected[key]['workloads']:
                        selected[key]['workloads'].append(tag)
                    if key not in coverage[tag]:
                        coverage[tag].append(key)
    return dict(schema=2, shapes=sorted(SHAPES), cases=list(selected.values()), coverage=coverage,
                scope='representative actual attention shapes, not every iteration or full-engine timing',
                implemented_controls=['split-K', 'QK/softmax overlap', 'register-fed PV',
                                      'KV tiles 64/128', 'compact mixed worklist including construction'],
                pending_features=['wider query tiles', 'loading-strategy variants', 'full-model qualification'])


def configs(case):
    # Cap scratch/reduction cost for large query cohorts; no combinatorial sweep.
    splits = (1, 2, 4, 8, 16) if case['kind'] == 'decode' else (1, 2, 4)
    return [dict(split_k=k, overlap_qk=overlap, tile_n=64, register_pv=False, compact=False)
            for k in splits for overlap in (True, False)]


def baseline_config(case):
    if case['kind'] != 'decode':
        return dict(split_k=1, overlap_qk=True, tile_n=64, register_pv=False, compact=False)
    batch = len(case['queries'])
    capacity = math.ceil(max(case['lengths']) / 16) * 16
    return dict(split_k=min(32, (capacity + 63) // 64, max(1, (256 + batch * 2 - 1) // (batch * 2))),
                overlap_qk=True, tile_n=64, register_pv=False, compact=False)


def architecture_configs(case, control):
    # A small interaction grid, not every tile x split x overlap combination.
    return [{**control, 'tile_n': n, 'register_pv': registers, 'compact': compact}
            for n in (64, 128) for registers in (False, True)
            for compact in ((False, True) if case['kind'] == 'mixed' else (False,))]


def evaluation_configs(case, control, chosen):
    rows = [('baseline', baseline_config(case)), ('tuned_control', control),
            ('register_pv_only', {**control, 'register_pv': True}),
            ('tile_128_only', {**control, 'tile_n': 128})]
    if case['kind'] == 'mixed':
        rows.append(('compact_only', {**control, 'compact': True}))
    return [*rows, ('selected', chosen)]


def best_passing(rows):
    passing = [r for r in rows if r['correctness_error'] is None]
    if not passing:
        raise ValueError('no correct configuration')
    return min(passing, key=lambda r: r['cold']['median_ms'])['config']


def fixtures(case, seed):
    """Identical CPU-generated FP16 fixtures in both environments; no large disk cache."""
    import numpy as np
    import torch
    rng = np.random.Generator(np.random.PCG64(seed))
    lengths = case['lengths']
    counts = [(n + 15) // 16 for n in lengths]
    pages = sum(counts)
    table = np.full((len(lengths), max(counts)), -1, dtype=np.int32)
    permutation = rng.permutation(pages).astype(np.int32)
    offset = 0
    for i, count in enumerate(counts):
        table[i, :count] = permutation[offset:offset + count]
        offset += count
    q = rng.standard_normal((sum(case['queries']), 12, 128), dtype=np.float32).astype(np.float16)
    k = rng.standard_normal((pages, 16, 2, 128), dtype=np.float32).astype(np.float16)
    v = rng.standard_normal(k.shape, dtype=np.float32).astype(np.float16)
    cu = np.array([0, *np.cumsum(case['queries']).tolist()], dtype=np.int32)
    arrays = (q, k, v, cu, table, np.asarray(lengths, dtype=np.int32))
    fingerprint = hashlib.sha256()
    for array in arrays:
        fingerprint.update(array.tobytes())
    return [torch.from_numpy(a).to('cuda') for a in arrays], fingerprint.hexdigest()


def capture(torch, operation):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            operation()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = operation()
    torch.cuda.current_stream().wait_stream(stream)
    graph.replay()
    return graph, output


def worker(args):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('H100 required')
    props = torch.cuda.get_device_properties(0)
    if (props.major, props.minor) != (9, 0) or 'H100' not in props.name:
        raise RuntimeError('this experiment is restricted to H100')
    reference = args.arm == 'reference'
    if reference:
        require_vllm_version(importlib.metadata.version('vllm'))
        from vllm.vllm_flash_attn import flash_attn_varlen_func
    else:
        from paged_flash_decode import flash_varlen, _extension
        extension = _extension()  # Includes source/binary hash and layout checks.
    environment = dict(torch=torch.__version__, cuda=torch.version.cuda,
                       gpu=props.name, memory=props.total_memory,
                       uuid=str(getattr(props, 'uuid', 'unavailable')),
                       vllm=importlib.metadata.version('vllm') if reference else None,
                       numpy=importlib.metadata.version('numpy'))
    if not reference:
        environment['kernel_binary_sha256'] = hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest()
        environment['kernel_variants'] = extension.kernel_info()
    env_path = args.output_dir / f'{args.arm}-environment.json'
    if env_path.exists() and json.loads(env_path.read_text()) != environment:
        raise ValueError('environment changed; use a fresh output directory')
    write_json(env_path, environment)
    from correctness.checks.check_flash_decode import varlen_reference, run_checks
    from grouped_splitk_validation import check_output
    if args.action == '_check':
        case = dict(queries=[1, 17], lengths=[129, 33])
        (q, k, v, cu, table, lengths), _ = fixtures(case, args.seed)
        expected = varlen_reference(q, k, v, cu, table, lengths)
        if reference:
            output = flash_attn_varlen_func(q=q, k=k, v=v, cu_seqlens_q=cu,
                seqused_k=lengths, block_table=table, max_seqlen_q=17,
                max_seqlen_k=129, causal=True, fa_version=3, num_splits=0)
            check_output(output, expected)
        else:
            from correctness.checks.check_flash_decode import variant_configs
            for config in variant_configs():
                check_output(flash_varlen(q, k, v, cu, table, lengths,
                    max_query_len=17, **config), expected)
        print(f'{args.arm}: small ragged CUDA smoke passed', flush=True)
        return
    if not reference and not (args.output_dir / 'adversarial-pass.json').exists():
        checks = run_checks()
        write_json(args.output_dir / 'adversarial-pass.json', dict(checks=checks))
    plan = json.loads((args.output_dir / 'plan.json').read_text())
    eviction = torch.zeros(128 * 1024 * 1024, device='cuda', dtype=torch.uint8)
    for index, case in enumerate(plan['cases']):
        path = args.output_dir / args.arm / f'{case["id"]}.json'
        if path.exists():
            saved = json.loads(path.read_text())
            if saved.get('status') != 'complete' or not saved.get('correctness_passed'):
                raise ValueError(f'{path}: failed case cannot be silently resumed')
            continue
        case_seed = args.seed + int(case['id'][:7], 16)
        chosen = None
        control = None
        tuning = []
        # Seed used for final measurement is held out from configuration selection.
        rounds = (('evaluation', case_seed + 1),) if reference else (
            ('tuning', case_seed), ('evaluation', case_seed + 1))
        for stage, seed in rounds:
            tensors, fingerprint = fixtures(case, seed)
            if reference:
                local_path = args.output_dir / 'local' / f'{case["id"]}.json'
                if json.loads(local_path.read_text())['fixture_sha256'] != fingerprint:
                    raise ValueError('held-out fixtures differ across environments; stopped before timing')
            q, k, v, cu, table, lengths = tensors
            expected = varlen_reference(q, k, v, cu, table, lengths)
            def operation(config):
                if reference:
                    return flash_attn_varlen_func(q=q, k=k, v=v, cu_seqlens_q=cu,
                        seqused_k=lengths, block_table=table, max_seqlen_q=max(case['queries']),
                        max_seqlen_k=max(case['lengths']), causal=True,
                        fa_version=3, num_splits=0)
                return flash_varlen(q, k, v, cu, table, lengths,
                    max_query_len=max(case['queries']), **config)
            measured = {}
            def measure_config(config, search_stage):
                key = digest(config)
                if key not in measured:
                    write_json(args.output_dir / 'progress.json', dict(arm=args.arm, case=case['id'],
                        stage=stage, search_stage=search_stage, config=config,
                        case_index=index + 1, total_cases=len(plan['cases'])))
                    graph, output = capture(torch, lambda: operation(config))
                    failure = None
                    try:
                        max_error = check_output(output, expected)
                    except AssertionError as error:
                        failure, max_error = str(error), None
                    reps = args.tune_repetitions if stage == 'tuning' else args.repetitions
                    row = dict(config=config, search_stage=search_stage,
                               correctness_error=failure, max_abs=max_error,
                               cold=measure(torch, graph.replay, reps, eviction))
                    if stage == 'evaluation':
                        row['warm'] = measure(torch, graph.replay, reps)
                    measured[key] = row
                    del graph, output
                return measured[key]
            if stage == 'tuning':
                import random
                rng = random.Random(seed)
                def sweep(candidates, label):
                    rng.shuffle(candidates)
                    for config in candidates:
                        measure_config(config, label)
                sweep([*configs(case), baseline_config(case)], 'split_overlap')
                try:
                    control = best_passing(list(measured.values()))
                    sweep(architecture_configs(case, control), 'architecture')
                    architecture = best_passing(list(measured.values()))
                    sweep([{**c, **{key: architecture[key] for key in ('tile_n', 'register_pv', 'compact')}}
                           for c in configs(case)], 'retune_selected_architecture')
                    chosen = best_passing(list(measured.values()))
                except ValueError:
                    write_json(path, dict(status='failed_correctness', case=case, tuning=list(measured.values())))
                    raise RuntimeError('no correct configuration; stopped before spending on other cases')
                tuning = list(measured.values())
            else:
                selections = [('reference', None)] if reference else evaluation_configs(case, control, chosen)
                # Preserve roles, but never remeasure an identical configuration
                # and accidentally report noise as an intervention's benefit.
                results = [{**measure_config(config, 'held_out'), 'role': role} for role, config in selections]
                write_json(path, dict(status='complete', case=case, fixture_sha256=fingerprint,
                    tuning=tuning, selected_config=chosen, evaluation=results,
                    tuned_control=control,
                    evaluation_seed=seed, numerical_tolerance=dict(atol=.002, rtol=.002),
                    correctness_passed=all(r['correctness_error'] is None for r in results)))
                if any(r['correctness_error'] is not None for r in results):
                    raise RuntimeError(f'{path}: held-out correctness failed; timings saved, remaining cases stopped')
            del expected, tensors, q, k, v, cu, table, lengths
        print(f'{args.arm} {index + 1}/{len(plan["cases"])}: {case["id"]} {case["kind"]}', flush=True)


def analyze(args, plan):
    environments = [json.loads((args.output_dir / f'{arm}-environment.json').read_text())
                    for arm in ('local', 'reference')]
    for field in ('gpu', 'memory', 'uuid'):
        if environments[0][field] != environments[1][field]:
            raise ValueError(f'reference hardware differs: {field}')
    rows = []
    for case in plan['cases']:
        local, reference = [json.loads((args.output_dir / arm / f'{case["id"]}.json').read_text())
                            for arm in ('local', 'reference')]
        if any(r['status'] != 'complete' for r in (local, reference)):
            raise ValueError('failed or incomplete case; inspect saved results')
        if local['fixture_sha256'] != reference['fixture_sha256']:
            raise ValueError('fixture mismatch across environments')
        by_role = {r['role']: r for r in local['evaluation']}
        baseline, tuned, control = [by_role[role] for role in ('baseline', 'selected', 'tuned_control')]
        ref = reference['evaluation'][0]
        rows.append(dict(id=case['id'], kind=case['kind'], workloads=case['workloads'],
            correct=local['correctness_passed'] and reference['correctness_passed'],
            selected_config=local['selected_config'],
            interventions_vs_tuned_control={role: {
                'correct': row['correctness_error'] is None,
                'speedup': {mode: control[mode]['median_ms'] / row[mode]['median_ms'] for mode in ('cold', 'warm')}}
                for role, row in by_role.items() if role.endswith('_only')},
            cold_tuning_speedup=baseline['cold']['median_ms'] / tuned['cold']['median_ms'],
            vs_fa3={mode: ref[mode]['median_ms'] / tuned[mode]['median_ms'] for mode in ('cold', 'warm')}))
    report = dict(status='complete', scope=plan['scope'], full_model_qualified=False,
                  parity_within_5_percent=all(r['correct'] and all(v >= 1 / 1.05 for v in r['vs_fa3'].values()) for r in rows),
                  rows=rows, pending_features=plan['pending_features'])
    write_json(args.output_dir / 'summary.json', report)
    write_json(args.output_dir / 'candidate-dispatch.json', dict(production_enabled=False,
        qualification='attention microbenchmark only', cases=[dict(case=c, config=r['selected_config'])
        for c, r in zip(plan['cases'], rows) if r['correct']]))
    print(f'{len(rows)} cases; FA3 5% parity gate={report["parity_within_5_percent"]}; full-model qualification pending')
    if not all(r['correct'] for r in rows):
        raise RuntimeError('numerical failures; timings retained but no correctness qualification')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'run', 'analyze', '_check', '_worker'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--suite-dir', type=Path, default=ROOT / 'experiments/results/full-checkpoint-20260916T033540Z')
    parser.add_argument('--vllm-python')
    parser.add_argument('--arm', choices=('local', 'reference'))
    parser.add_argument('--seed', type=int, default=20260914)
    parser.add_argument('--repetitions', type=int, default=21)
    parser.add_argument('--tune-repetitions', type=int, default=7)
    parser.add_argument('--max-seconds', type=int, default=600)
    parser.add_argument('--build', action='store_true', help='build local extension using existing CUTLASS_PATH; no downloads')
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.repetitions < 3 or args.tune_repetitions < 3 or args.max_seconds < 1:
        parser.error('require >=3 repetitions and a positive wall-time limit')
    if args.action in ('_check', '_worker'):
        if not args.arm:
            parser.error('--arm required for internal worker')
        worker(args)
        return
    if args.action == 'analyze':
        analyze(args, json.loads((args.output_dir / 'plan.json').read_text()))
        return
    plan = make_plan(args)
    contract = dict(source=source_digest(), plan=plan, seed=args.seed,
                    repetitions=args.repetitions, tune_repetitions=args.tune_repetitions,
                    reference_version=VLLM_VERSION)
    contract_path = args.output_dir / 'contract.json'
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError('source, inputs or measurement settings changed; use a fresh output directory')
    write_json(contract_path, contract)
    write_json(args.output_dir / 'plan.json', plan)
    print(f'8 workloads, burst + mixed; {len(plan["cases"])} deduplicated representative cases; '
          f'staged split/overlap -> architecture -> retune search (at most 30 configs/case)', flush=True)
    if args.action == 'plan':
        return
    if not args.vllm_python:
        parser.error('--vllm-python is required; do not install vLLM into the local environment')
    deadline = time.monotonic() + args.max_seconds
    def run(command, cwd=ROOT, limit=None):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('wall-time budget exhausted; completed cases are saved')
        process = subprocess.Popen(command, cwd=cwd, start_new_session=True)
        try:
            code = process.wait(timeout=min(remaining, limit) if limit else remaining)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise
        if code:
            raise subprocess.CalledProcessError(code, command)
    common = ['--output-dir', str(args.output_dir), '--seed', str(args.seed),
              '--repetitions', str(args.repetitions), '--tune-repetitions', str(args.tune_repetitions)]
    try:
        # Check external environment before spending time building/tuning local.
        run([args.vllm_python, str(Path(__file__)), '_check', '--arm', 'reference', *common], limit=90)
        if args.build:
            run([sys.executable, 'setup.py', 'build_ext', '--inplace'], ROOT / 'custom_kernels/hopper_attention')
        run([sys.executable, str(Path(__file__)), '_check', '--arm', 'local', *common], limit=90)
        for executable, arm in ((sys.executable, 'local'), (args.vllm_python, 'reference')):
            run([executable, str(Path(__file__)), '_worker', '--arm', arm, *common])
    except (subprocess.TimeoutExpired, TimeoutError):
        raise SystemExit('Wall-time budget reached. Completed cases retained; repeat the same command to resume.')
    analyze(args, plan)


if __name__ == '__main__':
    main()
