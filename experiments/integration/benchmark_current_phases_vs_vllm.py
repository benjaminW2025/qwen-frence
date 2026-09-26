#!/usr/bin/env python3
"""Synchronized prefill/decode/mixed step diagnostics for the eight-cell table.

This is a separate pass over each cell's burst and staggered workloads. It intentionally
synchronizes after every engine step so a step's wall time includes completed GPU
work; it must not be substituted for the unsynchronized whole-workload rate.
The two schedulers may do different amounts of work per step, so counts and
total phase time accompany every median step latency.
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for directory in (HERE, ROOT / "baseline", ROOT / "benchmarks",
                  ROOT / "engine/model_runner", ROOT / "engine/kvcache",
                  ROOT / "engine/cpp/build"):
    sys.path.insert(0, str(directory))

from benchmark_current_8_vs_vllm import (ENGINE_FLAGS, SHAPES, adapter_options as
                                         burst_adapter_options, atomic_json,
                                         commit_matches, dispatch_plan, input_contract,
                                         repository_commit, resume_options)
from benchmark_current_mixed_8_vs_vllm import (adapter_options, check_local_result,
                                               mixed_plan, vllm_once)
from benchmark_latest_vs_vllm import resolve_model_source
from fixed_regime import PREFILL_TOKENS_PER_STEP, verify_fixed_result
from reference_version import require_vllm_version

KINDS = ("prefill", "decode", "mixed")
TIMING_SCHEME = "synchronized-burst-and-mixed-step-wall-v1"


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("action", choices=("plan", "run-table", "run-cell",
                                           "run-local", "run-vllm", "analyze"))
    result.add_argument("--suite-dir", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--shape-id", choices=SHAPES)
    result.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--seed", type=int, default=20260914)
    result.add_argument("--warmups", type=int, default=1)
    result.add_argument("--repetitions", type=int, default=3)
    result.add_argument("--vllm-python")
    result.add_argument("--resume-commit", action="append")
    return result


def paths(args, shape_id):
    root = args.output_dir / "phases" / shape_id
    return root / "local.json", root / "vllm.json", root / "comparison.json"


def phase_summary(steps, kinds=KINDS):
    result = {}
    for kind in kinds:
        values = [step["wall_ms"] for step in steps if step["kind"] == kind]
        if not values or any(value <= 0 for value in values):
            raise AssertionError(f"no valid {kind} step latency")
        result[kind] = {"steps": len(values), "total_wall_ms": sum(values),
                        "median_step_wall_ms": statistics.median(values)}
    return result


def median_phases(runs):
    result = {}
    for kind in KINDS:
        counts = [run["phases"][kind]["steps"] for run in runs]
        if len(set(counts)) != 1:
            raise AssertionError(f"{kind} step count changed between repetitions: {counts}")
        result[kind] = {"steps": counts[0],
                        "total_wall_ms": statistics.median(
                            run["phases"][kind]["total_wall_ms"] for run in runs),
                        "median_step_wall_ms": statistics.median(
                            run["phases"][kind]["median_step_wall_ms"] for run in runs)}
    return result


def output_digest(outputs):
    normalized = {str(key): value for key, value in outputs.items()}
    return hashlib.sha256(json.dumps(normalized, sort_keys=True,
                          separators=(",", ":")).encode()).hexdigest()


def validate_saved(path, args, shape_id, fingerprint, model):
    if not path.is_file():
        return None
    row = json.loads(path.read_text())
    expected = {"status": "complete", "shape_id": shape_id,
                "timing_scheme": TIMING_SCHEME, "workload_sha256": fingerprint,
                "model": model, "warmups": args.warmups,
                "repetitions": args.repetitions}
    if (any(row.get(key) != value for key, value in expected.items())
            or not commit_matches(row.get("repository_commit"),
                                  repository_commit(), args.resume_commit)):
        raise ValueError(f"stale or mismatched phase result: {path}")
    if set(row.get("phases", {})) != set(KINDS) or len(row.get("runs", [])) != args.repetitions:
        raise ValueError(f"incomplete phase result: {path}")
    if path.name == "vllm.json":
        require_vllm_version(row.get("vllm_version"))
    elif row.get("engine_flags") != ENGINE_FLAGS:
        raise ValueError(f"stale local implementation in phase result: {path}")
    return row


def run_local(args, shape_id, model_source):
    case, requests, _, arrival, buckets, fingerprint, blocks = mixed_plan(args, shape_id)
    burst_case, burst_requests, _, _, burst_blocks = input_contract(args, shape_id)
    if blocks != burst_blocks:
        raise AssertionError(f"{shape_id}: burst/mixed KV capacities differ")
    burst_dispatch = dispatch_plan(burst_case, burst_requests, args.seed)
    path, _, _ = paths(args, shape_id)
    if validate_saved(path, args, shape_id, fingerprint, model_source):
        print(f"{shape_id}: phase local complete; reusing", flush=True)
        return
    from model_setup import check_startup, load_model_only
    from benchmark_scheduler_decode import execute, make_config
    from model_adapter import CppPackedMixedModelAdapter, allocate_pool
    from kernel_dispatch import _load
    import inference_engine_cpp as cpp
    import torch

    setup = check_startup(args.device)
    engine, _, _ = load_model_only(model_source, args.device, "float16",
                                   hub_transfer=setup["hub_transfer"])
    pool = allocate_pool(engine.cfg, blocks, engine.device)
    config = make_config(cpp, burst_case)
    config.packed_mixed_step = True
    adapter = CppPackedMixedModelAdapter(
        engine.model, pool, None,
        **burst_adapter_options(burst_case, burst_dispatch["prefill_buckets"]))
    if (adapter.graph_decoder.buckets != [burst_case["max_running"]]
            or not adapter.enable_residual_rmsnorm
            or not adapter.enable_native_decode_qkv_postprocess
            or not adapter.piecewise_prefill.enable_swiglu_fusion):
        raise AssertionError(f"{shape_id}: phase burst engine flags differ")
    burst_rows = []
    burst_digest = None
    for index in range(args.warmups + args.repetitions):
        for tensor in pool.k_pool + pool.v_pool:
            tensor.fill_(float("nan"))
        row = execute(torch, cpp.IterationLoop(config, torch.device(args.device)),
                      adapter, burst_requests, synchronize_steps=True)
        verify_fixed_result(row, shape_id)
        digest = output_digest(row["outputs"])
        if burst_digest is None:
            burst_digest = digest
        elif digest != burst_digest:
            raise AssertionError(f"{shape_id}: phase burst outputs changed across runs")
        if adapter.piecewise_prefill.eager_calls or not adapter.decisions.get("flash-local"):
            raise AssertionError(f"{shape_id}: phase burst missed graph dispatch")
        if index >= args.warmups:
            burst_rows.append(phase_summary(row["steps"], ("prefill", "decode")))
    if sorted(adapter.piecewise_prefill.shapes) != burst_dispatch["prefill_buckets"]:
        raise AssertionError(f"{shape_id}: phase burst prefill bucket missed capture")
    del adapter

    config = make_config(cpp, case)
    config.packed_mixed_step = True
    adapter = CppPackedMixedModelAdapter(
        engine.model, pool, None, **adapter_options(case, buckets))
    measured = []
    mixed_digest = None
    for index in range(args.warmups + args.repetitions):
        for tensor in pool.k_pool + pool.v_pool:
            tensor.fill_(float("nan"))
        row = execute(torch, cpp.IterationLoop(config, torch.device(args.device)),
                      adapter, requests, synchronize_steps=True)
        check_local_result(row, requests, arrival)
        if adapter.piecewise_prefill.eager_calls or not adapter.decisions.get(
                "packed_mixed_cpp_varlen") or not adapter.decisions.get("flash-local"):
            raise AssertionError(f"{shape_id}: phase mixed missed graph dispatch")
        digest = output_digest(row["outputs"])
        if mixed_digest is None:
            mixed_digest = digest
        elif digest != mixed_digest:
            raise AssertionError(f"{shape_id}: phase mixed outputs changed across runs")
        if index >= args.warmups:
            phases = dict(burst_rows[index - args.warmups])
            phases.update(phase_summary(row["steps"], ("mixed",)))
            measured.append({"phases": phases, "burst_outputs_sha256": burst_digest,
                             "mixed_outputs_sha256": mixed_digest})
    if sorted(adapter.piecewise_prefill.shapes) != buckets:
        raise AssertionError(f"{shape_id}: phase mixed prefill bucket missed capture")
    atomic_json(path, {"status": "complete", "timing_scheme": TIMING_SCHEME,
                       "shape_id": shape_id, "model": model_source,
                       "repository_commit": repository_commit(),
                       "workload_sha256": fingerprint, "warmups": args.warmups,
                       "repetitions": args.repetitions, "num_blocks": blocks,
                       "arrival_step": arrival, "prefill_buckets": buckets,
                       "burst_prefill_buckets": burst_dispatch["prefill_buckets"],
                       "engine_flags": ENGINE_FLAGS, "runs": measured,
                       "burst_outputs_sha256": burst_digest,
                       "mixed_outputs_sha256": mixed_digest,
                       "phases": median_phases(measured)})


def run_vllm(args, shape_id, model_source):
    require_vllm_version(importlib.metadata.version("vllm"))
    case, requests, first, arrival, _, fingerprint, blocks = mixed_plan(
        args, shape_id, validate_schedule=False)
    _, burst_requests, _, _, burst_blocks = input_contract(args, shape_id)
    if blocks != burst_blocks:
        raise AssertionError(f"{shape_id}: burst/mixed KV capacities differ")
    _, path, _ = paths(args, shape_id)
    if validate_saved(path, args, shape_id, fingerprint, model_source):
        print(f"{shape_id}: phase vLLM complete; reusing", flush=True)
        return
    from profile_prefill_mixed_vs_vllm import configure_vllm_step_mode
    configure_vllm_step_mode()
    from benchmark_backends import _matched_kv_cache_bytes
    from vllm import LLM

    version = importlib.metadata.version("vllm")
    require_vllm_version(version)
    kv_bytes = _matched_kv_cache_bytes(model_source, dtype="float16",
                                       block_size=16, num_blocks=blocks)
    llm = LLM(model=model_source, dtype="float16", seed=args.seed,
              max_num_seqs=case["max_running"],
              max_num_batched_tokens=PREFILL_TOKENS_PER_STEP,
              max_model_len=max(case["lengths"]) + max(case["outputs"]),
              block_size=16, enable_prefix_caching=False,
              enable_chunked_prefill=True, generation_config="vllm",
              enforce_eager=False, kv_cache_memory_bytes=kv_bytes)
    measured = []
    burst_digest = mixed_digest = None
    for index in range(args.warmups + args.repetitions):
        burst_row = vllm_once(llm, burst_requests, len(burst_requests), None,
                              f"phase-burst-{shape_id}-{index}", synchronize_steps=True)
        mixed_row = vllm_once(llm, requests, first, arrival,
                              f"phase-mixed-{shape_id}-{index}", synchronize_steps=True)
        current_burst = output_digest(burst_row["outputs"])
        current_mixed = output_digest(mixed_row["outputs"])
        if burst_digest is None:
            burst_digest, mixed_digest = current_burst, current_mixed
        elif current_burst != burst_digest or current_mixed != mixed_digest:
            raise AssertionError(f"{shape_id}: phase vLLM outputs changed across runs")
        if index >= args.warmups:
            phases = phase_summary(burst_row["phase_steps"], ("prefill", "decode"))
            phases.update(phase_summary(mixed_row["phase_steps"], ("mixed",)))
            measured.append({"phases": phases,
                             "burst_outputs_sha256": burst_digest,
                             "mixed_outputs_sha256": mixed_digest})
    atomic_json(path, {"status": "complete", "timing_scheme": TIMING_SCHEME,
                       "shape_id": shape_id, "model": model_source,
                       "repository_commit": repository_commit(),
                       "workload_sha256": fingerprint, "warmups": args.warmups,
                       "repetitions": args.repetitions, "num_blocks": blocks,
                       "arrival_step": arrival, "vllm_version": version,
                       "vllm_step_mode": True, "runs": measured,
                       "burst_outputs_sha256": burst_digest,
                       "mixed_outputs_sha256": mixed_digest,
                       "phases": median_phases(measured)})


def analyze(args, shape_id, model_source):
    _, requests, _, arrival, buckets, fingerprint, blocks = mixed_plan(args, shape_id)
    local_path, vllm_path, path = paths(args, shape_id)
    local = validate_saved(local_path, args, shape_id, fingerprint, model_source)
    vllm = validate_saved(vllm_path, args, shape_id, fingerprint, model_source)
    if local is None or vllm is None:
        raise ValueError(f"{shape_id}: missing local or vLLM phase result")
    require_vllm_version(vllm["vllm_version"])
    burst_case, burst_requests, _, _, _ = input_contract(args, shape_id)
    expected_burst_buckets = dispatch_plan(
        burst_case, burst_requests, args.seed)["prefill_buckets"]
    if (local["num_blocks"] != blocks or vllm["num_blocks"] != blocks
            or local["arrival_step"] != arrival or vllm["arrival_step"] != arrival
            or local["prefill_buckets"] != buckets
            or local["burst_prefill_buckets"] != expected_burst_buckets
            or local["engine_flags"] != ENGINE_FLAGS
            or not vllm["vllm_step_mode"]):
        raise ValueError(f"{shape_id}: phase comparator configurations differ")
    rows = {}
    for kind in KINDS:
        ours, theirs = local["phases"][kind], vllm["phases"][kind]
        rows[kind] = {"local": ours, "vllm": theirs,
                      "median_step_local_over_vllm":
                      ours["median_step_wall_ms"] / theirs["median_step_wall_ms"]}
    report = {"status": "complete", "timing_scheme": TIMING_SCHEME,
              "shape_id": shape_id, "workload_sha256": fingerprint,
              "local_repository_commit": local["repository_commit"],
              "vllm_repository_commit": vllm["repository_commit"],
              "same_output_requests_expected": len(requests),
              "same_burst_output_hash": (local["burst_outputs_sha256"] ==
                                         vllm["burst_outputs_sha256"]),
              "same_mixed_output_hash": (local["mixed_outputs_sha256"] ==
                                         vllm["mixed_outputs_sha256"]),
              "phases": rows,
              "note": "Separate synchronized-step diagnostic: pure prefill/decode "
                      "from burst workload, mixed from staggered workload; "
                      "scheduler step counts and work per step may differ. "
                      "Do not compare these medians as whole-workload throughput."}
    atomic_json(path, report)
    print(f"{shape_id}: " + "; ".join(
        f"{kind} local {rows[kind]['local']['median_step_wall_ms']:.3f} ms / "
        f"vLLM {rows[kind]['vllm']['median_step_wall_ms']:.3f} ms "
        f"(steps {rows[kind]['local']['steps']}/{rows[kind]['vllm']['steps']})"
        for kind in KINDS), flush=True)
    return report


def forwarded(args, action, shape_id):
    interpreter = (getattr(args, "vllm_python", None) or sys.executable) if action == "run-vllm" else sys.executable
    return [interpreter, str(Path(__file__)), action,
            "--suite-dir", str(args.suite_dir), "--output-dir", str(args.output_dir),
            "--shape-id", shape_id, "--model", args.model, "--device", args.device,
            "--seed", str(args.seed), "--warmups", str(args.warmups),
            "--repetitions", str(args.repetitions), *resume_options(args.resume_commit)]


def main():
    args = parser().parse_args()
    if args.device != "cuda:0" or args.warmups < 1 or args.repetitions < 1:
        raise ValueError("requires cuda:0, >=1 warmup, and >=1 repetition")
    if any(len(prefix) < 7 or any(char not in "0123456789abcdef"
                                   for char in prefix.lower())
           for prefix in (args.resume_commit or [])):
        raise ValueError("--resume-commit must be at least seven hexadecimal characters")
    if args.action in ("run-cell", "run-local", "run-vllm") and args.shape_id is None:
        raise ValueError(f"{args.action} requires --shape-id")
    selected = (args.shape_id,) if args.shape_id else SHAPES
    if args.action == "plan":
        print(json.dumps({shape_id: {"arrival_step": mixed_plan(args, shape_id)[3],
                                     "timing_scheme": TIMING_SCHEME}
                          for shape_id in selected}, indent=2))
        return
    model_source = resolve_model_source(args)
    if args.action == "run-local":
        run_local(args, args.shape_id, model_source)
    elif args.action == "run-vllm":
        run_vllm(args, args.shape_id, model_source)
    elif args.action in ("run-cell", "run-table"):
        reports = []
        for shape_id in selected:
            subprocess.run(forwarded(args, "run-local", shape_id), cwd=ROOT, check=True)
            subprocess.run(forwarded(args, "run-vllm", shape_id), cwd=ROOT, check=True)
            reports.append(analyze(args, shape_id, model_source))
        if args.action == "run-table":
            atomic_json(args.output_dir / "phase-summary.json",
                        {"status": "complete", "rows": reports})
    else:
        reports = [analyze(args, shape_id, model_source) for shape_id in selected]
        if args.shape_id is None:
            atomic_json(args.output_dir / "phase-summary.json",
                        {"status": "complete", "rows": reports})


if __name__ == "__main__":
    main()
