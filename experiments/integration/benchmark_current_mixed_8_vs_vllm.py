#!/usr/bin/env python3
"""Full-workload staggered mixed comparison for each frozen factorial shape.

The second half of each cohort arrives after the first half has entered decode.
Arrival is an iteration index in both local and in-process vLLM. Throughput is
measured over the entire drained workload, including both waves.
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
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for directory in (HERE, ROOT / "baseline", ROOT / "benchmarks",
                  ROOT / "engine/model_runner", ROOT / "engine/kvcache",
                  ROOT / "engine/cpp/build"):
    sys.path.insert(0, str(directory))

from benchmark_current_8_vs_vllm import (ENGINE_FLAGS, SHAPES, atomic_json,
                                         commit_matches, input_contract,
                                         repository_commit, resume_options,
                                         validate_capture_options)
from fixed_regime import PREFILL_TOKENS_PER_STEP
from reference_version import require_vllm_version


def mixed_plan(args, shape_id, *, validate_schedule=True):
    case, frozen, _, _, blocks = input_contract(args, shape_id)
    first = case["max_running"] // 2
    prompt = case["lengths"][0]
    arrival = (first * prompt + PREFILL_TOKENS_PER_STEP - 1) // PREFILL_TOKENS_PER_STEP + 3
    if arrival >= case["outputs"][0]:
        raise ValueError(f"{shape_id}: first wave could finish before mixed arrival")
    requests = [dict(id=row["id"], prompt=row["prompt"], output=row["output"],
                     arrival=0 if row["id"] < first else arrival) for row in frozen]
    mixed_case = {**case, "id": f"mixed-{shape_id}",
                  "arrivals": [row["arrival"] for row in requests]}
    fingerprint = hashlib.sha256(json.dumps(requests, sort_keys=True,
                               separators=(",", ":")).encode()).hexdigest()
    # The reference interpreter must never load our torch-linked C++ extension.
    # Workload construction is deterministic Python; bucket planning is local-only.
    if not validate_schedule:
        return mixed_case, requests, first, arrival, None, fingerprint, blocks
    import torch
    import inference_engine_cpp as cpp
    from benchmark_integrated_graph import dry_schedule

    schedule = dry_schedule(torch, cpp, mixed_case, args.seed, requests)
    if arrival >= len(schedule["steps"]) or schedule["steps"][arrival]["kind"] != "mixed":
        raise AssertionError(f"{shape_id}: planned second-wave step is not mixed")
    prefill = [call[1] for step in schedule["steps"]
               for call in step["calls"] if not call[0]]
    packed = [sum(call[1] for call in step["calls"])
              for step in schedule["steps"] if step["kind"] == "mixed"]
    if not prefill or not packed:
        raise AssertionError(f"{shape_id}: CPU schedule lacks prefill or mixed work")
    buckets = sorted({max(prefill), max(packed)})
    fingerprint = hashlib.sha256(json.dumps(requests, sort_keys=True,
                               separators=(",", ":")).encode()).hexdigest()
    return mixed_case, requests, first, arrival, buckets, fingerprint, blocks


def paths(args, shape_id):
    root = args.output_dir / "mixed" / shape_id
    return root / "local.json", root / "vllm.json", root / "comparison.json"


def adapter_options(case, buckets):
    if not buckets or min(buckets) < 1:
        raise ValueError("mixed graph buckets must be positive")
    return dict(max_running=case["max_running"],
                max_context_length=max(length + output for length, output in
                                       zip(case["lengths"], case["outputs"])),
                decode_attention_policy="flash",
                decode_buckets=[case["max_running"]],
                max_capture_tokens=max(buckets),
                max_prefill_shapes=len(buckets), prefill_buckets=buckets,
                enable_residual_rmsnorm=True,
                enable_native_decode_qkv_postprocess=True,
                enable_prefill_swiglu_fusion=True)


def validate_saved(path, *, shape_id, fingerprint, model, args):
    if not path.is_file():
        return None
    row = json.loads(path.read_text())
    expected = {"status": "complete", "shape_id": shape_id,
                "workload_sha256": fingerprint, "model": model,
                "warmups": args.warmups, "repetitions": args.repetitions}
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(f"{path}: {key} differs; use a new output directory")
    if not commit_matches(row.get("repository_commit"), repository_commit(),
                          args.resume_commit):
        raise ValueError(f"{path}: source commit differs; use --resume-commit only "
                         "if this prior result is known valid")
    if path.name == "vllm.json":
        require_vllm_version(row.get("vllm_version"))
    elif row.get("engine_flags") != ENGINE_FLAGS:
        raise ValueError(f"{path}: local implementation differs; use a new output directory")
    return row


def check_local_result(row, requests, arrival):
    if len(row["outputs"]) != len(requests) or any(
            len(row["outputs"][request["id"]]) != request["output"]
            for request in requests):
        raise AssertionError("local mixed workload has missing or short output")
    if arrival >= len(row["steps"]) or row["steps"][arrival]["kind"] != "mixed":
        raise AssertionError(f"local arrival step {arrival} was not mixed")
    if not any(call[0] for call in row["steps"][arrival]["calls"]) or not any(
            not call[0] for call in row["steps"][arrival]["calls"]):
        raise AssertionError("local mixed step omitted a decode or prefill cohort")
    if not any(step["kind"] == "mixed" for step in row["steps"]):
        raise AssertionError("local workload contained no mixed steps")


def run_local(args, shape_id, model_source):
    case, requests, first, arrival, buckets, fingerprint, blocks = mixed_plan(args, shape_id)
    local_path, _, _ = paths(args, shape_id)
    if validate_saved(local_path, shape_id=shape_id, fingerprint=fingerprint,
                      model=model_source, args=args):
        print(f"{shape_id}: mixed local complete; reusing", flush=True)
        return
    from model_setup import check_startup, load_model_only
    import torch
    import inference_engine_cpp as cpp
    from benchmark_scheduler_decode import execute, make_config
    from model_adapter import CppPackedMixedModelAdapter, allocate_pool
    from naive_forward import SWIGLU_FUSION_ROW_THRESHOLD

    setup = check_startup(args.device)
    engine, _, _ = load_model_only(model_source, args.device, "float16",
                                   hub_transfer=setup["hub_transfer"])
    config = make_config(cpp, case)
    config.packed_mixed_step = True
    pool = allocate_pool(engine.cfg, blocks, engine.device)
    adapter = CppPackedMixedModelAdapter(
        engine.model, pool, None, **adapter_options(case, buckets))
    if (adapter.graph_decoder.buckets != [case["max_running"]]
            or not adapter.enable_residual_rmsnorm
            or not adapter.enable_native_decode_qkv_postprocess
            or not adapter.piecewise_prefill.enable_swiglu_fusion):
        raise AssertionError(f"{shape_id}: mixed regime flags did not reach model")
    measured = []
    signature = expected_outputs = None
    for index in range(args.warmups + args.repetitions):
        for tensor in pool.k_pool + pool.v_pool:
            tensor.fill_(float("nan"))
        row = execute(torch, cpp.IterationLoop(config, torch.device(args.device)),
                      adapter, requests)
        check_local_result(row, requests, arrival)
        if adapter.piecewise_prefill.eager_calls:
            raise AssertionError(f"{shape_id}: piecewise graph missed mixed token bucket")
        if not adapter.decisions.get("packed_mixed_cpp_varlen"):
            raise AssertionError(f"{shape_id}: C++ packed mixed dispatch was not used")
        if not adapter.decisions.get("flash-local"):
            raise AssertionError(f"{shape_id}: independent decode graph was not used")
        actual_signature = [(step["kind"], step["calls"]) for step in row["steps"]]
        if signature is None:
            signature, expected_outputs = actual_signature, row["outputs"]
        elif actual_signature != signature or row["outputs"] != expected_outputs:
            raise AssertionError(f"{shape_id}: mixed local schedule/output changed across runs")
        if index >= args.warmups:
            measured.append({"wall_ms": row["wall_ms"],
                             "output_tokens_per_s": row["output_tokens_per_s"],
                             "mixed_steps": sum(step["kind"] == "mixed"
                                                for step in row["steps"]),
                             "packed_mixed_calls": adapter.decisions["packed_mixed_cpp_varlen"],
                             "outputs": row["outputs"]})
        print(f"{shape_id}: mixed local {'warmup' if index < args.warmups else 'run'} "
              f"{index + 1}/{args.warmups + args.repetitions} "
              f"{row['output_tokens_per_s']:.1f} tok/s", flush=True)
    if sorted(adapter.piecewise_prefill.shapes) != buckets:
        raise AssertionError(f"{shape_id}: expected mixed graph buckets not captured")
    atomic_json(local_path, {"status": "complete", "shape_id": shape_id,
                             "model": model_source, "repository_commit": repository_commit(),
                             "workload_sha256": fingerprint, "warmups": args.warmups,
                             "repetitions": args.repetitions, "first_wave": first,
                             "arrival_step": arrival, "prefill_buckets": buckets,
                             "swiglu_fused_buckets": [bucket for bucket in buckets
                                 if bucket > SWIGLU_FUSION_ROW_THRESHOLD],
                             "engine_flags": ENGINE_FLAGS, "num_blocks": blocks,
                             "runs": measured,
                             "median_output_tokens_per_s": statistics.median(
                                 item["output_tokens_per_s"] for item in measured),
                             "piecewise_graph_buckets": sorted(
                                 adapter.piecewise_prefill.shapes)})


def phase_kind(progress, expected):
    active = [progress[key] for key, length in expected.items()
              if progress[key] < length]
    if not active:
        raise AssertionError("vLLM step has no active requests")
    if all(value == 0 for value in active):
        return "prefill"
    if all(value > 0 for value in active):
        return "decode"
    return "mixed"


def vllm_once(llm, requests, first, arrival, run_id, *, synchronize_steps=False):
    import torch
    from profile_latest_vs_vllm import add_vllm_requests

    torch.cuda.synchronize()
    started = time.perf_counter()
    expected = add_vllm_requests(llm, requests[:first], run_id, cumulative=True)
    progress = {key: 0 for key in expected}
    outputs = {}
    injected = first == len(requests)
    step = 0
    limit = sum(len(row["prompt"]) + row["output"] for row in requests) + (arrival or 0) + 1
    target_advanced = injected
    phase_steps = []
    while llm.llm_engine.has_unfinished_requests() or not injected:
        if step == arrival:
            if not all(0 < progress[key] < length for key, length in expected.items()):
                raise AssertionError("vLLM first wave was not actively decoding at injection")
            later = add_vllm_requests(llm, requests[first:], run_id, cumulative=True)
            expected.update(later)
            progress.update({key: 0 for key in later})
            injected = True
            before = dict(progress)
        kind = phase_kind(progress, expected) if synchronize_steps else None
        step_started = time.perf_counter() if synchronize_steps else None
        step_outputs = llm.llm_engine.step()
        if synchronize_steps:
            torch.cuda.synchronize()
            phase_steps.append({"kind": kind,
                                "wall_ms": (time.perf_counter() - step_started) * 1000})
        for row in step_outputs:
            key = str(row.request_id)
            if key not in progress:
                raise AssertionError(f"unexpected vLLM request {key}")
            if row.outputs:
                progress[key] = len(row.outputs[0].token_ids)
            if row.finished:
                tokens = list(row.outputs[0].token_ids)
                if len(tokens) != expected[key]:
                    raise AssertionError(f"{key}: vLLM output length differs")
                outputs[key.split(":", 1)[1]] = tokens
        if step == arrival:
            target_advanced = all(progress[key] == before[key] + 1
                                  for key in list(before)[:first])
        step += 1
        if step > limit:
            raise RuntimeError("vLLM mixed workload exceeded step limit")
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - started) * 1000
    if not target_advanced:
        raise AssertionError("vLLM injection step did not advance decoding first wave")
    if len(outputs) != len(requests) or any(
            len(outputs[str(request["id"])]) != request["output"]
            for request in requests):
        raise AssertionError("vLLM mixed workload has missing or short output")
    return {"wall_ms": wall_ms,
            "output_tokens_per_s": sum(request["output"] for request in requests)
            * 1000 / wall_ms,
            "total_steps": step, "outputs": outputs,
            "first_wave_advanced_on_injection": True,
            **({"phase_steps": phase_steps} if synchronize_steps else {})}


def run_vllm(args, shape_id, model_source):
    require_vllm_version(importlib.metadata.version("vllm"))
    case, requests, first, arrival, _, fingerprint, blocks = mixed_plan(
        args, shape_id, validate_schedule=False)
    _, result_path, _ = paths(args, shape_id)
    if validate_saved(result_path, shape_id=shape_id, fingerprint=fingerprint,
                      model=model_source, args=args):
        print(f"{shape_id}: mixed vLLM complete; reusing", flush=True)
        return
    from profile_prefill_mixed_vs_vllm import configure_vllm_step_mode
    configure_vllm_step_mode()
    import torch
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
    signature = expected_outputs = None
    for index in range(args.warmups + args.repetitions):
        row = vllm_once(llm, requests, first, arrival, f"mixed-{shape_id}-{index}")
        if signature is None:
            signature, expected_outputs = row["total_steps"], row["outputs"]
        elif row["total_steps"] != signature or row["outputs"] != expected_outputs:
            raise AssertionError(f"{shape_id}: vLLM mixed schedule/output changed across runs")
        if index >= args.warmups:
            measured.append(row)
        print(f"{shape_id}: mixed vLLM {'warmup' if index < args.warmups else 'run'} "
              f"{index + 1}/{args.warmups + args.repetitions} "
              f"{row['output_tokens_per_s']:.1f} tok/s", flush=True)
    atomic_json(result_path, {"status": "complete", "shape_id": shape_id,
                              "model": model_source, "repository_commit": repository_commit(),
                              "workload_sha256": fingerprint, "warmups": args.warmups,
                              "repetitions": args.repetitions, "first_wave": first,
                              "arrival_step": arrival, "num_blocks": blocks,
                              "vllm_version": version, "vllm_step_mode": True,
                              "runs": measured,
                              "median_output_tokens_per_s": statistics.median(
                                  item["output_tokens_per_s"] for item in measured)})


def analyze(args, shape_id, model_source):
    _, requests, first, arrival, buckets, fingerprint, blocks = mixed_plan(args, shape_id)
    local_path, vllm_path, report_path = paths(args, shape_id)
    local = validate_saved(local_path, shape_id=shape_id, fingerprint=fingerprint,
                           model=model_source, args=args)
    vllm = validate_saved(vllm_path, shape_id=shape_id, fingerprint=fingerprint,
                          model=model_source, args=args)
    if local is None or vllm is None:
        raise ValueError(f"{shape_id}: missing local or vLLM mixed result")
    require_vllm_version(vllm["vllm_version"])
    if (local["num_blocks"] != blocks or vllm["num_blocks"] != blocks
            or local["prefill_buckets"] != buckets or local["arrival_step"] != arrival
            or vllm["arrival_step"] != arrival or local["first_wave"] != first
            or vllm["first_wave"] != first or local.get("engine_flags") != ENGINE_FLAGS):
        raise ValueError(f"{shape_id}: mixed comparator configurations differ")
    local_outputs, vllm_outputs = (local["runs"][0]["outputs"],
                                   vllm["runs"][0]["outputs"])
    exact = sum(local_outputs[str(request["id"])] ==
                vllm_outputs[str(request["id"])] for request in requests)
    local_rate, vllm_rate = (local["median_output_tokens_per_s"],
                             vllm["median_output_tokens_per_s"])
    report = {"status": "complete", "shape_id": shape_id, "scenario": "mixed",
              "local_repository_commit": local["repository_commit"],
              "vllm_repository_commit": vllm["repository_commit"],
              "first_wave": first, "arrival_step": arrival,
              "local_output_tokens_per_s": local_rate,
              "vllm_output_tokens_per_s": vllm_rate,
              "local_over_vllm": local_rate / vllm_rate,
              "local_mixed_steps": local["runs"][0]["mixed_steps"],
              "local_packed_mixed_calls": local["runs"][0]["packed_mixed_calls"],
              "exact_output_requests": exact, "total_requests": len(requests),
              "note": "step-index second-wave arrival; full drained-workload timing"}
    atomic_json(report_path, report)
    print(f"{shape_id} mixed: local {local_rate:.1f}, vLLM {vllm_rate:.1f} tok/s; "
          f"local/vLLM {local_rate / vllm_rate:.3f}x; "
          f"exact outputs {exact}/{len(requests)}", flush=True)
    return report


def forward(args, action, shape_id):
    interpreter = (getattr(args, "vllm_python", None) or sys.executable) if action == "run-vllm" else sys.executable
    command = [interpreter, str(Path(__file__)), action,
            "--suite-dir", str(args.suite_dir), "--output-dir", str(args.output_dir),
            "--shape-id", shape_id, "--model", args.model, "--device", args.device,
            "--seed", str(args.seed), "--warmups", str(args.warmups),
            "--repetitions", str(args.repetitions)]
    command += resume_options(args.resume_commit)
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run-cell", "run-local",
                                           "run-vllm", "analyze"))
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shape-id", choices=SHAPES, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--vllm-python")
    parser.add_argument("--resume-commit", action="append")
    args = parser.parse_args()
    if any(len(prefix) < 7 or any(char not in "0123456789abcdef"
                                   for char in prefix.lower())
           for prefix in (args.resume_commit or [])):
        raise ValueError("--resume-commit must be at least seven hexadecimal characters")
    if args.device != "cuda:0" or args.warmups < 1 or args.repetitions < 1:
        raise ValueError("requires cuda:0, >=1 warmup, and >=1 repetition")
    if args.action != "run-vllm":
        case, _, first, arrival, buckets, _, _ = mixed_plan(args, args.shape_id)
        options = adapter_options(case, buckets)
        validate_capture_options(options)
    if args.action == "plan":
        print(json.dumps({"shape_id": args.shape_id, "batch": case["max_running"],
                          "prompt_length": case["lengths"][0],
                          "output_length": case["outputs"][0], "first_wave": first,
                          "second_wave": case["max_running"] - first,
                          "arrival_step": arrival, "prefill_buckets": buckets,
                          "max_capture_tokens": options["max_capture_tokens"],
                          "token_budget": PREFILL_TOKENS_PER_STEP}, indent=2))
        return
    from benchmark_latest_vs_vllm import resolve_model_source
    model_source = resolve_model_source(args)
    if args.action == "run-local":
        run_local(args, args.shape_id, model_source)
    elif args.action == "run-vllm":
        run_vllm(args, args.shape_id, model_source)
    elif args.action == "analyze":
        analyze(args, args.shape_id, model_source)
    else:
        subprocess.run(forward(args, "run-local", args.shape_id), cwd=ROOT, check=True)
        subprocess.run(forward(args, "run-vllm", args.shape_id), cwd=ROOT, check=True)
        analyze(args, args.shape_id, model_source)


if __name__ == "__main__":
    main()
