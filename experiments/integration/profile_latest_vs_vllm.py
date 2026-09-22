#!/usr/bin/env python3
"""Matched one-step decode profiles for the latest local engine and vLLM 0.10.2.

Both arms consume the same frozen checkpoint workload.  Each is advanced to the
same zero-based occurrence of a pure, full-cohort decode step.  Model processes
are separate so their GPU allocations cannot contaminate each other.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for directory in (HERE, ROOT / "benchmarks", ROOT / "baseline"):
    sys.path.insert(0, str(directory))

from benchmark_backends import _matched_kv_cache_bytes
from benchmark_latest_vs_vllm import (MODEL, contract, load_frozen,
                                      resolve_model_source)
from fixed_regime import FIXED_SHAPES, PREFILL_TOKENS_PER_STEP, get_fixed_shape, shape_summary
from profile_cpp_control import cuda_kernel_category

PINNED_VLLM = "0.10.2"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check-setup", "run", "run-local", "run-vllm", "analyze"))
    parser.add_argument("--shape-id", choices=tuple(row["id"] for row in FIXED_SHAPES),
                        default="fixed-b8-l2048-o128")
    parser.add_argument("--suite-dir", type=Path, required=True,
                        help="existing full-checkpoint suite containing <shape>/workload.json")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--occurrence", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--prefill-budget", type=int,
                        help="local piecewise bucket; defaults to 4096 for B8 and 8192 for B64")
    parser.add_argument("--local-policy", choices=("splitk", "fa3"), default="fa3",
                        help="local decode attention implementation (default: current FA3 path)")
    parser.add_argument("--qkv-mode",
                        choices=("none", "native-k", "native", "packed", "full"),
                        default="native",
                        help="local decode QKV/RoPE/KV-write fusion arm")
    parser.add_argument("--enable-residual-rmsnorm", action="store_true",
                        help="enable local fused residual-add plus RMSNorm")
    parser.add_argument("--retry-failed", action="store_true")
    return parser


def validate_args(args):
    if args.device != "cuda:0":
        raise ValueError("matched profiler is pinned to cuda:0")
    if args.occurrence < 0 or args.warmups < 0 or args.repetitions < 1:
        raise ValueError("occurrence/warmups must be nonnegative and repetitions positive")
    shape = get_fixed_shape(args.shape_id)
    budget = args.prefill_budget or (4096 if shape["batch"] == 8 else 8192)
    if budget < 1:
        raise ValueError("prefill budget must be positive")
    args.prefill_budget = budget
    args.suite_dir = args.suite_dir.resolve()
    if args.output_dir is None:
        args.output_dir = ROOT / "experiments/results/matched-decode-profile" / args.shape_id
    args.output_dir = args.output_dir.resolve()
    args.workload_in = args.suite_dir / args.shape_id / "workload.json"
    # Reuse the checkpoint's exact validation rather than accepting a lookalike workload.
    checkpoint_args = argparse.Namespace(
        shape_id=args.shape_id, output_dir=args.suite_dir / args.shape_id,
        seed=args.seed, model=args.model, device=args.device, logit_atol=.05,
        trials=1, samples=1, repetitions=1, warmups=0, profile_occurrence=0)
    case, blocks = contract(checkpoint_args)
    load_frozen(checkpoint_args, case)
    return case, blocks


def verify_external_setup(args):
    try:
        version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError as error:
        raise ValueError("vLLM is not installed in this interpreter") from error
    if version != PINNED_VLLM:
        raise ValueError(f"requires vLLM {PINNED_VLLM}, found {version}")
    model = resolve_model_source(args)
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RequestOutputKind
    if not all(hasattr(LLM, name) for name in ("start_profile", "stop_profile")):
        raise ValueError("installed vLLM LLM API lacks worker profiling controls")
    SamplingParams(max_tokens=1, temperature=0.0,
                   output_kind=RequestOutputKind.CUMULATIVE)
    local_components = []
    if args.local_policy == "fa3":
        from kernel_dispatch import _load
        _load("paged_decode_fa3")._load_fa3()
        local_components.append("vllm_bundled_fa3")
    qkv_module = {
        "native-k": "rope_kv_write", "native": "rope_kv_write",
        "packed": "packed_qkv_rope_cache", "full": "fused_qkv_rope_cache",
    }.get(args.qkv_mode)
    if qkv_module is not None:
        from kernel_dispatch import _load
        _load(qkv_module)
        local_components.append(qkv_module)
    return {"vllm_version": version, "model": model,
            "profile_api": "LLM.start_profile/stop_profile",
            "local_components": local_components}


def json_files(directory):
    return sorted(directory.glob("*-report.json"))


def complete_local(directory, *, policy=None, qkv_mode=None, residual_rmsnorm=None):
    files = json_files(directory)
    if len(files) != 1:
        return False
    report = json.loads(files[0].read_text())
    complete = (report.get("schema_version") == 1
                and Path(report.get("trace", "missing")).is_file())
    if policy is not None:
        complete = complete and report.get("decode_attention_policy") == policy
    if qkv_mode is not None:
        complete = complete and report.get("qkv_mode") == qkv_mode
    if residual_rmsnorm is not None:
        complete = (complete and
                    report.get("enable_residual_rmsnorm") is residual_rmsnorm)
    return complete


def complete_vllm(directory):
    path = directory / "report.json"
    if not path.is_file():
        return False
    report = json.loads(path.read_text())
    return (report.get("status") == "complete"
            and Path(report.get("trace", "missing")).is_file())


def prepare_destination(directory, complete, retry_failed):
    if complete(directory):
        return False
    if directory.exists() and any(directory.iterdir()):
        if not retry_failed:
            raise ValueError(f"partial output at {directory}; rerun with --retry-failed")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        archive = directory.with_name(f"{directory.name}-failed-{stamp}")
        directory.rename(archive)
        print(f"archived interrupted output: {archive}", flush=True)
    directory.mkdir(parents=True, exist_ok=True)
    return True


def run_local(args):
    output = args.output_dir / "local"
    expected = lambda directory: complete_local(
        directory, policy=args.local_policy, qkv_mode=args.qkv_mode,
        residual_rmsnorm=args.enable_residual_rmsnorm)
    if not prepare_destination(output, expected, args.retry_failed):
        print("local profile already complete; skipping", flush=True)
        return
    model = resolve_model_source(args)
    command = [sys.executable, str(HERE / "profile_cpp_control.py"),
               "--preset", "fixed", "--case-id", args.shape_id,
               "--kind", "decode", "--occurrence", str(args.occurrence),
               "--adapter", "piecewise-prefill",
               "--decode-attention-policy", args.local_policy,
               "--qkv-mode", args.qkv_mode,
               "--prefill-budget", str(args.prefill_budget),
               "--model", model, "--device", args.device, "--seed", str(args.seed),
               "--workload-in", str(args.workload_in),
               "--warmups", str(args.warmups), "--repetitions", str(args.repetitions),
               "--output-dir", str(output)]
    if args.enable_residual_rmsnorm:
        command.append("--enable-residual-rmsnorm")
    subprocess.run(command, cwd=ROOT, check=True)


def add_vllm_requests(llm, requests, run_id, *, cumulative):
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    expected = {}
    for request in requests:
        request_id = f"{run_id}:{request['id']}"
        params = SamplingParams(temperature=0.0, max_tokens=request["output"],
                                ignore_eos=True, detokenize=False,
                                output_kind=(RequestOutputKind.CUMULATIVE if cumulative
                                             else RequestOutputKind.FINAL_ONLY))
        llm.llm_engine.add_request(request_id, {"prompt_token_ids": request["prompt"]}, params)
        expected[request_id] = request["output"]
    return expected


def discover_vllm_target(llm, requests, occurrence, run_id):
    """Locate a full-cohort step; this untimed pass may emit cumulative outputs."""
    expected = add_vllm_requests(llm, requests, run_id, cumulative=True)
    prompt_lengths = {f"{run_id}:{request['id']}": len(request["prompt"])
                      for request in requests}
    progress = {request_id: 0 for request_id in expected}
    finished = set()
    full_decode_index = 0
    target_before = None
    target_after = None
    target_step_index = None
    steps = 0
    limit = sum(len(row["prompt"]) + row["output"] for row in requests) + 1
    while llm.llm_engine.has_unfinished_requests():
        eligible = (not finished and all(progress[key] >= 1 for key in expected)
                    and all(progress[key] < expected[key] for key in expected))
        is_target = eligible and full_decode_index == occurrence
        if is_target:
            target_before = dict(progress)
            target_step_index = steps
            outputs = llm.llm_engine.step()
        else:
            outputs = llm.llm_engine.step()
        for output in outputs:
            request_id = str(output.request_id)
            if request_id not in progress:
                raise AssertionError(f"unexpected vLLM request ID {request_id}")
            if output.outputs:
                progress[request_id] = len(output.outputs[0].token_ids)
            if output.finished:
                finished.add(request_id)
        if eligible:
            if is_target and any(progress[key] != previous + 1
                                 for key, previous in target_before.items()):
                raise AssertionError("selected vLLM step did not advance every request by one token")
            if is_target:
                target_after = dict(progress)
            full_decode_index += 1
        steps += 1
        if steps > limit:
            raise RuntimeError("vLLM iteration limit exceeded")
    if target_step_index is None:
        raise ValueError(f"vLLM exposed only {full_decode_index} pure full-cohort decode steps; "
                         f"cannot select occurrence {occurrence}")
    if progress != expected or finished != set(expected):
        raise AssertionError("vLLM did not produce every requested output token")
    target_context = {key: prompt_lengths[key] + target_before[key] for key in expected}
    return {"target_engine_step_index": target_step_index,
            "target_progress_before": target_before,
            "target_progress_after": target_after, "pure_full_decode_steps": full_decode_index,
            "target_context_lengths": target_context, "total_engine_steps": steps}


def measure_vllm_target(llm, requests, target_step_index, run_id, *, profile=False):
    """Measure a discovered step with the benchmark's FINAL_ONLY output mode."""
    expected = add_vllm_requests(llm, requests, run_id, cumulative=False)
    completed = {}
    target_wall_ms = None
    steps = 0
    limit = sum(len(row["prompt"]) + row["output"] for row in requests) + 1
    while llm.llm_engine.has_unfinished_requests():
        is_target = steps == target_step_index
        if is_target and profile:
            llm.start_profile()
        if is_target:
            started = time.perf_counter()
        outputs = llm.llm_engine.step()
        if is_target:
            target_wall_ms = (time.perf_counter() - started) * 1000
        if is_target and profile:
            llm.stop_profile()
        for output in outputs:
            request_id = str(output.request_id)
            if request_id not in expected:
                raise AssertionError(f"unexpected vLLM request ID {request_id}")
            if output.finished:
                completed[request_id] = len(output.outputs[0].token_ids)
        steps += 1
        if steps > limit:
            raise RuntimeError("vLLM iteration limit exceeded")
    if target_wall_ms is None:
        raise ValueError(f"vLLM replay ended before discovered engine step {target_step_index}")
    if completed != expected:
        raise AssertionError("vLLM FINAL_ONLY replay did not produce every requested token")
    return {"target_wall_ms": target_wall_ms, "total_engine_steps": steps}


def open_trace(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        return json.load(stream)


def summarize_chrome_trace(path):
    payload = open_trace(path)
    rows = {}
    for event in payload.get("traceEvents", []):
        category = str(event.get("cat", "")).lower()
        duration = event.get("dur")
        if event.get("ph") != "X" or not isinstance(duration, (int, float)) or duration <= 0:
            continue
        if not any(token in category for token in ("kernel", "gpu_memcpy", "gpu_memset")):
            continue
        name = str(event.get("name", "unknown"))
        row = rows.setdefault(name, {"name": name, "category": cuda_kernel_category(name),
                                     "calls": 0, "total_us": 0.0})
        row["calls"] += 1
        row["total_us"] += float(duration)
    kernels = sorted(rows.values(), key=lambda row: row["total_us"], reverse=True)
    total = sum(row["total_us"] for row in kernels)
    categories = {}
    for row in kernels:
        row["mean_us"] = row["total_us"] / row["calls"]
        row["percent_of_cuda_activity"] = 100 * row["total_us"] / total if total else 0.0
        aggregate = categories.setdefault(row["category"], {"category": row["category"],
                                                             "calls": 0, "total_us": 0.0})
        aggregate["calls"] += row["calls"]
        aggregate["total_us"] += row["total_us"]
    category_rows = sorted(categories.values(), key=lambda row: row["total_us"], reverse=True)
    for row in category_rows:
        row["percent_of_cuda_activity"] = 100 * row["total_us"] / total if total else 0.0
    if not kernels:
        raise ValueError(f"no CUDA kernel activities found in {path}")
    return {"summed_cuda_activity_us": total,
            "activity_count": sum(row["calls"] for row in kernels),
            "categories": category_rows, "kernels": kernels}


def run_vllm(args, case, num_blocks):
    output = args.output_dir / "vllm"
    if not prepare_destination(output, complete_vllm, args.retry_failed):
        print("vLLM profile already complete; skipping", flush=True)
        return
    version = importlib.metadata.version("vllm")
    if version != PINNED_VLLM:
        raise ValueError(f"requires vLLM {PINNED_VLLM}, found {version}")
    trace_dir = output / "trace"
    trace_dir.mkdir()
    os.environ["VLLM_TORCH_PROFILER_DIR"] = str(trace_dir)
    os.environ["VLLM_TORCH_PROFILER_WITH_STACK"] = "0"
    model = resolve_model_source(args)
    from vllm import LLM

    max_model_len = shape_summary(get_fixed_shape(args.shape_id))["max_context_tokens_per_request"]
    kv_bytes = _matched_kv_cache_bytes(model, dtype="float16", block_size=16,
                                       num_blocks=num_blocks)
    started = time.perf_counter()
    llm = LLM(model=model, dtype="float16", seed=args.seed,
              max_num_seqs=case["max_running"],
              max_num_batched_tokens=PREFILL_TOKENS_PER_STEP,
              max_model_len=max_model_len, block_size=16,
              enable_prefix_caching=False, enable_chunked_prefill=True,
              generation_config="vllm", enforce_eager=False,
              kv_cache_memory_bytes=kv_bytes)
    load_seconds = time.perf_counter() - started
    _, requests = load_frozen(argparse.Namespace(
        output_dir=args.suite_dir / args.shape_id, shape_id=args.shape_id,
        seed=args.seed, model=args.model, device=args.device,
        logit_atol=.05, trials=1, samples=1, repetitions=1,
        warmups=0, profile_occurrence=0), case)
    discovery = discover_vllm_target(llm, requests, args.occurrence, "discovery")
    target_step = discovery["target_engine_step_index"]
    for index in range(args.warmups):
        warmup = measure_vllm_target(llm, requests, target_step, f"warmup-{index}")
        if warmup["total_engine_steps"] != discovery["total_engine_steps"]:
            raise AssertionError("vLLM schedule changed between discovery and warmup")
    baselines = [measure_vllm_target(llm, requests, target_step, f"baseline-{index}")
                 for index in range(args.repetitions)]
    if any(row["total_engine_steps"] != discovery["total_engine_steps"] for row in baselines):
        raise AssertionError("vLLM schedule changed between discovery and timing")
    before = set(trace_dir.rglob("*"))
    traced = measure_vllm_target(llm, requests, target_step, "profile", profile=True)
    if traced["total_engine_steps"] != discovery["total_engine_steps"]:
        raise AssertionError("vLLM schedule changed during profiling")
    candidates = [path for path in trace_dir.rglob("*") if path.is_file() and path not in before
                  and (path.name.endswith(".json") or path.name.endswith(".json.gz"))]
    if len(candidates) != 1:
        raise ValueError(f"expected one new vLLM trace, found {[str(path) for path in candidates]}")
    cuda = summarize_chrome_trace(candidates[0])
    report = {
        "schema_version": 1, "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(), "backend": "vllm",
        "vllm_version": version, "shape_id": args.shape_id, "model": model,
        "occurrence": args.occurrence, "prefill_token_budget": PREFILL_TOKENS_PER_STEP,
        "max_num_seqs": case["max_running"], "max_model_len": max_model_len,
        "matched_num_blocks": num_blocks, "kv_cache_memory_bytes": kv_bytes,
        "unprofiled_target_wall_ms": [row["target_wall_ms"] for row in baselines],
        "unprofiled_median_wall_ms": statistics.median(
            row["target_wall_ms"] for row in baselines),
        "profiled_target_wall_ms": traced["target_wall_ms"],
        "target_engine_step_index": target_step,
        "target_progress_before": discovery["target_progress_before"],
        "target_progress_after": discovery["target_progress_after"],
        "target_context_lengths": discovery["target_context_lengths"],
        "pure_full_decode_steps": discovery["pure_full_decode_steps"],
        "total_engine_steps": traced["total_engine_steps"],
        "timed_output_kind": "FINAL_ONLY",
        "cuda_activity": cuda, "trace": str(candidates[0]),
        "model_load_seconds": load_seconds,
        "engine_class": f"{type(llm.llm_engine).__module__}.{type(llm.llm_engine).__name__}",
        "vllm_use_v1_environment": os.environ.get("VLLM_USE_V1"),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"vLLM decode occurrence {args.occurrence}: "
          f"{report['unprofiled_median_wall_ms']:.3f} ms", flush=True)


def context_summary(values):
    values = sorted(values)
    return {"minimum": values[0], "median": statistics.median(values),
            "maximum": values[-1], "values": values}


def compare_categories(local, vllm):
    left = {row["category"]: row for row in local["categories"]}
    right = {row["category"]: row for row in vllm["categories"]}
    rows = []
    for category in sorted(set(left) | set(right)):
        local_row = left.get(category, {})
        vllm_row = right.get(category, {})
        local_us = float(local_row.get("total_us", 0.0))
        vllm_us = float(vllm_row.get("total_us", 0.0))
        rows.append({
            "category": category, "local_us": local_us, "vllm_us": vllm_us,
            "local_percent": float(local_row.get("percent_of_cuda_activity", 0.0)),
            "vllm_percent": float(vllm_row.get("percent_of_cuda_activity", 0.0)),
            "local_over_vllm": local_us / vllm_us if vllm_us else None,
        })
    return sorted(rows, key=lambda row: row["local_us"], reverse=True)


def analyze(args, num_blocks):
    local_files = json_files(args.output_dir / "local")
    if len(local_files) != 1 or not complete_vllm(args.output_dir / "vllm"):
        raise ValueError("both local and vLLM reports must be complete")
    local = json.loads(local_files[0].read_text())
    vllm = json.loads((args.output_dir / "vllm/report.json").read_text())
    if local["case"]["id"] != args.shape_id or vllm["shape_id"] != args.shape_id:
        raise ValueError("profile reports do not match requested shape")
    expected_call = [[True, vllm["max_num_seqs"], vllm["max_num_seqs"], 1]]
    if local["target_step_calls"] != expected_call:
        raise ValueError("local target was not exactly one pure full-cohort decode call")
    local_callbacks = local.get("target_callback_calls")
    if (not local_callbacks or len(local_callbacks) != 1
            or len(local_callbacks[0]["context_lengths"]) != vllm["max_num_seqs"]):
        raise ValueError("local target context metadata is incomplete")
    if (local.get("occurrence") != args.occurrence
            or vllm["occurrence"] != args.occurrence
            or local["kind"] != "decode"
            or local["adapter"] != "piecewise-prefill"
            or local["decode_attention_policy"] != args.local_policy
            or local.get("qkv_mode") != args.qkv_mode
            or local.get("enable_residual_rmsnorm") is not args.enable_residual_rmsnorm
            or local["prefill_budget"] != args.prefill_budget
            or vllm["prefill_token_budget"] != PREFILL_TOKENS_PER_STEP
            or vllm["matched_num_blocks"] != num_blocks
            or vllm["vllm_version"] != PINNED_VLLM
            or vllm["timed_output_kind"] != "FINAL_ONLY"
            or local["model"] != vllm["model"]):
        raise ValueError("profile execution settings do not match the requested contract")
    local_contexts = local_callbacks[0]["context_lengths"]
    vllm_contexts = list(vllm["target_context_lengths"].values())
    category_comparison = compare_categories(local["cuda_activity"],
                                             vllm["cuda_activity"])
    summary = {
        "schema_version": 1, "status": "complete", "shape_id": args.shape_id,
        "occurrence": args.occurrence,
        "local_policy": args.local_policy, "qkv_mode": args.qkv_mode,
        "enable_residual_rmsnorm": args.enable_residual_rmsnorm,
        "local": {"median_wall_ms": local["unprofiled_median_wall_ms"],
                  "cuda_activity": local["cuda_activity"],
                  "target_callback_calls": local_callbacks,
                  "context_lengths": context_summary(local_contexts)},
        "vllm": {"median_wall_ms": vllm["unprofiled_median_wall_ms"],
                 "cuda_activity": vllm["cuda_activity"],
                 "target_progress_before": vllm["target_progress_before"],
                 "context_lengths": context_summary(vllm_contexts)},
        "wall_time_ratio_local_over_vllm": (
            local["unprofiled_median_wall_ms"] / vllm["unprofiled_median_wall_ms"]),
        "summed_cuda_activity_ratio_local_over_vllm": (
            local["cuda_activity"]["summed_cuda_activity_us"]
            / vllm["cuda_activity"]["summed_cuda_activity_us"]),
        "category_comparison": category_comparison,
        "notes": [
            "Wall medians are uninstrumented; traced CUDA activity is diagnostic only.",
            "Both targets are pure full-cohort decode steps at the same occurrence, but each "
            "scheduler's earlier mixed prefill/decode history can produce different per-request contexts.",
        ],
    }
    path = args.output_dir / "comparison.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"local: {summary['local']['median_wall_ms']:.3f} ms", flush=True)
    print(f"vLLM:  {summary['vllm']['median_wall_ms']:.3f} ms", flush=True)
    print(f"local/vLLM: {summary['wall_time_ratio_local_over_vllm']:.3f}x", flush=True)
    print("\nCUDA activity category             local us    vLLM us   local/vLLM", flush=True)
    for row in category_comparison:
        ratio = (f"{row['local_over_vllm']:.3f}x"
                 if row["local_over_vllm"] is not None else "-")
        print(f"{row['category']:<32} {row['local_us']:>10.1f} "
              f"{row['vllm_us']:>10.1f} {ratio:>12}", flush=True)
    print(f"comparison: {path}", flush=True)


def run_all(args, case, blocks):
    # The subprocess boundary is deliberate: only one engine owns GPU memory at a time.
    setup = verify_external_setup(args)
    print(f"preflight complete: vLLM {setup['vllm_version']}, model {setup['model']}",
          flush=True)
    common = ["--shape-id", args.shape_id, "--suite-dir", str(args.suite_dir),
              "--output-dir", str(args.output_dir), "--model", args.model,
              "--device", args.device, "--seed", str(args.seed),
              "--occurrence", str(args.occurrence), "--warmups", str(args.warmups),
              "--repetitions", str(args.repetitions),
              "--prefill-budget", str(args.prefill_budget),
              "--local-policy", args.local_policy, "--qkv-mode", args.qkv_mode]
    if args.enable_residual_rmsnorm:
        common.append("--enable-residual-rmsnorm")
    if args.retry_failed:
        common.append("--retry-failed")
    for action in ("run-local", "run-vllm", "analyze"):
        subprocess.run([sys.executable, str(Path(__file__)), action, *common], cwd=ROOT, check=True)


def main():
    args = build_parser().parse_args()
    case, blocks = validate_args(args)
    if args.action == "check-setup":
        setup = verify_external_setup(args)
        result = {"workload": str(args.workload_in), "shape": shape_summary(get_fixed_shape(args.shape_id)),
                  "local_prefill_budget": args.prefill_budget,
                  "local_policy": args.local_policy,
                  "local_qkv_mode": args.qkv_mode,
                  "local_residual_rmsnorm": args.enable_residual_rmsnorm,
                  "vllm_prefill_budget": PREFILL_TOKENS_PER_STEP,
                  "vllm_installed": setup["vllm_version"],
                  "vllm_required": PINNED_VLLM,
                  "local_components": setup["local_components"],
                  "model": setup["model"], "status": "ready"}
        print(json.dumps(result, indent=2))
    elif args.action == "run-local":
        run_local(args)
    elif args.action == "run-vllm":
        run_vllm(args, case, blocks)
    elif args.action == "analyze":
        analyze(args, blocks)
    else:
        run_all(args, case, blocks)


if __name__ == "__main__":
    main()
