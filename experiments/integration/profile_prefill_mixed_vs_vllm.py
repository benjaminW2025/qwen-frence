#!/usr/bin/env python3
"""Matched one-step pure-prefill or mixed local/vLLM comparison.

Use the same frozen prompt IDs, two-wave arrival plan, model snapshot, token
budget, and KV reservation. A mixed target is one C++ decode callback followed
by one piecewise prefill callback; vLLM may implement it as one model pass.
Profiles are diagnostic; wall medians come from unprofiled replays.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for directory in (HERE, ROOT / "benchmarks", ROOT / "baseline",
                  ROOT / "engine/model_runner", ROOT / "engine/kvcache",
                  ROOT / "engine/cpp/build"):
    sys.path.insert(0, str(directory))

from benchmark_backends import _matched_kv_cache_bytes
from benchmark_latest_vs_vllm import MODEL, load_frozen, resolve_model_source
from benchmark_scheduler_decode import execute, make_config
from fixed_regime import get_fixed_case, get_fixed_shape, shape_summary
from profile_cpp_control import drive
from profile_latest_vs_vllm import (PINNED_VLLM, add_vllm_requests,
                                    prepare_destination, summarize_chrome_trace,
                                    verify_external_setup)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "check-setup", "run", "run-local",
                                           "run-vllm", "analyze"))
    parser.add_argument("--kind", choices=("prefill", "mixed"), required=True)
    parser.add_argument("--shape-id", choices=("fixed-b8-l256-o128", "fixed-b64-l256-o128"),
                        default="fixed-b8-l256-o128")
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--arrival-step", type=int, default=5,
                        help="mixed second-wave injection step, after pure decode has begun")
    parser.add_argument("--output-tokens", type=int, default=16,
                        help="enough generated tokens to keep the first wave active")
    parser.add_argument("--prefill-budget", type=int,
                        help="shared local/vLLM token budget: B8=2048, B64=8192")
    parser.add_argument("--local-policy", choices=("fa3", "splitk"), default="fa3")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--retry-failed", action="store_true")
    return parser


def validate(args):
    if args.device != "cuda:0":
        raise ValueError("comparison requires cuda:0")
    if args.arrival_step < 2 or args.output_tokens <= args.arrival_step + 1:
        raise ValueError("first wave must still decode after the second wave arrives")
    if args.warmups < 0 or args.repetitions < 1:
        raise ValueError("warmups must be nonnegative and repetitions positive")
    shape = get_fixed_shape(args.shape_id)
    args.prefill_budget = args.prefill_budget or (2048 if shape["batch"] == 8 else 8192)
    if args.prefill_budget < 1:
        raise ValueError("prefill budget must be positive")
    # The shared setup preflight checks the same accepted decode QKV path used
    # in the local mixed arm. It is inactive in a pure-prefill target step.
    args.qkv_mode = "native"
    args.suite_dir = args.suite_dir.resolve()
    args.output_dir = (args.output_dir or ROOT / "experiments/results/matched-phase-profile"
                       / args.kind / args.shape_id).resolve()
    frozen_case = get_fixed_case(args.shape_id)
    frozen_args = argparse.Namespace(
        output_dir=args.suite_dir / args.shape_id, shape_id=args.shape_id,
        seed=args.seed, model=args.model, device=args.device, logit_atol=.05,
        trials=1, samples=1, repetitions=1, warmups=0, profile_occurrence=0)
    _, frozen = load_frozen(frozen_args, frozen_case)
    # B64 pure prefill uses exactly one 8192-token half-batch; mixed uses two
    # 4096-token quarter-batches, leaving budget for concurrent decode tokens.
    count = (shape["batch"] if shape["batch"] == 8 else shape["batch"] // 2)
    if args.kind == "mixed":
        first = second = shape["batch"] // 2 if shape["batch"] == 8 else shape["batch"] // 4
        count = first + second
    else:
        first, second = count, 0
    prompt = shape["prompt_length"]
    if max(first, second) * prompt + (first if second else 0) > args.prefill_budget:
        raise ValueError("selected cohort cannot fit in one matched prefill step")
    requests = [dict(id=index, prompt=frozen[index]["prompt"], output=args.output_tokens,
                     arrival=args.arrival_step if index >= first else 0)
                for index in range(count)]
    case = {**frozen_case, "id": f"phase-{args.kind}-{args.shape_id}",
            "lengths": [prompt] * count, "outputs": [args.output_tokens] * count,
            "arrivals": [row["arrival"] for row in requests],
            "prefill_budget": args.prefill_budget}
    target_index = 0 if args.kind == "prefill" else args.arrival_step
    return case, requests, first, second, target_index


def plan(args, case, first, second, target_index):
    return {"kind": args.kind, "shape_id": args.shape_id,
            "first_wave": first, "second_wave": second,
            "prompt_length": case["lengths"][0], "output_tokens": args.output_tokens,
            "target_engine_step": target_index,
            "target_decode_tokens": first if second else 0,
            "target_prefill_tokens": (second or first) * case["lengths"][0],
            "shared_prefill_budget": args.prefill_budget,
            "local_model_passes": 2 if second else 1,
            "note": "fails before reporting a timing if either scheduler misses target work"}


def expected_local_calls(case, first, second):
    prompt = case["lengths"][0]
    prefill = (False, (second or first) * prompt, second or first, prompt)
    return [(True, first, first, 1), prefill] if second else [prefill]


def complete_local(directory, args):
    reports = list(directory.glob("*-report.json"))
    if len(reports) != 1:
        return False
    report = json.loads(reports[0].read_text())
    trace = Path(report.get("trace", "missing"))
    return (report.get("kind") == args.kind and
            report.get("shape_id") == args.shape_id and
            report.get("prefill_budget") == args.prefill_budget and
            report.get("arrival_step") == args.arrival_step and
            report.get("output_tokens") == args.output_tokens and
            report.get("local_policy") == args.local_policy and
            (trace.is_file() or len(list(directory.rglob(trace.name))) == 1))


def complete_vllm(directory, args):
    path = directory / "report.json"
    if not path.is_file():
        return False
    report = json.loads(path.read_text())
    trace = Path(report.get("trace", "missing"))
    return (report.get("status") == "complete" and report.get("kind") == args.kind
            and report.get("shape_id") == args.shape_id
            and report.get("prefill_budget") == args.prefill_budget
            and report.get("arrival_step") == args.arrival_step
            and report.get("output_tokens") == args.output_tokens
            and (trace.is_file() or len(list(directory.rglob(trace.name))) == 1))


def run_local(args, case, requests, first, second, target_index):
    output = args.output_dir / "local"
    if not prepare_destination(output, lambda path: complete_local(path, args),
                               args.retry_failed):
        print("local already complete; skipping", flush=True)
        return
    from model_adapter import PiecewiseGraphModelAdapter, allocate_pool
    from model_setup import check_startup, load_model_only
    import torch
    import inference_engine_cpp as cpp

    setup = check_startup(args.device)
    model_source = resolve_model_source(args)
    engine, load_seconds, _ = load_model_only(
        model_source, args.device, "float16", hub_transfer=setup["hub_transfer"])
    config = make_config(cpp, case)
    blocks = config.max_batch_size * ((config.max_context_length + 15) // 16 + 1)
    pool = allocate_pool(engine.cfg, blocks, engine.device)
    adapter = PiecewiseGraphModelAdapter(
        engine.model, pool, None, max_running=config.max_batch_size,
        max_context_length=config.max_context_length,
        decode_attention_policy=args.local_policy,
        max_capture_tokens=args.prefill_budget, max_prefill_shapes=1,
        prefill_buckets=[args.prefill_budget],
        enable_residual_rmsnorm=True,
        enable_native_decode_qkv_postprocess=True,
        enable_prefill_swiglu_fusion=True)

    def poison():
        for tensor in pool.k_pool + pool.v_pool:
            tensor.fill_(float("nan"))

    poison()
    preflight = execute(torch, cpp.IterationLoop(config, torch.device(args.device)),
                        adapter, requests)
    expected_steps = [(row["kind"], row["calls"], row["completed"])
                      for row in preflight["steps"]]
    if target_index >= len(expected_steps):
        raise AssertionError("local workload ended before target step")
    target = expected_steps[target_index]
    calls = expected_local_calls(case, first, second)
    if target[0] != args.kind or [tuple(row) for row in target[1]] != calls:
        raise AssertionError(f"local target work differs: {target}; expected {calls}")
    output_hash = hashlib.sha256(json.dumps(
        preflight["outputs"], sort_keys=True).encode()).hexdigest()

    def checked_run(*, trace=False):
        poison()
        row = drive(torch, cpp, config, requests, adapter, target_index,
                    profile_target=trace, collect_target_metadata=trace)
        if row["steps"] != expected_steps or row["outputs"] != preflight["outputs"]:
            raise AssertionError("local schedule or output changed between runs")
        return row

    for _ in range(args.warmups):
        checked_run()
    baseline = [checked_run()["target_wall_ms"] for _ in range(args.repetitions)]
    traced = checked_run(trace=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    trace_path = output / f"{args.kind}-{stamp}-trace.json"
    traced["profiler"].export_chrome_trace(str(trace_path))
    report = {"status": "complete", "kind": args.kind, "model": model_source,
              "shape_id": args.shape_id, "prefill_budget": args.prefill_budget,
              "arrival_step": args.arrival_step, "output_tokens": args.output_tokens,
              "local_policy": args.local_policy, "first_wave": first,
              "second_wave": second, "target_step_index": target_index,
              "target_step_calls": target[1], "target_callback_calls": traced["target_calls"],
              "outputs_sha256": output_hash, "model_load_seconds": load_seconds,
              "unprofiled_target_wall_ms": baseline,
              "unprofiled_median_wall_ms": statistics.median(baseline),
              "trace": str(trace_path)}
    (output / f"{args.kind}-{stamp}-report.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(f"local {args.kind}: {report['unprofiled_median_wall_ms']:.3f} ms", flush=True)


def vllm_run(llm, requests, first, target_index, run_id, *, cumulative=False,
             profile=False):
    first_requests = requests[:first]
    second_requests = requests[first:]
    expected = add_vllm_requests(llm, first_requests, run_id, cumulative=cumulative)
    progress = {key: 0 for key in expected}
    target_before = target_after = None
    target_wall_ms = None
    total_steps = 0
    limit = sum(len(row["prompt"]) + row["output"] for row in requests) + 1
    while llm.llm_engine.has_unfinished_requests() or (
            second_requests and total_steps <= target_index):
        if second_requests and total_steps == target_index:
            expected.update(add_vllm_requests(llm, second_requests, run_id,
                                              cumulative=cumulative))
            progress.update({key: 0 for key in expected if key not in progress})
        is_target = total_steps == target_index
        if is_target:
            target_before = dict(progress)
            if profile:
                llm.start_profile()
            started = time.perf_counter()
        outputs = llm.llm_engine.step()
        if is_target:
            target_wall_ms = (time.perf_counter() - started) * 1000
            if profile:
                llm.stop_profile()
        for row in outputs:
            key = str(row.request_id)
            if key not in progress:
                raise AssertionError(f"unexpected vLLM request {key}")
            if cumulative and row.outputs:
                progress[key] = len(row.outputs[0].token_ids)
            if row.finished:
                if len(row.outputs[0].token_ids) != expected[key]:
                    raise AssertionError("vLLM output length differs from requested length")
                progress[key] = expected[key]
        if is_target:
            target_after = dict(progress)
        total_steps += 1
        if total_steps > limit:
            raise RuntimeError("vLLM exceeded step limit")
    if target_wall_ms is None or set(progress) != set(expected):
        raise AssertionError("vLLM did not reach the selected target")
    if any(progress[key] != length for key, length in expected.items()):
        raise AssertionError("vLLM did not complete all requests")
    return {"target_wall_ms": target_wall_ms, "target_progress_before": target_before,
            "target_progress_after": target_after, "total_steps": total_steps}


def verify_vllm_target(discovery, first, second, kind):
    before = discovery["target_progress_before"]
    after = discovery["target_progress_after"]
    keys = list(before)
    if kind == "prefill":
        valid = len(keys) == first and all(before[key] == 0 and after[key] == 1
                                           for key in keys)
    else:
        valid = (len(keys) == first + second and
                 all(before[key] >= 1 and after[key] == before[key] + 1
                     for key in keys[:first]) and
                 all(before[key] == 0 and after[key] == 1
                     for key in keys[first:]))
    if not valid:
        raise AssertionError(f"vLLM target did not execute {kind} cohort: "
                             f"before={before}, after={after}")


def run_vllm(args, case, requests, first, second, target_index):
    output = args.output_dir / "vllm"
    if not prepare_destination(output, lambda path: complete_vllm(path, args),
                               args.retry_failed):
        print("vLLM already complete; skipping", flush=True)
        return
    setup = verify_external_setup(args)
    trace_dir = output / "trace"
    trace_dir.mkdir()
    import os
    os.environ["VLLM_TORCH_PROFILER_DIR"] = str(trace_dir)
    os.environ["VLLM_TORCH_PROFILER_WITH_STACK"] = "0"
    from vllm import LLM

    model = setup["model"]
    blocks = case["max_running"] * ((max(case["lengths"]) + args.output_tokens + 15) // 16 + 1)
    kv_bytes = _matched_kv_cache_bytes(model, dtype="float16", block_size=16,
                                       num_blocks=blocks)
    llm = LLM(model=model, dtype="float16", seed=args.seed,
              max_num_seqs=case["max_running"],
              max_num_batched_tokens=args.prefill_budget,
              max_model_len=max(case["lengths"]) + args.output_tokens,
              block_size=16, enable_prefix_caching=False,
              enable_chunked_prefill=True, generation_config="vllm",
              enforce_eager=False, kv_cache_memory_bytes=kv_bytes)
    discovery = vllm_run(llm, requests, first, target_index, "discovery",
                         cumulative=True)
    verify_vllm_target(discovery, first, second, args.kind)
    for index in range(args.warmups):
        row = vllm_run(llm, requests, first, target_index, f"warmup-{index}")
        if row["total_steps"] != discovery["total_steps"]:
            raise AssertionError("vLLM schedule changed during warmup")
    baseline = [vllm_run(llm, requests, first, target_index, f"baseline-{index}")
                for index in range(args.repetitions)]
    if any(row["total_steps"] != discovery["total_steps"] for row in baseline):
        raise AssertionError("vLLM schedule changed during timing")
    before = set(trace_dir.rglob("*"))
    traced = vllm_run(llm, requests, first, target_index, "profile", profile=True)
    if traced["total_steps"] != discovery["total_steps"]:
        raise AssertionError("vLLM schedule changed during profiling")
    traces = [path for path in trace_dir.rglob("*") if path.is_file() and path not in before
              and (path.name.endswith(".json") or path.name.endswith(".json.gz"))]
    if len(traces) != 1:
        raise ValueError(f"expected one vLLM trace, found {traces}")
    summarize_chrome_trace(traces[0])  # reject empty or malformed traces before completion
    report = {"status": "complete", "kind": args.kind, "shape_id": args.shape_id,
              "model": model, "vllm_version": PINNED_VLLM,
              "prefill_budget": args.prefill_budget,
              "arrival_step": args.arrival_step, "output_tokens": args.output_tokens,
              "first_wave": first, "second_wave": second,
              "target_step_index": target_index,
              "target_progress_before": discovery["target_progress_before"],
              "target_progress_after": discovery["target_progress_after"],
              "total_steps": discovery["total_steps"],
              "matched_num_blocks": blocks, "kv_cache_memory_bytes": kv_bytes,
              "unprofiled_target_wall_ms": [row["target_wall_ms"] for row in baseline],
              "unprofiled_median_wall_ms": statistics.median(
                  row["target_wall_ms"] for row in baseline),
              "trace": str(traces[0])}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"vLLM {args.kind}: {report['unprofiled_median_wall_ms']:.3f} ms", flush=True)


def resolve_trace(recorded, directory):
    path = Path(recorded)
    if path.is_file():
        return path
    matches = list(directory.rglob(path.name))
    if len(matches) != 1:
        raise ValueError(f"cannot resolve trace {recorded}: {matches}")
    return matches[0]


def analyze(args, case, first, second, target_index):
    local_paths = list((args.output_dir / "local").glob("*-report.json"))
    if len(local_paths) != 1 or not complete_vllm(args.output_dir / "vllm", args):
        raise ValueError("both local and vLLM reports must be complete")
    local = json.loads(local_paths[0].read_text())
    vllm = json.loads((args.output_dir / "vllm/report.json").read_text())
    calls = expected_local_calls(case, first, second)
    if (not complete_local(args.output_dir / "local", args)
            or [tuple(row) for row in local["target_step_calls"]] != calls
            or local["target_step_index"] != target_index
            or vllm["target_step_index"] != target_index
            or local["model"] != vllm["model"]
            or local["first_wave"] != vllm["first_wave"]
            or local["second_wave"] != vllm["second_wave"]):
        raise ValueError("local and vLLM work/settings do not match")
    verify_vllm_target(vllm, first, second, args.kind)
    local_cuda = summarize_chrome_trace(resolve_trace(local["trace"], args.output_dir / "local"))
    vllm_cuda = summarize_chrome_trace(resolve_trace(vllm["trace"], args.output_dir / "vllm"))
    from profile_latest_vs_vllm import compare_categories
    ratio = local["unprofiled_median_wall_ms"] / vllm["unprofiled_median_wall_ms"]
    report = {"status": "complete", "kind": args.kind, "shape_id": args.shape_id,
              "work": plan(args, case, first, second, target_index),
              "local_wall_ms": local["unprofiled_median_wall_ms"],
              "vllm_wall_ms": vllm["unprofiled_median_wall_ms"],
              "local_over_vllm": ratio,
              "local_cuda_activity": local_cuda, "vllm_cuda_activity": vllm_cuda,
              "category_comparison": compare_categories(local_cuda, vllm_cuda),
              "notes": ["Wall medians are unprofiled; CUDA activity is diagnostic.",
                        "Mixed compares the same requests/arrivals/token budget, not an "
                        "identical model-pass decomposition."]}
    path = args.output_dir / "comparison.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{args.kind}: local {report['local_wall_ms']:.3f} ms; "
          f"vLLM {report['vllm_wall_ms']:.3f} ms; local/vLLM {ratio:.3f}x")
    print(f"comparison: {path}")


def main():
    args = build_parser().parse_args()
    case, requests, first, second, target_index = validate(args)
    if args.action == "plan":
        print(json.dumps(plan(args, case, first, second, target_index), indent=2))
    elif args.action == "check-setup":
        print(json.dumps(verify_external_setup(args), indent=2))
    elif args.action == "run-local":
        run_local(args, case, requests, first, second, target_index)
    elif args.action == "run-vllm":
        run_vllm(args, case, requests, first, second, target_index)
    elif args.action == "analyze":
        analyze(args, case, first, second, target_index)
    else:
        verify_external_setup(args)
        common = ["--kind", args.kind, "--shape-id", args.shape_id,
                  "--suite-dir", str(args.suite_dir), "--output-dir", str(args.output_dir),
                  "--model", args.model, "--device", args.device,
                  "--seed", str(args.seed), "--arrival-step", str(args.arrival_step),
                  "--output-tokens", str(args.output_tokens),
                  "--prefill-budget", str(args.prefill_budget),
                  "--local-policy", args.local_policy,
                  "--warmups", str(args.warmups),
                  "--repetitions", str(args.repetitions)]
        if args.retry_failed:
            common.append("--retry-failed")
        for action in ("run-local", "run-vllm", "analyze"):
            subprocess.run([sys.executable, str(Path(__file__)), action, *common],
                           cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
