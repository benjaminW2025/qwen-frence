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
import os
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


def configure_vllm_step_mode():
    """Make request admission wait for step(), rather than racing a core process.

    With V1 multiprocessing the core may start executing the first request
    while later add_request() calls are still being submitted. The first
    LLMEngine.step() is then not an eight-request prefill step, even though the
    entire cohort was added before that call. In-process mode makes the target
    step a deterministic scheduler boundary for this *step* comparison.
    """
    if "vllm.envs" in sys.modules:
        raise RuntimeError("set vLLM step mode before importing vllm")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "check-setup", "run", "run-local",
                                           "run-vllm", "analyze", "run-ladder"))
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
    parser.add_argument("--local-mixed-policy",
                        choices=("separate", "packed", "packed-fa3",
                                 "packed-exact", "packed-exact-qkv"),
                        default="separate", help="packed shares projections; packed-fa3 "
                        "uses FA3 for its decode rows; exact variants specialize the "
                        "graph bucket to this mixed cohort")
    parser.add_argument("--vllm-reference-dir", type=Path,
                        help="reuse an already validated vLLM arm instead of loading it again")
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
    if args.local_mixed_policy != "separate" and args.kind != "mixed":
        raise ValueError("packed mixed policy requires --kind mixed")
    # The shared setup preflight checks the same accepted decode QKV path used
    # in the local mixed arm. It is inactive in a pure-prefill target step.
    args.qkv_mode = "native"
    args.suite_dir = args.suite_dir.resolve()
    phase_directory = (f"mixed-{args.local_mixed_policy}"
                       if args.local_mixed_policy != "separate" else args.kind)
    args.output_dir = (args.output_dir or ROOT / "experiments/results/matched-phase-profile"
                       / phase_directory / args.shape_id).resolve()
    if args.vllm_reference_dir is not None:
        args.vllm_reference_dir = args.vllm_reference_dir.resolve()
        if args.seed != 20260914:
            raise ValueError("saved vLLM reference was generated with seed 20260914")
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


def capture_configuration(args, case, first, second):
    """Keep the scheduler token budget fixed while changing only graph work."""
    exact = args.local_mixed_policy in ("packed-exact", "packed-exact-qkv")
    bucket = second * case["lengths"][0] + first if exact else args.prefill_budget
    if bucket < (second or first) * case["lengths"][0] or bucket > args.prefill_budget:
        raise ValueError("capture bucket does not fit the selected cohort and budget")
    return bucket, args.local_mixed_policy == "packed-exact-qkv"


def plan(args, case, first, second, target_index):
    bucket, qkv_fusion = capture_configuration(args, case, first, second)
    target_tokens = (second or first) * case["lengths"][0] + (first if second else 0)
    return {"kind": args.kind, "shape_id": args.shape_id,
            "first_wave": first, "second_wave": second,
            "prompt_length": case["lengths"][0], "output_tokens": args.output_tokens,
            "target_engine_step": target_index,
            "target_decode_tokens": first if second else 0,
            "target_prefill_tokens": (second or first) * case["lengths"][0],
            "shared_prefill_budget": args.prefill_budget,
            "capture_bucket_tokens": bucket,
            "capture_padding_tokens": bucket - target_tokens,
            "prefill_qkv_rope_cache": qkv_fusion,
            "local_model_passes": (2 if second and args.local_mixed_policy == "separate"
                                   else 1),
            "local_mixed_policy": args.local_mixed_policy,
            "vllm_reference_dir": (str(args.vllm_reference_dir)
                                   if args.vllm_reference_dir else None),
            "note": "fails before reporting a timing if either scheduler misses target work"}


def expected_local_calls(case, first, second):
    prompt = case["lengths"][0]
    prefill = (False, (second or first) * prompt, second or first, prompt)
    return [(True, first, first, 1), prefill] if second else [prefill]


class TargetLogits:
    """Read back just the target callbacks during an untimed validation pass."""

    def __init__(self, start, count):
        self.start, self.end = start, start + count
        self.cursor = 0
        self.rows = []

    def __call__(self, args, logits):
        if self.start <= self.cursor < self.end:
            self.rows.append({"metadata": tuple(t.detach().cpu().clone()
                                                for t in args[:6]),
                              "max_query": args[6], "decode": args[7],
                              "logits": logits.detach().cpu().clone()})
        self.cursor += 1


def compare_target_logits(torch, baseline, candidate):
    if len(baseline) != len(candidate):
        raise AssertionError("candidate target callback count differs")
    rows = []
    for old, new in zip(baseline, candidate):
        same_metadata = (old["decode"] == new["decode"]
                         and old["max_query"] == new["max_query"]
                         and all(torch.equal(left, right)
                                 for left, right in zip(old["metadata"], new["metadata"])))
        if not same_metadata:
            raise AssertionError("candidate target metadata differs from separate passes")
        reference, actual = old["logits"].float(), new["logits"].float()
        if reference.shape != actual.shape or not torch.isfinite(actual).all():
            raise AssertionError("candidate logits changed shape or became nonfinite")
        difference = (actual - reference).abs()
        outside = difference > .05 + .01 * reference.abs()
        rows.append({"decode": bool(old["decode"]), "elements": difference.numel(),
                     "outside_tolerance": int(outside.sum()),
                     "argmax_differences": int((actual.argmax(-1)
                                                 != reference.argmax(-1)).sum()),
                     "max_abs": float(difference.max()),
                     "mean_abs": float(difference.mean())})
    return {"status": "pass" if all(row["outside_tolerance"] == 0 and
                                    row["argmax_differences"] == 0 for row in rows)
            else "numerical_difference", "atol": .05, "rtol": .01, "rows": rows}


def snapshot_target_kv(torch, pool, slots):
    """Read only target-token cache entries, not the whole reserved KV pool."""
    indices = slots.to(device=pool.k_pool[0].device, dtype=torch.long)
    tensors = pool.k_pool + pool.v_pool
    return [tensor.view(-1, tensor.shape[-2], tensor.shape[-1])
            .index_select(0, indices).detach().cpu() for tensor in tensors]


def snapshot_observed_target_kv(torch, pool, observer, expected_callbacks):
    if len(observer.rows) != expected_callbacks:
        raise AssertionError("target KV observer missed a mixed callback")
    slots = torch.cat([row["metadata"][2] for row in observer.rows])
    return slots, snapshot_target_kv(torch, pool, slots)


def compare_target_kv(torch, baseline, candidate):
    if len(baseline) != len(candidate):
        raise AssertionError("KV layer count differs")
    elements = outside = 0
    max_abs = 0.0
    for expected, actual in zip(baseline, candidate):
        if expected.shape != actual.shape:
            raise AssertionError("KV target shape differs")
        reference, observed = expected.float(), actual.float()
        finite = torch.isfinite(reference) & torch.isfinite(observed)
        difference = (observed - reference).abs()
        outside += int((~finite | (difference > .05 + .01 * reference.abs())).sum())
        elements += difference.numel()
        max_abs = max(max_abs, float(difference[finite].max()) if finite.any() else 0.0)
    return {"status": "pass" if outside == 0 else "numerical_difference",
            "elements": elements, "outside_tolerance": outside, "max_abs": max_abs,
            "atol": .05, "rtol": .01}


def complete_local(directory, args):
    reports = list(directory.glob("*-report.json"))
    if len(reports) != 1:
        return False
    report = json.loads(reports[0].read_text())
    trace = Path(report.get("trace", "missing"))
    shape = get_fixed_shape(args.shape_id)
    first = (shape["batch"] // 2 if shape["batch"] == 8 else shape["batch"] // 4)
    second = first if args.kind == "mixed" else 0
    bucket, qkv_fusion = capture_configuration(
        args, {"lengths": [shape["prompt_length"]]}, first, second)
    return (report.get("kind") == args.kind and
            report.get("shape_id") == args.shape_id and
            report.get("prefill_budget") == args.prefill_budget and
            report.get("arrival_step") == args.arrival_step and
            report.get("output_tokens") == args.output_tokens and
            report.get("local_policy") == args.local_policy and
            report.get("local_mixed_policy", "separate") == args.local_mixed_policy and
            report.get("capture_bucket_tokens", args.prefill_budget) == bucket and
            report.get("prefill_qkv_rope_cache", False) == qkv_fusion and
            (not qkv_fusion or (report.get("same_history_exact_control_correctness") or {})
             .get("target_kv") is not None) and
            (trace.is_file() or len(list(directory.rglob(trace.name))) == 1))


def complete_vllm(directory, args):
    path = directory / "report.json"
    if not path.is_file():
        return False
    report = json.loads(path.read_text())
    trace = Path(report.get("trace", "missing"))
    return (report.get("status") == "complete" and report.get("kind") == args.kind
            and report.get("shape_id") == args.shape_id
            and report.get("vllm_version") == PINNED_VLLM
            and report.get("prefill_budget") == args.prefill_budget
            and report.get("vllm_enable_v1_multiprocessing") == "0"
            and report.get("arrival_step") == args.arrival_step
            and report.get("output_tokens") == args.output_tokens
            and (trace.is_file() or len(list(directory.rglob(trace.name))) == 1))


def run_local(args, case, requests, first, second, target_index):
    output = args.output_dir / "local"
    if not prepare_destination(output, lambda path: complete_local(path, args),
                               args.retry_failed):
        print("local already complete; skipping", flush=True)
        return
    from model_adapter import (PackedMixedPiecewiseGraphModelAdapter,
                               PiecewiseGraphModelAdapter, allocate_pool)
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
    adapter_class = (PackedMixedPiecewiseGraphModelAdapter
                     if args.local_mixed_policy != "separate"
                     else PiecewiseGraphModelAdapter)
    capture_bucket, qkv_fusion = capture_configuration(args, case, first, second)
    adapter_options = dict(
        max_running=config.max_batch_size,
        max_context_length=config.max_context_length,
        decode_attention_policy=args.local_policy,
        max_capture_tokens=args.prefill_budget, max_prefill_shapes=1,
        prefill_buckets=[capture_bucket],
        enable_residual_rmsnorm=True,
        enable_native_decode_qkv_postprocess=True,
        enable_prefill_swiglu_fusion=True,
        enable_prefill_packed_qkv_rope_cache=qkv_fusion)
    if args.local_mixed_policy == "packed-fa3":
        adapter_options["mixed_attention_policy"] = "fa3_hybrid"
    adapter = adapter_class(engine.model, pool, None, **adapter_options)

    def poison():
        for tensor in pool.k_pool + pool.v_pool:
            tensor.fill_(float("nan"))

    poison()
    preflight = execute(torch, cpp.IterationLoop(config, torch.device(args.device)),
                        adapter, requests)
    if (set(adapter.piecewise_prefill.shapes) != {capture_bucket}
            or adapter.piecewise_prefill.eager_calls):
        raise AssertionError("piecewise prefill missed the selected graph bucket")
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
    correctness = None
    exact_control_correctness = None
    candidate_kv = None
    candidate_slots = None
    if args.local_mixed_policy != "separate":
        first_call = sum(len(row["calls"]) for row in preflight["steps"][:target_index])
        reference = TargetLogits(first_call, len(calls))
        candidate = TargetLogits(first_call, len(calls))
        try:
            adapter.enable_packed_mixed = False
            adapter.observer = reference
            poison()
            execute(torch, cpp.IterationLoop(config, torch.device(args.device)),
                    adapter, requests)
            adapter.enable_packed_mixed = True
            adapter.observer = candidate
            poison()
            execute(torch, cpp.IterationLoop(config, torch.device(args.device)),
                    adapter, requests)
            if qkv_fusion:
                candidate_slots, candidate_kv = snapshot_observed_target_kv(
                    torch, pool, candidate, len(calls))
        finally:
            adapter.enable_packed_mixed = True
            adapter.observer = None
        if len(reference.rows) != len(calls) or len(candidate.rows) != len(calls):
            raise AssertionError("target logit observer missed a mixed callback")
        correctness = compare_target_logits(torch, reference.rows, candidate.rows)
        if qkv_fusion:
            # The packed-vs-separate check above holds the fusion constant.
            # Compare against the exact-bucket unfused model as a second,
            # untimed numerical gate so the fusion itself is not hidden.
            control_options = {**adapter_options,
                               "enable_prefill_packed_qkv_rope_cache": False}
            control_adapter = PackedMixedPiecewiseGraphModelAdapter(
                engine.model, pool, None, **control_options)
            control = TargetLogits(first_call, len(calls))
            control_adapter.observer = control
            try:
                poison()
                control_run = execute(
                    torch, cpp.IterationLoop(config, torch.device(args.device)),
                    control_adapter, requests)
                if len(control.rows) != len(calls):
                    raise AssertionError("exact control missed a target callback")
                exact_control_correctness = compare_target_logits(
                    torch, control.rows, candidate.rows)
                control_kv = snapshot_target_kv(torch, pool, candidate_slots)
                kv_check = compare_target_kv(torch, candidate_kv, control_kv)
                exact_control_correctness["target_kv"] = kv_check
                if kv_check["status"] != "pass":
                    exact_control_correctness["status"] = "numerical_difference"
                tokens_equal = control_run["outputs"] == preflight["outputs"]
                exact_control_correctness["whole_workload_tokens_equal"] = tokens_equal
                if not tokens_equal:
                    exact_control_correctness["status"] = "numerical_difference"
            except AssertionError as error:
                exact_control_correctness = {
                    "status": "history_diverged", "reason": str(error),
                    "whole_workload_tokens_equal": False}
            finally:
                control_adapter.observer = None
                del control_adapter

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
              "local_policy": args.local_policy,
              "local_mixed_policy": args.local_mixed_policy, "first_wave": first,
              "capture_bucket_tokens": capture_bucket,
              "prefill_qkv_rope_cache": qkv_fusion,
              "second_wave": second, "target_step_index": target_index,
              "target_step_calls": target[1], "target_callback_calls": traced["target_calls"],
              "outputs_sha256": output_hash, "model_load_seconds": load_seconds,
              "same_history_target_correctness": correctness,
              "same_history_exact_control_correctness": exact_control_correctness,
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
    if args.vllm_reference_dir is not None:
        raise ValueError("--vllm-reference-dir reuses a prior result; do not run-vllm")
    configure_vllm_step_mode()
    output = args.output_dir / "vllm"
    if not prepare_destination(output, lambda path: complete_vllm(path, args),
                               args.retry_failed):
        print("vLLM already complete; skipping", flush=True)
        return
    setup = verify_external_setup(args)
    trace_dir = output / "trace"
    trace_dir.mkdir()
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
              "vllm_enable_v1_multiprocessing": os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"],
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
    vllm_dir = ((args.vllm_reference_dir or args.output_dir) / "vllm")
    if len(local_paths) != 1 or not complete_vllm(vllm_dir, args):
        raise ValueError("both local and vLLM reports must be complete")
    local = json.loads(local_paths[0].read_text())
    vllm = json.loads((vllm_dir / "report.json").read_text())
    calls = expected_local_calls(case, first, second)
    if (not complete_local(args.output_dir / "local", args)
            or [tuple(row) for row in local["target_step_calls"]] != calls
            or local["target_step_index"] != target_index
            or vllm["target_step_index"] != target_index
            or local["model"] != vllm["model"]
            or local["first_wave"] != vllm["first_wave"]
            or local["second_wave"] != vllm["second_wave"]):
        raise ValueError("local and vLLM work/settings do not match")
    if vllm.get("vllm_enable_v1_multiprocessing") != "0":
        raise ValueError("vLLM result was not run with deterministic step scheduling")
    verify_vllm_target(vllm, first, second, args.kind)
    local_cuda = summarize_chrome_trace(resolve_trace(local["trace"], args.output_dir / "local"))
    vllm_cuda = summarize_chrome_trace(resolve_trace(vllm["trace"], vllm_dir))
    from profile_latest_vs_vllm import compare_categories
    ratio = local["unprofiled_median_wall_ms"] / vllm["unprofiled_median_wall_ms"]
    separate_baseline = None
    if args.vllm_reference_dir is not None:
        baseline_files = list((args.vllm_reference_dir / "local").glob("*-report.json"))
        if len(baseline_files) == 1:
            baseline = json.loads(baseline_files[0].read_text())
            if (baseline.get("kind") == args.kind
                    and baseline.get("shape_id") == args.shape_id
                    and baseline.get("prefill_budget") == args.prefill_budget
                    and baseline.get("arrival_step") == args.arrival_step
                    and baseline.get("output_tokens") == args.output_tokens
                    and baseline.get("local_policy") == args.local_policy
                    and baseline.get("local_mixed_policy", "separate") == "separate"):
                separate_baseline = {
                    "wall_ms": baseline["unprofiled_median_wall_ms"],
                    "speedup": (baseline["unprofiled_median_wall_ms"]
                                / local["unprofiled_median_wall_ms"]),
                    "category_comparison": compare_categories(
                        local_cuda, summarize_chrome_trace(resolve_trace(
                            baseline["trace"], args.vllm_reference_dir / "local"))),
                }
    report = {"status": "complete", "kind": args.kind, "shape_id": args.shape_id,
              "work": plan(args, case, first, second, target_index),
              "vllm_reference_dir": str(vllm_dir),
              "same_history_target_correctness": local.get("same_history_target_correctness"),
              "same_history_exact_control_correctness": local.get(
                  "same_history_exact_control_correctness"),
              "separate_local_baseline": separate_baseline,
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
    if report["same_history_target_correctness"] is not None:
        print("same-history target correctness: "
              f"{report['same_history_target_correctness']['status']}")
    if report["same_history_exact_control_correctness"] is not None:
        print("vs exact unfused correctness: "
              f"{report['same_history_exact_control_correctness']['status']}")
    if separate_baseline is not None:
        print(f"vs prior separate local: {separate_baseline['speedup']:.3f}x")
    print(f"comparison: {path}")


def run_ladder(args, case, first, second):
    """Run/reuse broad, exact, and exact-plus-QKV arms against one vLLM trace."""
    if args.kind != "mixed" or args.vllm_reference_dir is None:
        raise ValueError("run-ladder requires --kind mixed and --vllm-reference-dir")
    vllm_dir = args.vllm_reference_dir / "vllm"
    if not complete_vllm(vllm_dir, args):
        raise ValueError("saved vLLM reference is incomplete or has different settings")
    saved_vllm = json.loads((vllm_dir / "report.json").read_text())
    if saved_vllm["model"] != resolve_model_source(args):
        raise ValueError("saved vLLM model does not match selected model snapshot")
    verify_vllm_target(saved_vllm, first, second, "mixed")
    arms = ("packed", "packed-exact", "packed-exact-qkv")
    common = ["--kind", "mixed", "--shape-id", args.shape_id,
              "--suite-dir", str(args.suite_dir), "--model", args.model,
              "--device", args.device, "--seed", str(args.seed),
              "--arrival-step", str(args.arrival_step),
              "--output-tokens", str(args.output_tokens),
              "--prefill-budget", str(args.prefill_budget),
              "--local-policy", args.local_policy,
              "--warmups", str(args.warmups),
              "--repetitions", str(args.repetitions),
              "--vllm-reference-dir", str(args.vllm_reference_dir)]
    if args.retry_failed:
        common.append("--retry-failed")
    rows = []
    for arm in arms:
        directory = ROOT / "experiments/results/matched-phase-profile" / f"mixed-{arm}" / args.shape_id
        arm_args = argparse.Namespace(**{**vars(args), "local_mixed_policy": arm,
                                         "output_dir": directory})
        comparison = directory / "comparison.json"
        if not (complete_local(directory / "local", arm_args) and comparison.is_file()):
            print(f"running {arm}", flush=True)
            subprocess.run([sys.executable, str(Path(__file__)), "run", *common,
                            "--local-mixed-policy", arm], cwd=ROOT, check=True)
        else:
            print(f"reusing {arm}", flush=True)
        result = json.loads(comparison.read_text())
        work = result["work"]
        if (result.get("status") != "complete" or work["shape_id"] != args.shape_id
                or work["local_mixed_policy"] != arm
                or work["shared_prefill_budget"] != args.prefill_budget
                or result["vllm_wall_ms"] != saved_vllm["unprofiled_median_wall_ms"]):
            raise ValueError(f"{arm} comparison has mismatched work or vLLM reference")
        bucket, qkv = capture_configuration(arm_args, case, first, second)
        categories = {row["category"]: row["total_us"]
                      for row in result["local_cuda_activity"]["categories"]}
        rows.append({"arm": arm, "capture_bucket_tokens": bucket,
                     "prefill_qkv_rope_cache": qkv,
                     "local_wall_ms": result["local_wall_ms"],
                     "local_over_vllm": result["local_over_vllm"],
                     "profiled_gemm_us": categories.get("gemm", 0.0),
                     "profiled_attention_us": categories.get("attention", 0.0),
                     "profiled_rope_us": categories.get("rope", 0.0),
                     "target_correctness": result.get("same_history_target_correctness"),
                     "exact_control_correctness": result.get(
                         "same_history_exact_control_correctness"),
                     "comparison": str(comparison)})
    broad = rows[0]["local_wall_ms"]
    exact = rows[1]["local_wall_ms"]
    for row in rows:
        row["speedup_vs_broad"] = broad / row["local_wall_ms"]
        row["speedup_vs_exact"] = exact / row["local_wall_ms"]
    valid = all(row["target_correctness"] is not None and
                row["target_correctness"]["status"] == "pass" for row in rows)
    valid = valid and rows[-1]["exact_control_correctness"] is not None and \
        rows[-1]["exact_control_correctness"]["status"] == "pass"
    report = {"status": "complete" if valid else "numerical_difference",
              "shape_id": args.shape_id, "vllm_wall_ms": saved_vllm["unprofiled_median_wall_ms"],
              "notes": ["Existing arm reports are reused; wall medians may be from "
                        "different runs and should be reconfirmed if close.",
                        "Numerical differences are reported without suppressing timings."],
              "rows": rows}
    destination = (ROOT / "experiments/results/matched-phase-profile/mixed-ladder"
                   / args.shape_id)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "ladder.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print("arm                 bucket   local ms   GEMM us   vs broad   vs vLLM   logits")
    for row in rows:
        checks = row["target_correctness"] or {}
        print(f"{row['arm']:<19} {row['capture_bucket_tokens']:>6} "
              f"{row['local_wall_ms']:>10.3f} {row['profiled_gemm_us']:>9.0f} "
              f"{row['speedup_vs_broad']:>10.3f}x "
              f"{row['local_over_vllm']:>10.3f}x {checks.get('status', 'missing')}")
    print(f"ladder: {path} ({report['status']})")


def main():
    args = build_parser().parse_args()
    if args.action == "run-ladder" and args.output_dir is not None:
        raise ValueError("run-ladder uses fixed per-arm output directories; omit --output-dir")
    case, requests, first, second, target_index = validate(args)
    if args.action == "plan":
        print(json.dumps(plan(args, case, first, second, target_index), indent=2))
    elif args.action == "check-setup":
        configure_vllm_step_mode()
        print(json.dumps({**verify_external_setup(args),
                          "vllm_enable_v1_multiprocessing": "0"}, indent=2))
    elif args.action == "run-local":
        run_local(args, case, requests, first, second, target_index)
    elif args.action == "run-vllm":
        run_vllm(args, case, requests, first, second, target_index)
    elif args.action == "analyze":
        analyze(args, case, first, second, target_index)
    elif args.action == "run-ladder":
        run_ladder(args, case, first, second)
    else:
        configure_vllm_step_mode()
        verify_external_setup(args)
        common = ["--kind", args.kind, "--shape-id", args.shape_id,
                  "--suite-dir", str(args.suite_dir), "--output-dir", str(args.output_dir),
                  "--model", args.model, "--device", args.device,
                  "--seed", str(args.seed), "--arrival-step", str(args.arrival_step),
                  "--output-tokens", str(args.output_tokens),
                  "--prefill-budget", str(args.prefill_budget),
                  "--local-policy", args.local_policy,
                  "--local-mixed-policy", args.local_mixed_policy,
                  "--warmups", str(args.warmups),
                  "--repetitions", str(args.repetitions)]
        if args.vllm_reference_dir is not None:
            common.extend(("--vllm-reference-dir", str(args.vllm_reference_dir)))
        if args.retry_failed:
            common.append("--retry-failed")
        actions = (("run-local", "analyze") if args.vllm_reference_dir
                   else ("run-local", "run-vllm", "analyze"))
        for action in actions:
            subprocess.run([sys.executable, str(Path(__file__)), action, *common],
                           cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
