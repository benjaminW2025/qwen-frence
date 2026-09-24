#!/usr/bin/env python3
"""A/B one C++ packed mixed callback against the existing two-callback path.

Both arms run the same staggered B8 workload and packed-varlen FA3 attention.
The control still assembles mixed metadata in Python; the candidate builds it
directly in pinned C++ buffers and transfers/calls the model once. No vLLM load.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for directory in (HERE, ROOT / "baseline", ROOT / "engine/model_runner",
                  ROOT / "engine/kvcache", ROOT / "engine/cpp/build"):
    sys.path.insert(0, str(directory))

from benchmark_mixed_graph_churn import LogitObserver, workload


def work_signature(result):
    return [(step["kind"], step["calls"], step["completed"])
            for step in result["steps"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "check", "run"))
    parser.add_argument("--suite-dir", type=Path)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "experiments/results/cpp-packed-mixed-v1")
    args = parser.parse_args()
    if args.action == "plan":
        print(json.dumps({"shape": "B8, eight 256-token prompts, 18 output tokens",
                          "arrivals": [0, 0, 3, 4, 5, 6, 7, 8],
                          "control": "two C++ callbacks + Python metadata packing",
                          "candidate": "one C++ packed callback + one pinned H2D phase",
                          "same_attention": "packed-varlen FA3, piecewise model graphs",
                          "checks": ["all callback logits", "scheduler work",
                                     "output tokens", "full KV cache"],
                          "timing": "three interleaved unprofiled runs per arm; "
                                    "one diagnostic trace of the first mixed step"}, indent=2))
        return
    if args.action == "run":
        if args.suite_dir is None:
            parser.error("run requires --suite-dir with fixed-b8-l256-o128/workload.json")
        if args.output_dir.exists():
            parser.error(f"refusing to overwrite existing results: {args.output_dir}")
        from benchmark_latest_vs_vllm import load_frozen, resolve_model_source
        from fixed_regime import get_fixed_case
        frozen_case = get_fixed_case("fixed-b8-l256-o128")
        frozen_args = SimpleNamespace(output_dir=args.suite_dir / frozen_case["id"],
                                      shape_id=frozen_case["id"], seed=20260914,
                                      model=args.model, device=args.device,
                                      logit_atol=.05, trials=1, samples=1,
                                      repetitions=1, warmups=0, profile_occurrence=0)
        _, frozen = load_frozen(frozen_args, frozen_case)
        case, requests = workload(frozen)
        source = resolve_model_source(args)

    from model_setup import check_startup, load_model_only
    setup = check_startup(args.device)
    import inference_engine_cpp as cpp
    if not hasattr(cpp.SchedulerConfig(), "packed_mixed_step"):
        raise RuntimeError("C++ extension lacks packed_mixed_step; rebuild it")
    from kernel_dispatch import _load
    _load("paged_varlen_fa3").smoke_varlen_fa3(args.device)
    if args.action == "check":
        print(json.dumps({"startup": setup, "packed_mixed_extension": "pass",
                          "varlen_fa3_smoke": "pass", "model_loaded": False}, indent=2))
        return
    args.output_dir.mkdir(parents=True, exist_ok=False)

    import torch
    from benchmark_scheduler_decode import execute, make_config
    from model_adapter import (CppPackedMixedModelAdapter,
                               PackedMixedPiecewiseGraphModelAdapter, allocate_pool)
    from profile_cpp_control import drive, stage_summary, cuda_activity_summary

    engine, load_seconds, _ = load_model_only(
        source, args.device, "float16", hub_transfer=setup["hub_transfer"])
    config = make_config(cpp, case)
    packed_config = make_config(cpp, case)
    packed_config.packed_mixed_step = True
    blocks = config.max_batch_size * ((config.max_context_length + 15) // 16 + 1)
    pool = allocate_pool(engine.cfg, blocks, engine.device)
    options = dict(max_running=8, max_context_length=config.max_context_length,
                   decode_attention_policy="fa3", max_capture_tokens=2048,
                   max_prefill_shapes=4, prefill_buckets=[256, 512, 1024, 2048],
                   enable_residual_rmsnorm=True,
                   enable_native_decode_qkv_postprocess=True,
                   enable_prefill_swiglu_fusion=True)
    control = PackedMixedPiecewiseGraphModelAdapter(
        engine.model, pool, None, mixed_attention_policy="fa3_varlen",
        full_mixed_graph=False, **options)
    candidate = CppPackedMixedModelAdapter(engine.model, pool, None, **options)

    def run_one(adapter, scheduler_config, observer=None):
        for tensor in pool.k_pool + pool.v_pool:
            tensor.fill_(float("nan"))
        adapter.observer = observer
        try:
            return execute(torch, cpp.IterationLoop(scheduler_config,
                           torch.device(args.device)), adapter, requests)
        finally:
            adapter.observer = None

    def kv_snapshot():
        return [tensor.detach().cpu().clone() for tensor in pool.k_pool + pool.v_pool]

    control_logits = LogitObserver(torch)
    reference = run_one(control, config, control_logits)
    reference_kv = kv_snapshot()
    candidate_logits = LogitObserver(torch, control_logits.rows)
    checked = run_one(candidate, packed_config, candidate_logits)
    candidate_logits.finish()
    kv_equal = all(torch.allclose(a, b, atol=.05, rtol=.01, equal_nan=True)
                   for a, b in zip(reference_kv, kv_snapshot()))
    schedule_equal = work_signature(reference) == work_signature(checked)
    tokens_equal = reference["outputs"] == checked["outputs"]
    target_index = next((index for index, row in enumerate(reference["steps"])
                         if row["kind"] == "mixed"), None)
    if target_index is None:
        raise AssertionError("staggered workload did not produce a mixed step")

    # Alternate the two warmed arms; metadata observer and profiler are absent.
    samples = {"control": [], "candidate": []}
    for arm in ("control", "candidate", "candidate", "control", "control", "candidate"):
        adapter, scheduler_config = ((control, config) if arm == "control"
                                     else (candidate, packed_config))
        row = run_one(adapter, scheduler_config)
        samples[arm].append(row)
    warm_outputs_equal = all(row["outputs"] == reference["outputs"]
                             for rows in samples.values() for row in rows)

    # Instrumentation is diagnostic only; medians above exclude it.
    traces = {}
    for arm, adapter, scheduler_config in (("control", control, config),
                                           ("candidate", candidate, packed_config)):
        try:
            for tensor in pool.k_pool + pool.v_pool:
                tensor.fill_(float("nan"))
            profiled = drive(torch, cpp, scheduler_config, requests, adapter,
                             target_index, profile_target=True)
            if profiled["outputs"] != reference["outputs"]:
                warm_outputs_equal = False
            path = args.output_dir / f"{arm}-target-trace.json"
            profiled["profiler"].export_chrome_trace(str(path))
            traces[arm] = {"target_callback_count": len(profiled["target_calls"]),
                           "cpu_ranges": stage_summary(profiled["profiler"]),
                           "cuda_activity": cuda_activity_summary(
                               profiled["profiler"].events()),
                           "trace": str(path)}
        except Exception as error:
            traces[arm] = {"target_callback_count": 0,
                           "error": f"{type(error).__name__}: {error}"}

    control_mixed = statistics.median(row["mixed_wall_ms"]
                                       for row in samples["control"])
    candidate_mixed = statistics.median(row["mixed_wall_ms"]
                                         for row in samples["candidate"])
    valid = (not candidate_logits.mismatched_callbacks and kv_equal and
             schedule_equal and tokens_equal and warm_outputs_equal and
             traces["control"]["target_callback_count"] == 2 and
             traces["candidate"]["target_callback_count"] == 1)
    report = {"status": "complete" if valid else "invalid_comparison",
              "model": source, "model_load_seconds": load_seconds,
              "workload": {"arrivals": case["arrivals"], "lengths": case["lengths"],
                           "outputs": case["outputs"], "prefill_budget": 2048},
              "correctness": {
                  "all_callback_logits": "pass" if not candidate_logits.mismatched_callbacks
                                         else "numerical_difference",
                  "mismatched_callbacks": candidate_logits.mismatched_callbacks,
                  "max_abs_logit_error": candidate_logits.max_abs_error,
                  "scheduler_work": "pass" if schedule_equal else "different",
                  "output_tokens": "pass" if tokens_equal else "different",
                  "warm_output_tokens": "pass" if warm_outputs_equal else "different",
                  "kv_cache": "pass" if kv_equal else "different"},
              "cold_mixed_wall_ms": {"control": reference["mixed_wall_ms"],
                                     "candidate": checked["mixed_wall_ms"]},
              "warm_mixed_wall_ms": {
                  arm: [row["mixed_wall_ms"] for row in rows]
                  for arm, rows in samples.items()},
              "warm_total_wall_ms": {
                  arm: [row["wall_ms"] for row in rows]
                  for arm, rows in samples.items()},
              "warm_median_mixed_wall_ms": {"control": control_mixed,
                                            "candidate": candidate_mixed},
              "speedup_mixed": control_mixed / candidate_mixed,
              "traces": traces,
              "note": "CPU/CUDA trace timings are diagnostic; speedup uses only "
                      "unprofiled alternating runs. Incorrect results retain timings."}
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"mixed wall median: two-callback {control_mixed:.3f} ms; "
          f"single-callback {candidate_mixed:.3f} ms; "
          f"speedup {report['speedup_mixed']:.3f}x")
    print(f"target callbacks: {traces['control']['target_callback_count']} -> "
          f"{traces['candidate']['target_callback_count']}; "
          f"status={report['status']}; report={report_path}")
    if not valid:
        print("WARNING: correctness or trace gate failed; timings are diagnostic only",
              file=sys.stderr)


if __name__ == "__main__":
    main()
