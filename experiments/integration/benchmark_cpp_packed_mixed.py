#!/usr/bin/env python3
"""A/B packed-mixed control-plane and graph-boundary interventions.

Both arms run the same frozen prompts and packed-varlen FA3 attention.  The
shape-selectable boundary and metadata interventions compare against the current
single-callback engine. No vLLM load.
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

from benchmark_mixed_graph_churn import LogitObserver
from fixed_regime import FACTORIAL_SHAPES, get_fixed_case

SHAPES = tuple(row["id"] for row in FACTORIAL_SHAPES)


def probe_workload(frozen, case, tail_steps=8):
    """Create a short deterministic mixed probe while retaining the frozen prompts."""
    first = case["max_running"] // 2
    prompt = case["lengths"][0]
    arrival = (first * prompt + case["prefill_budget"] - 1) // case["prefill_budget"] + 3
    output = arrival + tail_steps
    requests = [dict(id=row["id"], prompt=row["prompt"], output=output,
                     arrival=0 if row["id"] < first else arrival) for row in frozen]
    return ({**case, "id": f"control-probe-{case['id']}",
             "outputs": [output] * len(requests),
             "arrivals": [row["arrival"] for row in requests]}, requests, arrival)


def work_signature(result):
    return [(step["kind"], step["calls"], step["completed"])
            for step in result["steps"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "check", "run"))
    parser.add_argument("--suite-dir", type=Path)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shape-id", choices=SHAPES, default="fixed-b8-l256-o128")
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--intervention", choices=("packed-callback", "boundary-buffers",
                        "resident-metadata", "buffers-and-metadata"), default="packed-callback",
                        help="new arms compare against the current single-callback FA3 path")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "experiments/results/cpp-packed-mixed-v1")
    args = parser.parse_args()
    if args.action == "plan":
        shape = get_fixed_case(args.shape_id)
        print(json.dumps({"shape_id": args.shape_id,
                          "batch": shape["max_running"],
                          "prompt_tokens": shape["lengths"][0],
                          "intervention": args.intervention,
                          "control": ("two callbacks" if args.intervention == "packed-callback"
                                      else "current single packed callback"),
                          "candidate": args.intervention,
                          "same_attention": "packed-varlen FA3, piecewise model graphs",
                          "checks": ["all callback logits", "scheduler work",
                                     "output tokens", "full KV cache"],
                          "timing": f"{args.repetitions} interleaved unprofiled runs per arm; "
                                    "one diagnostic trace of the first mixed step"}, indent=2))
        return
    if args.action == "run":
        if args.suite_dir is None:
            parser.error("run requires --suite-dir with the selected frozen workload")
        if args.output_dir.exists():
            parser.error(f"refusing to overwrite existing results: {args.output_dir}")
        from benchmark_latest_vs_vllm import load_frozen, resolve_model_source
        from fixed_regime import get_fixed_case
        frozen_case = get_fixed_case(args.shape_id)
        frozen_args = SimpleNamespace(output_dir=args.suite_dir / frozen_case["id"],
                                      shape_id=frozen_case["id"], seed=args.seed,
                                      model=args.model, device=args.device,
                                      logit_atol=.05, trials=1, samples=1,
                                      repetitions=1, warmups=0, profile_occurrence=0)
        _, frozen = load_frozen(frozen_args, frozen_case)
        case, requests, arrival = probe_workload(frozen, frozen_case)
        source = resolve_model_source(args)

    from model_setup import check_startup, load_model_only
    setup = check_startup(args.device)
    import inference_engine_cpp as cpp
    if not hasattr(cpp.SchedulerConfig(), "packed_mixed_step"):
        raise RuntimeError("C++ extension lacks packed_mixed_step; rebuild it")
    reuse_metadata = args.intervention in ("resident-metadata", "buffers-and-metadata")
    reuse_buffers = args.intervention in ("boundary-buffers", "buffers-and-metadata")
    if reuse_metadata and not hasattr(cpp.SchedulerConfig(), "reuse_stable_decode_metadata"):
        raise RuntimeError("C++ extension lacks reuse_stable_decode_metadata; rebuild it")
    from kernel_dispatch import _load
    _load("paged_varlen_fa3").smoke_varlen_fa3(args.device, boundary_buffers=reuse_buffers)
    if args.action == "check":
        print(json.dumps({"startup": setup, "packed_mixed_extension": "pass",
                          "varlen_fa3_smoke": "pass", "model_loaded": False}, indent=2))
        return
    args.output_dir.mkdir(parents=True, exist_ok=False)

    import torch
    from benchmark_scheduler_decode import execute, make_config
    from benchmark_integrated_graph import dry_schedule
    from model_adapter import (CppPackedMixedModelAdapter,
                               PackedMixedPiecewiseGraphModelAdapter, allocate_pool)
    from profile_cpp_control import drive, stage_summary, cuda_activity_summary

    engine, load_seconds, _ = load_model_only(
        source, args.device, "float16", hub_transfer=setup["hub_transfer"])
    config = make_config(cpp, case)
    packed_config = make_config(cpp, case)
    packed_config.packed_mixed_step = True
    if args.intervention != "packed-callback":
        config.packed_mixed_step = True
    if reuse_metadata:
        packed_config.reuse_stable_decode_metadata = True
    schedule = dry_schedule(torch, cpp, case, args.seed, requests)
    prefill = [call[1] for step in schedule["steps"] for call in step["calls"] if not call[0]]
    packed = [sum(call[1] for call in step["calls"])
              for step in schedule["steps"] if step["kind"] == "mixed"]
    if not prefill or not packed:
        raise AssertionError(
            f"{args.shape_id}: probe must contain both prefill and mixed work; "
            f"got prefill_calls={len(prefill)}, mixed_steps={len(packed)}")
    buckets = sorted({max(prefill), max(packed)})
    blocks = config.max_batch_size * ((config.max_context_length + 15) // 16 + 1)
    pool = allocate_pool(engine.cfg, blocks, engine.device)
    options = dict(max_running=case["max_running"], max_context_length=config.max_context_length,
                   decode_attention_policy="fa3", max_capture_tokens=max(buckets),
                   max_prefill_shapes=len(buckets), prefill_buckets=buckets,
                   enable_residual_rmsnorm=True,
                   enable_native_decode_qkv_postprocess=True,
                   enable_prefill_swiglu_fusion=True)
    if args.intervention != "packed-callback":
        # Match the accepted eight-cell --attention fa3 baseline: FA3 for pure
        # prefill too, not the Triton packed kernel that fa3 decode otherwise
        # pairs with. packed-callback keeps its original configuration.
        options["prefill_attention_policy"] = "fa3_varlen"
    if args.intervention == "packed-callback":
        control = PackedMixedPiecewiseGraphModelAdapter(
            engine.model, pool, None, mixed_attention_policy="fa3_varlen",
            full_mixed_graph=False, **options)
    else:
        control = CppPackedMixedModelAdapter(engine.model, pool, None, **options)
    candidate = CppPackedMixedModelAdapter(
        engine.model, pool, None, **options,
        enable_prefill_boundary_buffer_reuse=reuse_buffers,
        enable_stable_decode_table_cache=reuse_metadata)

    def run_one(adapter, scheduler_config, observer=None):
        for tensor in pool.k_pool + pool.v_pool:
            tensor.fill_(float("nan"))
        adapter.observer = observer
        try:
            loop = cpp.IterationLoop(scheduler_config, torch.device(args.device))
            before = adapter.piecewise_prefill.boundary_copies()
            result = execute(torch, loop, adapter, requests)
            after = adapter.piecewise_prefill.boundary_copies()
            result["device_decode_state_replays"] = loop.num_device_decode_state_replays()
            result["boundary_copies"] = {key: after[key] - before[key] for key in after}
            return result
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
    target_kind = "decode" if reuse_metadata else "mixed"
    targets = [index for index, row in enumerate(reference["steps"])
               if row["kind"] == target_kind]
    # The first pure decode seeds resident state; profile a later replay.
    target_index = (targets[1] if reuse_metadata and len(targets) > 1
                    else targets[0] if targets else None)
    if target_index is None:
        raise AssertionError(f"staggered workload did not produce an eligible {target_kind} step")

    # Alternate the two warmed arms; metadata observer and profiler are absent.
    samples = {"control": [], "candidate": []}
    order = [name for _ in range(args.repetitions) for name in ("control", "candidate")]
    for pair in range(0, len(order), 4):
        order[pair:pair + 4] = reversed(order[pair:pair + 4])
    for arm in order:
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
    medians = {phase: {arm: statistics.median(row[f"{phase}_wall_ms"]
                                               for row in rows)
                       for arm, rows in samples.items()}
               for phase in ("decode", "prefill", "mixed")}
    medians["wall"] = {arm: statistics.median(row["wall_ms"] for row in rows)
                       for arm, rows in samples.items()}
    prefill_mixed = {arm: statistics.median(
        row["prefill_wall_ms"] + row["mixed_wall_ms"] for row in rows)
        for arm, rows in samples.items()}
    valid = (not candidate_logits.mismatched_callbacks and kv_equal and
             schedule_equal and tokens_equal and warm_outputs_equal and
             traces["control"]["target_callback_count"] == (
                 2 if args.intervention == "packed-callback" else 1) and
             traces["candidate"]["target_callback_count"] == 1)
    metadata_executed = (not reuse_metadata or all(
        row["device_decode_state_replays"] > 0 for row in samples["candidate"]))
    # Buffer reuse must remove every boundary copy the control performs; a
    # silently disabled path would otherwise report ~1.0x as a valid result.
    boundary_executed = (not reuse_buffers or (
        all(row["boundary_copies"] == {"residual": 0, "attention": 0}
            for row in samples["candidate"])
        and all(row["boundary_copies"]["residual"] > 0
                and row["boundary_copies"]["attention"] > 0
                for row in samples["control"])))
    valid = valid and metadata_executed and boundary_executed
    report = {"status": "complete" if valid else "invalid_comparison",
              "intervention": args.intervention,
              "shape_id": args.shape_id,
              "metadata_reuse_executed": metadata_executed if reuse_metadata else None,
              "boundary_reuse_executed": boundary_executed if reuse_buffers else None,
              "boundary_copies": {arm: [row["boundary_copies"] for row in rows]
                                  for arm, rows in samples.items()},
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
              "warm_median_wall_ms": medians,
              "speedup_total": medians["wall"]["control"] / medians["wall"]["candidate"],
              "speedup_decode": medians["decode"]["control"] / medians["decode"]["candidate"],
              "speedup_prefill_mixed": prefill_mixed["control"] / prefill_mixed["candidate"],
              "traces": traces,
              "note": "CPU/CUDA trace timings are diagnostic; speedup uses only "
                      "unprofiled alternating runs. Incorrect results retain timings."}
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{args.shape_id} {args.intervention}: mixed wall median: control {control_mixed:.3f} ms; "
          f"candidate {candidate_mixed:.3f} ms; "
          f"speedup {report['speedup_mixed']:.3f}x")
    print(f"target callbacks: {traces['control']['target_callback_count']} -> "
          f"{traces['candidate']['target_callback_count']}; "
          f"status={report['status']}; report={report_path}")
    if not valid:
        print("WARNING: correctness or trace gate failed; timings are diagnostic only",
              file=sys.stderr)


if __name__ == "__main__":
    main()
