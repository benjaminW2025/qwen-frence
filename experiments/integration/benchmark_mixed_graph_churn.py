#!/usr/bin/env python3
"""Small local-only cold/warm test for changing mixed-step graph shapes.

Eight frozen prompts arrive as 2 initial requests plus 6 single-request waves.
This intentionally creates more mixed shapes than the four-entry full-graph cache.
No vLLM load, tokenizer, download, or profiler is involved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for directory in (HERE, ROOT / "baseline", ROOT / "engine/model_runner",
                  ROOT / "engine/kvcache", ROOT / "engine/cpp/build"):
    sys.path.insert(0, str(directory))


def workload(frozen, *, output_tokens=18):
    if len(frozen) < 8 or output_tokens < 12:
        raise ValueError("need eight frozen prompts and at least twelve output tokens")
    arrivals = [0, 0, 3, 4, 5, 6, 7, 8]
    requests = [dict(id=i, prompt=frozen[i]["prompt"], output=output_tokens,
                     arrival=arrival)
                for i, arrival in enumerate(arrivals)]
    lengths = [len(row["prompt"]) for row in requests]
    if lengths != [256] * 8:
        raise ValueError("this shape-churn probe requires eight 256-token prompts")
    case = {"id": "mixed-graph-churn-b8-l256", "lengths": lengths,
            "outputs": [output_tokens] * 8, "arrivals": arrivals,
            "max_running": 8, "prefill_budget": 2048}
    return case, requests


class LogitObserver:
    """Untimed full-logit check at every callback, not just the first shape."""

    def __init__(self, torch, reference=None):
        self.torch, self.reference, self.rows = torch, reference, []
        self.max_abs_error = 0.0
        self.mismatched_callbacks = []

    def __call__(self, args, logits):
        row = (bool(args[-1]), tuple(logits.shape), logits.detach().cpu().clone())
        index = len(self.rows)
        if self.reference is not None:
            if index >= len(self.reference) or row[:2] != self.reference[index][:2]:
                self.mismatched_callbacks.append(
                    {"callback": index, "reason": "type_or_shape"})
            else:
                expected = self.reference[index][2]
                close = self.torch.isclose(row[2], expected, atol=.05, rtol=.01)
                outside = int((~close).sum())
                if outside:
                    self.mismatched_callbacks.append({"callback": index,
                                                      "reason": "logits",
                                                      "outside_tolerance": outside})
                self.max_abs_error = max(
                    self.max_abs_error,
                    float((row[2].float() - expected.float()).abs().max()))
        self.rows.append(row)

    def finish(self):
        if self.reference is not None and len(self.rows) != len(self.reference):
            self.mismatched_callbacks.append(
                {"reason": "callback_count", "reference": len(self.reference),
                 "candidate": len(self.rows)})


def summarize_steps(result):
    return {"wall_ms": result["wall_ms"],
            "mixed_wall_ms": result["mixed_wall_ms"],
            "mixed_steps": [
                {"wall_ms": row["wall_ms"], "calls": row["calls"]}
                for row in result["steps"] if row["kind"] == "mixed"],
            "step_kinds": [row["kind"] for row in result["steps"]]}


def summarize_events(events):
    counts = {name: sum(event["outcome"] == name for event in events)
              for name in ("capture", "replay", "cache_full", "incompatible")}
    total = sum(counts.values())
    return {"counts": counts,
            "replay_hit_rate": counts["replay"] / total if total else 0.0,
            "capture_ms": sum(event["capture_ms"] for event in events),
            "events": events}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "check", "run"))
    parser.add_argument("--suite-dir", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/results/mixed-graph-churn.json")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.action == "plan":
        print(json.dumps({"arrival_steps": [0, 0, 3, 4, 5, 6, 7, 8],
                          "prompt_tokens_each": 256, "output_tokens_each": 18,
                          "max_running": 8, "prefill_budget": 2048,
                          "full_graph_cache_limit": 4,
                          "arms": ["piecewise-varlen", "full-mixed-clone",
                                   "full-mixed-no-clone"],
                          "measurements": ["cold/warm mixed-step wall time",
                                           "capture/replay/fallback counts", "capture time",
                                           "CUDA reserved memory", "all callback logits and outputs"]}, indent=2))
        return

    if args.action == "run":
        if args.suite_dir is None:
            parser.error("run requires --suite-dir with the frozen fixed-shape workloads")
        if args.output.exists():
            parser.error(f"refusing to overwrite existing report: {args.output}")
        from benchmark_latest_vs_vllm import load_frozen, resolve_model_source
        from fixed_regime import get_fixed_case
        from types import SimpleNamespace
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
    from kernel_dispatch import _load
    _load("paged_varlen_fa3").smoke_varlen_fa3(args.device, capture_graph=True)
    if args.action == "check":
        print(json.dumps({"startup": setup, "varlen_fa3_graph_smoke": "pass",
                          "model_loaded": False}, indent=2))
        return
    import torch
    import inference_engine_cpp as cpp
    from benchmark_scheduler_decode import execute, make_config
    from model_adapter import PackedMixedPiecewiseGraphModelAdapter, allocate_pool
    engine, load_seconds, _ = load_model_only(
        source, args.device, "float16", hub_transfer=setup["hub_transfer"])
    config = make_config(cpp, case)
    blocks = config.max_batch_size * ((config.max_context_length + 15) // 16 + 1)
    pool = allocate_pool(engine.cfg, blocks, engine.device)
    options = dict(max_running=8, max_context_length=config.max_context_length,
                   decode_attention_policy="fa3", max_capture_tokens=2048,
                   max_prefill_shapes=4, prefill_buckets=[256, 512, 1024, 2048],
                   enable_residual_rmsnorm=True,
                   enable_native_decode_qkv_postprocess=True,
                   enable_prefill_swiglu_fusion=True,
                   mixed_attention_policy="fa3_varlen")

    def make_adapter(full):
        started = time.perf_counter()
        adapter = PackedMixedPiecewiseGraphModelAdapter(
            engine.model, pool, None, full_mixed_graph=full, **options)
        return adapter, (time.perf_counter() - started) * 1000

    def execute_once(adapter, observer=None):
        for tensor in pool.k_pool + pool.v_pool:
            tensor.fill_(float("nan"))
        adapter.observer = observer
        try:
            return execute(torch, cpp.IterationLoop(config, torch.device(args.device)),
                           adapter, requests)
        finally:
            adapter.observer = None

    def kv_snapshot():
        # Untimed; compare the entire cache, including untouched NaN slots.
        return [tensor.detach().cpu().clone() for tensor in pool.k_pool + pool.v_pool]

    piecewise, piecewise_init_ms = make_adapter(False)
    control = LogitObserver(torch)
    reference = execute_once(piecewise, control)
    memory_before_full = torch.cuda.memory_reserved(args.device)
    full, full_init_ms = make_adapter(True)
    candidate = LogitObserver(torch, control.rows)
    cold = execute_once(full, candidate)
    candidate.finish()
    clone_kv = kv_snapshot()
    cold_outputs_equal = reference["outputs"] == cold["outputs"]
    scheduler_equal = [(step["kind"], step["calls"]) for step in reference["steps"]] == [
        (step["kind"], step["calls"]) for step in cold["steps"]]
    cold_events = summarize_events(full.full_mixed_events)
    exercised_capture_and_fallback = (
        cold_events["counts"]["capture"] > 0 and
        cold_events["counts"]["cache_full"] > 0)
    memory_after_cold = torch.cuda.memory_reserved(args.device)
    full.clone_decode_metadata = False
    full.full_mixed_events.clear()
    no_clone_observer = LogitObserver(torch, candidate.rows)
    no_clone_correctness_run = execute_once(full, no_clone_observer)
    no_clone_observer.finish()
    no_clone_kv_equal = all(
        torch.allclose(expected, actual, atol=.05, rtol=.01, equal_nan=True)
        for expected, actual in zip(clone_kv, kv_snapshot()))
    no_clone_outputs_equal = no_clone_correctness_run["outputs"] == cold["outputs"]
    no_clone_schedule_equal = [
        (step["kind"], step["calls"]) for step in no_clone_correctness_run["steps"]
    ] == [(step["kind"], step["calls"]) for step in cold["steps"]]

    # Alternate arms with one shared graph cache; no capture/recompile is hidden
    # in either timed arm, and order is balanced against clock/thermal drift.
    warm_full, warm_no_clone = [], []
    clone_events, no_clone_events = [], []
    for use_clone in (True, False, False, True, True, False):
        full.clone_decode_metadata = use_clone
        full.full_mixed_events.clear()
        result = execute_once(full)
        if use_clone:
            warm_full.append(result)
            clone_events.extend(full.full_mixed_events)
        else:
            warm_no_clone.append(result)
            no_clone_events.extend(full.full_mixed_events)
    warm_events = summarize_events(clone_events)
    no_clone_warm_events = summarize_events(no_clone_events)
    warm_piecewise = [execute_once(piecewise) for _ in range(3)]
    warm_outputs_equal = all(result["outputs"] == reference["outputs"]
                             for result in warm_full + warm_no_clone + warm_piecewise)
    piecewise_warm_median = statistics.median(
        r["mixed_wall_ms"] for r in warm_piecewise)
    full_warm_median = statistics.median(r["mixed_wall_ms"] for r in warm_full)
    no_clone_warm_median = statistics.median(
        r["mixed_wall_ms"] for r in warm_no_clone)
    warm_saving_ms = piecewise_warm_median - full_warm_median
    cold_penalty_ms = cold["mixed_wall_ms"] - reference["mixed_wall_ms"]
    valid = (not candidate.mismatched_callbacks and cold_outputs_equal and
             scheduler_equal and warm_outputs_equal and
             not no_clone_observer.mismatched_callbacks and
             no_clone_outputs_equal and no_clone_schedule_equal and
             no_clone_kv_equal and
             exercised_capture_and_fallback and
             warm_events["counts"]["capture"] == 0 and
             no_clone_warm_events["counts"]["capture"] == 0)
    report = {"status": "complete" if valid else "invalid_comparison", "model": source,
              "model_load_seconds": load_seconds,
              "workload": {"arrivals": case["arrivals"], "lengths": case["lengths"],
                           "outputs": case["outputs"], "prefill_budget": 2048},
              "correctness": {"all_callback_logits": (
                                  "pass" if not candidate.mismatched_callbacks
                                  else "numerical_difference"),
                              "max_abs_logit_error": candidate.max_abs_error,
                              "mismatched_callbacks": candidate.mismatched_callbacks,
                              "scheduler_work": "pass" if scheduler_equal else "different",
                              "output_tokens": "pass" if cold_outputs_equal else "different",
                              "warm_output_tokens": "pass" if warm_outputs_equal else "different",
                              "no_clone_all_callback_logits": (
                                  "pass" if not no_clone_observer.mismatched_callbacks
                                  else "numerical_difference"),
                              "no_clone_max_abs_logit_error":
                                  no_clone_observer.max_abs_error,
                              "no_clone_mismatched_callbacks":
                                  no_clone_observer.mismatched_callbacks,
                              "no_clone_output_tokens": (
                                  "pass" if no_clone_outputs_equal else "different"),
                              "no_clone_scheduler_work": (
                                  "pass" if no_clone_schedule_equal else "different"),
                              "no_clone_kv_cache": (
                                  "pass" if no_clone_kv_equal else "different"),
                              "churn_exercised_capture_and_fallback":
                                  exercised_capture_and_fallback},
              "adapter_init_ms": {"piecewise": piecewise_init_ms,
                                  "full": full_init_ms},
              "cold": {"piecewise": summarize_steps(reference),
                       "full": summarize_steps(cold), "full_graph": cold_events},
              "warm": {"piecewise_mixed_wall_ms": [r["mixed_wall_ms"] for r in warm_piecewise],
                       "full_clone_mixed_wall_ms": [r["mixed_wall_ms"] for r in warm_full],
                       "full_no_clone_mixed_wall_ms": [
                           r["mixed_wall_ms"] for r in warm_no_clone],
                       "piecewise_median_mixed_wall_ms": piecewise_warm_median,
                       "full_clone_median_mixed_wall_ms": full_warm_median,
                       "full_no_clone_median_mixed_wall_ms": no_clone_warm_median,
                       "no_clone_speedup_vs_clone": (
                           full_warm_median / no_clone_warm_median
                           if no_clone_warm_median else None),
                       "full_clone_graph": warm_events,
                       "full_no_clone_graph": no_clone_warm_events},
              "amortization": {
                  "warm_saving_ms_per_workload": warm_saving_ms,
                  "cold_penalty_ms_per_workload": cold_penalty_ms,
                  "additional_repeats_to_break_even": (
                      max(0.0, cold_penalty_ms) / warm_saving_ms
                      if warm_saving_ms > 0 else None),
                  "note": "Approximate: cold correctness observers and kernel compilation "
                          "are included; compare on the same H100 before production use."},
              "cuda_reserved_before_full_bytes": memory_before_full,
              "cuda_reserved_after_cold_bytes": memory_after_cold,
              "cuda_reserved_growth_cold_bytes": memory_after_cold - memory_before_full,
              "cuda_reserved_after_warm_bytes": torch.cuda.memory_reserved(args.device)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"cold mixed: piecewise {reference['mixed_wall_ms']:.3f} ms; "
          f"full {cold['mixed_wall_ms']:.3f} ms")
    print(f"warm mixed median: piecewise {piecewise_warm_median:.3f} ms; "
          f"full clone {full_warm_median:.3f} ms; "
          f"full no-clone {no_clone_warm_median:.3f} ms")
    print(f"cold graph: {cold_events['counts']}; "
          f"capture {cold_events['capture_ms']:.3f} ms")
    print(f"warm graph clone/no-clone replay hit rate: "
          f"{warm_events['replay_hit_rate']:.1%}/{no_clone_warm_events['replay_hit_rate']:.1%}; "
          f"report: {args.output}")
    if not valid:
        print("WARNING: comparison invalid; timings are diagnostic only", file=sys.stderr)


if __name__ == "__main__":
    main()
