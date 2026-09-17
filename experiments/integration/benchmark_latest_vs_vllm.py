#!/usr/bin/env python3
"""Frozen H100 checkpoint: prior regime-dispatched engine, C++/graphs, and vLLM.

Prepare the exact burst workload locally. Run the internal ablation and external
reference in separate processes, then analyze their matched full-workload output
throughput. This is a fixed-regime checkpoint, not a serving-SLO comparison.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for directory in (HERE, ROOT / "benchmarks"):
    sys.path.insert(0, str(directory))

from benchmark_core import RequestSpec, Workload
from benchmark_integrated_graph import (dry_schedule, requests_fingerprint,
                                        verify_actual_work)
from design import make_requests
from fixed_regime import (FACTORIAL_SHAPES, FIXED_SHAPES, MIN_FULL_DECODE_STEPS,
                          SPLITK_MIN_CONTEXT,
                          PREFILL_TOKENS_PER_STEP, get_fixed_case,
                          get_fixed_shape, shape_summary)

VOCAB = 151936
MODEL = "Qwen/Qwen2.5-1.5B"
MODEL_REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
ARMS = ("eager", "decode_graph", "piecewise", "piecewise_splitk")
TABLE_ACTIONS = ("plan-table", "run-table", "analyze-table", "retry-splitk-table")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("table", "plan-table", "run-table", "analyze-table", "retry-splitk-table",
                                           "plan", "prepare", "dry-schedule",
                                           "stage-model-cache", "check-model-cache",
                                           "run-profile", "run-ablation",
                                           "run-reference", "analyze"))
    parser.add_argument("--shape-id", choices=tuple(row["id"] for row in FIXED_SHAPES),
                        default=FIXED_SHAPES[0]["id"])
    parser.add_argument("--output-dir", type=Path,
                        help="fresh result directory; defaults to results/latest-vllm-checkpoint/<shape-id>")
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--logit-atol", type=float, default=.05,
                        help="absolute full-logit tolerance, recorded per cell; relative tolerance stays .01")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--profile-kind", choices=("decode", "prefill", "mixed"),
                        default="decode")
    parser.add_argument("--profile-occurrence", type=int, default=0)
    parser.add_argument("--profile-adapter", choices=("eager-prefill", "piecewise-prefill"),
                        default="piecewise-prefill")
    parser.add_argument("--profile-decode-policy", choices=("production", "splitk"),
                        default="production")
    parser.add_argument("--profile-with-stack", action="store_true")
    parser.add_argument("--include-context-probes", action="store_true",
                        help="include the two expensive 4096-token probes in table actions")
    parser.add_argument("--retry-failed", action="store_true",
                        help="archive and retry interrupted/error ablations in table actions")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", default=MODEL)
    return parser


def contract(args):
    if not 0 <= args.logit_atol < float("inf"):
        raise ValueError("logit-atol must be finite and nonnegative")
    if min(args.trials, args.samples, args.repetitions) < 1 or args.warmups < 0 or args.profile_occurrence < 0:
        raise ValueError("trials, samples, repetitions must be positive; "
                         "warmups and profile occurrence nonnegative")
    if args.device != "cuda:0":
        raise ValueError("the fixed checkpoint uses cuda:0 so vLLM and the local arms target one GPU")
    case = get_fixed_case(args.shape_id)
    if any(case["arrivals"]) or len(set(case["lengths"])) != 1:
        raise AssertionError("checkpoint requires a uniform burst cohort")
    shape = get_fixed_shape(args.shape_id)
    return case, shape_summary(shape)["logical_kv_blocks_with_headroom"]


def planned_workload(case, seed):
    requests = make_requests(case, seed, VOCAB)
    workload = Workload(
        name=f"latest-cpp-graph-vllm-{case['id']}",
        seed=seed,
        arrival_pattern="burst",
        requests=[RequestSpec(str(row["id"]), row["prompt"], row["output"])
                  for row in requests],
    )
    workload.validate()
    return workload, requests


def resolve_model_source(args):
    """Use one cached immutable snapshot for all three executors."""
    if args.model == MODEL:
        try:
            # This must run before importing huggingface_hub. RunPod images can
            # retain HF_HUB_ENABLE_HF_TRANSFER=1 without the optional package,
            # and the Hub client reads that flag during import.
            from model_setup import prepare_hub_transfer
            prepare_hub_transfer()
            from huggingface_hub import snapshot_download
            snapshot = Path(snapshot_download(repo_id=MODEL, revision=MODEL_REVISION,
                                              local_files_only=True))
            if snapshot.name != MODEL_REVISION:
                raise ValueError("Hub returned a model snapshot with the wrong resolved revision")
            validate_model_files(snapshot)
        except Exception as error:
            raise ValueError(
                f"cached Qwen snapshot {MODEL_REVISION} is missing or incomplete; run "
                "`python3 experiments/integration/benchmark_latest_vs_vllm.py "
                "stage-model-cache` before a GPU run, or pass a complete local directory "
                "with --model"
            ) from error
        return str(snapshot.resolve())
    local = Path(args.model)
    if local.is_dir():
        validate_model_files(local)
        return str(local.resolve())
    raise ValueError("--model must be the fixed Qwen repo ID or an existing local model directory")


def stage_model_cache(args):
    """Complete the immutable Hub snapshot before model loading or GPU timing."""
    if args.model != MODEL:
        source = resolve_model_source(args)
        print(f"complete local model directory: {source}")
        return
    # Disable a stale HF_HUB_ENABLE_HF_TRANSFER=1 before importing the Hub client.
    from model_setup import prepare_hub_transfer
    prepare_hub_transfer()
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(repo_id=MODEL, revision=MODEL_REVISION))
    if snapshot.name != MODEL_REVISION:
        raise ValueError("Hub downloaded a model snapshot with the wrong resolved revision")
    validate_model_files(snapshot)
    print(f"complete pinned model snapshot: {snapshot.resolve()}")


def validate_model_files(directory):
    """Fail on an incomplete offline snapshot before CUDA model construction."""
    if not (directory / "config.json").is_file():
        raise ValueError(f"{directory}: missing config.json")
    if not (directory / "tokenizer_config.json").is_file():
        raise ValueError(f"{directory}: missing tokenizer_config.json needed by vLLM")
    if not ((directory / "tokenizer.json").is_file()
            or (directory / "tokenizer.model").is_file()
            or ((directory / "vocab.json").is_file() and (directory / "merges.txt").is_file())):
        raise ValueError(f"{directory}: missing tokenizer assets needed by vLLM")
    safetensors_indices = list(directory.glob("*.safetensors.index.json"))
    indices = safetensors_indices or list(directory.glob("*.bin.index.json"))
    if indices:
        for index in indices:
            shards = set(json.loads(index.read_text())["weight_map"].values())
            missing = sorted(name for name in shards if not (directory / name).is_file())
            if missing:
                raise ValueError(f"{directory}: missing weight shards: {missing[:3]}")
    elif not any(path.is_file() for path in
                 (*directory.glob("*.safetensors"), *directory.glob("*.bin"))):
        raise ValueError(f"{directory}: missing model weight files")


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def table_shapes(args):
    return FIXED_SHAPES if args.include_context_probes else FACTORIAL_SHAPES


def cell_args(args, shape_id):
    values = vars(args).copy()
    values["shape_id"] = shape_id
    values["output_dir"] = args.output_dir / shape_id
    return argparse.Namespace(**values)


def prepare_or_validate(args, case):
    path = workload_path(args)
    if path.exists():
        load_frozen(args, case)
        return "validated"
    workload, _ = planned_workload(case, args.seed)
    atomic_json(path, workload.to_dict())
    return "created"


def result_state(args):
    ablation_files = [args.output_dir / "ablation" / name
                      for name in ("manifest.json", "report.json")]
    ablation_complete = False
    if all(path.is_file() for path in ablation_files):
        try:
            ablation_complete = json.loads(ablation_files[1].read_text()).get("status") == "ok"
        except (OSError, ValueError):
            pass
    reference_files = list((args.output_dir / "reference").glob("*.json"))
    return {"ablation_complete": ablation_complete,
            "ablation_partial": any(path.exists() for path in ablation_files)
                                and not ablation_complete,
            "reference_complete": len(reference_files) == 1,
            "reference_ambiguous": len(reference_files) > 1,
            "analysis_complete": (args.output_dir / "summary.json").is_file()}


def workload_path(args):
    return args.output_dir / "workload.json"


def load_frozen(args, case):
    path = workload_path(args)
    if not path.is_file():
        raise ValueError(f"missing {path}; run prepare first")
    workload = Workload.from_dict(json.loads(path.read_text()))
    expected, requests = planned_workload(case, args.seed)
    if workload.to_dict() != expected.to_dict():
        raise ValueError("frozen workload differs from the planned case/seed; use a fresh output dir")
    return workload, requests


def reference_outputs(result, workload):
    expected_ids = {request.request_id for request in workload.requests}
    maps = []
    for run in result["runs"]:
        values = {row["request_id"]: row["output_ids"] for row in run["requests"]}
        if set(values) != expected_ids:
            raise ValueError("reference run lost or duplicated request outputs")
        for request in workload.requests:
            if len(values[request.request_id]) != request.max_tokens:
                raise ValueError("reference run produced the wrong output-token count")
        maps.append(values)
    if not maps:
        raise ValueError("reference backend has no measured runs")
    return maps[0], all(values == maps[0] for values in maps[1:])


def token_agreement(expected, actual):
    total = matched = 0
    first = None
    if set(expected) != set(actual):
        raise ValueError("output request IDs differ")
    for request_id, tokens in expected.items():
        candidate = actual[request_id]
        if len(tokens) != len(candidate):
            raise ValueError("output lengths differ")
        for position, (left, right) in enumerate(zip(tokens, candidate)):
            total += 1
            matched += left == right
            if left != right and first is None:
                first = {"request_id": request_id, "position": position,
                         "reference_token": left, "candidate_token": right}
    return {"matched_tokens": matched, "total_tokens": total,
            "match_fraction": matched / total if total else 1.0,
            "exact": matched == total, "first_difference": first}


def run_ablation(args, case, *, splitk_only=False):
    load_frozen(args, case)
    output = args.output_dir / ("splitk-retry" if splitk_only else "ablation")
    if any((output / name).exists() for name in ("manifest.json", "report.json")):
        raise ValueError("ablation output already exists; refusing to overwrite it")
    model_source = resolve_model_source(args)
    command = [sys.executable, str(HERE / "benchmark_integrated_graph.py"),
               "--preset", "fixed", "--case-id", case["id"],
               "--model", model_source, "--device", args.device,
               "--seed", str(args.seed), "--workload-in", str(workload_path(args)),
               "--expected-max-decode-batch", str(case["max_running"]),
               "--min-full-decode-steps", str(MIN_FULL_DECODE_STEPS),
               "--expected-prefill-tokens", str(PREFILL_TOKENS_PER_STEP),
               "--prefill-buckets", str(PREFILL_TOKENS_PER_STEP),
               "--trials", str(args.trials), "--samples", str(args.samples),
               "--logit-atol", str(args.logit_atol),
               "--warmups", str(args.warmups), "--output-dir", str(output)]
    if splitk_only:
        command.append("--splitk-only")
    subprocess.run(command, cwd=ROOT, check=True)


def run_profile(args, case):
    load_frozen(args, case)
    model_source = resolve_model_source(args)
    command = [sys.executable, str(HERE / "profile_cpp_control.py"),
               "--preset", "fixed", "--case-id", case["id"],
               "--model", model_source, "--device", args.device,
               "--seed", str(args.seed), "--workload-in", str(workload_path(args)),
               "--kind", args.profile_kind, "--occurrence", str(args.profile_occurrence),
               "--adapter", args.profile_adapter,
               "--decode-attention-policy", args.profile_decode_policy,
               "--warmups", str(args.warmups), "--repetitions", str(args.repetitions),
               "--output-dir", str(args.output_dir / "profile")]
    if args.profile_with_stack:
        command.append("--with-stack")
    subprocess.run(command, cwd=ROOT, check=True)


def check_schedule(args, case):
    _, requests = load_frozen(args, case)
    import torch
    import inference_engine_cpp as cpp

    work = verify_actual_work(dry_schedule(torch, cpp, case, args.seed, requests),
                              case["max_running"], PREFILL_TOKENS_PER_STEP,
                              MIN_FULL_DECODE_STEPS)
    expected = get_fixed_shape(args.shape_id)["expected_full_decode_steps"]
    if work["pure_full_decode_steps_at_target_batch"] != expected:
        raise ValueError("CPU C++ schedule changed from the frozen table's expected full-batch count")
    print(json.dumps(work, indent=2))


def run_reference(args, case, num_blocks):
    load_frozen(args, case)
    output = args.output_dir / "reference"
    if list(output.glob("*.json")):
        raise ValueError("reference result already exists; refusing to add an ambiguous run")
    if importlib.util.find_spec("vllm") is None:
        raise ValueError("vLLM is not installed; install pinned benchmark requirements first")
    model_source = resolve_model_source(args)
    command = [sys.executable, str(ROOT / "benchmarks/run_benchmarks.py"),
               "--backends", "regime-dispatched,vllm", "--strict-backends",
               "--model", model_source, "--device", args.device, "--dtype", "float16",
               "--block-size", "16", "--max-running", str(case["max_running"]),
               "--max-num-batched-tokens", str(PREFILL_TOKENS_PER_STEP),
               "--num-blocks", str(num_blocks),
               "--vllm-kv-cache-mode", "matched", "--workload-in", str(workload_path(args)),
               "--warmups", str(args.warmups), "--repetitions", str(args.repetitions),
               "--seed", str(args.seed), "--output-dir", str(output)]
    environment = os.environ.copy()
    if environment.get("HF_HUB_ENABLE_HF_TRANSFER", "").lower() in ("1", "on", "yes", "true"):
        if importlib.util.find_spec("hf_transfer") is None:
            environment["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def analyze(args, case, num_blocks):
    workload, requests = load_frozen(args, case)
    ablation_manifest = json.loads((args.output_dir / "ablation/manifest.json").read_text())
    ablation = json.loads((args.output_dir / "ablation/report.json").read_text())
    retry_report = args.output_dir / "splitk-retry/report.json"
    retry_provenance = None
    if retry_report.is_file():
        retry = json.loads(retry_report.read_text())
        if retry.get("status") != "ok" or not retry.get("splitk_only"):
            raise ValueError("split-K retry is incomplete; retry it before analyzing")
        manifest = json.loads((args.output_dir / "splitk-retry/manifest.json").read_text())
        if (retry["case_id"] != case["id"] or manifest["model"] != ablation_manifest["model"]
                or manifest["plan"]["requests_sha256"] != requests_fingerprint(requests)
                or retry["actual_work"] != ablation["actual_work"]):
            raise ValueError("split-K retry does not match the original workload/model/schedule")
        arm = "piecewise_splitk"
        for field in ("checks", "capture", "measurements"):
            ablation.setdefault(field, {})[arm] = retry[field][arm]
        for phase, values in retry["trial_medians_ms"].items():
            ablation["trial_medians_ms"][phase][arm] = values[arm]
        outputs = ablation.setdefault("output_ids_by_arm", {a: ablation["output_ids"] for a in ARMS})
        outputs[arm] = retry["output_ids_by_arm"][arm]
        ablation.get("rejected_arms", {}).pop(arm, None)
        ablation["splitk_decision"] = retry["splitk_decision"]
        ablation["effects"] = {k: v for k, v in ablation["effects"].items() if "splitk" not in k}
        retry_provenance = {"report": str(retry_report), "system": manifest.get("system"),
                            "logit_tolerance": retry["logit_tolerance"],
                            "comparison": "separate run; not a paired split-K speedup experiment"}
    reference_files = list((args.output_dir / "reference").glob("*.json"))
    if len(reference_files) != 1:
        raise ValueError("expected exactly one reference JSON result")
    reference = json.loads(reference_files[0].read_text())
    if ablation.get("status") != "ok":
        raise ValueError("ablation did not complete successfully")
    if ablation["case_id"] != case["id"]:
        raise ValueError("ablation shape ID differs from the fixed input row")
    actual = ablation["actual_work"]
    if (actual["max_actual_decode_batch"] != case["max_running"]
            or actual["pure_full_decode_steps_at_target_batch"] !=
            get_fixed_shape(args.shape_id)["expected_full_decode_steps"]
            or PREFILL_TOKENS_PER_STEP not in actual["prefill_token_counts"]):
        raise ValueError("ablation did not meet the fixed input shape gates")
    rejected = ablation.get("rejected_arms", {})
    if set(rejected) - {"piecewise_splitk"}:
        raise ValueError("a required production arm was rejected")
    if rejected and (rejected["piecewise_splitk"].get("status") != "rejected_numerical_correctness"
                     or ablation["splitk_decision"]["choice"] != "rejected_numerical_correctness"
                     or "piecewise_splitk" in ablation["trial_medians_ms"]["wall_ms"]):
        raise ValueError("invalid split-K rejection record")
    active_arms = tuple(arm for arm in ARMS if arm not in rejected)
    for arm in (arm for arm in ("piecewise", "piecewise_splitk") if arm not in rejected):
        if ablation["capture"][arm]["timed_prefill_capture_calls"] < 1:
            raise ValueError(f"{arm} did not use piecewise capture during timing")
    splitk_expected = (shape_summary(get_fixed_shape(args.shape_id))["max_context_tokens_per_request"]
                       >= SPLITK_MIN_CONTEXT)
    splitk_executed = any(key != "production" for key in
                          ablation["checks"]["piecewise_splitk"]["decisions"])
    if splitk_executed != splitk_expected:
        raise ValueError("split-K execution does not match the frozen context regime")
    if reference["workload"] != workload.to_dict():
        raise ValueError("reference prompt IDs/outputs differ from the frozen workload")
    if ablation_manifest["plan"]["requests_sha256"] != requests_fingerprint(requests):
        raise ValueError("integrated prompts differ from the frozen workload")
    model_source = ablation_manifest["model"]
    if model_source != reference["configuration"]["model"]:
        raise ValueError("model identity differs across runs")
    if args.model == MODEL and Path(model_source).name != MODEL_REVISION:
        raise ValueError("benchmark did not use the pinned Qwen model revision")
    config = reference["configuration"]
    if (config["dtype"] != "float16" or config["block_size"] != 16
            or config["max_running"] != case["max_running"]
            or config["max_num_batched_tokens"] != PREFILL_TOKENS_PER_STEP
            or config["num_blocks"] != num_blocks or config["vllm_kv_cache_mode"] != "matched"):
        raise ValueError("reference configuration does not match the fixed checkpoint contract")
    reference_contract = reference["comparison_contract"]
    if (reference_contract["sampling"] != "greedy-temperature-0-ignore-eos"
            or reference_contract["max_concurrent_sequences"] != case["max_running"]
            or not reference_contract["identical_prompt_token_ids"]
            or not reference_contract["identical_requested_output_lengths"]):
        raise ValueError("reference sampling or workload contract differs")
    if set(("regime-dispatched", "vllm")) - set(reference["backends"]):
        raise ValueError("old engine or vLLM reference is missing")
    for run in reference["backends"]["vllm"]["runs"]:
        metadata = run["metadata"]
        if (metadata["max_num_seqs"] != case["max_running"]
                or metadata["max_num_batched_tokens"] != PREFILL_TOKENS_PER_STEP
                or metadata["max_model_len"] != shape_summary(get_fixed_shape(args.shape_id))["max_context_tokens_per_request"]
                or metadata["kv_cache_mode"] != "matched-local-pool"
                or metadata["prefix_caching"] is not False
                or metadata["cuda_graphs"] is not True):
            raise ValueError("vLLM execution settings differ from the matched checkpoint")
    old_outputs, old_repeatable = reference_outputs(reference["backends"]["regime-dispatched"],
                                                    workload)
    vllm_outputs, vllm_repeatable = reference_outputs(reference["backends"]["vllm"], workload)
    outputs_by_arm = ablation.get("output_ids_by_arm", {arm: ablation["output_ids"] for arm in ARMS})
    integrated_outputs = outputs_by_arm["piecewise"]
    agreements = {arm: {"vs_eager": token_agreement(outputs_by_arm["eager"], outputs_by_arm[arm]),
                        "vs_vllm": token_agreement(vllm_outputs, outputs_by_arm[arm]),
                        "vs_old": token_agreement(old_outputs, outputs_by_arm[arm])}
                  for arm in active_arms}
    if set(integrated_outputs) != {request.request_id for request in workload.requests}:
        raise ValueError("integrated ablation lost request outputs")
    for request in workload.requests:
        if len(integrated_outputs[request.request_id]) != request.max_tokens:
            raise ValueError("integrated ablation produced the wrong output-token count")
    total_output_tokens = sum(row["output"] for row in requests)
    integrated = {arm: total_output_tokens * 1000 /
                  statistics.median(ablation["trial_medians_ms"]["wall_ms"][arm])
                  for arm in active_arms}
    splitk_rate = integrated.get("piecewise_splitk")
    old = reference["backends"]["regime-dispatched"]["summary"]["output_throughput_tok_s"]
    vllm = reference["backends"]["vllm"]["summary"]["output_throughput_tok_s"]
    rows = {"prior_regime_dispatched_engine": old, "integrated_eager_cpp": integrated["eager"],
            "integrated_decode_graph": integrated["decode_graph"],
            "integrated_piecewise_production": integrated["piecewise"],
            "integrated_piecewise_splitk": splitk_rate,
            "vllm": vllm}
    summary = {"status": "complete", "created_at": datetime.now(timezone.utc).isoformat(),
               "scope": "matched fixed burst output throughput; scheduler semantics can differ; not TTFT/SLO",
               "model_source": model_source,
               "model_revision": MODEL_REVISION if args.model == MODEL else "unverified_local_override",
               "workload": {"case_id": case["id"], "requests_sha256": requests_fingerprint(requests),
                            "output_tokens": total_output_tokens, "logical_kv_blocks": num_blocks},
               "output_throughput_tok_s": rows,
               "integrated_production_vs_old": integrated["piecewise"] / old,
               "integrated_splitk_vs_old": splitk_rate / old if splitk_rate is not None else None,
               "vllm_vs_integrated_production": vllm / integrated["piecewise"],
               "vllm_vs_integrated_splitk": vllm / splitk_rate if splitk_rate is not None else None,
               "output_agreement": {"old_vs_integrated": old_outputs == integrated_outputs,
                                    "vllm_vs_integrated": vllm_outputs == integrated_outputs,
                                    "old_repetitions_identical": old_repeatable,
                                    "vllm_repetitions_identical": vllm_repeatable},
               "splitk_decision": ablation["splitk_decision"],
               "rejected_arms": rejected,
               "output_agreement_by_arm": agreements,
               "correctness_mode": ablation.get("correctness_mode", "legacy-exact-trajectory"),
               "logit_tolerance": ablation.get("logit_tolerance", {"atol": .05, "rtol": .01}),
               "numerical_checks": ablation["checks"],
               "splitk_retry": retry_provenance,
               "splitk_executed": splitk_executed,
               "ablation_effects": ablation["effects"],
               "reference_file": str(reference_files[0])}
    atomic_json(args.output_dir / "summary.json", summary)
    for name, value in rows.items():
        print(f"{name}: {value:.1f} output tok/s" if value is not None else
              f"{name}: REJECTED (numerical correctness); no performance result")
    for arm, check in ablation["checks"].items():
        if check.get("numerical_validation_passed") is False:
            print(f"{arm}: NUMERICAL WARNING; max absolute logit error={check['max_logit_error']:.6g}, "
                  f"outside tolerance={check['logits_outside_tolerance']}/{check['logits_compared']}; timing retained")
    print(f"piecewise production vs old: {summary['integrated_production_vs_old']:.3f}x")
    if splitk_rate is not None:
        print(f"piecewise split-K vs old: {summary['integrated_splitk_vs_old']:.3f}x")
        print(f"vLLM gap vs split-K: {summary['vllm_vs_integrated_splitk']:.3f}x")
    print(f"vLLM gap vs production: {summary['vllm_vs_integrated_production']:.3f}x")


def validate_resumed_measurements(args):
    """Do not mix a weaker or differently-shaped partial run into a table sweep."""
    state = result_state(args)
    if state["ablation_partial"]:
        if not args.retry_failed or args.action != "run-table":
            raise ValueError(f"interrupted/error ablation in {args.output_dir}; rerun the table "
                             "with --retry-failed to archive and retry it")
        source = args.output_dir / "ablation"
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        archive = args.output_dir / f"ablation-failed-{timestamp}"
        source.replace(archive)
        print(f"archived failed ablation: {archive}", flush=True)
        state = result_state(args)
    if state["reference_ambiguous"]:
        raise ValueError(f"multiple reference JSON files in {args.output_dir / 'reference'}")
    if state["ablation_complete"]:
        manifest = json.loads((args.output_dir / "ablation/manifest.json").read_text())
        plan = manifest["plan"]
        case = get_fixed_case(args.shape_id)
        _, requests = load_frozen(args, case)
        report = json.loads((args.output_dir / "ablation/report.json").read_text())
        recorded_atol = report.get("logit_tolerance", {"atol": .05})["atol"]
        if recorded_atol > args.logit_atol:
            raise ValueError("existing ablation used a looser logit tolerance; use a fresh cell directory")
        if report.get("rejected_arms") and recorded_atol < args.logit_atol:
            raise ValueError("existing split-K rejection used a stricter tolerance; use a fresh cell directory")
        if (report.get("case_id") != args.shape_id or
                plan["requests_sha256"] != requests_fingerprint(requests)):
            raise ValueError("existing ablation shape/workload differs; use a fresh cell directory")
        requested = {"trials": args.trials, "samples": args.samples,
                     "warmups": args.warmups}
        for name, value in requested.items():
            if int(plan[name]) < value:
                raise ValueError(f"existing {args.shape_id} ablation has {name}={plan[name]}, "
                                 f"below requested {value}; use a fresh suite directory")
    if state["reference_complete"]:
        reference_file = next((args.output_dir / "reference").glob("*.json"))
        config = json.loads(reference_file.read_text())["configuration"]
        if int(config["repetitions"]) < args.repetitions or int(config["warmups"]) < args.warmups:
            raise ValueError(f"existing {args.shape_id} reference is weaker than the requested "
                             "repetitions/warmups; use a fresh suite directory")
    return state


def aggregate_table(args):
    shapes = table_shapes(args)
    rows = []
    for shape in shapes:
        child = cell_args(args, shape["id"])
        path = child.output_dir / "summary.json"
        if not path.is_file():
            raise ValueError(f"missing analyzed table cell: {path}")
        summary = json.loads(path.read_text())
        if (summary.get("status") != "complete"
                or summary["workload"]["case_id"] != shape["id"]):
            raise ValueError(f"invalid analyzed table cell: {path}")
        throughput = summary["output_throughput_tok_s"]
        agreement = summary["output_agreement"]
        rows.append({
            "shape_id": shape["id"], "batch": shape["batch"],
            "prompt_length": shape["prompt_length"],
            "output_length": shape["output_length"],
            "splitk_executed": summary["splitk_executed"],
            "splitk_choice": summary["splitk_decision"]["choice"],
            "numerical_checks": json.dumps(summary.get("numerical_checks", {}), sort_keys=True),
            "old_tok_s": throughput["prior_regime_dispatched_engine"],
            "eager_cpp_tok_s": throughput["integrated_eager_cpp"],
            "decode_graph_tok_s": throughput["integrated_decode_graph"],
            "piecewise_production_tok_s": throughput["integrated_piecewise_production"],
            "piecewise_splitk_tok_s": throughput["integrated_piecewise_splitk"],
            "vllm_tok_s": throughput["vllm"],
            "production_vs_old": summary["integrated_production_vs_old"],
            "splitk_vs_old": summary["integrated_splitk_vs_old"],
            "vllm_vs_production": summary["vllm_vs_integrated_production"],
            "vllm_vs_splitk": summary["vllm_vs_integrated_splitk"],
            "old_vs_integrated_exact": agreement["old_vs_integrated"],
            "vllm_vs_integrated_exact": agreement["vllm_vs_integrated"],
        })
    payload = {"status": "complete", "created_at": datetime.now(timezone.utc).isoformat(),
               "scope": "eight factorial cells" if not args.include_context_probes
                        else "eight factorial cells plus two context probes",
               "shape_ids": [row["shape_id"] for row in rows], "rows": rows}
    atomic_json(args.output_dir / "table-summary.json", payload)
    csv_path = args.output_dir / "table-summary.csv"
    temporary = csv_path.with_suffix(".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(csv_path)
    print("\nshape                              production    split-K       vLLM  vLLM/prod")
    for row in rows:
        splitk_text = (f"{row['piecewise_splitk_tok_s']:.1f}"
                       if row['piecewise_splitk_tok_s'] is not None else "REJECTED")
        print(f"{row['shape_id']:<34} {row['piecewise_production_tok_s']:>10.1f} "
              f"{splitk_text:>10} {row['vllm_tok_s']:>10.1f} "
              f"{row['vllm_vs_production']:>10.3f}x")
    print(f"\naggregate JSON: {args.output_dir / 'table-summary.json'}")
    print(f"aggregate CSV:  {csv_path}")


def plan_table(args):
    rows = []
    totals = {"timed_ablation_prompt_tokens": 0, "timed_ablation_output_tokens": 0,
              "timed_reference_prompt_tokens": 0, "timed_reference_output_tokens": 0}
    for shape in table_shapes(args):
        summary = shape_summary(shape)
        ablation_workloads = len(ARMS) * args.trials * args.samples
        reference_workloads = 2 * args.repetitions
        row = {"shape_id": shape["id"], "ablation_workloads": ablation_workloads,
               "reference_workloads": reference_workloads,
               "timed_ablation_prompt_tokens": ablation_workloads * summary["total_prompt_tokens"],
               "timed_ablation_output_tokens": ablation_workloads * summary["total_output_tokens"],
               "timed_reference_prompt_tokens": reference_workloads * summary["total_prompt_tokens"],
               "timed_reference_output_tokens": reference_workloads * summary["total_output_tokens"]}
        rows.append(row)
        for name in totals:
            totals[name] += row[name]
    print(json.dumps({"rows": rows, "totals": totals,
                      "plus_warmups_and_correctness_per_cell": True,
                      "output_dir": str(args.output_dir)}, indent=2))


def retry_splitk_table(args):
    """Recover omitted split-K timings without rerunning production or vLLM."""
    cells = []
    # Check the entire original suite before spending GPU time on recovery.
    for shape in table_shapes(args):
        child = cell_args(args, shape["id"])
        case, blocks = contract(child)
        load_frozen(child, case)
        path = child.output_dir / "ablation/report.json"
        if not path.is_file() or json.loads(path.read_text()).get("status") != "ok":
            raise ValueError(f"original sweep must finish first: {path}")
        if not (child.output_dir / "summary.json").is_file():
            raise ValueError(f"original reference/analysis must finish first: {child.output_dir}")
        report = json.loads(path.read_text())
        cells.append((child, case, blocks, "piecewise_splitk" in report.get("rejected_arms", {})))
    for child, case, blocks, missing in cells:
        if not missing:
            print(f"{case['id']}: split-K timing already present; skipping", flush=True)
            continue
        output = child.output_dir / "splitk-retry"
        report_path, manifest_path = output / "report.json", output / "manifest.json"
        complete = (report_path.is_file() and manifest_path.is_file()
                    and json.loads(report_path.read_text()).get("status") == "ok")
        if output.exists() and not complete:
            if not args.retry_failed:
                raise ValueError(f"interrupted retry at {output}; use --retry-failed")
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            archive = output.with_name(f"splitk-retry-interrupted-{stamp}")
            output.rename(archive)
            print(f"archived interrupted retry: {archive}", flush=True)
        if not complete:
            print(f"{case['id']}: validating and timing split-K only", flush=True)
            run_ablation(child, case, splitk_only=True)
        analyze(child, case, blocks)
    aggregate_table(args)


def run_table(args):
    # Fail before launching any measured cell if the two external prerequisites are absent.
    resolve_model_source(args)
    if importlib.util.find_spec("vllm") is None:
        raise ValueError("vLLM is not installed in this interpreter")
    for index, shape in enumerate(table_shapes(args), 1):
        child = cell_args(args, shape["id"])
        case, num_blocks = contract(child)
        print(f"\n[{index}/{len(table_shapes(args))}] {shape['id']}", flush=True)
        prepared = prepare_or_validate(child, case)
        print(f"workload {prepared}: {workload_path(child)}", flush=True)
        check_schedule(child, case)
        state = validate_resumed_measurements(child)
        if not state["ablation_complete"]:
            run_ablation(child, case)
        else:
            print("resuming: validated existing ablation", flush=True)
        if not state["reference_complete"]:
            run_reference(child, case, num_blocks)
        else:
            print("resuming: validated existing reference", flush=True)
        analyze(child, case, num_blocks)
    aggregate_table(args)


def main():
    args = build_parser().parse_args()
    if args.output_dir is None:
        base = ROOT / "experiments/results/latest-vllm-checkpoint"
        args.output_dir = base if args.action in TABLE_ACTIONS else base / args.shape_id
    if args.action == "table":
        print(json.dumps([shape_summary(row) for row in FIXED_SHAPES], indent=2))
        return
    if args.action == "plan-table":
        contract(args)
        plan_table(args)
        return
    if args.action == "run-table":
        run_table(args)
        return
    if args.action == "retry-splitk-table":
        retry_splitk_table(args)
        return
    if args.action == "analyze-table":
        for shape in table_shapes(args):
            child = cell_args(args, shape["id"])
            case, num_blocks = contract(child)
            validate_resumed_measurements(child)
            analyze(child, case, num_blocks)
        aggregate_table(args)
        return
    case, num_blocks = contract(args)
    workload, requests = planned_workload(case, args.seed)
    if args.action == "plan":
        shape = shape_summary(get_fixed_shape(args.shape_id))
        ablation_workloads = len(ARMS) * args.trials * args.samples
        reference_workloads = 2 * args.repetitions
        print(json.dumps({"shape": shape,
                          "requests_sha256": requests_fingerprint(requests),
                          "ablation_workloads": ablation_workloads,
                          "reference_workloads": reference_workloads,
                          "timed_ablation_prompt_tokens": ablation_workloads * shape["total_prompt_tokens"],
                          "timed_ablation_output_tokens": ablation_workloads * shape["total_output_tokens"],
                          "timed_reference_prompt_tokens": reference_workloads * shape["total_prompt_tokens"],
                          "timed_reference_output_tokens": reference_workloads * shape["total_output_tokens"],
                          "plus_warmups_and_correctness": True,
                          "output_dir": str(args.output_dir)}, indent=2))
    elif args.action == "prepare":
        path = workload_path(args)
        if path.exists():
            raise ValueError(f"{path} already exists; refusing to overwrite it")
        atomic_json(path, workload.to_dict())
        print(f"frozen workload: {path}")
    elif args.action == "dry-schedule":
        check_schedule(args, case)
    elif args.action == "stage-model-cache":
        stage_model_cache(args)
    elif args.action == "check-model-cache":
        print(f"model snapshot: {resolve_model_source(args)}")
    elif args.action == "run-profile":
        run_profile(args, case)
    elif args.action == "run-ablation":
        run_ablation(args, case)
    elif args.action == "run-reference":
        run_reference(args, case, num_blocks)
    else:
        analyze(args, case, num_blocks)


if __name__ == "__main__":
    main()
