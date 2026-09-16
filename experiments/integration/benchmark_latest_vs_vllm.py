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
TABLE_ACTIONS = ("plan-table", "run-table", "analyze-table")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("table", "plan-table", "run-table", "analyze-table",
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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", default=MODEL)
    return parser


def contract(args):
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
    reference_files = list((args.output_dir / "reference").glob("*.json"))
    return {"ablation_complete": all(path.is_file() for path in ablation_files),
            "ablation_partial": any(path.exists() for path in ablation_files)
                                and not all(path.is_file() for path in ablation_files),
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


def run_ablation(args, case):
    load_frozen(args, case)
    output = args.output_dir / "ablation"
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
               "--warmups", str(args.warmups), "--output-dir", str(output)]
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
    for arm in ("piecewise", "piecewise_splitk"):
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
    integrated_outputs = ablation["output_ids"]
    if set(integrated_outputs) != {request.request_id for request in workload.requests}:
        raise ValueError("integrated ablation lost request outputs")
    for request in workload.requests:
        if len(integrated_outputs[request.request_id]) != request.max_tokens:
            raise ValueError("integrated ablation produced the wrong output-token count")
    total_output_tokens = sum(row["output"] for row in requests)
    integrated = {arm: total_output_tokens * 1000 /
                  statistics.median(ablation["trial_medians_ms"]["wall_ms"][arm])
                  for arm in ARMS}
    old = reference["backends"]["regime-dispatched"]["summary"]["output_throughput_tok_s"]
    vllm = reference["backends"]["vllm"]["summary"]["output_throughput_tok_s"]
    rows = {"prior_regime_dispatched_engine": old, "integrated_eager_cpp": integrated["eager"],
            "integrated_decode_graph": integrated["decode_graph"],
            "integrated_piecewise_production": integrated["piecewise"],
            "integrated_piecewise_splitk": integrated["piecewise_splitk"],
            "vllm": vllm}
    summary = {"status": "complete", "created_at": datetime.now(timezone.utc).isoformat(),
               "scope": "matched fixed burst output throughput; scheduler semantics can differ; not TTFT/SLO",
               "model_source": model_source,
               "model_revision": MODEL_REVISION if args.model == MODEL else "unverified_local_override",
               "workload": {"case_id": case["id"], "requests_sha256": requests_fingerprint(requests),
                            "output_tokens": total_output_tokens, "logical_kv_blocks": num_blocks},
               "output_throughput_tok_s": rows,
               "integrated_production_vs_old": integrated["piecewise"] / old,
               "integrated_splitk_vs_old": integrated["piecewise_splitk"] / old,
               "vllm_vs_integrated_production": vllm / integrated["piecewise"],
               "vllm_vs_integrated_splitk": vllm / integrated["piecewise_splitk"],
               "output_agreement": {"old_vs_integrated": old_outputs == integrated_outputs,
                                    "vllm_vs_integrated": vllm_outputs == integrated_outputs,
                                    "old_repetitions_identical": old_repeatable,
                                    "vllm_repetitions_identical": vllm_repeatable},
               "splitk_decision": ablation["splitk_decision"],
               "splitk_executed": splitk_executed,
               "ablation_effects": ablation["effects"],
               "reference_file": str(reference_files[0])}
    atomic_json(args.output_dir / "summary.json", summary)
    for name, value in rows.items():
        print(f"{name}: {value:.1f} output tok/s")
    print(f"piecewise production vs old: {summary['integrated_production_vs_old']:.3f}x")
    print(f"piecewise split-K vs old: {summary['integrated_splitk_vs_old']:.3f}x")
    print(f"vLLM gap vs production/split-K: "
          f"{summary['vllm_vs_integrated_production']:.3f}x / "
          f"{summary['vllm_vs_integrated_splitk']:.3f}x")


def validate_resumed_measurements(args):
    """Do not mix a weaker or differently-shaped partial run into a table sweep."""
    state = result_state(args)
    if state["ablation_partial"]:
        raise ValueError(f"partial ablation in {args.output_dir}; use a fresh suite directory")
    if state["reference_ambiguous"]:
        raise ValueError(f"multiple reference JSON files in {args.output_dir / 'reference'}")
    if state["ablation_complete"]:
        manifest = json.loads((args.output_dir / "ablation/manifest.json").read_text())
        plan = manifest["plan"]
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
        print(f"{row['shape_id']:<34} {row['piecewise_production_tok_s']:>10.1f} "
              f"{row['piecewise_splitk_tok_s']:>10.1f} {row['vllm_tok_s']:>10.1f} "
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
        plan_table(args)
        return
    if args.action == "run-table":
        run_table(args)
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
