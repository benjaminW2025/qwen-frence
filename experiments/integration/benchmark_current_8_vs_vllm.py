#!/usr/bin/env python3
"""Eight frozen burst workloads: current C++/FA3/graph engine versus vLLM.

Each backend runs in its own process. Completed cell stages can be resumed; an
incomplete or mismatched result is never silently treated as a measurement.
The factorial burst workloads do not exercise the mixed-only packed callback.
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
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for directory in (HERE, ROOT / "baseline", ROOT / "benchmarks",
                  ROOT / "engine/model_runner", ROOT / "engine/kvcache",
                  ROOT / "engine/cpp/build"):
    sys.path.insert(0, str(directory))

from benchmark_latest_vs_vllm import load_frozen, resolve_model_source
from fixed_regime import (FACTORIAL_SHAPES, PREFILL_TOKENS_PER_STEP,
                          get_fixed_case, shape_summary, verify_fixed_result)

SHAPES = tuple(row["id"] for row in FACTORIAL_SHAPES)
ENGINE_FLAGS = {
    "cpp_scheduler": True,
    "packed_mixed_step": True,
    "decode_attention_policy": "fa3",
    "decode_graph_bucket": "exact_batch",
    "prefill_graph_buckets": "CPU-scheduled pure/mixed exact upper bounds",
    "residual_rmsnorm": True,
    "native_decode_qkv_postprocess": True,
    "prefill_swiglu_fusion": True,
    "prefill_packed_qkv_rope_cache": False,
}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("action", choices=("plan", "check", "run-table", "run-cell",
                                           "run-local", "run-vllm", "analyze"))
    result.add_argument("--suite-dir", type=Path, required=True,
                        help="frozen checkpoint directory with all eight workload.json files")
    result.add_argument("--output-dir", type=Path, required=True,
                        help="new/resumable directory for this comparison")
    result.add_argument("--shape-id", choices=SHAPES)
    result.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--seed", type=int, default=20260914)
    result.add_argument("--warmups", type=int, default=1)
    result.add_argument("--repetitions", type=int, default=3)
    result.add_argument("--reuse-vllm-from", type=Path,
                        help="explicitly reuse validated burst vLLM results from a prior "
                             "output directory on the same GPU pod")
    result.add_argument("--resume-commit", action="append", help="explicitly permit "
                        "completed results from a prior benchmark commit (7+ hex chars); "
                        "repeat for multiple prior commits")
    return result


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def repository_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                   cwd=ROOT, text=True).strip()


def commit_matches(recorded, current, resume_commit=None):
    allowed = (resume_commit,) if isinstance(resume_commit, str) else (resume_commit or ())
    return recorded == current or bool(recorded and any(
        recorded.startswith(prefix) for prefix in allowed))


def resume_options(resume_commit):
    allowed = (resume_commit,) if isinstance(resume_commit, str) else (resume_commit or ())
    return [option for prefix in allowed for option in ("--resume-commit", prefix)]


def input_contract(args, shape_id):
    from benchmark_core import Workload
    from benchmark_latest_vs_vllm import planned_workload

    case = get_fixed_case(shape_id)
    source = args.suite_dir / shape_id / "workload.json"
    if not source.is_file():
        raise ValueError(f"missing frozen workload: {source}")
    workload, requests = load_frozen(
        SimpleNamespace(output_dir=source.parent, seed=args.seed), case)
    expected, _ = planned_workload(case, args.seed)
    if not isinstance(workload, Workload) or workload.to_dict() != expected.to_dict():
        raise ValueError(f"{shape_id}: frozen workload contract differs")
    digest = hashlib.sha256(json.dumps(workload.to_dict(), sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()
    blocks = shape_summary(next(row for row in FACTORIAL_SHAPES
                                if row["id"] == shape_id))["logical_kv_blocks_with_headroom"]
    return case, requests, workload, digest, blocks


def stage_paths(args, shape_id):
    cell = args.output_dir / shape_id
    return cell / "local.json", cell / "vllm", cell / "comparison.json"


def dispatch_plan(case, requests, seed):
    """Resolve graph buckets and expected mixed dispatch before loading weights."""
    import torch
    import inference_engine_cpp as cpp
    from benchmark_integrated_graph import dry_schedule
    from naive_forward import SWIGLU_FUSION_ROW_THRESHOLD

    schedule = dry_schedule(torch, cpp, case, seed, requests)
    packed = [sum(call[1] for call in step["calls"])
              for step in schedule["steps"] if step["kind"] == "mixed"]
    prefill = [call[1] for step in schedule["steps"]
               for call in step["calls"] if not call[0]]
    if not prefill or max(prefill) != PREFILL_TOKENS_PER_STEP:
        raise AssertionError(f"{case['id']}: CPU prefill work differs from fixed budget")
    buckets = sorted({PREFILL_TOKENS_PER_STEP, max(packed, default=0)})
    buckets = [bucket for bucket in buckets if bucket > 0]
    return {"prefill_buckets": buckets, "mixed_steps": len(packed),
            "max_packed_mixed_tokens": max(packed, default=0),
            "swiglu_fused_buckets": [bucket for bucket in buckets
                                     if bucket > SWIGLU_FUSION_ROW_THRESHOLD]}


def adapter_options(case, buckets):
    if not buckets or min(buckets) < 1:
        raise ValueError("burst graph buckets must be positive")
    return dict(max_running=case["max_running"],
                max_context_length=max(length + output for length, output in
                                       zip(case["lengths"], case["outputs"])),
                decode_attention_policy="fa3",
                decode_buckets=[case["max_running"]],
                max_capture_tokens=max(buckets),
                max_prefill_shapes=len(buckets), prefill_buckets=buckets,
                enable_residual_rmsnorm=True,
                enable_native_decode_qkv_postprocess=True,
                enable_prefill_swiglu_fusion=True)


def validate_capture_options(options):
    """Exercise the real graph bucket constructor without CUDA/model weights."""
    graph_dir = ROOT / "engine/graph"
    if str(graph_dir) not in sys.path:
        sys.path.insert(0, str(graph_dir))
    from piecewise_prefill import PiecewisePrefill

    model = SimpleNamespace(layers=[object()])
    pool = SimpleNamespace(k_pool=[SimpleNamespace(
        device=SimpleNamespace(type="cuda"))])
    graph = PiecewisePrefill(
        model, pool, max_capture_tokens=options["max_capture_tokens"],
        max_shapes=options["max_prefill_shapes"],
        token_buckets=options["prefill_buckets"])
    if graph.buckets != tuple(sorted(set(options["prefill_buckets"]))):
        raise AssertionError("captured graph buckets differ from plan")


def local_is_complete(path, digest, *, model=None, blocks=None,
                      warmups=None, repetitions=None, commit=None,
                      resume_commit=None):
    if not path.is_file():
        return False
    row = json.loads(path.read_text())
    if (row.get("status") != "complete" or row.get("workload_sha256") != digest
            or row.get("engine_flags") != ENGINE_FLAGS
            or (model is not None and row.get("model") != model)
            or (blocks is not None and row.get("num_blocks") != blocks)
            or (warmups is not None and row.get("warmups") != warmups)
            or (repetitions is not None and row.get("repetitions") != repetitions)
            or (commit is not None and not commit_matches(
                row.get("repository_commit"), commit, resume_commit))):
        raise ValueError(f"stale or mismatched local result: {path}")
    return True


def vllm_result(directory, workload, digest, case, blocks, model,
                warmups, repetitions, commit=None, resume_commit=None):
    files = sorted(directory.glob("*.json"))
    if not files:
        return None
    if len(files) != 1:
        raise ValueError(f"expected exactly one vLLM JSON in {directory}; found {len(files)}")
    payload = json.loads(files[0].read_text())
    from run_benchmarks import workload_fingerprint
    if payload.get("workload") != workload.to_dict():
        raise ValueError(f"{files[0]}: vLLM workload differs from frozen input")
    if workload_fingerprint(workload) != digest:
        raise ValueError(f"{files[0]}: workload fingerprint differs")
    configuration = payload.get("configuration", {})
    expected_config = {"model": model, "dtype": "float16", "block_size": 16,
                       "max_running": case["max_running"],
                       "max_num_batched_tokens": PREFILL_TOKENS_PER_STEP,
                       "num_blocks": blocks, "vllm_kv_cache_mode": "matched",
                       "warmups": warmups, "repetitions": repetitions}
    for field, expected in expected_config.items():
        if configuration.get(field) != expected:
            raise ValueError(f"{files[0]}: {field}={configuration.get(field)!r}, "
                             f"expected {expected!r}")
    result = payload.get("backends", {}).get("vllm")
    if result is None or len(result.get("runs", [])) != repetitions:
        raise ValueError(f"{files[0]}: missing measured vLLM repetitions")
    if payload.get("system", {}).get("packages", {}).get("vllm") != "0.10.2":
        raise ValueError(f"{files[0]}: expected pinned vLLM 0.10.2")
    if (commit is not None and not commit_matches(
            payload.get("system", {}).get("repository", {}).get("commit"),
            commit, resume_commit)):
        raise ValueError(f"{files[0]}: result came from a different source commit")
    expected_ids = {request.request_id for request in workload.requests}
    if result.get("summary", {}).get("output_throughput_tok_s", 0) <= 0:
        raise ValueError(f"{files[0]}: missing positive vLLM throughput")
    for run in result["runs"]:
        metadata = run.get("metadata", {})
        expected_metadata = {"max_num_seqs": case["max_running"],
                             "max_num_batched_tokens": PREFILL_TOKENS_PER_STEP,
                             "max_model_len": max(case["lengths"]) + max(case["outputs"]),
                             "kv_cache_mode": "matched-local-pool",
                             "matched_num_blocks": blocks}
        if any(metadata.get(field) != value for field, value in expected_metadata.items()):
            raise ValueError(f"{files[0]}: vLLM scheduling/KV settings differ")
        outputs = {row["request_id"]: row["output_ids"] for row in run["requests"]}
        if set(outputs) != expected_ids or any(
                len(outputs[request.request_id]) != request.max_tokens
                for request in workload.requests):
            raise ValueError(f"{files[0]}: incomplete vLLM output")
    return payload


def selected_vllm_result(args, shape_id, workload, digest, case, blocks, model):
    own_directory = stage_paths(args, shape_id)[1]
    own = vllm_result(own_directory, workload, digest, case, blocks, model,
                      args.warmups, args.repetitions, repository_commit(),
                      args.resume_commit)
    if own is not None:
        return own, own_directory
    if args.reuse_vllm_from is None:
        return None, own_directory
    source = args.reuse_vllm_from / shape_id / "vllm"
    reused = vllm_result(source, workload, digest, case, blocks, model,
                         args.warmups, args.repetitions)
    if reused is None:
        return None, own_directory
    import torch
    gpu = reused.get("system", {}).get("gpu", {})
    props = torch.cuda.get_device_properties(args.device)
    if (gpu.get("name") != props.name
            or gpu.get("total_memory_bytes") != props.total_memory
            or gpu.get("compute_capability") != f"{props.major}.{props.minor}"):
        raise ValueError(f"{source}: reused vLLM result is from different GPU hardware")
    return reused, source


def run_local(args, shape_id, model_source):
    case, requests, _, digest, blocks = input_contract(args, shape_id)
    dispatch = dispatch_plan(case, requests, args.seed)
    buckets = dispatch["prefill_buckets"]
    local_path, _, _ = stage_paths(args, shape_id)
    if local_is_complete(local_path, digest, model=model_source, blocks=blocks,
                         warmups=args.warmups, repetitions=args.repetitions,
                         commit=repository_commit(), resume_commit=args.resume_commit):
        print(f"{shape_id}: local complete; reusing", flush=True)
        return

    from model_setup import check_startup, load_model_only
    check_startup(args.device)
    import torch
    import inference_engine_cpp as cpp
    from benchmark_scheduler_decode import execute, make_config
    from model_adapter import CppPackedMixedModelAdapter, allocate_pool

    if not hasattr(cpp.SchedulerConfig(), "packed_mixed_step"):
        raise RuntimeError("C++ extension lacks packed_mixed_step; rebuild it")
    engine, _, _ = load_model_only(model_source, args.device, "float16")
    config = make_config(cpp, case)
    config.packed_mixed_step = True
    pool = allocate_pool(engine.cfg, blocks, engine.device)
    adapter = CppPackedMixedModelAdapter(
        engine.model, pool, None, **adapter_options(case, buckets))
    if (adapter.graph_decoder.buckets != [case["max_running"]]
            or not adapter.enable_residual_rmsnorm
            or not adapter.enable_native_decode_qkv_postprocess
            or not adapter.piecewise_prefill.enable_swiglu_fusion):
        raise AssertionError(f"{shape_id}: regime optimization flags did not reach model")
    runs = []
    for index in range(args.warmups + args.repetitions):
        # Excluded from timing; makes uninitialized/stale KV reads obvious.
        for tensor in pool.k_pool + pool.v_pool:
            tensor.fill_(float("nan"))
        result = execute(torch, cpp.IterationLoop(config, torch.device(args.device)),
                         adapter, requests)
        work = verify_fixed_result(result, shape_id)
        if len(result["outputs"]) != len(requests) or any(
                len(result["outputs"][request["id"]]) != request["output"]
                for request in requests):
            raise AssertionError(f"{shape_id}: missing or short local output")
        if adapter.piecewise_prefill.eager_calls:
            raise AssertionError(f"{shape_id}: prefill missed graph bucket")
        mixed_steps = sum(row["kind"] == "mixed" for row in result["steps"])
        if mixed_steps != dispatch["mixed_steps"]:
            raise AssertionError(f"{shape_id}: mixed step count changed from CPU plan")
        if adapter.decisions.get("packed_mixed_cpp_varlen", 0) != mixed_steps:
            raise AssertionError(f"{shape_id}: C++ packed mixed dispatch not used on every mixed step")
        if index >= args.warmups:
            runs.append({"wall_ms": result["wall_ms"],
                         "output_tokens_per_s": result["output_tokens_per_s"],
                         "outputs": result["outputs"], "work": work,
                         "mixed_steps": mixed_steps,
                         "packed_mixed_calls": adapter.decisions.get(
                             "packed_mixed_cpp_varlen", 0)})
        print(f"{shape_id}: local {'warmup' if index < args.warmups else 'run'} "
              f"{index + 1}/{args.warmups + args.repetitions} "
              f"{result['output_tokens_per_s']:.1f} tok/s", flush=True)
    if any(row["outputs"] != runs[0]["outputs"] for row in runs):
        raise AssertionError(f"{shape_id}: local generated tokens differ between repetitions")
    if (not adapter.decisions.get("FA3-auto")
            or not adapter.piecewise_prefill.graph_replays):
        raise AssertionError(f"{shape_id}: requested CUDA graph path did not replay")
    if sorted(adapter.piecewise_prefill.shapes) != buckets:
        raise AssertionError(f"{shape_id}: expected pure/mixed graph buckets not captured")
    atomic_json(local_path, {"status": "complete", "shape_id": shape_id,
                             "model": model_source, "workload_sha256": digest,
                             "repository_commit": repository_commit(),
                             "engine_flags": ENGINE_FLAGS, "warmups": args.warmups,
                             "repetitions": args.repetitions, "num_blocks": blocks,
                             "dispatch_plan": dispatch,
                             "runs": runs,
                             "median_output_tokens_per_s": statistics.median(
                                 row["output_tokens_per_s"] for row in runs),
                             "decode_graph_calls_last_run": adapter.decisions["FA3-auto"],
                             "piecewise_graph_replays": adapter.piecewise_prefill.graph_replays,
                             "piecewise_graph_buckets": sorted(adapter.piecewise_prefill.shapes)})


def run_vllm(args, shape_id, model_source):
    case, _, workload, digest, blocks = input_contract(args, shape_id)
    _, directory, _ = stage_paths(args, shape_id)
    existing, source = selected_vllm_result(
        args, shape_id, workload, digest, case, blocks, model_source)
    if existing is not None:
        print(f"{shape_id}: vLLM complete; reusing {source}", flush=True)
        return
    command = [sys.executable, str(ROOT / "benchmarks/run_benchmarks.py"),
               "--backends", "vllm", "--strict-backends",
               "--model", model_source, "--device", args.device,
               "--dtype", "float16", "--block-size", "16",
               "--max-running", str(case["max_running"]),
               "--max-num-batched-tokens", str(PREFILL_TOKENS_PER_STEP),
               "--num-blocks", str(blocks), "--vllm-kv-cache-mode", "matched",
               "--workload-in", str(args.suite_dir / shape_id / "workload.json"),
               "--warmups", str(args.warmups), "--repetitions", str(args.repetitions),
               "--seed", str(args.seed), "--output-dir", str(directory)]
    subprocess.run(command, cwd=ROOT, check=True)
    if vllm_result(directory, workload, digest, case, blocks,
                   model_source, args.warmups, args.repetitions,
                   repository_commit(), args.resume_commit) is None:
        raise AssertionError(f"{shape_id}: vLLM produced no validated result")


def analyze_cell(args, shape_id, model_source):
    case, requests, workload, digest, blocks = input_contract(args, shape_id)
    dispatch = dispatch_plan(case, requests, args.seed)
    local_path, directory, report_path = stage_paths(args, shape_id)
    if not local_is_complete(local_path, digest, model=model_source,
                             blocks=blocks, warmups=args.warmups,
                             repetitions=args.repetitions,
                             commit=repository_commit(),
                             resume_commit=args.resume_commit):
        raise ValueError(f"missing local result: {local_path}")
    local = json.loads(local_path.read_text())
    observed_dispatch = dict(local.get("dispatch_plan", {}))
    if ("swiglu_fused_buckets" not in observed_dispatch and commit_matches(
            local.get("repository_commit"), repository_commit(), args.resume_commit)):
        from naive_forward import SWIGLU_FUSION_ROW_THRESHOLD
        observed_dispatch["swiglu_fused_buckets"] = [
            bucket for bucket in observed_dispatch.get("prefill_buckets", [])
            if bucket > SWIGLU_FUSION_ROW_THRESHOLD]
    if (local["model"] != model_source or local["num_blocks"] != blocks
            or observed_dispatch != dispatch):
        raise ValueError(f"{shape_id}: local model/KV capacity differs")
    reference, reference_dir = selected_vllm_result(
        args, shape_id, workload, digest, case, blocks, model_source)
    if reference is None:
        raise ValueError(f"missing vLLM result: {directory}")
    vllm = reference["backends"]["vllm"]
    local_output = local["runs"][0]["outputs"]
    vllm_output = {row["request_id"]: row["output_ids"]
                   for row in vllm["runs"][-1]["requests"]}
    exact = sum(local_output[str(key)] == value for key, value in vllm_output.items())
    local_rate = local["median_output_tokens_per_s"]
    vllm_rate = vllm["summary"]["output_throughput_tok_s"]
    report = {"status": "complete", "shape_id": shape_id, "workload_sha256": digest,
              "local_repository_commit": local["repository_commit"],
              "vllm_repository_commit": reference["system"]["repository"]["commit"],
              "local_output_tokens_per_s": local_rate,
              "vllm_output_tokens_per_s": vllm_rate,
              "local_over_vllm": local_rate / vllm_rate,
              "exact_output_requests": exact, "total_requests": len(vllm_output),
              "vllm_result_dir": str(reference_dir),
              "mixed_steps": local["runs"][0]["mixed_steps"],
              "packed_mixed_calls": local["runs"][0]["packed_mixed_calls"],
              "prefill_buckets": dispatch["prefill_buckets"],
              "note": "burst fixed table; packed mixed callback exercised where C++ overlaps decode/prefill"}
    atomic_json(report_path, report)
    print(f"{shape_id}: local {local_rate:.1f}, vLLM {vllm_rate:.1f} tok/s; "
          f"local/vLLM {local_rate / vllm_rate:.3f}x; "
          f"exact outputs {exact}/{len(vllm_output)}", flush=True)
    return report


def forwarded(args, action, shape_id):
    command = [sys.executable, str(Path(__file__)), action,
            "--suite-dir", str(args.suite_dir), "--output-dir", str(args.output_dir),
            "--shape-id", shape_id, "--model", args.model, "--device", args.device,
            "--seed", str(args.seed), "--warmups", str(args.warmups),
            "--repetitions", str(args.repetitions)]
    if args.reuse_vllm_from is not None:
        command += ["--reuse-vllm-from", str(args.reuse_vllm_from)]
    command += resume_options(args.resume_commit)
    return command


def mixed_forwarded(args, shape_id):
    command = [sys.executable, str(HERE / "benchmark_current_mixed_8_vs_vllm.py"),
            "run-cell", "--suite-dir", str(args.suite_dir),
            "--output-dir", str(args.output_dir), "--shape-id", shape_id,
            "--model", args.model, "--device", args.device,
            "--seed", str(args.seed), "--warmups", str(args.warmups),
            "--repetitions", str(args.repetitions)]
    command += resume_options(args.resume_commit)
    return command


def phase_forwarded(args, shape_id):
    return [sys.executable, str(HERE / "benchmark_current_phases_vs_vllm.py"),
            "run-cell", "--suite-dir", str(args.suite_dir),
            "--output-dir", str(args.output_dir), "--shape-id", shape_id,
            "--model", args.model, "--device", args.device,
            "--seed", str(args.seed), "--warmups", str(args.warmups),
            "--repetitions", str(args.repetitions),
            *resume_options(args.resume_commit)]


def main():
    args = parser().parse_args()
    if any(len(prefix) < 7 or any(char not in "0123456789abcdef"
                                   for char in prefix.lower())
           for prefix in (args.resume_commit or [])):
        raise ValueError("--resume-commit must be at least seven hexadecimal characters")
    if args.warmups < 1 or args.repetitions < 1 or args.device != "cuda:0":
        raise ValueError("requires cuda:0, >=1 warmup, and >=1 repetition")
    if args.action in ("run-cell", "run-local", "run-vllm") and args.shape_id is None:
        raise ValueError(f"{args.action} requires --shape-id")
    selected = (args.shape_id,) if args.shape_id else SHAPES
    for shape_id in selected:
        input_contract(args, shape_id)
    from benchmark_current_mixed_8_vs_vllm import (
        mixed_plan, adapter_options as mixed_adapter_options,
        analyze as analyze_mixed)
    from benchmark_current_phases_vs_vllm import analyze as analyze_phases
    if args.action == "plan":
        shapes = {}
        for shape_id in selected:
            case, requests, _, _, _ = input_contract(args, shape_id)
            mixed_case, _, first, arrival, buckets, _, _ = mixed_plan(args, shape_id)
            burst = dispatch_plan(case, requests, args.seed)
            validate_capture_options(adapter_options(case, burst["prefill_buckets"]))
            validate_capture_options(mixed_adapter_options(mixed_case, buckets))
            shapes[shape_id] = {"burst": {**burst, "max_capture_tokens":
                                          adapter_options(case, burst["prefill_buckets"])
                                          ["max_capture_tokens"]},
                                "staggered_mixed": {
                                    "first_wave": first, "arrival_step": arrival,
                                    "prefill_buckets": buckets,
                                    "max_capture_tokens": mixed_adapter_options(
                                        mixed_case, buckets)["max_capture_tokens"],
                                    "batch": mixed_case["max_running"]}}
        print(json.dumps({"shapes": shapes, "engine": ENGINE_FLAGS,
                          "warmups": args.warmups, "repetitions": args.repetitions},
                         indent=2))
        return
    model_source = resolve_model_source(args)
    if args.action == "check":
        for shape_id in selected:
            case, requests, _, _, _ = input_contract(args, shape_id)
            burst = dispatch_plan(case, requests, args.seed)
            validate_capture_options(adapter_options(case, burst["prefill_buckets"]))
            mixed_case, _, _, _, mixed_buckets, _, _ = mixed_plan(args, shape_id)
            validate_capture_options(mixed_adapter_options(mixed_case, mixed_buckets))
            case, _, workload, digest, blocks = input_contract(args, shape_id)
            selected_vllm_result(args, shape_id, workload, digest, case,
                                 blocks, model_source)
        from model_setup import check_startup
        setup = check_startup(args.device)
        import torch
        import inference_engine_cpp as cpp
        from kernel_dispatch import _load
        if not hasattr(cpp.SchedulerConfig(), "packed_mixed_step"):
            raise RuntimeError("C++ extension lacks packed_mixed_step; rebuild it")
        decode_fa3 = _load("paged_decode_fa3")
        decode_fa3._load_fa3()
        _load("paged_varlen_fa3").smoke_varlen_fa3(args.device)
        version = importlib.metadata.version("vllm")
        if version != "0.10.2":
            raise ValueError(f"expected vLLM 0.10.2, got {version}")
        query = torch.zeros((1, 12, 128), device=args.device, dtype=torch.float16)
        kv = torch.zeros((1, 16, 2, 128), device=args.device, dtype=torch.float16)
        table = torch.zeros((1, 1), device=args.device, dtype=torch.int32)
        lengths = torch.ones(1, device=args.device, dtype=torch.int32)
        smoke = decode_fa3.fa3_paged_decode_attention(
            query, kv, kv, table, lengths)
        torch.cuda.synchronize()
        if smoke.shape != query.shape or not torch.isfinite(smoke).all():
            raise AssertionError("FA3 decode smoke did not return finite query-shaped output")
        print(json.dumps({"status": "pass", "model": model_source,
                          "startup": setup, "vllm_version": version,
                          "model_loaded": False}, indent=2))
        return
    if args.action == "run-local":
        run_local(args, args.shape_id, model_source)
    elif args.action == "run-vllm":
        run_vllm(args, args.shape_id, model_source)
    elif args.action in ("run-cell", "run-table"):
        # Cheap dependency/kernel smoke before any model load or long GPU run.
        subprocess.run(forwarded(args, "check", selected[0]), cwd=ROOT, check=True)
        reports = []
        for index, shape_id in enumerate(selected, 1):
            print(f"[{index}/{len(selected)}] {shape_id}", flush=True)
            subprocess.run(forwarded(args, "run-local", shape_id), cwd=ROOT, check=True)
            subprocess.run(forwarded(args, "run-vllm", shape_id), cwd=ROOT, check=True)
            burst = analyze_cell(args, shape_id, model_source)
            subprocess.run(mixed_forwarded(args, shape_id), cwd=ROOT, check=True)
            mixed = analyze_mixed(args, shape_id, model_source)
            subprocess.run(phase_forwarded(args, shape_id), cwd=ROOT, check=True)
            phases = analyze_phases(args, shape_id, model_source)
            reports.append({"shape_id": shape_id, "burst": burst,
                            "mixed": mixed, "phases": phases})
        if args.action == "run-table":
            atomic_json(args.output_dir / "summary.json", {"status": "complete",
                         "rows": reports})
    if args.action == "analyze":
        reports = [{"shape_id": shape_id,
                    "burst": analyze_cell(args, shape_id, model_source),
                    "mixed": analyze_mixed(args, shape_id, model_source),
                    "phases": analyze_phases(args, shape_id, model_source)}
                   for shape_id in selected]
        if args.shape_id is None:
            atomic_json(args.output_dir / "summary.json", {"status": "complete",
                         "rows": reports})


if __name__ == "__main__":
    main()
