#!/usr/bin/env python3
"""Eight frozen workloads: independent C++/tensor-core/graph engine versus current vLLM.

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
for directory in (ROOT, HERE, ROOT / "baseline", ROOT / "benchmarks",
                  ROOT / "engine/model_runner", ROOT / "engine/kvcache",
                  ROOT / "engine/cpp/build"):
    sys.path.insert(0, str(directory))

from benchmark_latest_vs_vllm import load_frozen, resolve_model_source
from reference_version import VLLM_VERSION, require_vllm_version
from fixed_regime import (FACTORIAL_SHAPES, PREFILL_TOKENS_PER_STEP,
                          get_fixed_case, shape_summary, verify_fixed_result)
from engine.accepted_config import retained_fusion_options

SHAPES = tuple(row["id"] for row in FACTORIAL_SHAPES)
# Local-engine attention. "project" runs the independent SM90a kernel.
# "fa3" calls vLLM's FlashAttention-3 for decode, prefill and mixed steps, so
# both engines run the identical attention kernel and the comparison isolates
# everything else; it must run from the vLLM environment, which owns that
# kernel. decode_decision is the action the graph decoder records per mode.
ATTENTION_MODES = {
    "project": {"decode": "flash", "prefill": "flash_varlen", "mixed": "flash_varlen",
                "decode_decision": "flash-local",
                "implementation": "project_owned_sm90a_tma_wgmma_candidate_unqualified"},
    "fa3": {"decode": "fa3", "prefill": "fa3_varlen", "mixed": "fa3_varlen",
            "decode_decision": "FA3-auto",
            "implementation": "external_vllm_flash_attn_fa3"},
}


def attention_options(attention):
    mode = ATTENTION_MODES[attention]
    return dict(decode_attention_policy=mode["decode"],
                prefill_attention_policy=mode["prefill"])


# Where the CUTLASS fused-epilogue GEMMs replace the cuBLAS + RMSNorm/SwiGLU/RoPE chain.
GEMM_EPILOGUE_MODES = ("off", "prefill", "decode", "all")


def greedy_output_head_config(rows):
    """Checked-in graph-warm winners for the fixed B8/B64 regimes."""
    if rows <= 16:
        return {"block_m": 16, "block_n": 256, "block_k": 64,
                "num_warps": 8, "num_stages": 3}
    return {"block_m": 64 if rows > 32 else 32, "block_n": 64, "block_k": 64,
            "num_warps": 4, "num_stages": 3}


def engine_flags(attention, budget=PREFILL_TOKENS_PER_STEP, graph_pool="private",
                 gemm_epilogues="off", boundary_buffers=False,
                 stable_decode_metadata=False, fused_greedy_output=False, gemm_intervention="all"):
    mode = ATTENTION_MODES[attention]
    if gemm_epilogues not in GEMM_EPILOGUE_MODES:
        raise ValueError(f"unknown GEMM epilogue mode {gemm_epilogues!r}")
    # Keys appear only off the frozen defaults, so results recorded before
    # budget, graph-pool and fusion variants keep matching their flags.
    extra = {} if budget == PREFILL_TOKENS_PER_STEP else {"prefill_token_budget": budget}
    if graph_pool != "private":
        extra["prefill_graph_pool"] = graph_pool
    if gemm_epilogues != "off":
        extra["gemm_epilogues"] = gemm_epilogues
        if gemm_intervention != "all":
            extra["gemm_intervention"] = gemm_intervention
    if boundary_buffers:
        extra["prefill_boundary_buffer_reuse"] = True
    if stable_decode_metadata:
        extra["stable_decode_metadata"] = True
    if fused_greedy_output:
        extra["fused_greedy_output"] = True
    return {**extra,
    "cpp_scheduler": True,
    "packed_mixed_step": True,
    "decode_attention_policy": mode["decode"],
    "mixed_attention_policy": mode["mixed"],
    "prefill_attention_policy": mode["prefill"],
    "attention_implementation": mode["implementation"],
    "decode_graph_bucket": "exact_batch",
    "prefill_graph_buckets": "CPU-scheduled pure/mixed exact upper bounds",
    "residual_rmsnorm": True,
    "native_decode_qkv_postprocess": True,
    "prefill_swiglu_fusion": True,
    "prefill_packed_qkv_rope_cache": False,
}


ENGINE_FLAGS = engine_flags("project")  # the default mode, for existing callers


def variant(args):
    """The local engine's configuration beyond attention, read off parsed CLI arguments."""
    return dict(budget=getattr(args, "prefill_budget", PREFILL_TOKENS_PER_STEP),
                graph_pool=getattr(args, "prefill_graph_pool", "private"),
                gemm_epilogues=getattr(args, "gemm_epilogues", "off"),
                boundary_buffers=bool(getattr(args, "boundary_buffers", False)),
                stable_decode_metadata=bool(getattr(args, "stable_decode_metadata", False)),
                fused_greedy_output=bool(getattr(args, "fused_greedy_output", False)),
                gemm_intervention=getattr(args, "gemm_intervention", "all"))


def variant_flags(args):
    return engine_flags(args.attention, **variant(args))


def variant_cli(args):
    """Forward the variant to a child harness process."""
    value = variant(args)
    return ["--prefill-budget", str(value["budget"]),
            "--prefill-graph-pool", value["graph_pool"],
            "--gemm-epilogues", value["gemm_epilogues"],
            *(["--gemm-intervention", value["gemm_intervention"]]
              if value["gemm_intervention"] != "all" else []),
            *(["--boundary-buffers"] if value["boundary_buffers"] else []),
            *(["--stable-decode-metadata"] if value["stable_decode_metadata"] else []),
            *(["--fused-greedy-output"] if value["fused_greedy_output"] else [])]


def add_variant_arguments(parser):
    parser.add_argument("--gemm-intervention", choices=("all", "residual-o", "gate-up", "residual-down", "qkv", "combined"),
                        default="all", help="select which fused GEMM operations execute; only all folds RMSNorm")
    parser.add_argument("--prefill-budget", type=int, default=PREFILL_TOKENS_PER_STEP,
                        help="packed prefill tokens per step for the local engine. The staggered "
                             "workload's arrival step stays derived from the frozen "
                             f"{PREFILL_TOKENS_PER_STEP}, so every budget runs the same workload")
    parser.add_argument("--prefill-graph-pool", choices=("private", "shared"), default="private",
                        help="shared captures every prefill segment into one CUDA graph memory "
                             "pool, which budgets above 8192 need")
    parser.add_argument("--gemm-epilogues", choices=GEMM_EPILOGUE_MODES, default="off",
                        help="CUTLASS GEMMs with fused residual/RMSNorm/SwiGLU/RoPE/cache "
                             "epilogues in prefill graphs, the decode graph, or both "
                             "(custom_kernels/gemm_epilogue; build it in this interpreter)")
    parser.add_argument("--boundary-buffers", action="store_true",
                        help="bind each prefill segment's residual input to its producer's "
                             "output and write FA3 attention in place (no boundary copies)")
    parser.add_argument("--stable-decode-metadata", action="store_true",
                        help="advance stable pure-decode IDs/positions/lengths/slots on GPU and "
                             "reuse its immutable page table")
    parser.add_argument("--fused-greedy-output", action="store_true",
                        help="return exact greedy token IDs directly from one fused final-norm/"
                             "LM-head/argmax path in decode, prefill and mixed callbacks")
    return parser


def variant_adapter_options(args):
    """Adapter keyword arguments for the variant, on top of the accepted configuration."""
    value = variant(args)
    extra = ({"gemm_epilogue_intervention": value["gemm_intervention"]}
             if value["gemm_intervention"] != "all" else {})
    return dict(**extra, enable_prefill_shared_graph_pool=value["graph_pool"] == "shared",
                enable_prefill_boundary_buffer_reuse=value["boundary_buffers"],
                enable_stable_decode_table_cache=value["stable_decode_metadata"],
                enable_prefill_fused_gemm_epilogues=value["gemm_epilogues"] in ("prefill", "all"),
                enable_decode_fused_gemm_epilogues=value["gemm_epilogues"] in ("decode", "all"),
                output_head_policy=("fused_argmax" if value["fused_greedy_output"]
                                    else "logits"))


def check_variant_reached(adapter, args, label):
    """Every flag of the variant must be live in the constructed adapter."""
    options = variant_adapter_options(args)
    prefill = adapter.piecewise_prefill
    decoder_heads = {
        decoder.output_head_policy
        for decoder in adapter.graph_decoder.decoders.values()
    }
    output_head = (prefill.output_head_policy
                   if decoder_heads == {prefill.output_head_policy}
                   else "inconsistent")
    actual = dict(enable_prefill_shared_graph_pool=prefill.graph_pool is not None,
                  enable_prefill_boundary_buffer_reuse=prefill.enable_boundary_buffer_reuse,
                  enable_stable_decode_table_cache=all(
                      decoder.enable_stable_decode_table_cache
                      for decoder in adapter.graph_decoder.decoders.values()),
                  enable_prefill_fused_gemm_epilogues=prefill.enable_fused_gemm_epilogues,
                  enable_decode_fused_gemm_epilogues=all(
                      decoder.enable_fused_gemm_epilogues
                      for decoder in adapter.graph_decoder.decoders.values()),
                  output_head_policy=output_head)
    if "gemm_epilogue_intervention" in options:
        selections = {prefill.gemm_epilogue_intervention, *(
            decoder.gemm_epilogue_intervention for decoder in adapter.graph_decoder.decoders.values())}
        actual["gemm_epilogue_intervention"] = (selections.pop() if len(selections) == 1 else "inconsistent")
    if actual != options:
        raise AssertionError(f"{label}: engine variant did not reach the model: "
                             f"requested {options}, constructed {actual}")


def apply_variant_config(config, args):
    """Apply scheduler-side variant flags paired with the adapter-side flags."""
    config.reuse_stable_decode_metadata = bool(
        getattr(args, "stable_decode_metadata", False))
    config.forward_returns_token_ids = bool(
        getattr(args, "fused_greedy_output", False))
    return config

# vLLM's scheduling budget. "matched" pins it to the frozen 2048 the local engine
# uses (the accepted baseline); "default" leaves max_num_batched_tokens unset so
# vLLM runs as shipped (16384 for the offline LLM class on H100 in 0.30.0). Every
# vLLM result records the budget it actually used.
VLLM_BUDGETS = ("matched", "default")


def vllm_budget_kwargs(mode):
    """max_num_batched_tokens for a directly constructed vllm.LLM."""
    if mode not in VLLM_BUDGETS:
        raise ValueError(f"unknown vLLM budget mode {mode!r}")
    return {} if mode == "default" else {"max_num_batched_tokens": PREFILL_TOKENS_PER_STEP}



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
    result.add_argument("--vllm-python", help="interpreter in the separate current-vLLM environment")
    add_variant_arguments(result)
    result.add_argument("--vllm-budget", choices=VLLM_BUDGETS, default="matched",
                        help="matched pins vLLM to the frozen 2048 budget; default runs vLLM as "
                             "shipped. Results of one mode are never reused as the other")
    result.add_argument("--attention", choices=tuple(ATTENTION_MODES), default="project",
                        help="local-engine attention: the independent kernel, or vLLM's FA3 "
                             "(identical to the reference; run from the vLLM environment)")
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


def repository_commit(root=ROOT):
    """Last commit that changed anything outside experiments/results.

    Saved cells are reusable only under the code that produced them, but a
    commit that only adds results must not make them look stale: pushing each
    cell as it completes is how a run survives losing the pod.
    """
    return subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", ".", ":(exclude)experiments/results"],
        cwd=root, text=True).strip()


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

    case = {**get_fixed_case(shape_id),
            "prefill_budget": getattr(args, "prefill_budget", PREFILL_TOKENS_PER_STEP)}
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


def dispatch_plan(case, requests, seed, *, with_schedule=False):
    """Resolve graph buckets and expected mixed dispatch before loading weights.

    with_schedule adds the CPU dry schedule's per-step (kind, calls) signature,
    which verify_planned_work compares against the GPU run off the frozen budget.
    """
    import torch
    import inference_engine_cpp as cpp
    from benchmark_integrated_graph import dry_schedule
    from naive_forward import SWIGLU_FUSION_ROW_THRESHOLD

    schedule = dry_schedule(torch, cpp, case, seed, requests)
    packed = [sum(call[1] for call in step["calls"])
              for step in schedule["steps"] if step["kind"] == "mixed"]
    prefill = [call[1] for step in schedule["steps"]
               for call in step["calls"] if not call[0]]
    # A full step packs the budget, or the whole cohort when it is smaller.
    expected = min(case["prefill_budget"], sum(case["lengths"]))
    if not prefill or max(prefill) != expected:
        raise AssertionError(f"{case['id']}: CPU prefill work differs from the {case['prefill_budget']}-token budget")
    buckets = sorted({max(prefill), max(packed, default=0)})
    buckets = [bucket for bucket in buckets if bucket > 0]
    plan = {"prefill_buckets": buckets, "mixed_steps": len(packed),
            "max_packed_mixed_tokens": max(packed, default=0),
            "swiglu_fused_buckets": [bucket for bucket in buckets
                                     if bucket > SWIGLU_FUSION_ROW_THRESHOLD]}
    if with_schedule:
        plan["schedule"] = [(step["kind"], [tuple(call) for call in step["calls"]])
                            for step in schedule["steps"]]
    return plan


def verify_planned_work(result, dispatch, case):
    """Off the frozen budget, the GPU run must reproduce the CPU dry schedule exactly."""
    actual = [(step["kind"], [tuple(call) for call in step["calls"]]) for step in result["steps"]]
    if actual != dispatch["schedule"]:
        first = next((index for index, (a, b) in enumerate(zip(actual, dispatch["schedule"])) if a != b),
                     min(len(actual), len(dispatch["schedule"])))
        raise AssertionError(f"{case['id']}: GPU schedule departs from the CPU dry schedule at step {first}")
    batches = [call[1] for step in result["steps"] for call in step["calls"] if call[0]]
    prefill = [call[1] for step in result["steps"] for call in step["calls"] if not call[0]]
    return {"max_actual_decode_batch": max(batches),
            "max_packed_prefill_tokens": max(prefill),
            "pure_full_decode_steps": sum(step["kind"] == "decode" and any(
                call[0] and call[1] == case["max_running"] for call in step["calls"])
                for step in result["steps"])}


def adapter_options(case, buckets, attention="project", variant_options=None):
    if not buckets or min(buckets) < 1:
        raise ValueError("burst graph buckets must be positive")
    variant_options = dict(variant_options or {})
    if variant_options.get("output_head_policy") == "fused_argmax":
        variant_options["output_head_config"] = greedy_output_head_config
    return dict(max_running=case["max_running"],
                **variant_options,
                max_context_length=max(length + output for length, output in
                                       zip(case["lengths"], case["outputs"])),
                **attention_options(attention),
                decode_buckets=[case["max_running"]],
                max_capture_tokens=max(buckets),
                max_prefill_shapes=len(buckets), prefill_buckets=buckets,
                **retained_fusion_options())


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


def local_is_complete(path, digest, *, flags=None, model=None, blocks=None,
                      warmups=None, repetitions=None, commit=None,
                      resume_commit=None):
    if not path.is_file():
        return False
    flags = ENGINE_FLAGS if flags is None else flags
    row = json.loads(path.read_text())
    if (row.get("status") != "complete" or row.get("workload_sha256") != digest
            or row.get("engine_flags") != flags
            or (model is not None and row.get("model") != model)
            or (blocks is not None and row.get("num_blocks") != blocks)
            or (warmups is not None and row.get("warmups") != warmups)
            or (repetitions is not None and row.get("repetitions") != repetitions)
            or (commit is not None and not commit_matches(
                row.get("repository_commit"), commit, resume_commit))):
        raise ValueError(f"stale or mismatched local result: {path}")
    return True


def vllm_result(directory, workload, digest, case, blocks, model,
                warmups, repetitions, commit=None, resume_commit=None, vllm_budget="matched"):
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
    # Results predating budget modes were all pinned.
    if configuration.get("vllm_budget", "matched") != vllm_budget:
        raise ValueError(f"{files[0]}: vLLM budget mode {configuration.get('vllm_budget', 'matched')!r}, "
                         f"expected {vllm_budget!r}; use a separate output directory per mode")
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
    require_vllm_version(payload.get("system", {}).get("packages", {}).get("vllm"))
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
                             "max_model_len": max(case["lengths"]) + max(case["outputs"]),
                             "kv_cache_mode": "matched-local-pool",
                             "matched_num_blocks": blocks}
        if vllm_budget == "matched":
            expected_metadata["max_num_batched_tokens"] = PREFILL_TOKENS_PER_STEP
        elif (metadata.get("max_num_batched_tokens_requested") is not None
              or not isinstance(metadata.get("max_num_batched_tokens"), int)):
            raise ValueError(f"{files[0]}: default-budget run did not record vLLM's own budget")
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
    mode = getattr(args, "vllm_budget", "matched")
    own = vllm_result(own_directory, workload, digest, case, blocks, model,
                      args.warmups, args.repetitions, repository_commit(),
                      args.resume_commit, vllm_budget=mode)
    if own is not None:
        return own, own_directory
    if args.reuse_vllm_from is None:
        return None, own_directory
    source = args.reuse_vllm_from / shape_id / "vllm"
    reused = vllm_result(source, workload, digest, case, blocks, model,
                         args.warmups, args.repetitions, vllm_budget=mode)
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
    dispatch = dispatch_plan(case, requests, args.seed, with_schedule=True)
    buckets = dispatch["prefill_buckets"]
    local_path, _, _ = stage_paths(args, shape_id)
    flags = variant_flags(args)
    if local_is_complete(local_path, digest, flags=flags,
                         model=model_source, blocks=blocks,
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
    apply_variant_config(config, args)
    pool = allocate_pool(engine.cfg, blocks, engine.device)
    adapter = CppPackedMixedModelAdapter(
        engine.model, pool, None,
        **adapter_options(case, buckets, args.attention, variant_adapter_options(args)))
    check_variant_reached(adapter, args, shape_id)
    if (adapter.graph_decoder.buckets != [case["max_running"]]
            or adapter.prefill_attention_policy != ATTENTION_MODES[args.attention]["prefill"]
            or not adapter.enable_residual_rmsnorm
            or not adapter.enable_native_decode_qkv_postprocess
            or not adapter.piecewise_prefill.enable_swiglu_fusion):
        raise AssertionError(f"{shape_id}: regime optimization flags did not reach model")
    runs = []
    for index in range(args.warmups + args.repetitions):
        # Excluded from timing; makes uninitialized/stale KV reads obvious.
        for tensor in pool.k_pool + pool.v_pool:
            tensor.fill_(float("nan"))
        loop = cpp.IterationLoop(config, torch.device(args.device))
        result = execute(torch, loop, adapter, requests)
        if args.stable_decode_metadata and loop.num_device_decode_state_replays() == 0:
            raise AssertionError(f"{shape_id}: stable decode metadata never replayed")
        work = (verify_fixed_result(result, shape_id) if args.prefill_budget == PREFILL_TOKENS_PER_STEP
                else verify_planned_work(result, dispatch, case))
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
    if (not adapter.decisions.get(ATTENTION_MODES[args.attention]["decode_decision"])
            or not adapter.piecewise_prefill.graph_replays):
        raise AssertionError(f"{shape_id}: requested CUDA graph path did not replay")
    if sorted(adapter.piecewise_prefill.shapes) != buckets:
        raise AssertionError(f"{shape_id}: expected pure/mixed graph buckets not captured")
    atomic_json(local_path, {"status": "complete", "shape_id": shape_id,
                             "model": model_source, "workload_sha256": digest,
                             "repository_commit": repository_commit(),
                             "engine_flags": flags, "warmups": args.warmups,
                             "repetitions": args.repetitions, "num_blocks": blocks,
                             "dispatch_plan": {key: value for key, value in dispatch.items()
                                               if key != "schedule"},
                             "prefill_graph_pool": args.prefill_graph_pool,
                             "prefill_graph_reserved_bytes": {
                                 str(bucket): value for bucket, value in
                                 adapter.piecewise_prefill.capture_memory.items()},
                             "peak_allocated_bytes": torch.cuda.max_memory_allocated(args.device),
                             "peak_reserved_bytes": torch.cuda.max_memory_reserved(args.device),
                             "runs": runs,
                             "median_output_tokens_per_s": statistics.median(
                                 row["output_tokens_per_s"] for row in runs),
                             "decode_graph_calls_last_run": adapter.decisions[
                                 ATTENTION_MODES[args.attention]["decode_decision"]],
                             "piecewise_graph_replays": adapter.piecewise_prefill.graph_replays,
                             "piecewise_graph_buckets": sorted(adapter.piecewise_prefill.shapes)})


def run_vllm(args, shape_id, model_source):
    require_vllm_version(importlib.metadata.version("vllm"))
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
    if args.vllm_budget == "default":
        command.append("--vllm-default-budget")
    subprocess.run(command, cwd=ROOT, check=True)
    if vllm_result(directory, workload, digest, case, blocks,
                   model_source, args.warmups, args.repetitions,
                   repository_commit(), args.resume_commit, vllm_budget=args.vllm_budget) is None:
        raise AssertionError(f"{shape_id}: vLLM produced no validated result")


def analyze_cell(args, shape_id, model_source):
    case, requests, workload, digest, blocks = input_contract(args, shape_id)
    dispatch = dispatch_plan(case, requests, args.seed)
    local_path, directory, report_path = stage_paths(args, shape_id)
    if not local_is_complete(local_path, digest, flags=variant_flags(args),
                             model=model_source,
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
              "vllm_budget": getattr(args, "vllm_budget", "matched"),
              "vllm_max_num_batched_tokens": vllm["runs"][0].get("metadata", {}).get("max_num_batched_tokens"),
              "local_prefill_token_budget": variant(args)["budget"],
              "local_engine_flags": local["engine_flags"],
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
    interpreter = (getattr(args, "vllm_python", None) or sys.executable) if action == "run-vllm" else sys.executable
    command = [interpreter, str(Path(__file__)), action,
            "--suite-dir", str(args.suite_dir), "--output-dir", str(args.output_dir),
            "--shape-id", shape_id, "--model", args.model, "--device", args.device,
            "--seed", str(args.seed), "--warmups", str(args.warmups),
            "--repetitions", str(args.repetitions), "--attention", args.attention,
            "--vllm-budget", getattr(args, "vllm_budget", "matched"), *variant_cli(args)]
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
            "--repetitions", str(args.repetitions), "--attention", args.attention,
            "--vllm-budget", getattr(args, "vllm_budget", "matched"), *variant_cli(args)]
    command += resume_options(args.resume_commit)
    if getattr(args, "vllm_python", None):
        command += ["--vllm-python", args.vllm_python]
    return command


def phase_forwarded(args, shape_id):
    command = [sys.executable, str(HERE / "benchmark_current_phases_vs_vllm.py"),
            "run-cell", "--suite-dir", str(args.suite_dir),
            "--output-dir", str(args.output_dir), "--shape-id", shape_id,
            "--model", args.model, "--device", args.device,
            "--seed", str(args.seed), "--warmups", str(args.warmups),
            "--repetitions", str(args.repetitions), "--attention", args.attention,
            "--vllm-budget", getattr(args, "vllm_budget", "matched"), *variant_cli(args),
            *resume_options(args.resume_commit)]
    if getattr(args, "vllm_python", None):
        command += ["--vllm-python", args.vllm_python]
    return command


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
    if args.prefill_budget < 1:
        raise ValueError("--prefill-budget must be positive")
    if args.prefill_budget > 8192 and args.prefill_graph_pool != "shared":
        raise ValueError("budgets above 8192 need --prefill-graph-pool shared")
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
            burst = dispatch_plan(case, requests, args.seed)
            validate_capture_options(adapter_options(case, burst["prefill_buckets"], args.attention))
            burst = {**burst, "prefill_budget": args.prefill_budget,
                     "max_capture_tokens": adapter_options(
                         case, burst["prefill_buckets"], args.attention)["max_capture_tokens"]}
            try:
                mixed_case, _, first, arrival, buckets, _, _ = mixed_plan(args, shape_id)
            except AssertionError as error:
                # A budget far below the frozen one can leave the first wave still
                # prefilling at the fixed arrival step; the mixed stage cannot run there.
                shapes[shape_id] = {"burst": burst, "staggered_mixed": {"unsupported": str(error)}}
                continue
            validate_capture_options(mixed_adapter_options(mixed_case, buckets, args.attention))
            shapes[shape_id] = {"burst": burst,
                                "staggered_mixed": {
                                    "first_wave": first, "arrival_step": arrival,
                                    "prefill_buckets": buckets,
                                    "max_capture_tokens": mixed_adapter_options(
                                        mixed_case, buckets, args.attention)["max_capture_tokens"],
                                    "batch": mixed_case["max_running"]}}
        print(json.dumps({"shapes": shapes, "engine": variant_flags(args),
                          "warmups": args.warmups, "repetitions": args.repetitions},
                         indent=2))
        return
    model_source = resolve_model_source(args)
    if args.action == "check":
        for shape_id in selected:
            case, requests, _, _, _ = input_contract(args, shape_id)
            burst = dispatch_plan(case, requests, args.seed)
            validate_capture_options(adapter_options(case, burst["prefill_buckets"], args.attention))
            mixed_case, _, _, _, mixed_buckets, _, _ = mixed_plan(args, shape_id)
            validate_capture_options(mixed_adapter_options(mixed_case, mixed_buckets,
                                                           args.attention))
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
        if (args.stable_decode_metadata
                and not hasattr(cpp.SchedulerConfig(), "reuse_stable_decode_metadata")):
            raise RuntimeError(
                "C++ extension lacks reuse_stable_decode_metadata; rebuild it")
        if (args.fused_greedy_output
                and not hasattr(cpp.SchedulerConfig(), "forward_returns_token_ids")):
            raise RuntimeError(
                "C++ extension lacks forward_returns_token_ids; rebuild it")
        query = torch.zeros((1, 12, 128), device=args.device, dtype=torch.float16)
        kv = torch.zeros((1, 16, 2, 128), device=args.device, dtype=torch.float16)
        table = torch.zeros((1, 1), device=args.device, dtype=torch.int32)
        lengths = torch.ones(1, device=args.device, dtype=torch.int32)
        # Smoke the attention this run will actually use, in this interpreter.
        if args.attention == "fa3":
            smoke = _load("paged_decode_fa3").fa3_paged_decode_attention(
                query, kv, kv, table, lengths, scale=128 ** -.5, num_splits=0)
        else:
            smoke = _load("paged_flash_decode").flash_decode(
                query, kv, kv, table, lengths)
        torch.cuda.synchronize()
        if smoke.shape != query.shape or not torch.isfinite(smoke).all():
            raise AssertionError("local attention did not return finite query-shaped output")
        if args.boundary_buffers and args.attention == "fa3":
            _load("paged_varlen_fa3").smoke_varlen_fa3(args.device, boundary_buffers=True)
        if args.gemm_epilogues != "off":
            # Loads the CUTLASS extension in this interpreter (ABI, source hash and
            # fragment proof), then checks one fused GEMM against torch.
            fused = _load("fused_gemm")
            x = torch.randn((64, 1536), device=args.device, dtype=torch.float16)
            weight = torch.randn((1536, 1536), device=args.device, dtype=torch.float16) * .02
            residual = torch.randn_like(x)
            out, _ = fused.residual_gemm(x, weight, residual)
            expected = (residual.float() + x.float() @ weight.float().T)
            if not torch.allclose(out.float(), expected, atol=.05, rtol=.01):
                raise AssertionError("fused GEMM epilogue extension disagrees with torch")
        if args.fused_greedy_output:
            hidden = torch.randn((8, 128), device=args.device, dtype=torch.float16)
            weight = torch.randn((256, 128), device=args.device, dtype=torch.float16)
            tokens = _load("fused_lm_head").fused_lm_head_argmax(hidden, weight)
            if tokens.shape != (8,) or tokens.dtype != torch.int64:
                raise AssertionError("fused greedy output head returned an invalid token tensor")
        print(json.dumps({"status": "pass", "model": model_source, "attention": args.attention,
                          "engine": variant_flags(args),
                          "startup": setup, "reference_version": VLLM_VERSION,
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
