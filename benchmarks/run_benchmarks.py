#!/usr/bin/env python3
"""Run identical synthetic inference workloads across engine backends."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

from benchmark_backends import BackendUnavailable, create_backend
from benchmark_core import (
    SCHEMA_VERSION,
    BackendRun,
    Workload,
    make_synthetic_workload,
    median_aggregate,
    parse_int_list,
)


BACKENDS = (
    "pytorch-baseline",
    "paged-kv",
    "continuous-batching",
    "bucketed-cuda-graphs",
    "custom-kernels",
    "regime-dispatched",
    "vllm",
)
DEFAULT_BACKENDS = BACKENDS[:4]
MODEL_ID = "Qwen/Qwen2.5-1.5B"
MATCHED_CONFIGURATION_FIELDS = (
    "model",
    "dtype",
    "block_size",
    "max_running",
    "max_num_batched_tokens",
    "max_prefill_chunk_size",
    "max_prefill_attention_pairs",
    "prefill_tile_policy",
)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def repository_metadata() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return {"commit": commit, "dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def system_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "repository": repository_metadata(),
        "packages": {
            name: package_version(name)
            for name in ("torch", "transformers", "triton", "vllm")
        },
    }
    try:
        import torch

        metadata["cuda_available"] = torch.cuda.is_available()
        metadata["torch_cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            index = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(index)
            metadata["gpu"] = {
                "name": props.name,
                "total_memory_bytes": props.total_memory,
                "compute_capability": f"{props.major}.{props.minor}",
            }
    except ImportError:
        metadata["cuda_available"] = False
    return metadata


def parse_backends(value: str) -> list[str]:
    if value.strip() == "all":
        return list(BACKENDS)
    names = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [name for name in names if name not in BACKENDS]
    if not names or unknown:
        raise argparse.ArgumentTypeError(
            f"unknown backend(s) {unknown}; choose from {', '.join(BACKENDS)}"
        )
    return names


def load_or_create_workload(args) -> Workload:
    if args.workload_in:
        workload = Workload.from_dict(json.loads(args.workload_in.read_text()))
    elif args.compare_with:
        source = json.loads(args.compare_with[0].read_text())
        workload = Workload.from_dict(source["workload"])
    else:
        workload = make_synthetic_workload(
            name=args.workload_name,
            num_requests=args.num_requests,
            prompt_lengths=parse_int_list(args.prompt_lengths),
            output_lengths=parse_int_list(args.output_lengths),
            vocab_size=args.vocab_size,
            seed=args.seed,
            request_rate=args.request_rate,
        )
    if args.workload_out:
        args.workload_out.parent.mkdir(parents=True, exist_ok=True)
        args.workload_out.write_text(json.dumps(workload.to_dict(), indent=2) + "\n")
    return workload


def workload_fingerprint(workload: Workload | dict[str, Any]) -> str:
    value = workload.to_dict() if isinstance(workload, Workload) else workload
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def load_comparison_results(
    paths: list[Path], workload: Workload, configuration: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Load compatible backends from earlier invocations.

    vLLM often lives in a separate environment, so matched results must be mergeable
    without rerunning the local engine. Exact workload equality is mandatory; the
    model, dtype, paging block size, and concurrency limit must also match.
    """
    combined: dict[str, dict[str, Any]] = {}
    sources: list[dict[str, Any]] = []
    expected_workload = workload.to_dict()
    for path in paths:
        payload = json.loads(path.read_text())
        if payload.get("schema_version") not in (1, SCHEMA_VERSION):
            raise ValueError(f"{path}: unsupported benchmark schema")
        if payload.get("workload") != expected_workload:
            raise ValueError(
                f"{path}: workload does not exactly match "
                f"{workload_fingerprint(expected_workload)}"
            )
        saved_configuration = payload.get("configuration", {})
        for field in MATCHED_CONFIGURATION_FIELDS:
            if saved_configuration.get(field) != configuration.get(field):
                raise ValueError(
                    f"{path}: {field} mismatch: saved={saved_configuration.get(field)!r}, "
                    f"current={configuration.get(field)!r}"
                )
        for backend, result in payload.get("backends", {}).items():
            if backend in combined:
                raise ValueError(f"{path}: duplicate imported backend {backend!r}")
            imported = dict(result)
            imported["provenance"] = {
                "result_file": str(path),
                "created_at": payload.get("created_at"),
                "system": payload.get("system"),
            }
            combined[backend] = imported
        sources.append({
            "path": str(path),
            "created_at": payload.get("created_at"),
            "workload_sha256": workload_fingerprint(payload["workload"]),
        })
    return combined, sources


def validate_run(run: BackendRun, workload: Workload) -> None:
    metrics = run.aggregate()
    expected_output = sum(request.max_tokens for request in workload.requests)
    if metrics["completed_requests"] != len(workload.requests):
        raise AssertionError(
            f"{run.backend}: completed {metrics['completed_requests']}/"
            f"{len(workload.requests)} requests"
        )
    if metrics["output_tokens"] != expected_output:
        raise AssertionError(
            f"{run.backend}: produced {metrics['output_tokens']} output tokens, "
            f"expected {expected_output}"
        )
    for trace in run.traces:
        if trace.output_tokens != len(trace.output_ids):
            raise AssertionError(
                f"{run.backend}/{trace.request_id}: output count does not match saved token IDs"
            )


def clear_accelerator_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except ImportError:
        pass


def run_backend(name: str, workload: Workload, args) -> tuple[list[BackendRun], dict[str, Any]]:
    backend = None
    try:
        backend = create_backend(
            name,
            workload=workload,
            model_id=args.model,
            device=args.device,
            dtype=args.dtype,
            block_size=args.block_size,
            max_running=args.max_running,
            num_blocks=args.num_blocks,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_prefill_chunk_size=args.max_prefill_chunk_size,
            max_prefill_attention_pairs=args.max_prefill_attention_pairs,
            prefill_tile_policy=args.prefill_tile_policy,
            decode_attention_policy=args.decode_attention_policy,
            seed=args.seed,
            vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            vllm_kv_cache_mode=args.vllm_kv_cache_mode,
        )
        for warmup in range(args.warmups):
            print(f"  warmup {warmup + 1}/{args.warmups}")
            warmup_run = backend.run(workload)
            validate_run(warmup_run, workload)

        runs = []
        for repetition in range(args.repetitions):
            print(f"  measured run {repetition + 1}/{args.repetitions}")
            run = backend.run(workload)
            validate_run(run, workload)
            runs.append(run)
        return runs, median_aggregate(runs)
    finally:
        if backend is not None:
            backend.close()
        # Drop the owning reference before empty_cache(); otherwise the model tensors
        # are still live and the next backend can inherit an artificially full device.
        backend = None
        clear_accelerator_cache()


def metric_value(summary: dict[str, Any], metric: str, statistic: str = "median"):
    value = summary.get(metric)
    if isinstance(value, dict):
        return value.get(statistic)
    return value


def matched_prefix(left: list[int], right: list[int]) -> int:
    for index, (left_token, right_token) in enumerate(zip(left, right)):
        if left_token != right_token:
            return index
    return min(len(left), len(right))


def attach_baseline_diagnostics(results: dict[str, dict[str, Any]]) -> None:
    baseline = results.get("pytorch-baseline")
    if baseline is None:
        return
    reference = {
        request["request_id"]: request["output_ids"]
        for request in baseline["runs"][-1]["requests"]
    }
    for backend, result in results.items():
        requests = result["runs"][-1]["requests"]
        prefixes = []
        exact = 0
        for request in requests:
            expected = reference[request["request_id"]]
            actual = request["output_ids"]
            prefixes.append(matched_prefix(actual, expected))
            exact += int(actual == expected)
        result["correctness_vs_pytorch_baseline"] = {
            "exact_requests": exact,
            "total_requests": len(requests),
            "exact_request_fraction": exact / len(requests) if requests else None,
            "min_matched_prefix_tokens": min(prefixes, default=None),
            "mean_matched_prefix_tokens": (
                sum(prefixes) / len(prefixes) if prefixes else None
            ),
        }


def attach_performance_comparisons(results: dict[str, dict[str, Any]]) -> None:
    references = {
        name: result["summary"].get("output_throughput_tok_s")
        for name, result in results.items()
        if name in ("pytorch-baseline", "custom-kernels")
    }
    for result in results.values():
        throughput = result["summary"].get("output_throughput_tok_s")
        result["relative_output_throughput"] = {
            f"vs_{name.replace('-', '_')}": (
                throughput / reference
                if throughput is not None and reference not in (None, 0) else None
            )
            for name, reference in references.items()
        }


def validate_comparison_contract(
    results: dict[str, dict[str, Any]],
    *,
    max_running: int,
    max_model_len: int,
    vllm_kv_cache_mode: str,
    max_num_batched_tokens: int | None = None,
) -> None:
    """Refuse to label an uncapped vLLM result as a matched comparison."""
    result = results.get("vllm")
    if result is None:
        return
    expected_kv_mode = (
        "matched-local-pool" if vllm_kv_cache_mode == "matched" else "vllm-native"
    )
    for index, run in enumerate(result.get("runs", [])):
        metadata = run.get("metadata", {})
        expected = {
            "max_num_seqs": max_running,
            "max_model_len": max_model_len,
            "kv_cache_mode": expected_kv_mode,
        }
        if max_num_batched_tokens is not None:
            expected["max_num_batched_tokens"] = max_num_batched_tokens
        for field, value in expected.items():
            if metadata.get(field) != value:
                raise ValueError(
                    f"vllm run {index} is not matched: {field}="
                    f"{metadata.get(field)!r}, expected {value!r}"
                )


def fmt(value, digits=2):
    return "-" if value is None else f"{value:.{digits}f}"


def print_summary(results: dict[str, dict[str, Any]]) -> None:
    print("\n" + "=" * 105)
    print(
        f"{'backend':>23} | {'req/s':>9} | {'out tok/s':>10} | {'TTFT p50':>10} | "
        f"{'TPOT p50':>10} | {'E2E p95':>10} | {'GPU GiB':>8}"
    )
    print("-" * 105)
    for backend, result in results.items():
        summary = result["summary"]
        peak = summary.get("peak_gpu_memory_bytes")
        peak_gib = peak / 1024**3 if peak is not None else None
        print(
            f"{backend:>23} | "
            f"{fmt(summary.get('request_throughput_rps')):>9} | "
            f"{fmt(summary.get('output_throughput_tok_s')):>10} | "
            f"{fmt(metric_value(summary, 'ttft_ms')):>10} | "
            f"{fmt(metric_value(summary, 'tpot_ms')):>10} | "
            f"{fmt(metric_value(summary, 'e2e_ms', 'p95')):>10} | "
            f"{fmt(peak_gib):>8}"
        )
    print("=" * 105)
    if "custom-kernels" in results:
        print("\noutput throughput relative to custom-kernels:")
        for backend, result in results.items():
            ratio = result.get("relative_output_throughput", {}).get(
                "vs_custom_kernels"
            )
            print(f"  {backend:>23}: {fmt(ratio)}x")


def write_csv(path: Path, results: dict[str, dict[str, Any]]) -> None:
    fields = (
        "backend",
        "request_throughput_rps",
        "output_throughput_tok_s",
        "total_throughput_tok_s",
        "ttft_median_ms",
        "ttft_p95_ms",
        "tpot_median_ms",
        "tpot_p95_ms",
        "itl_median_ms",
        "itl_p95_ms",
        "e2e_median_ms",
        "e2e_p95_ms",
        "peak_gpu_memory_bytes",
        "setup_time_s",
        "exact_request_fraction_vs_pytorch_baseline",
        "min_matched_prefix_tokens_vs_pytorch_baseline",
        "output_throughput_vs_pytorch_baseline",
        "output_throughput_vs_custom_kernels",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for backend, result in results.items():
            summary = result["summary"]
            correctness = result.get("correctness_vs_pytorch_baseline", {})
            relative = result.get("relative_output_throughput", {})
            writer.writerow({
                "backend": backend,
                "request_throughput_rps": summary.get("request_throughput_rps"),
                "output_throughput_tok_s": summary.get("output_throughput_tok_s"),
                "total_throughput_tok_s": summary.get("total_throughput_tok_s"),
                "ttft_median_ms": metric_value(summary, "ttft_ms"),
                "ttft_p95_ms": metric_value(summary, "ttft_ms", "p95"),
                "tpot_median_ms": metric_value(summary, "tpot_ms"),
                "tpot_p95_ms": metric_value(summary, "tpot_ms", "p95"),
                "itl_median_ms": metric_value(summary, "itl_ms"),
                "itl_p95_ms": metric_value(summary, "itl_ms", "p95"),
                "e2e_median_ms": metric_value(summary, "e2e_ms"),
                "e2e_p95_ms": metric_value(summary, "e2e_ms", "p95"),
                "peak_gpu_memory_bytes": summary.get("peak_gpu_memory_bytes"),
                "setup_time_s": summary.get("setup_time_s"),
                "exact_request_fraction_vs_pytorch_baseline": correctness.get(
                    "exact_request_fraction"
                ),
                "min_matched_prefix_tokens_vs_pytorch_baseline": correctness.get(
                    "min_matched_prefix_tokens"
                ),
                "output_throughput_vs_pytorch_baseline": relative.get(
                    "vs_pytorch_baseline"
                ),
                "output_throughput_vs_custom_kernels": relative.get(
                    "vs_custom_kernels"
                ),
            })


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", type=parse_backends,
                        default=list(DEFAULT_BACKENDS),
                        help="comma-separated backend names, or 'all'")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-running", type=int, default=4)
    parser.add_argument("--num-blocks", type=int,
                        help="override scheduler KV pool size; lower values exercise preemption")
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=4096,
        help="hard per-iteration token budget shared by decode and prefill",
    )
    parser.add_argument(
        "--max-prefill-chunk-size",
        type=int,
        help="maximum prompt tokens computed per request per prefill iteration",
    )
    parser.add_argument(
        "--max-prefill-attention-pairs",
        type=int,
        help=(
            "hard per-iteration prefill-attention work ceiling, measured as "
            "causal query-key pairs across packed prompt chunks"
        ),
    )
    parser.add_argument(
        "--prefill-tile-policy",
        choices=("static", "adaptive"),
        default="static",
        help="paged resumed-prefill tile dispatch policy",
    )
    parser.add_argument(
        "--decode-attention-policy",
        choices=("production", "adaptive"),
        default="production",
        help="paged-decode kernel dispatch policy for eager scheduler backends",
    )
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument(
        "--vllm-kv-cache-mode",
        choices=("matched", "native"),
        default="matched",
        help=(
            "'matched' gives vLLM the same logical KV pool as the local scheduler; "
            "'native' uses vLLM's gpu-memory-utilization reservation"
        ),
    )
    parser.add_argument(
        "--compare-with",
        action="append",
        type=Path,
        default=[],
        metavar="RESULT.json",
        help="merge a prior matched run (repeatable); exact workload/config equality is required",
    )

    parser.add_argument("--workload-name", default="synthetic-mixed")
    parser.add_argument("--num-requests", type=int, default=8)
    parser.add_argument("--prompt-lengths", default="128,512")
    parser.add_argument("--output-lengths", default="32,64")
    parser.add_argument("--request-rate", type=float, default=0.0,
                        help="Poisson arrivals per second; zero submits a burst")
    parser.add_argument("--vocab-size", type=int, default=151936)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workload-in", type=Path)
    parser.add_argument("--workload-out", type=Path)
    parser.add_argument("--write-workload-only", action="store_true")

    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--strict-backends", action="store_true",
                        help="fail instead of recording unavailable optional backends")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "results")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.warmups < 0 or args.repetitions < 1:
        raise ValueError("warmups must be >= 0 and repetitions must be >= 1")
    if args.max_running < 1 or args.block_size < 1:
        raise ValueError("max-running and block-size must be >= 1")
    if args.max_num_batched_tokens < args.max_running:
        raise ValueError("max-num-batched-tokens must be >= max-running")
    if args.max_prefill_chunk_size is not None and args.max_prefill_chunk_size < 1:
        raise ValueError("max-prefill-chunk-size must be >= 1")
    if (
        args.max_prefill_attention_pairs is not None
        and args.max_prefill_attention_pairs < 1
    ):
        raise ValueError("max-prefill-attention-pairs must be >= 1")
    if not 0 < args.vllm_gpu_memory_utilization <= 1:
        raise ValueError("vllm-gpu-memory-utilization must be in (0, 1]")
    workload = load_or_create_workload(args)
    if args.write_workload_only:
        print(json.dumps(workload.to_dict(), indent=2))
        return

    configuration = {
        "model": args.model,
        "device": args.device,
        "dtype": args.dtype,
        "block_size": args.block_size,
        "max_running": args.max_running,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "num_blocks": args.num_blocks,
        "max_prefill_chunk_size": args.max_prefill_chunk_size,
        "max_prefill_attention_pairs": args.max_prefill_attention_pairs,
        "prefill_tile_policy": args.prefill_tile_policy,
        "decode_attention_policy": args.decode_attention_policy,
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "vllm_kv_cache_mode": args.vllm_kv_cache_mode,
    }
    results, comparison_sources = load_comparison_results(
        args.compare_with, workload, configuration
    )
    print(
        f"workload={workload.name!r} requests={len(workload.requests)} "
        f"arrival={workload.arrival_pattern} "
        f"sha256={workload_fingerprint(workload)[:12]}"
    )
    if comparison_sources:
        print(f"imported matched results: {', '.join(str(p) for p in args.compare_with)}")
    unavailable: dict[str, str] = {}
    for name in args.backends:
        if name in results:
            raise ValueError(
                f"backend {name!r} is both imported via --compare-with and requested to run"
            )
        print(f"\n[{name}]")
        try:
            runs, summary = run_backend(name, workload, args)
        except BackendUnavailable as exc:
            if args.strict_backends:
                raise
            unavailable[name] = str(exc)
            print(f"  SKIP: {exc}")
            continue
        results[name] = {
            "summary": summary,
            "runs": [run.to_dict() for run in runs],
            "provenance": {"result_file": None, "system": system_metadata()},
        }

    attach_baseline_diagnostics(results)
    max_model_len = max(
        len(request.prompt_ids) + request.max_tokens
        for request in workload.requests
    )
    validate_comparison_contract(
        results,
        max_running=args.max_running,
        max_model_len=max_model_len,
        vllm_kv_cache_mode=args.vllm_kv_cache_mode,
        max_num_batched_tokens=args.max_num_batched_tokens,
    )
    attach_performance_comparisons(results)
    print_summary(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    stem = f"{workload.name}-{stamp}"
    json_path = args.output_dir / f"{stem}.json"
    csv_path = args.output_dir / f"{stem}.csv"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "system": system_metadata(),
        "workload": workload.to_dict(),
        "configuration": configuration,
        "comparison_contract": {
            "workload_sha256": workload_fingerprint(workload),
            "identical_prompt_token_ids": True,
            "identical_requested_output_lengths": True,
            "sampling": "greedy-temperature-0-ignore-eos",
            "max_concurrent_sequences": args.max_running,
            "vllm_context_limit": max_model_len,
            "latency_scope": (
                "per-request latency where backend API exposes token timestamps; "
                "vLLM V1 offline runs are throughput-only when those timestamps are absent"
            ),
            "memory_scope": (
                "matched logical KV-cache capacity; implementation overhead may differ"
                if args.vllm_kv_cache_mode == "matched" else
                "vLLM-native GPU reservation; not memory-matched"
            ),
        },
        "comparison_sources": comparison_sources,
        "unavailable_backends": unavailable,
        "backends": results,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    write_csv(csv_path, results)
    print(f"raw results: {json_path}")
    print(f"summary CSV: {csv_path}")


if __name__ == "__main__":
    main()
