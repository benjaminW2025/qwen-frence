#!/usr/bin/env python3
"""Run the complete inference-engine correctness suite in isolated processes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
CHECKS_DIR = HERE / "checks"
CHECKS = {
    "baseline-vs-hf": (
        "check_baseline_vs_hf.py",
        "custom PyTorch Qwen forward versus Hugging Face",
    ),
    "paged-attention": (
        "check_paged_decode_attention.py",
        "paged Triton decode attention versus SDPA/manual references",
    ),
    "paged-vs-baseline": (
        "check_paged_vs_baseline.py",
        "teacher-forced paged engine versus contiguous baseline",
    ),
    "cuda-graph": (
        "check_graph_vs_eager.py",
        "single-sequence CUDA graph versus baseline and eager paged decode",
    ),
    "scheduler": (
        "check_scheduler.py",
        "continuous batching, constrained KV pool, and preemption",
    ),
    "ragged-prefill": (
        "check_ragged_prefill.py",
        "mixed token-budget scheduling, resumable attention, logits, KV, and admission",
    ),
    "bucketed-graphs": (
        "check_bucketed_graph.py",
        "bucket selection, padding, replay parity, and scheduler integration",
    ),
    "custom-kernels": (
        "check_custom_kernels.py",
        "Triton RMSNorm, RoPE, SwiGLU, fused KV-write, and model parity",
    ),
}
PRIMARY_CHECKS = [
    "baseline-vs-hf",
    "paged-attention",
    "paged-vs-baseline",
    "scheduler",
    "ragged-prefill",
    "custom-kernels",
]


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_metadata() -> dict:
    root = HERE.parent
    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: package_version(name)
            for name in ("torch", "transformers", "triton", "vllm")
        },
    }
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout
        metadata["repository"] = {"commit": commit, "dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        metadata["repository"] = {"commit": None, "dirty": None}

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


def parse_checks(value: str) -> list[str]:
    if value.strip() == "all":
        return list(CHECKS)
    if value.strip() == "baseline":
        return PRIMARY_CHECKS
    selected = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [name for name in selected if name not in CHECKS]
    if not selected or unknown:
        raise argparse.ArgumentTypeError(
            f"unknown check(s) {unknown}; choose from {', '.join(CHECKS)}"
        )
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checks",
        type=parse_checks,
        default=list(CHECKS),
        help="comma-separated check names, 'baseline', or 'all'",
    )
    parser.add_argument("--list", action="store_true", help="list checks and exit")
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="stop after the first failure instead of collecting the full summary",
    )
    parser.add_argument("--json-out", type=Path, help="optional machine-readable summary")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.list:
        for name, (_, description) in CHECKS.items():
            print(f"{name:<20} {description}")
        return 0

    results = []
    suite_started = time.perf_counter()
    for index, name in enumerate(args.checks, start=1):
        filename, description = CHECKS[name]
        path = CHECKS_DIR / filename
        print("\n" + "#" * 88, flush=True)
        print(f"[{index}/{len(args.checks)}] {name}: {description}", flush=True)
        print(f"$ {sys.executable} {path}", flush=True)
        print("#" * 88, flush=True)

        started = time.perf_counter()
        completed = subprocess.run([sys.executable, str(path)], check=False)
        duration = time.perf_counter() - started
        passed = completed.returncode == 0
        results.append(
            {
                "name": name,
                "description": description,
                "script": str(path.relative_to(HERE)),
                "passed": passed,
                "exit_code": completed.returncode,
                "duration_s": duration,
            }
        )
        print(f"\n[{name}] {'PASS' if passed else 'FAIL'} ({duration:.1f}s)", flush=True)
        if not passed and args.fail_fast:
            break

    duration = time.perf_counter() - suite_started
    passed_count = sum(result["passed"] for result in results)
    print("\n" + "=" * 88)
    print(f"{'check':<20} | {'result':<6} | {'seconds':>9}")
    print("-" * 88)
    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"{result['name']:<20} | {status:<6} | {result['duration_s']:>9.1f}")
    print("=" * 88)
    print(f"OVERALL: {passed_count}/{len(results)} passed in {duration:.1f}s")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "command": sys.argv,
            "python": sys.executable,
            "system": environment_metadata(),
            "duration_s": duration,
            "passed": passed_count == len(results),
            "checks": results,
        }
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"summary: {args.json_out}")

    return 0 if passed_count == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
