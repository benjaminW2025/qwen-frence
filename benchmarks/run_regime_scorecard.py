#!/usr/bin/env python3
"""Run the final factorial H100 comparison and matched vLLM replay.

The burst matrix varies one workload axis at a time: active concurrency, prompt
length, and requested generation length. Local mode records the framework reference,
the pre-scheduler paged engine, the static scheduled engine, and the fitted
regime-dispatch candidate. VLLM mode imports each exact local workload and adds the
matched external result without silently changing cache capacity or scheduler limits.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
RUN_BENCHMARKS = HERE / "run_benchmarks.py"
REFERENCE_BACKENDS = (
    "pytorch-baseline",
    "paged-kv",
)
COMPETITIVE_BACKENDS = (
    "custom-kernels",
    "regime-dispatched",
)
REFERENCE_ANCHORS = frozenset({
    "lowB-shortP-shortO",   # launch-bound control
    "lowB-longP-shortO",    # prefill-heavy control
    "highB-shortP-longO",   # decode-heavy saturation
    "highB-longP-longO",    # dual-heavy mixed execution
})


@dataclass(frozen=True)
class Regime:
    name: str
    concurrency: int
    prompt_length: int
    output_length: int


def factorial_regimes() -> tuple[Regime, ...]:
    regimes = []
    for concurrency, concurrency_name in ((8, "lowB"), (64, "highB")):
        for prompt_length, prompt_name in ((256, "shortP"), (8192, "longP")):
            for output_length, output_name in ((32, "shortO"), (256, "longO")):
                regimes.append(Regime(
                    f"{concurrency_name}-{prompt_name}-{output_name}",
                    concurrency,
                    prompt_length,
                    output_length,
                ))
    return tuple(regimes)


REGIMES = factorial_regimes()
REGIME_BY_NAME = {regime.name: regime for regime in REGIMES}


def parse_regimes(value: str) -> list[Regime]:
    if value.strip() == "all":
        return list(REGIMES)
    names = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [name for name in names if name not in REGIME_BY_NAME]
    if not names or unknown:
        raise argparse.ArgumentTypeError(
            f"unknown regime(s) {unknown}; choose from {', '.join(REGIME_BY_NAME)}"
        )
    return [REGIME_BY_NAME[name] for name in names]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("local", "vllm"), default="local")
    parser.add_argument("--regimes", type=parse_regimes, default=list(REGIMES))
    parser.add_argument(
        "--local-suite",
        type=Path,
        help=(
            "existing suite to resume in local mode or replay in vllm mode; "
            "required for vllm mode"
        ),
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--reference-policy",
        choices=("none", "anchors", "all"),
        default="none",
        help=(
            "historical PyTorch/paged coverage: none (default), four anchors, or "
            "all eight regimes; reference runs use no warmup and one measurement"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path,
        default=HERE / "results" / "regime-scorecard",
    )
    return parser


def _common_command(
    args, regime: Regime, *, warmups: int | None = None, repetitions: int | None = None,
) -> list[str]:
    return [
        sys.executable,
        str(RUN_BENCHMARKS),
        "--model", args.model,
        "--device", args.device,
        "--dtype", args.dtype,
        "--block-size", str(args.block_size),
        "--max-running", str(regime.concurrency),
        "--max-num-batched-tokens", str(args.max_num_batched_tokens),
        "--warmups", str(args.warmups if warmups is None else warmups),
        "--repetitions", str(
            args.repetitions if repetitions is None else repetitions
        ),
        "--seed", str(args.seed),
        "--vllm-kv-cache-mode", "matched",
        "--strict-backends",
    ]


def _workload_args(regime: Regime) -> list[str]:
    return [
        "--workload-name", f"regime-{regime.name}",
        "--num-requests", str(regime.concurrency),
        "--prompt-lengths", str(regime.prompt_length),
        "--output-lengths", str(regime.output_length),
    ]


def reference_command(args, regime: Regime, case_dir: Path) -> list[str]:
    """Measure historical serial references once; their margin dwarfs timing noise."""
    return _common_command(args, regime, warmups=0, repetitions=1) + [
        "--backends", ",".join(REFERENCE_BACKENDS),
        *_workload_args(regime),
        "--workload-out", str(case_dir / "workload.json"),
        "--output-dir", str(case_dir / "reference"),
    ]


def competitive_command(
    args, regime: Regime, case_dir: Path, reference_result: Path | None = None,
) -> list[str]:
    command = _common_command(args, regime) + [
        "--backends", ",".join(COMPETITIVE_BACKENDS),
    ]
    if reference_result is not None:
        command += ["--compare-with", str(reference_result)]
    else:
        command += _workload_args(regime) + [
            "--workload-out", str(case_dir / "workload.json"),
        ]
    return command + [
        "--output-dir", str(case_dir / "local"),
    ]


def _single_result(directory: Path) -> Path:
    paths = sorted(directory.glob("*.json"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one result JSON in {directory}, found {paths}")
    return paths[0]


def _optional_result(directory: Path) -> Path | None:
    paths = sorted(directory.glob("*.json"))
    if len(paths) > 1:
        raise RuntimeError(f"expected at most one result JSON in {directory}, found {paths}")
    return paths[0] if paths else None


def validate_suite_path(path: Path) -> Path:
    """Reject documentation placeholders before they become result directories."""
    placeholders = {"SUITE_DIRECTORY", "suite-REPLACE_WITH_TIMESTAMP"}
    if path.name in placeholders:
        raise ValueError(
            f"{path} is a documentation placeholder; use the timestamped suite "
            "path printed by the completed local run"
        )
    return path.resolve()


def vllm_command(args, regime: Regime, local_suite: Path, case_dir: Path) -> list[str]:
    local_result = _single_result(local_suite / regime.name / "local")
    return _common_command(args, regime) + [
        "--backends", "vllm",
        "--compare-with", str(local_result),
        "--output-dir", str(case_dir / "vllm"),
    ]


def _metric(summary, family, statistic):
    values = summary.get(family)
    return values.get(statistic) if isinstance(values, dict) else None


def write_aggregate(suite_dir: Path, records: list[dict]) -> Path | None:
    rows = []
    for record in records:
        result_path = record.get("result")
        if record.get("returncode") != 0 or not result_path:
            continue
        payload = json.loads(Path(result_path).read_text())
        summaries = {
            backend: result["summary"]
            for backend, result in payload["backends"].items()
        }
        static_throughput = summaries.get("custom-kernels", {}).get(
            "output_throughput_tok_s"
        )
        for backend, summary in summaries.items():
            throughput = summary.get("output_throughput_tok_s")
            rows.append({
                **asdict(REGIME_BY_NAME[record["regime"]]),
                "backend": backend,
                "request_throughput_rps": summary.get("request_throughput_rps"),
                "output_throughput_tok_s": throughput,
                "total_throughput_tok_s": summary.get("total_throughput_tok_s"),
                "ttft_p50_ms": _metric(summary, "ttft_ms", "median"),
                "ttft_p95_ms": _metric(summary, "ttft_ms", "p95"),
                "tpot_p50_ms": _metric(summary, "tpot_ms", "median"),
                "tpot_p95_ms": _metric(summary, "tpot_ms", "p95"),
                "itl_p95_ms": _metric(summary, "itl_ms", "p95"),
                "e2e_p95_ms": _metric(summary, "e2e_ms", "p95"),
                "peak_gpu_memory_bytes": summary.get("peak_gpu_memory_bytes"),
                "output_throughput_vs_static": (
                    throughput / static_throughput
                    if throughput is not None and static_throughput not in (None, 0)
                    else None
                ),
            })
    if not rows:
        return None
    path = suite_dir / "summary.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> int:
    args = build_parser().parse_args()
    if args.warmups < 0 or args.repetitions < 1:
        raise ValueError("warmups must be non-negative and repetitions positive")
    if args.mode == "vllm" and args.local_suite is None:
        raise ValueError("--local-suite is required in vllm mode")

    if args.mode == "local" and args.local_suite is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        suite_dir = args.output_dir / f"suite-{stamp}"
    else:
        suite_dir = validate_suite_path(args.local_suite)

    if args.reference_policy == "all":
        reference_regimes = {regime.name for regime in args.regimes}
    elif args.reference_policy == "anchors":
        reference_regimes = REFERENCE_ANCHORS
    else:
        reference_regimes = frozenset()
    planned = []
    for regime in args.regimes:
        case_dir = suite_dir / regime.name
        if args.mode == "local":
            commands = []
            completed_local = _optional_result(case_dir / "local")
            completed_reference = _optional_result(case_dir / "reference")
            if completed_local is not None:
                commands.append(("reuse", None))
            elif regime.name in reference_regimes and completed_reference is not None:
                commands.append(("competitive", competitive_command(
                    args, regime, case_dir, completed_reference
                )))
            elif regime.name in reference_regimes:
                commands.append(("reference", reference_command(args, regime, case_dir)))
                # The result path is resolved after the reference subprocess finishes.
                commands.append(("competitive", None))
            else:
                commands.append(("competitive", competitive_command(
                    args, regime, case_dir
                )))
        else:
            commands = [("vllm", vllm_command(args, regime, suite_dir, case_dir))]
        planned.append((regime, case_dir, commands))

    if args.dry_run:
        print(f"suite: {suite_dir}")
        for regime, case_dir, commands in planned:
            for stage, command in commands:
                if stage == "reuse":
                    print(f"[{regime.name}/reuse] existing local result")
                    continue
                if command is None:
                    command = competitive_command(
                        args, regime, case_dir, Path("REFERENCE_RESULT.json")
                    )
                print(f"[{regime.name}/{stage}] {' '.join(command)}")
        return 0

    suite_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, (regime, case_dir, commands) in enumerate(planned, 1):
        print(f"\n{'=' * 88}")
        print(f"[{index}/{len(planned)}] {regime.name}")
        executed_commands = []
        returncode = 0
        for stage, command in commands:
            if stage == "reuse":
                print("[reuse] existing local result", flush=True)
                continue
            if command is None:
                reference_result = _single_result(case_dir / "reference")
                command = competitive_command(
                    args, regime, case_dir, reference_result
                )
            print(f"[{stage}] {' '.join(command)}", flush=True)
            completed = subprocess.run(command, check=False)
            executed_commands.append(command)
            returncode = completed.returncode
            if returncode:
                break
        result_dir = case_dir / ("local" if args.mode == "local" else "vllm")
        result = None
        if returncode == 0:
            result = str(_single_result(result_dir))
        records.append({
            "regime": regime.name,
            "returncode": returncode,
            "result": result,
            "commands": executed_commands,
            "reused": not executed_commands,
        })
        if returncode and args.fail_fast:
            break

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "systems": {
            "framework_reference": "pytorch-baseline",
            "pre_scheduler_engine": "paged-kv",
            "scheduled_static_control": "custom-kernels",
            "post_policy_engine": "regime-dispatched",
            "external_reference": "vllm",
        },
        "design": {
            "kind": "2x2x2-factorial-burst",
            "concurrency": [8, 64],
            "prompt_tokens": [256, 8192],
            "output_tokens": [32, 256],
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "reference_protocol": (
                "disabled"
                if args.reference_policy == "none" else
                f"one-run-no-warmup policy={args.reference_policy}"
            ),
            "reference_anchors": sorted(reference_regimes),
            "regimes": [asdict(regime) for regime in args.regimes],
        },
        "records": records,
        "passed": (
            len(records) == len(planned)
            and all(record["returncode"] == 0 for record in records)
        ),
    }
    manifest_path = suite_dir / f"manifest-{args.mode}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    summary_path = write_aggregate(suite_dir, records)
    print(f"\nmanifest: {manifest_path}")
    if summary_path:
        print(f"aggregate: {summary_path}")
    return 0 if manifest["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
