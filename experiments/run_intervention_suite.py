#!/usr/bin/env python3
"""Run every remaining isolated H100 intervention and write one manifest."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent

EXPERIMENTS = {
    "decode-causality": HERE / "decode" / "benchmark_decode_kernel_causality.py",
    "batched-sampling": HERE / "scheduler" / "benchmark_batched_sampling.py",
    "swiglu": HERE / "mlp" / "benchmark_swiglu_fusion.py",
    "metadata-staging": HERE / "scheduler" / "benchmark_metadata_staging.py",
    "rope-kv-fusion": HERE / "memory" / "benchmark_rope_kv_fusion.py",
    "attention-overlap": HERE / "attention" / "benchmark_attention_overlap.py",
    "launch-profile": HERE / "profiling" / "run_regime_atlas.py",
}


def parse_names(value):
    if value.strip() == "all":
        return list(EXPERIMENTS)
    names = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [name for name in names if name not in EXPERIMENTS]
    if not names or unknown:
        raise argparse.ArgumentTypeError(
            f"unknown experiments {unknown}; choose from {', '.join(EXPERIMENTS)}"
        )
    return names


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", type=parse_names, default=list(EXPERIMENTS))
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path,
        default=HERE / "results" / "intervention-suite",
    )
    return parser


def command_for(name, args, run_dir):
    common = [
        sys.executable, str(EXPERIMENTS[name]),
        "--device", args.device,
        "--output-dir", str(run_dir / name),
    ]
    if name != "metadata-staging":
        common += ["--dtype", args.dtype, "--seed", str(args.seed)]
    if name == "launch-profile":
        return common + [
            "--model", args.model,
            "--regimes", (
                "launch_bound,decode_low_concurrency,long_fresh_prefill"
            ),
            "--timing-repetitions", "20",
        ]
    return common


def main():
    args = build_parser().parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = args.output_dir / f"suite-{stamp}"
    commands = [
        (name, command_for(name, args, run_dir))
        for name in args.experiments
    ]
    if args.dry_run:
        for name, command in commands:
            print(f"[{name}] {' '.join(command)}")
        return 0

    run_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, (name, command) in enumerate(commands, 1):
        print(f"\n{'=' * 88}")
        print(f"[{index}/{len(commands)}] {name}")
        print(" ".join(command), flush=True)
        completed = subprocess.run(command, check=False)
        records.append({
            "name": name,
            "command": command,
            "returncode": completed.returncode,
            "output_dir": str(run_dir / name),
        })
        if completed.returncode and args.fail_fast:
            break

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "experiments": args.experiments,
            "model": args.model,
            "dtype": args.dtype,
            "device": args.device,
            "seed": args.seed,
        },
        "experiments": records,
        "passed": all(record["returncode"] == 0 for record in records)
        and len(records) == len(commands),
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nmanifest: {manifest_path}")
    failures = [record for record in records if record["returncode"]]
    if failures:
        print("failed: " + ", ".join(record["name"] for record in failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
