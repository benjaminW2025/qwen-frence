#!/usr/bin/env python3
"""Collect dense H100 crossover sweeps and fit constrained dispatch policies."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent

SWEEPS = {
    "decode": HERE / "decode" / "benchmark_decode_kernel_causality.py",
    "swiglu": HERE / "mlp" / "benchmark_swiglu_fusion.py",
    "rope-kv": HERE / "memory" / "benchmark_rope_kv_fusion.py",
    "paged-prefill": HERE / "prefill" / "benchmark_paged_prefill_tiles.py",
}
FITTER = HERE / "dispatch" / "fit_dispatch_policies.py"

DENSE_AXES = {
    "decode": {
        "--batch-sizes": "1,4,8,16,32,64",
        "--context-lengths": "128,256,512,1024,2048,4096,8192,16384",
        "--warmups": "3",
        "--repetitions": "20",
    },
    "swiglu": {
        "--rows": "128,256,384,512,768,1024,1280,1536,1792,2048,2560,3072,4096,6144,8192,12288,16384",
        "--warmups": "5",
        "--repetitions": "20",
    },
    "rope-kv": {
        "--token-counts": "64,128,256,512,768,1024,1536,2048,3072,4096,6144,8192,12288,16384",
        "--warmups": "3",
        "--repetitions": "20",
    },
    "paged-prefill": {
        "--batch-sizes": "1,2,4,8",
        "--query-lengths": "128,256,384,512,768,1024,1536,2048",
        "--prefix-lengths": "0,2048,8192,16384",
        "--warmups": "2",
        "--repetitions": "10",
    },
}


def parse_names(value: str) -> list[str]:
    if value.strip() == "all":
        return list(SWEEPS)
    names = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [name for name in names if name not in SWEEPS]
    if not names or unknown:
        raise argparse.ArgumentTypeError(
            f"unknown sweeps {unknown}; choose from {', '.join(SWEEPS)}"
        )
    return names


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweeps", type=parse_names, default=list(SWEEPS))
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--holdout-modulus", type=int, default=5)
    parser.add_argument("--max-training-regression", type=float, default=0.02)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path,
        default=HERE / "results" / "dispatch-policy-suite",
    )
    return parser


def sweep_command(name: str, args, run_dir: Path) -> list[str]:
    command = [
        sys.executable, str(SWEEPS[name]),
        "--device", args.device,
        "--dtype", args.dtype,
        "--seed", str(args.seed),
        "--output-dir", str(run_dir / name),
    ]
    for option, value in DENSE_AXES[name].items():
        command.extend((option, value))
    return command


def newest_result_json(directory: Path) -> Path:
    candidates = sorted(directory.glob("*.json"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one result JSON in {directory}, found {len(candidates)}"
        )
    return candidates[0]


def fit_command(args, run_dir: Path, results: dict[str, Path]) -> list[str]:
    command = [
        sys.executable, str(FITTER),
        "--max-depth", str(args.max_depth),
        "--holdout-modulus", str(args.holdout_modulus),
        "--max-training-regression", str(args.max_training_regression),
        "--output-dir", str(run_dir / "policy"),
    ]
    options = {
        "decode": "--decode-json",
        "swiglu": "--swiglu-json",
        "rope-kv": "--rope-kv-json",
        "paged-prefill": "--paged-prefill-json",
    }
    for name in args.sweeps:
        command.extend((options[name], str(results[name])))
    return command


def main() -> int:
    args = build_parser().parse_args()
    if args.max_depth < 0 or args.holdout_modulus < 2:
        raise ValueError("max depth must be non-negative and holdout modulus at least two")
    if not 0 <= args.max_training_regression < 1:
        raise ValueError("max training regression must be in [0, 1)")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = args.output_dir / f"suite-{stamp}"
    commands = [(name, sweep_command(name, args, run_dir)) for name in args.sweeps]
    if args.dry_run:
        for name, command in commands:
            print(f"[{name}] {' '.join(command)}")
        print(
            "[fit] fit_dispatch_policies.py consumes the four result JSONs, "
            f"max_depth={args.max_depth}, holdout_modulus={args.holdout_modulus}, "
            f"max_training_regression={args.max_training_regression}"
        )
        return 0

    run_dir.mkdir(parents=True, exist_ok=True)
    records = []
    results = {}
    for index, (name, command) in enumerate(commands, 1):
        print(f"\n{'=' * 88}")
        print(f"[{index}/{len(commands)}] dense {name} crossover sweep")
        print(" ".join(command), flush=True)
        completed = subprocess.run(command, check=False)
        record = {
            "name": name,
            "command": command,
            "returncode": completed.returncode,
            "output_dir": str(run_dir / name),
        }
        records.append(record)
        if completed.returncode:
            if args.fail_fast:
                break
            continue
        results[name] = newest_result_json(run_dir / name)

    fit_record = None
    if len(results) == len(args.sweeps):
        command = fit_command(args, run_dir, results)
        print(f"\n{'=' * 88}")
        print("[fit] constrained rule fitting and held-out oracle regret")
        print(" ".join(command), flush=True)
        completed = subprocess.run(command, check=False)
        fit_record = {
            "name": "fit",
            "command": command,
            "returncode": completed.returncode,
            "output_dir": str(run_dir / "policy"),
        }

    passed = (
        len(records) == len(commands)
        and all(record["returncode"] == 0 for record in records)
        and fit_record is not None
        and fit_record["returncode"] == 0
    )
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "sweeps": args.sweeps,
            "dtype": args.dtype,
            "device": args.device,
            "seed": args.seed,
            "max_depth": args.max_depth,
            "holdout_modulus": args.holdout_modulus,
            "max_training_regression": args.max_training_regression,
            "axes": {name: DENSE_AXES[name] for name in args.sweeps},
        },
        "sweeps": records,
        "fit": fit_record,
        "passed": passed,
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nmanifest: {manifest_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
