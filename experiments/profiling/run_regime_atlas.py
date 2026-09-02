#!/usr/bin/env python3
"""Collect detailed operator traces at representative workload-surface corners."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
ROOT = EXPERIMENTS.parent
PROFILE_SCRIPT = EXPERIMENTS / "scheduler" / "profile_mixed_batch.py"


@dataclass(frozen=True)
class Regime:
    name: str
    decode_requests: int
    decode_context: int
    prefill_requests: int
    prefill_chunk: int
    prefill_prefix: int
    rationale: str


REGIMES = (
    Regime("launch_bound", 8, 128, 2, 256, 0,
           "small fresh work where mixed packing previously saved 49%"),
    Regime("decode_low_concurrency", 8, 8192, 2, 256, 0,
           "long-context decode at low concurrency with a small fresh prefill"),
    Regime("balanced_fresh", 32, 2048, 2, 2048, 0,
           "representative fresh mixed iteration"),
    Regime("balanced_resumed", 32, 2048, 2, 2048, 4096,
           "matched resumed-prefix counterpart"),
    Regime("decode_bandwidth", 64, 8192, 2, 256, 0,
           "long-context high-batch decode with a small prefill"),
    Regime("long_fresh_prefill", 8, 128, 2, 8192, 0,
           "genuinely long fresh prefill during a small active decode batch"),
    Regime("prefill_compute", 8, 128, 2, 4096, 16384,
           "large resumed prefill with little decode work"),
    Regime("dual_heavy", 64, 8192, 2, 4096, 16384,
           "long-context decode and large resumed prefill together"),
)


def parse_names(value: str) -> list[str]:
    names = [part.strip() for part in value.split(",") if part.strip()]
    known = {regime.name for regime in REGIMES}
    unknown = [name for name in names if name not in known]
    if not names or unknown:
        raise argparse.ArgumentTypeError(
            f"unknown regimes {unknown}; choose from {', '.join(sorted(known))}"
        )
    return names


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regimes", type=parse_names,
                        default=[regime.name for regime in REGIMES])
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--implementation", default="custom-kernels")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--timing-repetitions", type=int, default=5)
    parser.add_argument("--prefill-tile-policy", choices=("static", "adaptive"),
                        default="static")
    parser.add_argument("--decode-attention-policy", choices=("production", "adaptive"),
                        default="production")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--output-dir", type=Path,
                        default=EXPERIMENTS / "results" / "regime-atlas")
    return parser


def command_for(args, regime: Regime, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(PROFILE_SCRIPT),
        "--model", args.model,
        "--implementation", args.implementation,
        "--decode-requests", str(regime.decode_requests),
        "--decode-context-length", str(regime.decode_context),
        "--prefill-requests", str(regime.prefill_requests),
        "--prefill-prefix-length", str(regime.prefill_prefix),
        "--prefill-chunk-size", str(regime.prefill_chunk),
        "--block-size", str(args.block_size),
        "--dtype", args.dtype,
        "--device", args.device,
        "--warmups", str(args.warmups),
        "--timing-repetitions", str(args.timing_repetitions),
        "--prefill-tile-policy", args.prefill_tile_policy,
        "--decode-attention-policy", args.decode_attention_policy,
        "--seed", str(args.seed),
        "--output-dir", str(output_dir / regime.name),
    ]


def main() -> None:
    args = build_parser().parse_args()
    selected = [regime for regime in REGIMES if regime.name in args.regimes]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = args.output_dir / f"atlas-{stamp}"
    commands = [command_for(args, regime, run_dir) for regime in selected]

    if args.dry_run:
        for regime, command in zip(selected, commands):
            print(f"[{regime.name}] {regime.rationale}")
            print(" ".join(command))
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, (regime, command) in enumerate(zip(selected, commands), 1):
        print(
            f"\n[{index}/{len(selected)}] {regime.name}: {regime.rationale}",
            flush=True,
        )
        completed = subprocess.run(command, check=False)
        record = {
            **regime.__dict__,
            "command": command,
            "returncode": completed.returncode,
        }
        records.append(record)
        if completed.returncode and not args.continue_on_error:
            break

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "regimes": records,
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nmanifest: {manifest_path}")
    failures = [record for record in records if record["returncode"]]
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
