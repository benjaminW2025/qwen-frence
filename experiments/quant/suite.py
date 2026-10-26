"""Run baseline quantization experiments and record actual hardware coverage.

Hopper: python experiments/quant/quant.py baselines --output-dir artifacts/quant-baselines
CPU:    python experiments/quant/quant.py baselines --reference-only --output-dir artifacts/quant-reference

All GPU results remain hardware-specific. CPU reference math does not validate
Triton or FA3 kernels. Missing stages are explicit and never counted as passes.
"""
import argparse
import json
from pathlib import Path
import traceback

import torch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--prefixes", nargs="+", type=int, default=[256, 2048, 8192])
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--input", type=Path, help="optional real QKV capture for local attention")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("use a fresh output directory")
    if (min(args.batch, args.samples, args.repeats, *args.prefixes) < 1
            or args.batch < 2 or max(args.prefixes) >= 32768 or args.warmup < 0):
        parser.error("invalid dimensions or repeat counts")
    args.output_dir.mkdir(parents=True)
    device = ("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    cuda = torch.device(device).type == "cuda" and torch.cuda.is_available()
    capability = torch.cuda.get_device_capability(device) if cuda else None
    report = {"status": "running", "device": device, "capability": capability,
              "stages": [], "note": "accuracy acceptance thresholds are not configured; no quality certification"}

    def save():
        (args.output_dir / "suite.json").write_text(json.dumps(report, indent=2) + "\n")

    def run_stage(name, callback, argv):
        try:
            code = callback(argv) or 0
            report["stages"].append({"name": name, "status": "complete" if code == 0 else "failed", "exit_code": code})
        except Exception as error:
            report["stages"].append({"name": name, "status": "failed",
                                     "error": str(error), "traceback": traceback.format_exc()})
        save()

    from experiments.quant import attention
    attention_args = ["--device", device, "--sweep-heads", "--warmup", str(args.warmup),
                      "--repeats", str(args.repeats), "--output", str(args.output_dir / "attention.json")]
    if args.input:
        attention_args += ["--input", str(args.input)]
    if args.reference_only or not cuda or capability[0] < 8:
        attention_args += ["--reference-only"]
    run_stage("attention-reference" if "--reference-only" in attention_args else "attention-tiles",
              attention.main, attention_args)

    if not args.reference_only and capability == (9, 0):
        from experiments.quant import full_model
        common = ["--device", device, "--model", args.model, "--batch", str(args.batch),
                  "--prefixes", *map(str, args.prefixes), "--samples", str(args.samples),
                  "--warmup", str(args.warmup), "--repeats", str(args.repeats)]
        run_stage("full-model-projections-staged-kv", full_model.main,
                  [*common, "--output", str(args.output_dir / "full-model.json")])
        run_stage("full-model-native-fa3", full_model.main,
                  [*common, "--variants", "--kv-modes", "off", "--scenarios", "decode",
                   "--native-fa3", "--output", str(args.output_dir / "native-fa3.json")])
    else:
        for name in ("full-model-projections-staged-kv", "full-model-native-fa3"):
            report["stages"].append({"name": name, "status": "not-run",
                                     "reason": "reference-only requested" if args.reference_only else "requires Hopper SM90 and final FA3 environment"})
    statuses = [row["status"] for row in report["stages"]]
    report["status"] = ("failed" if "failed" in statuses else
                        "reference-only-complete" if args.reference_only else
                        "partial" if "not-run" in statuses else "complete")
    save()
    print(f"Baseline suite: {report['status']}; report: {args.output_dir / 'suite.json'}")
    return 1 if report["status"] == "failed" else 2 if report["status"] == "partial" else 0
