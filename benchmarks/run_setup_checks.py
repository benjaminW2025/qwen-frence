#!/usr/bin/env python3
"""Fail-fast environment and smoke checks for paid GPU benchmark runs."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

from packaging.version import Version


HERE = Path(__file__).resolve().parent
RUN_BENCHMARKS = HERE / "run_benchmarks.py"
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B"
VLLM_VERSION = "0.10.2"
TRANSFORMERS_VERSION = "4.55.2"


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str
    remediation: str | None = None


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def version_contract(suite: str, versions: dict[str, str | None]) -> list[CheckResult]:
    required = ("torch", "transformers", "triton")
    if suite == "vllm":
        required += ("vllm",)
    results = []
    for package in required:
        value = versions.get(package)
        results.append(CheckResult(
            f"package:{package}",
            value is not None,
            value or "not installed",
            f"install the {suite} benchmark requirements" if value is None else None,
        ))

    if suite == "vllm":
        vllm = versions.get("vllm")
        transformers = versions.get("transformers")
        results.append(CheckResult(
            "vllm-version-contract",
            vllm == VLLM_VERSION and transformers == TRANSFORMERS_VERSION,
            f"vllm={vllm}, transformers={transformers}",
            (
                "install benchmarks/requirements-vllm-cu128.txt; the exact pin avoids "
                "the Qwen2Tokenizer compatibility failure"
            ),
        ))
    return results


def smoke_command(suite: str, model: str, output_dir: Path) -> list[str]:
    backends = "vllm" if suite == "vllm" else "custom-kernels,regime-dispatched"
    return [
        sys.executable,
        str(RUN_BENCHMARKS),
        "--backends", backends,
        "--model", model,
        "--device", "cuda",
        "--dtype", "float16",
        "--workload-name", f"{suite}-setup-smoke",
        "--num-requests", "1",
        "--prompt-lengths", "16",
        "--output-lengths", "2",
        "--max-running", "1",
        "--max-num-batched-tokens", "18",
        "--warmups", "0",
        "--repetitions", "1",
        "--strict-backends",
        "--output-dir", str(output_dir),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("local", "vllm"), required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--allow-non-h100",
        action="store_true",
        help="require CUDA but permit a GPU other than the scorecard's H100",
    )
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="only inspect imports, versions, CUDA, model config, and tokenizer",
    )
    parser.add_argument("--json-out", type=Path)
    return parser


def _append(results: list[CheckResult], name: str, operation, remediation: str) -> object | None:
    try:
        value, detail = operation()
        results.append(CheckResult(name, True, detail))
        return value
    except Exception as exc:  # Environment probes should report all independent failures.
        results.append(CheckResult(
            name,
            False,
            f"{type(exc).__name__}: {exc}",
            remediation,
        ))
        return None


def run_checks(args: argparse.Namespace) -> list[CheckResult]:
    results = [CheckResult(
        "python",
        Version(platform.python_version()) >= Version("3.10"),
        platform.python_version(),
        "use Python 3.10 or newer",
    )]
    versions = {
        name: package_version(name)
        for name in ("torch", "transformers", "triton", "vllm")
    }
    results.extend(version_contract(args.suite, versions))

    torch = _append(
        results,
        "torch-import",
        lambda: (__import__("torch"), f"torch {versions['torch']} imported"),
        "install the benchmark environment before running the GPU suite",
    )
    cuda_ready = False
    if torch is not None:
        cuda_ready = bool(torch.cuda.is_available())
        detail = "CUDA unavailable"
        if cuda_ready:
            index = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(index)
            detail = (
                f"{props.name}; compute capability {props.major}.{props.minor}; "
                f"torch CUDA {torch.version.cuda}"
            )
        results.append(CheckResult(
            "cuda",
            cuda_ready,
            detail,
            "run this suite inside the H100 GPU environment",
        ))
        if cuda_ready:
            gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
            is_h100 = "H100" in gpu_name.upper()
            results.append(CheckResult(
                "scorecard-gpu",
                is_h100 or args.allow_non_h100,
                gpu_name,
                "use an H100, or pass --allow-non-h100 for a non-scorecard smoke test",
            ))

    transformers = _append(
        results,
        "transformers-import",
        lambda: (
            __import__("transformers"),
            f"transformers {versions['transformers']} imported",
        ),
        "install the benchmark environment before loading the model metadata",
    )
    if transformers is not None:
        config = _append(
            results,
            "model-config",
            lambda: _load_model_config(transformers, args.model),
            f"confirm access to the exact model {args.model}",
        )
        if config is not None:
            architecture = set(getattr(config, "architectures", ()) or ())
            valid = (
                getattr(config, "model_type", None) == "qwen2"
                and "Qwen2ForCausalLM" in architecture
            )
            results.append(CheckResult(
                "model-identity",
                valid,
                (
                    f"requested={args.model}; model_type={getattr(config, 'model_type', None)}; "
                    f"architectures={sorted(architecture)}"
                ),
                f"use --model {DEFAULT_MODEL} for the matched scorecard",
            ))
        tokenizer = _append(
            results,
            "tokenizer",
            lambda: _load_tokenizer(transformers, args.model),
            (
                "confirm model access and, for vLLM 0.10.2, pin "
                f"transformers=={TRANSFORMERS_VERSION}"
            ),
        )
        if tokenizer is not None and args.suite == "vllm":
            compatible = hasattr(tokenizer, "all_special_tokens_extended")
            results.append(CheckResult(
                "vllm-tokenizer-api",
                compatible,
                (
                    f"{type(tokenizer).__name__}; "
                    f"all_special_tokens_extended={compatible}"
                ),
                f"pin transformers=={TRANSFORMERS_VERSION}",
            ))

    prerequisites_pass = all(result.passed for result in results)
    if not args.skip_smoke and prerequisites_pass:
        with tempfile.TemporaryDirectory(prefix=f"{args.suite}-setup-smoke-") as directory:
            command = smoke_command(args.suite, args.model, Path(directory))
            completed = subprocess.run(command, text=True)
        results.append(CheckResult(
            "end-to-end-smoke",
            completed.returncode == 0,
            f"exit code {completed.returncode}; 1 request, 16 prompt + 2 output tokens",
            "inspect the traceback above before starting the full scorecard",
        ))
    elif not args.skip_smoke:
        results.append(CheckResult(
            "end-to-end-smoke",
            False,
            "not run because a static prerequisite failed",
            "resolve the failed checks above, then rerun this command",
        ))
    return results


def _load_model_config(transformers, model: str):
    config = transformers.AutoConfig.from_pretrained(model)
    return config, f"loaded {model}"


def _load_tokenizer(transformers, model: str):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model)
    token_ids = tokenizer.encode("preflight", add_special_tokens=False)
    if not token_ids:
        raise RuntimeError("tokenizer returned no IDs for the smoke text")
    return tokenizer, f"{type(tokenizer).__name__}; encoded {len(token_ids)} token(s)"


def main() -> int:
    args = build_parser().parse_args()
    results = run_checks(args)
    print("\nSetup confirmation")
    print("=" * 80)
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.name}: {result.detail}")
        if not result.passed and result.remediation:
            print(f"       fix: {result.remediation}")
    passed = all(result.passed for result in results)
    print("=" * 80)
    print("SETUP PASS" if passed else "SETUP FAIL")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps({
            "suite": args.suite,
            "model": args.model,
            "passed": passed,
            "checks": [asdict(result) for result in results],
        }, indent=2) + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
