# Reproducing correctness and performance

The published measurements target one NVIDIA H100 80 GB and
`Qwen/Qwen2.5-1.5B` in FP16. Run commands from the repository root. Every GPU workflow
starts with a preflight and writes machine-readable results under one artifact root.

## 1. Local engine environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Confirm the control-plane tests without loading a model:

```bash
make unit
```

Run the complete local reproduction. `ARTIFACT_DIR` makes the output location
explicit and reusable across commands:

```bash
make reproduce-local \
  ARTIFACT_DIR=artifacts/reproduction/h100-local
```

This performs, in order:

1. CPU harness contract tests for benchmarks, correctness orchestration, and experiments.
2. CUDA/H100, model, tokenizer, and one-request local-engine preflight.
3. All eight isolated GPU correctness checks.
4. The eight-cell local factorial scorecard with one warmup and three repetitions.

The scorecard prints its timestamped suite path. Save that path for the external replay.
An interrupted local scorecard can resume directly:

```bash
python benchmarks/run_regime_scorecard.py \
  --mode local \
  --local-suite artifacts/reproduction/h100-local/regime-scorecard/suite-TIMESTAMP \
  --warmups 1 \
  --repetitions 3 \
  --seed 0 \
  --fail-fast
```

For a quick validation while developing, use `make correctness-primary` or
`make benchmark-smoke`. The primary correctness alias excludes only the two legacy
CUDA-graph checks.

## 2. Matched vLLM environment

vLLM uses a separate pinned environment because it owns its PyTorch/CUDA dependency
stack:

```bash
uv venv --python 3.12 /root/vllm-bench-env
uv pip install \
  --python /root/vllm-bench-env/bin/python \
  --torch-backend=cu128 \
  -r benchmarks/requirements-vllm-cu128.txt
```

Replay the exact saved local workloads and matched KV-cache capacities:

```bash
make reproduce-vllm \
  VLLM_PYTHON=/root/vllm-bench-env/bin/python \
  LOCAL_SUITE=artifacts/reproduction/h100-local/regime-scorecard/suite-TIMESTAMP \
  ARTIFACT_DIR=artifacts/reproduction/h100-vllm
```

The vLLM results are written into the supplied local suite so `summary.csv` contains
the complete matched comparison.

## 3. Phase and intervention evidence

The end-to-end scorecard is the headline comparison. Supporting phase surfaces and
focused causal ablations are separate because they are substantially more expensive:

```bash
make phase-scorecard \
  ARTIFACT_DIR=artifacts/reproduction/h100-experiments

make intervention-suite \
  ARTIFACT_DIR=artifacts/reproduction/h100-experiments

make dispatch-policy-suite \
  ARTIFACT_DIR=artifacts/reproduction/h100-experiments
```

`make reproduce-experiments` runs all three sequentially. Each experiment runner
writes its own timestamped manifest, configuration, raw samples, and summary files.

## Reproducibility contract

- Model, dtype, block size, concurrency, token budget, prompts, output lengths,
  sampling policy, and logical KV capacity are checked before results are merged.
- Synthetic workloads use seed 0 and save their exact token IDs plus a SHA-256
  fingerprint.
- Model loading, cache allocation, and graph capture are reported separately from
  request wall time.
- Correctness and benchmark JSON record the command, Python/package versions, GPU,
  CUDA version, Git commit, and whether the working tree was dirty.
- Timing results require the same GPU class and should use three or more measured
  repetitions. FP16 kernel launch shapes can change accumulation order, so correctness
  uses explicit tolerances and retains generated token IDs for audit.

The checked-in artifacts are indexed in `RESULTS.md` and `results/manifest.json`;
domain-specific retention notes live in the benchmark and experiment result folders.
