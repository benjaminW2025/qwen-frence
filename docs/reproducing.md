# Reproducing the final comparison

Run from the repository root on Linux with an NVIDIA H100 and a compatible CUDA driver/toolkit. The final engine uses vLLM's bundled FA3, so run both backends in the same vLLM 0.30.0 environment.

The retained reference records Python 3.12.3, Torch 2.13.0, Triton 3.7.1, Transformers 5.18.0, and CUDA 13.0. The root `requirements.txt` describes the earlier local-engine environment; it is not the final FA3 environment lock. The [reference requirements](../benchmarks/requirements-vllm-current.txt) pin vLLM only; a complete environment lock is not yet provided.

## Prepare

With that environment activated:

```bash
make cpp-scheduler-build PYTHON=python
python experiments/integration/benchmark_latest_vs_vllm.py stage-model-cache
```

The model helper downloads the pinned Qwen snapshot. The C++ extension must be rebuilt when its sources or the installed Torch version change.

## Compare

Use fresh output directories. Run the reference first, then the local engine against its saved outputs:

```bash
make final-reference PYTHON=python FINAL_REFERENCE=artifacts/final-reference
make final-local PYTHON=python FINAL_REFERENCE=artifacts/final-reference FINAL_OUTPUT=artifacts/final-local
```

Both commands use the final runner, one warmup, three repetitions, and a local prefill budget of 8192. vLLM uses its effective default budget. The local command validates and reuses the reference without launching vLLM. The final runner checks prerequisites before local measurements.

To inspect or customize the underlying command:

```bash
python experiments/integration/benchmark_final_8.py --help
```

The resulting `summary.json` contains the burst, mixed, and phase comparisons. Generated runs belong under `artifacts/`; the committed [final evidence](../experiments/results/final-accepted-8192/) stays unchanged.

## Checks

```bash
make unit PYTHON=python
make cpp-scheduler-test PYTHON=python
make correctness PYTHON=python
```

`unit` includes tests that need the built C++ extension and Torch. GPU-specific tests may skip without CUDA. The C++ suite's Python-scheduler comparison also imports Triton. `correctness` runs the broader numerical suite, including earlier engine paths; it is not a single certification of every final-engine option.

`make help` lists current and historical workflows. The older `scorecard-*` and `reproduce-*` targets retain the earlier custom-kernel/regime-dispatch comparison; use `final-*` for the README results.

[Results](results.md) · [Project overview](../README.md)
