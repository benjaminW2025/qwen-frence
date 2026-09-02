# Stable benchmark harness

This directory is the reproducible scorecard for complete inference backends. Focused
optimization hypotheses, profilers, sweep scripts, and their measurements live in
[`experiments/`](../experiments/README.md).

The harness runs the same token-ID workload across these project milestones:

- `pytorch-baseline`: contiguous KV cache, one request at a time
- `paged-kv`: paged KV cache and cache-aware decode attention
- `continuous-batching`: paged scheduler with eager decode
- `bucketed-cuda-graphs`: continuous batching with bucketed decode graphs
- `custom-kernels`: eager mixed execution with the project's Triton kernels
- `regime-dispatched`: the same eager executor plus measured attention dispatch and
  RoPE/KV-write and SwiGLU fusion policies
- `vllm`: optional offline reference

## End-to-end comparison

```bash
python3 benchmarks/run_benchmarks.py \
  --backends pytorch-baseline,paged-kv,continuous-batching,bucketed-cuda-graphs,custom-kernels,regime-dispatched \
  --num-requests 8 \
  --prompt-lengths 128,512 \
  --output-lengths 32,64 \
  --warmups 1 \
  --repetitions 3
```

Every measured run reuses loaded weights but resets logical scheduler and cache state.
EOS is ignored so every backend produces exactly the requested output length. Results
include TTFT, TPOT, ITL, end-to-end latency, output throughput, request traces, token
IDs, peak GPU memory, and environment metadata.

## Phase scorecard

```bash
python3 benchmarks/run_phase_sweep.py \
  --implementations custom-kernels,regime-dispatched \
  --phases prefill,decode \
  --batch-sizes 1,2,4,8 \
  --prefill-lengths 128,512,1024,2048,4096 \
  --context-lengths 128,512,1024,2048,4096,8192,16384,32736 \
  --decode-steps 32 \
  --warmups 1 \
  --repetitions 3
```

## CPU-only harness tests

```bash
python3 -m unittest discover -s benchmarks/tests -p 'test_*.py'
```

Before collecting GPU numbers, run the consolidated correctness suite:

```bash
python3 correctness/run_correctness.py --checks all
```

## Final H100 scorecard

### Setup confirmation

Run the appropriate preflight before any paid scorecard job. It checks package
versions, CUDA and H100 visibility, the exact Qwen model identity, tokenizer
compatibility, and a one-request end-to-end generation through the real benchmark
adapter. A static failure prevents the model-loading smoke test, so dependency errors
stay short and actionable.

For the local engine environment:

```bash
python3 benchmarks/run_setup_checks.py --suite local
```

Create the external-reference environment from the checked-in pins and confirm it:

```bash
uv venv --python 3.12 /root/vllm-bench-env
uv pip install \
  --python /root/vllm-bench-env/bin/python \
  --torch-backend=cu128 \
  -r benchmarks/requirements-vllm-cu128.txt

/root/vllm-bench-env/bin/python benchmarks/run_setup_checks.py --suite vllm
```

Use `--skip-smoke` for a fast package/model metadata inspection, or `--json-out PATH`
to retain a machine-readable preflight alongside benchmark results. Non-H100 devices
are rejected by default; `--allow-non-h100` permits development smoke tests without
making them valid scorecard measurements.

Run correctness before timing and retain its machine-readable summary:

```bash
python3 correctness/run_correctness.py \
  --checks all \
  --json-out artifacts/reproduction/manual/correctness.json
```

The primary end-to-end comparison is an eight-case 2x2x2 factorial: concurrency
8/64, prompt length 256/8192, and output length 32/256. These axes independently move
the engine between launch-bound, prefill-heavy, decode-heavy, and resumed mixed
execution. The suite retains `custom-kernels` as a static scheduled control; otherwise
the difference between the pre-scheduler `paged-kv` milestone and the final engine
would conflate scheduling, packing, custom kernels, fusion, and dispatch.

Because the historical serial engines can be roughly 50x slower, they are disabled by
default. The competitive static and regime-dispatched engines retain one warmup and
three measured repetitions across all eight cells. Use `--reference-policy anchors`
for one no-warmup measurement on four representative historical anchors, or
`--reference-policy all` only when every historical cell is worth the GPU cost.

Run the local systems first:

```bash
python3 benchmarks/run_regime_scorecard.py --mode local
```

An interrupted run can resume without repeating completed cells:

```bash
LOCAL_SUITE=benchmarks/results/regime-scorecard/suite-YYYYMMDDTHHMMSSffffffZ
python3 benchmarks/run_regime_scorecard.py \
  --mode local \
  --local-suite "$LOCAL_SUITE"
```

The command prints the timestamped suite directory. In the vLLM environment, replay
the exact saved workloads and matched KV-cache limits:

```bash
python3 benchmarks/run_regime_scorecard.py \
  --mode vllm \
  --local-suite "$LOCAL_SUITE"
```

Each case retains raw benchmark JSON/CSV, while the suite writes a manifest and a
cross-regime `summary.csv`. The four requested headline systems are PyTorch,
`paged-kv`, `regime-dispatched`, and vLLM; `custom-kernels` is the causal ablation for
the fitted policy. vLLM is restricted to the shared burst matrix because this
harness's offline adapter does not expose staggered arrivals. Online arrival-rate
tests should be reported separately rather than presented as a matched comparison.

### Phase-isolated diagnostic surfaces

Measure the static and regime-dispatched executors on separate prefill and decode
surfaces. Prefill uses smaller batch sizes to avoid turning the packed MLP activation
into an artificial out-of-memory test; decode includes B=64 to exercise the measured
high-concurrency rule.

```bash
python3 benchmarks/run_phase_sweep.py \
  --implementations custom-kernels,regime-dispatched \
  --phases prefill \
  --batch-sizes 1,2,4,8 \
  --prefill-lengths 128,512,1024,2048,4096,8192 \
  --warmups 2 \
  --repetitions 5 \
  --output-dir artifacts/reproduction/final-scorecard/prefill

python3 benchmarks/run_phase_sweep.py \
  --implementations custom-kernels,regime-dispatched \
  --phases decode \
  --batch-sizes 1,4,8,16,32,64 \
  --context-lengths 128,512,1024,2048,4096,8192,16384 \
  --decode-steps 64 \
  --warmups 2 \
  --repetitions 5 \
  --output-dir artifacts/reproduction/final-scorecard/decode
```

Then collect the end-to-end local control and candidate on one deterministic burst:

```bash
python3 benchmarks/run_benchmarks.py \
  --backends pytorch-baseline,custom-kernels,regime-dispatched \
  --workload-name final-mixed-burst \
  --num-requests 32 \
  --prompt-lengths 128,2048,8192 \
  --output-lengths 32,128 \
  --max-running 32 \
  --max-num-batched-tokens 8192 \
  --warmups 1 \
  --repetitions 3 \
  --workload-out artifacts/reproduction/final-scorecard/workload.json \
  --output-dir artifacts/reproduction/final-scorecard/local
```

If vLLM lives in a separate environment, replay the saved local result there. Replace
`LOCAL_RESULT.json` with the JSON emitted by the preceding command; `--compare-with`
imports its exact workload and refuses mismatched comparison settings.

```bash
python3 benchmarks/run_benchmarks.py \
  --backends vllm \
  --compare-with LOCAL_RESULT.json \
  --max-running 32 \
  --max-num-batched-tokens 8192 \
  --warmups 1 \
  --repetitions 3 \
  --strict-backends \
  --output-dir artifacts/reproduction/final-scorecard/vllm
```
