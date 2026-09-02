# Inference correctness suite

The suite runner lives here; individual executable checks are grouped under `checks/`:

```bash
python3 correctness/run_correctness.py
```

Each sweep executes in a fresh Python process. This releases its models, KV caches,
CUDA graphs, and allocator state before the next sweep starts. The runner continues
after failures so the final table shows the state of the complete suite, then exits
nonzero if any check failed.

## Included sweeps

| Check | Coverage |
|---|---|
| `baseline-vs-hf` | Custom PyTorch Qwen forward versus Hugging Face logits and top-1 |
| `paged-attention` | Triton paged decode attention across ragged lengths and block boundaries |
| `paged-vs-baseline` | Teacher-forced paged engine versus contiguous baseline |
| `cuda-graph` | Legacy captured single-sequence decode versus eager and baseline |
| `scheduler` | Continuous batching, admission, constrained pools, and preemption |
| `ragged-prefill` | Variable-length and resumable attention, mixed decode/prefill execution, global token-budget invariants, packed logits/top-1, logical KV parity, block boundaries, slot isolation, and batched admission |
| `bucketed-graphs` | Legacy bucket selection, input padding, replay parity, and lifecycle transitions |
| `custom-kernels` | Triton RMSNorm/Qwen RoPE and full-model prefill/decode parity |

Run a subset:

```bash
python3 correctness/run_correctness.py \
  --checks paged-attention,scheduler,bucketed-graphs
```

Run the primary fully eager scheduler baseline without the legacy CUDA-graph
ablations:

```bash
python3 correctness/run_correctness.py --checks baseline
```

Stop at the first failure:

```bash
python3 correctness/run_correctness.py --fail-fast
```

Save a machine-readable summary:

```bash
python3 correctness/run_correctness.py \
  --json-out artifacts/reproduction/manual/correctness.json
```

List available checks without importing PyTorch or loading a model:

```bash
python3 correctness/run_correctness.py --list
```

The teacher-forced checks are the strict numerical correctness gates. Free-running
scheduler comparisons retain matched-prefix allowances for benign fp16 greedy near-ties.
