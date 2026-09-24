# Fixed-regime integration benchmark

This directory contains the current C++ scheduler + CUDA graph + FA3 comparison
against vLLM. The final scorecard uses Qwen2.5-1.5B FP16, one H100, page size 16,
greedy sampling, and the fixed eight-cell workload table.

## Fixed workload

The source of truth is `fixed_regime.py`:

```bash
python3 experiments/integration/benchmark_latest_vs_vllm.py table
```

The eight factorial cells vary batch (8/64), prompt length (256/2048), and output
length (128/256). The two 4096-token rows are optional context probes, not part of
the eight-cell scorecard. Prefill work is capped per scheduler iteration; decode
graphs are captured for the exact supported batch/context buckets.

## Run the comparison

Inspect work before allocating GPU time:

```bash
/root/vllm-bench-env/bin/python \
  experiments/integration/benchmark_latest_vs_vllm.py plan-table \
  --output-dir experiments/results/current-eight-vs-vllm
```

Run the eight-cell smoke suite (resumable):

```bash
/root/vllm-bench-env/bin/python \
  experiments/integration/benchmark_latest_vs_vllm.py run-table \
  --output-dir experiments/results/current-eight-vs-vllm
```

For selection-quality timing, use a fresh directory:

```bash
/root/vllm-bench-env/bin/python \
  experiments/integration/benchmark_latest_vs_vllm.py run-table \
  --trials 3 --samples 3 --warmups 1 --repetitions 3 \
  --output-dir experiments/results/current-eight-vs-vllm-3x3
```

Add `--include-context-probes` only when the extra 4096-token rows are intentional.
Use `analyze-table` to rebuild summaries without launching model work.

## What is measured

Each cell reports pure prefill, pure decode, and mixed-iteration measurements where
the corresponding runner supports that phase. The local ladder includes:

1. eager/C++ control;
2. exact-batch decode graphs;
3. piecewise packed-prefill graphs;
4. FA3/paged decode and regime dispatch;
5. selected fused prefill and scheduler paths.

The vLLM reference uses the same saved workload and matched logical KV capacity.
Model loading, graph capture, correctness checks, and setup allocation are excluded
from steady-state timing.

## Safety checks

- A workload must pass the CPU schedule gate before GPU execution.
- Correctness is checked before timing and same-history checks remain enabled.
- Partial or ambiguous output fails closed; do not overwrite a failed cell manually.
- Resume interrupted runs only through the runner's retry behavior.

## Related artifacts

- Root [`README.md`](../../README.md): final evidence and intervention ranking.
- [`../FUSION_CHECKLIST.md`](../FUSION_CHECKLIST.md): fusion status and missing gates.
- [`../RESULTS.md`](../RESULTS.md): experiment conclusions.
- `fixed_regime.py`: exact shape definitions and dispatch assumptions.
- `benchmark_latest_vs_vllm.py`: table runner, phase comparison, and analysis.
