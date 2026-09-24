# Experiments

Focused performance experiments, profilers, and scheduler/control-plane checks.
The stable backend harness is in [`benchmarks/`](../benchmarks/); correctness gates
are in [`correctness/`](../correctness/).

## Start here

- [`FUSION_CHECKLIST.md`](FUSION_CHECKLIST.md): current fusion status and production gates.
- [`RESULTS.md`](RESULTS.md): concise experiment conclusions.
- [`INTERVENTIONS.md`](INTERVENTIONS.md): profile-driven hypotheses and decisions.
- [`integration/README.md`](integration/README.md): current fixed-regime local/vLLM table.
- [`decode/STAGE_POLICY.md`](decode/STAGE_POLICY.md): decode-stage policy protocol.

## Canonical experiments

Run the complete intervention suite:

```bash
python3 experiments/run_intervention_suite.py
```

Run the dispatch-policy sweeps:

```bash
python3 experiments/run_dispatch_policy_experiment.py
```

Run the decode grouping/split-K/stage sweep:

```bash
python3 experiments/decode/benchmark_decode_joint_sweep.py
```

Run the fixed-regime C++/graph/vLLM comparison:

```bash
/root/vllm-bench-env/bin/python \
  experiments/integration/benchmark_latest_vs_vllm.py run-table \
  --output-dir experiments/results/current-eight-vs-vllm
```

Inspect a plan before using a GPU:

```bash
python3 experiments/integration/benchmark_latest_vs_vllm.py table
python3 experiments/integration/benchmark_latest_vs_vllm.py plan-table \
  --output-dir experiments/results/plan-check
```

## Measurement rules

- Use the fixed shape table for final comparisons; do not tune shapes from results.
- Save each run in a fresh output directory or resume only through the runner.
- Run correctness before timing and retain raw JSON/CSV artifacts.
- Compare local and vLLM on identical tokenized workloads, dtype, page size,
  concurrency, and KV capacity.
- Treat microbenchmarks as evidence for dispatch decisions, not as end-to-end wins.

## Other entry points

- `experiments/decode/`: attention, split-K, grouping, stage, and graph studies.
- `experiments/integration/`: scheduler, graph, prefill, mixed, and vLLM studies.
- `experiments/model/`: projection and model-operation profiles.
- `experiments/tests/`: CPU planning and correctness tests.
- `experiments/results/`: raw artifacts; summaries should be linked from the root README.

Historical detail is intentionally kept in the focused protocol/evidence files rather
than duplicated in this index.
