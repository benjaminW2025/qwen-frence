# Experiment artifacts

This directory retains H100 evidence for focused hypotheses. The curated conclusions
and accept/reject decisions live in `../RESULTS.md`. Raw JSON is retained when it adds
samples or profiler structure not represented in CSV; exceptionally large duplicated
request traces are omitted and explicitly documented in their result folder.

The primary collections are:

- `dispatch-policy-suite/`: dense crossover measurements and fitted policies.
- `intervention-suite/`: isolated decode, sampling, fusion, metadata, and overlap tests.
- `regime-atlas/`: operator-level profiles across representative execution regimes.
- `token-budget/` and `work-budget-latency/`: scheduler-budget experiments.
- `packed-attention-tiles/`, `paged-prefill-tiles/`, and `ragged-prefill/`: prefill kernels.
- `profiles/`: the retained Nsight Systems summary for the representative dual-heavy
  workload; superseded framework-profiler traces are captured more comprehensively by
  `regime-atlas/`.

Only the newest complete run is retained when a later sweep strictly supersedes an
earlier grid. Distinct scheduler questions and before/after execution profiles remain
separate even when they use the same harness.
