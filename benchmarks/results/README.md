# Benchmark artifacts

The canonical final H100 factorial scorecard is
`regime-scorecard/suite-20260901T180925475938Z/`. It contains all eight matched local
and vLLM cells, both manifests, and the aggregate `summary.csv`.

`h100-burst-20260824T224748374089Z` is the canonical engine-milestone comparison;
`h100-poisson-20260824T195643307034Z` is the retained continuous-arrival workload.
Failed, unmatched, superseded, and incomplete attempts are intentionally not retained
here; the runner can resume an interrupted timestamped suite in place.

Large JSON files contain raw request traces and generated token IDs. CSV summaries are
the preferred inputs for plots and tables; JSON remains the audit trail.
