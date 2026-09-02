# Token-budget results

The two CSVs are distinct canonical experiments:

- `token-budget-20260830T043216132491Z.csv` sweeps static token budgets across prompt
  length, concurrency, and burst/Poisson arrivals.
- `token-budget-20260831T172914588812Z.csv` sweeps prefix-aware attention-work caps in
  the long-prompt Poisson regime.

The generated JSON duplicated per-request token traces for every sweep cell and added
96 MB without changing any plotted metric, so it is not retained. The exact command,
configuration axes, seed, and output schema are encoded by
`experiments/scheduler/benchmark_token_budget.py`; rerunning it regenerates raw JSON.
