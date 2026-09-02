# Engine layout

The runtime is organized by responsibility:

- `scheduler/`: request lifecycle, immutable per-iteration plans, admission,
  token-budget allocation, optional prefix-aware attention-work allocation, and
  execution dispatch.
- `model_runner/`: packed model forwards. `ragged_prefill.py` handles prefill-only
  batches; `mixed_batch.py` shares projection/MLP work across decode and prefill while
  dispatching their specialized attention kernels.
- `kvcache/`: paged KV storage, cache-aware attention operations, and the serial paged
  reference engine.
- `graph/`: legacy decode-only CUDA-graph experiments. These are not used by the eager
  mixed scheduler baseline.

Standalone Triton sources live in `../custom_kernels/` and are loaded lazily through
`../baseline/kernel_dispatch.py`.

The primary scheduler experiment is deliberately fully eager. This keeps executor
coverage constant while comparing fixed and adaptive prefill-token budgets.

The first cost-aware policy keeps the global token ceiling and optionally adds a
prefill attention ceiling measured in causal query-key pairs. For a chunk of `q`
tokens after cached prefix `p`, the planner charges `q*p + q*(q+1)/2` and shortens the
FCFS chunk until both ceilings fit when active decodes exist. Prefill-only iterations
ignore the attention ceiling and use the full token budget. This is a static control
surface for the later decode-SLO feedback controller, not a latency model by itself.
