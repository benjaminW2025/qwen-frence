# v0.1.0 Checkpoint

Qwen2.5-1.5B inference engine on H100. Custom kernels, paged KV, continuous batching, CUDA graphs.

## Architecture decisions

| Decision | Benefit | Measured |
|----------|---------|----------|
| Paged KV cache (BLOCK_SIZE=16) | Dynamic memory, no fragmentation, supports preemption | Enables continuous batching |
| Continuous batching | Amortize model execution across requests | 6.14x over paged-only |
| Bucketed CUDA graphs | Eliminate kernel launch overhead for decode | 1.71x over continuous batching |
| Custom Triton kernels | Replace PyTorch/SDPA with tuned ops | 1.21x over graphs |
| Packed ragged prefill | Concatenate prompts for projection/MLP GEMMs, isolate via offsets in attention | 1.46x over serial admission |
| Variable-length Triton attention | Replace per-request SDPA loop | 1.11x attention speedup |
| Mixed decode+prefill forward | Share projection/MLP work, specialize attention | 1.09x median over separate passes |
| Resumable chunked prefill | Bound prefill latency, interleave with decode | Prevents decode starvation |
| FCFS scheduling | Simple, predictable | Baseline policy |
| Adaptive prefill tile selection | 64×64 for light work, 64×32 for heavy resumed prefill | 6.5% attention improvement |
| SwiGLU fusion (rows>1408) | One kernel instead of separate SiLU + multiply | 1.31x-1.65x on fused shapes |
| RoPE+KV write fusion | Rotate and place KV in one kernel, skip materialization | 20%+ reduction in that region |
| GQA head grouping | Share KV reads across query heads | Rejected: no consistent win |
| Work-budget caps | Limit prefill to reduce decode stalls | Rejected: hurts total goodput |

## Where we are

| Milestone | Output tok/s |
|-----------|--------------|
| PyTorch baseline | 97 |
| + Paged KV | 63 (regression, expected) |
| + Continuous batching | 389 |
| + CUDA graphs | 666 |
| + Custom kernels | 805 |
| vLLM reference | 1,919 |

Gap: **2.06x - 5.26x** behind vLLM depending on regime.

## Scorecard (H100, 2x2x2 sweep)

| Regime | Us | vLLM | Gap |
|--------|---:|-----:|----:|
| lowB/shortP/shortO | 563 tok/s | 2,152 tok/s | 3.8x |
| lowB/shortP/longO | 563 tok/s | 2,343 tok/s | 4.2x |
| lowB/longP/shortO | 156 tok/s | 453 tok/s | 2.9x |
| lowB/longP/longO | 274 tok/s | 1,440 tok/s | 5.3x |
| highB/shortP/shortO | 3,432 tok/s | 9,457 tok/s | 2.8x |
| highB/shortP/longO | 4,232 tok/s | 14,134 tok/s | 3.3x |
| highB/longP/shortO | 267 tok/s | 550 tok/s | 2.1x |
| highB/longP/longO | 786 tok/s | 2,870 tok/s | 3.6x |

Worst case: low concurrency, long context, long output (5.3x gap).
Best case: high concurrency, long context, short output (2.1x gap).

## Time breakdown (mixed iter, fresh prefill)

- MLP: 46%
- Decode attention: 24%
- Paged prefill attention: 13%
- QKV + RoPE + KV write: 11%
- Other: 6%

## What's slow

**1. Decode attention launch**
Only B×12 programs for Qwen's 12 heads. At batch=8 that's 96 programs on 456 SMs. Sequential KV block loop inside each program. This is 80% of CUDA time in decode-heavy regimes.

**2. No CUDA graphs for mixed batches**
Graphs disabled when using regime fusions. Mixed batches run eager with full Python/H2D overhead per iteration.

**3. Tensor reallocation every iteration**
`cache_and_input_setup` rebuilds position, slot_mapping, cu_seqlens tensors from Python lists. 632μs/iter.

**4. Paged KV random access**
Page table lookup with div+mod per KV block. No prefetch. Continuous layout would be faster.

**5. Fusions off by default**
RoPE+KV fusion needs `enable_regime_fusions=True` AND tokens>64. SwiGLU fusion needs rows>1408.

## What we tried

| Intervention | Result |
|--------------|--------|
| Adaptive prefill tiles | +6.5% (near oracle) |
| SwiGLU fusion | +2-5% when active |
| RoPE+KV fusion | +3-4% when active |
| Grouped GQA decode | rejected, no consistent win |
| Prefill budget caps | rejected, hurts goodput |

Regime dispatch overall: 0-18% improvement over static. Not enough.

## Next targets

1. Parallelize decode attention across KV blocks (not just batch×heads)
2. Make fusions CUDA-graph compatible
3. Reuse metadata tensors
4. Enable fusions by default
5. Consider hybrid continuous/paged KV for hot sequences

The gap is executor overhead, not scheduler policy.

## Artifacts

- Burst benchmark: `benchmarks/results/h100-burst-*.json`
- Scorecard: `benchmarks/results/regime-scorecard/suite-20260901T*/summary.csv`
- Profiles: `experiments/results/regime-atlas/`, `experiments/results/profiles/`
- Intervention data: `experiments/results/intervention-suite/`
