# Experiment log

This log separates measured conclusions from planned work. Raw CSV/JSON and profiler
traces are versioned under `results/`; commands and measurement contracts are in the
main experiment README.

## Completed

| Experiment | Result | Decision |
|---|---|---|
| Packed ragged prefill | At B=8, ramped prompts through 4096 tokens: 1.46x total speedup over serial admission and 1.11x attention speedup over packed per-request SDPA | Ship packed projection/MLP and variable-length attention |
| Fresh packed-attention tiles | 64x32 won 42/56 shapes in the broad frontier; 64x64 won 13 and 128x64 won 1 | Keep 64x32 as the fresh-prefill default |
| Resumed paged-prefill tiles | The per-shape oracle was 6.61% faster than static 64x32 across 108 shapes. Choosing 64x64 below 2048 total packed query tokens and 64x32 otherwise achieved 6.51%, only 0.091% behind the oracle | Preserve an optional two-way dispatch for end-to-end validation; it helps short resumed chunks but not the previously profiled B=2, Q=2048 case |
| Resumable prefill correctness | Arbitrary chunks preserve prompt progress, RoPE/KV positions, cache isolation, final logits, and admission behavior within explicit FP16 tolerances | Ship resumable state and chunked execution |
| Mixed execution surface | A 32-decode, 2x2048-prefill iteration measured 45.7 ms with fresh prefixes and 57.3 ms after 4096 cached prefix tokens | Target paged resumed attention; do not model work using query-token count alone |
| Prefix-aware work cap | Uncapped scheduling had the highest SLO-compliant output goodput at every tested 40/50/75/100/150 ms threshold | Do not build the adaptive controller as the primary optimization |
| Grouped-GQA paged decode | All 240 candidate configurations passed the ragged manual-reference preflight. Sharing 3 or 6 query heads per program was slower; the best configuration at nearly every shape used one head per program. A separate single-head candidate with 4/8-warp dispatch improved most contexts above 128 tokens, including 1.17x at B=64, C=8K and 1.20x at B=64, C=16K | Reject this grouped-head implementation. Preserve the single-head result as an unattributed experimental lead; isolate launch settings from kernel-layout effects before integration |
| Fused SwiGLU activation | The fused elementwise path regressed packed rows through 520, then improved 1.31x/1.49x/1.59x/1.65x at 2048/4128/8200/16384 rows. A 512-element, 4-warp configuration was within 0.25% of the oracle at all winning atlas-sized shapes. Projected whole-iteration savings range from 1.1% to 3.0% across the dense atlas regimes | Preserve as a thresholded candidate and run one end-to-end balanced/prefill-heavy validation. Reject unconditional dispatch and a tile scheduler |
| Final regime scorecard | Against the static custom engine, the fitted backend ranged from small short-context regressions to a 17.96% long-context gain. vLLM remained 2.06x–5.26x faster across the eight matched cells; the gap was smallest for high-concurrency, prefill-heavy work and largest for low-concurrency long generation | Scope the scheduler/kernel study honestly. Preserve the measured long-context specialization; identify whole-executor decode launch, fusion, graph, and host overhead as future work rather than adding more scheduler policy |

The work-cap result is not “nothing worked.” It established that reducing the severity
of one mixed iteration can increase the number of affected decode intervals. Any future
latency controller therefore needs a production workload whose utility function values
hard tail protection enough to pay the measured throughput and TTFT cost.

## Final integration conclusion

The resumed paged-prefill result is available behind the `adaptive` tile-policy option
for later end-to-end validation. The grouped-decode hypothesis is closed for the first
implementation, while its unexpected single-head speedup is retained as a causal
ablation rather than mislabeled as GQA sharing. Production dispatch remains unchanged.

The dense regime-dispatch holdout selected four compact policies: a batch/context
decode-attention rule, fused SwiGLU above 1408 packed rows, two-warp fused RoPE/KV
placement from 64 packed tokens, and a 64x64/64x32 paged-prefill tile rule. They are
integrated behind the separate `regime-dispatched` benchmark backend while
`custom-kernels` remains unchanged. Their microbenchmark evidence is accepted and the
final engine-level H100 replay is retained under `benchmarks/results/regime-scorecard/`.

The first engine replay exposed an integration boundary absent from the packed-input
RoPE/KV microbenchmark: converting native decode `(B,H,1,D)` Q/K/V into packed
`(1,H,B,D)` tensors required three materializing copies per layer and reduced the
B=64, prompt-256, output-32 regime to about 90% of the static engine. Fused RoPE/KV
placement therefore remains enabled only where mixed/resumed execution already owns
the packed layout; native decode retains separate RoPE and cache placement.

Removing those native-layout copies recovered the B=64, prompt-256, output-256
regime from 0.897x to 0.958x of the static engine, but did not make short-context
adaptive dispatch a global win. The final executor policy is therefore deliberately
conservative: contexts below 1024 tokens resolve once to the exact static production
path before the layer loop, while contexts at or above 1024 retain the measured
adaptive candidate. This defines the intervention as a long-context specialization
with a safe short-context fallback, not as a universal kernel replacement.

The final comparison changes the project conclusion: additional token-budget or
dispatch complexity is not the highest-return work. The largest remaining gap is the
decode executor, where vLLM benefits from packed projections, whole-layer compilation,
CUDA graphs, persistent metadata, and reduced synchronization. Those are explicitly
future work, not unfinished claims in the present study.
