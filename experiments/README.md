# Experiments

Start with [the final comparison](integration/benchmark_final_8.py). It runs the accepted FA3 configuration across the eight frozen workloads.

| Entry point | Purpose |
|---|---|
| [benchmark_final_8.py](integration/benchmark_final_8.py) | Final burst, mixed, and synchronized phase comparison |
| [model_adapter.py](integration/model_adapter.py) | Shared C++/model integration used by the final runner |
| [divergence_report.py](integration/divergence_report.py) | First-divergence analysis, optionally against an FP32 reference |
| [results/final-accepted-8192/](results/final-accepted-8192/) | Final raw measurements and summary |
| [tests/](tests/) | Harness and integration contracts |

Other scripts retain focused experiments: `decode/`, `prefill/`, `scheduler/`, `model/`, `memory/`, `mlp/`, `attention/`, `profiling/`, and `dispatch/`. Older integration runners are retained because later experiments import their helpers. Their defaults may differ from the final configuration.

See [results](../docs/results.md) for the published comparison and [reproduction](../docs/reproducing.md) for commands.

<details>
<summary>Historical optimizations and rejected candidates</summary>


The final configuration retains C++ scheduling, packed mixed callbacks, exact-batch decode graphs, piecewise packed-prefill graphs, packed QKV/gate-up projections, decode QKV/RoPE/KV-write postprocessing, decode residual-add/RMSNorm, and conditional prefill SwiGLU.

The tables below preserve historical experiment notes. Some source artifacts are absent from the current checkout, so these numbers are not independently reproducible from this page. The final comparison has its own [retained evidence](results/final-accepted-8192/summary.json). Historical speedups use different controls and must not be multiplied together.

| # | Historical intervention | Type | Evidence | Experiment |
|---:|---|---|---|---|
| 1 | FA3 paged decode attention | Attention kernel | 3.892x decode-wall, 1.511x mixed-wall, and 2.244x complete-workload improvement over split-K at B64/C4096. | Paired full-workload ablation |
| 2 | Exact-batch decode graphs | CUDA graph capture | 2.835x whole-workload and 2.895x decode-phase improvement over eager at B8/P256/O128; standalone B8/C2048 replay was 1.981x eager without input copies. | Integrated ablation + microbench |
| 3 | Packed ragged/variable-length prefill | Packing + attention kernel | 1.46x over serial admission and 1.11x attention improvement in the original B8 packed-prefill sweep. | End-to-end + attention microbench |
| 4 | Prefill token budget | Scheduling policy | B64 control output throughput rose from 2056 tok/s at budget 2048 to 2411 tok/s at 8192 (1.173x). | Full prefill/mixed sweep |
| 5 | Packed mixed callback and exact packed buckets | Scheduler + graph capture | Representative B8 mixed target fell from 12.864 ms separate to 8.412 ms packed-exact-qkv (1.529x); all-history correctness passed. | Matched mixed phase |
| 6 | Packed QKV projection | GEMM/projection packing | 2.34x B8 and 2.35x B64 projection microbench speedups; packed weights are used in the final model path. | Projection microbench |
| 7 | Packed gate/up projection | GEMM/projection packing | 1.19x B8 and 1.14x B64 projection microbench speedups. | Projection microbench |
| 8 | QKV RoPE/KV-write postprocess (Triton) | Fusion: QKV → RoPE/KV write | Full FA3 B64/C4096 decode 1.062x; complete wall 1.026x; logit tolerance passed. | Full-model fusion ladder |
| 9 | Conditional SwiGLU | Fusion: activation + multiply | B64 budget-8192 throughput rose 2341→2416 tok/s; large-row kernel sweep reached about 1.65x. | Full prefill sweep + microbench |
| 10 | C++ scheduler / metadata reuse | Scheduler + metadata | Prior engine 759.5→integrated eager C++ 823.5 output tok/s; packed mixed warm wall 30.812→29.044 ms (1.061x), with logits/KV/scheduler checks passing. | Full workload + model callback gate |
| 11 | Resumed-prefill tile dispatch (earlier custom-attention path) | Kernel dispatch policy | Paged attention oracle 6.61% over static; the simple two-way rule captured 6.51%. | Paged-attention sweep |
| 12 | Residual add + RMSNorm | Fusion: residual + normalization | Despite a slower isolated kernel, full FA3 B64/C4096 decode improved 1.019x and wall 1.008x. | Full-model fusion ladder |
| 13 | Piecewise packed-prefill graphs | CUDA graph capture | Mutation-safe graph replay; integrated coverage passed correctness and avoided eager prefill calls in the fixed buckets. | Graph safety + integration gate |

### Other ideas

| Candidate | Direct evidence | Decision |
|---|---|---|
| Native CUDA grouped-decode attention | At B64/C4096, the historical matched warm test (artifact not retained) measured native CUDA at 0.425 ms/layer, old Triton at 0.717 ms/layer, and FA3 at 0.122 ms/layer. Native beat old Triton but was **3.49x slower than FA3**. | Do not replace FA3. This kernel accepts one query token per request and cannot run multi-token prefill as written. |
| K-only RoPE/KV write | Correct within tolerance; decode 1.052x and wall 1.021x, versus 1.062x and 1.026x for the integrated full QKV postprocess. | Superseded by the faster integrated fusion. |
| cuBLAS/packed QKV epilogue | About 2.04x in isolation; passed the full FA3 gate but tied the integrated Triton QKV postprocess. | Equivalent alternative, not an additive fusion. |
| Full mixed forward CUDA graph | Correctness passed for supported fixed buckets. | Dynamic arrivals and packed token counts prevent general coverage. |
| Grouped-GQA head sharing | Grouped variants were slower in the matched attention sweep. | No measured gain over the selected FA3 path. |
| Custom full-QKV fused GEMM | Isolated kernel was about 1.8x faster, but full decode fell to 0.892x, wall to 0.948x, and logits exceeded tolerance. | Rejected for full-model slowdown and numerical error. |
| Mixed attention overlap | Two-stream upper-bound tests did not show a stable end-to-end gain. | Resource contention and join dependencies erased the benefit. |


## Experimental options

CUTLASS GEMM epilogues, shared graph pools, boundary-buffer reuse, GPU-resident evolving metadata, fused greedy output, and K-step capture are not enabled in the final scorecard. No final fused-head A/B result is included. These candidates remain separate from the accepted configuration.


</details>
