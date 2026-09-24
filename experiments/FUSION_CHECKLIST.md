# Fusion ledger

(This is a checklist managing our kernel fusion ideas; the contents of this file are written with AI assistance but all ideas were validated through extensive experimental results)

This is the canonical checklist for model-operation fusion experiments. Consult and
update this file before proposing another fusion. `RESULTS.md` remains the broader
experiment narrative; this ledger answers the narrower question: **have we already
tried this fusion, what happened, and what is the next action?**

Last reconciled: 2026-09-22. Update this ledger with every implementation and result.

## Status contract

- **KEPT**: correctness passed and a relevant full-model/workload measurement justified
  retaining it.
- **REJECTED**: the tested implementation was incorrect or slower at its intended
  regime. This rejects that implementation, not every possible implementation.
- **INCONCLUSIVE**: implemented and measured, but the evidence does not justify shipping
  or deleting it. The exact missing experiment is listed.
- **NOT RUN**: implemented but has not completed its first H100 gate.
- **TODO**: discussed but not implemented.

An isolated microbenchmark win is never enough for **KEPT**. Promotion requires:

1. immediate tensor correctness;
2. full-model logits, sampled tokens, and KV-state correctness where applicable;
3. paired full-model timing using the same attention and graph path;
4. confirmation across the fixed regimes if the optimization is shape-sensitive.

## Decode: completed or implemented

| ID | Fusion | Status | What we know | Canonical implementation / evidence | Next action |
|---|---|---|---|---|---|
| D01 | Pack Q/K/V projections into one GEMM | **KEPT** | Isolated speedup was 2.34× at B8 and 2.35× at B64; packed weights are the current model layout. | `baseline/naive_forward.py`; `experiments/model/benchmark_packed_projection_decode.py`; `comprehensive-decode-profile-20260920/local/fusion.json` | None; keep enabled. |
| D02 | Pack gate/up projections into one GEMM | **KEPT** | Isolated speedup was 1.19× at B8 and 1.14× at B64; packed weights are the current model layout. | Same packed-projection experiment and fusion result as D01. | None; keep enabled. |
| D03 | FA3 paged decode attention | **KEPT** | Replaced the much slower split-K attention path, is graph-safe, and passed the long full-workload logit/history checks. | `custom_kernels/paged_decode_fa3.py`; `fa3-graph-capture/`; `fa3-vs-splitk-b64-l4096-o256/` | Keep as the decode-attention baseline for every new fusion test. |
| D04 | Post-GEMM Q/K RoPE + K/V cache write | **KEPT** | Native postprocess was 1.67–1.68× faster in isolation. On B64/C4096 FA3 it improved pure decode 1.064× and whole workload 1.026×; max logit error was 0.03369 with zero values outside tolerance and identical reference-history argmax. | `custom_kernels/rope_kv_write.py`; `benchmark_fa3_fusion_kernels.py`; `fa3-fusions-b64-l4096-o256/report.json` | Use `qkv-mode=native` as the accepted control. |
| D05 | Residual add + RMSNorm | **KEPT** | The isolated kernel was slower (0.945× B8, 0.935× B64), but the full B64/C4096 FA3 path improved decode 1.020× and wall time 1.008× with passing logits. | `custom_kernels/fused_rms.py`; FA3 fusion kernel and full-engine reports | Keep in the accepted combined-path candidate; reconfirm across the fixed table before making it unconditional. |
| D06 | K-only RoPE + KV write, preserving the production Q path | **KEPT, superseded by D04** | The completed B64/C4096 FA3 ladder passed all-step validation with max logit error 0.03125 and zero values outside tolerance. It improved decode 1.052× and wall time 1.021×, but D04 improved the same boundary by 1.062×/1.026×. | `rope-kv-fusion.json`; `fa3-fusions-b64-c4096-v2/report.json` | Preserve as evidence/control; use D04 in the accepted path. |
| D07 | Custom full QKV GEMM + RoPE + KV write | **REJECTED** | Looked 1.80–1.82× faster in the isolated microbenchmark, but B64/C4096 FA3 decode regressed to 0.891×, whole workload to 0.948×, max logit error reached 0.19531, and 1,350,304 logits were outside tolerance. | `custom_kernels/fused_qkv_rope_cache.py`; FA3 fusion reports | Do not rerun this Triton GEMM. A future tensor-core/CUTLASS epilogue is a different implementation (D13). |
| D08 | cuBLAS QKV GEMM + packed RoPE/KV-write epilogue | **KEPT, equivalent to D04** | The corrected H100 gate passed Q/K/V and measured 2.041× at B8 and 2.046× at B64 in isolation. The B64/C4096 FA3 ladder passed 2.499B logit comparisons with max error 0.03369 and improved decode 1.062× and wall time 1.025×—effectively tied with D04, not additive with it. | `custom_kernels/packed_qkv_rope_cache.py`; `fa3-fusion-kernel-gate-v2/`; `fa3-fusions-b64-c4096-v2/report.json` | Keep one implementation after maintainability testing across the fixed table; do not combine D04 and D08. |
| D09 | SwiGLU activation + multiply | **REJECTED** for decode; **KEPT** conditionally for large packed prefill | At decode B8/B64 it achieved only 0.389×/0.394×. Older packed-row sweeps won above roughly 1,408 rows, reaching 1.65× at 16,384 rows. | `custom_kernels/swiglu.py`; `experiments/mlp/benchmark_swiglu_fusion.py`; fusion results | Do not propose it again for decode. Retain only the measured large-row prefill dispatch. |
| D10 | LM-head projection + exact greedy argmax without materializing `[B,V]` logits | **INCONCLUSIVE** | Already implemented and tested. In the completed FA3 run it passed exact token and full-logit validation and improved a materialized K-step chunk by about 0.5–0.8%. Against the current synchronized materialized-K1 loop, the combined fused-head/K-step path improved roughly 1.4–1.8%. | `custom_kernels/fused_lm_head.py`; `fa3-kstep-head-b8-c4096-v1/`; `fa3-kstep-head-b64-c4096-v1/` | Do not reimplement. Confirm the small effect in production C++ integration before enabling it. |
| D11 | K=1/2/4/8 whole-model graph experiment | **INCONCLUSIVE** (implemented and measured; not production-integrated) | The completed FA3 validation retained every logit and passed exact token/KV checks. Graph composition alone improved only about 0.2–0.55% versus a GPU-chained K1 replay. The production-like comparison—K-step with one final D2H versus K1 with a synchronization each step—improved about 0.5–1.1%; D10 raises the combined result to roughly 1.4–1.8%. | `engine/graph/unrolled_graph_decoder.py`; `benchmark_decode_control_plane.py`; `fa3-kstep-head-b8-c4096-v1/`; `fa3-kstep-head-b64-c4096-v1/` | Implement one C++ chunk commit, first-EOS truncation, and scheduler fallback to K1; do not recapture or rewrite the graph experiment. |

## Decode metadata and fixed-regime specialization

These entries prevent already-shipped graph/scheduler machinery from being proposed as
new cache work. They are control-plane optimizations, not model-operation fusions.

| ID | Optimization | Status | Current implementation / missing work |
|---|---|---|---|
| C01 | Preallocated pinned-host and GPU decode metadata with asynchronous double buffering | **KEPT** | `IterationLoop` already owns reusable host/device buffers and overlaps copies on a copy stream. Do not propose basic metadata preallocation again. |
| C02 | GPU-owned evolving token, position, sequence-length, and slot state | **NOT RUN** | An opt-in stable-cohort path now retains sampled IDs and advances positions, lengths, and slots on device, with automatic fallback on cohort/mixed-work changes. CPU correctness passes; H100 timing and CUDA correctness are pending. The first implementation uses ATen device operations and may need one fused update kernel. |
| C03 | Incremental page-table maintenance | **NOT RUN** | The same opt-in path seeds every page already reserved for each request once and reuses the immutable device table until the cohort changes; the graph adapter skips the redundant D2D table copy while its source address is stable. H100 measurement is pending. |
| C04 | Fixed-regime deterministic/preallocated physical page sequences | **PARTIAL** | Admission already reserves every page needed for prompt plus maximum output, which enables C03. Page IDs still come from the dynamic free list; a deterministic per-slot mapping has not been implemented or benchmarked. |
| C05 | Graph-owned fixed-address FA3 metadata and captured workspace reuse | **KEPT** | `CUDAGraphDecoder` already owns fixed input buffers; FA3 allocations and launches are captured and reused by replay. The missing optimization is C02/C03, not another static workspace wrapper. |
| C06 | Exact B8/B64 decode graph capture | **KEPT** | Exact B8 and B64 graphs are already among the bucket captures, and the targeted experiments explicitly use `decode_buckets=[max_batch_size]`. Context-specific FA3 split/tile specialization remains untested, but merely recapturing the same FA3 launch at another context does not constitute a new optimization. |
| C07 | K=2/4/8 graph capture with one token transfer | **IMPLEMENTED EXPERIMENTALLY** | D11 already captures these graphs, benchmarks one blocking or pinned-async D2H transfer, validates first-EOS scanning, and retains logits in validation. Only production C++ chunk commit, request truncation, and fallback policy remain. |

## Decode: genuinely untested fusion boundaries

| ID | Fusion | Status | Why it might help | Required experiment |
|---|---|---|---|---|
| D12 | Attention output projection + residual add + RMSNorm epilogue | **TODO** | D05 still launches after the output GEMM; a GEMM epilogue could remove another read/write and launch. | Tensor-core-preserving epilogue microbenchmark, then FA3 full model with layerwise/full-logit checks. |
| D13 | Tensor-core QKV projection with RoPE/KV-write epilogue | **TODO** | This is the sound version of D07: retain a competitive tensor-core GEMM while fusing its consumer. | CUTLASS/cuBLASLt-compatible prototype; compare against D04 and D08, not against separate projections. |
| D14 | Gate/up projection + SwiGLU epilogue | **TODO** | D09 only tested a standalone elementwise kernel; it did not eliminate the write/read of both projection outputs. | Tensor-core projection epilogue producing the activated product, followed by full MLP timing. |
| D15 | MLP down projection + residual add + next RMSNorm epilogue | **TODO** | D05 fuses residual and norm but not the preceding GEMM output write/read. | Tensor-core-preserving epilogue and FA3 layer/full-model validation. |

D12–D15 are not equally cheap. They require custom GEMM epilogues, not another small
Triton elementwise kernel. Profile the accepted FA3 path first and implement them in
descending measured boundary cost.

## Prefill-specific boundaries

| ID | Fusion | Status | Current conclusion / next action |
|---|---|---|---|
| P01 | Packed QKV and packed gate/up GEMMs | **KEPT** | Same packed weights as D01/D02; already part of packed prefill. |
| P02 | Masked KV write for bucket-padded piecewise capture | **KEPT** | Required for padded packed-prefill graph correctness; already integrated. |
| P03 | Standalone SwiGLU | **KEPT conditionally** | Use only above the measured packed-row threshold; see D09. |
| P04 | cuBLAS packed QKV + fused RoPE/KV-write consumer | **NOT RUN** | Implemented in piecewise capture with a device-resident live-token mask, zeroed padded Q rows, and no padded KV writes. Run `benchmark_prefill_fusions.py` for the micro gate and `benchmark_integrated_graph.py --prefill-fusions` for all-step logits/KV behavior and paired timing. |
| P05 | Residual add + next RMSNorm in packed prefill | **NOT RUN** | D05 is now wired independently at both residual boundaries in every piecewise-prefill layer, including the final model norm. The same four-arm full-engine gate tests it alone and with P04. |
| P06 | Prefill projection/down-GEMM residual/norm epilogues | **TODO** | P05 remains a standalone consumer kernel. Apply true D12/D15 GEMM epilogues only after profiling proves these boundaries matter in the fixed prefill regimes. |

## Immediate queue

Execute these in order; do not reopen rejected rows unless the implementation strategy
changes materially.

1. Run the P04/P05 micro gate, then the four-arm piecewise full-engine gate at
   the fixed 2048-token bucket. Follow it with the FA3 sequential budget/fusion
   sweep at 8192 tokens; the earlier split-K B64 budget sweep showed a 1.17×
   whole-workload gain from 2048 to 8192. Retain each fusion only on full-engine
   evidence at the selected budget.
2. Reconfirm D04/D05 or D08/D05 across the remaining fixed-table decode regimes.
3. Run the opt-in C02/C03 prototype on H100; measure metadata bytes, state-update GPU
   time, host time, and full-step wall time against the current C01 path. Fuse the
   device update only if the initial ATen operations erase the saved host/copy time.
4. Integrate D11/C07 into the C++ scheduler only after defining chunk admission,
   first-EOS truncation, and K1 fallback semantics.
5. Use the corrected matched FA3/vLLM trace to rank D12–D15; do not use the superseded
   category table that double-counted parent profiler ranges.

## Update rule

Every new run must update one existing row or add a new stable ID. Record the exact
attention backend, graph mode, QKV mode, shape, numerical tolerance, micro result, and
full-model result. Never describe an **INCONCLUSIVE** row as a new idea; state the
specific missing validation instead.
