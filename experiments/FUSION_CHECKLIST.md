# Fusion ledger

This is the canonical checklist for model-operation fusion experiments. Consult and
update this file before proposing another fusion. `RESULTS.md` remains the broader
experiment narrative; this ledger answers the narrower question: **have we already
tried this fusion, what happened, and what is the next action?**

Last reconciled: 2026-09-21, through commit `99da47d` and the checked-in H100 results.

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
| D06 | K-only RoPE + KV write, preserving the production Q path | **INCONCLUSIVE** | B8 full model improved 1.063× and passed. B64 improved 1.107× with exact sampled tokens, but 1/9,723,904 logits missed the effective tolerance: max error 0.11719 versus 0.1 allowed. It is now wired into the FA3 ladder but that extended run has not landed. | `rope-kv-fusion.json`; `piecewise_fa3_k_rope_kv` in `benchmark_integrated_graph.py` | Run the FA3 full-workload ladder; keep timing even if the numerical flag fails. |
| D07 | Custom full QKV GEMM + RoPE + KV write | **REJECTED** | Looked 1.80–1.82× faster in the isolated microbenchmark, but B64/C4096 FA3 decode regressed to 0.891×, whole workload to 0.948×, max logit error reached 0.19531, and 1,350,304 logits were outside tolerance. | `custom_kernels/fused_qkv_rope_cache.py`; FA3 fusion reports | Do not rerun this Triton GEMM. A future tensor-core/CUTLASS epilogue is a different implementation (D13). |
| D08 | cuBLAS QKV GEMM + packed RoPE/KV-write epilogue | **NOT RUN** | Implemented to preserve the proven cuBLAS projection and eliminate split/view/RoPE/cache-write traffic. No checked-in H100 result yet. | `custom_kernels/packed_qkv_rope_cache.py`; `piecewise_fa3_packed_epilogue`; commit `3ff667f` | Run the cheap kernel gate, then the FA3 full-engine arm only if correctness passes. |
| D09 | SwiGLU activation + multiply | **REJECTED** for decode; **KEPT** conditionally for large packed prefill | At decode B8/B64 it achieved only 0.389×/0.394×. Older packed-row sweeps won above roughly 1,408 rows, reaching 1.65× at 16,384 rows. | `custom_kernels/swiglu.py`; `experiments/mlp/benchmark_swiglu_fusion.py`; fusion results | Do not propose it again for decode. Retain only the measured large-row prefill dispatch. |
| D10 | LM-head projection + exact greedy argmax without materializing `[B,V]` logits | **INCONCLUSIVE** | Already implemented and tested. Isolated graph-warm speedup was 1.055× at B8 and 1.186× at B64. In the complete 28-layer split-K graph it was only 1.009×/1.001×. Inside old K-step graphs it ranged from 0.995× to 1.009× versus the materialized head, with exact tokens and KV state. | `custom_kernels/fused_lm_head.py`; `output-head.json`; `output-head-full-model.json`; `unrolled-decode.json` | **Rerun the existing fused head inside FA3 K=1/2/4/8. Do not reimplement it.** |
| D11 | K=2/4/8 whole-model graph unrolling | **INCONCLUSIVE** (graph composition, not a kernel fusion) | The original split-K run retained every validation logit and passed logit/token/KV checks, but produced roughly 0.995–1.011×. A newer FA3 control-plane runner exists, but currently checks tokens/KV rather than full logits and does not include D10. | `engine/graph/unrolled_graph_decoder.py`; old `unrolled-decode.json`; `benchmark_decode_control_plane.py` | Add the existing validation-logit and fused-head arms to the FA3 runner, then rerun. |

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
| P04 | Fresh-prefill QKV projection + RoPE + KV write | **TODO** | This remains a real missing fusion; it must support packed token positions and masked padded rows. |
| P05 | Prefill projection/down-GEMM residual/norm epilogues | **TODO** | Apply D12/D15 only after profiling proves these boundaries matter in the fixed prefill regimes. |

## Immediate queue

Execute these in order; do not reopen rejected rows unless the implementation strategy
changes materially.

1. **D08 cheap gate**, followed by its FA3 full-engine arm if correct.
2. **D06 FA3 full-workload validation** (already wired).
3. **D10 + D11 FA3 rerun**: materialized versus fused output head at K=1/2/4/8,
   retaining every logit only in the correctness capture.
4. Run the current matched FA3/vLLM component trace and use its boundary costs to rank
   D12, D13, D14, and D15.

## Update rule

Every new run must update one existing row or add a new stable ID. Record the exact
attention backend, graph mode, QKV mode, shape, numerical tolerance, micro result, and
full-model result. Never describe an **INCONCLUSIVE** row as a new idea; state the
specific missing validation instead.
