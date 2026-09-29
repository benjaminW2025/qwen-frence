# GEMM epilogue fusion (SM90a)

**Uncompiled / unvalidated on Hopper; no performance claim.**

`fused_gemm.cu` runs one decoder layer's non-attention work as four GEMMs with
project-owned epilogues and no RMSNorm kernel. The mainloop is CUTLASS v3.9.2's
warp-specialized cooperative TMA/WGMMA collective (`CollectiveBuilder`),
unmodified; only the epilogues are ours. They read the FP32 accumulators in
registers. The binding picks one of two configurations by row count:

| Configuration | Rows | Tile | Cluster | Scheduler | Why |
|---|---|---|---|---|---|
| Prefill | > 64 | 128x256x64 | 1x2 | persistent | 1x2 along N: a 128-row bucket is one M tile, so a 2x1 cluster would idle half its CTAs |
| Decode | <= 64 | 128x128x64 | 1x1 | stream-K | decode GEMMs stream weights; o_proj/down_proj outputs are only 12 tiles wide, so K is split across SMs |

| GEMM | Epilogue | Replaces |
|---|---|---|
| o_proj | `residual`: fp16(acc + residual), per-row partial sums of squares | cuBLAS store, `residual_add_rms_norm` |
| gate_up | `swiglu`: row scale, silu(g) * u, half-width output | cuBLAS store of [M, 2F], SwiGLU kernel |
| down_proj | `residual` (as o_proj) | cuBLAS store, `residual_add_rms_norm` |
| QKV | `qkv`: row scale, bias, rotate-half RoPE, Q out, K/V into the paged cache | cuBLAS store, `packed_qkv_rope_cache` |

## RMSNorm without a kernel

`rmsnorm(x) @ W.T == r * (x @ (W * gamma).T)` with `r = rsqrt(mean(x^2) + eps)`
per row. So the norm is split three ways:

1. the producer's `residual` epilogue writes each row's sum of squares for its
   256 columns (`[M, hidden / 256]` FP32, fixed order: deterministic), computed
   from the FP16-rounded sum exactly as `residual_add_rms_norm` reads it;
2. gamma is folded into the consumer's weight once at load (`prepare_gate_up`,
   `prepare_qkv`);
3. the consumer's epilogue scales its accumulator rows by `r`, which commutes
   with the matmul, so it costs one multiply on values already in registers.

The consumer GEMM therefore reads the raw residual stream. Layer 0's input has
no producer GEMM; `row_square_partials` computes its partials. Consumers sum
however many partial columns they are given, so prefill (256-column) and decode
(128-column) producers are interchangeable.

The final norm goes into the LM head's input load instead
(`fused_lm_head_argmax(..., norm_weight=, row_partials=)`): the head weight is
tied to the embedding table, so folding gamma into it would need a separate
467 MB copy, while the head's input is only the selected rows.

## Why this shape

- A prior Triton GEMM with a fused QKV/RoPE epilogue (`full_qkv`,
  `experiments/results/fa3-fusions-b64-c4096-v2`) was 5.7% slower end to end
  than cuBLAS plus separate fused kernels: the slower mainloop outweighed the
  saved traffic. Keep the mainloop cuBLAS-class; change only the epilogue.
- At a 2048-token prefill step the unfused MLP alone writes a 73 MB gate/up
  tensor and re-reads it, per layer; the norms add two more full passes.

## Fragment contract

Every epilogue's index math assumes the SM90 accumulator geometry: a thread's
element j is at column `c0 + j % 2 + 8 * (j / 4)` and row `m0 + 8 * ((j / 2) % 2)`,
with `c0 = 2 * (lane % 4)` and quads sharing rows. From it: SwiGLU partners
(block-8 interleaved gate/up) are j and j + 4; RoPE partners (d, d + 64) are
j and j + 32; row partial sums reduce across a quad. `validate_fragment` proves
the geometry for every thread of the mainloop's real `TiledMma` at import.
`experiments/tests/test_fused_gemm.py` proves the algebra and index arithmetic
against float64 models of the unfused path.

Numerics differ slightly from the unfused path (norms and SiLU/RoPE applied to
FP32 accumulators rather than FP16-rounded intermediates). Compare logits with
tolerance, not tokens exactly.

## Gates before integration

1. Build (full CUTLASS v3.9.2 checkout; torch built for the installed nvcc)
   and import: the fragment proof runs at import. Build it in the interpreter
   that runs the engine: for `--attention fa3` that is the vLLM env, whose torch
   is cu130, so it needs a CUDA 13 nvcc (`CUDA_HOME`); torch refuses a major
   version mismatch. CUTLASS 3.9.2 under CUDA 13 is untested here; if it fails
   to compile, the same CollectiveBuilder API is in CUTLASS 4.x.
2. `experiments/prefill/benchmark_mlp_epilogue.py`: SwiGLU alone plus mainloop
   parity (`cutlass_mainloop_vs_cublas` near 1.0, or a win is not the epilogue's).
3. `experiments/prefill/benchmark_layer_epilogues.py`: the whole segment vs the
   engine's current kernels; fused must be no less accurate than baseline
   against an FP32 chain, padded rows must not touch the cache.
4. Integration behind an off-by-default flag, then the all-step logit A/B:
   `experiments/integration/ab_gemm_epilogues.py` (prefill buckets 2048-16384,
   decode B=8/64 at 256/2048 context, teacher-forced; passes when every greedy
   disagreement is a near tie in the control's logits).

## Engine integration

`fused_gemm.prepare_model(model)` builds each layer's folded weights once
(gate/up with the post-attention gamma, QKV with the input gamma; about 1.7 GB
extra, the originals stay for the eager fallback). Then per layer:

    q    = attention_inputs(w[i], x, partials, ...)     # QKV+bias+RoPE+cache
    ...attention...
    x, partials = layer_tail(w[i], attention, x, cfg)  # o+res, gate/up+SwiGLU, down+res

`PiecewisePrefill(enable_fused_gemm_epilogues=True)` makes each captured segment
exactly those GEMMs; `CUDAGraphDecoder(enable_fused_gemm_epilogues=True)` uses
`fused_graph_decode_forward`. The eight-cell harnesses take
`--gemm-epilogues {off,prefill,decode,all}` and verify the flag reached the model.

Decode rows use 128-row tiles; rows past M are zero-filled by TMA and cost
tensor-core time, not memory bandwidth, which is what decode GEMMs spend.
