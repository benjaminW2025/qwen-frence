# Profile-driven intervention record

This document preserves the hypotheses and their original ship/reject criteria. The
experiments are complete; final decisions are summarized in `RESULTS.md`.

## Evidence and objective

The H100 profiles show that mixed iterations are dominated by model execution rather
than scheduler arithmetic. In the representative fresh-prefix trace, MLP work consumes
about 46% of CUDA time, decode attention about 24%, paged prefill attention about 13%,
and QKV/RoPE/KV movement roughly another 11%. Resuming a 2K prefill chunk after a 4K
prefix increases median iteration latency from 45.7 ms to 57.3 ms. The work-budget
sweep reduced individual tail events but did not improve SLO-compliant output goodput.

The primary score is output tokens per second subject to an ITL target. Prefill token
throughput is diagnostic. Every intervention must also report TTFT, TPOT/ITL p95 and
p99, peak memory, and exact output-token agreement with its baseline.

## Failure-signature rubric

Classify each measured regime before choosing an intervention. The relevant signatures
are poor scaling with batch/concurrency or context length, device idle gaps, occupancy
collapse, an unexpected memory-bandwidth ceiling, synchronization between independent
work, reduced CUDA-graph effectiveness, and a gap between isolated-kernel speed and
end-to-end throughput. These are diagnoses, not assumptions: latency sweeps establish
scaling, Nsight establishes gaps/occupancy/bandwidth/synchronization, and matched
microkernel plus end-to-end runs establish whether a local win matters to the executor.

An intervention only needs to test the signatures implicated by its hypothesis. For
example, grouped-GQA decode targets context/concurrency scaling and KV bandwidth; it
does not need a CUDA-graph claim. This keeps each ablation falsifiable and prevents a
single aggregate throughput number from hiding a regime-specific regression.

## Selected interventions

### 0. Share paged KV reads across grouped-query heads

Status: the first implementation falsified the head-sharing hypothesis. Across the
30-shape H100 sweep, the winning candidate almost always kept one query head per
program. Grouping two heads produced only isolated small wins at B=64; grouping three
or six was substantially slower. Do not integrate grouped head sharing as a production
optimization from this result.

Hypothesis: the current decode grid launches one program per query head, causing all
six Qwen GQA heads in a group to traverse the same K/V pages independently. Grouping
multiple query heads per program should reduce redundant KV traffic in the long-context
decode regime.

Experiment: sweep 1/2/3/6 query heads per program and 4/8 warps across batches
1/8/32/64 and contexts 128/2K/8K/16K. Compare against the unchanged production kernel,
check numerical parity, and report both latency and effective logical KV bandwidth.
Then replay winning configurations end to end to measure how much of the microkernel
gain reaches output throughput and ITL.

The same experiment exposed a separate result: the candidate kernel with one head per
program and a batch-dependent 4/8-warp choice beat production on most nontrivial
contexts. This is not evidence for KV sharing because both variants use one head per
program, and the candidate is not source-identical to production. Before integration,
run a causal ablation with matched launch settings:

1. production kernel with its default launch,
2. the exact production kernel with 4/8 warps and a stage sweep,
3. the candidate single-head kernel with those identical launch settings, and
4. candidate grouped variants with those identical settings.

On B=64, C=16K, use Nsight Compute to compare DRAM bytes, L2 hit rate,
registers/thread, local loads/stores, achieved occupancy, and active warps. This tells
us whether the measured win is launch tuning, compiler/layout behavior, or something
else. Keep the measured dispatch behind an experimental flag until this attribution
and an end-to-end replay pass.

Ship criterion: move the profiled long-context decode saturation knee or improve the
B32/B64, C8K/C16K kernel shapes by at least 10%, followed by at least 5% improvement in
a decode-heavy end-to-end regime without a short-context regression.

### 1. Specialize resumed paged-prefill attention

Hypothesis: the production 64x32 tile selected for fresh contiguous prefill is not
optimal when a short query chunk traverses a long paged prefix.

Experiment: sweep query/key tiles and warp/stage configurations over query lengths
64 through 2048 and prefixes 0 through 16K. Compare the paged Triton kernel with the
paged SDPA reference, production default, best static configuration, and per-shape
oracle.

Ship criterion: at least 8% median attention-kernel improvement over the representative
resumed shapes, no material fresh-prefill regression, and correctness within existing
FP16 tolerances.

Adaptive-selection criterion: the per-shape oracle must beat the best single static
configuration by at least 5%, and winning configurations must form repeatable regions
expressible using batch size, query length, and total KV context. Otherwise retain one
static configuration.

### 2. Fuse SwiGLU's gate activation and elementwise multiply

Status: the isolated H100 sweep passed correctness for all eight launch
configurations. Fusion loses below 520 packed rows, then wins by 1.31x at 2048 rows,
1.49x at 4128, 1.59x at 8200, and 1.65x at 16384. A single 512-element, 4-warp
configuration stays within 0.25% of the oracle at every winning atlas-sized shape.
This supports a simple packed-row threshold, not an adaptive tile selector.

Hypothesis: separate SiLU and multiply kernels account for meaningful memory traffic
inside the largest profiled region.

Experiment: replace `silu(gate) * up` with one Triton operation, keeping both GEMMs and
the down projection unchanged so the ablation isolates elementwise fusion.

Ship criterion: at least 3% end-to-end mixed-iteration improvement with no output-token
change. A second ablation may pack gate/up projection weights into one GEMM if fusion
alone is launch-bound.

The microkernel savings projected over 28 layers are 1.24 ms for the 4128-row balanced
shape and 2.66 ms for the 8200-row prefill-heavy shape. Relative to the existing atlas,
that is an upper-bound estimate of 3.0% for balanced fresh, 2.3% for balanced resumed,
1.5% for prefill-heavy, and 1.1% for dual-heavy. Validate end to end behind a policy
that retains PyTorch for small packed batches; do not enable the fused path globally.

### 3. Fuse RoPE with paged KV placement

Hypothesis: materializing rotated K, making contiguous transposes, and issuing two
`index_copy_` operations wastes bandwidth and launches on every layer.

Experiment: one kernel rotates Q/K, writes rotated K and unrotated V directly to their
paged slots, and emits only the Q tensor required by attention.

Harness status: an isolated candidate now compares the exact current sequence of two
RoPE launches, contiguous materialization, and two `index_copy_` calls against one
direct-write kernel over packed row counts 64 through 16K. It checks rotated Q, logical
K placement, and exact V placement independently before timing.

Ship criterion: at least 20% reduction in the combined RoPE-plus-KV-write region and at
least 3% mixed-iteration improvement without increasing persistent memory.

### 4. Overlap decode and prefill attention within each layer

Hypothesis: paged decode attention is comparatively bandwidth-heavy while prefill
attention has more matrix compute, so independent CUDA streams may overlap part of the
two kernels before their outputs are joined.

Experiment: compare the current sequential launches with a two-stream implementation
using explicit events at the QKV-ready and output-join boundaries. Measure fresh and
resumed prefills across low/high decode counts.

Harness status: the attention-only upper-bound experiment covers all eight regime-atlas
corners and reports each kernel alone, their sequential sum, two-stream latency, and
the fraction of the smaller kernel hidden. This rejects the idea cheaply if resource
contention prevents concurrency before the executor is modified.

Ship criterion: at least 5% mixed-iteration improvement on two or more realistic shapes,
no decode-only/prefill-only regression, and no unstable latency tail. Reject if resource
contention merely slows both kernels.

### 5. Reuse iteration metadata and remove host synchronizations

Hypothesis: Python list construction, pageable H2D metadata copies, allocations, and
per-token host reads inflate wall time beyond CUDA event time, especially for smaller
batches.

Experiment: reuse capacity-bounded device buffers for positions, slot mappings,
sequence offsets, context lengths, and block tables; update them with nonblocking
copies from pinned host storage. Separately count allocations and synchronization calls.

Harness status: sampling and metadata are separated. One test measures per-row versus
batched argmax plus host transfer across decode batches 1 through 128. The other
decomposes current Python-list-to-device construction, prebuilt pageable tensors, and
pinned host plus reused device buffers at every atlas corner. The pinned result is an
upper bound because host-value generation occurs outside its timed region.

Ship criterion: at least 10% wall-time improvement for small/mid-size mixed iterations
or a demonstrable reduction in CPU submission time without GPU regression.

## Decision protocol

Each intervention is an independent baseline/variant comparison on the same loaded
model and deterministic workload. Run warmups before randomized alternating baseline
and variant repetitions. Keep raw samples and environment metadata. Do not stack an
intervention into the production path until it passes correctness and its ship
criterion; rejected hypotheses remain documented results rather than disappearing.

Order: attribute the single-head decode result, then continue regime diagnosis with
SwiGLU fusion, RoPE/KV fusion, attention overlap, and metadata reuse. Resumed-attention
tiling has already been swept. Re-profile after each accepted change because the next
bottleneck may move.
