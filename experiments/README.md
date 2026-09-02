# Optimization experiments

This directory contains focused performance hypotheses, ablations, profilers, their
CPU control-plane tests, and the measurements collected from them. The stable
cross-backend benchmark harness remains in `benchmarks/`; numerical and state-machine
gates remain in `correctness/`.

## One-shot H100 intervention suite

Run every remaining isolated hypothesis plus the updated launch-bound profile with one
command:

```bash
python3 experiments/run_intervention_suite.py
```

The runner continues after an individual failure, gives every experiment its own
output directory, and writes a top-level manifest under
`experiments/results/intervention-suite/`. The suite contains:

- exact production launch tuning versus the single-head candidate and grouped-GQA,
- serial versus batched decode sampling,
- the existing fused SwiGLU frontier,
- current metadata construction versus pageable and pinned/reused staging,
- separate RoPE/cache placement versus a fused direct-write candidate,
- sequential versus two-stream decode/prefill attention, and
- a real launch-bound mixed-iteration profile after batched sampling.

Each microbenchmark has its own correctness preflight and raw samples. The attention
overlap result is explicitly an attention-only upper bound; the metadata pinned-buffer
result excludes host-value generation and is likewise labeled as an upper bound. Use
`--experiments name1,name2` to rerun a subset or `--dry-run` to print every command.

## Regime-dispatch policy experiment

The intervention suite identifies promising regions; the dispatch-policy suite measures
the crossover boundaries densely and converts them into small host-side rules:

```bash
python3 experiments/run_dispatch_policy_experiment.py
```

It runs four matched sweeps:

- decode batch size versus KV context for production and single-head candidates,
- packed SwiGLU rows around the baseline/fusion crossover,
- packed RoPE/KV rows across the warp-count frontier, and
- paged-prefill batch/query/prefix shapes around the tile crossover.

The final stage fits latency-aware decision trees with depth at most two. Every leaf is
guarded against more than a 2% regression on any training shape. A deterministic 20%
holdout is excluded from fitting and reports aggregate, p95, and worst-case oracle
regret. Winner margins distinguish stable regions from shapes where launch noise makes
the nominal fastest action fragile. Outputs live under
`experiments/results/dispatch-policy-suite/`.

Run one frontier while iterating:

```bash
python3 experiments/run_dispatch_policy_experiment.py --sweeps decode
```

The fitter can also analyze existing result files without rerunning CUDA work:

```bash
python3 experiments/dispatch/fit_dispatch_policies.py \
  --decode-json DECODE_RESULT.json \
  --swiglu-json SWIGLU_RESULT.json \
  --rope-kv-json ROPE_KV_RESULT.json \
  --paged-prefill-json PAGED_PREFILL_RESULT.json
```

These fitted rules are evidence, not automatically installed production settings. We
first inspect held-out regret and boundary stability, then encode the accepted rules and
run matched end-to-end static/rule/oracle replays.

An experiment belongs here when it changes one factor to explain or improve the
system. A benchmark belongs in `benchmarks/` when it is a stable scorecard used to
compare complete implementations.

This harness runs the same token-ID workload across the project milestones:

- `pytorch-baseline`: contiguous KV cache, one request at a time
- `paged-kv`: paged KV cache and paged decode attention, one request at a time
- `continuous-batching`: paged scheduler with eager decode
- `bucketed-cuda-graphs`: the same scheduler with bucketed graph replay
- `custom-kernels`: fully eager scheduler with custom Triton RMSNorm and Qwen RoPE
- `regime-dispatched`: the eager custom-kernel control plus H100-fitted attention
  dispatch, SwiGLU fusion, and fused RoPE/KV placement
- `vllm`: optional offline vLLM reference

The output contains raw per-request traces, repetition-level aggregates, a median
summary, system/package metadata, and a flat CSV for plots and ablation tables.

The `paged-kv` milestone is not a cache-only micro-ablation: in the current codebase,
the paged engine introduces both the paged cache and its cache-aware decode-attention
kernel. Their isolated kernel/memory results remain in the existing focused benchmarks.

## Measurement contract

- Model loading is excluded from request timing and reported as metadata.
- Persistent KV-pool allocation and optional legacy CUDA-graph capture are excluded
  from steady-state request timing and reported as `setup_time_s`.
- Every measured repetition reuses the same model and cache buffers, but resets all
  logical cache and scheduler state.
- EOS is ignored, so every backend must produce exactly the requested output length.
- TTFT begins at the workload's scheduled arrival, so it includes queueing.
- TPOT is `(finished - first_token) / (output_tokens - 1)`.
- ITL contains the observed gaps between individual output tokens.
- CUDA peak memory includes the loaded model and persistent cache pools.
- A run fails if any request is missing or produces the wrong number of tokens.
- Generated token IDs are retained. Free-running exact match and matched-prefix
  diagnostics are reported against the PyTorch baseline when it is present; these are
  diagnostic rather than strict gates because fp16 near-ties can change greedy output.

The offline vLLM adapter supports burst workloads only. This keeps the comparison
in-process and avoids including HTTP client/server overhead for only one backend.
Depending on the installed vLLM version, offline per-request latency metrics may be
unavailable; throughput and token counts are always reported, and unavailable latency
fields remain null rather than being estimated.

The primary vLLM comparison is matched on the exact prompt token IDs, requested output
lengths, greedy sampling/EOS policy, model dtype, paging block size, maximum concurrent
sequences, maximum context length, and logical KV-cache capacity. vLLM keeps its own
optimized scheduler, graph implementation, and kernels: those are the implementation
being compared. `--vllm-kv-cache-mode native` is available as a separately labeled
upper-bound configuration, but it is not a memory-matched result.

## Quick start

Run all five integrated local stages:

```bash
python3 inference-engine/benchmarks/run_benchmarks.py \
  --backends pytorch-baseline,paged-kv,continuous-batching,bucketed-cuda-graphs,custom-kernels,regime-dispatched \
  --num-requests 8 \
  --prompt-lengths 128,512 \
  --output-lengths 32,64 \
  --warmups 1 \
  --repetitions 3
```

Exercise staggered Poisson arrivals at four requests/second:

```bash
python3 inference-engine/benchmarks/run_benchmarks.py \
  --backends pytorch-baseline,paged-kv,continuous-batching,bucketed-cuda-graphs,custom-kernels \
  --request-rate 4 \
  --num-requests 16
```

Bound each request's prefill work per scheduler iteration while preserving its
paged-KV and RoPE continuation state:

```bash
python3 inference-engine/benchmarks/run_benchmarks.py \
  --backends custom-kernels \
  --prompt-lengths 10000 \
  --max-prefill-chunk-size 2048
```

Intermediate chunks do not sample. The first output token is emitted only after the
request's final prompt chunk; result metadata records both request counts and token
counts for every prefill iteration.

The global scheduler limit is shared by decode and prefill. Active decodes consume one
token each first; prompt chunks fill the remaining capacity:

```bash
python3 inference-engine/benchmarks/run_benchmarks.py \
  --backends custom-kernels \
  --max-num-batched-tokens 4096 \
  --num-requests 16 \
  --prompt-lengths 512,8192 \
  --output-lengths 64
```

`max_prefill_chunk_size` is optional and remains only a per-request fairness cap. It
does not replace or override `max_num_batched_tokens`.

Force a smaller scheduler KV pool to exercise preemption:

```bash
python3 inference-engine/benchmarks/run_benchmarks.py \
  --backends continuous-batching,bucketed-cuda-graphs,custom-kernels \
  --num-blocks 32
```

## Reusing exactly the same workload with vLLM

vLLM commonly needs its own environment because it pins CUDA/PyTorch versions. Save
the token trace during the local run, then load that trace in the vLLM environment:

```bash
python3 inference-engine/benchmarks/run_benchmarks.py \
  --backends pytorch-baseline,paged-kv,continuous-batching,bucketed-cuda-graphs,custom-kernels \
  --workload-out /tmp/qwen-burst.json

python3 inference-engine/benchmarks/run_benchmarks.py \
  --backends vllm \
  --workload-in /tmp/qwen-burst.json \
  --max-running 8 \
  --block-size 16 \
  --vllm-kv-cache-mode matched \
  --compare-with inference-engine/benchmarks/results/LOCAL_RESULT.json
```

`--compare-with` imports the prior local rows, checks exact workload equality plus the
matched model/dtype/block-size/concurrency settings, and prints one combined table with
throughput ratios relative to `custom-kernels`. If `--workload-in` is omitted, the
workload is loaded directly from the first comparison result. An old unconstrained
vLLM result is rejected rather than being mislabeled as matched.

Use `--strict-backends` when an unavailable optional backend should fail the entire
command. Without it, unavailable backends are recorded in the JSON and skipped.

## Dense MLP activation fusion

The SwiGLU experiment isolates the two elementwise launches between the gate/up and
down-projection GEMMs. Its packed-row defaults include the launch-bound, balanced, and
prefill-heavy regime-atlas shapes:

```bash
python3 experiments/mlp/benchmark_swiglu_fusion.py
```

This is a microkernel diagnostic, not an end-to-end claim. It records raw timings,
correctness error, and an explicit traffic model for the unfused and fused paths. A
winning configuration must still be replayed through the mixed executor before it is
accepted.

## Phase-separated prefill and long-context decode

`run_phase_sweep.py` prevents the mixed end-to-end workload from hiding which phase
dominates. Prefill uses real identical prompts and mirrors the current scheduler's
serial admission behavior. Decode seeds zero-valued paged KV histories directly, so
long-context measurements isolate steady-state O(L) cache traversal without first
paying quadratic prompt attention. Decode is teacher-forced to exclude per-step
device-to-host sampling synchronization.

The default H100 sweep covers batches 1/2/4/8, real prefill lengths through 4K, and
decode contexts through 32K while preserving 32 tokens of generation headroom:

```bash
python3 benchmarks/run_phase_sweep.py \
  --implementations custom-kernels \
  --phases prefill,decode \
  --batch-sizes 1,2,4,8 \
  --prefill-lengths 128,512,1024,2048,4096 \
  --context-lengths 128,512,1024,2048,4096,8192,16384,32736 \
  --decode-steps 32 \
  --warmups 1 \
  --repetitions 3
```

Every row includes raw timings, median phase latency, aggregate and per-sequence
throughput, peak PyTorch allocation, a shape-specific correctness check, and executor
setup time as excluded metadata. JSON and CSV are written under
`benchmarks/results/phase-sweeps/`. OOM shapes are recorded rather than discarding the
rest of the sweep.

## Mixed token-budget frontier

The scheduler frontier sweep loads the model once, varies prompt length, request count,
arrival timing, and the shared token budget, and writes full request traces plus a flat
CSV under `experiments/results/token-budget/`:

```bash
python3 experiments/scheduler/benchmark_token_budget.py \
  --implementation custom-kernels \
  --prompt-lengths 128,512,2048,8192 \
  --request-counts 1,4,16 \
  --request-rates 0,20 \
  --max-num-batched-tokens 512,1024,2048,4096,8192,16384 \
  --output-length 64 \
  --max-running 16 \
  --warmups 1 \
  --repetitions 3
```

Every measured run asserts that its observed iteration token count stays within the
configured budget. Output IDs are compared across budgets for exact-match and
matched-prefix diagnostics, while TTFT, TPOT, ITL, end-to-end latency, and throughput
form the performance frontier. This sweep refuses to run with CUDA graphs: decode-only,
prefill-only, and mixed iterations all use the same eager custom-kernel executor so the
budget policy is the only changing variable.

The same harness can add a prefix-aware attention-work ceiling. Work is counted as
causal query-key pairs: a chunk of `q` tokens after a cached prefix of `p` costs
`q*p + q*(q+1)/2`. The token ceiling remains active, so the two limits independently
bound linear projection/MLP work and attention work.

Use this focused H100 comparison before introducing feedback control:

```bash
python3 experiments/scheduler/benchmark_token_budget.py \
  --implementation custom-kernels \
  --prompt-lengths 8192 \
  --request-counts 16 \
  --request-rates 20 \
  --max-num-batched-tokens 16384 \
  --max-prefill-chunk-size 2048 \
  --max-prefill-attention-pairs none,262144,1048576,4194304,16777216 \
  --output-length 64 \
  --max-running 16 \
  --warmups 1 \
  --repetitions 3
```

`none` is the existing token-only baseline. Each capped run asserts that measured
mixed-iteration attention work never exceeds its ceiling and reports mixed pair-budget
utilization alongside TTFT, TPOT, ITL, and throughput. Prefill-only iterations remain
token-budgeted and are not throttled by this ceiling.

### Work-budget decode-latency distribution

The comprehensive work-budget profiler down-sweeps from the uncapped regime and writes
both a summary frontier and one raw row per scheduler iteration:

```bash
python3 experiments/scheduler/profile_work_budget_latency.py \
  --prompt-length 8192 \
  --request-count 16 \
  --request-rate 20 \
  --output-length 64 \
  --max-running 16 \
  --max-num-batched-tokens 16384 \
  --max-prefill-chunk-size 2048 \
  --max-prefill-attention-pairs \
none,67108864,50331648,33554432,25165824,16777216,12582912,8388608,6291456,4194304,3145728,2097152,1048576 \
  --latency-thresholds-ms 40,50,75,100,150 \
  --warmups 1 \
  --repetitions 3
```

The iteration CSV aligns wall latency with iteration type, decode count, every decode
context length, prefill tokens, attention pairs, and every prefill prefix length. The
summary weights each decode-bearing iteration by its decode count, reports threshold
violation rates, and includes output throughput plus SLO-compliant decode goodput.
Input/prefill throughput remains diagnostic: it is not treated as interchangeable with
the more expensive and user-visible output-token throughput.

## Packed ragged-prefill ablation

The focused ablation loads one model and compares request-serial admission, packed
ragged projection/MLP with per-sequence SDPA, and packed variable-length Triton
attention over uniform and heterogeneous prompt batches:

```bash
python3 experiments/prefill/benchmark_ragged_prefill.py \
  --batch-sizes 1,2,4,8 \
  --prompt-lengths 128,512,2048,4096 \
  --patterns uniform,ramp \
  --warmups 2 \
  --repetitions 5
```

All modes execute the same prompts, model, paged cache, greedy first-token policy, and
scheduler lifecycle. This separates the gain from packing projection/MLP work from the
gain due specifically to replacing the per-sequence SDPA loop. Timestamped JSON/CSV
files are written under `experiments/results/ragged-prefill/`; any first-token or
admission-shape mismatch fails the command.

### Packed-attention tile sweep

After the end-to-end ablation, isolate the variable-length attention operation and
sweep query/key tile geometry while keeping Qwen's 12:2 GQA layout fixed:

```bash
python3 experiments/prefill/benchmark_packed_attention_tiles.py \
  --batch-sizes 4,8 \
  --prompt-lengths 512,2048,4096 \
  --patterns uniform,ramp \
  --block-ms 32,64,128 \
  --block-ns 32,64,128 \
  --warmups 2 \
  --repetitions 10
```

Every configuration compiles before timing and must match the per-sequence SDPA
reference. Compilation, resource, and correctness failures are recorded without
discarding the rest of the sweep. JSON/CSV results are written under
`experiments/results/packed-attention-tiles/`. The current production default is
64x32, selected from the H100 frontier results.

After identifying promising geometries, run the focused frontier sweep to check
whether one choice generalizes across occupancy-starved short prompts and long-context
attention:

```bash
python3 experiments/prefill/benchmark_packed_attention_tiles.py \
  --batch-sizes 1,2,4,8 \
  --prompt-lengths 128,256,512,2048,4096,8192,16384 \
  --patterns uniform,ramp \
  --tile-configs 32x32,64x32,64x64,128x64 \
  --warmups 2 \
  --repetitions 10
```

`--tile-configs` evaluates only the listed `(BLOCK_M, BLOCK_N)` pairs; without it,
the benchmark retains the Cartesian-product behavior of `--block-ms` and
`--block-ns`. The focused command produces 224 rows rather than 504 and directly
tests whether the 64x32 result remains robust outside the original B=4/8,
L=512..4096 range.

### Resumed paged-prefill tile sweep

The fresh-prefill result does not establish the best geometry for a short query chunk
attending through a long paged prefix. This experiment holds the model layout and page
size fixed, validates every configuration against a paged SDPA reference, and reports
speedup relative to the production 64x32x4x2 control:

```bash
python3 experiments/prefill/benchmark_paged_prefill_tiles.py \
  --batch-sizes 1,2,4 \
  --query-lengths 64,128,256,512,1024,2048 \
  --prefix-lengths 0,128,512,2048,8192,16384 \
  --configs 32x16x4x2,32x32x2x2,32x32x4x2,64x32x4x2,64x64x4x2,128x64x8x2 \
  --warmups 2 \
  --repetitions 10
```

Results are written under `experiments/results/paged-prefill-tiles/`. Production is not
changed until the resumed-shape weighted result clears the criterion in
[`INTERVENTIONS.md`](INTERVENTIONS.md). The summary reports the best single static
configuration and a per-shape oracle. An adaptive selector is only justified when the
oracle materially beats the best static configuration and winner regions are stable.

The measured two-way policy is available for later end-to-end A/B testing without
changing the production default:

```bash
python3 benchmarks/run_benchmarks.py \
  --backends custom-kernels \
  --prompt-lengths 16640 \
  --max-prefill-chunk-size 256 \
  --max-num-batched-tokens 1024 \
  --prefill-tile-policy adaptive \
  --num-requests 4 \
  --output-lengths 32 \
  --warmups 1 \
  --repetitions 5
```

Run the identical command with `--prefill-tile-policy static` as the control. The
adaptive path chooses 64x64 below 2048 packed query tokens and 64x32 otherwise.

## Grouped-GQA decode intervention

The regime atlas found that paged decode attention consumes 86% of the long-context,
high-batch iteration. Qwen's six query heads per KV head currently reread the same KV
history independently. The experimental kernel shares each KV page across 1/2/3/6
query heads and records compilation/resource failures without aborting the sweep:

```bash
python3 experiments/decode/benchmark_grouped_gqa_decode.py \
  --batch-sizes 1,8,32,64 \
  --context-lengths 128,2048,8192,16384 \
  --heads-per-program 1,2,3,6 \
  --num-warps 4,8 \
  --warmups 2 \
  --repetitions 10
```

Before timing, every configuration is checked against an independent manual reference
using ragged lengths around page boundaries and non-contiguous, isolated page tables.
Every full-sweep shape must then match the production paged-decode kernel. Batch and
context axes test whether the candidate moves the decode saturation knee instead of
winning at one convenient shape. Raw results and a per-shape winner summary are written
under `experiments/results/grouped-gqa-decode/`; the production path is not changed
until a stable batch/context dispatch boundary is measured.

## Profiling

### Detailed regime atlas

The 144-point mixed surface identifies latency regimes but intentionally avoids tracing
overhead. The atlas runner collects full operator tables and Chrome traces at six
predeclared corners: launch-bound, balanced fresh, balanced resumed, decode-bandwidth,
prefill-compute, and dual-heavy. Synthetic resident KV staging keeps long-context setup
outside the target trace.

```bash
python3 experiments/profiling/run_regime_atlas.py \
  --warmups 2 \
  --timing-repetitions 5 \
  --prefill-tile-policy static
```

Use `--dry-run` to print the exact six commands. Results are grouped under
`experiments/results/regime-atlas/`; each regime receives independent metadata,
operator tables, and a Chrome trace, plus one top-level manifest.

Map the decode/prefill tradeoff with matched controls. The surface profiler measures
decode-only `T(D,0)`, prefill-only `T(0,P)`, and mixed `T(D,P)` iterations on one
loaded model:

```bash
python3 experiments/scheduler/profile_mixed_surface.py \
  --decode-requests 1,8,32,64 \
  --decode-context-lengths 128,2048,8192 \
  --prefill-tokens 512,2048,4096,8192 \
  --prefill-prefix-lengths 0,4096,16384 \
  --prefill-requests 2 \
  --warmups 1 \
  --repetitions 5
```

Decode histories and prefill prefixes are staged directly in resident paged KV outside
timing. KV values do not affect launch geometry or attention work; target iterations
execute the real eager scheduler and model kernels. JSON retains raw CUDA-event and
wall-clock samples. The flat CSV reports both medians and the corresponding tradeoff
metrics:

`--prefill-tokens` is the total scheduled prefill work across
`--prefill-requests`; each request receives an equal chunk. Every decode request
contributes one scheduled decode token.

- `T(D,P) - T(D,0)`: incremental prefill interference seen by active decodes.
- `T(D,P) / T(D,0)`: decode-iteration stretch.
- `T(D,0) + T(0,P) - T(D,P)`: time saved by mixed packing.
- `(T(D,0) + T(0,P)) / T(D,P)`: mixed-versus-separate execution speedup.

Use a small smoke surface before the full run:

```bash
python3 experiments/scheduler/profile_mixed_surface.py \
  --decode-requests 8,32 \
  --decode-context-lengths 2048 \
  --prefill-tokens 512,2048,4096 \
  --prefill-prefix-lengths 0,4096 \
  --warmups 1 \
  --repetitions 3
```

Results are written under `experiments/results/mixed-surface/`. Use the existing
single-point profiler below after the surface identifies important boundary points;
it supplies the operator tables and Chrome/NVTX trace for kernel attribution.

Profile the scheduler's real prefill path. `--requests 1` measures the serial reference;
values greater than one exercise packed ragged prefill:

```bash
python3 experiments/prefill/profile_prefill.py \
  --implementation custom-kernels \
  --prompt-length 2048 \
  --requests 1 \
  --warmups 3 \
  --timing-repetitions 5
```

This produces three timestamped artifacts under `experiments/results/profiles/`: a
Chrome trace, CUDA/CPU operator tables, and JSON metadata containing low-overhead
CUDA-event latency, throughput, memory, configuration, and environment details. The
trace contains nested regions for cache/input setup, RMSNorm, QKV projections, RoPE,
KV-cache writes, attention, output projection, MLP, final norm/head, and sampling.
Use the CUDA-event latency for performance comparisons; profiler timings are for
attribution and include tracing overhead.

Profile a true mixed iteration after staging real decode histories and optional
prefill-prefix KV state:

```bash
python3 experiments/scheduler/profile_mixed_batch.py \
  --implementation custom-kernels \
  --decode-requests 32 \
  --decode-context-length 2048 \
  --prefill-requests 2 \
  --prefill-prefix-length 0 \
  --prefill-chunk-size 2048 \
  --warmups 3 \
  --timing-repetitions 5
```

The mixed trace separates cache/input construction, packed QKV projection, RoPE, KV
writes, decode attention, prefill attention, output projection, and MLP. Staging is
excluded from the torch-profiler trace and CUDA-event latency.

The same scheduler regions are emitted as NVTX ranges. For an Nsight Systems capture:

```bash
nsys profile --trace=cuda,nvtx,osrt,cublas,cudnn \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --output=experiments/results/profiles/mixed-balanced \
  --force-overwrite=true \
python3 experiments/scheduler/profile_mixed_batch.py \
  --decode-requests 32 --decode-context-length 2048 \
  --prefill-requests 2 --prefill-chunk-size 2048 \
  --nvtx-only --cuda-profiler-range
```

The capture range excludes model loading, synthetic KV staging, warmups, and timing
runs from the report. Remove `--cuda-profiler-range` when invoking the script without
Nsight Systems.

## CPU-only control-plane tests

```bash
python3 -m unittest discover -s experiments/tests -p 'test_*.py'
```

These test deterministic workload generation, serialization, metric definitions,
percentiles, and repetition summaries without loading a model or importing Triton.

Before collecting benchmark numbers, run the consolidated GPU correctness suite:

```bash
python3 correctness/run_correctness.py --checks baseline-vs-hf
```

Use `--checks all` when also validating the legacy CUDA-graph ablations.
