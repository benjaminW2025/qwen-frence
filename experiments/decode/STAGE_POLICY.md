# Pages per program and decode stage dispatch

## Question and falsifiable hypothesis

For this Qwen2.5-1.5B decode kernel, does pipeline depth pay off according to
**page iterations per program**, **available parallel work**, and the resources
required by the generated pipeline? Longer contexts do not inherently require
more stages. At fixed per-page computation, they can amortize startup costs;
changing K also changes parallel work and reduction cost.

Keep H=1, 12 query heads / 2 KV heads, head dimension 128, page size 16, four
warps, and FP16 fixed. This experiment supports a dispatch policy for that scope,
not a claim about arbitrary attention implementations or GPUs. It uses the
existing kernel math. An optional `launches` argument exposes the exact
preallocated partial, reduction, and combined launches for measurement.

## Design

The full preset contains 290 cases and 4,240 candidate case/configuration pairs:

| Suite | Cases | Purpose |
| --- | ---: | --- |
| mechanism | 220 | Vary exact full pages/program, K, and batch independently where possible |
| train | 20 | Fit simple dispatch rules on fixed workloads |
| validation | 12 | Select rule complexity using different batch/context shapes |
| test | 20 | Final uniform/tail-shape evaluation; no fitting or selection |
| ragged | 18 | Held-out ramp and strongly skewed lengths; test max-length dispatch robustness |

Mechanism axes:

- Batch: 1, 8, 32, 128.
- K: 1, 2, 4, 8, 16, 32, 64.
- Pages per program: 1, 2, 3, 4, 8, 16, 32, 64.
- Stages: 1, 2, 3, 4; stage 1 is always the matched control.
- Construct `context = 16 * K * pages_per_program`; omit contexts above 32,768.

One page/program is a **negative control**: no cross-page overlap is possible.
Extra stages may compile away or still incur costs. Do not assume they must
make the kernel slower. Compare generated code to see what actually happened.

At fixed batch/K, varying pages/program holds the launch grid fixed. At fixed
pages/program, varying K changes the grid, total workload, and cache footprint.
The latter comparison is not a pure occupancy experiment. The plan also contains
cohorts with equal `batch * K` and pages/program, e.g. `(B,K)=(1,32),(8,4),(32,1)`:
these hold the launch grid and allocated KV bytes constant while changing batch/K.
Per-shape features record exact min/max/mean page iterations, empty-program
fraction, active program count, launched programs/SM, and allocated KV bytes.
**Launched programs/SM is not measured residency or active SM count.**

Dispatch suites use a common set of K values: the explicit grid plus the union
of auto-K choices across their shapes. This lets us compare a fixed policy,
auto-K, and learned policies using actually measured actions. Auto-K targets four
launched programs/SM with a two-page minimum-chunk heuristic. Train/validation/test
batch-context pairs are disjoint. Test includes non-page-aligned contexts and
batches absent from training. Ragged cases are not used for fitting/selection.

## Timing and controls

Default protocol: **5 independent tensor-seed trials**, each with **9 interleaved
sample rounds**. Within each trial, sample order is shuffled across K, stages,
operation role, cache condition, and production. Compilation and graph capture
occur before timing. Every timed replay executes one attention operation.
Stage-1 comparisons are paired within the same case/trial/cache condition.

Record these separately:

1. Full attention: partial work plus reduction; the dispatch objective.
2. Partial attention kernel only: mechanism diagnosis.
3. Reduction only (K>1): overhead diagnosis, using valid precomputed partials.

For K=1, the partial kernel includes final normalization and equals full
attention. Do not interpret its timing as identical work to a K>1 partial kernel.
Nor should isolated partial and reduction timings be added to predict full
latency: cache state and launch boundaries can differ.

Two cache conditions are measured separately:

- `warm`: replay the same operation immediately before its timed replay.
- `evict`: touch a 256 MiB buffer before the timed replay. This is eviction
  **stress**, not proof that every load misses cache. Buffer work is outside the
  event interval. Check L2/DRAM counters on the target GPU.

Timing covers GPU graph execution, not scheduling, Python/C++ dispatch,
allocation, or an end-to-end serving step. For very short kernels, graph/event
measurement overhead can limit resolution; inspect sample variability rather
than interpreting tiny differences. Trial checkpoints include GPU temperatures,
clocks, power, utilization, wall duration, sample order, all raw timings, and
register/spill/shared-memory metadata. The harness observes clocks; it does not
change them or guarantee an otherwise idle GPU.

Independent dense FP32 reference checks cover every measured input. Each unique
kernel specialization also passes poisoned padding, strided Q/V/metadata,
empty-partition/negative-score, and accumulator-overflow cases. Full graph replay
outputs are checked again after timing. Failures abort rather than generating a
winner from an incomplete candidate set.

## Run it

First inspect the plan without CUDA:

```bash
python3 experiments/decode/benchmark_decode_stage_policy.py plan \
  --preset full --suite all --num-sms 132 > /tmp/stage-plan.json
```

Then run the small protocol check on the GPU. This still exercises all suites,
stage 4, independent component launches, reference checks, and policy analysis:

```bash
python3 experiments/decode/benchmark_decode_stage_policy.py run \
  --preset smoke --suite all --trials 2 --samples 3 --cache-modes warm \
  --output-dir experiments/results/decode-stage-policy-smoke
python3 experiments/decode/benchmark_decode_stage_policy.py analyze \
  --output-dir experiments/results/decode-stage-policy-smoke
```

Smoke data intentionally cannot pass the evidence gate. For real measurements:

```bash
python3 experiments/decode/benchmark_decode_stage_policy.py run --suite mechanism
python3 experiments/decode/benchmark_decode_stage_policy.py analyze
python3 experiments/decode/benchmark_decode_stage_policy.py run --suite train
python3 experiments/decode/benchmark_decode_stage_policy.py run --suite validation
python3 experiments/decode/benchmark_decode_stage_policy.py run --suite test
python3 experiments/decode/benchmark_decode_stage_policy.py run --suite ragged
python3 experiments/decode/benchmark_decode_stage_policy.py analyze
```

`--suite all` runs the complete plan in one invocation. This is substantially
larger than the prior sweep: plan output reports timed replay count. An exact
`--case-id` from the plan selects one case for a targeted rerun/profile.

Each completed case/trial is written atomically under `trials/`. Rerun the same
command to resume; completed checkpoints are skipped. An interrupted trial is
repeated in full so paired comparisons are not spliced across sessions. Changing
source code, GPU identity, software/driver versions, seed, trial/sample count,
cache protocol, or preset requires a **new output directory**. Suite/case selectors
can change to populate the same design in stages. Do not run two writers against
the same output directory concurrently.

## Profile representative cases separately

Use Nsight Compute outside timing runs. For example, profile one page/program,
then repeat at 2, 8, and 64 pages/program and stages 1–4 with the same batch/K:

```bash
ncu --set full --cache-control none --nvtx --nvtx-include 'decode_stage_profile/' \
  -o /tmp/decode-one-page-s2 \
  python3 experiments/decode/benchmark_decode_stage_policy.py profile \
  --case-id mechanism-b8-l128-uniform-k8 --k 8 --stages 2 \
  --role partial --cache-modes warm
```

The stable NVTX range contains only the selected launch(es); warmup and JIT
compilation are outside it. `--role full` includes reduction; `--role reduce`
profiles just the reducer. TTGIR/PTX is dumped for inspection. Compare achieved
occupancy, eligible warps, long-scoreboard/memory stalls, SM utilization, L2/DRAM
traffic, registers, spills, and shared memory. Check for actual asynchronous
loads and their scheduling. Metric availability depends on GPU/Nsight version;
`--set full` avoids hardcoding architecture-specific metric names.

Profiler replay and clock control can change conditions; profiler durations are
not replacements for the ordinary benchmark timings. The CLI's NVTX push/pop
filter syntax and cache controls are documented in the
[Nsight Compute CLI guide](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html).
CUDA graph capture retains tensor addresses; the prepared launch closures keep
buffers alive, following [PyTorch's graph lifetime requirements](https://docs.pytorch.org/docs/main/notes/cuda.html#cuda-graphs).

## Analysis and policy evidence

`stage-effects.json` contains stage-1/candidate speedup estimates with paired
95% bootstrap intervals, resampling **whole independent trials**, not treating
correlated event samples as independent observations. These are exploratory,
pointwise intervals without a multiple-comparison correction. They diagnose the
hypothesis; they are not significance certificates for thousands of winners.
Mechanism-only runs can produce this file before dispatch suites finish.

Once all dispatch suites are complete, `policy-report.json`:

- Fits depth 0/1/2 trees on training shapes only, using batch and maximum page count.
  Leaves select a measured `(K, stages)` pair; H remains 1.
- Minimizes mean log slowdown relative to each training shape's measured oracle.
  A greedy tree is a compact baseline, not an exhaustive policy optimizer.
- Uses validation shapes to select depth, preferring the simpler tree when within
  1% of the best validation loss. No refitting on test/ragged cases occurs.
- Reports held-out geometric-mean, p95, and worst-case slowdown versus the measured
  oracle. Also evaluates the train-selected fixed pair, auto-K/stage-1, and K=1/stage-1.
- Reports paired uncertainty against production on each held-out shape.
- Flags cases whose observed winner hits the maximum K or stage candidate: they
  may justify expanding a future training sweep, not declaring a global optimum.

The policy is fitted on warm-cache data by default (`--fit-cache evict` is an
explicit alternate experiment), and evaluated under every collected cache mode.
The provisional microbenchmark gate requires at least five trials, validation
p95 slowdown <=10%, every held-out suite/cache p95 slowdown <=10%, worst slowdown
<=20%, and a per-shape policy/production speedup interval lower bound >=0.95.
The test oracle itself is estimated from finite measurements; regret is not exact.
These thresholds are declared before running, not chosen after seeing results.

A passing artifact is still **not production-ready**. If test results inform
policy changes, use a new final holdout. Confirm borderline gains with fresh
sessions, verify profiling explanations, and run the candidate in the full C++
engine under latency/throughput objectives and realistic mixed batches. Cache
state, weight traffic, scheduler behavior, and neighboring kernels can change
which configuration wins.

## Local tests

```bash
python3 -m unittest discover -s experiments/tests -p 'test_decode_stage_policy.py' -v
python3 -m unittest discover -s experiments/tests -p 'test_grouped_splitk_pipelined.py' -v
```

CPU tests cover exact partition geometry, common controls, constant-grid/footprint
cohorts, held-out isolation, tree fitting, trial-level intervals, checkpoint
compatibility, and a complete synthetic analysis. CUDA tests verify preallocated
partial/reduce/full replay, including stage 4. GPU tests skip explicitly when
CUDA/Triton is unavailable.
