# C++ scheduler and decode-policy integration experiment

## Frozen input-shape table

`fixed_regime.py` is the source of truth for this checkpoint. All rows use
Qwen2.5-1.5B FP16 on one H100, page16, uniform burst arrivals, greedy sampling,
ignore EOS, factorial prompt lengths 256/2048 and output lengths 128/256
per request, plus two targeted 4096-prompt/256-output context probes,
prefill work capped at 2048 packed tokens per step, and the 2048-token graph
bucket. No row is tuned from timing results.

| Shape ID | B/max running | Cohort prompt tokens | Cohort output tokens | Max context/request | Max packed prefill/step | CPU dry-run pure B decode steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fixed-b8-l256-o128` | 8 | 2,048 | 1,024 | 384 | 2,048 | 127 |
| `fixed-b8-l256-o256` | 8 | 2,048 | 2,048 | 512 | 2,048 | 255 |
| `fixed-b8-l2048-o128` | 8 | 16,384 | 1,024 | 2,176 | 2,048 | 120 |
| `fixed-b8-l2048-o256` | 8 | 16,384 | 2,048 | 2,304 | 2,048 | 248 |
| `fixed-b64-l256-o128` | 64 | 16,384 | 8,192 | 384 | 2,048 | 120 |
| `fixed-b64-l256-o256` | 64 | 16,384 | 16,384 | 512 | 2,048 | 248 |
| `fixed-b64-l2048-o128` | 64 | 131,072 | 8,192 | 2,176 | 2,048 | 64 |
| `fixed-b64-l2048-o256` | 64 | 131,072 | 16,384 | 2,304 | 2,048 | 192 |
| `probe-b8-l4096-o256` | 8 | 32,768 | 2,048 | 4,352 | 2,048 | 241 |
| `probe-b64-l4096-o256` | 64 | 262,144 | 16,384 | 4,352 | 2,048 | 129 |

The pure-decode counts are predictions from the current CPU C++ scheduler dry
run. Every measured row must actually reach its declared batch for **at least
64 pure decode steps**; the GPU preflight and timed runs assert this. “Total
tokens” in the table means the whole request cohort, whereas the 2048-token
limit is **per prefill iteration**. Inspect the machine-readable table with
`python3 experiments/integration/benchmark_latest_vs_vllm.py table`.
The old scorecard used 8192-token long prompts; this table caps the eight-cell
factorial at 2048 and adds only two 4096-token probes to limit GPU spend.
Split-K is intentionally inactive for
256-token prompts because their maximum context is below its 1024-token policy
threshold; those rows still compare eager, decode graphs, and piecewise graphs.

## Tonight's fixed-regime vLLM checkpoint

`benchmark_latest_vs_vllm.py` freezes one selected row from the table, then runs
the four-arm C++/graph ablation and
a separate reference process using the previous `regime-dispatched` Python
engine plus vLLM. Both read the same saved prompt token IDs and output lengths.
The reference uses FP16, page16, the row's concurrent-sequence count, a 2048-token
prefill/batched-token budget, matched logical KV capacity, greedy sampling,
ignore-EOS behavior, and offline burst timing. Model loading, correctness, and
graph capture are outside measured workloads. Runs are separate processes to
avoid cross-backend GPU memory ownership.

Start locally, before the H100 session:

```bash
python3 experiments/integration/benchmark_latest_vs_vllm.py table
python3 experiments/integration/benchmark_latest_vs_vllm.py plan \
  --shape-id fixed-b8-l256-o128
python3 experiments/integration/benchmark_latest_vs_vllm.py prepare \
  --shape-id fixed-b8-l256-o128 \
  --output-dir experiments/results/latest-vllm-b8-short-main
python3 experiments/integration/benchmark_latest_vs_vllm.py dry-schedule \
  --shape-id fixed-b8-l256-o128 \
  --output-dir experiments/results/latest-vllm-b8-short-main
```

On H100, run the internal ablation, then the external reference, then analysis.
The checkpoint defaults to a **one-sample smoke** (one trial/sample, no warmup,
one reference repetition), not a selection-quality benchmark:

```bash
python3 experiments/integration/benchmark_latest_vs_vllm.py check-model-cache
python3 experiments/integration/benchmark_latest_vs_vllm.py run-ablation \
  --shape-id fixed-b8-l256-o128 \
  --output-dir experiments/results/latest-vllm-b8-short-main
python3 experiments/integration/benchmark_latest_vs_vllm.py run-reference \
  --shape-id fixed-b8-l256-o128 \
  --output-dir experiments/results/latest-vllm-b8-short-main
python3 experiments/integration/benchmark_latest_vs_vllm.py analyze \
  --shape-id fixed-b8-l256-o128 \
  --output-dir experiments/results/latest-vllm-b8-short-main
```

Repeat for the other seven factorial IDs in the table, each in its **own fresh**
output directory. Use the default one-trial smoke for all eight cells first,
then the two context probes only if the basic run passes.
Start with B=8 before the larger rows. For selection-quality measurements, use
a **new** directory, run `prepare` there, then pass
`--trials 3 --samples 3 --warmups 1` to `run-ablation` and
`--repetitions 3 --warmups 1` to `run-reference`. Do not treat the smoke result
as a split-K winner. The worst B=64/P=2048 full plan includes 36 timed ablation
workloads and about 4.72 million timed ablation prompt tokens, plus reference workloads,
preflight, and warmup; inspect `plan` with the full flags before committing GPU time.
For the 4096-token B=64 probe, the one-trial smoke alone processes 1,048,576
timed ablation prompt tokens and 524,288 reference prompt tokens, plus
correctness and capture work; avoid launching the 3×3 plan across all ten rows.

All integration benchmark and profile entry points accept the same table through
`--preset fixed --case-id <shape-id>`; `make_plan("fixed")` reads
`fixed_regime.py`, rather than duplicating shapes. The integrated ablation and
vLLM checkpoint use the same saved `workload.json`. For a focused full-batch
decode trace on a selected row:

```bash
python3 experiments/integration/benchmark_latest_vs_vllm.py run-profile \
  --shape-id probe-b64-l4096-o256
```

The checkpoint action passes the pinned model and exact saved workload, and
defaults to one unprofiled repetition with no warmup. The fixed profiler
chooses a pure full-batch decode step. To trace long-context prefill, pass
`--profile-kind prefill --profile-occurrence 1` on a 4096-token probe; that
selects its second 2048-token chunk with an existing prefix. The legacy C++ graph and piecewise
benchmarks can also select rows from the fixed table, but they are diagnostic
sub-ablations, not additional arms in the vLLM checkpoint.

The analyzer refuses mismatched prompt IDs, output lengths, model/config, or
missing arms. It reports net output-throughput change against the prior engine,
production-attention and split-K versions of the integrated path, and the
remaining vLLM throughput gap. It also records whether generated token IDs
match across the prior engine, integrated path, and vLLM; unequal IDs are
reported, not mislabeled as an identical-trajectory comparison. Use a fresh
directory for every run. Install the pinned vLLM environment in
`benchmarks/requirements-vllm-cu128.txt` before the reference run. Both run
commands resolve the same cached immutable Qwen snapshot
(`8faed761d45a263340a0528343f099c05c9a4323`) before launching model work;
they fail if it is not staged. `--model` may instead point to an existing local
model directory, but that override is labeled unverified in the summary.

This checkpoint measures fixed-shape offline burst **output throughput**, not
TTFT/SLO serving behavior. The C++/graph adapter is currently an experimental
real-model executor and is **not** a strict superset of all fusions in the prior
`regime-dispatched` production path. Thus its ratio against that engine is the
net package effect, not the isolated benefit of each new optimization. The
four-arm ablation supplies attribution inside the new executor. A production
backend integration and serving-traffic replay remain separate milestones.

## One integrated graph experiment

`benchmark_integrated_graph.py` runs four paired arms against the same real-model
C++ scheduler, requests, weights, and KV pool: eager model execution, full-model
production decode graph, that decode graph plus piecewise packed prefill, and the
same piecewise path with split-K decode attention. It is an implementation-path
comparison: the eager and captured decoders have different executor code, so the
first effect is not a pure CUDA-graph-only ablation. It does **not** compare with
vLLM or the production Python scheduler.

Before renting a benchmark window, inspect the CPU-only plan, build the C++
extension, check setup, and run one measured sample:

```bash
python3 experiments/integration/benchmark_integrated_graph.py --plan
make cpp-scheduler-build
python3 experiments/integration/benchmark_integrated_graph.py --dry-schedule
python3 experiments/integration/benchmark_integrated_graph.py --check-setup
python3 experiments/integration/benchmark_integrated_graph.py \
  --trials 1 --samples 1 --warmups 0 \
  --output-dir experiments/results/integrated-graph-smoke
```

The default row is B=8, 2048-token prompts, 128 outputs each, and a 2048-token
prefill budget. The preflight requires an **observed** maximum
decode batch of eight and at least 64 pure B=8 decode steps, an actual
2048-token packed-prefill call, exact scheduled
work and generated tokens across arms, and representative full logits within
the existing integration tolerance. It also checks that split-K and piecewise
capture actually execute. The real C++ scheduler runs a CPU-only dry schedule
first, so a case-name/shape mismatch fails before model loading or graph capture;
the GPU preflight repeats the shape check before warmup or timing.
The one-sample run validates the path, but is not enough to select a winner.

For the paired measurement, use the default three trials and three shuffled
sample rounds, with compilation/capture and correctness outside timing:

```bash
python3 experiments/integration/benchmark_integrated_graph.py \
  --output-dir experiments/results/integrated-graph-main
```

The report contains per-arm workload and phase times, actual batch histograms,
captured prefill buckets, fallback/replay counts, and paired trial speedups.
`splitk_decision` labels split-K a candidate only if every one of at least three
paired trials improves full-workload wall time by 2% or more; otherwise it
conservatively retains production attention. That threshold is a practical
decision rule, **not** a statistical significance claim. For a different case,
set `--expected-max-decode-batch`, `--min-full-decode-steps`,
`--expected-prefill-tokens`, and
`--prefill-buckets` explicitly; do not infer live work from the case ID.

## Focused C++ control-plane profile

`profile_cpp_control.py` first runs an uninstrumented preflight and chooses one
deterministic decode, prefill, or mixed step. It warms the model and graph shapes,
then times that step across separate unprofiled full-workload repetitions.
Finally it traces only that step with PyTorch's CPU/CUDA profiler and drains the
same workload to verify generated tokens and the schedule. The report contains
the unprofiled wall-time distribution and named C++ ranges for scheduling,
metadata building/transfers and waits, callbacks, argmax, sampled-token D2H,
and request updates. Profiler CPU ranges are *inclusive* and may overlap; do not
sum them or interpret callback CPU duration as GPU compute time. Use the trace
timeline to see whether a host stall leaves the GPU idle.

Rebuild the extension after the range instrumentation change, then run a decode
step and a packed-prefill step in fixed regimes:

```bash
make cpp-scheduler-build
python3 experiments/integration/profile_cpp_control.py --check-setup
python3 experiments/integration/profile_cpp_control.py \
  --preset smoke --case-id uniform-b4-l513 --kind decode
python3 experiments/integration/profile_cpp_control.py \
  --preset full --case-id uniform-b8-l512 --kind prefill
```

`--check-setup` verifies CUDA, Triton, and the current C++ extension without
loading weights or capturing graphs. It does not prove that model files are
cached or that the CUDA correctness gates pass; run a one-sample smoke case
before reserving a long benchmark window. These synthetic-token experiments
load model weights directly and do not download an unused tokenizer. If an older
Hub environment sets `HF_HUB_ENABLE_HF_TRANSFER=1` without the optional
`hf_transfer` package, setup warns and selects the standard download path before
importing Hugging Face. The selected mode is recorded in output metadata.

Use `--occurrence` to choose a later step of the same kind and
`--adapter eager-prefill` to compare the control path without piecewise prefill. Traces
are deliberately one step long; repeated timings are collected without profiler
overhead. The current synchronous `step()` reads sampled tokens on the CPU before
the next scheduling decision, so a D2H wait is expected but its *cost* must be
measured, not inferred from the call site.

## Piecewise packed-prefill capture

`benchmark_piecewise_prefill.py` compares the same C++ scheduler and captured
production decode in both arms. The treatment captures one segment between each
pair of attention calls: post-attention output projection/MLP of one layer plus
QKV/RoPE/KV write of the next. The first segment includes embedding; the final
segment includes the last layer's post-attention work. Thus a 28-layer model
replays 29 graphs per captured prefill call, not 56. Packed paged attention
stays eager so it can use live ragged offsets, context lengths, and launch
geometry. Final norm/head remain eager. The implementation pads non-attention
work to packed-token buckets (128, 256, 512, 1024, 2048 by default). A masked
KV-placement kernel reads the real token count from a device scalar, so padded
rows cannot write into live cache slots; eager attention receives only real
query rows and the unmodified ragged metadata. Captures are lazy and bounded
to eight buckets by default. Calls above the largest bucket or beyond that
shape limit explicitly fall back to eager prefill. The report records configured
and captured buckets, graph replays, and capture/fallback call counts. Capture
occurs during correctness preflight, before warmup and measurement.

On a CUDA machine with the C++ extension built, start with the short ragged case:

```bash
python3 experiments/integration/benchmark_piecewise_prefill.py \
  --preset smoke --case-id ragged-b4-l769 --trials 1 --samples 1
```

For a fixed larger regime, pass explicit buckets such as
`--prefill-buckets 512 1024 2048`; this prevents capturing rarely used sizes.
For example, a 2048-token-budget case is:

```bash
python3 experiments/integration/benchmark_piecewise_prefill.py \
  --preset full --case-id uniform-b8-l512 \
  --prefill-buckets 512 1024 2048 --trials 1 --samples 1
```

Inspect `prefill_plus_mixed_ms` and full-workload `wall_ms` separately. The
preflight compares scheduler metadata and representative full logits, then exact
generated tokens and work schedule; timed runs must preserve the latter. This is
an exploratory exact-shape experiment, not yet a production graph policy. More
shape buckets, padding, or mixed prefill/decode capture should be decided from
the measured coverage, memory use, and latency, not assumed to help.

## Captured C++ decode bridge

`benchmark_cpp_graph.py` is a separate paired experiment for the next integration
step. The C++ scheduler supplies persistent device metadata; `GraphModelAdapter`
uses it to replay a full-model decode graph with either production or split-K
attention. Packed prefill stays eager and identical in both arms. Each case checks
metadata, logits at representative shapes, output tokens, and the scheduled work
before accepting timings. Capture and model loading are outside timed workloads.

After building the C++ extension on a CUDA machine, start with one long-context
case that selects split-K:

```bash
make cpp-scheduler-build
python3 experiments/integration/benchmark_cpp_graph.py \
  --case-id uniform-b4-l2048 --trials 1 --samples 1
```

Then increase trials and use `--all-cases` to cover the full long-context preset.
This compares two C++-scheduled graph arms; it is not yet a vLLM comparison.

This sequence answers three questions with the real Qwen2.5-1.5B weights:

1. What does the C++ scheduling/metadata implementation change at fixed model execution?
2. Which ungrouped split-K/stage policy survives the independent decode study?
3. Do the scheduler and decode improvements compose in the real model?

No dispatch winner is hardcoded. The combined experiment requires a passing
`policy-report.json` from the [stage study](../decode/STAGE_POLICY.md), freezes its
selected tree, and evaluates it without fitting on integration results.

## Controls and scope

| Variant | Scheduling/metadata | Decode attention |
| --- | --- | --- |
| python-production | Matched Python reference | Existing production kernel |
| cpp-production | C++ loop with pinned staging | Existing production kernel |
| python-selected | Matched Python reference | Frozen H=1, K/stages tree |
| cpp-selected | C++ loop with pinned staging | Frozen H=1, K/stages tree |

`--phase scheduler` measures the first two. `--phase combined` measures all four
again, interleaved in the same experiment; do not divide timings from different
sessions to estimate the combined gain.

The Python control intentionally mirrors C++ FCFS admission, full-request page
reservation, prefill-only token budget, separate decode/prefill model calls,
sampling of intermediate chunks, and page-release order. It builds pageable
metadata from Python lists. The C++ treatment includes its persistent pinned
staging, not only a change of language. Both execute the **same real model adapter**.
The adapter owns K/V tensors, and the scheduler alone allocates physical page IDs;
there is no duplicate Python cache allocator or metadata readback in the adapter.

This control is **not** `engine/scheduler/scheduler.py`, which has a global token
budget, preemption, different admission rules, and a combined mixed forward.
Consequently these results isolate the C++ implementation package under matched
semantics; they are not a measured speedup over the current production Python
engine, a serving benchmark, or a comparison with vLLM. A production-backend
comparison remains a separate integration milestone.

All arms use FP16, page16, 12 query/2 KV heads, dimension128, custom RMSNorm/RoPE,
static paged prefill, separate phase forwards, greedy sampling, and no EOS stopping.
Inference is eager throughout: scheduler, model launches, attention allocations,
H2D metadata, sampling, CPU token readback, and request updates are timed. Weight
loading, KV/metadata buffer allocation, compilation preflight, and warmup are
outside timing. The selected attention wrapper currently allocates its partial
buffers on each invocation; this overhead is intentionally included.

The CPU scheduler supplies maximum decode context through
`max_decode_context_length()`. Policy selection never calls `.item()` on GPU
metadata. Outside the sweep's batch/page bounding ranges, selected arms explicitly
fall back to production attention and record that decision. These bounds do not
mean every interior shape was measured. H remains 1 because that is the stage
study's fixed scope; this experiment does not refit head grouping.

## Workloads and timing

Full preset: twelve cases. Eight uniform cohorts use B=1,8,32,128 and prompt
length=512,8192, with 64 output tokens per request. Ragged and staggered cases use
32 requests, maximum prompt length6145, 16 active slots, and 64–66 output tokens.
Their queued requests exercise completion and physical-page reuse. Two additional
saturation cases use 32/128 requests at prompt length8192 and 192/576 output
tokens respectively, keeping early requests alive until the full decode batch
forms. Ordinary long-context cohorts with only 64 outputs may reach only 16
concurrent decode requests; recorded batch histograms distinguish cohort size
from actual decode concurrency. The prefill
budget is 2048 tokens per iteration, so uniform cohorts can also contain mixed
iterations while requests are admitted.

Smoke has four smaller cases with real weights, chunked prompts, queued requests,
and staggered arrivals. Arrivals are declared in **iteration indices**, ensuring
the same scheduled work across arms. This is not a wall-clock arrival or TTFT/SLO
experiment. Synthetic token prompts isolate fixed lengths; validate meaningful
text workloads separately before making serving claims.

Default measurements use five independent prompt-seed trials and three sample
rounds per trial. Every round shuffles the complete variant list. Analysis takes
within-trial medians and bootstraps whole paired trials. Confidence intervals are
pointwise/exploratory, not adjusted for selecting among many workloads. A run with
fewer than three trials reports no interval. Per-step wall times, phase/work
counts, actual dispatch counts, raw full-workload times, and output tokens/s are
saved, along with per-trial GPU telemetry. Full defaults comprise 360 timed scheduler workloads or 720 combined
workloads, **plus** preflight and warmup. Use `plan` and a smoke run to budget GPU
time rather than assuming a short completion time.

## Correctness before timing

- A small chunked-prefill/decode probe compares the adapter against the existing
  dense SDPA model path, with poisoned unused KV storage.
- Every case/trial checks every arm before timing. All callback metadata must
  match exactly. Full logits are compared at the first occurrence of each
  `(phase, token count, sequence count, max query length)` shape (atol=.05,
  rtol=.01). Generated tokens and the full work/completion schedule must match
  exactly across all arms. This is not a claim of checking every logit row at
  every iteration.
- KV pools are poisoned before each arm's preflight to expose unwritten reads.
- Timed outputs and work schedules must still equal preflight. A token divergence
  aborts that trial rather than silently timing different trajectories. If this
  happens, investigate numerical margins in a separate correctness experiment.
- The selected kernel has already passed the stage sweep's independent attention
  checks and held-out evidence gate. This is additional stateful validation.

## Run sequence on H100

First build and check the native extension and adapter:

```bash
make cpp-scheduler-test
python3 -m unittest discover -s experiments/tests -p 'test_scheduler_decode_integration.py' -v
```

1. Measure the scheduler at production decode:

```bash
python3 experiments/integration/benchmark_scheduler_decode.py plan --phase scheduler
python3 experiments/integration/benchmark_scheduler_decode.py run \
  --phase scheduler --preset smoke --trials 1 --samples 1 \
  --output-dir experiments/results/scheduler-model-smoke
python3 experiments/integration/benchmark_scheduler_decode.py run \
  --phase scheduler --output-dir experiments/results/scheduler-model
python3 experiments/integration/benchmark_scheduler_decode.py analyze \
  --output-dir experiments/results/scheduler-model
```

2. Run the existing decode sweep, then select and assess its policy:

```bash
python3 experiments/decode/benchmark_decode_stage_policy.py run \
  --preset smoke --suite all --trials 2 --samples 3 --cache-modes warm \
  --output-dir experiments/results/decode-stage-policy-smoke
python3 experiments/decode/benchmark_decode_stage_policy.py analyze \
  --output-dir experiments/results/decode-stage-policy-smoke
python3 experiments/decode/benchmark_decode_stage_policy.py run \
  --preset full --suite all --output-dir experiments/results/decode-stage-policy
python3 experiments/decode/benchmark_decode_stage_policy.py analyze \
  --output-dir experiments/results/decode-stage-policy
```

Inspect `policy-report.json`: the selected tree comes from train/validation only.
Its status must be `microbenchmark_candidate`. An incomplete study, smoke study,
or failed held-out gate cannot be promoted by this integration harness. A failed
gate means more investigation is required; do not retune using the final holdout.

3. Evaluate the frozen policy and scheduler together:

```bash
python3 experiments/integration/benchmark_scheduler_decode.py run \
  --phase combined --preset smoke --trials 1 --samples 1 \
  --policy-dir experiments/results/decode-stage-policy \
  --output-dir experiments/results/scheduler-decode-smoke
python3 experiments/integration/benchmark_scheduler_decode.py run \
  --phase combined --policy-dir experiments/results/decode-stage-policy \
  --output-dir experiments/results/scheduler-decode-combined
python3 experiments/integration/benchmark_scheduler_decode.py analyze \
  --output-dir experiments/results/scheduler-decode-combined
```

`--case-id` runs one planned case for debugging or targeted completion. Trials are
atomic/resumable; rerunning validates all paired measurements and correctness
checks before skipping completed trials. Use a new directory after
changing source, compiled extension, GPU identity, software, model revision,
policy, or protocol. Do not run concurrent writers in the same directory. The
model Hub revision is resolved and pinned for loading, and recorded in the
manifest. Rebuild the C++ extension after pulling changes. CUDA and real-weight
runs require the GPU environment; local CPU tests do not certify this path.

## Reading the result

For latencies A=python-production, B=cpp-production, C=python-selected,
D=cpp-selected, `report.json` reports paired trial ratios:

- Scheduler: A/B.
- Decode on Python: A/C; decode on C++: B/D.
- Scheduler with selected decode: C/D.
- Combined: A/D.
- Interaction: `(B*C)/(A*D)`. Above1 means combined gain exceeds the product of
  the separately measured gains; below1 means less benefit than that product.

Values above1 indicate speedups except that interaction is a relative interaction
factor. The report also computes these ratios for cumulative decode-only, prefill-only,
and mixed step times where those phases occur. Partially populated cases remain
pending until all paired trials/samples arrive. Inspect each workload, confidence
intervals, dispatch/fallback coverage,
and raw step timings. A whole-workload gain can be diluted by prefill even when
decode-only steps improve. No artifact from this harness is automatically
production-ready, and the program never modifies production dispatch defaults.
