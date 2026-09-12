# C++ scheduler and decode-policy integration experiment

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
