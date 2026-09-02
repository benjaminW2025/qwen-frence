# Results: from reference engine to regime-aware execution

This is the curated result narrative. Raw JSON retains configurations, samples,
generated token IDs, and environment metadata; CSV files are the preferred plotting
inputs. `results/manifest.json` provides the same canonical map in machine-readable
form.

## 1. Build the inference engine

The first matched H100 benchmark follows the implementation milestones on one
heterogeneous 16-request burst with concurrency capped at eight:

| Engine milestone | Output tok/s | Speedup over previous |
|---|---:|---:|
| PyTorch reference | 97 | — |
| Paged KV reference | 63 | 0.65x |
| Continuous batching | 389 | 6.14x |
| Bucketed CUDA graphs | 666 | 1.71x |
| Custom kernels | 805 | 1.21x |
| Matched vLLM reference | 1,919 | 2.39x over custom |

The paged-only stage intentionally regressed: paging adds indirection without yet
amortizing model execution across requests. Continuous batching supplied that missing
benefit. Canonical artifact: [`h100-burst-20260824T224748374089Z.json`](benchmarks/results/h100-burst-20260824T224748374089Z.json).

A separate Poisson-arrival workload preserves the continuous-serving result:
[`h100-poisson-20260824T195643307034Z.json`](benchmarks/results/h100-poisson-20260824T195643307034Z.json).

## 2. Pack and resume prefill

Packed ragged prefill concatenates prompt tokens for projection and MLP GEMMs while
sequence offsets isolate causal attention. At batch eight with ramped prompts through
4096 tokens, packing was 1.46x faster end to end than serial admission; replacing the
per-request SDPA loop with variable-length Triton attention added a 1.11x attention
speedup. Token outputs and admission shapes matched.

- Canonical packed-prefill sweep: [`ragged-prefill-20260830T041410084304Z.csv`](experiments/results/ragged-prefill/ragged-prefill-20260830T041410084304Z.csv)
- Fresh-attention tile frontier: [`packed-attention-tiles-20260827T045100937371Z.csv`](experiments/results/packed-attention-tiles/packed-attention-tiles-20260827T045100937371Z.csv)
- Resumed paged-attention frontier: [`paged-prefill-tiles-20260831T182927183128Z-summary.csv`](experiments/results/paged-prefill-tiles/paged-prefill-tiles-20260831T182927183128Z-summary.csv)

The resumed frontier found 6.61% oracle improvement over the static tile. A two-way
rule recovered 6.51%, only 0.091 percentage points behind the oracle.

## 3. Mix decode and prefill under one budget

The mixed executor shares projection, output-projection, and MLP work while launching
specialized decode and prefill attention. The dense execution surface showed that a
representative 32-decode plus 2x2048-prefill iteration rose from 45.7 ms with fresh
prefill to 57.3 ms after a 4096-token prefix.

- Mixed execution surface: [`mixed-surface-20260830T053933170071Z.csv`](experiments/results/mixed-surface/mixed-surface-20260830T053933170071Z.csv)
- Static token-budget sweep: [`token-budget-20260830T043216132491Z.csv`](experiments/results/token-budget/token-budget-20260830T043216132491Z.csv)
- Attention-work budget sweep: [`token-budget-20260831T172914588812Z.csv`](experiments/results/token-budget/token-budget-20260831T172914588812Z.csv)
- Decode-latency distribution: [`work-budget-latency-20260831T180304797246Z-summary.csv`](experiments/results/work-budget-latency/work-budget-latency-20260831T180304797246Z-summary.csv)

The important negative result was stable: restricting prefill attention reduced the
largest individual stalls but created more medium stalls. The uncapped scheduler
maximized SLO-compliant output goodput at every tested 40–150 ms threshold.

## 4. Profile and test focused interventions

The regime atlas attributed the representative fresh mixed iteration to roughly 46%
MLP, 24% decode attention, 13% paged prefill attention, and 11% QKV/RoPE/KV movement.
That evidence motivated isolated decode, fusion, metadata, and overlap experiments.

- Regime atlas: [`atlas-20260831T184645839237Z/`](experiments/results/regime-atlas/atlas-20260831T184645839237Z/)
- Nsight Systems dual-heavy summary: [`nsys-dual-heavy-stats.txt`](experiments/results/profiles/nsys-dual-heavy-stats.txt)
- Complete seven-intervention suite: [`suite-20260831T231615153086Z/`](experiments/results/intervention-suite/suite-20260831T231615153086Z/)
- Dense dispatch fit and holdout: [`suite-20260901T070849420362Z/`](experiments/results/dispatch-policy-suite/suite-20260901T070849420362Z/)
- Grouped-GQA rejection study: [`grouped-gqa-decode-20260831T223906154588Z-summary.csv`](experiments/results/grouped-gqa-decode/grouped-gqa-decode-20260831T223906154588Z-summary.csv)

The accepted executor uses conservative long-context attention dispatch, thresholded
SwiGLU fusion, fused packed RoPE/KV placement, and resumed-prefill tile selection.

## 5. Final external scorecard

The final 2x2x2 H100 scorecard varies concurrency 8/64, prompt length 256/8192, and
output length 32/256. Regime dispatch ranged from small short-context regressions to a
17.96% gain over the static custom engine in the high-concurrency, long-prompt,
long-output cell.

vLLM remained 2.06x–5.26x faster. The smallest gap occurred in the most
prefill-dominated cell; the largest occurred during low-concurrency long generation.
This identifies whole-executor decode launch, graph, fusion, and synchronization
overhead—not additional scheduler policy—as the highest-return future work.

- Aggregate: [`summary.csv`](benchmarks/results/regime-scorecard/suite-20260901T180925475938Z/summary.csv)
- Local manifest: [`manifest-local.json`](benchmarks/results/regime-scorecard/suite-20260901T180925475938Z/manifest-local.json)
- vLLM manifest: [`manifest-vllm.json`](benchmarks/results/regime-scorecard/suite-20260901T180925475938Z/manifest-vllm.json)
- Final correctness: [`final-h100.json`](correctness/results/final-h100.json)

The final external scorecard used one warmup and one measured repetition because of
GPU cost; treat its magnitudes as observed rather than confidence intervals. Isolated
kernel and policy sweeps use larger repetition counts as recorded in their artifacts.
