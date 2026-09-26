# Independent Hopper attention candidate

**Uncompiled / unvalidated on Hopper; no performance claim.** Do not treat CPU
unit tests, the FA3-style design, or successful import as CUDA qualification.

Scope: SM90a, FP16 Qwen2.5-1.5B, 12 query / 2 KV heads, head dimension 128,
page size 16. Supports paged decode and packed variable-length bottom-right
causal attention, including mixed decode/prefill. Unsupported types fail;
there is no external-attention fallback.

The attention implementation is project-owned. NVIDIA CUTLASS v3.9.2 supplies
CuTe layouts and WGMMA instruction wrappers, not an attention implementation.
Algorithm reference: [FlashAttention-3](https://tridao.me/publications/flash3/flash3.pdf).
It uses double-buffered TMA, warp specialization and asynchronous WGMMA,
overlapping next-tile QK with softmax. These features alone do not establish
performance parity with FA3. The initial split schedule is not tuned.

## Qualification gates

1. Build with CUDA 12.8+ and `CUTLASS_PATH` pointing at CUTLASS v3.9.2.
   From `custom_kernels/hopper_attention`, run
   `python setup.py build_ext --inplace`, then return to the repository root.
2. Run `correctness/checks/check_flash_decode.py`: independent FP32 oracle,
   poisoned KV padding, empty partitions, ragged causal masks, strided Q and
   graph replay after changing all address-backed inputs. Run under Compute
   Sanitizer memcheck, racecheck and synccheck before performance qualification.
3. Run `experiments/decode/qualify_flash_decode.py run --output-dir <fresh-dir>
   --vllm-python <separate-vllm-0.30.0-python>` with the local interpreter.
   Shared fixtures cover B8/B64 decode/mixed and resumed prefill at C256/2048/4096,
   plus fresh-prefill chunks up to an 8192-token total; report both
   warm/cold replay medians and all samples. Default parity gate is <=1.05x
   FA3 latency in **every** case, not an average. This is a microbenchmark gate.
4. Run integrated same-history full-logit checks, direct live-KV comparisons,
   token checks and the complete eight-cell burst/mixed/phase suite. The
   microbenchmark explicitly reports `full_model_qualified=false`; it does
   not perform or substitute for these checks.

None of these GPU gates has passed yet. The current eight-cell scripts have
candidate routing and separate reference interpreters wired; older experimental
FA3 arms remain explicitly external references. A complete historical experiment
rerun/migration is still pending. Do not reuse old scorecards for this candidate.

## Optimization audit / next experiments

The current eight-cell adapter selects this backend for decode, pure prefill,
and mixed; pure/mixed prefill eager fallbacks also retain this selection.
Historical oracle arms are not renamed or silently converted. The optional
whole-forward mixed-graph prototype is rejected for this backend until migrated.

| Intervention | Current candidate | Next controlled comparison |
|---|---|---|
| FP16 / D128 / 12:2 heads / page16 specialization | Compiled constants | Keep identical math and precision |
| TMA + producer/consumer WGMMA/softmax overlap | Written, unvalidated | Validate before measuring |
| Causal tile skipping | Written, unvalidated | Fresh and resumed prefill, including partial tiles |
| Register-fed PV instead of shared-memory P | Written, unvalidated; shared-P control retained | Avoid P shared-memory traffic; measure register pressure and latency |
| Regime-specific tile configuration | M64/N64 and M64/N128 written; fixed warp count | Larger KV tile amortizes work but uses more shared memory/registers |
| Packed mixed work scheduling | GPU-built compact query worklist written | Reduce rectangular-grid empty CTAs; include construction cost |
| Split-K schedule | Initial decode heuristic; varlen K1 | Tune B8/B64 and context, including partial/reduction traffic |
| Small-page loading | Page-wise TMA | Compare vector asynchronous loads; TMA is not automatically best |
| Metadata/descriptors shared across layers | Optional prepared query-worklist API; not integrated | Rebuild when query offsets change; TMA descriptor reuse still pending |

vLLM 0.30.0 pins its attention fork at
`506341a143fcabd4bb79052a7605ada727d6b3f5`. Its
[Hopper tile selector](https://github.com/vllm-project/flash-attention/blob/506341a143fcabd4bb79052a7605ada727d6b3f5/hopper/tile_size.h)
already specializes by head dimension, causal/local/paged mode and warp-group
choice, including register-fed PV for FP16 D128. Merely fixing the model shape
is not a demonstrated advantage over that implementation. The actual winning
launch configuration must be recorded from execution, not inferred from source.

## One bounded workload-tuning run

Use the local-engine interpreter, an existing CUTLASS v3.9.2 checkout in
`CUTLASS_PATH`, and a separate vLLM 0.30.0 interpreter. No packages or models are
downloaded by this command. CUDA compilation and GPU behavior are not verified
on the development machine; an H100 run must first establish those gates.

```bash
python experiments/decode/tune_hopper_workloads.py run \
  --vllm-python /root/vllm-current-env/bin/python \
  --output-dir experiments/results/hopper-workload-tuning-v2 \
  --build --max-seconds 600
```

`plan` instead of `run` exercises the real C++ scheduler on CPU only. The frozen
eight burst/staggered workloads currently reduce to 38 representative attention
shapes: prefill/mixed highest-work steps, and decode beginning, last-full-cohort,
and tail steps. Identical shapes share measurements. This is not exhaustive
coverage of every context length or a full-engine throughput benchmark.

The staged search tunes splits/overlap, compares 64/128 KV tiles and shared/register
PV (plus compact scheduling for mixed), then retunes splits/overlap for the chosen
architecture. It is bounded, not an exhaustive search or proof of optimality.
Held-out input seeds measure the baseline, tuned control, each single intervention,
selected combination, and vLLM FA3 with warm/cold cache samples. Compact-worklist
construction is inside the timed graph. Compiler register/local-memory attributes
and shared-memory requirements are recorded, not presented as measured occupancy.
FP16 attention checks use atol=rtol=0.002 against dense FP32 attention. All 32
tile/PV/scheduling/overlap/split test combinations also undergo ragged correctness
checks and graph replay after metadata changes before tuning.

The 600-second subprocess budget includes optional compilation, not CPU planning.
Smoke workers are additionally limited to 90 seconds. A timeout terminates the
owned process group, saves completed cases, and returns; it does **not** stop pod
billing. Repeat the same command/settings to resume. Source/environment changes
require a fresh output directory. Failed correctness cases stop further spending
and cannot be silently resumed. Fixtures are generated reproducibly in memory,
not stored as gigabytes on the network volume.

Outputs: `plan.json`, per-case raw tuning/evaluation samples, `progress.json`,
`summary.json`, and a **disabled** `candidate-dispatch.json`. No production
dispatch is changed, and kernel parity never implies full-model qualification.
