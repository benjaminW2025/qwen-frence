# v0.1.1 Plan

Goal: Close the 2-5x gap with vLLM.

## Diagnosis

Two distinct bottlenecks depending on regime:

| Regime | Bottleneck | Evidence |
|--------|------------|----------|
| Long context, decode-heavy | Decode attention occupancy | 90% of CUDA time, sequential KV loop |
| Short context, any batch | Python/launch overhead | 6.6ms CUDA vs 45ms wall time |

## Work streams

### 1. Split-K decode attention

**Why:** Parallelize across KV blocks, not just batch×heads. Fill SMs.

**Approach:**
```
current_programs = B × 12
target_programs = 456 SMs × 4 warps = 1824
K = ceil(target / current)
K = min(K, num_kv_blocks / 16)  # min 16 blocks per chunk
```

**Experiment:**
- Implement split-K paged decode kernel
- Sweep K = {1, 2, 4, 8, 16, auto} across regime grid
- Measure: kernel latency, memory bandwidth, reduction overhead
- Compare: production kernel, vLLM decode

**Success:** >2x speedup on decode_bandwidth regime (D=64, C=8192).

### 2. C++ scheduler

**Why:** Python dispatch is 85% of wall time in short-context regimes.

**Approach:**
- Move hot path to C++ extension
- Keep Python for request management, config
- Eliminate per-iteration Python calls in forward pass

**Scope:**
- Phase 1: C++ iteration loop (batch assembly, forward call, sampling)
- Phase 2: C++ request queue management
- Phase 3: Async Python interface

**Experiment:**
- Implement minimal C++ forward loop
- Benchmark launch_bound regime before/after
- Measure: wall time, kernel launch count, CPU utilization

**Success:** >3x speedup on launch_bound regime (D=8, C=128).

### 3. Kernel fusion

**Why:** 434 kernel launches per iteration. Each has overhead + memory round-trip.

**Candidates (by ROI):**

| Fusion | Current | Fused | Expected gain |
|--------|---------|-------|---------------|
| QKV + RoPE + KV write | 3 kernels | 1 kernel | 20-30% of attention region |
| Gate + Up + SiLU + Mul | 4 kernels | 1 kernel | 10-15% of MLP region |
| RMSNorm + QKV proj | 2 kernels | 1 kernel | Minor |

**Approach:**
- Enable existing fusions by default (currently opt-in)
- Make fusions CUDA-graph compatible
- Measure impact per-fusion

**Experiment:**
- Baseline: all fusions off
- Test: enable each fusion independently
- Measure: kernel count, wall time, memory traffic

**Success:** <200 kernel launches per iteration, 10%+ end-to-end speedup.

## Sequence

```
Week 1: Split-K decode attention
  - Implement kernel
  - Run experiment sweep
  - Ship if >2x on target regime

Week 2: Enable fusions + graph compatibility
  - Fix graph conflicts with regime fusions
  - Enable by default
  - Validate correctness

Week 3-4: C++ scheduler
  - Implement C++ forward loop
  - Benchmark
  - Integrate with existing Python API
```

## Non-goals (for now)

- Pure CUDA kernels (Triton is fine for iteration speed)
- Custom GEMM kernels (cuBLAS is fine)
- Speculative decoding
- Tensor parallelism

## Metrics

Track across regime grid (2×2×2 sweep):

| Metric | Current | Target |
|--------|---------|--------|
| Output tok/s (decode-heavy) | 142-666 | 400-1500 |
| Output tok/s (prefill-heavy) | 563-3432 | 1500-8000 |
| Gap vs vLLM | 2.06-5.26x | <1.5x |
| Kernel launches/iter | 434 | <200 |
| GPU utilization (short ctx) | ~15% | >50% |

## Dependencies

- Triton 3.x for split-K reduction primitives
- pybind11 or nanobind for C++ scheduler
- CUDA graphs API for fused kernel capture
