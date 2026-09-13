# Qwen projection hotspot profile

The projection profiler measures the real Qwen2.5-1.5B weights at decode
shapes (one hidden-state row per active request).  It separates Q/K/V, MLP
gate/up, SwiGLU, MLP down, vocabulary logits, and logits plus greedy argmax.
It also compares separate QKV and gate/up GEMMs with a candidate packed-weight
layout.  Weight packing is outside the measured region.

Run CUDA-event measurements first:

```bash
python experiments/model/profile_qwen_projection_hotspots.py bench \
  --batches 1,2,4,8,16,32,64,96,128,256 --warmups 20 --repetitions 100
```

Use Nsight Systems after that to see launch gaps, stream serialization, and
which GEMMs dominate a realistic decode batch.  This captures only the marked
target region, after warmup:

```bash
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --stop-on-range-end=true \
  -o experiments/results/model-projections/nsys-b32 \
  python experiments/model/profile_qwen_projection_hotspots.py trace \
    --batches 1,2,4,8,16,32,64,96,128,256 --trace-batch 32 --trace-repetitions 20 \
    --cuda-profiler-range
```

Read the named NVTX ranges as follows:

- `qkv_separate` vs `qkv_packed`: whether combining the three attention input
  projections saves launches and improves GEMM shape efficiency.
- `mlp_gate_up_separate` vs `mlp_gate_up_packed`: the same comparison for the
  two SwiGLU inputs. `mlp_swiglu` and `mlp_down` show whether those projections
  are material enough to change the full MLP path.
- `lm_head_logits` and `lm_head_logits_argmax`: the vocabulary projection alone
  and its end-to-end greedy sampling cost.

This is a projection experiment, not an end-to-end vLLM comparison.  Use it to
choose which fusion candidate merits wiring into the model, then measure the
result in the scheduler/decode integration benchmark.

## Real decode ablation

`benchmark_packed_projection_decode.py` compares the separate layout, QKV-only
packing, and QKV plus gate/up packing through the full eager 28-layer decode
forward. It checks logits and greedy tokens, then measures paired CUDA-event
latency with warm and L2-evicted KV cache. The scheduler, model loading, weight
packing, and cache staging are outside the timed interval.

```bash
python experiments/model/benchmark_packed_projection_decode.py \
  --preset smoke --output-dir experiments/results/packed-projection-decode-smoke
python experiments/model/benchmark_packed_projection_decode.py \
  --preset full --output-dir experiments/results/packed-projection-decode-v1
```

Each completed batch/context cell is saved under `trials/` before the next cell
starts. Repeating the same command resumes those validated cells. The aggregate
`decode-ablation-results.json` is written after all cells finish. Use a new
output directory if the workload or source changes.
