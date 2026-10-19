# Results

The final results are in [summary.json](../experiments/results/final-accepted-8192/summary.json).

- NVIDIA H100 80GB HBM3; Qwen2.5-1.5B; FP16; greedy decoding.
- Batch sizes 8/64, prompt lengths 256/2048, output lengths 128/256.
- One warmup and **three measured runs per cell and engine**; medians reported.
- Local prefill token budget: **8192**. vLLM uses its default budget, which resolves to **3072**, **4096**, or **16384** depending on the workload (see below).
- Throughput speedup = local output tok/s ÷ vLLM output tok/s.
- FA3 attention uses the external vLLM FlashAttention wrapper while scheduling, graph integration, and the retained custom fusions are implemented in this repository

### Complete-workload throughput

Burst submits all requests together. Mixed submits requests in two waves, with the second arriving at a fixed iteration index rather than a wall-clock timestamp. Both timings include prefill and decode. Throughput is generated output tokens per second.

| Batch / prompt / output | Our burst tok/s | vLLM burst tok/s | Burst speedup | Our mixed tok/s | vLLM mixed tok/s | Mixed speedup |
|---|---:|---:|---:|---:|---:|---:|
| 8 / 256 / 128 | 3,415 | 3,097 | 1.103× | 3,281 | 3,253 | 1.009× |
| 8 / 256 / 256 | 3,436 | 3,264 | 1.052× | 3,355 | 3,346 | 1.003× |
| 8 / 2048 / 128 | 2,529 | 2,521 | 1.003× | 2,435 | 2,485 | 0.980× |
| 8 / 2048 / 256 | 2,817 | 2,852 | 0.988× | 2,775 | 2,851 | 0.973× |
| 64 / 256 / 128 | 20,140 | 18,621 | 1.082× | 19,410 | 18,697 | 1.038× |
| 64 / 256 / 256 | 22,132 | 21,026 | 1.053× | 21,642 | 21,287 | 1.017× |
| 64 / 2048 / 128 | 6,883 | 6,986 | 0.985× | 6,483 | 6,612 | 0.981× |
| 64 / 2048 / 256 | 9,645 | 9,923 | 0.972× | 9,395 | 9,474 | 0.992× |

Geometric-mean speedup: **1.029× burst**, **0.999× mixed**.

### Pure decode step latency

[Phase reports](../experiments/results/final-accepted-8192/phases/) measure synchronized pure-decode iterations. Each step generates one token per active request. Step latency is the median within each run, then the median across three runs.

**Local means our engine. Latencies are local / vLLM, in milliseconds. Decode speedup = vLLM ms ÷ local ms.**

| Batch / prompt / output | Decode step ms, local / vLLM | Decode speedup |
|---|---:|---:|
| 8 / 256 / 128 | 2.283 / 2.874 | 1.258× |
| 8 / 256 / 256 | 2.292 / 2.909 | 1.269× |
| 8 / 2048 / 128 | 2.520 / 3.076 | 1.221× |
| 8 / 2048 / 256 | 2.505 / 3.076 | 1.228× |
| 64 / 256 / 128 | 2.551 / 3.492 | 1.369× |
| 64 / 256 / 256 | 2.590 / 3.592 | 1.387× |
| 64 / 2048 / 128 | 3.718 / 4.686 | 1.260× |
| 64 / 2048 / 256 | 3.747 / 4.719 | 1.260× |

Both engines record 127 pure-decode steps for 128-token outputs and 255 for 256-token outputs. Prefill and mixed step timings remain in the raw reports; they are omitted here because the scheduled token counts differ between engines.

### Correctness

Both engines produce the same output-token counts: **110,592 tokens each** across the 16 burst/mixed comparisons.

| Workload | Entire output sequences identical to vLLM |
|---|---:|
| Burst | 285 / 288 requests — 98.96% |
| Mixed | 286 / 288 requests — 99.31% |
| Combined | 571 / 576 requests — 99.13% |

These counts compare complete generated sequences. Equal output counts establish completed work; the five sequence mismatches still need numerical diagnosis. Candidate tests also check same-history logits, KV-cache writes, scheduler plans, and graph replay safety.


## Effective vLLM budgets

| Batch / prompt / output | Token budget |
|---|---:|
| 8 / 256 / 128 | 3072 |
| 8 / 256 / 256 | 4096 |
| 8 / 2048 / 128 | 16384 |
| 8 / 2048 / 256 | 16384 |
| 64 / 256 / 128 | 16384 |
| 64 / 256 / 256 | 16384 |
| 64 / 2048 / 128 | 16384 |
| 64 / 2048 / 256 | 16384 |

## Interpretation

These are warm measurements on eight fixed synthetic workloads. Three repetitions describe this run; they do not establish confidence intervals or general serving performance. Graph buckets are planned from the frozen workloads.

The decode table comes from a separate pass that synchronizes every step and runs vLLM in deterministic in-process step mode. It does not measure the same execution conditions as the offline burst throughput comparison. Pure-decode step counts include changing active populations; they are not necessarily all full-batch steps.

[Reproduce the comparison](reproducing.md) · [Project overview](../README.md)
