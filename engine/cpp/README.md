# C++ Iteration Loop

This module eliminates Python overhead in the inference hot path.

## Why C++?

From profiling (`experiments/results/regime-atlas/.../launch_bound`):
```
CUDA time: 6.6ms
Wall time: 45ms
```

Python overhead is **85% of execution time** in short-context regimes.

## Architecture

```
Python (cold path)                    C++ (hot path)
─────────────────                    ───────────────
submit_request() ──────────────────► pending_queue_
                                           │
                                           ▼
                                     ┌───────────┐
                                     │ schedule()│ ◄─── FCFS + admission
                                     └─────┬─────┘
                                           │
                                           ▼
                                    ┌────────────┐
                                    │build_batch()│ ◄─── Write to pre-allocated buffers
                                    └──────┬─────┘
                                           │
forward_fn() ◄─────────────────────────────┤ (callback into PyTorch)
                                           │
                                           ▼
                                     ┌──────────┐
                                     │ sample() │
                                     └────┬─────┘
                                           │
                                           ▼
                                   ┌──────────────┐
                                   │update_state()│
                                   └──────┬───────┘
                                           │
pop_completed() ◄──────────────────────────┘
```

## Key Optimization: Pre-allocated Buffers

Python version (slow):
```python
# Every iteration allocates new tensors
tokens = []
for req in decode_requests:
    tokens.append(req.last_token())
input_ids = torch.tensor(tokens, device='cuda')  # Allocation + H2D copy
```

The C++ loop allocates device metadata and matching CPU staging tensors once.
For CUDA, staging uses `TensorOptions().device(torch::kCPU).pinned_memory(true)`.
`build_batch()` writes directly into that storage; no temporary pageable metadata
arrays are created each iteration. CPU execution uses ordinary reusable buffers.

For each CUDA step:

1. Wait for the previous H2D copy before rewriting its pinned source buffer.
2. Build all metadata synchronously on the calling CPU thread.
3. On a persistent copy stream, wait for prior device-buffer consumers and copy
   only active rows with `copy_(source, /*non_blocking=*/true)`.
4. Record a copy-complete event and make the caller's current compute stream wait
   for it before invoking `forward_fn`.
5. Record completion of consumers before the next device-buffer reuse. Callback
   exceptions drain both streams; destruction waits before freeing buffers.

The stream wrappers restore the caller's stream/device. `forward_fn` must consume
metadata on the current stream, or join any side-stream consumers back to it
before returning. Its arguments are temporary views into reused storage: clone
anything that must persist beyond the step. Call the loop from one thread; it is
not reentrant. Capturing the entire `step()` in a CUDA graph is unsupported because
it schedules and reads sampled tokens on the CPU.

This implements pinned asynchronous **transfers**, while preserving the existing
synchronous `step()` API. It still reads sampled tokens back before scheduling the
next iteration. Consequently, it does **not** yet overlap next-batch construction
with the current forward. One buffer set is sufficient under this API, and retains
stable device addresses. Cross-iteration overlap requires a separate scheduling
change and multiple buffer sets; no throughput improvement is assumed here.

## Implementation Status

- [x] FCFS scheduling and admission limits
- [x] Decode metadata construction
- [x] Packed-paged prefill metadata construction
- [x] Separate decode/prefill forward calls and mixed-logit ordering
- [x] Greedy batched sampling
- [x] EOS/output-limit completion and block reclamation
- [x] Request, configuration, and callback-logit validation
- [x] Reusable pinned CPU metadata, asynchronous H2D copies, and stream/event ordering
- [ ] Integrate a real model/KV-pool `forward_fn`
- [ ] Size the physical block pool from actual KV-cache memory

## Build and test

The setuptools path uses PyTorch's extension tooling without requiring Ninja:

```bash
make cpp-scheduler-build
make cpp-scheduler-test
```

The behavioral suite runs on CPU and exercises the real compiled extension. It
checks packed/chunked metadata, mixed decode/prefill logit ordering, EOS cleanup,
invalid-input handling, and exact deterministic logit/output parity with the
Python scheduler. On CUDA machines the same command also exercises metadata
parity across alternating streams, delayed consumers and destruction, callback
failure recovery, and (with two GPUs) device restoration. CUDA tests skip on CPU
machines. Run `make cpp-scheduler-test` on the H100 before benchmarking this path.

Run the scheduler-overhead comparison with:

```bash
make cpp-scheduler-benchmark
```

Or select a device and workload explicitly:

```bash
python3 experiments/scheduler/benchmark_cpp_scheduler.py \
    --device cuda \
    --num-requests 8 \
    --prompt-length 128 \
    --output-length 16
```

This benchmark deliberately replaces transformer execution with identical
deterministic logits. It measures scheduler/control-plane overhead and performs
an exact-logit correctness preflight; it is not a model-throughput benchmark.

## Alternative CMake build

```bash
cd engine/cpp
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')"
make -j
```

## Testing

```python
import torch
import sys
sys.path.insert(0, 'engine/cpp/build')
import inference_engine_cpp as cpp

config = cpp.SchedulerConfig()
config.max_batch_size = 64
config.block_size = 16

loop = cpp.IterationLoop(config, torch.device('cuda'))

# Submit request
req_id = loop.submit_request([1, 2, 3, 4, 5], max_output_tokens=10)

# Dummy forward function
def forward_fn(
    input_ids,
    positions,
    slot_mapping,
    cu_seqlens,
    context_lens,
    block_table,
    max_query_length,
    is_decode,
):
    # The callback returns one sampling row per sequence. During prefill,
    # input_ids.shape[0] is the number of packed tokens, not the batch size.
    batch_size = context_lens.shape[0]
    return torch.randn(batch_size, 32000, device='cuda')  # fake logits

# Run
while loop.num_running() > 0 or loop.num_pending() > 0:
    loop.step(forward_fn)

# Get outputs
for req_id, output_ids in loop.pop_completed():
    print(f"Request {req_id}: {output_ids}")
```

## KV-cache metadata ownership

This teaching implementation leaves the actual K/V pools behind `forward_fn`.
`IterationLoop` acts as a small block manager: its free-list allocates physical
page IDs, and each `Request::block_ids` vector persistently records that
request's logical-to-physical page mapping.

`BatchMetadata::{decode,prefill}_block_table` do not own page allocations. They
are temporary padded GPU views assembled in current batch order for the paged
attention kernels.

Packed-paged prefill has three distinct pieces of sequence metadata:

- `cu_seqlens`: boundaries of current query chunks in the packed token tensor.
- `context_lens`: visible lengths after adding the current chunks.
- `block_table`: locations of those complete contexts in the paged K/V pools.

For decode, `cu_seqlens` is an empty tensor, every query has length one, and
`context_lens` is the existing `decode_seq_lens` tensor.

## Performance Checklist

- [ ] Pre-allocated GPU buffers (done)
- [ ] Pre-allocated CPU staging buffers
- [ ] Pinned memory for async H2D
- [ ] Double buffering (prepare N+1 while N runs)
- [ ] Minimize host-device syncs
