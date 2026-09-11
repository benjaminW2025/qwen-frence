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

C++ version (fast):
```cpp
// One-time allocation
batch_metadata_ = BatchMetadata::allocate(...);

// Every iteration: just write to existing buffer
auto acc = cpu_staging.accessor<int64_t, 1>();
for (int i = 0; i < n; ++i) {
    acc[i] = requests[i]->last_token();
}
batch_metadata_.decode_input_ids.slice(0, 0, n).copy_(cpu_staging);
```

## Your Implementation Tasks

### 1. Prefill Batch Building (iteration_loop.cpp:200-230)
Build the packed ragged format for prefill:
- `prefill_input_ids`: all tokens concatenated
- `prefill_cu_seqlens`: cumulative sequence lengths

### 2. Block Table (iteration_loop.cpp:180)
Fill the block table for paged attention:
- `decode_block_table[i, j]` = physical block ID for request i, logical block j

### 3. Completion Handling (iteration_loop.cpp:290-300)
Move completed requests out of `running_requests_`:
- Free their KV cache blocks
- Store outputs for `pop_completed()`

### 4. Mixed Forward (iteration_loop.cpp:330)
Handle both decode and prefill in one step:
- Concatenate outputs if both present
- Extract correct logits for sampling

## Building

```bash
cd engine/cpp
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
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
