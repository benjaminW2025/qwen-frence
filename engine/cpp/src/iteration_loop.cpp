/**
 * iteration_loop.cpp - Main C++ iteration loop
 *
 * KEY INSIGHT: The Python scheduler spends 85% of wall time on:
 *   1. Python list operations (building token lists)
 *   2. torch.tensor() calls (creating new tensors each iteration)
 *   3. Kernel launch overhead from Python
 *
 * This C++ version pre-allocates buffers and writes directly to them,
 * eliminating per-iteration allocation overhead.
 */

#include "iteration_loop.hpp"
#include <algorithm>
#include <stdexcept>

namespace inference_engine {

// =============================================================================
// BatchMetadata - Pre-allocated buffers
// =============================================================================

BatchMetadata BatchMetadata::allocate(
    int64_t max_decode_batch,
    int64_t max_prefill_tokens,
    int64_t max_prefill_seqs,
    int64_t max_blocks,
    torch::Device device
) {
    BatchMetadata meta;

    auto opts_long = torch::TensorOptions().dtype(torch::kInt64).device(device);
    auto opts_int = torch::TensorOptions().dtype(torch::kInt32).device(device);

    // Decode buffers
    meta.decode_input_ids = torch::empty({max_decode_batch}, opts_long);
    meta.decode_positions = torch::empty({max_decode_batch}, opts_long);
    meta.decode_slot_mapping = torch::empty({max_decode_batch}, opts_long);
    meta.decode_seq_lens = torch::empty({max_decode_batch}, opts_int);
    meta.decode_block_table = torch::empty({max_decode_batch, max_blocks}, opts_int);

    // Prefill buffers
    meta.prefill_input_ids = torch::empty({max_prefill_tokens}, opts_long);
    meta.prefill_positions = torch::empty({max_prefill_tokens}, opts_long);
    meta.prefill_slot_mapping = torch::empty({max_prefill_tokens}, opts_long);
    meta.prefill_cu_seqlens = torch::empty({max_prefill_seqs + 1}, opts_int);
    meta.prefill_context_lens = torch::empty({max_prefill_seqs}, opts_int);
    meta.prefill_block_table = torch::empty({max_prefill_seqs, max_blocks}, opts_int);

    meta.reset();
    return meta;
}

void BatchMetadata::reset() {
    num_decode_tokens = 0;
    num_prefill_tokens = 0;
    num_prefill_seqs = 0;
    max_prefill_chunk_length = 0;
}

// =============================================================================
// IterationLoop - Constructor
// =============================================================================

IterationLoop::IterationLoop(SchedulerConfig config, torch::Device device)
    : config_(config)
    , device_(device)
{
    // Pre-allocate batch metadata buffers
    int64_t max_blocks = (config_.max_context_length + config_.block_size - 1)
                         / config_.block_size;

    batch_metadata_ = BatchMetadata::allocate(
        config_.max_batch_size,
        config_.max_prefill_tokens_per_iter,
        config_.max_batch_size,  // max prefill seqs
        max_blocks,
        device_
    );

    // Initialize block allocator
    // TODO(you): Calculate total_blocks_ based on GPU memory
    //
    // DESIGN CHOICE: How much memory to reserve for KV cache?
    //   - Need: num_blocks * block_size * num_kv_heads * head_dim * 2 (K+V) * 2 (bytes for fp16)
    //   - H100 80GB: ~70GB usable after model weights
    //   - Each block: 16 * 2 * 128 * 2 * 2 = 16KB
    //   - Max blocks: ~4M blocks (way more than needed)
    //   - Practical limit: max_batch_size * max_context_length / block_size
    //
    total_blocks_ = config_.max_batch_size * max_blocks;
    for (int64_t i = 0; i < total_blocks_; ++i) {
        free_blocks_.push_back(i);
    }
}

// =============================================================================
// Request submission
// =============================================================================

int64_t IterationLoop::submit_request(
    std::vector<int64_t> prompt_ids,
    int64_t max_output_tokens
) {
    auto request = std::make_unique<Request>();
    request->request_id = next_request_id_++;
    request->prompt_ids = std::move(prompt_ids);
    request->output_ids = {};
    request->num_prompt_tokens_computed = 0;
    request->max_output_tokens = max_output_tokens;
    request->block_ids = {}; // NEED THIS FOR BLOCK_TABLE
    request->current_slot = 0;
    request->status = Request::Status::PENDING;

    int64_t id = request->request_id;
    pending_queue_.push(std::move(request));
    return id;
}

// =============================================================================
// Scheduling
// =============================================================================

IterationPlan IterationLoop::schedule() {
    IterationPlan plan;

    // ==========================================================================
    // TODO(you): Implement FCFS scheduling
    //
    // DESIGN CHOICES:
    //   1. Decode-first vs prefill-first?
    //      - Decode-first: lower latency for in-flight requests
    //      - Prefill-first: better throughput (bigger batches)
    //      - We do decode-first to minimize TTFT for decoding requests
    //
    //   2. How to handle mixed batches?
    //      - Run decode + prefill together (current approach)
    //      - Separate passes (simpler but slower)
    //
    //   3. Prefill chunking?
    //      - Long prompts split across iterations
    //      - Prevents decode starvation
    //      - config_.max_prefill_tokens_per_iter controls this
    //
    // ALGORITHM:
    //   1. Add all decoding requests (they only need 1 token each)
    //   2. Fill remaining token budget with prefill work
    //   3. Admit new requests from pending queue if space
    // ==========================================================================

    int64_t prefill_budget = config_.max_prefill_tokens_per_iter;

    // Step 1: Schedule all decode requests
    for (auto& req : running_requests_) {
        if (req->is_prefill_complete()) {
            plan.decode_requests.push_back(req.get());
        }
    }

    // Step 2: Schedule prefill work for running requests
    for (auto& req : running_requests_) {
        if (!req->is_prefill_complete() && prefill_budget > 0) {
            int64_t remaining = req->remaining_prefill();
            int64_t chunk = std::min(remaining, prefill_budget);

            plan.prefill_requests.push_back(req.get());
            plan.prefill_chunk_sizes.push_back(chunk);
            prefill_budget -= chunk;
        }
    }

    // Step 3: Admit new requests from pending queue
    // TODO(you): Implement admission control
    //
    // DESIGN CHOICE: When to admit new requests?
    //   - Always admit if memory available (aggressive)
    //   - Only admit if decode batch < threshold (conservative)
    //   - We go aggressive for throughput
    //
    // HINT: You need to allocate KV cache blocks for new requests
    //       Use allocate_block() and check for std::nullopt

    while (!pending_queue_.empty() && prefill_budget > 0) {
        auto& req = pending_queue_.front();

        // Calculate blocks needed for this request
        int64_t blocks_needed = (req->prompt_ids.size() + req->max_output_tokens
                                 + config_.block_size - 1) / config_.block_size;

        // Try to allocate blocks
        std::vector<int64_t> allocated;
        bool success = true;
        for (int64_t i = 0; i < blocks_needed; ++i) {
            auto block = allocate_block();
            if (!block.has_value()) {
                success = false;
                break;
            }
            allocated.push_back(*block);
        }

        if (!success) {
            // Return allocated blocks and stop admitting
            for (int64_t b : allocated) {
                free_block(b);
            }
            break;
        }

        // Admit request
        req->block_ids = std::move(allocated);
        req->status = Request::Status::PREFILLING;

        int64_t chunk = std::min(static_cast<int64_t>(req->prompt_ids.size()), prefill_budget);
        plan.prefill_requests.push_back(req.get());
        plan.prefill_chunk_sizes.push_back(chunk);
        prefill_budget -= chunk;

        running_requests_.push_back(std::move(pending_queue_.front()));
        pending_queue_.pop();
    }

    return plan;
}

// =============================================================================
// Batch building - THE KEY OPTIMIZATION
// =============================================================================

void IterationLoop::build_batch(const IterationPlan& plan) {
    batch_metadata_.reset();

    // Build decode batch
    if (!plan.decode_requests.empty()) {
        int64_t n = plan.decode_requests.size();
        batch_metadata_.num_decode_tokens = n;

        // Stage on CPU for now (TODO: optimize with pinned memory)
        auto cpu_ids = torch::empty({n}, torch::kInt64);
        auto cpu_pos = torch::empty({n}, torch::kInt64);
        auto cpu_slots = torch::empty({n}, torch::kInt64);
        auto cpu_seq_lens = torch::empty({n}, torch::kInt32);

        auto ids_acc = cpu_ids.accessor<int64_t, 1>();
        auto pos_acc = cpu_pos.accessor<int64_t, 1>();
        auto slots_acc = cpu_slots.accessor<int64_t, 1>();
        auto lens_acc = cpu_seq_lens.accessor<int32_t, 1>();

        for (int64_t i = 0; i < n; ++i) {
            Request* req = plan.decode_requests[i];

            // Last token (either last output or last prompt if no output yet)
            int64_t last_token = req->output_ids.empty()
                ? req->prompt_ids.back()
                : req->output_ids.back();

            ids_acc[i] = last_token;
            pos_acc[i] = req->total_tokens() - 1;
            lens_acc[i] = static_cast<int32_t>(req->total_tokens());

            // Compute slot mapping
            // TODO(you): Implement slot mapping calculation
            //
            // DESIGN: slot = physical_position_in_kv_cache
            //   slot = block_id * block_size + offset_in_block
            //   block_id = req->block_ids[position / block_size]
            //   offset = position % block_size
            //
            int64_t pos = req->total_tokens() - 1;
            int64_t block_idx = pos / config_.block_size;
            int64_t offset = pos % config_.block_size;
            slots_acc[i] = req->block_ids[block_idx] * config_.block_size + offset;
        }

        // Copy to GPU (TODO: use async copy with pinned memory)
        batch_metadata_.decode_input_ids.slice(0, 0, n).copy_(cpu_ids);
        batch_metadata_.decode_positions.slice(0, 0, n).copy_(cpu_pos);
        batch_metadata_.decode_slot_mapping.slice(0, 0, n).copy_(cpu_slots);
        batch_metadata_.decode_seq_lens.slice(0, 0, n).copy_(cpu_seq_lens);

        int64_t max_blocks = batch_metadata_.decode_block_table.size(1); // <- get the max blocks

        auto cpu_block_table = torch::zeros({n, max_blocks}, torch::kInt32); // 2D tensor
        auto block_table_acc = cpu_block_table.accessor<int32_t, 2>(); // Get accessor
        // Now loop through requests
        for (int64_t i = 0; i < n; ++i) {
            Request* req = plan.decode_requests[i]; // Get the request at index i
            for (size_t j = 0; j < req->block_ids.size(); ++j) { // Loop through block_ids
                block_table_acc[i][j] =  static_cast<int32_t>(req->block_ids[j]); // Cast down to int32
            }
        }

        // Copy to GPU
        batch_metadata_.decode_block_table.slice(0, 0, n).copy_(cpu_block_table);
    }

    // Build prefill batch (packed ragged format)
    if (!plan.prefill_requests.empty()) {
        int64_t total_tokens = 0;
        int64_t num_seqs = static_cast<int64_t>(plan.prefill_requests.size());
        int64_t max_blocks = batch_metadata_.prefill_block_table.size(1);

        for (size_t i = 0; i < plan.prefill_chunk_sizes.size(); ++i) {
            // First we should acc
            total_tokens += plan.prefill_chunk_sizes[i];
        }

        // First we want to build a tensor input that is the length of all prefill tokens added up
        auto cpu_ids = torch::empty({total_tokens}, torch::kInt64); // Token ids
        auto cpu_positions = torch::empty({total_tokens}, torch::kInt64); // Absolute positions within sequence
        auto cpu_slots = torch::empty({total_tokens}, torch::kInt64);
        auto cpu_cu_seqlens = torch::empty({num_seqs + 1}, torch::kInt32); // Keeps ending position for each sequence + 0 as starting position
        // Unlike cu_seqlens (chunk boundaries), context_lens contains each
        // request's full visible length: previous prefix + this chunk.
        auto cpu_context_lens = torch::empty({num_seqs}, torch::kInt32);
        auto cpu_block_table = torch::zeros({num_seqs, max_blocks}, torch::kInt32);

        batch_metadata_.num_prefill_tokens = total_tokens;
        batch_metadata_.num_prefill_seqs = num_seqs;

        auto ids_acc = cpu_ids.accessor<int64_t, 1>(); // Accessor for ids
        auto positions_acc = cpu_positions.accessor<int64_t, 1>(); // Accessor for positions
        auto slots_acc = cpu_slots.accessor<int64_t, 1>(); // Accessor for slots
        auto cu_seqlens_acc = cpu_cu_seqlens.accessor<int32_t, 1>(); // Accessor for end points
        auto context_lens_acc = cpu_context_lens.accessor<int32_t, 1>(); // Accessor for context lens
        auto block_table_acc = cpu_block_table.accessor<int32_t, 2>(); // Accessor for block table

        int64_t packed_idx = 0; // Index of packed input
        cu_seqlens_acc[0] = 0; // Set the begin index

        // Loop through every request
        for (size_t i = 0; i < plan.prefill_requests.size(); ++i) {
            // Get pointer to this request
            Request* req = plan.prefill_requests[i];
            int64_t chunk_size = plan.prefill_chunk_sizes[i]; // Get the chunk size for chunk i
            int64_t start = req->num_prompt_tokens_computed; // How many tokens in the prompt have been processed already
            // Update max_prefill_chunk_length
            if (batch_metadata_.max_prefill_chunk_length < chunk_size) {
                batch_metadata_.max_prefill_chunk_length = chunk_size;
            }
            // Now we want to get its prompt_ids
            // Loop through each token in the chunk
            for (int64_t j = 0; j < chunk_size; ++j) {
                // For each token in prompt_ids write it into cpu_ids
                int64_t prompt_idx = start + j;
                ids_acc[packed_idx] = req->prompt_ids[prompt_idx];
                positions_acc[packed_idx] = prompt_idx;

                // Now update slots by
                // 1) Compute the token's logical block
                // 2) Compute the token's offset
                // 3) Get the physical block and compute physical index
                int64_t logical_block = prompt_idx / config_.block_size;
                int64_t block_offset = prompt_idx % config_.block_size;
                int64_t physical_block_id = req->block_ids[logical_block];
                slots_acc[packed_idx] = physical_block_id * config_.block_size + block_offset;

                ++packed_idx; // Increment the token index within the packed input

            }
            // Build block table
            for (size_t j = 0; j < req->block_ids.size(); ++j) {
                block_table_acc[i][j] = static_cast<int32_t>(req->block_ids[j]); // Write i'th request's block ids into i'th row of block_table
            }
            // Update end point
            cu_seqlens_acc[i+1] = static_cast<int32_t>(packed_idx);

            // Update the context length
            context_lens_acc[i] = static_cast<int32_t>(
                req->num_prompt_tokens_computed + chunk_size
            );
        }

        // Copy the active portions of the CPU staging tensors to the
        // pre-allocated device buffers.
        batch_metadata_.prefill_input_ids
            .slice(0, 0, total_tokens)
            .copy_(cpu_ids);
        batch_metadata_.prefill_positions
            .slice(0, 0, total_tokens)
            .copy_(cpu_positions);
        batch_metadata_.prefill_slot_mapping
            .slice(0, 0, total_tokens)
            .copy_(cpu_slots);
        batch_metadata_.prefill_cu_seqlens
            .slice(0, 0, num_seqs + 1)
            .copy_(cpu_cu_seqlens);
        batch_metadata_.prefill_context_lens
            .slice(0, 0, num_seqs)
            .copy_(cpu_context_lens);
        batch_metadata_.prefill_block_table
            .slice(0, 0, num_seqs)
            .copy_(cpu_block_table);
    }
}

// =============================================================================
// Sampling
// =============================================================================

torch::Tensor IterationLoop::sample(torch::Tensor logits) {
    // ==========================================================================
    // TODO(you): Implement sampling
    //
    // DESIGN CHOICES:
    //   1. Greedy (argmax) - simplest, deterministic
    //   2. Temperature sampling - more diverse
    //   3. Top-k / Top-p - controlled diversity
    //
    // For now, implement greedy:
    //   next_tokens = logits.argmax(dim=-1)
    //
    // logits shape: (num_decode + num_prefill_seqs, vocab_size)
    // We only care about:
    //   - Last token of each prefill sequence
    //   - Each decode token
    //
    // HINT: For prefill, you need to extract the last logit of each sequence
    //       using cu_seqlens to find the boundaries.
    // ==========================================================================

    // Simple greedy for now
    return logits.argmax(-1);
}

// =============================================================================
// Update requests after sampling
// =============================================================================

void IterationLoop::update_requests(
    const IterationPlan& plan,
    torch::Tensor next_tokens
) {
    // ==========================================================================
    // TODO(you): Update request state after sampling
    //
    // For decode requests:
    //   1. Append next_token to output_ids
    //   2. Check for EOS or max_output_tokens
    //   3. Mark complete if done
    //
    // For prefill requests:
    //   1. Update num_prompt_tokens_computed += chunk_size
    //   2. If prefill complete, append next_token and switch to DECODING
    //   3. If prefill not complete, continue next iteration
    //
    // HINT: next_tokens is on GPU, use .item<int64_t>() to get scalar
    //       or copy to CPU first for batch access
    // ==========================================================================

    auto cpu_tokens = next_tokens.to(torch::kCPU);
    auto tokens_acc = cpu_tokens.accessor<int64_t, 1>();

    int64_t token_idx = 0;

    // Update decode requests
    for (Request* req : plan.decode_requests) {
        int64_t next_token = tokens_acc[token_idx++];
        req->output_ids.push_back(next_token);

        // Check completion
        // TODO(you): Add EOS token check
        if (static_cast<int64_t>(req->output_ids.size()) >= req->max_output_tokens) {
            req->status = Request::Status::COMPLETED;
        }
    }

    // Update prefill requests
    for (size_t i = 0; i < plan.prefill_requests.size(); ++i) {
        Request* req = plan.prefill_requests[i];
        int64_t chunk = plan.prefill_chunk_sizes[i];

        req->num_prompt_tokens_computed += chunk;

        if (req->is_prefill_complete()) {
            // Prefill done, get the next token
            int64_t next_token = tokens_acc[token_idx++];
            req->output_ids.push_back(next_token);
            req->status = Request::Status::DECODING;
        }
    }

    // Move completed requests to output
    // TODO(you): Implement this
    //
    // HINT: Use std::remove_if with erase idiom
    //   running_requests_.erase(
    //       std::remove_if(running_requests_.begin(), running_requests_.end(),
    //           [this](auto& req) {
    //               if (req->status == Request::Status::COMPLETED) {
    //                   completed_outputs_.emplace_back(req->request_id, req->output_ids);
    //                   // Free blocks
    //                   for (int64_t b : req->block_ids) free_block(b);
    //                   return true;
    //               }
    //               return false;
    //           }),
    //       running_requests_.end()
    //   );
}

// =============================================================================
// Main step function
// =============================================================================

int64_t IterationLoop::step(
    const std::function<torch::Tensor(
        torch::Tensor, torch::Tensor, torch::Tensor,
        torch::Tensor, torch::Tensor, torch::Tensor,
        int64_t, bool
    )>& forward_fn
) {
    // Schedule
    IterationPlan plan = schedule();
    if (plan.empty()) {
        return 0;
    }

    // Build batch (writes to pre-allocated buffers)
    build_batch(plan);

    torch::Tensor logits;

    if (batch_metadata_.num_decode_tokens > 0) {
        int64_t n = batch_metadata_.num_decode_tokens;
        auto decode_logits = forward_fn(
            batch_metadata_.decode_input_ids.slice(0, 0, n),
            batch_metadata_.decode_positions.slice(0, 0, n),
            batch_metadata_.decode_slot_mapping.slice(0, 0, n),
            // Decode has no packed-query boundaries. This zero-length view is a
            // real device tensor, which is cleaner to pass through pybind than
            // an undefined torch::Tensor().
            batch_metadata_.decode_seq_lens.slice(0, 0, 0),
            batch_metadata_.decode_seq_lens.slice(0, 0, n),
            batch_metadata_.decode_block_table.slice(0, 0, n),
            /*max_query_length=*/1,
            /*is_decode=*/true
        );
        logits = decode_logits;
    }

    if (batch_metadata_.num_prefill_tokens > 0) {
        int64_t num_tokens = batch_metadata_.num_prefill_tokens;
        int64_t num_seqs = batch_metadata_.num_prefill_seqs;

        auto prefill_logits = forward_fn(
            batch_metadata_.prefill_input_ids.slice(0, 0, num_tokens),
            batch_metadata_.prefill_positions.slice(0, 0, num_tokens),
            batch_metadata_.prefill_slot_mapping.slice(0, 0, num_tokens),
            batch_metadata_.prefill_cu_seqlens.slice(0, 0, num_seqs + 1),
            batch_metadata_.prefill_context_lens.slice(0, 0, num_seqs),
            batch_metadata_.prefill_block_table.slice(0, 0, num_seqs),
            batch_metadata_.max_prefill_chunk_length,
            /*is_decode=*/false
        );

        if (logits.defined()) {
            logits = torch::cat({logits, prefill_logits}, 0);
        } else {
            logits = prefill_logits;
        }
    }

    // Sample
    torch::Tensor next_tokens = sample(logits);

    // Update state
    int64_t prev_completed = completed_outputs_.size();
    update_requests(plan, next_tokens);

    return completed_outputs_.size() - prev_completed;
}

// =============================================================================
// Completed outputs
// =============================================================================

std::vector<std::pair<int64_t, std::vector<int64_t>>> IterationLoop::pop_completed() {
    auto result = std::move(completed_outputs_);
    completed_outputs_.clear();
    return result;
}

// =============================================================================
// Block allocation (simple free list)
// =============================================================================

std::optional<int64_t> IterationLoop::allocate_block() {
    if (free_blocks_.empty()) {
        return std::nullopt;
    }
    int64_t block = free_blocks_.back();
    free_blocks_.pop_back();
    return block;
}

void IterationLoop::free_block(int64_t block_id) {
    free_blocks_.push_back(block_id);
}

}  // namespace inference_engine
