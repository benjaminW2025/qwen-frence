#pragma once

#include <torch/torch.h>
#include <vector>
#include <queue>
#include <memory>
#include <optional>
#include <functional>

namespace inference_engine {

// Forward declarations
struct Request;
struct IterationPlan;
struct BatchMetadata;

// Request state mirrors Python RequestState
struct Request {
    int64_t request_id;
    std::vector<int64_t> prompt_ids;
    std::vector<int64_t> output_ids;
    int64_t num_prompt_tokens_computed;
    int64_t max_output_tokens;

    // Persistent logical -> physical page mapping for this request.
    //
    // The Python engine keeps this mapping in PagedKVCache.block_tables. This
    // teaching implementation does not own the K/V pools (forward_fn does), so
    // it keeps the mapping on the request and manages page IDs with the simple
    // free-list below. BatchMetadata only contains temporary, padded GPU views
    // of these per-request mappings.
    std::vector<int64_t> block_ids;
    int64_t current_slot;

    enum class Status {
        PENDING,
        PREFILLING,
        DECODING,
        COMPLETED,
        PREEMPTED
    };
    Status status;

    // Computed properties
    int64_t total_tokens() const {
        return prompt_ids.size() + output_ids.size();
    }
    int64_t remaining_prefill() const {
        return static_cast<int64_t>(prompt_ids.size()) - num_prompt_tokens_computed;
    }
    bool is_prefill_complete() const {
        return num_prompt_tokens_computed >= static_cast<int64_t>(prompt_ids.size());
    }
};

// Pre-allocated batch metadata buffers (eliminates per-iteration allocation)
struct BatchMetadata {
    // Decode tokens
    torch::Tensor decode_input_ids;      // (max_decode_batch,)
    torch::Tensor decode_positions;       // (max_decode_batch,)
    torch::Tensor decode_slot_mapping;    // (max_decode_batch,)
    torch::Tensor decode_seq_lens;        // (max_decode_batch,)
    torch::Tensor decode_block_table;     // (max_decode_batch, max_blocks)

    // Prefill tokens (packed ragged)
    torch::Tensor prefill_input_ids;      // (max_prefill_tokens,)
    torch::Tensor prefill_positions;      // (max_prefill_tokens,)
    torch::Tensor prefill_slot_mapping;   // (max_prefill_tokens,)
    torch::Tensor prefill_cu_seqlens;     // packed-query boundaries: (max_prefill_seqs + 1,)
    torch::Tensor prefill_context_lens;   // prefix + current chunk: (max_prefill_seqs,)
    torch::Tensor prefill_block_table;    // temporary padded view: (max_prefill_seqs, max_blocks)

    // Current batch sizes (updated each iteration)
    int64_t num_decode_tokens;
    int64_t num_prefill_tokens;
    int64_t num_prefill_seqs;
    int64_t max_prefill_chunk_length;

    // Pre-allocate buffers
    static BatchMetadata allocate(
        int64_t max_decode_batch,
        int64_t max_prefill_tokens,
        int64_t max_prefill_seqs,
        int64_t max_blocks,
        torch::Device device,
        bool pinned_memory = false
    );

    // Reset for new iteration (just reset counts, don't reallocate)
    void reset();
};

// Iteration plan built by scheduler
struct IterationPlan {
    std::vector<Request*> decode_requests;
    std::vector<Request*> prefill_requests;
    std::vector<int64_t> prefill_chunk_sizes;  // tokens per prefill request this iter

    bool empty() const {
        return decode_requests.empty() && prefill_requests.empty();
    }
};

// Configuration
struct SchedulerConfig {
    int64_t eos_token_id = -1;
    int64_t max_batch_size = 256;
    int64_t max_prefill_tokens_per_iter = 2048;
    int64_t max_context_length = 32768;
    int64_t block_size = 16;
    int64_t num_kv_heads = 2;
    int64_t head_dim = 128;
};

// The main C++ iteration loop
class IterationLoop {
public:
    IterationLoop(
        SchedulerConfig config,
        torch::Device device
    );

    ~IterationLoop();

    // Submit new request (called from Python)
    int64_t submit_request(
        std::vector<int64_t> prompt_ids,
        int64_t max_output_tokens
    );

    // Run one iteration: schedule → build batch → forward → sample → update.
    // Single-threaded. Callback views are reused next step; consume on the
    // current stream (or join side streams before returning). CUDA transfers
    // are asynchronous, but step waits for sampled tokens to update requests.
    // Returns number of completed requests
    int64_t step(
        const std::function<torch::Tensor(
            torch::Tensor,  // input_ids
            torch::Tensor,  // positions
            torch::Tensor,  // slot_mapping
            torch::Tensor,  // cu_seqlens (empty for decode)
            torch::Tensor,  // context_lens
            torch::Tensor,  // block_table
            int64_t,        // max query length (1 for decode)
            bool            // is_decode
        )>& forward_fn
    );

    // Get completed request outputs
    std::vector<std::pair<int64_t, std::vector<int64_t>>> pop_completed();

    // Stats
    // Host-side dispatch feature; never reads a CUDA tensor.
    int64_t max_decode_context_length() const { return max_decode_context_length_; }
    int64_t num_pending() const { return pending_queue_.size(); }
    int64_t num_running() const { return running_requests_.size(); }

private:
    SchedulerConfig config_;
    torch::Device device_;
    BatchMetadata batch_metadata_;
    BatchMetadata host_metadata_;
    struct MetadataTransfer;
    std::unique_ptr<MetadataTransfer> metadata_transfer_;

    // Request management
    std::queue<std::unique_ptr<Request>> pending_queue_;
    std::vector<std::unique_ptr<Request>> running_requests_;
    std::vector<std::pair<int64_t, std::vector<int64_t>>> completed_outputs_;

    int64_t next_request_id_ = 0;
    int64_t max_decode_context_length_ = 0;

    // Simplified block manager. It owns page-ID allocation, but not the actual
    // K/V tensors; those remain behind forward_fn in this prototype.
    std::vector<int64_t> free_blocks_;
    int64_t total_blocks_;

    // Internal methods
    IterationPlan schedule();
    void build_batch(const IterationPlan& plan);
    void copy_batch();
    torch::Tensor sample(torch::Tensor logits);
    void update_requests(const IterationPlan& plan, torch::Tensor next_tokens);

    // Block allocation
    std::optional<int64_t> allocate_block();
    void free_block(int64_t block_id);
};

}  // namespace inference_engine
