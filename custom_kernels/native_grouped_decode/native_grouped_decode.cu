#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>

namespace {

constexpr int kHeadDim = 128;
constexpr int kPageSize = 16;
constexpr int kGroup = 6;
constexpr int kWarpSize = 32;
constexpr int kThreads = kGroup * kWarpSize;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return __shfl_sync(0xffffffffu, value, 0);
}

template <typename scalar_t, typename index_t>
__global__ void grouped_gqa_splitk_partial_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k_pool,
    const scalar_t* __restrict__ v_pool,
    const index_t* __restrict__ block_table,
    const index_t* __restrict__ seq_lens,
    float* __restrict__ partial_out,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    int64_t q_stride_b,
    int64_t q_stride_h,
    int64_t q_stride_d,
    int64_t pool_stride_block,
    int64_t pool_stride_token,
    int64_t pool_stride_kv,
    int64_t pool_stride_d,
    int64_t v_stride_block,
    int64_t v_stride_token,
    int64_t v_stride_kv,
    int64_t v_stride_d,
    int64_t table_stride_b,
    int64_t table_stride_page,
    int64_t seq_stride,
    int batch,
    int query_heads,
    int kv_heads,
    int split_k,
    float scale) {
  const int sequence = blockIdx.x;
  const int kv_head = blockIdx.y;
  const int split = blockIdx.z;
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int query_head = kv_head * kGroup + warp;

  extern __shared__ __align__(16) unsigned char shared_raw[];
  scalar_t* shared_k = reinterpret_cast<scalar_t*>(shared_raw);
  scalar_t* shared_v = shared_k + kPageSize * kHeadDim;

  float query[4];
  float accumulator[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int item = 0; item < 4; ++item) {
    const int dim = lane + item * kWarpSize;
    query[item] = static_cast<float>(
        q[sequence * q_stride_b + query_head * q_stride_h + dim * q_stride_d]);
  }

  const int seq_len = static_cast<int>(seq_lens[sequence * seq_stride]);
  const int total_pages = (seq_len + kPageSize - 1) / kPageSize;
  const int start_page = split * total_pages / split_k;
  const int end_page = (split + 1) * total_pages / split_k;
  float running_max = -CUDART_INF_F;
  float running_sum = 0.f;

  for (int logical_page = start_page; logical_page < end_page; ++logical_page) {
    const int page_id = static_cast<int>(
        block_table[sequence * table_stride_b + logical_page * table_stride_page]);

    // All six warps stage the one K/V page cooperatively. Every staged value is
    // then consumed by all six query-head warps instead of fetched six times.
    for (int linear = threadIdx.x; linear < kPageSize * kHeadDim;
         linear += blockDim.x) {
      const int token = linear / kHeadDim;
      const int dim = linear - token * kHeadDim;
      const int64_t k_offset =
          static_cast<int64_t>(page_id) * pool_stride_block +
          token * pool_stride_token + kv_head * pool_stride_kv +
          dim * pool_stride_d;
      const int64_t v_offset =
          static_cast<int64_t>(page_id) * v_stride_block +
          token * v_stride_token + kv_head * v_stride_kv +
          dim * v_stride_d;
      const bool valid = logical_page * kPageSize + token < seq_len;
      shared_k[linear] = valid ? k_pool[k_offset] : static_cast<scalar_t>(0.f);
      shared_v[linear] = valid ? v_pool[v_offset] : static_cast<scalar_t>(0.f);
    }
    __syncthreads();

    float scores[kPageSize];
    const int remaining_tokens = seq_len - logical_page * kPageSize;
    const int valid_tokens = remaining_tokens < kPageSize ? remaining_tokens : kPageSize;
#pragma unroll
    for (int token = 0; token < kPageSize; ++token) {
      float dot = 0.f;
#pragma unroll
      for (int item = 0; item < 4; ++item) {
        const int dim = lane + item * kWarpSize;
        dot += query[item] * static_cast<float>(shared_k[token * kHeadDim + dim]);
      }
      dot = warp_sum(dot) * scale;
      scores[token] = token < valid_tokens ? dot : -CUDART_INF_F;
    }

    float page_max = -CUDART_INF_F;
#pragma unroll
    for (int token = 0; token < kPageSize; ++token) {
      page_max = fmaxf(page_max, scores[token]);
    }
    const float new_max = fmaxf(running_max, page_max);
    const float correction = expf(running_max - new_max);
    running_sum *= correction;
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      accumulator[item] *= correction;
    }

#pragma unroll
    for (int token = 0; token < kPageSize; ++token) {
      const float probability = token < valid_tokens
          ? expf(scores[token] - new_max)
          : 0.f;
      running_sum += probability;
#pragma unroll
      for (int item = 0; item < 4; ++item) {
        const int dim = lane + item * kWarpSize;
        accumulator[item] += probability *
            static_cast<float>(shared_v[token * kHeadDim + dim]);
      }
    }
    running_max = new_max;
    __syncthreads();
  }

  const int64_t partial_head =
      (static_cast<int64_t>(sequence) * split_k + split) * query_heads + query_head;
#pragma unroll
  for (int item = 0; item < 4; ++item) {
    const int dim = lane + item * kWarpSize;
    partial_out[partial_head * kHeadDim + dim] = accumulator[item];
  }
  if (lane == 0) {
    partial_max[partial_head] = running_max;
    partial_sum[partial_head] = running_sum;
  }
}

template <typename scalar_t, typename index_t>
void launch_partial(
    const torch::Tensor& q,
    const torch::Tensor& k_pool,
    const torch::Tensor& v_pool,
    const torch::Tensor& block_table,
    const torch::Tensor& seq_lens,
    torch::Tensor& partial_out,
    torch::Tensor& partial_max,
    torch::Tensor& partial_sum,
    int split_k,
    double scale) {
  const int batch = static_cast<int>(q.size(0));
  const int query_heads = static_cast<int>(q.size(1));
  const int kv_heads = static_cast<int>(k_pool.size(2));
  const dim3 grid(batch, kv_heads, split_k);
  const size_t shared_bytes =
      2 * kPageSize * kHeadDim * sizeof(scalar_t);
  const auto stream = at::cuda::getCurrentCUDAStream(q.get_device());
  grouped_gqa_splitk_partial_kernel<scalar_t, index_t>
      <<<grid, kThreads, shared_bytes, stream>>>(
          q.data_ptr<scalar_t>(),
          k_pool.data_ptr<scalar_t>(),
          v_pool.data_ptr<scalar_t>(),
          block_table.data_ptr<index_t>(),
          seq_lens.data_ptr<index_t>(),
          partial_out.data_ptr<float>(),
          partial_max.data_ptr<float>(),
          partial_sum.data_ptr<float>(),
          q.stride(0), q.stride(1), q.stride(2),
          k_pool.stride(0), k_pool.stride(1), k_pool.stride(2), k_pool.stride(3),
          v_pool.stride(0), v_pool.stride(1), v_pool.stride(2), v_pool.stride(3),
          block_table.stride(0), block_table.stride(1),
          seq_lens.stride(0),
          batch, query_heads, kv_heads, split_k, static_cast<float>(scale));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void grouped_gqa_splitk_partial_out(
    const torch::Tensor& q,
    const torch::Tensor& k_pool,
    const torch::Tensor& v_pool,
    const torch::Tensor& block_table,
    const torch::Tensor& seq_lens,
    torch::Tensor partial_out,
    torch::Tensor partial_max,
    torch::Tensor partial_sum,
    int64_t split_k,
    double scale) {
  TORCH_CHECK(q.is_cuda(), "q must be CUDA");
  TORCH_CHECK(k_pool.is_cuda() && v_pool.is_cuda() && block_table.is_cuda() &&
              seq_lens.is_cuda(), "all inputs must be CUDA");
  TORCH_CHECK(partial_out.is_cuda() && partial_max.is_cuda() && partial_sum.is_cuda(),
              "all outputs must be CUDA");
  TORCH_CHECK(q.device() == k_pool.device() && q.device() == v_pool.device() &&
              q.device() == block_table.device() && q.device() == seq_lens.device(),
              "all inputs must share a device");
  TORCH_CHECK(q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16,
              "Q/K/V must be float16 or bfloat16");
  TORCH_CHECK(k_pool.scalar_type() == q.scalar_type() &&
              v_pool.scalar_type() == q.scalar_type(), "Q/K/V dtypes must match");
  TORCH_CHECK(block_table.scalar_type() == seq_lens.scalar_type() &&
              (block_table.scalar_type() == at::kInt ||
               block_table.scalar_type() == at::kLong),
              "block_table and seq_lens must both be int32 or int64");
  TORCH_CHECK(q.dim() == 3 && k_pool.dim() == 4 && v_pool.sizes() == k_pool.sizes(),
              "invalid Q/K/V ranks or shapes");
  TORCH_CHECK(block_table.dim() == 2 && seq_lens.dim() == 1,
              "block table and lengths must have rank two and one");
  TORCH_CHECK(q.size(0) == block_table.size(0) && q.size(0) == seq_lens.size(0),
              "batch dimensions must match");
  TORCH_CHECK(q.size(2) == kHeadDim && k_pool.size(1) == kPageSize &&
              k_pool.size(3) == kHeadDim,
              "native kernel currently requires page_size=16 and head_dim=128");
  TORCH_CHECK(q.size(1) == k_pool.size(2) * kGroup,
              "native kernel currently requires six query heads per KV head");
  TORCH_CHECK(split_k > 1 && split_k <= 65535, "split_k must be in [2, 65535]");
  TORCH_CHECK(partial_out.scalar_type() == at::kFloat &&
              partial_max.scalar_type() == at::kFloat &&
              partial_sum.scalar_type() == at::kFloat,
              "partial outputs must be float32");
  TORCH_CHECK(partial_out.is_contiguous() && partial_max.is_contiguous() &&
              partial_sum.is_contiguous(), "partial outputs must be contiguous");
  TORCH_CHECK(partial_out.dim() == 4 && partial_out.size(0) == q.size(0) &&
              partial_out.size(1) == split_k && partial_out.size(2) == q.size(1) &&
              partial_out.size(3) == kHeadDim,
              "partial_out has the wrong shape");
  TORCH_CHECK(partial_max.dim() == 3 && partial_max.size(0) == q.size(0) &&
              partial_max.size(1) == split_k && partial_max.size(2) == q.size(1) &&
              partial_sum.sizes() == partial_max.sizes(),
              "partial max/sum have the wrong shape");

  c10::cuda::CUDAGuard device_guard(q.device());
  if (q.scalar_type() == at::kHalf) {
    if (block_table.scalar_type() == at::kInt) {
      launch_partial<at::Half, int32_t>(q, k_pool, v_pool, block_table, seq_lens,
                                        partial_out, partial_max, partial_sum,
                                        static_cast<int>(split_k), scale);
    } else {
      launch_partial<at::Half, int64_t>(q, k_pool, v_pool, block_table, seq_lens,
                                        partial_out, partial_max, partial_sum,
                                        static_cast<int>(split_k), scale);
    }
  } else if (block_table.scalar_type() == at::kInt) {
    launch_partial<at::BFloat16, int32_t>(q, k_pool, v_pool, block_table, seq_lens,
                                          partial_out, partial_max, partial_sum,
                                          static_cast<int>(split_k), scale);
  } else {
    launch_partial<at::BFloat16, int64_t>(q, k_pool, v_pool, block_table, seq_lens,
                                          partial_out, partial_max, partial_sum,
                                          static_cast<int>(split_k), scale);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("grouped_gqa_splitk_partial_out", &grouped_gqa_splitk_partial_out,
             "CTA-shared grouped-GQA split-K partial attention (CUDA)");
}
