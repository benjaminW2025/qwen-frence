// Project-owned Hopper attention candidate. NOT yet GPU-validated or performance-qualified.
// CuTe supplies WGMMA instruction/layout primitives; no upstream attention kernel is used.
// Algorithm attribution: FlashAttention-3 (Shah et al., 2024), FP16 asynchronous pipeline.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cutlass/arch/reg_reconfig.h>
#include <cmath>
#include <cstdint>
#include <type_traits>

using namespace cute;
using H = cutlass::half_t;
constexpr int M = 64, D = 128, G = 6, HQ = 12, HKV = 2;
using LQ = decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<H>{}, Shape<_64,_128>{}, Step<_1,_2>{}));
template<int N, bool RegisterPV>
struct Traits {
    static_assert(N == 64 || N == 128);
    using LK = decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<H>{}, Shape<Int<N>,_128>{}, Step<_1,_2>{}));
    using LV = decltype(tile_to_shape(GMMA::Layout_MN_SW128_Atom<H>{}, Shape<_128,Int<N>>{}, Step<_2,_1>{}));
    using LP = decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<H>{}, Shape<_64,Int<N>>{}));
    using QKOp = std::conditional_t<N == 64,
        SM90_64x64x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::K>,
        SM90_64x128x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::K>>;
    using PVOp = std::conditional_t<RegisterPV,
        SM90_64x128x16_F32F16F16_RS<GMMA::Major::K, GMMA::Major::MN>,
        SM90_64x128x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::MN>>;
    using QKMma = decltype(make_tiled_mma(QKOp{}));
    using PVMma = decltype(make_tiled_mma(PVOp{}));
};

template<int N, bool RegisterPV>
struct alignas(128) Shared {
    using T = Traits<N, RegisterPV>;
    alignas(128) H q[cosize_v<LQ>];
    alignas(128) H k[2][cosize_v<typename T::LK>];
    alignas(128) H v[2][cosize_v<typename T::LV>];
    // Register-fed PV neither stores nor loads P in shared memory.
    alignas(128) H p[RegisterPV ? 1 : cosize_v<typename T::LP>];
    alignas(8) uint64_t ready[2], empty[2];
};

__device__ uint32_t shared_address(void const* pointer) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
}
__device__ void consumer_sync() { asm volatile("bar.sync 1, 128;" ::: "memory"); }
__device__ void async_fence() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ void barrier_init(uint64_t* barrier) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(shared_address(barrier)) : "memory");
}
__device__ void wait_phase(uint64_t* barrier, int phase) {
    uint32_t ready = 0;
    do {
        asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; "
                     "selp.u32 %0, 1, 0, p; }" : "=r"(ready)
                     : "r"(shared_address(barrier)), "r"(phase) : "memory");
    } while (!ready);
}
__device__ void arrive(uint64_t* barrier) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(shared_address(barrier)) : "memory");
}
template<int N>
__device__ void expect_bytes(uint64_t* barrier) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                 :: "r"(shared_address(barrier)), "r"(2 * N * D * int(sizeof(H))) : "memory");
}
__device__ void load_page(CUtensorMap const* map, H* dest, uint64_t* barrier,
                         int feature, int head, int physical_page) {
    asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%3, %4, %5, %6}], [%2];"
                 :: "r"(shared_address(dest)), "l"(map), "r"(shared_address(barrier)),
                    "r"(feature), "r"(head), "r"(0), "r"(physical_page) : "memory");
}

// row distribution of SM90 m64 WGMMA accumulators: four adjacent lanes share
// each row, and each thread holds two rows separated by eight.
__device__ float row_max(float value) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, 1));
    return fmaxf(value, __shfl_xor_sync(0xffffffff, value, 2));
}
__device__ float row_sum(float value) {
    value += __shfl_xor_sync(0xffffffff, value, 1);
    return value + __shfl_xor_sync(0xffffffff, value, 2);
}

template<int N, bool RegisterPV>
__global__ __launch_bounds__(256)
void attention(H const* q, int const* cu, int const* table, int const* lengths,
               H* output, float* partial, float* stats, int width, int splits,
               int query_tiles, float scale, bool causal, bool overlap_qk,
               int64_t qs0, int64_t qs1, int64_t qs2, int const* worklist,
               const __grid_constant__ CUtensorMap km,
               const __grid_constant__ CUtensorMap vm) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    using T = Traits<N, RegisterPV>;
    using LK = typename T::LK;
    using LV = typename T::LV;
    using LP = typename T::LP;
    using QKMma = typename T::QKMma;
    using PVMma = typename T::PVMma;
    int sequence = blockIdx.x, kv_head = blockIdx.y;
    int split = blockIdx.z / query_tiles, query_tile = blockIdx.z % query_tiles;
    if (worklist != nullptr) {
        sequence = worklist[2 * blockIdx.x];
        if (sequence < 0) return;
        query_tile = worklist[2 * blockIdx.x + 1];
        split = blockIdx.z;
    }
    int begin_q = cu[sequence], q_len = cu[sequence + 1] - begin_q;
    int begin_row = query_tile * M;
    if (begin_row >= q_len * G) return;
    int length = lengths[sequence];
    // Entire future tiles are invisible to every row in this query CTA. Avoid
    // their TMA loads and both WGMMA operations, not merely their softmax terms.
    int last_query_exclusive = min(q_len, (begin_row + M + G - 1) / G);
    int visible_length = causal ? max(0, min(length, length - q_len + last_query_exclusive)) : length;
    int tiles = (visible_length + N - 1) / N;
    int first = tiles * split / splits, last = tiles * (split + 1) / splits;
    if (first == last) {
        for (int i = threadIdx.x; i < M * D; i += blockDim.x) {
            int row = begin_row + i / D;
            if (row < q_len * G) {
                int id = (begin_q + row / G) * HQ + kv_head * G + row % G;
                if (splits == 1) output[id * D + i % D] = H(0.f);
                else partial[(id * splits + split) * D + i % D] = 0.f;
            }
        }
        if (splits > 1 && threadIdx.x < M && begin_row + int(threadIdx.x) < q_len * G) {
            int row = begin_row + threadIdx.x;
            int id = (begin_q + row / G) * HQ + kv_head * G + row % G;
            stats[(id * splits + split) * 2] = -INFINITY;
            stats[(id * splits + split) * 2 + 1] = 0.f;
        }
        return;
    }
    extern __shared__ __align__(128) unsigned char shared_bytes[];
    Shared<N, RegisterPV>& sm = *reinterpret_cast<Shared<N, RegisterPV>*>(shared_bytes);
    auto sq = make_tensor(make_smem_ptr(sm.q), LQ{});
    auto sp = make_tensor(make_smem_ptr(sm.p), LP{});
    for (int i = threadIdx.x; i < M * D; i += blockDim.x) {
        int row = begin_row + i / D;
        sq(i / D, i % D) = row < q_len * G
            ? q[(begin_q + row / G) * qs0 + (kv_head * G + row % G) * qs1 + (i % D) * qs2] : H(0.f);
    }
    if (threadIdx.x == 0) {
        for (int stage = 0; stage < 2; ++stage) {
            barrier_init(&sm.ready[stage]); barrier_init(&sm.empty[stage]);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    async_fence();
    __syncthreads();
    if (threadIdx.x < 128) {
        cutlass::arch::warpgroup_reg_dealloc<32>();
        if (threadIdx.x == 0) {
            for (int tile = first; tile < last; ++tile) {
                int i = tile - first, stage = i % 2;
                if (i >= 2) wait_phase(&sm.empty[stage], (i / 2 - 1) % 2);
                expect_bytes<N>(&sm.ready[stage]);
                // Page-wise TMA, two feature blocks and two tensors.
                // Invalid pages use page zero; the consumer masks token tails.
                for (int page = 0; page < N / 16; ++page) {
                    int logical = tile * (N / 16) + page;
                    int physical = logical * 16 < length ? table[sequence * width + logical] : 0;
                    for (int chunk = 0; chunk < 2; ++chunk) {
                        load_page(&km, sm.k[stage] + LK{}(page * 16, chunk * 64),
                                  &sm.ready[stage], chunk * 64, kv_head, physical);
                        load_page(&vm, sm.v[stage] + LV{}(chunk * 64, page * 16),
                                  &sm.ready[stage], chunk * 64, kv_head, physical);
                    }
                }
            }
        }
        return;
    }
    cutlass::arch::warpgroup_reg_alloc<232>();
    int lane = threadIdx.x - 128;
    int base_row = (lane / 32) * 16 + (lane % 32) / 4;
    QKMma qk_mma;
    PVMma pv_mma;
    auto tqk = qk_mma.get_slice(lane);
    auto tpv = pv_mma.get_slice(lane);
    auto score_coord = tqk.partition_C(make_identity_tensor(Shape<_64,Int<N>>{}));
    auto out_coord = tpv.partition_C(make_identity_tensor(Shape<_64,_128>{}));
    auto score = tqk.make_fragment_C(score_coord);
    auto next_score = tqk.make_fragment_C(score_coord);
    auto acc = tpv.make_fragment_C(out_coord);
    clear(acc);
    auto qa = tqk.make_fragment_A(tqk.partition_A(sq));
    auto pa = tpv.make_fragment_A(tpv.partition_A(sp));
    if constexpr (RegisterPV) {
        CUTE_STATIC_ASSERT_V(size(pa) == size(score));
    }
    float maximum[2] = {-INFINITY, -INFINITY}, denominator[2] = {0.f, 0.f};
    auto issue_qk = [&](int stage, auto& fragment) {
        auto sk = make_tensor(make_smem_ptr(sm.k[stage]), LK{});
        auto kb = tqk.make_fragment_B(tqk.partition_B(sk));
        clear(fragment);
        warpgroup_fence_operand(fragment);
        warpgroup_arrive();
        cute::gemm(qk_mma, qa, kb, fragment);
        warpgroup_commit_batch();
    };
    wait_phase(&sm.ready[0], 0);
    issue_qk(0, score);
    warpgroup_wait<0>();
    warpgroup_fence_operand(score);
    for (int tile = first; tile < last; ++tile) {
        int i = tile - first, stage = i % 2;
        bool has_next = tile + 1 < last;
        if (has_next && overlap_qk) {
            wait_phase(&sm.ready[(i + 1) % 2], ((i + 1) / 2) % 2);
            issue_qk((i + 1) % 2, next_score);
        }
        // QK for the next tile is outstanding while CUDA cores do softmax.
        float tile_max[2] = {-INFINITY, -INFINITY};
        CUTE_UNROLL
        for (int j = 0; j < size(score); ++j) {
            int row = get<0>(score_coord(j)), col = get<1>(score_coord(j));
            int r = (row - base_row) / 8, query = (begin_row + row) / G;
            bool live = begin_row + row < q_len * G && tile * N + col < length;
            live = live && (!causal || tile * N + col <= length - q_len + query);
            score(j) = live ? score(j) * scale * 1.4426950408889634f : -INFINITY;
            tile_max[r] = fmaxf(tile_max[r], score(j));
        }
        float alpha[2], safe_max[2], sum[2] = {0.f, 0.f};
        for (int r = 0; r < 2; ++r) {
            float updated = fmaxf(maximum[r], row_max(tile_max[r]));
            safe_max[r] = isfinite(updated) ? updated : 0.f;
            alpha[r] = isfinite(maximum[r]) ? exp2f(maximum[r] - safe_max[r]) : 0.f;
            maximum[r] = updated;
        }
        CUTE_UNROLL
        for (int j = 0; j < size(score); ++j) {
            int row = get<0>(score_coord(j)), col = get<1>(score_coord(j));
            int r = (row - base_row) / 8;
            float p = exp2f(score(j) - safe_max[r]);
            sum[r] += p;
            if constexpr (RegisterPV) pa(j) = H(p);
            else sp(row, col) = H(p);
        }
        for (int r = 0; r < 2; ++r) denominator[r] = denominator[r] * alpha[r] + row_sum(sum[r]);
        CUTE_UNROLL
        for (int j = 0; j < size(acc); ++j) {
            int r = (int(get<0>(out_coord(j))) - base_row) / 8;
            acc(j) *= alpha[r];
        }
        auto sv = make_tensor(make_smem_ptr(sm.v[stage]), LV{});
        // Avoid 0 * NaN from poisoned or uninitialized cache padding in PV.
        for (int j = lane; j < N * D; j += 128) {
            if (tile * N + j / D >= length) sv(j % D, j / D) = H(0.f);
        }
        async_fence();
        consumer_sync();
        auto vb = tpv.make_fragment_B(tpv.partition_B(sv));
        if constexpr (RegisterPV) warpgroup_fence_operand(pa);
        warpgroup_fence_operand(acc);
        warpgroup_arrive();
        cute::gemm(pv_mma, pa, vb, acc);
        warpgroup_commit_batch();
        warpgroup_wait<0>();
        warpgroup_fence_operand(acc);
        if (has_next && !overlap_qk) {
            wait_phase(&sm.ready[(i + 1) % 2], ((i + 1) / 2) % 2);
            issue_qk((i + 1) % 2, next_score);
            warpgroup_wait<0>();
        }
        if (has_next) {
            warpgroup_fence_operand(next_score);
            copy(next_score, score);
        }
        consumer_sync();
        if (lane == 0) arrive(&sm.empty[stage]);
    }
    CUTE_UNROLL
    for (int j = 0; j < size(acc); ++j) {
        int local_row = get<0>(out_coord(j)), dim = get<1>(out_coord(j));
        int row = begin_row + local_row, r = (local_row - base_row) / 8;
        if (row < q_len * G) {
            int id = (begin_q + row / G) * HQ + kv_head * G + row % G;
            if (splits == 1) output[id * D + dim] = H(acc(j) / fmaxf(denominator[r], 1.e-30f));
            else partial[(id * splits + split) * D + dim] = acc(j);
        }
    }
    if (splits > 1 && lane % 4 == 0) {
        for (int r = 0; r < 2; ++r) {
            int row = begin_row + base_row + r * 8;
            if (row < q_len * G) {
                int id = (begin_q + row / G) * HQ + kv_head * G + row % G;
                stats[(id * splits + split) * 2] = maximum[r];
                stats[(id * splits + split) * 2 + 1] = denominator[r];
            }
        }
    }
#endif
}

__global__ void combine(float const* partial, float const* stats, H* output, int splits) {
    int id = blockIdx.x, dim = threadIdx.x;
    float maximum = -INFINITY, sum = 0.f, value = 0.f;
    for (int s = 0; s < splits; ++s) {
        float m = stats[(id * splits + s) * 2], l = stats[(id * splits + s) * 2 + 1];
        if (l > 0.f) {
            float next = fmaxf(maximum, m);
            float old_weight = sum > 0.f ? exp2f(maximum - next) : 0.f;
            float new_weight = exp2f(m - next);
            value = value * old_weight + partial[(id * splits + s) * D + dim] * new_weight;
            sum = sum * old_weight + l * new_weight;
            maximum = next;
        }
    }
    output[id * D + dim] = H(value / fmaxf(sum, 1.e-30f));
}

CUtensorMap tensor_map(torch::Tensor const& pool) {
    CUtensorMap map{};
    uint64_t dims[4] = {D, HKV, 16, uint64_t(pool.size(0))};
    uint64_t strides[3] = {D * 2, HKV * D * 2, 16 * HKV * D * 2};
    uint32_t box[4] = {64, 1, 16, 1}, element_strides[4] = {1,1,1,1};
    auto result = cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 4,
        pool.data_ptr(), dims, strides, box, element_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(result == CUDA_SUCCESS, "TMA descriptor encoding failed: ", int(result));
    return map;
}

template<int N, bool RegisterPV>
bool validate_layouts_for() {
    using T = Traits<N, RegisterPV>;
    using LK = typename T::LK;
    using LV = typename T::LV;
    using QKMma = typename T::QKMma;
    using PVMma = typename T::PVMma;
    // Check the exact layout assumed by the sixteen page-wise TMA transfers.
    // These checks are host-only and also guard changes to the CuTe version.
    for (int token = 0; token < N; ++token) {
        for (int feature = 0; feature < D; ++feature) {
            int physical = (feature / 64) * N * 64 + token * 64
                           + ((feature % 64) ^ ((token % 8) * 8));
            auto k_offset = int(LK{}(token, feature));
            auto v_offset = int(LV{}(feature, token));
            TORCH_CHECK(k_offset == physical,
                        "K TMA/WGMMA layout mismatch: N=", N,
                        " register_pv=", RegisterPV, " token=", token,
                        " feature=", feature, " CuTe offset=", k_offset,
                        " expected TMA offset=", physical);
            TORCH_CHECK(v_offset == physical,
                        "V TMA/WGMMA layout mismatch: N=", N,
                        " register_pv=", RegisterPV, " token=", token,
                        " feature=", feature, " CuTe offset=", v_offset,
                        " expected TMA offset=", physical);
        }
    }
    for (int lane = 0; lane < 128; ++lane) {
        auto score_coord = QKMma{}.get_slice(lane).partition_C(make_identity_tensor(Shape<_64,Int<N>>{}));
        auto out_coord = PVMma{}.get_slice(lane).partition_C(make_identity_tensor(Shape<_64,_128>{}));
        int base = (lane / 32) * 16 + (lane % 32) / 4;
        for (int j = 0; j < size(score_coord); ++j) {
            int row = get<0>(score_coord(j));
            TORCH_CHECK(row == base || row == base + 8, "QK softmax row distribution differs");
        }
        for (int j = 0; j < size(out_coord); ++j) {
            int row = get<0>(out_coord(j));
            TORCH_CHECK(row == base || row == base + 8, "PV accumulator row distribution differs");
        }
        if constexpr (RegisterPV) {
            // Register-source A uses the same lane/element order as QK's C
            // fragment for our M64 FP16 specialization. Prove this for every
            // lane and element, rather than assuming a flatten/cast is valid.
            auto a_coord = PVMma{}.get_slice(lane).partition_A(make_identity_tensor(Shape<_64,Int<N>>{}));
            TORCH_CHECK(int(size(a_coord)) == int(size(score_coord)), "PV register count differs");
            for (int j = 0; j < size(a_coord); ++j) {
                TORCH_CHECK(int(get<0>(a_coord(j))) == int(get<0>(score_coord(j))) &&
                            int(get<1>(a_coord(j))) == int(get<1>(score_coord(j))),
                            "QK-to-register-PV lane/element mapping differs");
            }
        }
    }
    return true;
}

bool validate_layouts() {
    return validate_layouts_for<64, false>() && validate_layouts_for<64, true>() &&
           validate_layouts_for<128, false>() && validate_layouts_for<128, true>();
}

// Construct a compact sequence/query-tile worklist on device. A host-computable
// capacity avoids a GPU->CPU count readback: sum ceil(q_i*G/M) <=
// ceil(total_q*G/M) + batch - 1. Unused capacity is marked -1 and exits early.
__global__ void construct_worklist(int const* cu, int* work, int batch, int capacity) {
    __shared__ int prefix[257];
    if (threadIdx.x == 0) {
        prefix[0] = 0;
        for (int i = 0; i < batch; ++i)
            prefix[i + 1] = prefix[i] + ((cu[i + 1] - cu[i]) * G + M - 1) / M;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < capacity; i += blockDim.x) {
        int seq = -1, tile = -1;
        if (i < prefix[batch]) {
            int lo = 0, hi = batch;
            while (lo < hi) {
                int middle = (lo + hi) / 2;
                if (prefix[middle + 1] <= i) lo = middle + 1;
                else hi = middle;
            }
            seq = lo;
            tile = i - prefix[lo];
        }
        work[i * 2] = seq;
        work[i * 2 + 1] = tile;
    }
}

torch::Tensor prepare_worklist(torch::Tensor cu, int64_t total_queries) {
    TORCH_CHECK(cu.is_cuda() && cu.is_contiguous() && cu.scalar_type() == at::kInt &&
                cu.dim() == 1 && cu.numel() >= 2 && cu.numel() <= 257,
                "compact scheduling requires contiguous CUDA int32 offsets for 1..256 sequences");
    TORCH_CHECK(total_queries > 0 && total_queries <= 1048576, "invalid compact query count");
    c10::cuda::CUDAGuard guard(cu.device());
    int batch = cu.numel() - 1;
    int capacity = (total_queries * G + M - 1) / M + batch - 1;
    auto work = torch::empty({capacity, 2}, cu.options());
    construct_worklist<<<1, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        cu.data_ptr<int>(), work.data_ptr<int>(), batch, capacity);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return work;
}

template<int N, bool RegisterPV>
void launch_attention(torch::Tensor q, torch::Tensor cu, torch::Tensor table, torch::Tensor lengths,
                      torch::Tensor out, torch::Tensor partial, torch::Tensor stats,
                      torch::Tensor work, int splits, int query_tiles, float scale,
                      bool causal, bool overlap_qk, CUtensorMap km, CUtensorMap vm) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(attention<N, RegisterPV>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(Shared<N, RegisterPV>)));
    dim3 grid(work.defined() ? work.size(0) : lengths.numel(), HKV,
              work.defined() ? splits : query_tiles * splits);
    attention<N, RegisterPV><<<grid, 256, sizeof(Shared<N, RegisterPV>), at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<H const*>(q.data_ptr()), cu.data_ptr<int>(), table.data_ptr<int>(),
        lengths.data_ptr<int>(), reinterpret_cast<H*>(out.data_ptr()), partial.data_ptr<float>(),
        stats.data_ptr<float>(), table.size(1), splits, query_tiles, scale, causal, overlap_qk,
        q.stride(0), q.stride(1), q.stride(2), work.defined() ? work.data_ptr<int>() : nullptr, km, vm);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor hopper_attention_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                      torch::Tensor cu, torch::Tensor table, torch::Tensor lengths,
                      int64_t max_query, int64_t splits, double scale, bool causal, bool overlap_qk,
                      int64_t tile_n, bool register_pv, bool compact, c10::optional<torch::Tensor> prepared_worklist) {
    TORCH_CHECK(q.is_cuda(), "Hopper attention requires CUDA");
    c10::cuda::CUDAGuard guard(q.device());
    auto props = at::cuda::getCurrentDeviceProperties();
    static bool const layouts_checked = validate_layouts();
    (void)layouts_checked;
    TORCH_CHECK(props->major == 9 && props->minor == 0, "requires SM90 Hopper");
    for (auto const& tensor : {k,v,cu,table,lengths})
        TORCH_CHECK(tensor.is_cuda() && tensor.device() == q.device() && tensor.is_contiguous(),
                    "all inputs must be contiguous on one CUDA device");
    TORCH_CHECK(q.scalar_type() == at::kHalf && k.scalar_type() == at::kHalf && v.scalar_type() == at::kHalf,
                "this specialization requires FP16");
    TORCH_CHECK(q.dim()==3 && q.size(1)==HQ && q.size(2)==D, "Q must be [tokens,12,128]");
    TORCH_CHECK(k.dim()==4 && k.size(0)>0 && k.size(1)==16 && k.size(2)==HKV && k.size(3)==D
                && k.sizes()==v.sizes(), "KV must be [pages,16,2,128]");
    TORCH_CHECK(cu.scalar_type()==at::kInt && table.scalar_type()==at::kInt && lengths.scalar_type()==at::kInt,
                "metadata must be int32");
    TORCH_CHECK(lengths.dim()==1 && cu.dim()==1 && cu.numel()==lengths.numel()+1
                && table.dim()==2 && table.size(0)==lengths.numel() && table.size(1)>0,
                "metadata dimensions differ");
    TORCH_CHECK(max_query>0 && max_query<=q.size(0) && splits>0 && splits<=64, "invalid bounds/splits");
    TORCH_CHECK(std::isfinite(scale), "attention scale must be finite");
    TORCH_CHECK(tile_n == 64 || tile_n == 128, "KV tile must be 64 or 128");
    TORCH_CHECK(q.numel() * splits <= INT32_MAX, "split buffers exceed int32 indexing limit");
    torch::Tensor work;
    if (prepared_worklist.has_value()) {
        TORCH_CHECK(compact, "prepared worklist requires compact scheduling");
        work = prepared_worklist.value();
        TORCH_CHECK(work.device() == q.device() && work.scalar_type() == at::kInt &&
                    work.is_contiguous() && work.dim() == 2 && work.size(1) == 2 && work.size(0) > 0,
                    "invalid prepared worklist");
    } else if (compact) work = prepare_worklist(cu, q.size(0));
    auto out = torch::empty(q.sizes(), q.options());
    auto partial = torch::empty({splits > 1 ? q.numel() * splits : 0}, q.options().dtype(at::kFloat));
    auto stats = torch::empty({splits > 1 ? q.size(0) * HQ * splits * 2 : 0}, q.options().dtype(at::kFloat));
    CUtensorMap km = tensor_map(k), vm = tensor_map(v);
    int query_tiles = (max_query * G + M - 1) / M;
    TORCH_CHECK(int64_t(query_tiles) * splits <= 65535, "query grid exceeds CUDA z limit");
    auto stream = at::cuda::getCurrentCUDAStream();
    if (tile_n == 64) {
        if (register_pv) launch_attention<64, true>(q,cu,table,lengths,out,partial,stats,work,splits,query_tiles,scale,causal,overlap_qk,km,vm);
        else launch_attention<64, false>(q,cu,table,lengths,out,partial,stats,work,splits,query_tiles,scale,causal,overlap_qk,km,vm);
    } else {
        if (register_pv) launch_attention<128, true>(q,cu,table,lengths,out,partial,stats,work,splits,query_tiles,scale,causal,overlap_qk,km,vm);
        else launch_attention<128, false>(q,cu,table,lengths,out,partial,stats,work,splits,query_tiles,scale,causal,overlap_qk,km,vm);
    }
    if (splits > 1) {
        combine<<<q.size(0)*HQ, D, 0, stream>>>(partial.data_ptr<float>(), stats.data_ptr<float>(),
                                              reinterpret_cast<H*>(out.data_ptr()), splits);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}

template<int N, bool RegisterPV>
pybind11::dict variant_info() {
    cudaFuncAttributes attrs{};
    C10_CUDA_CHECK(cudaFuncGetAttributes(&attrs, attention<N, RegisterPV>));
    pybind11::dict result;
    result["tile_n"] = N;
    result["register_pv"] = RegisterPV;
    result["compiler_registers_per_thread"] = attrs.numRegs;
    result["local_bytes_per_thread"] = attrs.localSizeBytes;
    result["dynamic_shared_bytes_per_cta"] = sizeof(Shared<N, RegisterPV>);
    // Dynamic setmaxnreg changes the producer/consumer split; the compiler
    // register count alone is not an achieved-occupancy measurement.
    return result;
}

pybind11::list kernel_info() {
    pybind11::list result;
    result.append(variant_info<64, false>());
    result.append(variant_info<64, true>());
    result.append(variant_info<128, false>());
    result.append(variant_info<128, true>());
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.attr("abi_version") = 3;
    module.attr("source_sha256") = HOPPER_SOURCE_HASH;
    module.def("forward", &::hopper_attention_forward);
    module.def("validate_layouts", &validate_layouts);
    module.def("prepare_worklist", &prepare_worklist);
    module.def("kernel_info", &kernel_info);
}
