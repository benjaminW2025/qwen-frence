// Project-owned Hopper attention candidate. NOT yet GPU-validated or performance-qualified.
// CuTe supplies WGMMA instruction/layout primitives; no upstream attention kernel is used.
// Algorithm attribution: FlashAttention-3 (Shah et al., 2024), FP16 asynchronous pipeline.
// Paged KV uses CuTe tensor views for both stores and WGMMA reads. GMMA layout
// flags move the byte-address swizzle into the tensor pointer; raw-array stores
// indexed by Layout{}(coord) do NOT implement that address transformation.
//
// One CTA covers one query tile of one sequence, one KV head and one split:
//   warpgroup 0       producer. Gathers paged K/V tiles in 16-byte chunks into a
//                     Stages-deep shared-memory ring signalled by mbarriers.
//   warpgroups 1..C   consumers, 64 packed (query, GQA head) rows each. Softmax
//                     of tile i runs while the PV GEMM of tile i-1 is in flight
//                     (FA3 intra-warpgroup pipelining). With two consumers a
//                     named-barrier ping-pong orders GEMM issue, so one
//                     warpgroup's softmax runs under the other's tensor-core work.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cutlass/arch/reg_reconfig.h>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include "shared_layouts.cuh"

using namespace cute;
using H = cutlass::half_t;
constexpr int M = 64, D = 128, G = 6, HQ = 12, HKV = 2;
constexpr int MaxDynamicShared = 227 * 1024;
using LQ = hopper_layouts::Q<H>;
template<int N, bool RegisterPV>
struct Traits {
    static_assert(N == 64 || N == 128);
    using LK = hopper_layouts::K<N, H>;
    using LV = hopper_layouts::V<N, H>;
    using LP = hopper_layouts::P<N, H>;
    using QKOp = std::conditional_t<N == 64,
        SM90_64x64x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::K>,
        SM90_64x128x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::K>>;
    using PVOp = std::conditional_t<RegisterPV,
        SM90_64x128x16_F32F16F16_RS<GMMA::Major::K, GMMA::Major::MN>,
        SM90_64x128x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::MN>>;
    using QKMma = decltype(make_tiled_mma(QKOp{}));
    using PVMma = decltype(make_tiled_mma(PVOp{}));
};

// setmaxnreg moves registers within the CTA's launch allocation: the producer
// releases what the consumers claim. 32/232 is the split already proven with
// one consumer; 40/232 over 384 threads is CUTLASS's warp-specialized
// cooperative split and needs the 168-register launch allocation checked at
// the first launch.
template<int Consumers> struct Registers;
template<> struct Registers<1> { static constexpr int producer = 32, consumer = 232; };
template<> struct Registers<2> { static constexpr int producer = 40, consumer = 232; };

// Three KV stages whenever they fit in the 227 KiB opt-in limit, otherwise two.
// Only N=128 with shared P and two consumers falls back to two.
template<int N, bool RegisterPV, int Consumers>
constexpr int stages_for() {
    constexpr int q = Consumers * M * D * 2;
    constexpr int kv = 2 * N * D * 2;
    constexpr int p = RegisterPV ? 0 : Consumers * M * N * 2;
    return q + 3 * kv + p + 1024 <= MaxDynamicShared ? 3 : 2;
}

template<int N, bool RegisterPV, int Consumers>
struct alignas(128) Shared {
    using T = Traits<N, RegisterPV>;
    static constexpr int Stages = stages_for<N, RegisterPV, Consumers>();
    alignas(128) H q[Consumers][cosize_v<LQ>];
    alignas(128) H k[Stages][cosize_v<typename T::LK>];
    alignas(128) H v[Stages][cosize_v<typename T::LV>];
    // Register-fed PV neither stores nor loads P in shared memory.
    alignas(128) H p[Consumers][RegisterPV ? 1 : cosize_v<typename T::LP>];
    alignas(8) uint64_t ready[Stages], empty[Stages];
};

// Named barriers; 0 is __syncthreads. Consumer and schedule IDs add the
// consumer warpgroup index.
constexpr int ProducerBarrier = 1, ConsumerBarrier = 2, ScheduleBarrier = 4;

__device__ uint32_t shared_address(void const* pointer) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
}
__device__ void named_sync(int id, int threads) {
    asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(threads) : "memory");
}
__device__ void named_arrive(int id, int threads) {
    asm volatile("bar.arrive %0, %1;" :: "r"(id), "r"(threads) : "memory");
}
__device__ void async_fence() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ void barrier_init(uint64_t* barrier, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(shared_address(barrier)), "r"(count) : "memory");
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
// Row distribution of SM90 m64 WGMMA accumulators: four adjacent lanes share
// each row, and each thread holds two rows separated by eight.
__device__ float row_max(float value) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, 1));
    return fmaxf(value, __shfl_xor_sync(0xffffffff, value, 2));
}
__device__ float row_sum(float value) {
    value += __shfl_xor_sync(0xffffffff, value, 1);
    return value + __shfl_xor_sync(0xffffffff, value, 2);
}
// Accumulator element j of a lane sits at row base_row + 8 * row_half(j) and
// column 2 * (lane % 4) + column_offset(j). validate_layouts_for proves this
// for every lane and element. Compile-time row indices keep per-row softmax
// state in registers instead of local memory.
__host__ __device__ constexpr int row_half(int j) { return (j / 2) % 2; }
__host__ __device__ constexpr int column_offset(int j) { return j % 2 + 8 * (j / 4); }

template<int N, bool RegisterPV, int Consumers>
__global__ __launch_bounds__(128 * (1 + Consumers), 1)
void attention(H const* q, H const* k_pool, H const* v_pool,
               int const* cu, int const* table, int const* lengths,
               H* output, float* partial, float* stats, int width, int splits,
               int query_tiles, float scale, bool causal, bool overlap,
               int64_t qs0, int64_t qs1, int64_t qs2, int const* worklist) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    using T = Traits<N, RegisterPV>;
    using LK = typename T::LK;
    using LV = typename T::LV;
    using LP = typename T::LP;
    using QKMma = typename T::QKMma;
    using PVMma = typename T::PVMma;
    using SharedT = Shared<N, RegisterPV, Consumers>;
    constexpr int Stages = SharedT::Stages;
    constexpr int Rows = M * Consumers;
    int sequence = blockIdx.x, kv_head = blockIdx.y;
    int split = blockIdx.z / query_tiles, query_tile = blockIdx.z % query_tiles;
    if (worklist != nullptr) {
        // Row 0 records the query-tile height the worklist was built for; a
        // mismatch would silently skip or duplicate rows, so fail loudly.
        if (worklist[0] != Rows) __trap();
        sequence = worklist[2 * (blockIdx.x + 1)];
        if (sequence < 0) return;
        query_tile = worklist[2 * (blockIdx.x + 1) + 1];
        split = blockIdx.z;
    }
    int begin_q = cu[sequence], q_len = cu[sequence + 1] - begin_q;
    int begin_row = query_tile * Rows;
    if (begin_row >= q_len * G) return;
    int length = lengths[sequence];
    // Entire future tiles are invisible to every row in this query CTA. Avoid
    // their KV gathers and both WGMMA operations, not merely their softmax terms.
    int last_query_exclusive = min(q_len, (begin_row + Rows + G - 1) / G);
    int visible_length = causal ? max(0, min(length, length - q_len + last_query_exclusive)) : length;
    int tiles = (visible_length + N - 1) / N;
    int first = tiles * split / splits, last = tiles * (split + 1) / splits;
    if (first == last) {
        for (int i = threadIdx.x; i < Rows * D; i += blockDim.x) {
            int row = begin_row + i / D;
            if (row < q_len * G) {
                int id = (begin_q + row / G) * HQ + kv_head * G + row % G;
                if (splits == 1) output[id * D + i % D] = H(0.f);
                else partial[(id * splits + split) * D + i % D] = 0.f;
            }
        }
        if (splits > 1 && int(threadIdx.x) < Rows && begin_row + int(threadIdx.x) < q_len * G) {
            int row = begin_row + threadIdx.x;
            int id = (begin_q + row / G) * HQ + kv_head * G + row % G;
            stats[(id * splits + split) * 2] = -INFINITY;
            stats[(id * splits + split) * 2 + 1] = 0.f;
        }
        return;
    }
    extern __shared__ __align__(128) unsigned char shared_bytes[];
    SharedT& sm = *reinterpret_cast<SharedT*>(shared_bytes);
    // Q uses the same 16-byte chunking as K/V when features are unit-stride and
    // 16-byte aligned; any other stride keeps the elementwise path.
    bool const vector_q = qs2 == 1 && qs1 % 8 == 0 && qs0 % 8 == 0 &&
                          (reinterpret_cast<uintptr_t>(q) & 15) == 0;
    if (vector_q) {
        for (int i = threadIdx.x; i < Rows * (D / 8); i += blockDim.x) {
            int row = i / (D / 8), f8 = (i % (D / 8)) * 8, packed = begin_row + row;
            auto sq = make_tensor(make_smem_ptr(sm.q[row / M]), LQ{});
            uint4 value = make_uint4(0, 0, 0, 0);
            if (packed < q_len * G)
                value = *reinterpret_cast<uint4 const*>(
                    q + (begin_q + packed / G) * qs0 + (kv_head * G + packed % G) * qs1 + f8);
            *reinterpret_cast<uint4*>(&sq(row % M, f8)) = value;
        }
    } else {
        for (int i = threadIdx.x; i < Rows * D; i += blockDim.x) {
            int row = i / D, packed = begin_row + row;
            auto sq = make_tensor(make_smem_ptr(sm.q[row / M]), LQ{});
            sq(row % M, i % D) = packed < q_len * G
                ? q[(begin_q + packed / G) * qs0 + (kv_head * G + packed % G) * qs1 + (i % D) * qs2] : H(0.f);
        }
    }
    if (threadIdx.x == 0) {
        for (int stage = 0; stage < Stages; ++stage) {
            barrier_init(&sm.ready[stage], 1);
            barrier_init(&sm.empty[stage], Consumers);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    async_fence();
    __syncthreads();
    if (threadIdx.x < 128) {
        cutlass::arch::warpgroup_reg_dealloc<Registers<Consumers>::producer>();
        for (int tile = first; tile < last; ++tile) {
            int i = tile - first, stage = i % Stages;
            if (i >= Stages) wait_phase(&sm.empty[stage], (i / Stages - 1) & 1);
            auto sk = make_tensor(make_smem_ptr(sm.k[stage]), LK{});
            auto sv = make_tensor(make_smem_ptr(sm.v[stage]), LV{});
            // Each producer owns distinct logical (token, feature) elements.
            // Index the TENSORS, not raw arrays with LK/LV offsets: CuTe's
            // smem_ptr_flag transfers the swizzle into the pointer engine.
            // One uint4 moves eight halves: the 128B swizzle permutes 16B
            // chunks without splitting one, and both the paged pool and the
            // K/V layouts are feature-contiguous within a 64-element block,
            // so an eight-aligned feature run is 16 contiguous bytes in both.
            // Threads sharing a token also share its one block-table load.
            // Tokens at or past the context length are written as zeros, so
            // poisoned cache padding never reaches either GEMM.
            constexpr int VEC = 8, OCTETS = D / VEC;
            for (int token = threadIdx.x / OCTETS; token < N; token += 128 / OCTETS) {
                int f8 = int(threadIdx.x % OCTETS) * VEC;
                int absolute = tile * N + token;
                uint4 key = make_uint4(0, 0, 0, 0), value = key;
                if (absolute < length) {
                    int physical = table[sequence * width + absolute / 16];
                    int pool_offset = ((physical * 16 + absolute % 16) * HKV + kv_head) * D + f8;
                    key = *reinterpret_cast<uint4 const*>(k_pool + pool_offset);
                    value = *reinterpret_cast<uint4 const*>(v_pool + pool_offset);
                }
                *reinterpret_cast<uint4*>(&sk(token, f8)) = key;
                *reinterpret_cast<uint4*>(&sv(f8, token)) = value;
            }
            async_fence();
            named_sync(ProducerBarrier, 128);
            if (threadIdx.x == 0) arrive(&sm.ready[stage]);
        }
        return;
    }
    cutlass::arch::warpgroup_reg_alloc<Registers<Consumers>::consumer>();
    int const wg = (threadIdx.x - 128) / 128;
    int const lane = threadIdx.x - 128 * (wg + 1);
    int const base_row = (lane / 32) * 16 + (lane % 32) / 4;
    int const base_col = 2 * (lane % 4);
    int const wg_row = begin_row + wg * M;
    // A warpgroup whose 64 rows all lie past this sequence's queries keeps the
    // pipeline protocol (stage waits, releases, schedule turns) but skips its
    // GEMMs and softmax. The flag is uniform across the warpgroup.
    bool const active = wg_row < q_len * G;
    QKMma qk_mma;
    PVMma pv_mma;
    auto tqk = qk_mma.get_slice(lane);
    auto tpv = pv_mma.get_slice(lane);
    auto sq = make_tensor(make_smem_ptr(sm.q[wg]), LQ{});
    auto sp = make_tensor(make_smem_ptr(sm.p[wg]), LP{});
    auto score = tqk.make_fragment_C(tqk.partition_C(make_identity_tensor(Shape<_64,Int<N>>{})));
    auto acc = tpv.make_fragment_C(tpv.partition_C(make_identity_tensor(Shape<_64,_128>{})));
    clear(acc);
    auto qa = tqk.make_fragment_A(tqk.partition_A(sq));
    auto pa = tpv.make_fragment_A(tpv.partition_A(sp));
    if constexpr (RegisterPV) {
        CUTE_STATIC_ASSERT_V(size(pa) == size(score));
    }
    float const scale_log2 = scale * 1.4426950408889634f;
    float maximum[2] = {-INFINITY, -INFINITY}, denominator[2] = {0.f, 0.f}, alpha[2] = {1.f, 1.f};

    auto issue_qk = [&](int stage) {
        auto kb = tqk.make_fragment_B(tqk.partition_B(make_tensor(make_smem_ptr(sm.k[stage]), LK{})));
        warpgroup_fence_operand(score);
        warpgroup_arrive();
        // The first K block overwrites the fragment; no separate clear.
        qk_mma.accumulate_ = GMMA::ScaleOut::Zero;
        CUTE_UNROLL
        for (int block = 0; block < size<2>(qa); ++block) {
            cute::gemm(qk_mma, qa(_, _, block), kb(_, _, block), score);
            qk_mma.accumulate_ = GMMA::ScaleOut::One;
        }
        warpgroup_commit_batch();
    };
    auto issue_pv = [&](int stage) {
        auto vb = tpv.make_fragment_B(tpv.partition_B(make_tensor(make_smem_ptr(sm.v[stage]), LV{})));
        if constexpr (RegisterPV) warpgroup_fence_operand(pa);
        warpgroup_fence_operand(acc);
        warpgroup_arrive();
        cute::gemm(pv_mma, pa, vb, acc);
        warpgroup_commit_batch();
    };
    // Scale into the log2 domain, apply masking, and take per-row maxima.
    // Tiles that every row of this warpgroup sees in full skip the mask.
    auto scale_and_max = [&](auto masked, int column0, float (&tile_max)[2]) {
        CUTE_UNROLL
        for (int j = 0; j < size(score); ++j) {
            int const r = row_half(j);
            float value = score(j) * scale_log2;
            if constexpr (decltype(masked)::value) {
                int const packed = wg_row + base_row + 8 * r;
                int const column = column0 + base_col + column_offset(j);
                bool const live = packed < q_len * G && column < length &&
                    (!causal || column <= length - q_len + packed / G);
                value = live ? value : -INFINITY;
            }
            score(j) = value;
            tile_max[r] = fmaxf(tile_max[r], value);
        }
    };
    // Online softmax for one tile. Leaves exp2 probabilities in the score
    // fragment and the correction factor for the running output in alpha.
    // Row sums stay thread-local: alpha is quad-uniform because maximum is
    // quad-reduced, so the quad reduction is deferred to the epilogue.
    auto softmax = [&](int tile) {
        int const column0 = tile * N;
        bool const unmasked = wg_row + M <= q_len * G && column0 + N <= length &&
            (!causal || column0 + N - 1 <= length - q_len + wg_row / G);
        float tile_max[2] = {-INFINITY, -INFINITY};
        if (unmasked) scale_and_max(std::false_type{}, column0, tile_max);
        else scale_and_max(std::true_type{}, column0, tile_max);
        float safe_max[2], sum[2] = {0.f, 0.f};
        CUTE_UNROLL
        for (int r = 0; r < 2; ++r) {
            float const updated = fmaxf(maximum[r], row_max(tile_max[r]));
            safe_max[r] = isfinite(updated) ? updated : 0.f;
            alpha[r] = isfinite(maximum[r]) ? exp2f(maximum[r] - safe_max[r]) : 0.f;
            maximum[r] = updated;
        }
        CUTE_UNROLL
        for (int j = 0; j < size(score); ++j) {
            float const p = exp2f(score(j) - safe_max[row_half(j)]);
            score(j) = p;
            sum[row_half(j)] += p;
        }
        CUTE_UNROLL
        for (int r = 0; r < 2; ++r) denominator[r] = denominator[r] * alpha[r] + sum[r];
    };
    // Publish probabilities as the PV A operand. Only called once the previous
    // PV has retired, because that GEMM reads the same registers or buffer.
    auto store_p = [&]() {
        if constexpr (RegisterPV) {
            CUTE_UNROLL
            for (int j = 0; j < size(score); ++j) pa(j) = H(score(j));
        } else {
            CUTE_UNROLL
            for (int j = 0; j < size(score); ++j)
                sp(base_row + 8 * row_half(j), base_col + column_offset(j)) = H(score(j));
            async_fence();
            named_sync(ConsumerBarrier + wg, 128);
        }
    };
    auto rescale = [&]() {
        CUTE_UNROLL
        for (int j = 0; j < size(acc); ++j) acc(j) *= alpha[row_half(j)];
    };
    auto release = [&](int stage) {
        named_sync(ConsumerBarrier + wg, 128);
        if (lane == 0) arrive(&sm.empty[stage]);
    };
    // Ping-pong: phase k of warpgroup w issues its GEMMs after phase k-1 of
    // the other (warpgroup 0 starts unblocked). The final phase of warpgroup 1
    // has no successor, so every named barrier's arrivals match its syncs.
    int const phases = last - first + 1;
    auto schedule_begin = [&](int phase) {
        if constexpr (Consumers == 2) {
            if (!(wg == 0 && phase == 0)) named_sync(ScheduleBarrier + wg, 256);
        }
    };
    auto schedule_end = [&](int phase) {
        if constexpr (Consumers == 2) {
            if (!(wg == 1 && phase == phases - 1)) named_arrive(ScheduleBarrier + 1 - wg, 256);
        }
    };

    // Invariant at the top of iteration i: acc holds tiles < i-1 at the
    // running maximum of tile i-1, and P(i-1) is published. Issue QK(i) and
    // PV(i-1) together; softmax(i) runs while PV(i-1) is in flight; once PV
    // retires, publish P(i) and move acc to tile i's maximum.
    wait_phase(&sm.ready[0], 0);
    schedule_begin(0);
    if (active) issue_qk(0);
    schedule_end(0);
    if (active) {
        warpgroup_wait<0>();
        warpgroup_fence_operand(score);
        softmax(first);
        store_p();  // acc is still zero, so tile 0 needs no rescale
    }
    for (int tile = first + 1; tile < last; ++tile) {
        int const i = tile - first, stage = i % Stages, previous = (i - 1) % Stages;
        wait_phase(&sm.ready[stage], (i / Stages) & 1);
        schedule_begin(i);
        if (active) {
            issue_qk(stage);
            if (!overlap) warpgroup_wait<0>();
            issue_pv(previous);
            if (!overlap) warpgroup_wait<0>();
        }
        schedule_end(i);
        if (active) {
            warpgroup_wait<1>();  // QK(i) retired; PV(i-1) may still run
            warpgroup_fence_operand(score);
            softmax(tile);
            warpgroup_wait<0>();  // PV(i-1) retired: acc, P and the stage are free
            warpgroup_fence_operand(acc);
        }
        release(previous);
        if (active) {
            store_p();
            rescale();
        }
    }
    int const final_stage = (last - first - 1) % Stages;
    schedule_begin(phases - 1);
    if (active) issue_pv(final_stage);
    schedule_end(phases - 1);
    if (active) {
        warpgroup_wait<0>();
        warpgroup_fence_operand(acc);
    }
    release(final_stage);

    CUTE_UNROLL
    for (int r = 0; r < 2; ++r) denominator[r] = row_sum(denominator[r]);
    float inverse[2];
    CUTE_UNROLL
    for (int r = 0; r < 2; ++r) inverse[r] = 1.f / fmaxf(denominator[r], 1.e-30f);
    // Elements j and j+1 (j even) are adjacent columns of one row starting at
    // an even column, so each pair is one 4-byte (FP16) or 8-byte (FP32) store.
    CUTE_UNROLL
    for (int j = 0; j < size(acc); j += 2) {
        int const r = row_half(j);
        int const packed = wg_row + base_row + 8 * r;
        int const dim = base_col + column_offset(j);
        if (packed < q_len * G) {
            int const id = (begin_q + packed / G) * HQ + kv_head * G + packed % G;
            if (splits == 1) {
                *reinterpret_cast<__half2*>(output + id * D + dim) =
                    __floats2half2_rn(acc(j) * inverse[r], acc(j + 1) * inverse[r]);
            } else {
                *reinterpret_cast<float2*>(partial + (id * splits + split) * D + dim) =
                    make_float2(acc(j), acc(j + 1));
            }
        }
    }
    if (splits > 1 && lane % 4 == 0) {
        CUTE_UNROLL
        for (int r = 0; r < 2; ++r) {
            int const packed = wg_row + base_row + 8 * r;
            if (packed < q_len * G) {
                int const id = (begin_q + packed / G) * HQ + kv_head * G + packed % G;
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

template<int N, bool RegisterPV>
bool validate_layouts_for() {
    using T = Traits<N, RegisterPV>;
    using QKMma = typename T::QKMma;
    using PVMma = typename T::PVMma;
    hopper_layouts::validate_storage<N>();
    for (int lane = 0; lane < 128; ++lane) {
        auto score_coord = QKMma{}.get_slice(lane).partition_C(make_identity_tensor(Shape<_64,Int<N>>{}));
        auto out_coord = PVMma{}.get_slice(lane).partition_C(make_identity_tensor(Shape<_64,_128>{}));
        int base = (lane / 32) * 16 + (lane % 32) / 4, column = 2 * (lane % 4);
        // The kernel derives every accumulator element's row and column from
        // (lane, j) at compile time; prove that mapping exhaustively.
        for (int j = 0; j < size(score_coord); ++j) {
            TORCH_CHECK(int(get<0>(score_coord(j))) == base + 8 * row_half(j) &&
                        int(get<1>(score_coord(j))) == column + column_offset(j),
                        "QK accumulator element mapping differs");
        }
        for (int j = 0; j < size(out_coord); ++j) {
            TORCH_CHECK(int(get<0>(out_coord(j))) == base + 8 * row_half(j) &&
                        int(get<1>(out_coord(j))) == column + column_offset(j),
                        "PV accumulator element mapping differs");
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

// Construct a compact sequence/query-tile worklist on device. Row 0 is a
// header [query rows per tile, capacity] that the kernel checks against its
// own tile height. A host-computable capacity avoids a GPU->CPU count readback:
// sum ceil(q_i*G/R) <= ceil(total_q*G/R) + batch - 1. Unused capacity is
// marked -1 and exits early.
__global__ void construct_worklist(int const* cu, int* work, int batch, int capacity, int rows) {
    __shared__ int prefix[257];
    if (threadIdx.x == 0) {
        prefix[0] = 0;
        for (int i = 0; i < batch; ++i)
            prefix[i + 1] = prefix[i] + ((cu[i + 1] - cu[i]) * G + rows - 1) / rows;
        work[0] = rows;
        work[1] = capacity;
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
        work[(i + 1) * 2] = seq;
        work[(i + 1) * 2 + 1] = tile;
    }
}

torch::Tensor prepare_worklist(torch::Tensor cu, int64_t total_queries, int64_t consumers) {
    TORCH_CHECK(cu.is_cuda() && cu.is_contiguous() && cu.scalar_type() == at::kInt &&
                cu.dim() == 1 && cu.numel() >= 2 && cu.numel() <= 257,
                "compact scheduling requires contiguous CUDA int32 offsets for 1..256 sequences");
    TORCH_CHECK(total_queries > 0 && total_queries <= 1048576, "invalid compact query count");
    TORCH_CHECK(consumers == 1 || consumers == 2, "consumer warpgroups must be 1 or 2");
    c10::cuda::CUDAGuard guard(cu.device());
    int batch = cu.numel() - 1;
    int rows = M * int(consumers);
    int capacity = (total_queries * G + rows - 1) / rows + batch - 1;
    auto work = torch::empty({capacity + 1, 2}, cu.options());
    construct_worklist<<<1, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        cu.data_ptr<int>(), work.data_ptr<int>(), batch, capacity, rows);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return work;
}

struct LaunchArgs {
    torch::Tensor q, cu, table, lengths, k, v, out, partial, stats, work;
    int splits, query_tiles;
    float scale;
    bool causal, overlap;
};

template<int N, bool RegisterPV, int Consumers>
void launch_attention(LaunchArgs const& a) {
    using SharedT = Shared<N, RegisterPV, Consumers>;
    static_assert(sizeof(SharedT) <= MaxDynamicShared, "shared-memory plan exceeds the 227 KiB limit");
    auto kernel = attention<N, RegisterPV, Consumers>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(SharedT)));
    static bool const registers_checked = [&] {
        cudaFuncAttributes attrs{};
        C10_CUDA_CHECK(cudaFuncGetAttributes(&attrs, kernel));
        // setmaxnreg.inc can only claim what setmaxnreg.dec released inside
        // this CTA's launch allocation. Fail here rather than hang there.
        TORCH_CHECK(attrs.numRegs * (1 + Consumers) >=
                    Registers<Consumers>::producer + Consumers * Registers<Consumers>::consumer,
                    "launch register allocation (", attrs.numRegs, " per thread) cannot fund ",
                    Consumers, " consumer warpgroups at ", Registers<Consumers>::consumer, " registers");
        return true;
    }();
    (void)registers_checked;
    dim3 grid(a.work.defined() ? a.work.size(0) - 1 : a.lengths.numel(), HKV,
              a.work.defined() ? a.splits : a.query_tiles * a.splits);
    kernel<<<grid, 128 * (1 + Consumers), sizeof(SharedT), at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<H const*>(a.q.data_ptr()), reinterpret_cast<H const*>(a.k.data_ptr()),
        reinterpret_cast<H const*>(a.v.data_ptr()), a.cu.data_ptr<int>(), a.table.data_ptr<int>(),
        a.lengths.data_ptr<int>(), reinterpret_cast<H*>(a.out.data_ptr()), a.partial.data_ptr<float>(),
        a.stats.data_ptr<float>(), a.table.size(1), a.splits, a.query_tiles, a.scale, a.causal, a.overlap,
        a.q.stride(0), a.q.stride(1), a.q.stride(2), a.work.defined() ? a.work.data_ptr<int>() : nullptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int N, bool RegisterPV>
void launch_for_consumers(LaunchArgs const& a, int consumers) {
    if (consumers == 2) launch_attention<N, RegisterPV, 2>(a);
    else launch_attention<N, RegisterPV, 1>(a);
}

torch::Tensor hopper_attention_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                      torch::Tensor cu, torch::Tensor table, torch::Tensor lengths,
                      int64_t max_query, int64_t splits, double scale, bool causal, bool overlap_qk,
                      int64_t tile_n, bool register_pv, int64_t consumers, bool compact,
                      c10::optional<torch::Tensor> prepared_worklist) {
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
    TORCH_CHECK(consumers == 1 || consumers == 2, "consumer warpgroups must be 1 or 2");
    TORCH_CHECK(q.numel() * splits <= INT32_MAX, "split buffers exceed int32 indexing limit");
    torch::Tensor work;
    if (prepared_worklist.has_value()) {
        TORCH_CHECK(compact, "prepared worklist requires compact scheduling");
        work = prepared_worklist.value();
        TORCH_CHECK(work.device() == q.device() && work.scalar_type() == at::kInt &&
                    work.is_contiguous() && work.dim() == 2 && work.size(1) == 2 && work.size(0) > 1,
                    "invalid prepared worklist");
    } else if (compact) work = prepare_worklist(cu, q.size(0), consumers);
    auto out = torch::empty(q.sizes(), q.options());
    auto partial = torch::empty({splits > 1 ? q.numel() * splits : 0}, q.options().dtype(at::kFloat));
    auto stats = torch::empty({splits > 1 ? q.size(0) * HQ * splits * 2 : 0}, q.options().dtype(at::kFloat));
    int rows = M * int(consumers);
    int query_tiles = (max_query * G + rows - 1) / rows;
    TORCH_CHECK(int64_t(query_tiles) * splits <= 65535, "query grid exceeds CUDA z limit");
    LaunchArgs args{q, cu, table, lengths, k, v, out, partial, stats, work,
                    int(splits), query_tiles, float(scale), causal, overlap_qk};
    int c = int(consumers);
    if (tile_n == 64) {
        if (register_pv) launch_for_consumers<64, true>(args, c);
        else launch_for_consumers<64, false>(args, c);
    } else {
        if (register_pv) launch_for_consumers<128, true>(args, c);
        else launch_for_consumers<128, false>(args, c);
    }
    if (splits > 1) {
        combine<<<q.size(0)*HQ, D, 0, at::cuda::getCurrentCUDAStream()>>>(
            partial.data_ptr<float>(), stats.data_ptr<float>(), reinterpret_cast<H*>(out.data_ptr()), splits);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}

template<int N, bool RegisterPV, int Consumers>
pybind11::dict variant_info() {
    using SharedT = Shared<N, RegisterPV, Consumers>;
    cudaFuncAttributes attrs{};
    C10_CUDA_CHECK(cudaFuncGetAttributes(&attrs, attention<N, RegisterPV, Consumers>));
    pybind11::dict result;
    result["tile_n"] = N;
    result["register_pv"] = RegisterPV;
    result["consumers"] = Consumers;
    result["stages"] = SharedT::Stages;
    result["threads_per_cta"] = 128 * (1 + Consumers);
    result["compiler_registers_per_thread"] = attrs.numRegs;
    result["local_bytes_per_thread"] = attrs.localSizeBytes;
    result["dynamic_shared_bytes_per_cta"] = sizeof(SharedT);
    result["producer_registers"] = Registers<Consumers>::producer;
    result["consumer_registers"] = Registers<Consumers>::consumer;
    // Dynamic setmaxnreg changes the producer/consumer split; the compiler
    // register count alone is not an achieved-occupancy measurement. Nonzero
    // local bytes mean spills and should be treated as a performance defect.
    return result;
}

pybind11::list kernel_info() {
    pybind11::list result;
    result.append(variant_info<64, false, 1>());
    result.append(variant_info<64, true, 1>());
    result.append(variant_info<128, false, 1>());
    result.append(variant_info<128, true, 1>());
    result.append(variant_info<64, false, 2>());
    result.append(variant_info<64, true, 2>());
    result.append(variant_info<128, false, 2>());
    result.append(variant_info<128, true, 2>());
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.attr("abi_version") = 5;
    module.attr("source_sha256") = HOPPER_SOURCE_HASH;
    module.def("forward", &::hopper_attention_forward);
    module.def("validate_layouts", &validate_layouts);
    module.def("prepare_worklist", &prepare_worklist);
    module.def("kernel_info", &kernel_info);
}
