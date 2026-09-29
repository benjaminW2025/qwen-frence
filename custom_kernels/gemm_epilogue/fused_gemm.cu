// SM90a GEMMs with project-owned epilogues on CUTLASS's unmodified mainloop.
//
// The mainloop is CUTLASS v3.9.2's warp-specialized cooperative TMA/WGMMA
// collective (CollectiveBuilder), unmodified, in two configurations:
//   Prefill  128x256x64 tiles, 1x2 cluster, persistent tile scheduler.
//   Decode   128x128x64 tiles, no cluster, stream-K, for M <= 64: decode GEMMs
//            stream weights, and an o_proj/down_proj output is only 12 tiles
//            wide, so K is split across SMs to use the whole memory system.
// Only the epilogue differs: it consumes the FP32 accumulators in registers. An earlier
// Triton GEMM with a fused epilogue lost 5.7% end to end because its mainloop
// was slower than cuBLAS; this keeps the mainloop and changes only the epilogue.
//
// Epilogues (one decoder layer, prefill-sized M):
//   residual   o_proj / down_proj: out = fp16(acc + residual); per-row partial
//              sums of squares of the rounded output, one per 256-column tile.
//   swiglu     gate_up: silu(g * r) * (u * r) from block-8 interleaved weights.
//   qkv        QKV: acc * r + bias, rotate-half RoPE on Q/K, Q out, K/V straight
//              into the paged cache (live rows only; padded Q rows are zero).
// r = rsqrt(sum(partials) / hidden + eps) is the next RMSNorm's per-row factor.
// The norm weight gamma is folded into the consumer's weight at load time:
// rmsnorm(x) @ W.T == r * (x @ (W * gamma).T), so no RMSNorm kernel remains.
//
// Every index scheme relies on the SM90 accumulator geometry: a thread's
// element j sits at column c0 + (j % 2) + 8 * (j / 4) and row m0 + 8 * ((j / 2) % 2),
// with c0 < 8 and quads (lanes 4k..4k+3) sharing rows. validate_fragment proves
// this for every thread of the mainloop's real TiledMma at import.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/default_epilogue.hpp>
#include <cutlass/epilogue/collective/detail.hpp>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/util/packed_stride.hpp>
#include <cstdint>
#include <string>
#include <tuple>
#include <type_traits>

using namespace cute;
using H = cutlass::half_t;
constexpr int Block = 8;          // gate/up interleave granularity (columns)
constexpr int TileM = 128;
constexpr int DecodeRows = 64;    // M at or below this uses the decode configuration
constexpr int HeadDim = 128, RopeHalf = 64, QHeads = 12, KVHeads = 2;
constexpr int QkvWidth = (QHeads + 2 * KVHeads) * HeadDim;  // 2048

CUTLASS_DEVICE bool aligned4(void const* pointer) { return reinterpret_cast<uintptr_t>(pointer) % 4 == 0; }
bool host_aligned4(void const* pointer) { return reinterpret_cast<uintptr_t>(pointer) % 4 == 0; }

// Shared collective-interface plumbing; derived classes add can_implement and operator().
template <class Args>
struct EpilogueBase {
    using ThreadEpilogueOp = cutlass::epilogue::thread::LinearCombination<H, 1, float, float>;  // unused
    using ElementC = H;
    using StrideC = cute::Stride<int64_t, cute::Int<1>, int64_t>;
    using ElementD = H;
    using StrideD = StrideC;
    using ElementAccumulator = float;
    struct SharedStorage { };
    using TensorStorage = SharedStorage;
    using Arguments = Args;
    using Params = Args;

    template <class P> static constexpr Params to_underlying_arguments(P const&, Arguments const& a, void*) { return a; }
    template <class P> static size_t get_workspace_size(P const&, Arguments const&) { return 0; }
    template <class P> static cutlass::Status initialize_workspace(P const&, Arguments const&, void*, cudaStream_t,
                                                                   cutlass::CudaHostAdapter* = nullptr) {
        return cutlass::Status::kSuccess;
    }
    CUTLASS_HOST_DEVICE EpilogueBase(Params const& p, SharedStorage const& = SharedStorage()) : params(p) { }
    CUTLASS_DEVICE bool is_source_needed() { return false; }
    Params params;
};

// Global (m, n) coordinates of every accumulator element this thread holds.
template <class ProblemShape, class BlockShape, class BlockCoord, class TiledMma>
CUTLASS_DEVICE auto thread_coordinates(ProblemShape problem, BlockShape block, BlockCoord coord,
                                       TiledMma tiled_mma, int thread_idx) {
    auto [m_coord, n_coord, k_coord, l_coord] = coord;
    Tensor tile = local_tile(make_identity_tensor(make_shape(get<0>(problem), get<1>(problem))),
                             take<0, 2>(block), make_coord(m_coord, n_coord));
    return tiled_mma.get_thread_slice(thread_idx).partition_C(tile);
}

// RMSNorm factor of row m from the producer's per-tile partial sums of squares.
struct RowScale {
    float const* partials = nullptr;  // [M, count] FP32; null disables scaling
    int64_t ld = 0;
    int count = 0;
    float hidden = 0.f;
    float eps = 0.f;
    CUTLASS_DEVICE float operator()(int m) const {
        if (partials == nullptr) return 1.f;
        float sum = 0.f;
        for (int t = 0; t < count; ++t) sum += partials[int64_t(m) * ld + t];  // fixed order: deterministic
        return 1.f / sqrtf(sum / hidden + eps);
    }
};

// ---------------------------------------------------------------- residual
struct ResidualArgs {
    H const* residual = nullptr; int64_t ld_residual = 0;   // may alias out
    H* out = nullptr; int64_t ld_out = 0;
    float* partials = nullptr; int64_t ld_partials = 0;     // optional [M, N / TileN]
};
template <int TileN>
struct ResidualEpilogue : EpilogueBase<ResidualArgs> {
    using EpilogueBase<ResidualArgs>::EpilogueBase;
    using typename EpilogueBase<ResidualArgs>::Arguments;
    template <class P> static bool can_implement(P const& shape, Arguments const& a) {
        auto n = get<1>(shape);
        return a.residual && a.out && n % TileN == 0 && a.ld_residual >= n && a.ld_out >= n &&
               a.ld_residual % 2 == 0 && a.ld_out % 2 == 0 && host_aligned4(a.residual) && host_aligned4(a.out) &&
               (a.partials == nullptr || a.ld_partials >= n / TileN);
    }
    template <class ProblemShapeMNKL, class BlockShapeMNK, class BlockCoordMNKL,
              class FrgEngine, class FrgLayout, class TiledMma, class ResidueMNK>
    CUTLASS_DEVICE void operator()(ProblemShapeMNKL problem, BlockShapeMNK block, BlockCoordMNKL coord,
                                   cute::Tensor<FrgEngine, FrgLayout> const& acc, TiledMma tiled_mma,
                                   ResidueMNK, int thread_idx, char*) {
        auto M = get<0>(problem);
        auto N = get<1>(problem);
        Tensor c = thread_coordinates(problem, block, coord, tiled_mma, thread_idx);
        float square[2] = {0.f, 0.f};
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < size(acc); j += 2) {
            int const m = get<0>(c(j)), n = get<1>(c(j));
            if (m < M && n < N) {
                auto const& params = this->params;
                float2 const r = __half22float2(*reinterpret_cast<__half2 const*>(
                    reinterpret_cast<__half const*>(params.residual) + int64_t(m) * params.ld_residual + n));
                __half2 const sum = __floats2half2_rn(acc(j) + r.x, acc(j + 1) + r.y);
                *reinterpret_cast<__half2*>(reinterpret_cast<__half*>(params.out) + int64_t(m) * params.ld_out + n) = sum;
                // The next RMSNorm reads the rounded sum, exactly as residual_add_rms_norm does.
                float2 const s = __half22float2(sum);
                square[(j / 2) % 2] += s.x * s.x + s.y * s.y;
            }
        }
        auto const& params = this->params;
        if (params.partials != nullptr) {
            auto [m_coord, n_coord, k_coord, l_coord] = coord;
            CUTLASS_PRAGMA_UNROLL
            for (int half = 0; half < 2; ++half) {
                // Quad lanes share a row; all lanes shuffle, one writes.
                square[half] += __shfl_xor_sync(0xffffffff, square[half], 1);
                square[half] += __shfl_xor_sync(0xffffffff, square[half], 2);
                int const m = get<0>(c(2 * half));
                if (thread_idx % 4 == 0 && m < M)
                    params.partials[int64_t(m) * params.ld_partials + n_coord] = square[half];
            }
        }
    }
};

// ---------------------------------------------------------------- swiglu
struct SwigluArgs {
    H* out = nullptr; int64_t ld_out = 0;                   // [M, F]
    RowScale scale;
};
struct SwigluEpilogue : EpilogueBase<SwigluArgs> {
    using EpilogueBase::EpilogueBase;
    template <class P> static bool can_implement(P const& shape, Arguments const& a) {
        auto n = get<1>(shape);
        return a.out && n % (2 * Block) == 0 && a.ld_out >= n / 2 && a.ld_out % 2 == 0 && host_aligned4(a.out);
    }
    template <class ProblemShapeMNKL, class BlockShapeMNK, class BlockCoordMNKL,
              class FrgEngine, class FrgLayout, class TiledMma, class ResidueMNK>
    CUTLASS_DEVICE void operator()(ProblemShapeMNKL problem, BlockShapeMNK block, BlockCoordMNKL coord,
                                   cute::Tensor<FrgEngine, FrgLayout> const& acc, TiledMma tiled_mma,
                                   ResidueMNK, int thread_idx, char*) {
        auto M = get<0>(problem);
        auto N = get<1>(problem);
        Tensor c = thread_coordinates(problem, block, coord, tiled_mma, thread_idx);
        float row_scale[2];
        CUTLASS_PRAGMA_UNROLL
        for (int half = 0; half < 2; ++half) {
            int const m = get<0>(c(2 * half));
            row_scale[half] = m < M ? params.scale(m) : 1.f;
        }
        // Elements j..j+3 are gate columns c, c+1 (two rows); j+4..j+7 their up partners, +8 columns.
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(acc); i += 8) {
            CUTLASS_PRAGMA_UNROLL
            for (int half = 0; half < 2; ++half) {
                int const gate = i + 2 * half;
                int const m = get<0>(c(gate)), n = get<1>(c(gate));
                if (m < M && n < N) {
                    float const s = row_scale[half];
                    float const g0 = acc(gate) * s, g1 = acc(gate + 1) * s;
                    float const u0 = acc(gate + 4) * s, u1 = acc(gate + 5) * s;
                    int64_t const column = int64_t(n / (2 * Block)) * Block + n % Block;
                    *reinterpret_cast<__half2*>(reinterpret_cast<__half*>(params.out) + int64_t(m) * params.ld_out + column) =
                        __floats2half2_rn(g0 / (1.f + expf(-g0)) * u0, g1 / (1.f + expf(-g1)) * u1);
                }
            }
        }
    }
};

// ---------------------------------------------------------------- qkv
struct QkvArgs {
    H* q = nullptr; int64_t ld_q = 0;                       // [M, 12 * 128]
    H* k_cache = nullptr; H* v_cache = nullptr;             // flat [slots, 2, 128]
    int64_t const* positions = nullptr;                     // [M]
    int64_t const* slots = nullptr;                         // [M]
    int const* valid_tokens = nullptr;                      // optional device scalar
    H const* bias = nullptr;                                // optional [2048]
    float log_theta = 0.f;
    RowScale scale;
};
template <int TileN>
struct QkvEpilogue : EpilogueBase<QkvArgs> {
    static_assert(TileN % HeadDim == 0, "a tile must hold whole heads");
    using EpilogueBase<QkvArgs>::EpilogueBase;
    using typename EpilogueBase<QkvArgs>::Arguments;
    template <class P> static bool can_implement(P const& shape, Arguments const& a) {
        return get<1>(shape) == QkvWidth && a.q && a.k_cache && a.v_cache && a.positions && a.slots &&
               a.ld_q >= QHeads * HeadDim && a.ld_q % 2 == 0 && host_aligned4(a.q) &&
               host_aligned4(a.k_cache) && host_aligned4(a.v_cache);
    }
    template <class ProblemShapeMNKL, class BlockShapeMNK, class BlockCoordMNKL,
              class FrgEngine, class FrgLayout, class TiledMma, class ResidueMNK>
    CUTLASS_DEVICE void operator()(ProblemShapeMNKL problem, BlockShapeMNK block, BlockCoordMNKL coord,
                                   cute::Tensor<FrgEngine, FrgLayout> const& acc, TiledMma tiled_mma,
                                   ResidueMNK, int thread_idx, char*) {
        static_assert(decltype(size(acc))::value % 64 == 0, "RoPE partners must stay inside the fragment");
        auto const& params = this->params;
        auto M = get<0>(problem);
        Tensor c = thread_coordinates(problem, block, coord, tiled_mma, thread_idx);
        auto [m_coord, n_coord, k_coord, l_coord] = coord;
        // Every tile holds whole heads of one kind: Q heads 0-11, then K 12-13, then V 14-15.
        bool const is_q = n_coord < QHeads * HeadDim / TileN;
        bool const is_v = n_coord >= (QHeads + KVHeads) * HeadDim / TileN;
        int const live = params.valid_tokens != nullptr ? *params.valid_tokens : int(M);
        float row_scale[2];
        float position[2];
        int64_t slot[2];
        CUTLASS_PRAGMA_UNROLL
        for (int half = 0; half < 2; ++half) {
            int const m = get<0>(c(2 * half));
            bool const live_row = m < M && m < live;
            row_scale[half] = m < M ? params.scale(m) : 1.f;
            position[half] = live_row ? float(params.positions[m]) : 0.f;
            slot[half] = live_row ? params.slots[m] : 0;
        }
        __half* const cache = reinterpret_cast<__half*>(is_v ? params.v_cache : params.k_cache);
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < size(acc); j += 2) {
            // Element j is in the first rotation half (d < 64) iff (j / 4) % 16 < 8, since c0 < 8.
            if ((j / 4) % 16 >= 8) continue;
            int const half = (j / 2) % 2;
            int const m = get<0>(c(j)), n = get<1>(c(j));
            if (m >= M) continue;
            bool const live_row = m < live;
            int const head = n / HeadDim, d = n % HeadDim;
            float const s = row_scale[half];
            auto value = [&](int element, int column) {
                float v = acc(element) * s;
                return params.bias != nullptr ? v + float(params.bias[column]) : v;
            };
            float first[2] = {value(j, n), value(j + 1, n + 1)};
            float second[2] = {value(j + 32, n + RopeHalf), value(j + 33, n + RopeHalf + 1)};
            if (!is_v) {
                CUTLASS_PRAGMA_UNROLL
                for (int e = 0; e < 2; ++e) {
                    float const inverse = expf(float(d + e) * (-params.log_theta / RopeHalf));
                    float sine, cosine;
                    sincosf(position[half] * inverse, &sine, &cosine);
                    float const a = first[e], b = second[e];
                    first[e] = a * cosine - b * sine;
                    second[e] = a * sine + b * cosine;
                }
            }
            if (is_q) {
                __half* const row = reinterpret_cast<__half*>(params.q) + int64_t(m) * params.ld_q;
                // Padded rows get deterministic zeros, matching packed_qkv_rope_cache.
                *reinterpret_cast<__half2*>(row + n) =
                    live_row ? __floats2half2_rn(first[0], first[1]) : __floats2half2_rn(0.f, 0.f);
                *reinterpret_cast<__half2*>(row + n + RopeHalf) =
                    live_row ? __floats2half2_rn(second[0], second[1]) : __floats2half2_rn(0.f, 0.f);
            } else if (live_row) {
                int const kv_head = head - (is_v ? QHeads + KVHeads : QHeads);
                int64_t const base = (slot[half] * KVHeads + kv_head) * HeadDim + d;
                *reinterpret_cast<__half2*>(cache + base) = __floats2half2_rn(first[0], first[1]);
                *reinterpret_cast<__half2*>(cache + base + RopeHalf) = __floats2half2_rn(second[0], second[1]);
            }
        }
    }
};

// ---------------------------------------------------------------- kernels
using LayoutA = cutlass::layout::RowMajor;     // activations [M, K]
using LayoutB = cutlass::layout::ColumnMajor;  // nn.Linear weight [N, K] row-major == [K, N] column-major
using Schedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;

template <int TileN_, int ClusterM, int ClusterN, bool StreamK>
struct Config {
    static constexpr int TileN = TileN_;
    using TileShape = Shape<Int<TileM>, Int<TileN>, _64>;
    using ClusterShape = Shape<Int<ClusterM>, Int<ClusterN>, _1>;
    using Scheduler = std::conditional_t<StreamK, cutlass::gemm::StreamKScheduler, void>;
    static constexpr char const* name() {
        return StreamK ? "stream-K" : "persistent";
    }
    static constexpr int cluster_m = ClusterM, cluster_n = ClusterN;
    static constexpr bool stream_k = StreamK;
};
// Defaults. Prefill: 1x2 along N, since a 128-row bucket is one M tile and a
// 2x1 cluster would idle half its CTAs. Decode (M <= 64): stream-K, because
// decode GEMMs stream weights and o_proj/down_proj are only 12 tiles wide.
using PrefillConfig = Config<256, 1, 2, false>;
using DecodeConfig = Config<128, 1, 1, true>;
// Candidates 0 and 1 are always the defaults. A GEMM_EPILOGUE_SWEEP build adds
// the rest for experiments/prefill/sweep_epilogue_tiles.py; the default build
// keeps compile time down.
#ifdef GEMM_EPILOGUE_SWEEP
using Candidates = std::tuple<PrefillConfig, DecodeConfig,
                              Config<256, 1, 1, false>, Config<256, 2, 1, false>, Config<256, 1, 1, true>,
                              Config<128, 1, 2, false>, Config<128, 2, 1, false>, Config<128, 1, 1, false>>;
constexpr bool SweepBuild = true;
#else
using Candidates = std::tuple<PrefillConfig, DecodeConfig>;
constexpr bool SweepBuild = false;
#endif
constexpr int CandidateCount = int(std::tuple_size_v<Candidates>);

template <class T> struct Tag { using type = T; };

// Call f(Tag<Config>{}) for candidate `index` (C++17: a generic lambda reads the type from the tag).
template <int I = 0, class F>
void with_candidate(int index, F&& f) {
    if constexpr (I < CandidateCount) {
        if (index == I) return f(Tag<std::tuple_element_t<I, Candidates>>{});
        return with_candidate<I + 1>(index, std::forward<F>(f));
    } else {
        TORCH_CHECK(false, "GEMM config ", index, " is not in this build (", CandidateCount, " candidates)");
    }
}

int resolve_config(int64_t config, int64_t rows) {
    if (config < 0) return rows <= DecodeRows ? 1 : 0;
    TORCH_CHECK(config < CandidateCount, "GEMM config ", config, " is not in this build (", CandidateCount,
                " candidates); rebuild with GEMM_EPILOGUE_SWEEP=1");
    return int(config);
}

template <class Config, class Epilogue>
using MainloopFor = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    H, LayoutA, 8, H, LayoutB, 8, float, typename Config::TileShape, typename Config::ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
    Schedule>::CollectiveOp;

template <class Epilogue>
using Adapted = cutlass::epilogue::collective::detail::Sm90TmaWarpSpecializedAdapter<Epilogue>;
template <class Config, class Epilogue>
using GemmFor = cutlass::gemm::device::GemmUniversalAdapter<cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, MainloopFor<Config, Adapted<Epilogue>>, Adapted<Epilogue>, typename Config::Scheduler>>;

template <class Config> using ResidualGemm = GemmFor<Config, ResidualEpilogue<Config::TileN>>;
template <class Config> using SwigluGemm = GemmFor<Config, SwigluEpilogue>;
template <class Config> using QkvGemm = GemmFor<Config, QkvEpilogue<Config::TileN>>;

// Same prefill mainloop, plain [M, N] output: attributes any fused win to the epilogue.
using PlainEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    typename PrefillConfig::TileShape, typename PrefillConfig::ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto, float, float,
    H, cutlass::layout::RowMajor, 8, H, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::NoSmemWarpSpecialized>::CollectiveOp;
using PlainGemm = cutlass::gemm::device::GemmUniversalAdapter<cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, MainloopFor<PrefillConfig, PlainEpilogue>, PlainEpilogue>>;

template <class Config>
using TiledMmaFor = typename MainloopFor<Config, Adapted<SwigluEpilogue>>::TiledMma;
template <class Config>
constexpr bool shares_tiled_mma =
    std::is_same_v<TiledMmaFor<Config>, typename MainloopFor<Config, Adapted<ResidualEpilogue<Config::TileN>>>::TiledMma> &&
    std::is_same_v<TiledMmaFor<Config>, typename MainloopFor<Config, Adapted<QkvEpilogue<Config::TileN>>>::TiledMma>;
template <class Tuple> struct AllShare;
template <class... Cs> struct AllShare<std::tuple<Cs...>> {
    static constexpr bool value = (shares_tiled_mma<Cs> && ...);
};
static_assert(AllShare<Candidates>::value, "all epilogues of a configuration must share one mainloop");

// Prove the fragment geometry every epilogue's index math assumes, for every
// thread of a configuration's real TiledMma, instead of trusting the documented layout.
template <class Config>
void validate_fragment_for(char const* name) {
    TiledMmaFor<Config> tiled_mma;
    int const threads = int(size(tiled_mma));
    TORCH_CHECK(threads == 256, name, ": cooperative mainloop expected 256 MMA threads, got ", threads);
    auto coords_of = [&](int thread) {
        return tiled_mma.get_thread_slice(thread).partition_C(
            make_identity_tensor(make_shape(Int<TileM>{}, Int<Config::TileN>{})));
    };
    for (int thread = 0; thread < threads; ++thread) {
        auto coords = coords_of(thread);
        int const m0 = int(get<0>(coords(0))), c0 = int(get<1>(coords(0)));
        int const count = int(size(coords));
        TORCH_CHECK(count % 64 == 0, name, ": fragment size ", count, " breaks RoPE pairing");
        TORCH_CHECK(c0 == 2 * (thread % 4), name, ": thread ", thread, " starts at column ", c0);
        for (int j = 0; j < count; ++j) {
            TORCH_CHECK(int(get<0>(coords(j))) == m0 + 8 * ((j / 2) % 2) &&
                        int(get<1>(coords(j))) == c0 + j % 2 + 8 * (j / 4),
                        name, ": accumulator element ", j, " of thread ", thread, " is not where the epilogues expect");
        }
        for (int partner : {thread ^ 1, thread ^ 2}) {
            TORCH_CHECK(int(get<0>(coords_of(partner)(0))) == m0,
                        name, ": quad lanes ", thread, "/", partner, " do not share rows");
        }
    }
}

bool validate_fragment() {
    for (int index = 0; index < CandidateCount; ++index) {
        std::string const name = "config " + std::to_string(index);
        with_candidate(index, [&](auto tag) { using C = typename decltype(tag)::type; validate_fragment_for<C>(name.c_str()); });
    }
    return true;
}

pybind11::list configs() {
    pybind11::list result;
    for (int index = 0; index < CandidateCount; ++index) {
        with_candidate(index, [&](auto tag) { using C = typename decltype(tag)::type;
            pybind11::dict row;
            row["index"] = index;
            row["tile"] = pybind11::make_tuple(TileM, C::TileN, 64);
            row["cluster"] = pybind11::make_tuple(C::cluster_m, C::cluster_n);
            row["scheduler"] = C::name();
            row["default_for"] = index == 0 ? "prefill" : index == 1 ? "decode" : "";
            result.append(row);
        });
    }
    return result;
}

cutlass::KernelHardwareInfo hardware_info(int device) {
    static cutlass::KernelHardwareInfo cached[16] = {};
    static bool ready[16] = {};
    TORCH_CHECK(device >= 0 && device < 16, "unsupported device index");
    if (!ready[device]) {
        cached[device].device_id = device;
        cached[device].sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device);
        ready[device] = true;
    }
    return cached[device];
}

void check_operands(torch::Tensor const& x, torch::Tensor const& w) {
    TORCH_CHECK(x.is_cuda() && w.is_cuda() && x.device() == w.device(), "operands must share one CUDA device");
    TORCH_CHECK(x.scalar_type() == at::kHalf && w.scalar_type() == at::kHalf, "FP16 operands required");
    TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && x.size(1) == w.size(1), "x [M, K] and weight [N, K] required");
    TORCH_CHECK(x.is_contiguous() && w.is_contiguous(), "operands must be contiguous");
    TORCH_CHECK(x.size(1) % 8 == 0 && w.size(0) % 8 == 0, "K and N must be multiples of 8 for 16-byte TMA");
    TORCH_CHECK(x.size(0) > 0 && x.size(0) <= INT32_MAX && w.size(0) <= INT32_MAX, "invalid GEMM extent");
    auto props = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(props->major == 9 && props->minor == 0, "requires SM90 Hopper");
    static bool const proven = validate_fragment();
    (void)proven;
}

RowScale row_scale(c10::optional<torch::Tensor> const& partials, torch::Tensor const& x, double hidden, double eps) {
    RowScale scale;
    if (!partials.has_value()) return scale;
    auto const& p = partials.value();
    TORCH_CHECK(p.is_cuda() && p.device() == x.device() && p.scalar_type() == at::kFloat && p.dim() == 2 &&
                p.stride(1) == 1 && p.size(0) >= x.size(0) && p.size(1) > 0,
                "partials must be CUDA float32 [M, tiles] with unit column stride");
    TORCH_CHECK(hidden > 0 && eps > 0, "RMSNorm hidden size and eps must be positive");
    scale.partials = p.data_ptr<float>();
    scale.ld = p.stride(0);
    scale.count = int(p.size(1));
    scale.hidden = float(hidden);
    scale.eps = float(eps);
    return scale;
}

template <class Gemm>
void run(torch::Tensor const& x, torch::Tensor const& w, typename Gemm::GemmKernel::EpilogueArguments epilogue) {
    int M = x.size(0), N = w.size(0), K = x.size(1);
    using Kernel = typename Gemm::GemmKernel;
    auto stride_a = cutlass::make_cute_packed_stride(typename Kernel::StrideA{}, {M, K, 1});
    auto stride_b = cutlass::make_cute_packed_stride(typename Kernel::StrideB{}, {N, K, 1});
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
        {reinterpret_cast<H const*>(x.data_ptr()), stride_a, reinterpret_cast<H const*>(w.data_ptr()), stride_b},
        epilogue, hardware_info(x.get_device())};
    Gemm gemm;
    TORCH_CHECK(gemm.can_implement(args) == cutlass::Status::kSuccess, "CUTLASS cannot implement this GEMM");
    // Stream-K needs a reduction workspace; allocating it through the caching
    // allocator keeps it on the capturing graph's pool under CUDA graph capture.
    size_t const bytes = Gemm::get_workspace_size(args);
    torch::Tensor workspace;
    if (bytes > 0) workspace = torch::empty({int64_t(bytes)}, x.options().dtype(at::kByte));
    auto stream = at::cuda::getCurrentCUDAStream();
    TORCH_CHECK(gemm.initialize(args, bytes > 0 ? workspace.data_ptr() : nullptr, stream) == cutlass::Status::kSuccess,
                "CUTLASS initialize failed");
    TORCH_CHECK(gemm.run(stream) == cutlass::Status::kSuccess, "CUTLASS launch failed");
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::tuple<torch::Tensor, c10::optional<torch::Tensor>> residual_gemm(
        torch::Tensor x, torch::Tensor w, torch::Tensor residual, bool partials, c10::optional<torch::Tensor> out,
        int64_t config) {
    check_operands(x, w);
    c10::cuda::CUDAGuard guard(x.device());
    TORCH_CHECK(residual.is_cuda() && residual.device() == x.device() && residual.scalar_type() == at::kHalf &&
                residual.dim() == 2 && residual.stride(1) == 1 && residual.size(0) >= x.size(0) &&
                residual.size(1) == w.size(0), "residual must be FP16 [M, N] with unit column stride");
    auto result = out.has_value() ? out.value() : torch::empty({x.size(0), w.size(0)}, x.options());
    TORCH_CHECK(result.is_cuda() && result.device() == x.device() && result.scalar_type() == at::kHalf &&
                result.dim() == 2 && result.stride(1) == 1 && result.size(0) >= x.size(0) &&
                result.size(1) == w.size(0), "out must be FP16 [M, N] with unit column stride");
    c10::optional<torch::Tensor> sums;
    with_candidate(resolve_config(config, x.size(0)), [&](auto tag) { using C = typename decltype(tag)::type;
        TORCH_CHECK(w.size(0) % C::TileN == 0, "residual output width must be a multiple of ", C::TileN);
        if (partials) sums = torch::empty({x.size(0), w.size(0) / C::TileN}, x.options().dtype(at::kFloat));
        run<ResidualGemm<C>>(x, w, ResidualArgs{
            reinterpret_cast<H const*>(residual.data_ptr()), residual.stride(0),
            reinterpret_cast<H*>(result.data_ptr()), result.stride(0),
            partials ? sums.value().data_ptr<float>() : nullptr, partials ? sums.value().stride(0) : 0});
    });
    return {result, sums};
}

torch::Tensor gate_up_swiglu(torch::Tensor x, torch::Tensor w, c10::optional<torch::Tensor> partials,
                             double hidden, double eps, int64_t config) {
    check_operands(x, w);
    c10::cuda::CUDAGuard guard(x.device());
    TORCH_CHECK(w.size(0) % (2 * Block) == 0, "interleaved gate/up weight needs 16-row blocks");
    auto out = torch::empty({x.size(0), w.size(0) / 2}, x.options());
    SwigluArgs args{reinterpret_cast<H*>(out.data_ptr()), out.stride(0), row_scale(partials, x, hidden, eps)};
    with_candidate(resolve_config(config, x.size(0)), [&](auto tag) { using C = typename decltype(tag)::type; run<SwigluGemm<C>>(x, w, args); });
    return out;
}

torch::Tensor qkv_rope_cache(torch::Tensor x, torch::Tensor w, c10::optional<torch::Tensor> bias,
                             c10::optional<torch::Tensor> partials, double hidden, double eps,
                             torch::Tensor positions, torch::Tensor slots, torch::Tensor k_pool, torch::Tensor v_pool,
                             c10::optional<torch::Tensor> valid_tokens, double theta, int64_t config) {
    check_operands(x, w);
    c10::cuda::CUDAGuard guard(x.device());
    TORCH_CHECK(w.size(0) == QkvWidth, "packed QKV weight must be [2048, K]");
    for (auto const& t : {positions, slots})
        TORCH_CHECK(t.is_cuda() && t.device() == x.device() && t.scalar_type() == at::kLong && t.dim() == 1 &&
                    t.is_contiguous() && t.size(0) >= x.size(0), "positions/slots must be CUDA int64 [M]");
    TORCH_CHECK(k_pool.sizes() == v_pool.sizes() && k_pool.dim() == 4 && k_pool.size(1) == 16 &&
                k_pool.size(2) == KVHeads && k_pool.size(3) == HeadDim, "KV pools must be [pages, 16, 2, 128]");
    for (auto const& t : {k_pool, v_pool})
        TORCH_CHECK(t.is_cuda() && t.device() == x.device() && t.scalar_type() == at::kHalf && t.is_contiguous(),
                    "KV pools must be contiguous FP16 on the operands' device");
    H const* bias_ptr = nullptr;
    if (bias.has_value()) {
        auto const& b = bias.value();
        TORCH_CHECK(b.is_cuda() && b.device() == x.device() && b.scalar_type() == at::kHalf && b.dim() == 1 &&
                    b.is_contiguous() && b.size(0) == QkvWidth, "bias must be contiguous FP16 [2048]");
        bias_ptr = reinterpret_cast<H const*>(b.data_ptr());
    }
    int const* valid_ptr = nullptr;
    if (valid_tokens.has_value()) {
        auto const& v = valid_tokens.value();
        TORCH_CHECK(v.is_cuda() && v.device() == x.device() && v.scalar_type() == at::kInt && v.dim() == 0,
                    "valid_tokens must be a CUDA int32 scalar");
        valid_ptr = v.data_ptr<int>();
    }
    TORCH_CHECK(theta > 1, "RoPE theta must exceed 1");
    auto q = torch::empty({x.size(0), QHeads, HeadDim}, x.options());
    QkvArgs args{reinterpret_cast<H*>(q.data_ptr()), int64_t(QHeads) * HeadDim,
                 reinterpret_cast<H*>(k_pool.data_ptr()), reinterpret_cast<H*>(v_pool.data_ptr()),
                 positions.data_ptr<int64_t>(), slots.data_ptr<int64_t>(), valid_ptr, bias_ptr,
                 float(std::log(theta)), row_scale(partials, x, hidden, eps)};
    with_candidate(resolve_config(config, x.size(0)), [&](auto tag) { using C = typename decltype(tag)::type; run<QkvGemm<C>>(x, w, args); });
    return q;
}

torch::Tensor gemm_plain(torch::Tensor x, torch::Tensor w) {
    check_operands(x, w);
    c10::cuda::CUDAGuard guard(x.device());
    int M = x.size(0), N = w.size(0);
    auto out = torch::empty({x.size(0), w.size(0)}, x.options());
    auto stride_d = cutlass::make_cute_packed_stride(typename PlainGemm::GemmKernel::StrideD{}, {M, N, 1});
    run<PlainGemm>(x, w, {{1.f, 0.f}, nullptr, stride_d, reinterpret_cast<H*>(out.data_ptr()), stride_d});
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.attr("abi_version") = 4;
    module.attr("source_sha256") = GEMM_EPILOGUE_SOURCE_HASH;
    module.attr("interleave_block") = Block;
    module.attr("partial_tile") = PrefillConfig::TileN;
    module.attr("decode_partial_tile") = DecodeConfig::TileN;
    module.attr("decode_rows") = DecodeRows;
    module.attr("sweep_build") = SweepBuild;
    module.def("configs", &configs);
    module.def("residual_gemm", &residual_gemm);
    module.def("gate_up_swiglu", &gate_up_swiglu);
    module.def("qkv_rope_cache", &qkv_rope_cache);
    module.def("gemm_plain", &gemm_plain);
    module.def("validate_fragment", &validate_fragment);
}
