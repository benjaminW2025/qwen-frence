#pragma once

#include <cute/tensor.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace hopper_layouts {
using namespace cute;

template<class E>
using Q = decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<E>{}, Shape<_64,_128>{}, Step<_1,_2>{}));
template<int N, class E>
using K = decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<E>{}, Shape<Int<N>,_128>{}, Step<_1,_2>{}));
template<int N, class E>
using V = decltype(tile_to_shape(GMMA::Layout_MN_SW128_Atom<E>{}, Shape<_128,Int<N>>{}, Step<_2,_1>{}));
template<int N, class E>
using P = decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<E>{}, Shape<_64,Int<N>>{}));

// Runs entirely on the host. uint16_t exercises the same 16-bit address mapping
// as FP16 without involving floating-point arithmetic or a CUDA device.
// Check real tensor REFERENCES: Layout{}(coord) alone applies a byte swizzle to
// an element index and is not a shared-memory physical address.
template<class Layout, int Rows, int Columns, bool Transpose = false>
int validate_tensor_storage(char const* name, bool reproduce_raw_bug = false) {
    static_assert(Columns % 64 == 0);
    static_assert(cosize_v<Layout> == Rows * Columns);
    alignas(1024) std::array<uint16_t, Rows * Columns + 512> storage{};
    int raw_mismatches = 0;
    // WGMMA buffers are 128B aligned, so cover every base within a 1KiB
    // swizzle period, rather than assuming that the base's swizzle is zero.
    for (int base_bytes = 0; base_bytes < 1024; base_bytes += 128) {
        auto* base = storage.data() + base_bytes / 2;
        auto tensor = make_tensor(make_smem_ptr(base), Layout{});
        auto address = reinterpret_cast<uintptr_t>(base);
        for (int row = 0; row < Rows; ++row) {
            for (int col = 0; col < Columns; ++col) {
                int a = Transpose ? col : row, b = Transpose ? row : col;
                int linear = (col / 64) * Rows * 64 + row * 64 + col % 64;
                uintptr_t unswizzled = address + 2 * linear;
                // SM90 128B swizzle exchanges 16B chunks using address bits
                // 7..9. This independently expresses the hardware mapping.
                uintptr_t expected = unswizzled ^ ((unswizzled & 0x380u) >> 3);
                uintptr_t actual = reinterpret_cast<uintptr_t>(&tensor(a, b));
                if (actual != expected)
                    throw std::runtime_error(std::string(name) + " tensor address mismatch at row=" +
                        std::to_string(row) + " col=" + std::to_string(col) +
                        " base_bytes=" + std::to_string(base_bytes));
                uint16_t tag = uint16_t(row * Columns + col + 1);
                tensor(a, b) = tag;
                if (base[(expected - address) / 2] != tag)
                    throw std::runtime_error(std::string(name) + " tensor store mismatch");
            }
        }
        if (reproduce_raw_bug && base_bytes == 0) {
            for (int row = 0; row < Rows; ++row)
                for (int col = 0; col < Columns; ++col) {
                    int a = Transpose ? col : row, b = Transpose ? row : col;
                    base[Layout{}(a, b)] = uint16_t(row * Columns + col + 1);
                }
            for (int row = 0; row < Rows; ++row)
                for (int col = 0; col < Columns; ++col) {
                    int a = Transpose ? col : row, b = Transpose ? row : col;
                    raw_mismatches += tensor(a, b) != uint16_t(row * Columns + col + 1);
                }
        }
    }
    return raw_mismatches;
}

template<int N>
void validate_storage() {
    validate_tensor_storage<Q<uint16_t>, 64, 128>("Q");
    validate_tensor_storage<K<N, uint16_t>, N, 128>("K");
    validate_tensor_storage<V<N, uint16_t>, N, 128, true>("V");
    validate_tensor_storage<P<N, uint16_t>, 64, N>("P");
}
} // namespace hopper_layouts
