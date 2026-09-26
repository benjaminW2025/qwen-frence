// CPU regression using the exact layout definitions compiled into the kernel.
// Build with a host C++17 compiler, CUTLASS and CUDA headers; no GPU/runtime link.
#include "../../custom_kernels/hopper_attention/shared_layouts.cuh"
#include <iostream>

template<int N>
void check() {
    using namespace hopper_layouts;
    validate_storage<N>();
    int bad_k = validate_tensor_storage<K<N, uint16_t>, N, 128>("K", true);
    int bad_v = validate_tensor_storage<V<N, uint16_t>, N, 128, true>("V", true);
    if (bad_k == 0 || bad_v == 0)
        throw std::runtime_error("regression did not detect the previous raw-array stores");
    std::cout << "N=" << N << ": Q/K/V/P tensor addresses and stores pass all 8 base alignments; "
              << "old raw stores fail K=" << bad_k << ", V=" << bad_v << " elements\n";
}

int main() {
    check<64>();
    check<128>();
}
