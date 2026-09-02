"""Fused Triton RMSNorm used by the inference engine."""

import torch
import triton
import triton.language as tl

DEVICE = torch.device("cuda")

@triton.jit
def fused_rms_norm(x_ptr,
                   weight_ptr,
                   out_ptr,
                   N, # Number of elements per row
                   x_row_stride,
                   out_row_stride,
                   epsilon,
                   BLOCK_SIZE: tl.constexpr,):
    """
    Given an N long vector x, we divide by (epsilon + 1/N (sum of x_i^2))^(1/2)
    i.e., the root mean square of x.
    """

    # First implement under assumption that a row can load into cache
    pid = tl.program_id(0) # Block ID

    # Load in weights
    weight_offsets = tl.arange(0, BLOCK_SIZE)
    weight_ptrs = weight_ptr + (weight_offsets)
    mask = weight_offsets < N
    weights = tl.load(weight_ptrs, mask=mask, other=0.0)

    # Generate offsets
    row_offsets = tl.arange(0, BLOCK_SIZE)
    row_ptrs = x_ptr + (row_offsets + x_row_stride * pid) # Change row accordingly
    mask = row_offsets < N
    row = tl.load(row_ptrs, mask=mask, other=0.0)

    # Compute norm factor want to accumulate in fp32
    row_f32 = row.to(tl.float32)
    root_mean_square = tl.sqrt(epsilon + tl.sum(row_f32 * row_f32, axis=-1) / N)

    # Normalize input row
    row = (weights * row_f32 / root_mean_square).to(row.dtype)

    # Now generate output offsets
    out_offsets = tl.arange(0, BLOCK_SIZE)
    out_ptrs = out_ptr + (out_offsets + pid * out_row_stride)
    mask = out_offsets < N

    # Store the normalized outputs
    tl.store(out_ptrs, row, mask=mask)


def rms_norm(x, weight, epsilon=1e-6):
    assert x.ndim == 2, "expected a (rows, N) input"
    assert x.is_contiguous(), "kernel assumes row-major contiguous input"
    assert weight.ndim == 1 and weight.shape[0] == x.shape[1], "weight must be (N,)"

    M, N = x.shape
    out = torch.empty_like(x)

    BLOCK_SIZE = triton.next_power_of_2(N)
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)

    grid = (M,)
    fused_rms_norm[grid](
        x, weight, out,
        N,
        x.stride(0), out.stride(0),
        epsilon,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    return out


def rms_norm_reference(x, weight, epsilon=1e-6):
    x_f32 = x.float()
    rms = torch.sqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + epsilon)
    return (weight * x_f32 / rms).to(x.dtype)


def rms_norm_naive(x, weight, epsilon=1e-6):
    x_f32 = x.float()
    rms = torch.rsqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + epsilon)
    return (x_f32 * rms * weight).to(x.dtype)


_compiled_rms = None


def rms_norm_compiled(x, weight, epsilon=1e-6):
    """torch.compile-fused RMSNorm - the *actually* optimized PyTorch baseline.
    TorchInductor fuses the naive decomposition into (ideally) a single kernel."""
    global _compiled_rms
    if _compiled_rms is None:
        _compiled_rms = torch.compile(rms_norm_naive)
    return _compiled_rms(x, weight, epsilon)


def check_correctness(shapes=((128, 512), (256, 1000), (17, 4096), (4096, 8192)),
                       dtype=torch.float16, epsilon=1e-6):
    print(f"{'shape':>16} | {'max_err':>10} | {'status':>6}")

    all_ok = True
    for M, N in shapes:
        x = torch.randn((M, N), device=DEVICE, dtype=dtype)
        weight = torch.randn((N,), device=DEVICE, dtype=dtype)

        out = rms_norm(x, weight, epsilon)
        ref = rms_norm_reference(x, weight, epsilon)

        max_err = (out.float() - ref.float()).abs().max().item()
        ok = max_err < 2e-2  # fp16 has ~1e-3 relative precision; leave headroom
        all_ok &= ok

        print(f"{str((M, N)):>16} | {max_err:>10.5f} | {'OK' if ok else 'FAIL':>6}")

    return all_ok


# Set to your GPU's datasheet peak (GB/s) to see a %-of-peak column; None hides it.
PEAK_BANDWIDTH_GBPS = None


def benchmark_bandwidth(M=8192, Ns=(512, 1024, 2048, 4096, 8192, 16384),
                         dtype=torch.float16, epsilon=1e-6):
    dtype_bytes = torch.tensor([], dtype=dtype).element_size()

    header = f"{'N':>8} | {'time (ms)':>10} | {'GB/s':>10}"
    if PEAK_BANDWIDTH_GBPS:
        header += f" | {'% peak':>8}"
    print(header)

    results = []
    for N in Ns:
        x = torch.randn((M, N), device=DEVICE, dtype=dtype)
        weight = torch.randn((N,), device=DEVICE, dtype=dtype)

        ms = triton.testing.do_bench(lambda: rms_norm(x, weight, epsilon))

        # Read x + write out. Weight is only N elements (vs. x/out's M*N) and
        # stays hot in L2 across the M row-programs, so its traffic is
        # negligible and left out here.
        bytes_moved = 2 * M * N * dtype_bytes
        gbps = bytes_moved / (ms * 1e-3) / 1e9

        row = f"{N:>8} | {ms:>10.4f} | {gbps:>10.2f}"
        if PEAK_BANDWIDTH_GBPS:
            row += f" | {gbps / PEAK_BANDWIDTH_GBPS * 100:>7.1f}%"
        print(row)

        results.append((N, ms, gbps))

    return results


def benchmark_vs_torch(M=8192, Ns=(512, 1024, 2048, 4096, 8192, 16384),
                       dtype=torch.float16, epsilon=1e-6):
    header = (f"{'N':>8} | {'triton (ms)':>12} | {'compiled (ms)':>14} | "
              f"{'naive (ms)':>12} | {'vs compiled':>12} | {'vs naive':>9} | {'max_err':>8}")
    print(header)
    print("-" * len(header))

    results = []
    for N in Ns:
        x = torch.randn((M, N), device=DEVICE, dtype=dtype)
        weight = torch.randn((N,), device=DEVICE, dtype=dtype)

        triton_ms = triton.testing.do_bench(lambda: rms_norm(x, weight, epsilon))
        compiled_ms = triton.testing.do_bench(lambda: rms_norm_compiled(x, weight, epsilon))
        naive_ms = triton.testing.do_bench(lambda: rms_norm_naive(x, weight, epsilon))

        # speedup > 1 means the Triton kernel is faster than that baseline.
        vs_compiled = compiled_ms / triton_ms
        vs_naive = naive_ms / triton_ms

        # Sanity: kernel still matches the fp32 reference.
        max_err = (rms_norm(x, weight, epsilon).float()
                   - rms_norm_reference(x, weight, epsilon).float()).abs().max().item()

        print(f"{N:>8} | {triton_ms:>12.4f} | {compiled_ms:>14.4f} | "
              f"{naive_ms:>12.4f} | {vs_compiled:>11.2f}x | {vs_naive:>8.2f}x | {max_err:>8.4f}")

        results.append((N, triton_ms, compiled_ms, naive_ms, vs_compiled, vs_naive))

    return results


if __name__ == "__main__":
    check_correctness()
    print()
    benchmark_bandwidth()
    print()
    benchmark_vs_torch()
