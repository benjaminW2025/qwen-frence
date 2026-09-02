"""Qwen-compatible Triton RoPE used by the inference engine."""

import math

import torch
import triton
import triton.language as tl

DEVICE = torch.device("cuda")
@triton.jit
def rope_kernel(x_ptr,   # (batch, n_heads, seq_len, head_dim)
                out_ptr,
                positions_ptr,
                x_row_stride,
                head_stride,
                batch_stride,
                out_row_stride,
                out_head_stride,
                out_batch_stride,
                position_batch_stride,
                position_row_stride,
                log_theta,
                n_heads,
                seq_len,
                half: tl.constexpr,     # head_dim // 2 (compile-time constant)
                BLOCK_SIZE: tl.constexpr):

    # Flatten (batch, head, token) into grid-X. CUDA grid-Y/Z are limited to 65,535,
    # which a packed prefill can exceed even when every individual prompt is within
    # the model context limit (for example 32 * 2,048 tokens).
    program = tl.program_id(0)
    row = program // seq_len
    seq = program - row * seq_len
    batch = row // n_heads
    head = row - batch * n_heads

    h = tl.arange(0, BLOCK_SIZE // 2)
    mask = h < half

    # Qwen/Hugging Face rotate-half pairing: i is paired with i + D/2.
    first_offsets = h
    second_offsets = h + half
    x_row = x_ptr + batch * batch_stride + head * head_stride + seq * x_row_stride
    first_x = tl.load(x_row + first_offsets, mask=mask)
    second_x = tl.load(x_row + second_offsets, mask=mask)

    # Explicit positions support prefill, ragged decode, and CUDA-graph buffers.
    position = tl.load(positions_ptr + batch * position_batch_stride + seq * position_row_stride)
    scale = -log_theta / half
    inv_freq = tl.exp(h.to(tl.float32) * scale)
    theta = position.to(tl.float32) * inv_freq
    cos_t = tl.cos(theta)
    sin_t = tl.sin(theta)

    first_out = first_x * cos_t - second_x * sin_t
    second_out = second_x * cos_t + first_x * sin_t

    # Cast back to the output dtype (loads promote fp16*fp32 -> fp32).
    first_out = first_out.to(out_ptr.dtype.element_ty)
    second_out = second_out.to(out_ptr.dtype.element_ty)
    out_row = out_ptr + batch * out_batch_stride + head * out_head_stride + seq * out_row_stride
    tl.store(out_row + first_offsets, first_out, mask=mask)
    tl.store(out_row + second_offsets, second_out, mask=mask)


def rope(x, positions=None, base=10000.0):
    assert x.ndim == 4, "expected a (batch, n_heads, seq_len, head_dim) input"
    assert x.stride(-1) == 1, "head_dim must be contiguous"

    batch, n_heads, seq_len, dim = x.shape
    assert dim % 2 == 0, "head_dim must be even for paired rotation"
    if positions is None:
        positions = torch.arange(seq_len, device=x.device, dtype=torch.int32)
        positions = positions.unsqueeze(0).expand(batch, -1)
    assert positions.shape == (batch, seq_len), "positions must be (batch, seq_len)"
    assert positions.device == x.device, "positions and x must share a device"

    half = dim // 2

    out = torch.empty_like(x)

    BLOCK_SIZE = triton.next_power_of_2(dim)
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)

    # One program per (batch, head, position), flattened onto grid-X so large packed
    # token counts do not overflow CUDA's 65,535 limit for grid-Y/grid-Z.
    grid = (batch * n_heads * seq_len,)
    rope_kernel[grid](
        x, out, positions,
        x.stride(2), x.stride(1), x.stride(0),           # seq / head / batch strides
        out.stride(2), out.stride(1), out.stride(0),
        positions.stride(0), positions.stride(1),
        math.log(base),
        n_heads,
        seq_len,
        half=half,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    return out


def _apply_rope_torch(x, base, positions=None):
    # Independent rotate-half implementation matching Qwen/Hugging Face.
    batch, n_heads, seq_len, dim = x.shape
    half = dim // 2

    x_f = x.float()
    if positions is None:
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch, -1)
    positions = positions.to(torch.float32).unsqueeze(-1)
    indices = torch.arange(half, device=x.device, dtype=torch.float32)
    theta = positions * (base ** (-indices / half))

    cos_t = torch.cos(theta)[:, None]
    sin_t = torch.sin(theta)[:, None]

    x_first, x_second = x_f.chunk(2, dim=-1)

    out_first = x_first * cos_t - x_second * sin_t
    out_second = x_second * cos_t + x_first * sin_t
    return out_first, out_second


def rope_reference(x, base=10000.0, positions=None):
    # Independent of the kernel path (recomputes cos/sin from scratch).
    out_first, out_second = _apply_rope_torch(x, base, positions)
    return torch.cat((out_first, out_second), dim=-1).to(x.dtype)


def rope_naive(x, base=10000.0, positions=None):
    """Idiomatic plain-PyTorch RoPE apply. Computes cos/sin the same way the
    kernel does in-kernel, so benchmark_vs_naive compares equal work."""
    out_first, out_second = _apply_rope_torch(x, base, positions)
    return torch.cat((out_first, out_second), dim=-1).to(x.dtype)


_compiled_rope = None


def rope_compiled(x, base=10000.0):
    """torch.compile-fused RoPE - the optimized PyTorch baseline. TorchInductor
    fuses the naive decomposition (arange/cos/sin/mul/stack/flatten) into ~one
    kernel, so the fp32 intermediates stay in registers instead of round-tripping
    through DRAM the way eager `rope_naive` does."""
    global _compiled_rope
    if _compiled_rope is None:
        _compiled_rope = torch.compile(rope_naive)
    return _compiled_rope(x, base)


# (batch, n_heads, seq_len, head_dim). Shapes stress masking: non-power-of-2
# head_dim (100), odd seq_len (257), small batch/head counts.
DEFAULT_SHAPES = (
    (2, 4, 128, 64),
    (1, 8, 257, 100),
    (4, 8, 512, 128),
    (1, 2, 16, 256),
)


def check_correctness(shapes=DEFAULT_SHAPES, dtype=torch.float16):
    print(f"{'shape (B,H,S,D)':>20} | {'max_err':>10} | {'status':>6}")

    all_ok = True
    for shape in shapes:
        x = torch.randn(shape, device=DEVICE, dtype=dtype)

        out = rope(x)
        ref = rope_reference(x)

        max_err = (out.float() - ref.float()).abs().max().item()
        ok = max_err < 2e-2  # fp16 has ~1e-3 relative precision; leave headroom
        all_ok &= ok

        print(f"{str(shape):>20} | {max_err:>10.5f} | {'OK' if ok else 'FAIL':>6}")

    return all_ok


# Set to your GPU's datasheet peak (GB/s) to see a %-of-peak column; None hides it.
PEAK_BANDWIDTH_GBPS = None

# Realistic transformer shapes: fixed batch/heads/head_dim, sweeping seq_len.
BENCH_SHAPES = (
    (4, 32, 512, 128),
    (4, 32, 1024, 128),
    (4, 32, 2048, 128),
    (4, 32, 4096, 128),
    (2, 32, 8192, 128),
)


def benchmark_bandwidth(shapes=BENCH_SHAPES, dtype=torch.float16):
    x_bytes = torch.tensor([], dtype=dtype).element_size()

    header = f"{'shape (B,H,S,D)':>20} | {'time (ms)':>10} | {'GB/s':>10}"
    if PEAK_BANDWIDTH_GBPS:
        header += f" | {'% peak':>8}"
    print(header)

    results = []
    for shape in shapes:
        x = torch.randn(shape, device=DEVICE, dtype=dtype)

        ms = triton.testing.do_bench(lambda: rope(x))

        # cos/sin are computed in-kernel now, so traffic is just x read + out write.
        bytes_moved = 2 * x.numel() * x_bytes
        gbps = bytes_moved / (ms * 1e-3) / 1e9

        row = f"{str(shape):>20} | {ms:>10.4f} | {gbps:>10.2f}"
        if PEAK_BANDWIDTH_GBPS:
            row += f" | {gbps / PEAK_BANDWIDTH_GBPS * 100:>7.1f}%"
        print(row)

        results.append((shape, ms, gbps))

    return results


def benchmark_vs_naive(shapes=BENCH_SHAPES, dtype=torch.float16):
    header = (f"{'shape (B,H,S,D)':>20} | {'triton (ms)':>12} | {'naive (ms)':>12} | "
              f"{'compiled (ms)':>14} | {'vs naive':>9} | {'vs compiled':>12} | {'max_err':>8}")
    print(header)
    print("-" * len(header))

    results = []
    for shape in shapes:
        x = torch.randn(shape, device=DEVICE, dtype=dtype)

        triton_ms = triton.testing.do_bench(lambda: rope(x))
        naive_ms = triton.testing.do_bench(lambda: rope_naive(x))
        compiled_ms = triton.testing.do_bench(lambda: rope_compiled(x))

        # speedup > 1 means the Triton kernel is faster than that baseline.
        vs_naive = naive_ms / triton_ms
        vs_compiled = compiled_ms / triton_ms

        # Sanity: all paths should agree (same rotation).
        max_err = (rope(x).float() - rope_compiled(x).float()).abs().max().item()

        print(f"{str(shape):>20} | {triton_ms:>12.4f} | {naive_ms:>12.4f} | "
              f"{compiled_ms:>14.4f} | {vs_naive:>8.2f}x | {vs_compiled:>11.2f}x | {max_err:>8.4f}")

        results.append((shape, triton_ms, naive_ms, compiled_ms, vs_naive, vs_compiled))

    return results


if __name__ == "__main__":
    check_correctness()
    print()
    benchmark_bandwidth()
    print()
    benchmark_vs_naive()
