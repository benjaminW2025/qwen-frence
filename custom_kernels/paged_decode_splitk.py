"""Split-K paged decode attention for improved SM occupancy.

Instead of one program per (batch, head) iterating over all KV blocks,
we launch K programs per (batch, head), each handling a chunk of KV blocks.
Partial softmax states are reduced to produce the final output.
"""

import torch
import triton
import triton.language as tl

DEVICE = torch.device("cuda")

# Tuning constants
TARGET_OCCUPANCY = 4  # warps per SM
MIN_BLOCKS_PER_CHUNK = 8  # don't split too fine

_num_sms_cache = None

def get_num_sms() -> int:
    """Query SM count from device, cached after first call."""
    global _num_sms_cache
    if _num_sms_cache is None:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        _num_sms_cache = props.multi_processor_count
    return _num_sms_cache


def compute_split_k(batch_size: int, n_heads: int, num_kv_blocks: int) -> int:
    """Compute optimal K to fill SMs without excessive reduction overhead."""
    current_programs = batch_size * n_heads
    target_programs = get_num_sms() * TARGET_OCCUPANCY

    if current_programs >= target_programs:
        return 1  # already saturated

    if num_kv_blocks < MIN_BLOCKS_PER_CHUNK * 2:
        return 1  # not enough blocks to split

    # K to reach target occupancy
    k = (target_programs + current_programs - 1) // current_programs

    # Cap by available blocks
    max_k = num_kv_blocks // MIN_BLOCKS_PER_CHUNK
    k = min(k, max_k)

    return max(k, 1)


@triton.jit
def splitk_decode_partial_kernel(
    q_ptr,            # (B, n_heads, d_head)
    k_pool_ptr,       # (num_blocks, block_size, n_kv_heads, d_head)
    v_pool_ptr,       # (num_blocks, block_size, n_kv_heads, d_head)
    block_table_ptr,  # (B, max_blocks) int32
    seq_lens_ptr,     # (B,) int32
    # Partial outputs - one per (B, n_heads, K)
    partial_m_ptr,    # (B, n_heads, K) float32 - max scores
    partial_d_ptr,    # (B, n_heads, K) float32 - denominators
    partial_acc_ptr,  # (B, n_heads, K, d_head) float32 - weighted values
    # Strides
    stride_qb, stride_qh, stride_qd,
    stride_pblk, stride_pt, stride_pkv, stride_pd,
    stride_btb, stride_btm,
    stride_pmb, stride_pmh, stride_pmk,  # partial_m strides
    stride_pab, stride_pah, stride_pak, stride_pad,  # partial_acc strides
    # Constants
    scale,
    K: tl.constexpr,           # number of splits
    GROUP: tl.constexpr,       # n_heads // n_kv_heads
    BLOCK_SIZE: tl.constexpr,  # tokens per KV block
    D_HEAD: tl.constexpr,
):
    """Each program computes partial attention over a chunk of KV blocks."""
    b = tl.program_id(0)
    h = tl.program_id(1)
    k_idx = tl.program_id(2)  # which chunk

    kv_h = h // GROUP
    seq_len = tl.load(seq_lens_ptr + b)
    num_blocks = tl.cdiv(seq_len, BLOCK_SIZE)

    # Compute this chunk's block range
    blocks_per_chunk = tl.cdiv(num_blocks, K)
    start_block = k_idx * blocks_per_chunk
    end_block = tl.minimum(start_block + blocks_per_chunk, num_blocks)

    # Load query
    offs_d = tl.arange(0, D_HEAD)
    q = tl.load(q_ptr + b * stride_qb + h * stride_qh + offs_d * stride_qd)

    # Online softmax state for this chunk
    m = float("-inf")
    denom = 0.0
    acc = tl.zeros([D_HEAD], dtype=tl.float32)

    offs_t = tl.arange(0, BLOCK_SIZE)

    # Process this chunk's blocks
    for i in range(start_block, end_block):
        blk_id = tl.load(block_table_ptr + b * stride_btb + i * stride_btm)
        abs_pos = i * BLOCK_SIZE + offs_t
        mask = abs_pos < seq_len

        # Load K, V for this block
        base = stride_pblk * blk_id + stride_pkv * kv_h
        kv_off = offs_t[:, None] * stride_pt + offs_d[None, :] * stride_pd
        k = tl.load(k_pool_ptr + base + kv_off, mask=mask[:, None], other=0.0)
        v = tl.load(v_pool_ptr + base + kv_off, mask=mask[:, None], other=0.0)

        # Compute attention scores
        score = tl.sum(q[None, :].to(tl.float32) * k.to(tl.float32), axis=1) * scale
        score = tl.where(mask, score, float("-inf"))

        # Online softmax update
        new_max = tl.maximum(m, tl.max(score, axis=0))
        alpha = tl.exp(m - new_max)
        p = tl.exp(score - new_max)
        denom = alpha * denom + tl.sum(p, axis=0)
        acc = alpha * acc + tl.sum(p[:, None] * v.to(tl.float32), axis=0)
        m = new_max

    # Write partial results
    partial_idx = b * stride_pmb + h * stride_pmh + k_idx * stride_pmk
    tl.store(partial_m_ptr + partial_idx, m)
    tl.store(partial_d_ptr + partial_idx, denom)

    acc_base = b * stride_pab + h * stride_pah + k_idx * stride_pak
    tl.store(partial_acc_ptr + acc_base + offs_d * stride_pad, acc)


@triton.jit
def splitk_decode_reduce_kernel(
    partial_m_ptr,    # (B, n_heads, K) float32
    partial_d_ptr,    # (B, n_heads, K) float32
    partial_acc_ptr,  # (B, n_heads, K, d_head) float32
    out_ptr,          # (B, n_heads, d_head)
    stride_pmb, stride_pmh, stride_pmk,
    stride_pab, stride_pah, stride_pak, stride_pad,
    stride_ob, stride_oh, stride_od,
    K: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    """Reduce K partial results into final output."""
    b = tl.program_id(0)
    h = tl.program_id(1)

    offs_d = tl.arange(0, D_HEAD)

    # Load first chunk as initial state
    base_m = b * stride_pmb + h * stride_pmh
    base_acc = b * stride_pab + h * stride_pah

    m = tl.load(partial_m_ptr + base_m)
    denom = tl.load(partial_d_ptr + base_m)
    acc = tl.load(partial_acc_ptr + base_acc + offs_d * stride_pad)

    # Reduce remaining chunks
    for k in range(1, K):
        m_k = tl.load(partial_m_ptr + base_m + k * stride_pmk)
        d_k = tl.load(partial_d_ptr + base_m + k * stride_pmk)
        acc_k = tl.load(partial_acc_ptr + base_acc + k * stride_pak + offs_d * stride_pad)

        # Combine using online softmax math
        new_max = tl.maximum(m, m_k)
        alpha = tl.exp(m - new_max)
        alpha_k = tl.exp(m_k - new_max)

        denom = alpha * denom + alpha_k * d_k
        acc = alpha * acc + alpha_k * acc_k
        m = new_max

    # Normalize and store
    out = acc / denom
    tl.store(out_ptr + b * stride_ob + h * stride_oh + offs_d * stride_od,
             out.to(out_ptr.dtype.element_ty))


def paged_decode_attention_splitk(
    q,
    k_pool,
    v_pool,
    block_table,
    seq_lens,
    scale=None,
    *,
    k_splits=None,  # None = auto, or specify manually
):
    """Split-K paged decode attention.

    q: (B, n_heads, d_head)
    k_pool, v_pool: (num_blocks, block_size, n_kv_heads, d_head)
    block_table: (B, max_blocks) int32
    seq_lens: (B,) int32
    """
    B, n_heads, d_head = q.shape
    _, block_size, n_kv_heads, _ = k_pool.shape

    if scale is None:
        scale = 1.0 / (d_head ** 0.5)

    max_seq_len = int(seq_lens.max().item())
    num_kv_blocks = (max_seq_len + block_size - 1) // block_size

    # Compute K
    if k_splits is None:
        K = compute_split_k(B, n_heads, num_kv_blocks)
    else:
        K = k_splits

    # Fast path: K=1 means no split needed
    if K == 1:
        from engine.kvcache.paged_decode_attention import paged_decode_attention
        return paged_decode_attention(q, k_pool, v_pool, block_table, seq_lens, scale=scale)

    # Allocate partial buffers
    partial_m = torch.empty(B, n_heads, K, device=q.device, dtype=torch.float32)
    partial_d = torch.empty(B, n_heads, K, device=q.device, dtype=torch.float32)
    partial_acc = torch.empty(B, n_heads, K, d_head, device=q.device, dtype=torch.float32)

    # Launch partial kernel
    grid_partial = (B, n_heads, K)
    splitk_decode_partial_kernel[grid_partial](
        q, k_pool, v_pool, block_table, seq_lens,
        partial_m, partial_d, partial_acc,
        q.stride(0), q.stride(1), q.stride(2),
        k_pool.stride(0), k_pool.stride(1), k_pool.stride(2), k_pool.stride(3),
        block_table.stride(0), block_table.stride(1),
        partial_m.stride(0), partial_m.stride(1), partial_m.stride(2),
        partial_acc.stride(0), partial_acc.stride(1), partial_acc.stride(2), partial_acc.stride(3),
        scale,
        K=K,
        GROUP=n_heads // n_kv_heads,
        BLOCK_SIZE=block_size,
        D_HEAD=d_head,
    )

    # Launch reduce kernel
    out = torch.empty_like(q)
    grid_reduce = (B, n_heads)
    splitk_decode_reduce_kernel[grid_reduce](
        partial_m, partial_d, partial_acc, out,
        partial_m.stride(0), partial_m.stride(1), partial_m.stride(2),
        partial_acc.stride(0), partial_acc.stride(1), partial_acc.stride(2), partial_acc.stride(3),
        out.stride(0), out.stride(1), out.stride(2),
        K=K,
        D_HEAD=d_head,
    )

    return out


def check_correctness(
    B=8,
    n_heads=12,
    n_kv_heads=2,
    d_head=128,
    block_size=16,
    max_len=8192,
    dtype=torch.float16,
    k_splits=None,
):
    """Verify Split-K matches reference implementation."""
    from engine.kvcache.paged_decode_attention import paged_decode_attention

    torch.manual_seed(42)

    seq_lens = torch.randint(max_len // 2, max_len + 1, (B,), device=DEVICE, dtype=torch.int32)
    max_blocks = (int(seq_lens.max()) + block_size - 1) // block_size
    num_blocks = B * max_blocks

    # Each sequence gets contiguous blocks
    block_table = torch.zeros(B, max_blocks, device=DEVICE, dtype=torch.int32)
    for b in range(B):
        nb = (int(seq_lens[b]) + block_size - 1) // block_size
        block_table[b, :nb] = torch.arange(b * max_blocks, b * max_blocks + nb,
                                           device=DEVICE, dtype=torch.int32)

    k_pool = torch.randn(num_blocks, block_size, n_kv_heads, d_head, device=DEVICE, dtype=dtype)
    v_pool = torch.randn(num_blocks, block_size, n_kv_heads, d_head, device=DEVICE, dtype=dtype)
    q = torch.randn(B, n_heads, d_head, device=DEVICE, dtype=dtype)

    # Reference
    out_ref = paged_decode_attention(q, k_pool, v_pool, block_table, seq_lens)

    # Split-K
    out_splitk = paged_decode_attention_splitk(
        q, k_pool, v_pool, block_table, seq_lens, k_splits=k_splits
    )

    err = (out_splitk.float() - out_ref.float()).abs().max().item()

    num_kv_blocks = (int(seq_lens.max()) + block_size - 1) // block_size
    K = k_splits if k_splits else compute_split_k(B, n_heads, num_kv_blocks)

    print(f"B={B}, max_seq_len={int(seq_lens.max())}, num_kv_blocks={num_kv_blocks}")
    print(f"K={K} (programs: {B * n_heads} -> {B * n_heads * K})")
    print(f"max abs error: {err:.6f}")

    return err


def benchmark(
    B=8,
    n_heads=12,
    n_kv_heads=2,
    d_head=128,
    block_size=16,
    seq_len=8192,
    dtype=torch.float16,
    warmup=10,
    iters=100,
):
    """Benchmark Split-K vs production kernel."""
    from engine.kvcache.paged_decode_attention import paged_decode_attention

    torch.manual_seed(42)

    seq_lens = torch.full((B,), seq_len, device=DEVICE, dtype=torch.int32)
    max_blocks = (seq_len + block_size - 1) // block_size
    num_blocks = B * max_blocks

    block_table = torch.zeros(B, max_blocks, device=DEVICE, dtype=torch.int32)
    for b in range(B):
        block_table[b, :max_blocks] = torch.arange(b * max_blocks, (b + 1) * max_blocks,
                                                    device=DEVICE, dtype=torch.int32)

    k_pool = torch.randn(num_blocks, block_size, n_kv_heads, d_head, device=DEVICE, dtype=dtype)
    v_pool = torch.randn(num_blocks, block_size, n_kv_heads, d_head, device=DEVICE, dtype=dtype)
    q = torch.randn(B, n_heads, d_head, device=DEVICE, dtype=dtype)

    num_kv_blocks = (seq_len + block_size - 1) // block_size
    K = compute_split_k(B, n_heads, num_kv_blocks)

    print(f"B={B}, seq_len={seq_len}, num_kv_blocks={num_kv_blocks}")
    print(f"K={K} (programs: {B * n_heads} -> {B * n_heads * K})")

    # Warmup
    for _ in range(warmup):
        _ = paged_decode_attention(q, k_pool, v_pool, block_table, seq_lens)
        _ = paged_decode_attention_splitk(q, k_pool, v_pool, block_table, seq_lens)
    torch.cuda.synchronize()

    # Benchmark production
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        _ = paged_decode_attention(q, k_pool, v_pool, block_table, seq_lens)
    end.record()
    torch.cuda.synchronize()
    prod_ms = start.elapsed_time(end) / iters

    # Benchmark Split-K
    start.record()
    for _ in range(iters):
        _ = paged_decode_attention_splitk(q, k_pool, v_pool, block_table, seq_lens)
    end.record()
    torch.cuda.synchronize()
    splitk_ms = start.elapsed_time(end) / iters

    speedup = prod_ms / splitk_ms
    print(f"Production: {prod_ms:.3f}ms")
    print(f"Split-K:    {splitk_ms:.3f}ms")
    print(f"Speedup:    {speedup:.2f}x")

    return prod_ms, splitk_ms, speedup


if __name__ == "__main__":
    print("=== Correctness Check ===")
    check_correctness(B=8, max_len=8192, k_splits=8)
    print()

    print("=== Benchmark: Low Batch, Long Context ===")
    benchmark(B=8, seq_len=16384)
    print()

    print("=== Benchmark: Medium Batch, Long Context ===")
    benchmark(B=32, seq_len=8192)
    print()

    print("=== Benchmark: High Batch (should skip Split-K) ===")
    benchmark(B=128, seq_len=4096)
