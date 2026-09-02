"""Cache-aware paged decode attention kernel with GQA support.

Decode = one query token per (batch, head), attended over that sequence's cached
K/V, read directly from the paged pool via its block table (no gather/pad/mask).
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

DEVICE = torch.device("cuda")

# The adaptive candidate wins as a microkernel at some shorter shapes, but the
# end-to-end engine does not recover its host-side selection overhead there.  Keep
# short-context decode on the exact production path and reserve adaptive dispatch
# for the long-context regime it measurably improves.
MIN_ADAPTIVE_DECODE_CONTEXT_LENGTH = 1024


def resolve_decode_attention_policy(policy, max_context_length):
    """Resolve the host policy once, before entering the transformer layer loop."""
    if policy == "production":
        return "production"
    if policy != "adaptive":
        raise ValueError("decode attention policy must be 'production' or 'adaptive'")
    if max_context_length is None:
        raise ValueError("adaptive decode attention requires a host context length")
    if max_context_length < MIN_ADAPTIVE_DECODE_CONTEXT_LENGTH:
        return "production"
    return "adaptive"


@triton.jit
def decode_attention_kernel(
    q_ptr,            # (B, n_heads, d_head)
    k_pool_ptr,       # (num_blocks, block_size, n_kv_heads, d_head)
    v_pool_ptr,       # (num_blocks, block_size, n_kv_heads, d_head)  (same layout as k)
    block_table_ptr,  # (B, max_blocks) int32
    seq_lens_ptr,     # (B,) int32
    out_ptr,          # (B, n_heads, d_head)
    stride_qb, stride_qh, stride_qd,               # q strides
    stride_pblk, stride_pt, stride_pkv, stride_pd,  # pool strides (k and v share these)
    stride_btb, stride_btm,                        # block_table strides
    stride_ob, stride_oh, stride_od,               # out strides
    scale,
    GROUP: tl.constexpr,       # n_heads // n_kv_heads  (query heads per kv head)
    BLOCK_SIZE: tl.constexpr,  # tokens per block
    D_HEAD: tl.constexpr,      # head dim
):
    # One program per (batch, query-head).
    b = tl.program_id(0)
    h = tl.program_id(1)
    kv_h = h // GROUP                      # GQA: which kv head this query head reads

    seq_len = tl.load(seq_lens_ptr + b)    # this sequence's cached length

    offs_d = tl.arange(0, D_HEAD)          # feature offsets
    offs_t = tl.arange(0, BLOCK_SIZE)      # within-block token offsets

    # Load the query vector q for (b, h) -> shape (D_HEAD,)
    q = tl.load(q_ptr + b * stride_qb + h * stride_qh + offs_d * stride_qd)

    # init online-softmax state
    # Keep the max value
    m = float("-inf")
    acc = tl.zeros([D_HEAD], dtype=tl.float32)  # Initialize value state
    denom = 0.0  # Keep track of the denominator for softmax

    # loop over this sequence's blocks
    for i in range(tl.cdiv(seq_len, BLOCK_SIZE)):
        blk_id = tl.load(block_table_ptr + b * stride_btb + i * stride_btm)
        # Build masks for k, v
        abs_pos = i * BLOCK_SIZE + offs_t
        mask = abs_pos < seq_len
        # Load in the kv heads
        base = stride_pblk * blk_id + stride_pkv * kv_h
        kv_off = offs_t[:, None] * stride_pt + offs_d[None, :] * stride_pd
        k = tl.load(k_pool_ptr + base + kv_off, mask=mask[:, None], other=0.0)  # (BLOCK_SIZE, d_head) <- one k head per token here
        v = tl.load(v_pool_ptr + base + kv_off, mask=mask[:, None], other=0.0)  # (BLOCK_SIZE, d_head)

        # We have Q, K, and V loaded in
        # Need to first compute max(max(Q @ K), max)
        # Rescale exponents of denominator and value multiplication, then aggregate
        # QK in fp32: an fp16 dot-product accumulation can overflow to inf on real
        # activations, and inf - inf in the softmax below becomes NaN.
        score = tl.sum(q[None, :].to(tl.float32) * k.to(tl.float32), axis=1) * scale  # <- attention score computation
        score = tl.where(mask, score, float("-inf"))

        # Store local maximum
        new_max = tl.maximum(m, tl.max(score, axis=0))
        # First update denominator by scaling old term and new term respectively
        alpha = tl.exp(m - new_max)
        p = tl.exp(score - new_max)
        denom = alpha * denom + tl.sum(p, axis=0)
        # Next update value aggregation
        acc = alpha * acc + tl.sum(p[:, None] * v, axis=0)  # <- scaled aggregation
        # Update the maximum
        m = new_max

    # normalize (acc / denom) and store to out[b, h]
    attn_out = acc / denom  # apply denominator
    # Need to store
    tl.store(out_ptr + b * stride_ob + h * stride_oh + offs_d * stride_od,
             attn_out.to(out_ptr.dtype.element_ty))


def paged_decode_attention(
    q,
    k_pool,
    v_pool,
    block_table,
    seq_lens,
    scale=None,
    *,
    num_warps=None,
    num_stages=None,
):
    """q: (B, n_heads, d_head). pools: (num_blocks, block_size, n_kv_heads, d_head).
    block_table: (B, max_blocks) int32. seq_lens: (B,) int32. -> out (B, n_heads, d_head)."""
    B, n_heads, d_head = q.shape
    _, block_size, n_kv_heads, _ = k_pool.shape
    assert k_pool.shape == v_pool.shape, "k and v pools must share layout"
    assert n_heads % n_kv_heads == 0, "n_heads must be a multiple of n_kv_heads"

    if scale is None:
        scale = 1.0 / (d_head ** 0.5)

    out = torch.empty_like(q)
    grid = (B, n_heads)
    launch_options = {}
    if num_warps is not None:
        if num_warps not in (1, 2, 4, 8):
            raise ValueError("num_warps must be 1, 2, 4, or 8")
        launch_options["num_warps"] = num_warps
    if num_stages is not None:
        if num_stages < 1:
            raise ValueError("num_stages must be positive")
        launch_options["num_stages"] = num_stages
    decode_attention_kernel[grid](
        q, k_pool, v_pool, block_table, seq_lens, out,
        q.stride(0), q.stride(1), q.stride(2),
        k_pool.stride(0), k_pool.stride(1), k_pool.stride(2), k_pool.stride(3),
        block_table.stride(0), block_table.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        scale,
        GROUP=n_heads // n_kv_heads,
        BLOCK_SIZE=block_size,
        D_HEAD=d_head,
        **launch_options,
    )
    return out


def paged_decode_attention_dispatch(
    q,
    k_pool,
    v_pool,
    block_table,
    seq_lens,
    *,
    policy="production",
    max_context_length=None,
    scale=None,
):
    """Dispatch without reading device metadata or changing the production default."""
    if policy == "production":
        return paged_decode_attention(
            q, k_pool, v_pool, block_table, seq_lens, scale=scale
        )
    policy = resolve_decode_attention_policy(policy, max_context_length)
    if policy == "production":
        return paged_decode_attention(
            q, k_pool, v_pool, block_table, seq_lens, scale=scale
        )
    from kernel_dispatch import (
        paged_decode_attention_candidate,
        select_paged_decode_candidate_config,
    )

    config = select_paged_decode_candidate_config(
        q.shape[0], max_context_length, policy=policy
    )
    if config is None:
        return paged_decode_attention(
            q, k_pool, v_pool, block_table, seq_lens, scale=scale
        )
    return paged_decode_attention_candidate(
        q,
        k_pool,
        v_pool,
        block_table,
        seq_lens,
        scale=scale,
        **config,
    )


def decode_attention_reference(q, k_pool, v_pool, block_table, seq_lens, scale=None):
    """Ground-truth: gather each sequence's KV via its block table, attend per head."""
    B, n_heads, d_head = q.shape
    _, block_size, n_kv_heads, _ = k_pool.shape
    group = n_heads // n_kv_heads
    if scale is None:
        scale = 1.0 / (d_head ** 0.5)

    out = torch.empty_like(q)
    for b in range(B):
        L = int(seq_lens[b])
        n_blk = (L + block_size - 1) // block_size
        bt = block_table[b, :n_blk]
        K = k_pool[bt].reshape(-1, n_kv_heads, d_head)[:L]   # (L, n_kv_heads, d_head)
        V = v_pool[bt].reshape(-1, n_kv_heads, d_head)[:L]
        for h in range(n_heads):
            kv_h = h // group
            k_h = K[:, kv_h, :].float()                      # (L, d_head)
            v_h = V[:, kv_h, :].float()
            scores = (q[b, h].float() @ k_h.T) * scale        # (L,)
            p = torch.softmax(scores, dim=-1)
            out[b, h] = (p @ v_h).to(out.dtype)              # (d_head,)
    return out


def decode_attention_sdpa(q, k_pool, v_pool, block_table, seq_lens, scale=None):
    B, n_heads, d_head = q.shape
    _, block_size, n_kv_heads, _ = k_pool.shape
    group = n_heads // n_kv_heads
    if scale is None:
        scale = 1.0 / (d_head ** 0.5)

    out = torch.empty_like(q)
    for b in range(B):
        L = int(seq_lens[b])
        n_blk = (L + block_size - 1) // block_size
        bt = block_table[b, :n_blk]
        K = k_pool[bt].reshape(-1, n_kv_heads, d_head)[:L]       # (L, n_kv_heads, d_head)
        V = v_pool[bt].reshape(-1, n_kv_heads, d_head)[:L]
        K = K.repeat_interleave(group, dim=1)                    # (L, n_heads, d_head)  GQA expand
        V = V.repeat_interleave(group, dim=1)
        q_b = q[b][None, :, None, :]                             # (1, n_heads, 1, d_head)
        k_b = K.permute(1, 0, 2)[None]                           # (1, n_heads, L, d_head)
        v_b = V.permute(1, 0, 2)[None]
        o = F.scaled_dot_product_attention(q_b, k_b, v_b, scale=scale)  # (1, n_heads, 1, d_head)
        out[b] = o[0, :, 0, :]
    return out


def check_correctness(B=2, n_heads=12, n_kv_heads=2, d_head=128,
                      block_size=16, max_len=100, dtype=torch.float16,
                      seq_lens_override=None):
    torch.manual_seed(0)
    if seq_lens_override is None:
        seq_lens = torch.randint(1, max_len + 1, (B,), device=DEVICE, dtype=torch.int32)
    else:
        if not seq_lens_override or any(length < 1 for length in seq_lens_override):
            raise ValueError("seq_lens_override must contain positive lengths")
        B = len(seq_lens_override)
        seq_lens = torch.tensor(seq_lens_override, device=DEVICE, dtype=torch.int32)
    max_blocks = (int(seq_lens.max()) + block_size - 1) // block_size
    num_blocks = B * max_blocks                    # enough for each sequence its own range

    # Give each sequence a distinct contiguous run of physical blocks.
    block_table = torch.zeros(B, max_blocks, device=DEVICE, dtype=torch.int32)
    for b in range(B):
        nb = (int(seq_lens[b]) + block_size - 1) // block_size
        block_table[b, :nb] = torch.arange(b * max_blocks, b * max_blocks + nb,
                                           device=DEVICE, dtype=torch.int32)

    k_pool = torch.randn(num_blocks, block_size, n_kv_heads, d_head, device=DEVICE, dtype=dtype)
    v_pool = torch.randn(num_blocks, block_size, n_kv_heads, d_head, device=DEVICE, dtype=dtype)
    q = torch.randn(B, n_heads, d_head, device=DEVICE, dtype=dtype)

    out_kernel = paged_decode_attention(q, k_pool, v_pool, block_table, seq_lens)
    out_sdpa = decode_attention_sdpa(q, k_pool, v_pool, block_table, seq_lens)
    out_ref = decode_attention_reference(q, k_pool, v_pool, block_table, seq_lens)

    err_sdpa = (out_kernel.float() - out_sdpa.float()).abs().max().item()
    err_ref = (out_kernel.float() - out_ref.float()).abs().max().item()
    print(f"seq_lens          : {seq_lens.tolist()}")
    print(f"max abs err vs torch.sdpa   : {err_sdpa:.5f}")
    print(f"max abs err vs manual ref   : {err_ref:.5f}")
    return max(err_sdpa, err_ref)


if __name__ == "__main__":
    check_correctness()
