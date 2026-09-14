"""Grouped split-K paged decode attention, CUDA-graph safe.

The two Triton kernels are taken verbatim from the decode stage study
(`experiments/decode/paged_decode_grouped_splitk_pipelined.py`), which fitted the
frozen K/stages policy against them. Only the host wrapper differs: the study's
entry point auto-computes K via `seq_lens.max().item()` and carries diagnostics
and ablation aliases, none of which belong on a captured path. A device-to-host
sync cannot be recorded into a CUDA graph at all.

Split-K shortens the critical path of the online-softmax reduction: one program
per (sequence, KV head, head tile) makes graph depth equal to the page count,
which at decode is the whole cached sequence. K programs over disjoint page
ranges cut that to pages/K, then one combine pass rescales the partials. The
saving grows with pages while the host cost of the extra launch is fixed, so this
only pays above a context/batch dependent break-even.

`split_k` is always explicit here. The caller resolves it once, before capture,
and the resulting action is baked into the graph.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _grouped_splitk_kernel(
    q_ptr, k_pool_ptr, v_pool_ptr, block_table_ptr, seq_lens_ptr,
    partial_out_ptr, partial_max_ptr, partial_sum_ptr,
    stride_qb, stride_qh, stride_qd,
    stride_pblk, stride_pt, stride_pkv, stride_pd,
    stride_vblk, stride_vt, stride_vkv, stride_vd,
    stride_sl,
    stride_btb, stride_btm,
    stride_pob, stride_pok, stride_poh, stride_pod,
    stride_pmb, stride_pmk, stride_pmh,
    scale,
    GROUP: tl.constexpr,
    HEADS_PER_PROGRAM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    D_HEAD: tl.constexpr,
    SPLIT_K: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
):
    """Identical arithmetic/layout across ablations; only loop stages change."""
    sequence = tl.program_id(0)
    kv_head = tl.program_id(1)
    head_tile = tl.program_id(2) % tl.cdiv(GROUP, HEADS_PER_PROGRAM)
    k_idx = tl.program_id(2) // tl.cdiv(GROUP, HEADS_PER_PROGRAM)

    head_offsets = tl.arange(0, BLOCK_H)
    dim_offsets = tl.arange(0, D_HEAD)
    token_offsets = tl.arange(0, PAGE_SIZE)
    heads_in_group = head_tile * HEADS_PER_PROGRAM + head_offsets
    head_mask = (head_offsets < HEADS_PER_PROGRAM) & (heads_in_group < GROUP)
    query_heads = kv_head * GROUP + heads_in_group

    q = tl.load(
        q_ptr
        + sequence * stride_qb
        + query_heads[:, None] * stride_qh
        + dim_offsets[None, :] * stride_qd,
        mask=head_mask[:, None],
        other=0.0,
    )

    seq_len = tl.load(seq_lens_ptr + sequence * stride_sl)
    total_pages = tl.cdiv(seq_len, PAGE_SIZE)

    # Balanced contiguous partitions: no empty programs when SPLIT_K <= pages.
    start_page = (k_idx * total_pages) // SPLIT_K
    end_page = ((k_idx + 1) * total_pages) // SPLIT_K

    running_max = tl.full([BLOCK_H], float("-inf"), tl.float32)
    running_sum = tl.zeros([BLOCK_H], tl.float32)
    accumulator = tl.zeros([BLOCK_H, D_HEAD], tl.float32)

    pool_offsets = token_offsets[:, None] * stride_pt + dim_offsets[None, :] * stride_pd

    value_offsets = token_offsets[:, None] * stride_vt + dim_offsets[None, :] * stride_vd

    # Loop-level stages also target loads without tl.dot (Triton 3.2+).
    # One stage disables software pipelining for the sequential control.
    for page_index in tl.range(start_page, end_page, num_stages=PIPELINE_STAGES):
        page_id = tl.load(
            block_table_ptr + sequence * stride_btb + page_index * stride_btm
        )
        positions = page_index * PAGE_SIZE + token_offsets
        token_mask = positions < seq_len
        base = page_id * stride_pblk + kv_head * stride_pkv

        keys = tl.load(
            k_pool_ptr + base + pool_offsets,
            mask=token_mask[:, None],
            other=0.0,
        )
        values = tl.load(
            v_pool_ptr + page_id * stride_vblk + kv_head * stride_vkv + value_offsets,
            mask=token_mask[:, None],
            other=0.0,
        )

        # Compute attention
        scores = tl.sum(
            q[:, None, :].to(tl.float32) * keys[None, :, :].to(tl.float32),
            axis=2,
        ) * scale

        score_mask = head_mask[:, None] & token_mask[None, :]
        scores = tl.where(score_mask, scores, float("-inf"))
        scores = tl.where(head_mask[:, None], scores, 0.0)

        page_max = tl.max(scores, axis=1)
        new_max = tl.maximum(running_max, page_max)
        correction = tl.exp(running_max - new_max)
        probabilities = tl.exp(scores - new_max[:, None])
        probabilities = tl.where(score_mask, probabilities, 0.0)

        accumulator = (
            accumulator * correction[:, None]
            + tl.sum(probabilities[:, :, None] * values[None, :, :].to(tl.float32), axis=1)
        )
        running_sum = running_sum * correction + tl.sum(probabilities, axis=1)
        running_max = new_max

    if SPLIT_K == 1:
        # Normalize while the accumulator is still FP32, including empty rows.
        accumulator = accumulator / tl.where(running_sum > 0, running_sum, 1.0)[:, None]

    tl.store(
        partial_out_ptr
        + sequence * stride_pob
        + k_idx * stride_pok
        + query_heads[:, None] * stride_poh
        + dim_offsets[None, :] * stride_pod,
        accumulator.to(partial_out_ptr.dtype.element_ty),
        mask=head_mask[:, None],
    )
    if SPLIT_K > 1:
        tl.store(
            partial_max_ptr
            + sequence * stride_pmb
            + k_idx * stride_pmk
            + query_heads * stride_pmh,
            running_max,
            mask=head_mask,
        )
        tl.store(
            partial_sum_ptr
            + sequence * stride_pmb
            + k_idx * stride_pmk
            + query_heads * stride_pmh,
            running_sum,
            mask=head_mask,
        )


# =============================================================================
# REDUCE KERNEL (shared by both)
# =============================================================================

@triton.jit
def _splitk_reduce_kernel(
    partial_out_ptr, partial_max_ptr, partial_sum_ptr, out_ptr,
    stride_pob, stride_pok, stride_poh, stride_pod,
    stride_pmb, stride_pmk, stride_pmh,
    stride_ob, stride_oh, stride_od,
    SPLIT_K: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    """Reduce partial results using online softmax combination."""
    sequence = tl.program_id(0)
    query_head = tl.program_id(1)

    dim_offsets = tl.arange(0, D_HEAD)

    global_max = tl.load(
        partial_max_ptr + sequence * stride_pmb + 0 * stride_pmk + query_head * stride_pmh
    )
    global_sum = tl.load(
        partial_sum_ptr + sequence * stride_pmb + 0 * stride_pmk + query_head * stride_pmh
    )
    accumulator = tl.load(
        partial_out_ptr
        + sequence * stride_pob
        + 0 * stride_pok
        + query_head * stride_poh
        + dim_offsets * stride_pod,
    ).to(tl.float32)

    for k in range(1, SPLIT_K):
        partial_max = tl.load(
            partial_max_ptr + sequence * stride_pmb + k * stride_pmk + query_head * stride_pmh
        )
        partial_sum = tl.load(
            partial_sum_ptr + sequence * stride_pmb + k * stride_pmk + query_head * stride_pmh
        )
        partial_out = tl.load(
            partial_out_ptr
            + sequence * stride_pob
            + k * stride_pok
            + query_head * stride_poh
            + dim_offsets * stride_pod,
        ).to(tl.float32)

        # Empty partitions must not participate in the maximum or rescaling.
        # This also handles an empty first partition when pages < SPLIT_K.
        if partial_sum > 0:
            new_max = tl.maximum(global_max, partial_max)
            alpha_global = tl.where(global_sum > 0, tl.exp(global_max - new_max), 0.0)
            alpha_partial = tl.exp(partial_max - new_max)
            accumulator = accumulator * alpha_global + partial_out * alpha_partial
            global_sum = global_sum * alpha_global + partial_sum * alpha_partial
            global_max = new_max

    output = accumulator / tl.where(global_sum > 0, global_sum, 1.0)
    tl.store(
        out_ptr
        + sequence * stride_ob
        + query_head * stride_oh
        + dim_offsets * stride_od,
        output.to(out_ptr.dtype.element_ty),
    )

def grouped_splitk_decode_attention(
    q,
    k_pool,
    v_pool,
    block_table,
    seq_lens,
    *,
    split_k,
    num_stages=2,
    heads_per_program=1,
    num_warps=4,
    scale=None,
    partials=None,
):
    """Explicit-configuration split-K decode attention.

    q: (B, n_heads, d_head). pools: (num_pages, page_size, n_kv_heads, d_head).
    block_table: (B, max_pages) int32. seq_lens: (B,) int32. -> out (B, n_heads, d_head)

    Validation reads shapes and dtypes only; no tensor is inspected on the host, so
    this is safe to call inside a graph capture. `partials` optionally supplies
    preallocated (out, max, sum) buffers for the eager path, where a fresh
    allocation per layer per step is a measurable cost. Under capture the
    allocations are served from the graph's private pool and replayed at fixed
    addresses, so leaving it None is free there.
    """
    if q.ndim != 3 or any(size < 1 for size in q.shape):
        raise ValueError("q must have non-empty shape [batch, query_heads, head_dim]")
    if k_pool.ndim != 4 or k_pool.shape != v_pool.shape:
        raise ValueError("K/V pools must have matching rank-four shapes")
    if k_pool.shape[2] < 1 or k_pool.shape[-1] != q.shape[-1]:
        raise ValueError("KV heads must be positive and head dimensions must match")
    if q.shape[1] % k_pool.shape[2]:
        raise ValueError("query heads must be divisible by KV heads")
    if block_table.ndim != 2 or seq_lens.ndim != 1:
        raise ValueError("block table and sequence lengths must be rank two and one")
    if block_table.shape[0] != q.shape[0] or seq_lens.numel() != q.shape[0]:
        raise ValueError("batch dimensions must match")
    for size in (q.shape[-1], k_pool.shape[1]):
        if size < 1 or size & (size - 1):
            raise ValueError("head dimension and page size must be powers of two")
    if k_pool.dtype != q.dtype or v_pool.dtype != q.dtype:
        raise ValueError("Q/K/V dtypes must match")
    if any(t.dtype not in (torch.int32, torch.int64) for t in (block_table, seq_lens)):
        raise ValueError("page table and lengths must use integer tensors")
    if not isinstance(split_k, int) or split_k < 1:
        raise ValueError("split_k must be a positive integer")
    if num_warps not in (1, 2, 4, 8) or not isinstance(num_stages, int) or num_stages < 1:
        raise ValueError("num_warps must be 1, 2, 4, or 8 and num_stages a positive integer")

    batch, query_heads, head_dim = q.shape
    page_size, kv_heads = k_pool.shape[1], k_pool.shape[2]
    group = query_heads // kv_heads
    if heads_per_program not in (1, 2, 3, 6) or group % heads_per_program:
        raise ValueError("heads_per_program must divide the query/KV head group evenly")
    head_tiles = group // heads_per_program
    if scale is None:
        scale = head_dim ** -0.5

    if partials is None:
        partial_out = torch.empty((batch, split_k, query_heads, head_dim),
                                  dtype=q.dtype if split_k == 1 else torch.float32,
                                  device=q.device)
        partial_max = torch.empty((batch, split_k, query_heads), dtype=torch.float32, device=q.device)
        partial_sum = torch.empty_like(partial_max)
    else:
        partial_out, partial_max, partial_sum = partials
        expected = (batch, split_k, query_heads, head_dim)
        if tuple(partial_out.shape) != expected or tuple(partial_max.shape) != expected[:3]:
            raise ValueError("preallocated partials do not match this launch shape")

    _grouped_splitk_kernel[(batch, kv_heads, head_tiles * split_k)](
        q, k_pool, v_pool, block_table, seq_lens,
        partial_out, partial_max, partial_sum,
        q.stride(0), q.stride(1), q.stride(2),
        k_pool.stride(0), k_pool.stride(1), k_pool.stride(2), k_pool.stride(3),
        v_pool.stride(0), v_pool.stride(1), v_pool.stride(2), v_pool.stride(3),
        seq_lens.stride(0),
        block_table.stride(0), block_table.stride(1),
        partial_out.stride(0), partial_out.stride(1), partial_out.stride(2), partial_out.stride(3),
        partial_max.stride(0), partial_max.stride(1), partial_max.stride(2),
        scale,
        GROUP=group,
        HEADS_PER_PROGRAM=heads_per_program,
        BLOCK_H=triton.next_power_of_2(heads_per_program),
        PAGE_SIZE=page_size,
        D_HEAD=head_dim,
        SPLIT_K=split_k,
        PIPELINE_STAGES=num_stages,
        num_warps=num_warps,
        num_stages=1,
    )
    if split_k == 1:
        return partial_out.squeeze(1)

    out = torch.empty_like(q)
    _splitk_reduce_kernel[(batch, query_heads)](
        partial_out, partial_max, partial_sum, out,
        partial_out.stride(0), partial_out.stride(1), partial_out.stride(2), partial_out.stride(3),
        partial_max.stride(0), partial_max.stride(1), partial_max.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        SPLIT_K=split_k,
        D_HEAD=head_dim,
        num_warps=4,
        num_stages=1,
    )
    return out
