"""Grouped + Split-K + Pipelined decode attention experiment.

Combines three optimizations:
1. Grouping (hpp=6): Minimize KV reads by sharing across GQA heads (6x reduction)
2. Split-K: Parallelize across KV blocks to saturate SMs
3. Pipelining: Async loads to hide memory latency within each program

The goal: saturate SMs with just enough programs, then hide latency via pipelining,
while minimizing total memory traffic via grouping.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


def get_num_sms() -> int:
    """Query SM count from current CUDA device."""
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


def compute_split_k(
    batch_size: int,
    kv_heads: int,
    head_tiles: int,
    num_kv_blocks: int,
    target_occupancy: int = 4,
) -> int:
    """Compute K to reach target_occupancy programs per SM.

    Args:
        batch_size: Number of sequences in batch
        kv_heads: Number of KV heads
        head_tiles: Number of head tiles (ceil(GROUP / heads_per_program))
        num_kv_blocks: Total KV blocks per sequence
        target_occupancy: Target programs per SM (1, 2, 4, or 8)

    Returns:
        K value for split-k parallelization
    """
    num_sms = get_num_sms()
    current_programs = batch_size * kv_heads * head_tiles
    target_programs = num_sms * target_occupancy

    if current_programs >= target_programs:
        return 1  # Already saturated

    # How much do we need to split?
    k = (target_programs + current_programs - 1) // current_programs

    # Don't split more than we have blocks
    # Each chunk should have at least 2 blocks for pipelining to help
    min_blocks_per_chunk = 2
    max_k = max(1, num_kv_blocks // min_blocks_per_chunk)

    return min(k, max_k)


@triton.jit
def _grouped_splitk_partial_kernel(
    q_ptr, k_pool_ptr, v_pool_ptr, block_table_ptr, seq_lens_ptr,
    partial_out_ptr, partial_max_ptr, partial_sum_ptr,
    stride_qb, stride_qh, stride_qd,
    stride_pblk, stride_pt, stride_pkv, stride_pd,
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
):
    """Compute partial attention for a chunk of KV blocks.

    Uses double-buffering for pipelined loads.
    """
    sequence = tl.program_id(0)
    kv_head = tl.program_id(1)
    head_tile = tl.program_id(2) % tl.cdiv(GROUP, HEADS_PER_PROGRAM)
    k_idx = tl.program_id(2) // tl.cdiv(GROUP, HEADS_PER_PROGRAM)

    # Head indexing
    head_offsets = tl.arange(0, BLOCK_H)
    dim_offsets = tl.arange(0, D_HEAD)
    token_offsets = tl.arange(0, PAGE_SIZE)
    heads_in_group = head_tile * HEADS_PER_PROGRAM + head_offsets
    head_mask = (head_offsets < HEADS_PER_PROGRAM) & (heads_in_group < GROUP)
    query_heads = kv_head * GROUP + heads_in_group

    # Load query (once per program)
    q = tl.load(
        q_ptr
        + sequence * stride_qb
        + query_heads[:, None] * stride_qh
        + dim_offsets[None, :] * stride_qd,
        mask=head_mask[:, None],
        other=0.0,
    )

    seq_len = tl.load(seq_lens_ptr + sequence)
    total_pages = tl.cdiv(seq_len, PAGE_SIZE)

    # This program handles pages [start_page, end_page)
    pages_per_chunk = tl.cdiv(total_pages, SPLIT_K)
    start_page = k_idx * pages_per_chunk
    end_page = tl.minimum(start_page + pages_per_chunk, total_pages)

    # Online softmax state
    running_max = tl.full([BLOCK_H], float("-inf"), tl.float32)
    running_sum = tl.zeros([BLOCK_H], tl.float32)
    accumulator = tl.zeros([BLOCK_H, D_HEAD], tl.float32)

    # Main loop with software pipelining
    # Load first page
    if start_page < end_page:
        page_id_curr = tl.load(
            block_table_ptr + sequence * stride_btb + start_page * stride_btm
        )
        positions_curr = start_page * PAGE_SIZE + token_offsets
        token_mask_curr = positions_curr < seq_len
        base_curr = page_id_curr * stride_pblk + kv_head * stride_pkv
        pool_offsets = token_offsets[:, None] * stride_pt + dim_offsets[None, :] * stride_pd

        keys_curr = tl.load(
            k_pool_ptr + base_curr + pool_offsets,
            mask=token_mask_curr[:, None],
            other=0.0,
        )
        values_curr = tl.load(
            v_pool_ptr + base_curr + pool_offsets,
            mask=token_mask_curr[:, None],
            other=0.0,
        )

    for page_index in range(start_page, end_page):
        # Current page data (already loaded)
        keys = keys_curr
        values = values_curr
        positions = page_index * PAGE_SIZE + token_offsets
        token_mask = positions < seq_len

        # Prefetch next page (if exists)
        next_page = page_index + 1
        if next_page < end_page:
            page_id_next = tl.load(
                block_table_ptr + sequence * stride_btb + next_page * stride_btm
            )
            positions_next = next_page * PAGE_SIZE + token_offsets
            token_mask_next = positions_next < seq_len
            base_next = page_id_next * stride_pblk + kv_head * stride_pkv

            keys_curr = tl.load(
                k_pool_ptr + base_next + pool_offsets,
                mask=token_mask_next[:, None],
                other=0.0,
            )
            values_curr = tl.load(
                v_pool_ptr + base_next + pool_offsets,
                mask=token_mask_next[:, None],
                other=0.0,
            )

        # Compute attention for current page
        scores = tl.sum(
            q[:, None, :].to(tl.float32) * keys[None, :, :].to(tl.float32),
            axis=2,
        ) * scale

        score_mask = head_mask[:, None] & token_mask[None, :]
        scores = tl.where(score_mask, scores, float("-inf"))
        scores = tl.where(head_mask[:, None], scores, 0.0)

        # Online softmax update
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

    # Handle empty chunk
    running_max = tl.where(running_sum > 0, running_max, 0.0)

    # Store partial results
    tl.store(
        partial_out_ptr
        + sequence * stride_pob
        + k_idx * stride_pok
        + query_heads[:, None] * stride_poh
        + dim_offsets[None, :] * stride_pod,
        accumulator.to(partial_out_ptr.dtype.element_ty),
        mask=head_mask[:, None],
    )
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

    # Load first partial
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

    # Combine remaining partials
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

        # Online softmax combination
        new_max = tl.maximum(global_max, partial_max)
        alpha_global = tl.exp(global_max - new_max)
        alpha_partial = tl.exp(partial_max - new_max)

        accumulator = accumulator * alpha_global + partial_out * alpha_partial
        global_sum = global_sum * alpha_global + partial_sum * alpha_partial
        global_max = new_max

    # Normalize and store
    output = accumulator / global_sum
    tl.store(
        out_ptr
        + sequence * stride_ob
        + query_head * stride_oh
        + dim_offsets * stride_od,
        output.to(out_ptr.dtype.element_ty),
    )


def grouped_splitk_pipelined_attention(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    heads_per_program: int = 6,
    split_k: int | None = None,
    target_occupancy: int = 4,
    scale: float | None = None,
    num_warps: int = 4,
    num_stages: int = 2,
) -> torch.Tensor:
    """Grouped + Split-K + Pipelined decode attention.

    Args:
        q: Query tensor [batch, query_heads, head_dim]
        k_pool: Key pool [num_pages, page_size, kv_heads, head_dim]
        v_pool: Value pool [num_pages, page_size, kv_heads, head_dim]
        block_table: Page table [batch, max_pages]
        seq_lens: Sequence lengths [batch]
        heads_per_program: Query heads per program (1, 2, 3, or 6)
        split_k: Manual K value, or None for auto-compute
        target_occupancy: Programs per SM target when auto-computing K
        scale: Attention scale (default: 1/sqrt(head_dim))
        num_warps: Warps per program
        num_stages: Pipeline stages

    Returns:
        Output tensor [batch, query_heads, head_dim]
    """
    batch_size, query_heads, head_dim = q.shape
    page_size = k_pool.shape[1]
    kv_heads = k_pool.shape[2]
    group = query_heads // kv_heads

    if heads_per_program not in (1, 2, 3, 6) or group % heads_per_program:
        raise ValueError("heads_per_program must divide GROUP evenly")

    head_tiles = (group + heads_per_program - 1) // heads_per_program
    max_context = seq_lens.max().item()
    num_kv_blocks = (max_context + page_size - 1) // page_size

    # Auto-compute split_k if not provided
    if split_k is None:
        split_k = compute_split_k(
            batch_size, kv_heads, head_tiles, num_kv_blocks, target_occupancy
        )

    if scale is None:
        scale = head_dim ** -0.5

    # For K=1, skip partial storage and reduction
    if split_k == 1:
        output = torch.empty_like(q)
        block_h = triton.next_power_of_2(heads_per_program)
        grid = (batch_size, kv_heads, head_tiles)

        # Allocate dummy partials (won't be used, but kernel expects them)
        partial_out = torch.empty(
            (batch_size, 1, query_heads, head_dim),
            dtype=q.dtype, device=q.device
        )
        partial_max = torch.empty(
            (batch_size, 1, query_heads),
            dtype=torch.float32, device=q.device
        )
        partial_sum = torch.empty_like(partial_max)

        _grouped_splitk_partial_kernel[grid](
            q, k_pool, v_pool, block_table, seq_lens,
            partial_out, partial_max, partial_sum,
            q.stride(0), q.stride(1), q.stride(2),
            k_pool.stride(0), k_pool.stride(1), k_pool.stride(2), k_pool.stride(3),
            block_table.stride(0), block_table.stride(1),
            partial_out.stride(0), partial_out.stride(1), partial_out.stride(2), partial_out.stride(3),
            partial_max.stride(0), partial_max.stride(1), partial_max.stride(2),
            scale,
            GROUP=group,
            HEADS_PER_PROGRAM=heads_per_program,
            BLOCK_H=block_h,
            PAGE_SIZE=page_size,
            D_HEAD=head_dim,
            SPLIT_K=1,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        # For K=1, partial_out is the final output (after normalization)
        # Actually we need to normalize - let's just copy from partial
        output = partial_out.squeeze(1)
        # Normalize
        output = output / partial_sum.squeeze(1).unsqueeze(-1)
        return output

    # Allocate partial results
    partial_out = torch.empty(
        (batch_size, split_k, query_heads, head_dim),
        dtype=q.dtype, device=q.device
    )
    partial_max = torch.empty(
        (batch_size, split_k, query_heads),
        dtype=torch.float32, device=q.device
    )
    partial_sum = torch.empty_like(partial_max)

    # Launch partial kernel
    block_h = triton.next_power_of_2(heads_per_program)
    grid_partial = (batch_size, kv_heads, head_tiles * split_k)

    _grouped_splitk_partial_kernel[grid_partial](
        q, k_pool, v_pool, block_table, seq_lens,
        partial_out, partial_max, partial_sum,
        q.stride(0), q.stride(1), q.stride(2),
        k_pool.stride(0), k_pool.stride(1), k_pool.stride(2), k_pool.stride(3),
        block_table.stride(0), block_table.stride(1),
        partial_out.stride(0), partial_out.stride(1), partial_out.stride(2), partial_out.stride(3),
        partial_max.stride(0), partial_max.stride(1), partial_max.stride(2),
        scale,
        GROUP=group,
        HEADS_PER_PROGRAM=heads_per_program,
        BLOCK_H=block_h,
        PAGE_SIZE=page_size,
        D_HEAD=head_dim,
        SPLIT_K=split_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    # Launch reduce kernel
    output = torch.empty_like(q)
    grid_reduce = (batch_size, query_heads)

    _splitk_reduce_kernel[grid_reduce](
        partial_out, partial_max, partial_sum, output,
        partial_out.stride(0), partial_out.stride(1), partial_out.stride(2), partial_out.stride(3),
        partial_max.stride(0), partial_max.stride(1), partial_max.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        SPLIT_K=split_k,
        D_HEAD=head_dim,
        num_warps=4,
        num_stages=1,
    )

    return output
