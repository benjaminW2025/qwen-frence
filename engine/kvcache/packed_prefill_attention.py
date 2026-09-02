"""Dispatch and reference paths for packed variable-length prefill attention."""

from __future__ import annotations

import torch

from paged_prefill_attention import paged_prefill_attention


def packed_prefill_attention_sdpa(q, k, v, cu_seqlens, group, scale=None):
    """Reference implementation: slice each sequence and invoke causal SDPA."""
    offsets = (
        cu_seqlens
        if isinstance(cu_seqlens, (list, tuple))
        else cu_seqlens.tolist()
    )
    parts = [
        paged_prefill_attention(
            q[:, :, start:end],
            k[:, :, start:end],
            v[:, :, start:end],
            group,
            scale=scale,
        )
        for start, end in zip(offsets[:-1], offsets[1:])
    ]
    return torch.cat(parts, dim=2)


def packed_prefill_attention(
    q,
    k,
    v,
    cu_seqlens,
    max_seqlen,
    group,
    scale=None,
    backend="triton",
):
    """Apply causal attention independently to every sequence in packed Q/K/V."""
    if q.shape[1] != k.shape[1] * group:
        raise ValueError("group must equal query_heads // kv_heads")
    if backend == "sdpa":
        return packed_prefill_attention_sdpa(q, k, v, cu_seqlens, group, scale)
    if backend != "triton":
        raise ValueError(f"unknown packed prefill attention backend: {backend}")
    from kernel_dispatch import packed_prefill_attention as triton_attention

    return triton_attention(
        q,
        k,
        v,
        cu_seqlens,
        max_seqlen,
        scale=scale,
    )
