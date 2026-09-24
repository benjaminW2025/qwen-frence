"""Packed variable-query-length FA3 over the engine's paged KV cache."""

from __future__ import annotations

import torch


def _load_fa3_varlen():
    try:
        from vllm.vllm_flash_attn import flash_attn_varlen_func
    except (ImportError, AttributeError):
        try:
            from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func
        except (ImportError, AttributeError) as error:
            raise RuntimeError("installed vLLM does not expose packed FA3 varlen") from error
    return flash_attn_varlen_func


def fa3_paged_varlen_attention(q, k_pool, v_pool, cu_seqlens_q,
                              block_table, seq_lens, *, max_query_len,
                              max_context_len=None, scale=None):
    """Return packed (total_q, query_heads, head_dim) causal attention.

    K/V for the new query tokens must already be in the paged cache. FA3's
    bottom-right causal alignment handles decode and prefill in the same call.
    """
    if q.ndim != 3 or k_pool.ndim != 4 or k_pool.shape != v_pool.shape:
        raise ValueError("expected packed Q and matching paged K/V pools")
    if (cu_seqlens_q.ndim != 1 or block_table.ndim != 2 or seq_lens.ndim != 1
            or cu_seqlens_q.numel() != seq_lens.numel() + 1
            or block_table.shape[0] != seq_lens.numel()):
        raise ValueError("packed query offsets, lengths, and page-table rows differ")
    if q.shape[1] % k_pool.shape[2] or q.shape[2] != k_pool.shape[3]:
        raise ValueError("Q/K/V head dimensions or GQA grouping differ")
    if (cu_seqlens_q.dtype != torch.int32 or block_table.dtype != torch.int32
            or seq_lens.dtype != torch.int32):
        raise ValueError("FA3 offsets, page table, and lengths must be int32")
    if (q.dtype not in (torch.float16, torch.bfloat16)
            or k_pool.dtype != q.dtype or v_pool.dtype != q.dtype):
        raise ValueError("FA3 Q/K/V must share a half-precision dtype")
    if not all(t.is_cuda and t.device == q.device
               for t in (q, k_pool, v_pool, cu_seqlens_q, block_table, seq_lens)):
        raise ValueError("FA3 inputs must share one CUDA device")
    if max_query_len < 1 or (max_context_len is not None and max_context_len < 1):
        raise ValueError("FA3 query and context bounds must be positive")
    if scale is None:
        scale = q.shape[-1] ** -0.5
    result = _load_fa3_varlen()(
        q=q.contiguous(), k=k_pool, v=v_pool,
        cu_seqlens_q=cu_seqlens_q, seqused_k=seq_lens,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_context_len or block_table.shape[1] * k_pool.shape[1],
        block_table=block_table, softmax_scale=scale,
        causal=True, fa_version=3,
    )
    return result[0] if isinstance(result, tuple) else result


def smoke_varlen_fa3(device, *, capture_graph=False):
    """Cheap GPU API/numerics/capture gate before the model is allocated."""
    generator = torch.Generator(device=device).manual_seed(23)
    q = torch.randn((3, 12, 128), device=device, dtype=torch.float16,
                    generator=generator)
    k = torch.randn((2, 16, 2, 128), device=device, dtype=torch.float16,
                    generator=generator)
    v = torch.randn((2, 16, 2, 128), device=device, dtype=torch.float16,
                    generator=generator)
    cu = torch.tensor([0, 1, 3], device=device, dtype=torch.int32)
    lengths = torch.tensor([3, 2], device=device, dtype=torch.int32)
    table = torch.tensor([[0], [1]], device=device, dtype=torch.int32)

    def forward():
        return fa3_paged_varlen_attention(
            q, k, v, cu, table, lengths, max_query_len=2,
            max_context_len=3)

    actual = forward()
    expected = []
    for batch, (start, end, kv_length) in enumerate(((0, 1, 3), (1, 3, 2))):
        query = q[start:end].float().transpose(0, 1)
        keys = k[batch, :kv_length].float().repeat_interleave(6, dim=1).transpose(0, 1)
        values = v[batch, :kv_length].float().repeat_interleave(6, dim=1).transpose(0, 1)
        scores = query @ keys.transpose(-1, -2) * (128 ** -0.5)
        query_index = torch.arange(end - start, device=device)[:, None]
        key_index = torch.arange(kv_length, device=device)[None, :]
        visible = key_index <= kv_length - (end - start) + query_index
        scores = scores.masked_fill(~visible[None, :, :], float("-inf"))
        expected.append((torch.softmax(scores, dim=-1) @ values)
                        .transpose(0, 1))
    reference = torch.cat(expected).to(actual.dtype)
    torch.testing.assert_close(actual, reference, atol=.05, rtol=.02)
    if capture_graph:
        current = torch.cuda.current_stream(device)
        side = torch.cuda.Stream(device=device)
        side.wait_stream(current)
        with torch.cuda.stream(side):
            forward()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=side):
                captured = forward()
        current.wait_stream(side)
        graph.replay()
        torch.testing.assert_close(captured, reference, atol=.05, rtol=.02)
    return {"packed_queries": 3, "sequences": 2, "graph_capture": capture_graph}
