"""Independent SM90a TMA/WGMMA attention candidate; no external attention imports.

The CUDA implementation is in hopper_attention/attention.cu. It implements a
two-stage producer/consumer pipeline and overlaps next-QK WGMMA with softmax.
It is not yet GPU-validated or performance-qualified. There is no fallback.
"""
from functools import lru_cache
import hashlib
from pathlib import Path
import sys

import torch


@lru_cache(maxsize=1)
def _extension():
    sys.path.insert(0, str(Path(__file__).parent / 'hopper_attention'))
    try:
        import inference_hopper_attention
    except ImportError as error:
        raise RuntimeError('Build custom_kernels/hopper_attention/setup.py with CUTLASS_PATH set '
                           'to CUTLASS v3.9.2; this candidate has no external attention fallback') from error
    inference_hopper_attention.validate_layouts()
    if getattr(inference_hopper_attention, 'abi_version', None) != 3:
        raise RuntimeError('Hopper extension ABI changed; rebuild custom_kernels/hopper_attention')
    expected = hashlib.sha256((Path(__file__).parent / 'hopper_attention/attention.cu').read_bytes()).hexdigest()
    if getattr(inference_hopper_attention, 'source_sha256', None) != expected:
        raise RuntimeError('Hopper extension does not match CUDA source; rebuild before measuring')
    return inference_hopper_attention


@lru_cache(maxsize=32)
def _decode_offsets(batch, device):
    # Created during eager warmup, then fixed-address and immutable for replay.
    return torch.arange(batch + 1, dtype=torch.int32, device=device)


def flash_varlen(q, k_pool, v_pool, cu_seqlens_q, block_table, seq_lens, *,
                 max_query_len, split_k=1, scale=None, causal=True, overlap_qk=True,
                 tile_n=64, register_pv=False, compact=False, worklist=None):
    """Qwen FP16 packed attention with bottom-right causal alignment.

    Q: [tokens,12,128]; KV: [pages,16,2,128]. Metadata: CUDA int32.
    Caller owns valid cu offsets, query lengths, live page IDs and context lengths.
    The wrapper only reads tensor metadata, so the forward can be captured.
    compact=True builds a worklist on GPU unless one is explicitly supplied.
    A supplied worklist must be rebuilt when query offsets change; KV lengths
    and page mappings can change without changing the query-tile worklist.
    """
    if q.ndim != 3 or q.shape[1:] != (12, 128) or q.dtype != torch.float16:
        raise ValueError('Hopper specialization requires FP16 Q [tokens,12,128]')
    if not isinstance(max_query_len, int) or max_query_len < 1:
        raise ValueError('max_query_len must be a positive host integer')
    if not isinstance(split_k, int) or not 1 <= split_k <= 64:
        raise ValueError('split_k must be an integer in [1,64]')
    if tile_n not in (64, 128):
        raise ValueError('tile_n must be 64 or 128')
    if worklist is not None and not compact:
        raise ValueError('worklist requires compact=True')
    if any(t.dtype != torch.int32 for t in (cu_seqlens_q, block_table, seq_lens)):
        raise ValueError('query offsets, block table and lengths must be int32')
    if any(not t.is_contiguous() for t in (k_pool, v_pool, cu_seqlens_q, block_table, seq_lens)):
        raise ValueError('KV and metadata must be contiguous; implicit cache/metadata copies are forbidden')
    return _extension().forward(q, k_pool, v_pool,
        cu_seqlens_q, block_table, seq_lens,
        max_query_len, split_k, 128 ** -.5 if scale is None else float(scale), bool(causal), bool(overlap_qk),
        tile_n, bool(register_pv), bool(compact), worklist)


def prepare_flash_worklist(cu_seqlens_q, total_queries):
    """GPU query-tile metadata that can be shared across layers of one iteration.

    No device count readback. Rebuild after changing cu_seqlens_q, including
    within a CUDA graph if replay can change the query-length distribution.
    """
    if not isinstance(total_queries, int) or total_queries < 1:
        raise ValueError('total_queries must be a positive host integer')
    return _extension().prepare_worklist(cu_seqlens_q, total_queries)


def flash_decode(q, k_pool, v_pool, block_table, seq_lens, *, split_k=None,
                 scale=None, max_context_length=None, tile_n=64,
                 register_pv=False, overlap_qk=True, compact=False, worklist=None):
    """One query per sequence through the same TMA/WGMMA pipeline as varlen."""
    if q.ndim != 3 or q.shape[0] < 1:
        raise ValueError('decode Q must have nonempty shape [batch,12,128]')
    if block_table.ndim != 2 or block_table.shape[0] != q.shape[0]:
        raise ValueError('decode block table must have one row per query')
    if seq_lens.shape != (q.shape[0],):
        raise ValueError('decode lengths must have one entry per query')
    if tile_n not in (64, 128):
        raise ValueError('tile_n must be 64 or 128')
    capacity = block_table.shape[1] * 16
    context = capacity if max_context_length is None else max_context_length
    if not isinstance(context, int) or not 1 <= context <= capacity:
        raise ValueError('context bound must fit the block table')
    if split_k is None:
        # Starting schedule, not an autotuned or performance-qualified rule.
        split_k = min(32, (context + tile_n - 1) // tile_n, max(1, (256 + q.shape[0] * 2 - 1) // (q.shape[0] * 2)))
    return flash_varlen(q, k_pool, v_pool, _decode_offsets(q.shape[0], q.device),
                        block_table, seq_lens, max_query_len=1, split_k=split_k,
                        scale=scale, causal=False, tile_n=tile_n, register_pv=register_pv,
                        overlap_qk=overlap_qk, compact=compact, worklist=worklist)
