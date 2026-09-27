"""Independent SM90a WGMMA attention candidate; no external attention imports.

The CUDA implementation is in hopper_attention/attention.cu: a producer
warpgroup gathers paged KV with cp.async into a multi-stage shared-memory ring
(FA3's own loader for 16-token pages), and one or two
consumer warpgroups (64 query rows each) overlap softmax with the PV GEMM,
ping-ponging GEMM issue when there are two. It is not yet GPU-validated or
performance-qualified. There is no fallback.
"""
from functools import lru_cache
import hashlib
from pathlib import Path
import sys

import torch

ABI_VERSION = 5
GQA_GROUP = 6
CONSUMER_ROWS = 64


def default_consumers(max_query_len):
    """One 64-row consumer warpgroup unless a sequence can fill more than one.

    Starting rule, not an autotuned policy: decode (6 packed rows) keeps one
    consumer; prefill and mixed steps whose longest sequence exceeds 64 packed
    (query, GQA head) rows use two.
    """
    return 2 if max_query_len * GQA_GROUP > CONSUMER_ROWS else 1


def _check_consumers(consumers):
    if consumers not in (1, 2):
        raise ValueError('consumers must be 1 or 2 warpgroups')


@lru_cache(maxsize=1)
def _extension():
    sys.path.insert(0, str(Path(__file__).parent / 'hopper_attention'))
    try:
        import inference_hopper_attention
    except ImportError as error:
        raise RuntimeError('Build custom_kernels/hopper_attention/setup.py with CUTLASS_PATH set '
                           'to CUTLASS v3.9.2; this candidate has no external attention fallback') from error
    binary = Path(inference_hopper_attention.__file__)
    if getattr(inference_hopper_attention, 'abi_version', None) != ABI_VERSION:
        raise RuntimeError(f'Stale Hopper extension at {binary}: expected ABI {ABI_VERSION}, got '
                           f'{getattr(inference_hopper_attention, "abi_version", None)}. '
                           'Rebuild custom_kernels/hopper_attention with build_ext --inplace --force')
    source_dir = Path(__file__).parent / 'hopper_attention'
    expected = hashlib.sha256(b'\0'.join((source_dir / name).read_bytes()
        for name in ('attention.cu', 'shared_layouts.cuh'))).hexdigest()
    if getattr(inference_hopper_attention, 'source_sha256', None) != expected:
        raise RuntimeError(f'Stale Hopper extension at {binary}: CUDA source hash differs. '
                           'Rebuild custom_kernels/hopper_attention with build_ext --inplace --force')
    inference_hopper_attention.validate_layouts()
    return inference_hopper_attention


@lru_cache(maxsize=32)
def _decode_offsets(batch, device):
    # Created during eager warmup, then fixed-address and immutable for replay.
    return torch.arange(batch + 1, dtype=torch.int32, device=device)


def flash_varlen(q, k_pool, v_pool, cu_seqlens_q, block_table, seq_lens, *,
                 max_query_len, split_k=1, scale=None, causal=True, overlap_qk=True,
                 tile_n=64, register_pv=False, consumers=None, compact=False, worklist=None):
    """Qwen FP16 packed attention with bottom-right causal alignment.

    Q: [tokens,12,128]; KV: [pages,16,2,128]. Metadata: CUDA int32.
    Caller owns valid cu offsets, query lengths, live page IDs and context lengths.
    The wrapper only reads tensor metadata, so the forward can be captured.
    compact=True builds a worklist on GPU unless one is explicitly supplied.
    A supplied worklist must be rebuilt when query offsets change; KV lengths
    and page mappings can change without changing the query-tile worklist.
    overlap_qk=False serializes each warpgroup's GEMMs with its softmax (a
    control for measuring the pipelining). consumers=None picks
    default_consumers(max_query_len); a prepared worklist fixes the query-tile
    height, so it requires consumers to be passed explicitly and to match.
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
    if consumers is None:
        if worklist is not None:
            raise ValueError('a prepared worklist fixes the query-tile height; pass consumers explicitly')
        consumers = default_consumers(max_query_len)
    _check_consumers(consumers)
    if any(t.dtype != torch.int32 for t in (cu_seqlens_q, block_table, seq_lens)):
        raise ValueError('query offsets, block table and lengths must be int32')
    if any(not t.is_contiguous() for t in (k_pool, v_pool, cu_seqlens_q, block_table, seq_lens)):
        raise ValueError('KV and metadata must be contiguous; implicit cache/metadata copies are forbidden')
    return _extension().forward(q, k_pool, v_pool,
        cu_seqlens_q, block_table, seq_lens,
        max_query_len, split_k, 128 ** -.5 if scale is None else float(scale), bool(causal), bool(overlap_qk),
        tile_n, bool(register_pv), consumers, bool(compact), worklist)


def prepare_flash_worklist(cu_seqlens_q, total_queries, *, consumers):
    """GPU query-tile metadata that can be shared across layers of one iteration.

    No device count readback. Rebuild after changing cu_seqlens_q, including
    within a CUDA graph if replay can change the query-length distribution.
    Tiles are 64 * consumers packed rows; the kernel traps on a worklist built
    for a different consumer count, so pass the same value to flash_varlen.
    """
    if not isinstance(total_queries, int) or total_queries < 1:
        raise ValueError('total_queries must be a positive host integer')
    _check_consumers(consumers)
    return _extension().prepare_worklist(cu_seqlens_q, total_queries, consumers)


def flash_decode(q, k_pool, v_pool, block_table, seq_lens, *, split_k=None,
                 scale=None, max_context_length=None, tile_n=64,
                 register_pv=False, overlap_qk=True, consumers=1, compact=False, worklist=None):
    """One query per sequence through the same paged-KV/WGMMA pipeline as varlen.

    A decode sequence packs six (query, GQA head) rows, so one consumer
    warpgroup is the default; a second would only run the pipeline protocol.
    """
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
                        overlap_qk=overlap_qk, consumers=consumers, compact=compact,
                        worklist=worklist)
