"""SM90a GEMMs with fused epilogues on CUTLASS's mainloop (gemm_epilogue/fused_gemm.cu).

One decoder layer's non-attention work as four GEMMs, with no RMSNorm kernel:

    x1, p1 = residual_gemm(attn, o_proj.weight, x0)            # x1 = x0 + attn @ Wo.T
    act    = gate_up_swiglu(x1, gate_up_weight, p1)            # post-attention norm folded in
    x2, p2 = residual_gemm(act, down_proj.weight, x1)          # x2 = x1 + act @ Wd.T
    q      = qkv_rope_cache(x2, qkv_weight, qkv_bias, p2, ...)  # next layer's input norm folded in

RMSNorm is split three ways: the producer's epilogue writes per-row partial sums
of squares; gamma is folded into the consumer's weight once (prepare_*); the
consumer's epilogue scales each accumulator row by rsqrt(mean + eps), which
commutes with the matmul. Opt-in and unqualified until benchmarked.
"""
from functools import lru_cache
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

ABI_VERSION = 4
INTERLEAVE_BLOCK = 8
PARTIAL_TILE = 256
QKV_WIDTH = 2048


@lru_cache(maxsize=1)
def _extension():
    sys.path.insert(0, str(Path(__file__).parent / 'gemm_epilogue'))
    try:
        import inference_gemm_epilogue
    except ImportError as error:
        raise RuntimeError('Build custom_kernels/gemm_epilogue/setup.py with CUTLASS_PATH set '
                           'to CUTLASS v3.9.2') from error
    binary = Path(inference_gemm_epilogue.__file__)
    if getattr(inference_gemm_epilogue, 'abi_version', None) != ABI_VERSION:
        raise RuntimeError(f'Stale GEMM epilogue extension at {binary}: expected ABI {ABI_VERSION}. '
                           'Rebuild custom_kernels/gemm_epilogue with build_ext --inplace --force')
    source = Path(__file__).parent / 'gemm_epilogue/fused_gemm.cu'
    if getattr(inference_gemm_epilogue, 'source_sha256', None) != hashlib.sha256(source.read_bytes()).hexdigest():
        raise RuntimeError(f'Stale GEMM epilogue extension at {binary}: CUDA source hash differs. '
                           'Rebuild custom_kernels/gemm_epilogue with build_ext --inplace --force')
    if (inference_gemm_epilogue.interleave_block, inference_gemm_epilogue.partial_tile) != \
            (INTERLEAVE_BLOCK, PARTIAL_TILE):
        raise RuntimeError('GEMM epilogue layout constants differ from the Python contract')
    inference_gemm_epilogue.validate_fragment()
    return inference_gemm_epilogue


# ---------------------------------------------------------------- weight preparation (load time)

def fold_norm(weight, gamma):
    """W * gamma along K, rounded once: rmsnorm(x) @ W.T == r * (x @ fold_norm(W, gamma).T)."""
    if weight.dim() != 2 or gamma.shape != (weight.shape[1],):
        raise ValueError('gamma must match the weight input dimension')
    return (weight.float() * gamma.float()[None, :]).to(weight.dtype).contiguous()


def interleave_gate_up(gate_weight, up_weight):
    """[F, K] gate and up -> [2F, K], rows [g0..g7, u0..u7, g8..g15, u8..u15, ...].

    The GEMM then produces every gate column eight columns before its up partner,
    inside the same thread's registers.
    """
    if gate_weight.shape != up_weight.shape or gate_weight.dim() != 2:
        raise ValueError('gate and up weights must share one [F, K] shape')
    rows, columns = gate_weight.shape
    if rows % INTERLEAVE_BLOCK:
        raise ValueError(f'F must be a multiple of {INTERLEAVE_BLOCK}')
    blocks = rows // INTERLEAVE_BLOCK
    return torch.stack((gate_weight.reshape(blocks, INTERLEAVE_BLOCK, columns),
                        up_weight.reshape(blocks, INTERLEAVE_BLOCK, columns)), dim=1).reshape(2 * rows, columns).contiguous()


def split_interleaved(output):
    """Inverse column order: [M, 2F] -> ([M, F] gate, [M, F] up)."""
    rows, width = output.shape
    blocks = output.reshape(rows, width // (2 * INTERLEAVE_BLOCK), 2, INTERLEAVE_BLOCK)
    return blocks[:, :, 0, :].reshape(rows, width // 2), blocks[:, :, 1, :].reshape(rows, width // 2)


def prepare_gate_up(gate_weight, up_weight, post_attention_gamma):
    """The fused gate/up weight: norm folded, then interleaved."""
    return interleave_gate_up(fold_norm(gate_weight, post_attention_gamma),
                              fold_norm(up_weight, post_attention_gamma))


def prepare_qkv(qkv_weight, input_gamma):
    """The fused packed QKV weight [2048, K] with the input norm folded in (bias unchanged)."""
    if qkv_weight.shape[0] != QKV_WIDTH:
        raise ValueError('packed QKV weight must have 2048 rows')
    return fold_norm(qkv_weight, input_gamma)


def prepare_model(model):
    """Attach each layer's fused weights once; idempotent.

    Norm folding needs the *next* consumer's gamma: gate/up takes the layer's
    post-attention norm and QKV its input norm. o_proj and down_proj are used as
    loaded. The originals stay for the eager fallback, so this costs one extra
    copy of the gate/up and QKV weights (about 1.7 GB for Qwen2.5-1.5B).
    """
    prepared = getattr(model, 'fused_gemm_layers', None)
    if prepared is not None:
        return prepared
    prepared = []
    for layer in model.layers:
        if layer.qkv_proj is None or layer.gate_up_proj is None:
            raise ValueError('fused GEMM epilogues require packed QKV and gate/up weights')
        gate, up = layer.gate_up_proj.weight.detach().chunk(2, dim=0)
        prepared.append(SimpleNamespace(
            qkv=prepare_qkv(layer.qkv_proj.weight.detach(), layer.input_norm.weight.detach()),
            qkv_bias=None if layer.qkv_proj.bias is None else layer.qkv_proj.bias.detach().contiguous(),
            o=layer.o_proj.weight.detach().contiguous(),
            gate_up=prepare_gate_up(gate, up, layer.post_attn_norm.weight.detach()),
            down=layer.down_proj.weight.detach().contiguous()))
    model.fused_gemm_layers = prepared
    return prepared


def attention_inputs(weights, x, partials, cfg, *, positions, slots, k_pool, v_pool, valid_tokens=None):
    """The layer's input norm, QKV, bias, RoPE and cache write in one GEMM; returns Q [M, 12, 128]."""
    return qkv_rope_cache(x, weights.qkv, weights.qkv_bias, partials, positions=positions, slots=slots,
                          k_pool=k_pool, v_pool=v_pool, valid_tokens=valid_tokens,
                          eps=cfg.rms_norm_eps, theta=cfg.rope_theta)


def layer_tail(weights, attention, residual, cfg):
    """o_proj + residual, then the MLP with its norm folded in: returns (x, partials) for the next layer."""
    x, partials = residual_gemm(attention, weights.o, residual)
    act = gate_up_swiglu(x, weights.gate_up, partials, eps=cfg.rms_norm_eps)
    return residual_gemm(act, weights.down, x)


def row_square_partials(x):
    """Per-row partial sums of squares in PARTIAL_TILE-column chunks, as residual_gemm writes them.

    For inputs no GEMM produced, such as the embedding output before layer 0.
    """
    rows, width = x.shape
    if width % PARTIAL_TILE:
        raise ValueError(f'hidden size must be a multiple of {PARTIAL_TILE}')
    return x.float().square().reshape(rows, width // PARTIAL_TILE, PARTIAL_TILE).sum(-1).contiguous()


# ---------------------------------------------------------------- kernels
# Thin forwards: argument validation lives in the C++ bindings (TORCH_CHECK).
# ``config`` selects a compiled tile/cluster/scheduler candidate (see configs());
# -1 picks the default for the row count (decode config at M <= 64).

def configs():
    """The candidates compiled into this build; all of them in a GEMM_EPILOGUE_SWEEP build."""
    return _extension().configs()


def residual_gemm(x, weight, residual, *, partials=True, out=None, config=-1):
    """(residual + x @ weight.T, [M, N/tile] FP32 partial sums of squares of the FP16 output); out may alias residual."""
    return _extension().residual_gemm(x, weight, residual, bool(partials), out, int(config))


def gate_up_swiglu(x, weight, partials=None, *, hidden=None, eps=1e-6, config=-1):
    """silu(g) * u from one GEMM on an interleaved weight; with partials, rows are RMS-normalized first."""
    return _extension().gate_up_swiglu(x, weight, partials, float(hidden or x.shape[1]), float(eps), int(config))


def qkv_rope_cache(x, weight, bias, partials, *, positions, slots, k_pool, v_pool,
                   valid_tokens=None, hidden=None, eps=1e-6, theta=1_000_000.0, config=-1):
    """Normalized QKV + bias, rotate-half RoPE, K/V into the paged cache; returns Q [M, 12, 128].

    Padded rows (>= valid_tokens) get zero Q and never touch the cache, as in packed_qkv_rope_cache.
    """
    return _extension().qkv_rope_cache(x, weight, bias, partials, float(hidden or x.shape[1]), float(eps),
                                       positions, slots, k_pool, v_pool, valid_tokens, float(theta), int(config))


def gemm_plain(x, weight):
    """x @ weight.T through the same CUTLASS mainloop, plain output (mainloop parity diagnostic)."""
    return _extension().gemm_plain(x, weight)
