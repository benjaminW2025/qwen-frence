"""Experimental tile-local E4M3 KV decode followed by FP16 Tensor Core dots.

Not FA3, not paged, not a throughput-tuned kernel. Q is FP16. KV is stored
as E4M3 bytes and converted only in the on-chip tile. Explicit byte decoding
avoids requiring native FP8 arithmetic, enabling Ampere-or-newer experiments.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _e4m3(bits):
    value = bits.to(tl.int32)
    exponent = (value >> 3) & 15
    mantissa = value & 7
    magnitude = tl.where(exponent == 0, mantissa.to(tl.float32) * (2. ** -9),
                         (1. + mantissa.to(tl.float32) * .125) * tl.exp2(exponent - 7.))
    magnitude = tl.where((exponent == 15) & (mantissa == 7), float("nan"), magnitude)
    return tl.where((value & 128) != 0, -magnitude, magnitude)


@triton.jit
def _attention(Q, K, V, K8, V8, KS, VS, MASK, O,
               HQ: tl.constexpr, HK: tl.constexpr, NQ: tl.constexpr,
               NK: tl.constexpr, D: tl.constexpr, K_SCALE_ROWS: tl.constexpr,
               V_SCALE_ROWS: tl.constexpr, QUANT_K: tl.constexpr,
               QUANT_V: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    query_tile, head, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    kvhead = head // (HQ // HK)
    selected = tl.load(MASK + kvhead) != 0
    rows = query_tile * BM + tl.arange(0, BM)
    cols = tl.arange(0, BN)
    features = tl.arange(0, D)
    q = tl.load(Q + ((batch * HQ + head) * NQ + rows[:, None]) * D + features[None, :],
                rows[:, None] < NQ, 0)
    maximum = tl.full((BM,), float("-inf"), tl.float32)
    denominator = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, D), tl.float32)
    # Bottom-right causal alignment: queries are the final NQ tokens of NK.
    for tile in range(tl.cdiv(NK, BN)):
        positions = tile * BN + cols
        base = ((batch * HK + kvhead) * NK + positions) * D
        if QUANT_K:
            if selected:
                bits = tl.load(K8 + base[None, :] + features[:, None], positions[None, :] < NK, 0)
                scale_pos = positions if K_SCALE_ROWS != 1 else tl.full((BN,), 0, tl.int32)
                scales = tl.load(KS + (batch * HK + kvhead) * K_SCALE_ROWS + scale_pos,
                                 positions < NK, 0)
                k = (_e4m3(bits) * scales[None, :]).to(tl.float16)
            else:
                k = tl.load(K + base[None, :] + features[:, None], positions[None, :] < NK, 0)
        else:
            k = tl.load(K + base[None, :] + features[:, None], positions[None, :] < NK, 0)
        scores = tl.dot(q, k, out_dtype=tl.float32) * (D ** -.5)
        visible = (positions[None, :] < NK) & (positions[None, :] <= NK - NQ + rows[:, None])
        scores = tl.where(visible, scores, float("-inf"))
        next_max = tl.maximum(maximum, tl.max(scores, 1))
        alpha = tl.exp(maximum - next_max)
        probability = tl.exp(scores - next_max[:, None])
        denominator = denominator * alpha + tl.sum(probability, 1)
        acc = acc * alpha[:, None]
        if QUANT_V:
            if selected:
                bits = tl.load(V8 + base[:, None] + features[None, :], positions[:, None] < NK, 0)
                scale_pos = positions if V_SCALE_ROWS != 1 else tl.full((BN,), 0, tl.int32)
                scales = tl.load(VS + (batch * HK + kvhead) * V_SCALE_ROWS + scale_pos,
                                 positions < NK, 0)
                v = (_e4m3(bits) * scales[:, None]).to(tl.float16)
            else:
                v = tl.load(V + base[:, None] + features[None, :], positions[:, None] < NK, 0)
        else:
            v = tl.load(V + base[:, None] + features[None, :], positions[:, None] < NK, 0)
        acc = tl.dot(probability.to(tl.float16), v, acc, out_dtype=tl.float32)
        maximum = next_max
    result = acc / denominator[:, None]
    tl.store(O + ((batch * HQ + head) * NQ + rows[:, None]) * D + features[None, :],
             result, rows[:, None] < NQ)


def attention(q, k, v, k8, v8, k_scales, v_scales, head_mask, mode="kv"):
    if mode not in ("off", "k", "v", "kv"):
        raise ValueError("invalid KV quantization mode")
    if (q.ndim != 4 or k.ndim != 4 or k.shape != v.shape
            or q.shape[0] != k.shape[0] or q.shape[-1] != k.shape[-1]
            or q.shape[1] % k.shape[1] or not 0 < q.shape[2] <= k.shape[2]
            or q.shape[-1] not in (32, 64, 128)):
        raise ValueError("expected compatible B,H,Q,D and B,HK,L,D tensors")
    tensors = (q, k, v, k8, v8, k_scales, v_scales, head_mask)
    if any(not x.is_cuda or x.device != q.device or not x.is_contiguous() for x in tensors):
        raise ValueError("inputs must be contiguous on one CUDA device")
    if any(x.dtype != torch.float16 for x in (q, k, v)):
        raise ValueError("Q and FP16-control K/V must be FP16")
    if any(x.dtype != torch.uint8 or x.shape != k.shape for x in (k8, v8)):
        raise ValueError("FP8 storage must have K/V shape and uint8 dtype")
    for scales in (k_scales, v_scales):
        if (scales.dtype != torch.float32 or scales.shape[:2] != k.shape[:2]
                or scales.ndim != 3 or scales.shape[2] not in (1, k.shape[2])):
            raise ValueError("scales must be FP32 B,HK,1 or B,HK,L")
    if head_mask.shape != (k.shape[1],) or head_mask.dtype != torch.int32:
        raise ValueError("head mask must be int32 with one entry per KV head")
    output = torch.empty_like(q)
    with torch.cuda.device(q.device):
        _attention[(triton.cdiv(q.shape[2], 16), q.shape[1], q.shape[0])](
            *tensors[:-1], head_mask, output,
            q.shape[1], k.shape[1], q.shape[2], k.shape[2], q.shape[3],
            k_scales.shape[2], v_scales.shape[2], mode in ("k", "kv"),
            mode in ("v", "kv"), 16, 32, num_warps=4, num_stages=2)
    return output
