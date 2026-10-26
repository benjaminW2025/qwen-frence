"""Accuracy-first FP8 Tensor Core GEMM, not a tuned throughput candidate.

tl.dot uses FP32 accumulators and max_num_imprecise_acc=0 to request the
compiler's accuracy-oriented FP8 accumulation. Hardware instruction precision
is not equivalent to an exact IEEE FP32 multiplication of original inputs.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _quantize(X, Q, S, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * K + col, col < K, 0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(x), 0) / 448.0, 1e-12)
    q = tl.minimum(tl.maximum(x / scale, -448.0), 448.0)
    tl.store(Q + row * K + col, q, col < K)
    tl.store(S + row, scale)


@triton.jit
def _gemm(A, W, AS, WS, BIAS, Y, M: tl.constexpr, N: tl.constexpr,
          K: tl.constexpr, HAS_BIAS: tl.constexpr,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        kk = start * BK + k
        a = tl.load(A + m[:, None] * K + kk[None, :],
                    (m[:, None] < M) & (kk[None, :] < K), 0)
        w = tl.load(W + n[None, :] * K + kk[:, None],
                    (n[None, :] < N) & (kk[:, None] < K), 0)
        acc = tl.dot(a, w, acc, out_dtype=tl.float32, max_num_imprecise_acc=0)
    sa = tl.load(AS + m, m < M, 0)
    sw = tl.load(WS + n, n < N, 0)
    result = (acc * sa[:, None]) * sw[None, :]
    if HAS_BIAS:
        result += tl.load(BIAS + n, n < N, 0).to(tl.float32)[None, :]
    tl.store(Y + m[:, None] * N + n[None, :], result,
             (m[:, None] < M) & (n[None, :] < N))


def quantize_activations(x):
    if not x.is_cuda or x.dtype != torch.float16 or x.ndim != 2 or not x.numel():
        raise ValueError("activation quantization requires a nonempty CUDA FP16 matrix")
    x = x.contiguous()
    q = torch.empty(x.shape, device=x.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty(x.shape[0], device=x.device, dtype=torch.float32)
    with torch.cuda.device(x.device):
        _quantize[(x.shape[0],)](x, q, scales, x.shape[1],
                                triton.next_power_of_2(x.shape[1]), num_warps=4)
    return q, scales


def fp8_linear(x, weights, weight_scales, bias=None):
    if (weights.ndim != 2 or x.ndim < 1 or x.shape[-1] != weights.shape[1]
            or weights.dtype != torch.float8_e4m3fn
            or not weights.is_contiguous() or weights.device != x.device
            or weight_scales.shape != (weights.shape[0],)
            or weight_scales.dtype != torch.float32 or weight_scales.device != x.device
            or not weight_scales.is_contiguous()):
        raise ValueError("incompatible FP8 projection operands")
    if bias is not None and (bias.shape != (weights.shape[0],) or bias.device != x.device
                             or bias.dtype != torch.float16 or not bias.is_contiguous()):
        raise ValueError("incompatible bias")
    a, scales = quantize_activations(x.reshape(-1, x.shape[-1]))
    m, k = a.shape
    n = weights.shape[0]
    out = torch.empty((m, n), device=x.device, dtype=torch.float16)
    with torch.cuda.device(x.device):
        _gemm[(triton.cdiv(m, 16), triton.cdiv(n, 64))](
            a, weights, scales, weight_scales, bias, out, m, n, k,
            bias is not None, 16, 64, 32, num_warps=4, num_stages=3,
            enable_fp_fusion=False)
    return out.view(*x.shape[:-1], n)
