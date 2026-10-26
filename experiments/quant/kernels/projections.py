"""FP8 projections using PyTorch scaled_mm (cuBLASLt on the target stack).

Target: PyTorch 2.13+, CUDA 12.9+, Hopper. Triton only quantizes activations;
there is no custom matrix multiplication. Bias uses the library GEMM, as in the original linear projection.
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
    if weights.shape[0] % 16 or weights.shape[1] % 16:
        raise ValueError("cuBLASLt FP8 requires weight dimensions divisible by 16")
    # A is row-major; transposing the stored row-major weights gives the
    # column-major B layout required by cuBLASLt without copying weights.
    # Only the required FP8 operand scales are applied by scaled_mm.
    # Library bias matches nn.Linear; no activation or residual fusion.
    out = torch._scaled_mm(
        a, weights.T, scale_a=scales.view(-1, 1),
        scale_b=weight_scales.view(1, -1), bias=bias, out_dtype=torch.float16,
        use_fast_accum=False)
    return out.view(*x.shape[:-1], weights.shape[0])
