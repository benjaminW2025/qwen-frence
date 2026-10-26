"""Experimental row-scaled E4M3 projections; never enabled by accepted config.

Weights use one FP32 dequantization scale per output channel. Activations use
one scale per token. The CUDA implementation is imported only on execution so
weight preparation and experiment controls can be tested without Triton.
"""
from contextlib import contextmanager

import torch
from torch import nn

GROUPS = {"gate-up": "gate_up_proj", "down": "down_proj",
          "output": "o_proj", "qkv": "qkv_proj"}
FP8_MAX = 448.0


@torch.no_grad()
def quantize_rows_reference(x):
    """Setup/reference implementation, including a well-defined zero row."""
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("quantization expects a nonempty matrix")
    if not torch.isfinite(x).all():
        raise ValueError("quantization input must be finite")
    values = x.float()
    scales = (values.abs().amax(dim=1) / FP8_MAX).clamp_min(1e-12)
    quantized = (values / scales[:, None]).clamp(-FP8_MAX, FP8_MAX)
    return quantized.to(torch.float8_e4m3fn).contiguous(), scales.contiguous()


class FP8Linear(nn.Module):
    """No resident FP16 weight copy inside the experimental replacement."""
    def __init__(self, linear):
        super().__init__()
        if not isinstance(linear, nn.Linear) or linear.weight.dtype != torch.float16:
            raise ValueError("FP8 replacement requires an FP16 nn.Linear")
        q, scales = quantize_rows_reference(linear.weight.detach())
        self.register_buffer("weight_fp8", q)
        self.register_buffer("weight_scales", scales)
        self.register_buffer("bias", None if linear.bias is None else linear.bias.detach().clone())
        self.in_features, self.out_features = linear.in_features, linear.out_features

    def forward(self, x):
        from experiments.quant.kernels.projections import fp8_linear
        return fp8_linear(x, self.weight_fp8, self.weight_scales, self.bias)


def parse_groups(value):
    groups = tuple(value.split(","))
    if not groups or any(g not in GROUPS for g in groups) or len(set(groups)) != len(groups):
        raise ValueError(f"groups must be unique names from {tuple(GROUPS)}")
    return groups


@contextmanager
def replace_projections(model, groups):
    """Prepare all replacements before mutation; always restore module identity."""
    if not groups or len(set(groups)) != len(groups) or any(g not in GROUPS for g in groups):
        raise ValueError("invalid projection groups")
    replacements = []
    for layer in model.layers:
        for group in groups:
            name = GROUPS[group]
            original = getattr(layer, name)
            replacements.append((layer, name, original, FP8Linear(original)))
    try:
        for layer, name, _, replacement in replacements:
            setattr(layer, name, replacement)
        yield
    finally:
        for layer, name, original, _ in replacements:
            setattr(layer, name, original)


@contextmanager
def strict_accumulation():
    """Control PyTorch GEMMs; Triton FP8 has its own explicit precision setting."""
    backend = torch.backends.cuda.matmul
    names = ["allow_fp16_reduced_precision_reduction", "allow_tf32"]
    if hasattr(backend, "allow_fp16_accumulation"):
        names.append("allow_fp16_accumulation")
    old = {name: getattr(backend, name) for name in names}
    try:
        for name in names:
            setattr(backend, name, False)
        yield {name: getattr(backend, name) for name in names}
    finally:
        for name, value in old.items():
            setattr(backend, name, value)
