"""Lazy adapters from the inference engine to the standalone Triton kernels."""

from __future__ import annotations

from functools import lru_cache
import importlib.util
from pathlib import Path


_KERNEL_DIR = Path(__file__).resolve().parents[1] / "custom_kernels"


@lru_cache(maxsize=None)
def _load(name: str):
    path = _KERNEL_DIR / f"{name}.py"
    if not path.is_file():
        raise RuntimeError(f"custom kernel source not found: {path}")
    spec = importlib.util.spec_from_file_location(f"_inference_engine_{name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load custom kernel: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rms_norm(x, weight, epsilon=1e-6):
    """Apply the Triton RMSNorm kernel to every row of an N-D tensor."""
    shape = x.shape
    rows = x.reshape(-1, shape[-1])
    if not rows.is_contiguous():
        rows = rows.contiguous()
    return _load("fused_rms").rms_norm(rows, weight, epsilon).view(shape)


def rope(x, positions, theta):
    """Apply Qwen rotate-half RoPE using explicit per-sequence positions."""
    if positions.ndim == 1:
        positions = positions[:, None]
    return _load("rope").rope(x, positions=positions, base=theta)


def packed_prefill_attention(*args, **kwargs):
    """Run packed variable-length causal attention."""
    return _load("packed_prefill_attention").packed_prefill_attention_triton(
        *args, **kwargs
    )


def packed_paged_prefill_attention(*args, **kwargs):
    """Run packed query-chunk attention over paged K/V state."""
    return _load(
        "packed_paged_prefill_attention"
    ).packed_paged_prefill_attention_triton(*args, **kwargs)


def select_paged_decode_candidate_config(*args, **kwargs):
    """Select the measured paged-decode candidate without inspecting device data."""
    return _load("paged_decode_candidate").select_paged_decode_candidate_config(
        *args, **kwargs
    )


def paged_decode_attention_candidate(*args, **kwargs):
    """Run the tuned paged-decode candidate."""
    return _load("paged_decode_candidate").paged_decode_attention_candidate(
        *args, **kwargs
    )


def swiglu(*args, **kwargs):
    """Fuse the SiLU activation and gated elementwise product."""
    return _load("swiglu").swiglu(*args, **kwargs)


def rope_kv_write(*args, **kwargs):
    """Fuse packed Q/K RoPE with paged K/V placement."""
    return _load("rope_kv_write").rope_kv_write(*args, **kwargs)
