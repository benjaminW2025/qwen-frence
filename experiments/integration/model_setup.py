"""Model-only setup for synthetic-token C++ integration experiments."""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from types import SimpleNamespace
import warnings
from pathlib import Path


def prepare_hub_transfer():
    """Avoid a broken optional download path before importing Hugging Face.

    Older huggingface_hub releases error if the fast-transfer flag is enabled
    without the optional ``hf_transfer`` package. The model identity and weights
    do not change when the standard downloader is used instead.
    """
    enabled = os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "").lower() in ("1", "on", "yes", "true")
    if not enabled or importlib.util.find_spec("hf_transfer") is not None:
        return "unchanged"
    if "huggingface_hub.constants" in sys.modules:
        raise RuntimeError(
            "HF_HUB_ENABLE_HF_TRANSFER=1 but hf_transfer is not installed, and "
            "huggingface_hub was already imported. Restart with "
            "HF_HUB_ENABLE_HF_TRANSFER=0 or install hf_transfer."
        )
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    warnings.warn(
        "hf_transfer is unavailable; using the standard Hugging Face download path "
        "for this synthetic-token experiment",
        RuntimeWarning,
        stacklevel=2,
    )
    return "standard_fallback_missing_hf_transfer"


def load_model_only(model_id, device, dtype, *, hub_transfer=None):
    """Load the same Qwen weights as PagedEngine without an unused tokenizer."""
    if hub_transfer is None:
        hub_transfer = prepare_hub_transfer()
    import torch
    from naive_forward import Qwen2Config
    from weight_loader import QwenWeightLoader

    cfg = Qwen2Config(use_custom_kernels=True)
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    started = time.perf_counter()
    model = QwenWeightLoader(cfg).load_pretrained(model_id, device, torch_dtype)
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    return SimpleNamespace(cfg=cfg, model=model, device=device), time.perf_counter() - started, hub_transfer


def check_startup(device):
    """Fail before weight loading/capture if core runtime dependencies are missing."""
    hub_transfer = prepare_hub_transfer()
    import torch

    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("a CUDA device is required for this integration experiment")
    if importlib.util.find_spec("triton") is None:
        raise RuntimeError("Triton is not installed; install the experiment's CUDA dependencies")
    try:
        import triton  # noqa: F401 - fail before loading weights if the install is broken
    except ImportError as error:
        raise RuntimeError("Triton cannot be imported in this environment") from error
    try:
        import inference_engine_cpp as cpp
    except ImportError as error:
        raise RuntimeError("C++ scheduler extension is missing; run make cpp-scheduler-build") from error
    root = Path(__file__).resolve().parents[2]
    extension = Path(cpp.__file__)
    sources = (list((root / "engine/cpp/src").glob("*.cpp"))
               + list((root / "engine/cpp/include").glob("*.hpp")))
    if any(source.stat().st_mtime_ns > extension.stat().st_mtime_ns for source in sources):
        raise RuntimeError("C++ scheduler extension is older than source; run make cpp-scheduler-build")
    return {"hub_transfer": hub_transfer, "cuda_available": True,
            "triton_available": True, "cpp_extension": str(extension),
            "note": "No model files downloaded and no model/graph GPU memory allocated."}
