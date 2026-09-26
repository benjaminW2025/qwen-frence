"""Version contract for new reference measurements (historical artifacts are immutable)."""

from packaging.version import Version

VLLM_VERSION = "0.30.0"


def require_vllm_version(version):
    # Official CUDA-specific wheels carry a local version such as +cu129.
    if version is None or Version(version).base_version != VLLM_VERSION:
        raise ValueError(f"requires vLLM {VLLM_VERSION}; got {version!r}")
    return version
