#!/usr/bin/env bash
# Fresh-pod setup for the FA3-mode eight-cell session (run_integrated_8.py).
# Idempotent: rerun after any pod restart; finished steps are skipped.
#
#   bash scripts/pod_setup.sh            # everything
#   PERSIST=/workspace bash scripts/...  # where envs, caches and CUTLASS live
#
# Everything downloadable goes under $PERSIST. If $PERSIST is a network volume
# (check: `df -h / /workspace` shows different filesystems) restarts reuse it;
# otherwise this script simply reinstalls. The CUDA 13 toolkit goes to /usr/local
# (container disk) and is reinstalled when missing.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PERSIST="${PERSIST:-/workspace}"
ENV="$PERSIST/vllm-env"
CUTLASS="$PERSIST/cutlass"
export HF_HOME="$PERSIST/hf"
export PIP_CACHE_DIR="$PERSIST/pip-cache"
PY="$ENV/bin/python"

step() { printf '\n==> %s\n' "$*"; }

step "filesystems (a durable PERSIST must differ from /)"
df -h / "$PERSIST" || true

step "system packages"
if ! command -v python3.12 >/dev/null || ! command -v git >/dev/null; then
  apt-get update -qq
  apt-get install -y -qq python3.12 python3.12-venv python3.12-dev git wget >/dev/null
fi

step "vLLM 0.30.0 env at $ENV (both engines run here in FA3 mode)"
if ! "$PY" -c "import vllm, sys; sys.exit(vllm.__version__ != '0.30.0')" 2>/dev/null; then
  python3.12 -m venv "$ENV"
  "$PY" -m pip install -q --upgrade pip
  "$PY" -m pip install -q -r "$REPO/benchmarks/requirements-vllm-current.txt"
fi
"$PY" -m pip install -q ninja "pybind11>=2.13" "setuptools>=77,<81" wheel

TORCH_CUDA="$("$PY" -c 'import torch; print(torch.version.cuda)')"
MAJOR="${TORCH_CUDA%%.*}"
MINOR="$(echo "$TORCH_CUDA" | cut -d. -f2)"
step "CUDA toolkit matching torch (cu$TORCH_CUDA) for the CUTLASS extension"
CUDA_HOME="/usr/local/cuda-$MAJOR.$MINOR"
if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
  . /etc/os-release
  DISTRO="ubuntu${VERSION_ID//./}"
  if ! apt-cache policy 2>/dev/null | grep -q developer.download.nvidia.com; then
    wget -q "https://developer.download.nvidia.com/compute/cuda/repos/$DISTRO/x86_64/cuda-keyring_1.1-1_all.deb" -O /tmp/cuda-keyring.deb
    dpkg -i /tmp/cuda-keyring.deb >/dev/null
  fi
  apt-get update -qq
  apt-get install -y -qq "cuda-toolkit-$MAJOR-$MINOR" >/dev/null
fi
export CUDA_HOME PATH="$CUDA_HOME/bin:$PATH"
nvcc --version | tail -1

step "model snapshot into $HF_HOME"
(cd "$REPO" && "$PY" experiments/integration/benchmark_latest_vs_vllm.py check-model-cache >/dev/null 2>&1) || \
  (cd "$REPO" && "$PY" experiments/integration/benchmark_latest_vs_vllm.py stage-model-cache)

step "C++ scheduler extension against this env's torch"
(cd "$REPO/engine/cpp" && "$PY" setup.py -q build_ext --build-lib build --force)

step "CUTLASS fused-epilogue extension"
build_epilogues() {
  (cd "$REPO/custom_kernels/gemm_epilogue" && CUTLASS_PATH="$1" "$PY" setup.py -q build_ext --inplace --force)
}
if [ ! -d "$CUTLASS/v3.9.2" ]; then
  git clone -q --depth 1 -b v3.9.2 https://github.com/NVIDIA/cutlass "$CUTLASS/v3.9.2"
fi
if ! build_epilogues "$CUTLASS/v3.9.2"; then
  # 3.9.2 predates CUDA 13; 4.x keeps the same SM90 CollectiveBuilder API.
  echo "CUTLASS v3.9.2 failed under CUDA $TORCH_CUDA; retrying with v4.2.1"
  [ -d "$CUTLASS/v4.2.1" ] || git clone -q --depth 1 -b v4.2.1 https://github.com/NVIDIA/cutlass "$CUTLASS/v4.2.1"
  build_epilogues "$CUTLASS/v4.2.1"
fi

step "CPU contracts and the session plan"
cd "$REPO"
"$PY" -m unittest discover -s experiments/tests -p 'test_*.py' 2>&1 | tail -2
VLLM_USE_FLASHINFER_SAMPLER=0 "$PY" experiments/integration/run_integrated_8.py \
  --output-dir experiments/results/integrated-v1 --plan >/dev/null
cat > "$PERSIST/pod_env.sh" <<EOF
export HF_HOME="$HF_HOME" PIP_CACHE_DIR="$PIP_CACHE_DIR" CUDA_HOME="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:\$PATH" VLLM_USE_FLASHINFER_SAMPLER=0
source "$ENV/bin/activate"
EOF
step "ready: source $PERSIST/pod_env.sh"
