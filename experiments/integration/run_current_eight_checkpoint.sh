#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^(plan|check|run-table|analyze)$ ]]; then
  echo "usage: bash experiments/integration/run_current_eight_checkpoint.sh {plan|check|run-table|analyze}" >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
cd "$repo_root"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  python_bin="$PYTHON_BIN"
elif [[ -x /root/local-engine-env/bin/python ]]; then
  python_bin=/root/local-engine-env/bin/python
else
  python_bin=python3
fi

exec "$python_bin" experiments/integration/benchmark_current_8_vs_vllm.py "$1" \
  --suite-dir experiments/results/full-checkpoint-20260916T033540Z \
  --output-dir "${OUTPUT_DIR:-experiments/results/independent-vllm-0.30.0-v1}" \
  --vllm-python "${VLLM_PYTHON_BIN:-/root/vllm-current-env/bin/python}"
