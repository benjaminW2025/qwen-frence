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
elif [[ -x /root/vllm-bench-env/bin/python ]]; then
  python_bin=/root/vllm-bench-env/bin/python
elif [[ -x /workspace/vllm-bench-env/bin/python ]]; then
  python_bin=/workspace/vllm-bench-env/bin/python
else
  python_bin=python3
fi

exec "$python_bin" experiments/integration/benchmark_current_8_vs_vllm.py "$1" \
  --suite-dir experiments/results/full-checkpoint-20260916T033540Z \
  --output-dir experiments/results/current-eight-vs-vllm-v2 \
  --reuse-vllm-from experiments/results/current-eight-vs-vllm-v1 \
  --resume-commit f318493 \
  --resume-commit 2627dcc \
  --resume-commit 0c2c698
