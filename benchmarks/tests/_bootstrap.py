"""Shared import-path setup for benchmark control-plane tests."""

from pathlib import Path
import sys


BENCHMARKS_DIR = Path(__file__).resolve().parents[1]
ROOT = BENCHMARKS_DIR.parent
for path in (
    BENCHMARKS_DIR,
    BENCHMARKS_DIR / "prefill",
    BENCHMARKS_DIR / "scheduler",
    ROOT / "engine" / "scheduler",
):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)
