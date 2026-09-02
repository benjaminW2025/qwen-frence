"""Shared import-path setup for optimization experiment control-plane tests."""

from pathlib import Path
import sys


EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENTS_DIR.parent
BENCHMARKS_DIR = ROOT / "benchmarks"
for path in (
    BENCHMARKS_DIR,
    EXPERIMENTS_DIR / "prefill",
    EXPERIMENTS_DIR / "scheduler",
    ROOT / "engine" / "scheduler",
):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)
