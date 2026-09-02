"""Shared import-path setup for directly executable correctness checks."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
for path in (
    ROOT / "baseline",
    ROOT / "engine" / "kvcache",
    ROOT / "engine" / "model_runner",
    ROOT / "engine" / "graph",
    ROOT / "engine" / "scheduler",
):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)
