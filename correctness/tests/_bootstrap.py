"""Shared import-path setup for correctness-runner control-plane tests."""

from pathlib import Path
import sys


CORRECTNESS_DIR = Path(__file__).resolve().parents[1]
value = str(CORRECTNESS_DIR)
if value not in sys.path:
    sys.path.insert(0, value)
