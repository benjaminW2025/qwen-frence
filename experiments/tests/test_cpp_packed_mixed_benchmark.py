"""CPU-only contracts for the one-callback mixed benchmark."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "experiments/integration/benchmark_cpp_packed_mixed.py"
SPEC = importlib.util.spec_from_file_location("benchmark_cpp_packed_mixed", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CppPackedMixedBenchmarkTests(unittest.TestCase):
    def test_work_signature_compares_logical_cohorts_and_completion(self):
        run = {"steps": [
            {"kind": "prefill", "calls": [(False, 512, 2, 256)], "completed": 0},
            {"kind": "mixed", "calls": [(True, 2, 2, 1),
                                         (False, 256, 1, 256)], "completed": 0},
        ]}
        self.assertEqual(MODULE.work_signature(run)[1],
                         ("mixed", [(True, 2, 2, 1), (False, 256, 1, 256)], 0))


if __name__ == "__main__":
    unittest.main()
