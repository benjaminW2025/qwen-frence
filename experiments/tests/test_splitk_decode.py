"""CPU-only controls for the Split-K decode benchmark protocol."""

from pathlib import Path
import sys
import unittest

import _bootstrap  # noqa: F401


DECODE_DIR = Path(__file__).resolve().parents[1] / "decode"
if str(DECODE_DIR) not in sys.path:
    sys.path.insert(0, str(DECODE_DIR))

from benchmark_splitk_decode import (  # noqa: E402
    _attention_operation,
    materialize_k_candidates,
    parse_auto_k_neighbor_factors,
)


class SplitKDecodeSweepTests(unittest.TestCase):
    def test_auto_neighbor_factors_are_optional_and_positive(self):
        self.assertEqual(parse_auto_k_neighbor_factors(""), [])
        self.assertEqual(
            parse_auto_k_neighbor_factors("0.5, 1.25"), [0.5, 1.25],
        )
        for value in ("0", "-1", "0.5,0"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_auto_k_neighbor_factors(value)

    def test_neighbors_bracket_auto_without_duplicate_k_measurements(self):
        candidates = materialize_k_candidates(
            [1, 2, 4, 8, 16, "auto"], 24, 256,
            [0.5, 0.75, 1.25, 1.5],
        )
        self.assertEqual(
            candidates,
            [
                ("1", 1, False), ("2", 2, False), ("4", 4, False),
                ("8", 8, False), ("16", 16, False), ("auto(24)", 24, True),
                ("auto*0.5(12)", 12, False), ("auto*0.75(18)", 18, False),
                ("auto*1.25(30)", 30, False), ("auto*1.5(36)", 36, False),
            ],
        )

    def test_k_one_uses_direct_production_attention(self):
        calls = []

        def production(*args):
            calls.append(("production", args))

        def splitk(*args, **kwargs):
            calls.append(("splitk", args, kwargs))

        _attention_operation(production, splitk, ("q", "k"), 1)()
        self.assertEqual(calls, [("production", ("q", "k"))])


if __name__ == "__main__":
    unittest.main()
