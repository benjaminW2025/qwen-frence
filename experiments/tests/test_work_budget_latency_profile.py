"""CPU-only contracts for work-budget iteration profiling."""

from types import SimpleNamespace
import unittest

import _bootstrap  # noqa: F401

from profile_work_budget_latency import _iteration_rows, parse_positive_float_list


class WorkBudgetLatencyProfileTests(unittest.TestCase):
    def test_positive_latency_thresholds(self):
        self.assertEqual(parse_positive_float_list("40,50.5"), [40.0, 50.5])
        for value in ("", "0", "-1"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_positive_float_list(value)

    def test_iteration_rows_preserve_alignment_and_lists(self):
        run = SimpleNamespace(metadata={
            "iteration_wall_ms": [12.5],
            "iteration_kinds": ["mixed"],
            "iteration_decode_token_counts": [2],
            "iteration_decode_context_lengths": [[128, 256]],
            "iteration_prefill_token_counts": [64],
            "iteration_prefill_attention_pairs": [2080],
            "iteration_prefill_prefix_lengths": [[0]],
            "iteration_token_counts": [66],
        })
        row = _iteration_rows(run, 4096, 1)[0]
        self.assertEqual(row["iteration_latency_ms"], 12.5)
        self.assertEqual(row["decode_context_lengths"], "[128,256]")
        self.assertEqual(row["decode_context_mean"], 192)
        self.assertEqual(row["prefill_attention_pairs"], 2080)

    def test_iteration_rows_reject_misalignment(self):
        run = SimpleNamespace(metadata={
            "iteration_wall_ms": [],
            "iteration_kinds": ["decode_only"],
            "iteration_decode_token_counts": [1],
            "iteration_decode_context_lengths": [[128]],
            "iteration_prefill_token_counts": [0],
            "iteration_prefill_attention_pairs": [0],
            "iteration_prefill_prefix_lengths": [[]],
            "iteration_token_counts": [1],
        })
        with self.assertRaisesRegex(AssertionError, "not aligned"):
            _iteration_rows(run, None, 0)


if __name__ == "__main__":
    unittest.main()
