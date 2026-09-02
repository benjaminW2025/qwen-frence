"""CPU-only controls for token and prefix-aware attention-work sweeps."""

import unittest

import _bootstrap  # noqa: F401

from benchmark_token_budget import parse_optional_int_list


class TokenBudgetBenchmarkTests(unittest.TestCase):
    def test_optional_attention_pair_budgets(self):
        self.assertEqual(
            parse_optional_int_list("none,1048576,4194304"),
            [None, 1_048_576, 4_194_304],
        )

    def test_attention_pair_budgets_reject_nonpositive_values(self):
        for value in ("", "0", "-1"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_optional_int_list(value)


if __name__ == "__main__":
    unittest.main()
