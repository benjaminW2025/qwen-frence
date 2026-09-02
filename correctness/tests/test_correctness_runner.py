"""CPU-only contracts for correctness suite selection."""

import unittest

import _bootstrap  # noqa: F401

from run_correctness import CHECKS, PRIMARY_CHECKS, parse_checks


class CorrectnessRunnerTests(unittest.TestCase):
    def test_all_selects_every_check_in_stable_order(self):
        self.assertEqual(parse_checks("all"), list(CHECKS))

    def test_baseline_alias_selects_primary_nonlegacy_checks(self):
        self.assertEqual(parse_checks("baseline"), PRIMARY_CHECKS)
        self.assertNotIn("cuda-graph", PRIMARY_CHECKS)
        self.assertNotIn("bucketed-graphs", PRIMARY_CHECKS)

    def test_unknown_check_is_rejected(self):
        with self.assertRaisesRegex(Exception, "unknown check"):
            parse_checks("not-a-check")


if __name__ == "__main__":
    unittest.main()
