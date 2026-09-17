"""CPU checks for the integrated packed-prefill budget sweep."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/integration"))
PATH = ROOT / "experiments/integration/benchmark_prefill_budget.py"
SPEC = importlib.util.spec_from_file_location("benchmark_prefill_budget", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PrefillBudgetBenchmarkTests(unittest.TestCase):
    def test_default_plan_halves_prefill_calls_per_budget_step(self):
        args = MODULE.build_parser().parse_args(["--plan"])
        base = MODULE.validate_args(args)
        requests = MODULE.resolve_requests(args, base)
        plan = MODULE.plan_payload(args, base, requests)
        self.assertEqual(plan["budgets"], [2048, 4096, 8192])
        self.assertEqual(plan["decode_attention_policy"], "splitk")
        self.assertEqual(plan["expected_prefill_calls"],
                         {"2048": 8, "4096": 4, "8192": 2})
        self.assertEqual(plan["cohort_prompt_tokens"], 16384)

    def test_rejects_tail_buckets_and_oversized_budgets(self):
        parser = MODULE.build_parser()
        with self.assertRaisesRegex(ValueError, "divide total prompt tokens"):
            MODULE.validate_args(parser.parse_args(["--budgets", "2048", "3000"]))
        with self.assertRaisesRegex(ValueError, "exceeds the cohort"):
            MODULE.validate_args(parser.parse_args(["--budgets", "32768"]))

    def test_call_summary_reports_realized_packing(self):
        result = {"steps": [
            {"calls": [(False, 4096, 2, 2048)]},
            {"calls": [(True, 1, 1, 1), (False, 4096, 2, 2048)]},
        ]}
        summary = MODULE.call_summary(result, 4096)
        self.assertEqual(summary["prefill_calls"], 2)
        self.assertEqual(summary["sequences_per_call"], [2, 2])
        self.assertEqual(summary["token_budget_utilization"], 1.0)


if __name__ == "__main__":
    unittest.main()
