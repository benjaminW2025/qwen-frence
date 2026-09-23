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
        with self.assertRaisesRegex(ValueError, "start with control"):
            MODULE.validate_args(parser.parse_args([
                "--fusion-modes", "qkv", "control",
            ]))

    def test_fa3_fusion_plan_and_two_axis_comparison(self):
        args = MODULE.build_parser().parse_args([
            "--shape-id", "fixed-b8-l2048-o128",
            "--budgets", "2048", "8192",
            "--fusion-modes", "control", "qkv", "residual", "both",
            "--decode-attention-policy", "fa3",
        ])
        base = MODULE.validate_args(args)
        plan = MODULE.plan_payload(args, base, MODULE.resolve_requests(args, base))
        self.assertEqual(plan["fusion_modes"], list(MODULE.FUSION_MODES))
        self.assertEqual(plan["correctness_workloads_per_budget"], 5)
        self.assertEqual(plan["expected_prefill_calls"], {"2048": 8, "8192": 2})

        rows = [
            {"budget": budget, "fusion_mode": mode,
             "medians": {"wall_ms": wall, "prefill_plus_mixed_ms": wall / 2,
                         "output_tokens_per_s": 1000 / wall},
             "work": {"prefill_calls": calls}}
            for budget, calls, mode, wall in (
                (2048, 8, "control", 100),
                (2048, 8, "both", 90),
                (8192, 2, "control", 80),
                (8192, 2, "both", 70),
            )
        ]
        MODULE.aggregate(rows)
        self.assertAlmostEqual(rows[3]["relative_to_smallest_budget"]
                               ["end_to_end_speedup"], 90 / 70)
        self.assertAlmostEqual(rows[3]["relative_to_control_same_budget"]
                               ["end_to_end_speedup"], 80 / 70)

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
