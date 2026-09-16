"""CPU-only contract checks for the integrated graph experiment."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "experiments/integration/benchmark_integrated_graph.py"
SPEC = importlib.util.spec_from_file_location("benchmark_integrated_graph", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
sys.path.insert(0, str(ROOT / "engine/cpp/build"))
try:
    import inference_engine_cpp as cpp
except ImportError:
    cpp = None


class IntegratedGraphDesignTests(unittest.TestCase):
    def test_default_plan_is_one_fixed_regime(self):
        args = MODULE.build_parser().parse_args([])
        case = MODULE.validate_args(args)
        self.assertEqual(case["id"], "fixed-b8-l256-o128")
        self.assertEqual(len(case["lengths"]), 8)
        self.assertEqual(args.expected_max_decode_batch, 8)
        self.assertEqual(args.expected_prefill_tokens, 2048)

    def test_rejects_mismatched_declared_targets(self):
        args = MODULE.build_parser().parse_args(["--expected-max-decode-batch", "9"])
        with self.assertRaisesRegex(ValueError, "max_running"):
            MODULE.validate_args(args)
        args = MODULE.build_parser().parse_args(["--prefill-buckets", "512"])
        with self.assertRaisesRegex(ValueError, "no configured bucket"):
            MODULE.validate_args(args)

    def test_actual_work_gate_checks_calls_not_case_name(self):
        result = {"steps": [{"kind": "prefill", "calls": [(False, 2048, 1, 512)]},
                            {"kind": "decode", "calls": [(True, 1, 1, 1)]}],
                  "decode_batch_histogram": {"1": 1}}
        with self.assertRaisesRegex(AssertionError, "actual max decode batch 1"):
            MODULE.verify_actual_work(result, 4, 2048)
        result["steps"][1]["calls"] = [(True, 4, 4, 1)]
        result["decode_batch_histogram"] = {"4": 1}
        self.assertEqual(MODULE.verify_actual_work(result, 4, 2048)
                         ["max_actual_decode_batch"], 4)

    def test_splitk_gate_requires_repeated_practical_gain(self):
        effects = {"piecewise_splitk_vs_piecewise": {
            "wall_ms": {"trial_speedups": [1.03, 1.04, 1.025]}}}
        self.assertEqual(MODULE.splitk_decision(effects)["choice"], "splitk_candidate")
        self.assertEqual(MODULE.splitk_decision(effects, executed=False)["choice"],
                         "inactive_short_context")
        effects["piecewise_splitk_vs_piecewise"]["wall_ms"]["trial_speedups"][1] = 1.01
        self.assertEqual(MODULE.splitk_decision(effects)["choice"], "retain_production")

    def test_paired_effect_uses_same_trial_and_sample_medians(self):
        measurements = {}
        for arm, wall in (("eager", 12.0), ("decode_graph", 10.0),
                          ("piecewise", 8.0), ("piecewise_splitk", 7.0)):
            measurements[arm] = [[{phase: wall for phase in MODULE.PHASES}
                                  for _ in range(2)] for _ in range(3)]
        _, effects = MODULE.paired_summary(measurements)
        self.assertAlmostEqual(effects["decode_graph_vs_eager"]["wall_ms"]
                               ["median_speedup"], 1.2)
        self.assertAlmostEqual(effects["piecewise_splitk_vs_piecewise"]["wall_ms"]
                               ["median_speedup"], 8 / 7)

    @unittest.skipIf(cpp is None, "build C++ extension first")
    def test_cpu_scheduler_dry_run_reaches_real_target_shapes(self):
        args = MODULE.build_parser().parse_args([])
        case = MODULE.validate_args(args)
        work = MODULE.verify_actual_work(MODULE.dry_schedule(torch, cpp, case, args.seed),
                                         args.expected_max_decode_batch,
                                         args.expected_prefill_tokens)
        self.assertEqual(work["pure_full_decode_steps_at_target_batch"], 127)
        self.assertEqual(work["prefill_token_counts"], [2048])

    @unittest.skipIf(cpp is None, "build C++ extension first")
    def test_all_fixed_rows_sustain_at_least_64_full_batch_steps(self):
        from fixed_regime import (CONTEXT_PROBES, FACTORIAL_SHAPES, FIXED_SHAPES,
                                  get_fixed_case)

        self.assertEqual(len(FACTORIAL_SHAPES), 8)
        self.assertEqual(len(CONTEXT_PROBES), 2)
        self.assertEqual(len(FIXED_SHAPES), 10)
        for shape in FIXED_SHAPES:
            case = get_fixed_case(shape["id"])
            work = MODULE.verify_actual_work(
                MODULE.dry_schedule(torch, cpp, case, 20260914),
                shape["batch"], 2048, 64)
            self.assertEqual(work["pure_full_decode_steps_at_target_batch"],
                             shape["expected_full_decode_steps"])


if __name__ == "__main__":
    unittest.main()
