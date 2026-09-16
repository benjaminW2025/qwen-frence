"""CPU-only contract checks for the integrated graph experiment."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest
import io
from contextlib import redirect_stdout
from unittest.mock import patch
from types import SimpleNamespace

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
    @unittest.skipIf(cpp is None, "build C++ extension first")
    def test_long_cell_runs_full_ablation_with_close_but_different_tokens(self):
        args = MODULE.build_parser().parse_args([
            "--case-id", "fixed-b8-l2048-o128", "--trials", "1", "--samples", "1",
            "--warmups", "0"])
        case = MODULE.validate_args(args)

        class Adapter:
            corrupt_splitk = False
            def __init__(self, candidate=False, pieces=False, splitk=False):
                self.decisions, self.step_calls, self.observer = {}, [], None
                self.candidate, self.pieces = candidate, pieces
                self.action = "H1-K22-S3" if splitk else "production"
                if pieces:
                    self.piecewise_prefill = SimpleNamespace(
                        shapes={2048: []}, buckets=[2048], captured_calls=0,
                        eager_calls=0, graph_replays=0)

            def __call__(self, ids, positions, slots, cu, context, table, max_query, decode):
                raw = (ids, positions, slots, cu, context, table, max_query, decode)
                self.step_calls.append((decode, ids.numel(), context.numel(), max_query))
                if decode:
                    self.decisions[self.action] = self.decisions.get(self.action, 0) + 1
                elif self.pieces:
                    self.piecewise_prefill.captured_calls += 1
                    self.piecewise_prefill.graph_replays += 29
                logits = torch.tensor([[1.002, 1.0] if self.candidate else [1.0, 1.001]])
                if decode and self.corrupt_splitk and self.action != "production":
                    logits = torch.tensor([[10., 1.]])
                logits = logits.repeat(context.numel(), 1)
                if self.observer is not None:
                    replacement = self.observer(raw, logits)
                    if replacement is not None:
                        return replacement
                return logits

        engine = SimpleNamespace(cfg=SimpleNamespace(vocab=16), model=None, device="cpu")
        pool = lambda *a: SimpleNamespace(k_pool=[torch.zeros(1)], v_pool=[torch.zeros(1)])
        with (patch.object(torch.cuda, "synchronize"),
              patch.object(MODULE, "allocate_pool", side_effect=pool),
              patch.object(MODULE, "ModelAdapter", side_effect=lambda *a: Adapter()),
              patch.object(MODULE, "GraphModelAdapter", side_effect=lambda *a, **k: Adapter(True)),
              patch.object(MODULE, "PiecewiseGraphModelAdapter", side_effect=lambda *a, **k:
                           Adapter(True, True, k.get("decode_attention_policy") == "splitk")),
              redirect_stdout(io.StringIO())):
            row = MODULE.run_case(torch, cpp, engine, case, args,
                                  MODULE.make_requests(case, args.seed, 16))
            Adapter.corrupt_splitk = True
            rejected_row = MODULE.run_case(torch, cpp, engine, case, args,
                                           MODULE.make_requests(case, args.seed, 16))
            args.splitk_only = True
            retry_row = MODULE.run_case(torch, cpp, engine, case, args,
                                       MODULE.make_requests(case, args.seed, 16))
        self.assertEqual(row["status"], "ok")
        self.assertTrue(row["splitk_executed"])
        self.assertNotEqual(row["output_ids_by_arm"]["eager"],
                            row["output_ids_by_arm"]["piecewise_splitk"])
        self.assertGreater(row["checks"]["piecewise_splitk"]
                           ["argmax_differences_on_reference_history"], 0)
        self.assertEqual(set(row["measurements"]), set(MODULE.ARMS))
        self.assertEqual(rejected_row["splitk_decision"]["choice"], "timed_with_numerical_warning")
        self.assertIn("piecewise_splitk", rejected_row["measurements"])
        self.assertIn("piecewise_splitk", rejected_row["output_ids_by_arm"])
        self.assertEqual(set(rejected_row["measurements"]), set(MODULE.ARMS))
        check = rejected_row["checks"]["piecewise_splitk"]
        self.assertFalse(check["numerical_validation_passed"])
        self.assertGreater(check["logits_outside_tolerance"], 0)
        self.assertAlmostEqual(check["max_logit_error"], 9.)
        self.assertEqual(set(retry_row["measurements"]), {"piecewise_splitk"})
        self.assertTrue(retry_row["splitk_only"])
        self.assertEqual(retry_row["effects"], {})
        self.assertEqual(retry_row["splitk_decision"]["choice"], "timed_with_numerical_warning")

    def test_teacher_forced_preflight_and_free_generation_use_distinct_histories(self):
        from python_control import PythonControl
        from benchmark_scheduler_decode import execute

        config = SimpleNamespace(max_batch_size=2, max_context_length=16,
                                 max_prefill_tokens_per_iter=4, block_size=16,
                                 eos_token_id=-1)
        requests = [{"id": i, "arrival": 0, "prompt": [1] * 4, "output": 4}
                    for i in range(2)]

        class Adapter:
            def __init__(self, candidate=False, observer=None):
                self.decisions, self.step_calls = {}, []
                self.candidate, self.observer = candidate, observer

            def __call__(self, ids, positions, slots, cu, context, table, max_query, decode):
                args = (ids, positions, slots, cu, context, table, max_query, decode)
                self.step_calls.append((decode, ids.numel(), context.numel(), max_query))
                logits = torch.tensor([[1.002, 1.0] if self.candidate else [1.0, 1.001]])
                logits = logits.repeat(context.numel(), 1)
                if self.observer is not None:
                    replacement = self.observer(args, logits)
                    if replacement is not None:
                        return replacement
                return logits

        with patch.object(torch.cuda, "synchronize"):
            reference = MODULE.TraceCheck(torch)
            expected = execute(torch, PythonControl(config, "cpu"), Adapter(observer=reference), requests)
            checker = MODULE.SameHistoryCheck(torch, Adapter(), reference.rows, "candidate")
            checked = execute(torch, PythonControl(config, "cpu"),
                              Adapter(True, checker), requests)
            checker.finish()
            natural = execute(torch, PythonControl(config, "cpu"), Adapter(True), requests)
        self.assertEqual(checked["outputs"], expected["outputs"])
        self.assertNotEqual(natural["outputs"], expected["outputs"])
        self.assertEqual(MODULE.schedule_of(natural), MODULE.schedule_of(expected))
        self.assertGreater(checker.argmax_differences, 0)

    def test_same_history_accepts_close_logits_with_different_argmax(self):
        args = (torch.tensor([1]), torch.tensor([3]), torch.tensor([3]),
                torch.tensor([], dtype=torch.int32), torch.tensor([4]),
                torch.tensor([[0]]), 1, True)
        expected = torch.tensor([[1.0, 1.001]])
        reference = MODULE.TraceCheck(torch)
        reference(args, expected)
        checker = MODULE.SameHistoryCheck(torch, lambda *a: expected,
                                          reference.rows, "splitk")
        actual = torch.tensor([[1.002, 1.0]])
        returned = checker(args, actual)
        self.assertIs(returned, expected)
        self.assertEqual(checker.argmax_differences, 1)
        checker.finish()

    def test_explicit_tolerance_accepts_small_outlier_but_retains_strict_default(self):
        args = (torch.tensor([1]), torch.tensor([3]), torch.tensor([3]),
                torch.tensor([], dtype=torch.int32), torch.tensor([4]),
                torch.tensor([[0]]), 1, True)
        expected, actual = torch.tensor([[0., 2.]]), torch.tensor([[.064, 2.]])
        reference = MODULE.TraceCheck(torch)
        reference(args, expected)
        strict = MODULE.SameHistoryCheck(torch, lambda *a: expected, reference.rows, "splitk")
        with self.assertRaises(MODULE.NumericalMismatch):
            strict(args, actual)
        relaxed = MODULE.SameHistoryCheck(torch, lambda *a: expected, reference.rows, "splitk", atol=.075)
        relaxed(args, actual)
        self.assertAlmostEqual(relaxed.max_logit_error, .064, places=6)
        diagnostic = MODULE.SameHistoryCheck(torch, lambda *a: expected, reference.rows,
                                             "splitk", report_only=True)
        diagnostic(args, actual)
        self.assertEqual(diagnostic.logits_outside_tolerance, 1)
        self.assertEqual(diagnostic.logits_compared, 2)

    def test_same_history_checks_repeated_shapes_and_rejects_corruption(self):
        args = (torch.tensor([1]), torch.tensor([3]), torch.tensor([3]),
                torch.tensor([], dtype=torch.int32), torch.tensor([4]),
                torch.tensor([[0]]), 1, True)
        expected = torch.tensor([[1.0, 2.0]])
        reference = MODULE.TraceCheck(torch)
        reference(args, expected)
        reference(args, expected)
        checker = MODULE.SameHistoryCheck(torch, lambda *a: expected,
                                          reference.rows, "piecewise")
        checker(args, expected.clone())
        with self.assertRaisesRegex(AssertionError, "piecewise: same-history"):
            checker(args, torch.tensor([[10.0, 2.0]]))

    def test_metadata_diagnostic_identifies_field(self):
        args = (torch.tensor([1]), torch.tensor([3]), torch.tensor([3]),
                torch.tensor([], dtype=torch.int32), torch.tensor([4]),
                torch.tensor([[0]]), 1, True)
        logits = torch.tensor([[1.0, 2.0]])
        reference = MODULE.TraceCheck(torch)
        reference(args, logits)
        checker = MODULE.TraceCheck(torch, reference.rows)
        changed = (torch.tensor([2]), *args[1:])
        with self.assertRaisesRegex(AssertionError, "fields=\\['input_ids'\\]"):
            checker(changed, logits)

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
