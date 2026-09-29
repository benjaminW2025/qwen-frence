"""CPU contracts for the eight-cell local prefill-budget sweep."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from experiments.integration import benchmark_current_8_vs_vllm as benchmark
from experiments.integration import sweep_prefill_budget_8 as sweep

ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / "experiments/results/full-checkpoint-20260916T033540Z"


class BudgetSweepContract(unittest.TestCase):
    def test_budgets_past_a_cohort_are_measured_once(self):
        shapes = [row for row in sweep.FACTORIAL_SHAPES
                  if row["id"] in ("fixed-b8-l256-o128", "fixed-b64-l2048-o128")]
        runs = sweep.plan_runs(shapes, [1024, 2048, 4096, 8192])
        short = {budget: equivalent for shape, budget, equivalent in runs if shape == "fixed-b8-l256-o128"}
        long = {budget: equivalent for shape, budget, equivalent in runs if shape == "fixed-b64-l2048-o128"}
        # The B8 x 256 cohort is 2048 tokens: 4096 and 8192 schedule exactly like 2048.
        self.assertEqual(short, {1024: None, 2048: None, 4096: 2048, 8192: 2048})
        self.assertEqual(long, {1024: None, 2048: None, 4096: None, 8192: None})

    def test_frozen_budget_keeps_recorded_flags_and_others_differ(self):
        self.assertEqual(benchmark.engine_flags("fa3"), benchmark.engine_flags("fa3", 2048))
        self.assertNotIn("prefill_token_budget", benchmark.engine_flags("fa3"))
        self.assertEqual(benchmark.engine_flags("fa3", 4096)["prefill_token_budget"], 4096)

    def test_variant_flags_are_recorded_and_reach_adapter_options(self):
        self.assertNotIn("prefill_graph_pool", benchmark.engine_flags("fa3"))
        self.assertEqual(benchmark.engine_flags("fa3", 16384, "shared")["prefill_graph_pool"], "shared")
        flags = benchmark.engine_flags("fa3", 16384, "shared", "all", True)
        self.assertEqual((flags["gemm_epilogues"], flags["prefill_boundary_buffer_reuse"]), ("all", True))
        for key in ("gemm_epilogues", "prefill_boundary_buffer_reuse"):
            self.assertNotIn(key, benchmark.engine_flags("fa3"))
        case = dict(max_running=8, lengths=[2048] * 8, outputs=[128] * 8)
        args = SimpleNamespace(attention="fa3", prefill_budget=16384, prefill_graph_pool="shared",
                               gemm_epilogues="prefill", boundary_buffers=True)
        options = benchmark.adapter_options(case, [16384], "fa3", benchmark.variant_adapter_options(args))
        self.assertTrue(options["enable_prefill_shared_graph_pool"])
        self.assertTrue(options["enable_prefill_boundary_buffer_reuse"])
        self.assertTrue(options["enable_prefill_fused_gemm_epilogues"])
        self.assertFalse(options["enable_decode_fused_gemm_epilogues"])
        self.assertEqual(benchmark.variant_flags(args), flags | {"gemm_epilogues": "prefill"})
        self.assertNotIn("enable_prefill_shared_graph_pool", benchmark.adapter_options(case, [2048], "fa3"))

    def test_variant_is_forwarded_to_every_child_harness(self):
        args = SimpleNamespace(attention="fa3", suite_dir=SUITE, output_dir=Path("out"),
                               model="m", device="cuda:0", seed=1, warmups=1, repetitions=3,
                               reuse_vllm_from=None, resume_commit=None, vllm_budget="default",
                               prefill_budget=8192, prefill_graph_pool="shared",
                               gemm_epilogues="all", boundary_buffers=True)
        shape = benchmark.SHAPES[0]
        for command in (benchmark.forwarded(args, "run-local", shape),
                        benchmark.mixed_forwarded(args, shape),
                        benchmark.phase_forwarded(args, shape)):
            joined = " ".join(command)
            for fragment in ("--prefill-budget 8192", "--prefill-graph-pool shared",
                             "--gemm-epilogues all", "--boundary-buffers", "--vllm-budget default"):
                self.assertIn(fragment, joined)

    def test_variant_reaching_the_adapter_is_checked(self):
        args = SimpleNamespace(attention="fa3", prefill_budget=2048, prefill_graph_pool="private",
                               gemm_epilogues="decode", boundary_buffers=False)
        prefill = SimpleNamespace(graph_pool=None, enable_boundary_buffer_reuse=False,
                                  enable_fused_gemm_epilogues=False)
        decoder = SimpleNamespace(enable_fused_gemm_epilogues=True)
        adapter = SimpleNamespace(piecewise_prefill=prefill,
                                  graph_decoder=SimpleNamespace(decoders={8: decoder}))
        benchmark.check_variant_reached(adapter, args, "cell")
        decoder.enable_fused_gemm_epilogues = False
        with self.assertRaisesRegex(AssertionError, "did not reach the model"):
            benchmark.check_variant_reached(adapter, args, "cell")

    def test_sweep_refuses_large_budgets_without_the_shared_pool(self):
        argv = ["sweep", "--output-dir", "out", "--prefill-graph-pool", "private"]
        with mock.patch.object(sys, "argv", argv), mock.patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                sweep.main()

    def test_large_budget_needs_the_shared_pool_in_every_action(self):
        for action in ("run-local", "run-table", "run-cell"):
            argv = ["benchmark", action, "--suite-dir", str(SUITE), "--output-dir", "out",
                    "--shape-id", "fixed-b8-l2048-o128", "--prefill-budget", "16384"]
            with mock.patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(ValueError, "shared"):
                    benchmark.main()

    def test_planned_work_must_match_the_dry_schedule_step_for_step(self):
        case = {"id": "cell", "max_running": 2}
        schedule = [("prefill", [(False, 8, 2, 4)]), ("decode", [(True, 2, 2, 1)])]
        result = {"steps": [{"kind": kind, "calls": [list(call) for call in calls]} for kind, calls in schedule]}
        work = benchmark.verify_planned_work(result, {"schedule": schedule}, case)
        self.assertEqual(work, {"max_actual_decode_batch": 2, "max_packed_prefill_tokens": 8,
                                "pure_full_decode_steps": 1})
        result["steps"][1]["calls"][0][1] = 1
        with self.assertRaisesRegex(AssertionError, "departs from the CPU dry schedule at step 1"):
            benchmark.verify_planned_work(result, {"schedule": schedule}, case)

    def test_real_scheduler_plans_a_larger_budget(self):
        args = SimpleNamespace(suite_dir=SUITE, seed=20260914, prefill_budget=4096)
        case, requests, *_ = benchmark.input_contract(args, "fixed-b8-l2048-o128")
        plan = benchmark.dispatch_plan(case, requests, args.seed, with_schedule=True)
        self.assertEqual(plan["prefill_buckets"], [4096, 4102])
        self.assertEqual(max(call[1] for kind, calls in plan["schedule"] for call in calls if not call[0]), 4096)
        self.assertNotIn("schedule", benchmark.dispatch_plan(case, requests, args.seed))

    def test_summary_picks_best_budgets_and_folds_equivalent_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(output_dir=root, budgets=[1024, 2048, 4096], attention="fa3",
                                   prefill_graph_pool="shared",
                                   shape_ids=["fixed-b8-l256-o128", "fixed-b64-l2048-o128"])
            throughput = {("fixed-b8-l256-o128", 1024): 900, ("fixed-b8-l256-o128", 2048): 1000,
                          ("fixed-b64-l2048-o128", 1024): 5000, ("fixed-b64-l2048-o128", 2048): 6000,
                          ("fixed-b64-l2048-o128", 4096): 7200}
            for (shape, budget), value in throughput.items():
                path = sweep.budget_dir(root, budget) / shape / "local.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({"median_output_tokens_per_s": value, "runs": [{"outputs": {"0": [1]}}]}))
            shapes = [row for row in sweep.FACTORIAL_SHAPES if row["id"] in args.shape_ids]
            with mock.patch("builtins.print"):
                summary = sweep.summarize(args, sweep.plan_runs(shapes, args.budgets))
        short = summary["cells"]["fixed-b8-l256-o128"]
        self.assertEqual(short["best_budget"], 2048)          # 4096 folds onto the 2048 run
        self.assertEqual(short["measured_as"], {4096: 2048})
        self.assertEqual(summary["cells"]["fixed-b64-l2048-o128"]["best_budget"], 4096)
        self.assertAlmostEqual(summary["geomean_speedup_vs_2048"][4096], (1.0 * 1.2) ** .5)
        self.assertEqual(summary["best_single_budget"], 4096)
        self.assertEqual(summary["status"], "complete")

    def test_summary_checks_frozen_budget_tokens_against_the_reference_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference"
            args = SimpleNamespace(output_dir=root / "sweep", budgets=[2048], attention="fa3",
                                   prefill_graph_pool="shared", reference_dir=reference,
                                   shape_ids=["fixed-b8-l256-o128", "fixed-b8-l256-o256"])
            for shape, sweep_tokens, reference_tokens in (("fixed-b8-l256-o128", [1, 2], [1, 2]),
                                                          ("fixed-b8-l256-o256", [1, 2], [1, 3])):
                for base, tokens in ((sweep.budget_dir(args.output_dir, 2048), sweep_tokens), (reference, reference_tokens)):
                    path = base / shape / "local.json"
                    path.parent.mkdir(parents=True)
                    path.write_text(json.dumps({"median_output_tokens_per_s": 1.0,
                                                "runs": [{"outputs": {"0": tokens}}]}))
            shapes = [row for row in sweep.FACTORIAL_SHAPES if row["id"] in args.shape_ids]
            with mock.patch("builtins.print"):
                summary = sweep.summarize(args, sweep.plan_runs(shapes, args.budgets))
        self.assertTrue(summary["cells"]["fixed-b8-l256-o128"]["outputs_equal_to_reference_2048"])
        self.assertFalse(summary["cells"]["fixed-b8-l256-o256"]["outputs_equal_to_reference_2048"])


if __name__ == "__main__":
    unittest.main()
