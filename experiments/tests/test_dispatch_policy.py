"""CPU-only controls for constrained regime-dispatch fitting."""

import importlib.util
from pathlib import Path
import sys
import unittest


EXPERIMENTS = Path(__file__).resolve().parents[1]
FITTER_PATH = EXPERIMENTS / "dispatch" / "fit_dispatch_policies.py"
RUNNER_PATH = EXPERIMENTS / "run_dispatch_policy_experiment.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


FITTER = load_module("fit_dispatch_policies", FITTER_PATH)
RUNNER = load_module("run_dispatch_policy_experiment", RUNNER_PATH)


class DispatchPolicyTests(unittest.TestCase):
    def shape(self, value, baseline, candidate):
        return FITTER.Shape(
            component="swiglu",
            key=(value,),
            features={"rows": value},
            latencies={"baseline": baseline, "candidate": candidate},
            baseline_action="baseline",
        )

    def test_one_split_recovers_a_clean_crossover(self):
        shapes = [
            self.shape(1, 1.0, 1.2),
            self.shape(2, 1.0, 1.1),
            self.shape(3, 1.0, 0.8),
            self.shape(4, 1.0, 0.7),
        ]
        tree = FITTER.fit_tree(
            shapes, ("rows",), max_depth=1, min_leaf_shapes=1,
            min_split_improvement=0.0, max_training_regression=0.02,
        )
        self.assertEqual(tree["kind"], "split")
        self.assertEqual(tree["threshold"], 2.5)
        self.assertEqual(FITTER.select_action(tree, {"rows": 2}), "baseline")
        self.assertEqual(FITTER.select_action(tree, {"rows": 3}), "candidate")

    def test_regression_guard_keeps_the_baseline_without_a_split(self):
        shapes = [self.shape(1, 1.0, 1.03), self.shape(2, 1.0, 0.5)]
        tree = FITTER.fit_tree(
            shapes, ("rows",), max_depth=0, min_leaf_shapes=1,
            min_split_improvement=0.0, max_training_regression=0.02,
        )
        self.assertEqual(tree["action"], "baseline")

    def test_evaluation_reports_oracle_regret(self):
        shapes = [self.shape(1, 1.0, 1.2), self.shape(3, 1.0, 0.8)]
        tree = {
            "kind": "split", "feature": "rows", "threshold": 2,
            "left": {"kind": "leaf", "action": "baseline"},
            "right": {"kind": "leaf", "action": "candidate"},
        }
        rows = FITTER.evaluate(tree, shapes, "holdout", winner_margin=1.02)
        summary = FITTER.summarize(rows)
        self.assertAlmostEqual(summary["aggregate_oracle_regret"], 1.0)
        self.assertAlmostEqual(summary["oracle_action_agreement"], 1.0)

    def test_dense_runner_covers_each_observed_crossover(self):
        self.assertIn("1024,2048,4096,8192", RUNNER.DENSE_AXES["decode"]["--context-lengths"])
        self.assertIn("1536,1792,2048,2560", RUNNER.DENSE_AXES["swiglu"]["--rows"])
        self.assertIn("512,768,1024,1536", RUNNER.DENSE_AXES["rope-kv"]["--token-counts"])
        self.assertIn("384,512,768,1024", RUNNER.DENSE_AXES["paged-prefill"]["--query-lengths"])

    def test_dry_run_has_isolated_outputs_and_a_fit_stage(self):
        args = RUNNER.build_parser().parse_args([])
        run_dir = Path("/tmp/dispatch-policy-suite-test")
        for name in RUNNER.SWEEPS:
            command = RUNNER.sweep_command(name, args, run_dir)
            self.assertIn(str(run_dir / name), command)
        self.assertTrue(RUNNER.FITTER.name.endswith("fit_dispatch_policies.py"))


if __name__ == "__main__":
    unittest.main()
