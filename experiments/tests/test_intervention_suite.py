"""CPU-only contracts for the unified intervention suite."""

import importlib.util
from pathlib import Path
import unittest


EXPERIMENTS = Path(__file__).resolve().parents[1]
RUNNER_PATH = EXPERIMENTS / "run_intervention_suite.py"
SPEC = importlib.util.spec_from_file_location("run_intervention_suite", RUNNER_PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class InterventionSuiteTests(unittest.TestCase):
    def test_runner_contains_independent_remaining_hypotheses(self):
        self.assertEqual(
            set(RUNNER.EXPERIMENTS),
            {
                "decode-causality", "batched-sampling", "swiglu",
                "metadata-staging", "rope-kv-fusion", "attention-overlap",
                "launch-profile",
            },
        )

    def test_dry_run_commands_have_isolated_output_directories(self):
        args = RUNNER.build_parser().parse_args([])
        run_dir = Path("/tmp/intervention-suite-test")
        for name in RUNNER.EXPERIMENTS:
            command = RUNNER.command_for(name, args, run_dir)
            self.assertIn("--output-dir", command)
            self.assertIn(str(run_dir / name), command)

    def test_decode_causality_matches_launch_settings(self):
        source = (EXPERIMENTS / "decode" / "benchmark_decode_kernel_causality.py").read_text()
        self.assertIn('"production-w4-s2"', source)
        self.assertIn('"candidate-h1-w4-s2"', source)
        self.assertIn('"candidate-h6-w4-s2"', source)
        self.assertIn("decode_attention_reference", source)

    def test_launch_bound_tests_cover_sampling_and_metadata(self):
        sampling = (EXPERIMENTS / "scheduler" / "benchmark_batched_sampling.py").read_text()
        metadata = (EXPERIMENTS / "scheduler" / "benchmark_metadata_staging.py").read_text()
        self.assertIn("logits.argmax(dim=-1).tolist()", sampling)
        self.assertIn("pinned-reused-device", metadata)
        self.assertIn("current-python-to-device", metadata)

    def test_missing_user_regimes_are_explicit_controls(self):
        atlas = (EXPERIMENTS / "profiling" / "run_regime_atlas.py").read_text()
        overlap = (EXPERIMENTS / "attention" / "benchmark_attention_overlap.py").read_text()
        self.assertIn('"decode_low_concurrency"', atlas)
        self.assertIn('"long_fresh_prefill"', atlas)
        self.assertIn('"decode-heavy-low"', overlap)
        self.assertIn('"decode-heavy-high"', overlap)
        self.assertIn('"balanced-resumed"', overlap)

    def test_architectural_candidates_have_correctness_controls(self):
        rope = (EXPERIMENTS / "memory" / "benchmark_rope_kv_fusion.py").read_text()
        overlap = (EXPERIMENTS / "attention" / "benchmark_attention_overlap.py").read_text()
        self.assertIn("torch.testing.assert_close", rope)
        self.assertIn("torch.testing.assert_close", overlap)
        self.assertIn("sequential", overlap)
        self.assertIn("overlapped", overlap)


if __name__ == "__main__":
    unittest.main()
