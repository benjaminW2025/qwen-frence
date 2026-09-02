"""CPU-only contracts for the final factorial regime scorecard."""

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import _bootstrap  # noqa: F401

from run_regime_scorecard import (
    COMPETITIVE_BACKENDS,
    REFERENCE_ANCHORS,
    REFERENCE_BACKENDS,
    REGIMES,
    competitive_command,
    _optional_result,
    parse_regimes,
    reference_command,
    validate_suite_path,
    vllm_command,
)


class RegimeScorecardTests(unittest.TestCase):
    def _args(self):
        return SimpleNamespace(
            model="model",
            device="cuda",
            dtype="float16",
            block_size=16,
            max_num_batched_tokens=8192,
            warmups=1,
            repetitions=3,
            seed=0,
        )

    def test_plan_is_complete_two_by_two_by_two_factorial(self):
        self.assertEqual(len(REGIMES), 8)
        self.assertEqual({case.concurrency for case in REGIMES}, {8, 64})
        self.assertEqual({case.prompt_length for case in REGIMES}, {256, 8192})
        self.assertEqual({case.output_length for case in REGIMES}, {32, 256})
        self.assertEqual(parse_regimes("all"), list(REGIMES))

    def test_competitive_command_preserves_static_ablation(self):
        command = competitive_command(self._args(), REGIMES[0], Path("case"))
        backend_value = command[command.index("--backends") + 1]
        self.assertEqual(backend_value, ",".join(COMPETITIVE_BACKENDS))
        self.assertIn("custom-kernels", backend_value)
        self.assertIn("regime-dispatched", backend_value)
        self.assertEqual(
            command[command.index("--max-running") + 1],
            str(REGIMES[0].concurrency),
        )

    def test_historical_references_use_sparse_single_sample_protocol(self):
        self.assertEqual(len(REFERENCE_ANCHORS), 4)
        command = reference_command(self._args(), REGIMES[0], Path("case"))
        self.assertEqual(
            command[command.index("--backends") + 1],
            ",".join(REFERENCE_BACKENDS),
        )
        self.assertEqual(command[command.index("--warmups") + 1], "0")
        self.assertEqual(command[command.index("--repetitions") + 1], "1")

    def test_competitive_command_can_merge_reference_result(self):
        command = competitive_command(
            self._args(), REGIMES[0], Path("case"), Path("reference.json")
        )
        self.assertEqual(
            command[command.index("--compare-with") + 1], "reference.json"
        )
        self.assertNotIn("--workload-out", command)

    def test_vllm_replays_the_exact_local_result(self):
        with tempfile.TemporaryDirectory() as directory:
            suite = Path(directory)
            result_dir = suite / REGIMES[0].name / "local"
            result_dir.mkdir(parents=True)
            local_result = result_dir / "result.json"
            local_result.write_text("{}")
            command = vllm_command(
                self._args(), REGIMES[0], suite, suite / REGIMES[0].name
            )
            self.assertEqual(command[command.index("--backends") + 1], "vllm")
            self.assertEqual(
                command[command.index("--compare-with") + 1], str(local_result)
            )
            self.assertEqual(
                command[command.index("--vllm-kv-cache-mode") + 1], "matched"
            )

    def test_optional_result_supports_resuming_completed_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            result_dir = Path(directory)
            self.assertIsNone(_optional_result(result_dir))
            result = result_dir / "result.json"
            result.write_text("{}")
            self.assertEqual(_optional_result(result_dir), result)

    def test_documentation_placeholder_cannot_become_a_result_directory(self):
        with self.assertRaisesRegex(ValueError, "documentation placeholder"):
            validate_suite_path(Path("benchmarks/results/SUITE_DIRECTORY"))


if __name__ == "__main__":
    unittest.main()
