"""CPU contracts for the deliberately scope-limited final comparison."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.integration import benchmark_final_8 as final


class FinalEightTests(unittest.TestCase):
    def test_selects_smallest_within_one_percent_and_caps_private_pool(self):
        report = {"cells": {"cell": {"tokens_per_s": {
            "2048": 100, "4096": 119, "8192": 120, "16384": 150}}}}
        rates = final.candidates(report, "cell", 8192)
        self.assertNotIn(16384, rates)
        self.assertEqual(final.choose_budget(rates, .01), 4096)

    def test_rejects_nonfinite_and_empty_rates(self):
        report = {"cells": {"cell": {"tokens_per_s": {"8192": float("nan")}}}}
        with self.assertRaises(ValueError):
            final.candidates(report, "cell", 8192)

    def test_no_candidate_flags_or_sweep_commands(self):
        args = SimpleNamespace(suite_dir=Path("suite"), output_dir=Path("out"),
            model="model", seed=1, warmups=1, repetitions=3)
        command = final.command(args, "run-cell", final.SHAPES[0], 8192)
        for flag in ("--boundary-buffers", "--stable-decode-metadata", "--fused-greedy-output"):
            self.assertNotIn(flag, command)
        self.assertEqual(command[command.index("--gemm-epilogues") + 1], "off")
        self.assertEqual(command[command.index("--prefill-graph-pool") + 1], "private")
        self.assertEqual(command[command.index("--vllm-budget") + 1], "default")

    def test_plan_does_not_run_gpu_or_create_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new"
            with patch("sys.argv", ["final", "--output-dir", str(output), "--plan"]), \
                 patch.object(final, "supported_plan", return_value={"validated": True}) as plan, \
                 patch.object(final, "stream_run") as run, patch("builtins.print"):
                final.main()
            self.assertEqual(plan.call_count, 8)
            run.assert_not_called()
            self.assertFalse(output.exists())

    def test_incomplete_or_fused_calibration_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            for report in ({"status": "partial", "attention": "fa3"},
                           {"status": "complete", "attention": "fa3", "gemm_epilogues": "all"}):
                path.write_text(json.dumps(report))
                with self.assertRaises(ValueError):
                    final.read_sweep(path)

    def test_full_progress_is_written_after_every_cell(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for shape in final.SHAPES:
                for prefix in (root, root / "mixed", root / "phases"):
                    path = prefix / shape / "comparison.json"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps({"status": "complete"}))
            with patch("sys.argv", ["final", "--output-dir", str(root)]), \
                 patch.object(final, "supported_plan", return_value={}), \
                 patch.object(final, "stream_run") as run, patch("builtins.print"):
                final.main()
            self.assertEqual(run.call_count, 8)
            summary = json.loads((root / "summary.json").read_text())
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(len(summary["rows"]), 8)


if __name__ == "__main__":
    unittest.main()
