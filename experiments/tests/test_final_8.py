"""CPU contracts for the deliberately scope-limited final comparison."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.integration import benchmark_final_8 as final


class FinalEightTests(unittest.TestCase):
    def test_local_only_never_launches_vllm_and_head_is_explicit(self):
        args = SimpleNamespace(suite_dir=Path("suite"), output_dir=Path("out"),
            model="model", seed=1, warmups=1, repetitions=3,
            backend="local", reference_dir=Path("reference"), fused_greedy_output=True,
            reference_commits=["abcdef1"])
        commands = final.cell_commands(args, final.SHAPES[0], 8192)
        self.assertEqual([cmd[2] for cmd in commands],
                         ["check", "run-local", "run-local", "run-local", "analyze"])
        for cmd in commands:
            self.assertNotIn("run-vllm", cmd)
            self.assertNotIn("run-cell", cmd)
            self.assertIn("--fused-greedy-output", cmd)
            self.assertIn("--resume-commit", cmd)

    def test_reference_only_never_launches_local(self):
        args = SimpleNamespace(suite_dir=Path("suite"), output_dir=Path("out"),
            model="model", seed=1, warmups=1, repetitions=3,
            backend="vllm", reference_dir=None, fused_greedy_output=False)
        commands = final.cell_commands(args, final.SHAPES[0], 8192)
        self.assertEqual([cmd[2] for cmd in commands], ["run-vllm"] * 3)

    def test_reference_copy_preserves_contents_and_rejects_overwrite(self):
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "reference.json"
            original = b'{"repository_commit":"abcdef1"}\n'
            source.write_bytes(original)
            args = SimpleNamespace(output_dir=root / "out", reference_commits=["abcdef1"])
            item = {"source": source, "relative": Path("mixed/cell/vllm.json"),
                    "sha256": hashlib.sha256(original).hexdigest()}
            final.stage_reference(args, [item])
            destination = args.output_dir / item["relative"]
            self.assertEqual(destination.read_bytes(), original)
            final.stage_reference(args, [item])
            destination.write_text("changed")
            with self.assertRaises(ValueError):
                final.stage_reference(args, [item])

    def test_local_requires_complete_reference_before_running(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv = ["final", "--backend", "local", "--reference-dir", str(root / "missing"),
                    "--output-dir", str(root / "out")]
            with patch("sys.argv", argv), patch.object(final, "supported_plan", return_value={}), \
                 patch.object(final, "stream_run") as run:
                with self.assertRaisesRegex(ValueError, "reference needs"):
                    final.main()
            run.assert_not_called()
            self.assertFalse((root / "out").exists())

    def test_complete_local_run_uses_only_local_actions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for shape in final.SHAPES:
                for prefix in (root, root / "mixed", root / "phases"):
                    path = prefix / shape / "comparison.json"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps({"status": "complete"}))
            argv = ["final", "--backend", "local", "--reference-dir", str(root / "ref"),
                    "--output-dir", str(root), "--fused-greedy-output"]
            with patch("sys.argv", argv), patch.object(final, "supported_plan", return_value={}), \
                 patch.object(final, "validate_reference", return_value=[]), \
                 patch.object(final, "stream_run") as run, patch("builtins.print"):
                final.main()
            self.assertEqual(run.call_count, 8 * 5)
            for call in run.call_args_list:
                self.assertNotIn("run-vllm", call.args[0])
            summary = json.loads((root / "summary.json").read_text())
            self.assertTrue(summary["configuration"]["fused_greedy_output"])
            self.assertFalse(summary["configuration"]["k_step_production"])

    def test_reference_contracts_checked_before_preserving_original_commits(self):
        import benchmark_current_8_vs_vllm as burst
        import benchmark_current_mixed_8_vs_vllm as mixed
        import benchmark_current_phases_vs_vllm as phases
        import benchmark_latest_vs_vllm as setup
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shape = final.SHAPES[0]
            source_commit = "abcdef1"
            rows = ((root / shape / "vllm/result.json", {
                "system": {"repository": {"commit": source_commit}}}),
                (root / "mixed" / shape / "vllm.json", {
                    "repository_commit": source_commit, "num_blocks": 200,
                    "arrival_step": 4, "first_wave": 4, "vllm_step_mode": True,
                    "runs": [{}, {}, {}]}),
                (root / "phases" / shape / "vllm.json", {
                    "repository_commit": source_commit, "num_blocks": 200,
                    "arrival_step": 4, "vllm_step_mode": True, "runs": [{}, {}, {}]}))
            for path, row in rows:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(row))
            args = SimpleNamespace(reference_dir=root, suite_dir=root, output_dir=root / "local",
                model="model", seed=1, warmups=1, repetitions=3, fused_greedy_output=True)
            with patch.object(final, "SHAPES", (shape,)), \
                 patch.object(burst, "input_contract", return_value=({}, [], object(), "digest", 200)), \
                 patch.object(burst, "vllm_result") as burst_check, \
                 patch.object(mixed, "mixed_plan", return_value=({}, [], 4, 4, None, "fingerprint", 200)), \
                 patch.object(mixed, "validate_saved") as mixed_check, \
                 patch.object(phases, "validate_saved") as phase_check, \
                 patch.object(setup, "resolve_model_source", return_value="model"):
                staged = final.validate_reference(args, {shape: 8192}, check_hardware=False)
                self.assertEqual(len(staged), 3)
                self.assertEqual(args.reference_commits, [source_commit])
                burst_check.assert_called_once()
                mixed_check.assert_called_once()
                phase_check.assert_called_once()
                wrong = dict(rows[1][1], arrival_step=5)
                rows[1][0].write_text(json.dumps(wrong))
                with self.assertRaisesRegex(ValueError, "arrival"):
                    final.validate_reference(args, {shape: 8192}, check_hardware=False)

    def test_reference_only_saves_reference_progress_without_comparisons(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("sys.argv", ["final", "--output-dir", str(root), "--backend", "vllm"]), \
                 patch.object(final, "supported_plan", return_value={}), \
                 patch.object(final, "stream_run") as run, patch("builtins.print"):
                final.main()
            self.assertEqual(run.call_count, 8 * 3)
            summary = json.loads((root / "reference-summary.json").read_text())
            self.assertEqual(summary["status"], "complete")
            self.assertFalse((root / "summary.json").exists())

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
