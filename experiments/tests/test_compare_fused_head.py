"""Keep head selection tied to both timing and actual generated output."""
import json
from pathlib import Path
import tempfile
import unittest

from experiments.integration import benchmark_final_8  # establish legacy import paths
from experiments.integration import compare_fused_head as ab


class HeadComparisonTests(unittest.TestCase):
    def fixture(self, root, speedup=1.05, exact=True):
        baseline, head = root / "accepted", root / "head"
        for directory, fused in ((baseline, False), (head, True)):
            rows = []
            config = {"model": "model", "suite_dir": "suite", "seed": 1,
                      "warmups": 1, "repetitions": 3, "vllm_version": "0.30.0",
                      "vllm_budget": "default", "local_graph_pool": "private",
                      "budgets": {shape: 8192 for shape in ab.SHAPES},
                      "fused_greedy_output": fused}
            for shape in ab.SHAPES:
                phase = {kind: {"local": {"median_step_wall_ms": 1 / speedup if fused else 1}}
                         for kind in ("prefill", "decode", "mixed")}
                rate = 100 * speedup if fused else 100
                rows.append({"shape_id": shape, "burst": {"local_output_tokens_per_s": rate},
                             "mixed": {"local_output_tokens_per_s": rate}, "phases": {"phases": phase}})
                flags = {"cpp_scheduler": True}
                if fused:
                    flags["fused_greedy_output"] = True
                for prefix in (Path(), Path("mixed")):
                    path = directory / prefix / shape / "local.json"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    outputs = {"0": [1, 2 if exact or not fused else 3]}
                    path.write_text(json.dumps({"engine_flags": flags,
                                                "runs": [{"outputs": outputs}] * 3}))
                path = directory / "phases" / shape / "local.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"engine_flags": flags, "burst_outputs_sha256": "same",
                                            "mixed_outputs_sha256": "same"}))
                for path in (directory / shape / "vllm/result.json",
                             directory / "mixed" / shape / "vllm.json",
                             directory / "phases" / shape / "vllm.json"):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text('{"reference":"same"}')
            (directory / "summary.json").write_text(json.dumps({
                "status": "complete", "configuration": config, "rows": rows}))
        return baseline, head

    def test_selects_head_only_on_meaningful_gain_and_exact_outputs(self):
        for speedup, exact, selected in ((1.05, True, "fused-head"),
                                         (1.005, True, "accepted"),
                                         (.98, True, "accepted"),
                                         (1.05, False, "accepted")):
            with self.subTest(speedup=speedup, exact=exact), tempfile.TemporaryDirectory() as temporary:
                baseline, head = self.fixture(Path(temporary), speedup, exact)
                report = ab.compare(baseline, head)
                self.assertEqual(report["selected"], selected)
                self.assertEqual(report["exact_baseline_tokens"], exact)

    def test_rejects_different_reference(self):
        with tempfile.TemporaryDirectory() as temporary:
            baseline, head = self.fixture(Path(temporary))
            (head / "mixed" / ab.SHAPES[0] / "vllm.json").write_text("changed")
            with self.assertRaisesRegex(ValueError, "same saved"):
                ab.compare(baseline, head)

    def test_rejects_other_optimizations_in_head_arm(self):
        with tempfile.TemporaryDirectory() as temporary:
            baseline, head = self.fixture(Path(temporary))
            path = head / ab.SHAPES[0] / "local.json"
            value = json.loads(path.read_text())
            value["engine_flags"]["gemm_epilogues"] = "all"
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "more than"):
                ab.compare(baseline, head)

    def test_rejects_budget_changes_before_gpu_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            baseline, head = self.fixture(Path(temporary))
            config = json.loads((head / "summary.json").read_text())["configuration"]
            config["budgets"][ab.SHAPES[0]] = 2048
            with self.assertRaisesRegex(ValueError, "budgets"):
                ab.validate_baseline(baseline, config)


if __name__ == "__main__":
    unittest.main()
