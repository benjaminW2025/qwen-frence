"""CPU-only gates for the eight-cell latest-engine comparison."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from experiments.integration import benchmark_current_8_vs_vllm as benchmark


ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / "experiments/results/full-checkpoint-20260916T033540Z"


class CurrentEightTests(unittest.TestCase):
    def test_all_eight_frozen_workloads_validate(self):
        args = SimpleNamespace(suite_dir=SUITE, seed=20260914)
        self.assertEqual(len(benchmark.SHAPES), 8)
        for shape in benchmark.SHAPES:
            with self.subTest(shape=shape):
                case, requests, workload, digest, blocks = benchmark.input_contract(args, shape)
                self.assertEqual(len(requests), case["max_running"])
                self.assertEqual(len(workload.requests), case["max_running"])
                self.assertEqual(len(digest), 64)
                self.assertGreater(blocks, 0)

    def test_cpu_dispatch_plan_includes_packed_mixed_rows(self):
        args = SimpleNamespace(suite_dir=SUITE, seed=20260914)
        expected_mixed = (0, 0, 7, 7, 7, 7, 63, 63)
        expected_max = (0, 0, 2055, 2055, 2104, 2104, 2111, 2111)
        for shape, mixed, maximum in zip(benchmark.SHAPES, expected_mixed,
                                         expected_max):
            with self.subTest(shape=shape):
                case, requests, _, _, _ = benchmark.input_contract(args, shape)
                plan = benchmark.dispatch_plan(case, requests, args.seed)
                self.assertEqual(plan["mixed_steps"], mixed)
                self.assertEqual(plan["max_packed_mixed_tokens"], maximum)
                self.assertEqual(plan["prefill_buckets"],
                                 [2048] if not mixed else [2048, maximum])
                self.assertEqual(plan["swiglu_fused_buckets"],
                                 plan["prefill_buckets"])

    def test_vllm_result_refuses_mismatched_capacity(self):
        args = SimpleNamespace(suite_dir=SUITE, seed=20260914)
        shape = benchmark.SHAPES[0]
        case, _, workload, digest, blocks = benchmark.input_contract(args, shape)
        source = next((SUITE / shape / "reference").glob("*.json"))
        payload = json.loads(source.read_text())
        model = payload["configuration"]["model"]
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "reference.json"
            destination.write_text(json.dumps(payload))
            self.assertIsNotNone(benchmark.vllm_result(
                destination.parent, workload, digest, case, blocks, model, 1, 3))
            payload["configuration"]["num_blocks"] += 1
            destination.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "num_blocks"):
                benchmark.vllm_result(
                    destination.parent, workload, digest, case, blocks, model, 1, 3)

    def test_completed_local_result_must_match_engine_flags(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "local.json"
            path.write_text(json.dumps({"status": "complete", "workload_sha256": "frozen",
                                        "engine_flags": benchmark.ENGINE_FLAGS}))
            self.assertTrue(benchmark.local_is_complete(path, "frozen"))
            with self.assertRaisesRegex(ValueError, "mismatched"):
                benchmark.local_is_complete(path, "different")

    def test_prior_commit_requires_explicit_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "local.json"
            path.write_text(json.dumps({"status": "complete", "workload_sha256": "frozen",
                                        "engine_flags": benchmark.ENGINE_FLAGS,
                                        "repository_commit": "f318493" + "0" * 33}))
            with self.assertRaisesRegex(ValueError, "mismatched"):
                benchmark.local_is_complete(path, "frozen", commit="newcommit")
            self.assertTrue(benchmark.local_is_complete(
                path, "frozen", commit="newcommit", resume_commit="f318493"))

    def test_multiple_prior_commits_are_forwarded_and_matched(self):
        allowed = ["f318493", "2627dcc"]
        self.assertTrue(benchmark.commit_matches("2627dcc" + "0" * 33,
                                                 "newcommit", allowed))
        self.assertTrue(benchmark.commit_matches("f318493" + "0" * 33,
                                                 "newcommit", allowed))
        self.assertFalse(benchmark.commit_matches("unrelated", "newcommit", allowed))
        self.assertEqual(benchmark.resume_options(allowed),
                         ["--resume-commit", "f318493",
                          "--resume-commit", "2627dcc"])

    def test_mixed_child_does_not_receive_burst_only_reuse_option(self):
        args = SimpleNamespace(suite_dir=SUITE, output_dir=Path("out"),
                               shape_id=benchmark.SHAPES[0],
                               model="Qwen/Qwen2.5-1.5B", device="cuda:0",
                               seed=20260914, warmups=1, repetitions=3,
                               reuse_vllm_from=Path("old-burst"),
                               resume_commit="f318493")
        self.assertIn("--reuse-vllm-from", benchmark.forwarded(
            args, "run-vllm", args.shape_id))
        child = benchmark.mixed_forwarded(args, args.shape_id)
        self.assertNotIn("--reuse-vllm-from", child)
        self.assertIn("--resume-commit", child)
        self.assertEqual(child[2], "run-cell")

    def test_analysis_joins_local_and_vllm_by_request_id(self):
        shape = benchmark.SHAPES[0]
        source = next((SUITE / shape / "reference").glob("*.json"))
        payload = json.loads(source.read_text())
        model = payload["configuration"]["model"]
        commit = payload["system"]["repository"]["commit"]
        args_for_input = SimpleNamespace(suite_dir=SUITE, seed=20260914)
        case, _, _, digest, blocks = benchmark.input_contract(args_for_input, shape)
        outputs = {row["request_id"]: row["output_ids"]
                   for row in payload["backends"]["vllm"]["runs"][-1]["requests"]}
        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(suite_dir=SUITE, output_dir=Path(temporary),
                                   seed=20260914, warmups=1, repetitions=3,
                                   reuse_vllm_from=None, resume_commit=None)
            local_path, reference_dir, comparison = benchmark.stage_paths(args, shape)
            reference_dir.mkdir(parents=True)
            (reference_dir / "reference.json").write_bytes(source.read_bytes())
            benchmark.atomic_json(local_path, {
                "status": "complete", "shape_id": shape, "model": model,
                "workload_sha256": digest, "engine_flags": benchmark.ENGINE_FLAGS,
                "repository_commit": commit,
                "num_blocks": blocks, "warmups": 1, "repetitions": 3,
                "dispatch_plan": benchmark.dispatch_plan(case,
                    benchmark.input_contract(args_for_input, shape)[1], args.seed),
                "runs": [{"outputs": outputs, "mixed_steps": 0,
                          "packed_mixed_calls": 0}],
                "median_output_tokens_per_s": 2500.0,
            })
            with mock.patch.object(benchmark, "repository_commit", return_value=commit):
                report = benchmark.analyze_cell(args, shape, model)
            self.assertEqual(report["exact_output_requests"], case["max_running"])
            self.assertTrue(comparison.is_file())

            # The previous runner's completed cell predates the additional
            # derived SwiGLU field, but its actual dispatch buckets are unchanged.
            old_commit = "f318493" + "0" * 33
            payload["system"]["repository"]["commit"] = old_commit
            (reference_dir / "reference.json").write_text(json.dumps(payload))
            previous = json.loads(local_path.read_text())
            previous["repository_commit"] = old_commit
            del previous["dispatch_plan"]["swiglu_fused_buckets"]
            benchmark.atomic_json(local_path, previous)
            args.resume_commit = "f318493"
            with mock.patch.object(benchmark, "repository_commit", return_value="newcommit"):
                resumed = benchmark.analyze_cell(args, shape, model)
            self.assertEqual(resumed["exact_output_requests"], case["max_running"])


if __name__ == "__main__":
    unittest.main()
