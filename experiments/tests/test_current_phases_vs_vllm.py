"""CPU checks for synchronized eight-cell phase diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from experiments.integration import benchmark_current_phases_vs_vllm as phases
from experiments.integration.benchmark_integrated_graph import dry_schedule
import inference_engine_cpp as cpp
import torch


ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / "experiments/results/full-checkpoint-20260916T033540Z"


class PhaseTests(unittest.TestCase):
    def test_all_eight_cells_plan_all_three_phase_kinds(self):
        args = SimpleNamespace(suite_dir=SUITE, seed=20260914)
        self.assertEqual(len(phases.SHAPES), 8)
        for shape_id in phases.SHAPES:
            with self.subTest(shape_id=shape_id):
                case, requests, first, arrival, buckets, digest, blocks = (
                    phases.mixed_plan(args, shape_id))
                self.assertEqual(len(requests), case["max_running"])
                self.assertGreater(first, 0)
                self.assertGreater(arrival, 0)
                self.assertTrue(buckets)
                self.assertEqual(len(digest), 64)
                self.assertGreater(blocks, 0)
                burst_case, burst_requests, _, _, _ = phases.input_contract(
                    args, shape_id)
                burst = dry_schedule(torch, cpp, burst_case, args.seed,
                                     burst_requests)
                staggered = dry_schedule(torch, cpp, case, args.seed, requests)
                self.assertTrue({"prefill", "decode"}.issubset(
                    {step["kind"] for step in burst["steps"]}))
                self.assertIn("mixed", {step["kind"] for step in staggered["steps"]})

    def test_phase_summary_rejects_missing_kind(self):
        steps = [{"kind": kind, "wall_ms": value}
                 for kind, value in (("prefill", 2.0), ("decode", 1.0),
                                     ("decode", 3.0), ("mixed", 4.0))]
        summary = phases.phase_summary(steps)
        self.assertEqual(summary["decode"]["steps"], 2)
        self.assertEqual(summary["decode"]["median_step_wall_ms"], 2.0)
        with self.assertRaisesRegex(AssertionError, "no valid mixed"):
            phases.phase_summary(steps[:-1])

    def test_output_digest_joins_local_integer_and_vllm_string_ids(self):
        self.assertEqual(phases.output_digest({0: [1, 2], 1: [3]}),
                         phases.output_digest({"0": [1, 2], "1": [3]}))

    def test_saved_result_rejects_wrong_timing_scheme_and_commit(self):
        args = SimpleNamespace(warmups=1, repetitions=3, resume_commit=None)
        shape = phases.SHAPES[0]
        payload = {"status": "complete", "timing_scheme": phases.TIMING_SCHEME,
                   "shape_id": shape, "workload_sha256": "frozen", "model": "model",
                   "warmups": 1, "repetitions": 3, "repository_commit": "oldcommit",
                   "phases": {kind: {} for kind in phases.KINDS},
                   "runs": [{}, {}, {}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "local.json"
            path.write_text(json.dumps(payload))
            with mock.patch.object(phases, "repository_commit", return_value="newcommit"):
                with self.assertRaisesRegex(ValueError, "stale"):
                    phases.validate_saved(path, args, shape, "frozen", "model")
                args.resume_commit = ["oldcommit"]
                self.assertIsNotNone(phases.validate_saved(
                    path, args, shape, "frozen", "model"))
                payload["timing_scheme"] = "unsynchronized"
                path.write_text(json.dumps(payload))
                with self.assertRaisesRegex(ValueError, "stale"):
                    phases.validate_saved(path, args, shape, "frozen", "model")


if __name__ == "__main__":
    unittest.main()
