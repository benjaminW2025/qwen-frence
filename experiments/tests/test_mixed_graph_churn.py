"""CPU contracts for the evolving-batch graph-churn probe."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "experiments/integration/benchmark_mixed_graph_churn.py"
SPEC = importlib.util.spec_from_file_location("benchmark_mixed_graph_churn", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MixedGraphChurnTests(unittest.TestCase):
    def test_staggered_workload_exercises_changing_decode_population(self):
        frozen = [{"prompt": list(range(256))} for _ in range(8)]
        case, requests = MODULE.workload(frozen)
        self.assertEqual(case["arrivals"], [0, 0, 3, 4, 5, 6, 7, 8])
        self.assertEqual(case["outputs"], [18] * 8)
        self.assertEqual([r["arrival"] for r in requests], case["arrivals"])

    def test_full_logit_observer_checks_every_callback(self):
        reference = MODULE.LogitObserver(torch)
        args = (None, None, None, None, None, None, 1, True)
        reference(args, torch.tensor([[1., 2.]]))
        reference(args, torch.tensor([[3., 4.]]))
        candidate = MODULE.LogitObserver(torch, reference.rows)
        candidate(args, torch.tensor([[1., 2.]]))
        candidate(args, torch.tensor([[30., 4.]]))
        candidate.finish()
        self.assertEqual(candidate.mismatched_callbacks[0]["callback"], 1)
        self.assertEqual(candidate.mismatched_callbacks[0]["reason"], "logits")

    def test_graph_event_summary_separates_capture_replay_and_fallback(self):
        rows = [{"outcome": name, "capture_ms": ms}
                for name, ms in (("capture", 8.), ("replay", 0.),
                                 ("cache_full", 0.))]
        result = MODULE.summarize_events(rows)
        self.assertEqual(result["counts"]["capture"], 1)
        self.assertEqual(result["counts"]["replay"], 1)
        self.assertEqual(result["counts"]["cache_full"], 1)
        self.assertEqual(result["capture_ms"], 8.)
        self.assertAlmostEqual(result["replay_hit_rate"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
