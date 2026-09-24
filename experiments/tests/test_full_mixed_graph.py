"""CPU-side shape gate for whole-step mixed graph reuse."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT / "engine/graph", ROOT / "baseline",
                  ROOT / "engine/model_runner", ROOT / "engine/kvcache"):
    sys.path.insert(0, str(directory))

from full_mixed import FullMixedGraph


class FullMixedGraphTests(unittest.TestCase):
    def test_replay_stages_changed_values_into_fixed_inputs(self):
        graph = object.__new__(FullMixedGraph)
        graph.ids = torch.zeros(2, dtype=torch.long)
        graph.positions = torch.zeros(2, dtype=torch.long)
        graph.slots = torch.zeros(2, dtype=torch.long)
        graph.cu = torch.zeros(3, dtype=torch.int32)
        graph.context = torch.zeros(2, dtype=torch.int32)
        graph.table = torch.zeros((2, 2), dtype=torch.int32)
        graph.max_query = 1
        called = []
        graph.graph = type("Replay", (), {"replay": lambda self: called.append(True)})()
        graph.logits = torch.tensor([[1., 2.], [3., 4.]])
        args = (torch.tensor([5, 6]), torch.tensor([7, 8]), torch.tensor([9, 10]),
                torch.tensor([0, 1, 2], dtype=torch.int32),
                torch.tensor([11, 12], dtype=torch.int32),
                torch.tensor([[1, 2], [3, 4]], dtype=torch.int32), 1)
        self.assertIs(graph.forward(*args), graph.logits)
        self.assertEqual(called, [True])
        self.assertEqual(graph.ids.tolist(), [5, 6])
        self.assertEqual(graph.table.tolist(), [[1, 2], [3, 4]])
        self.assertIsNone(graph.forward(*args[:-2], torch.zeros(2, 3, dtype=torch.int32), 1))

    def test_replay_requires_same_metadata_shapes_and_query_bound(self):
        graph = object.__new__(FullMixedGraph)
        graph.ids = torch.empty(1028)
        graph.positions = torch.empty(1028)
        graph.slots = torch.empty(1028)
        graph.cu = torch.empty(9)
        graph.context = torch.empty(8)
        graph.table = torch.empty(8, 18)
        graph.max_query = 256
        args = (torch.empty(1028), torch.empty(1028), torch.empty(1028),
                torch.empty(9), torch.empty(8), torch.empty(8, 18), 256)
        self.assertTrue(graph.compatible(*args))
        self.assertFalse(graph.compatible(*args[:-2], torch.empty(8, 19), 256))
        self.assertFalse(graph.compatible(*args[:-1], 128))


if __name__ == "__main__":
    unittest.main()
