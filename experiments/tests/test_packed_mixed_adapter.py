"""CPU contracts for one-pass packed mixed metadata and delayed logits."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch


ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT / "experiments/integration", ROOT / "baseline",
                  ROOT / "engine/model_runner", ROOT / "engine/kvcache"):
    sys.path.insert(0, str(directory))

from model_adapter import (PackedMixedPiecewiseGraphModelAdapter,
                           combine_mixed_metadata)


def inputs():
    decode = (
        torch.tensor([10, 11]), torch.tensor([100, 50]),
        torch.tensor([9, 13]), torch.empty(0, dtype=torch.int32),
        torch.tensor([101, 51], dtype=torch.int32),
        torch.tensor([[4, 5], [6, 7]], dtype=torch.int32), 1, True,
    )
    prefill = (
        torch.tensor([20, 21, 22]), torch.tensor([0, 1, 0]),
        torch.tensor([17, 18, 33]), torch.tensor([0, 2, 3], dtype=torch.int32),
        torch.tensor([2, 1], dtype=torch.int32),
        torch.tensor([[8], [9]], dtype=torch.int32), 2, False,
    )
    return decode, prefill


class PackedMixedTests(unittest.TestCase):
    def test_whole_mixed_graph_dispatch_fills_deferred_decode_logits(self):
        adapter = object.__new__(PackedMixedPiecewiseGraphModelAdapter)
        adapter.model = SimpleNamespace(cfg=SimpleNamespace(vocab=4),
                                        lm_head=SimpleNamespace(weight=torch.empty(4)))
        adapter.loop = SimpleNamespace(current_step_is_mixed=lambda: True)
        adapter._pending_mixed = None
        adapter.enable_packed_mixed = True
        adapter.mixed_attention_policy = "fa3_varlen"
        adapter.full_mixed_graph = True
        adapter.full_mixed_hits = 0
        adapter.step_calls = []
        adapter.decisions = {}
        adapter.observer = None
        seen = []

        def full_forward(*args):
            seen.append((args[0].numel(), args[3].numel(), args[6]))
            return torch.arange(20, dtype=torch.float32).view(5, 4)

        key = (5, 5, torch.Size((4, 2)), 2)
        adapter.full_mixed_graphs = {key: SimpleNamespace(forward=full_forward)}
        adapter.piecewise_prefill = SimpleNamespace(
            forward=lambda *_args, **_kwargs: self.fail("unexpected piecewise fallback"))
        decode, prefill = inputs()
        placeholder = adapter(*decode)
        prefill_logits = adapter(*prefill)
        self.assertEqual(seen, [(5, 5, 2)])
        self.assertEqual(adapter.full_mixed_hits, 1)
        self.assertEqual(placeholder.tolist(), [[0., 1., 2., 3.], [4., 5., 6., 7.]])
        self.assertEqual(prefill_logits.shape, (3, 4))

    def test_metadata_keeps_decode_first_and_pads_page_tables(self):
        combined = combine_mixed_metadata(*inputs())
        self.assertEqual(combined[0].tolist(), [10, 11, 20, 21, 22])
        self.assertEqual(combined[3].tolist(), [0, 1, 2, 4, 5])
        self.assertEqual(combined[4].tolist(), [101, 51, 2, 1])
        self.assertEqual(combined[5].tolist(), [[4, 5], [6, 7], [8, 0], [9, 0]])
        self.assertEqual(combined[6], 2)

    def test_decode_logits_are_filled_after_prefill_without_stale_metadata(self):
        adapter = object.__new__(PackedMixedPiecewiseGraphModelAdapter)
        adapter.model = SimpleNamespace(cfg=SimpleNamespace(vocab=4),
                                        lm_head=SimpleNamespace(weight=torch.empty(4)))
        adapter.loop = SimpleNamespace(current_step_is_mixed=lambda: True)
        adapter._pending_mixed = None
        adapter.enable_packed_mixed = True
        adapter.mixed_attention_policy = "packed_paged"
        adapter.full_mixed_graph = False
        adapter.step_calls = []
        adapter.decisions = {}
        adapter.observer = None
        seen = []

        def forward(*args, **kwargs):
            seen.append((args[0].tolist(), args[3].tolist(), kwargs))
            return torch.arange(20, dtype=torch.float32).view(5, 4)

        adapter.piecewise_prefill = SimpleNamespace(forward=forward)
        decode, prefill = inputs()
        placeholder = adapter(*decode)
        decode[0][0] = 999  # C++ reuses the decode metadata buffer for prefill.
        prefill_logits = adapter(*prefill)
        self.assertEqual(seen, [([10, 11, 20, 21, 22], [0, 1, 2, 4, 5],
                                 {"mixed_decode_count": 0,
                                  "mixed_attention_policy": "packed_paged"})])
        self.assertEqual(placeholder.tolist(), [[0., 1., 2., 3.], [4., 5., 6., 7.]])
        self.assertEqual(prefill_logits.tolist(), [[8., 9., 10., 11.],
                                                   [12., 13., 14., 15.],
                                                   [16., 17., 18., 19.]])
        self.assertEqual(adapter.step_calls, [(True, 2, 2, 1), (False, 3, 2, 2)])

    def test_fa3_hybrid_receives_decode_row_count(self):
        adapter = object.__new__(PackedMixedPiecewiseGraphModelAdapter)
        adapter.model = SimpleNamespace(cfg=SimpleNamespace(vocab=4),
                                        lm_head=SimpleNamespace(weight=torch.empty(4)))
        adapter.loop = SimpleNamespace(current_step_is_mixed=lambda: True)
        adapter._pending_mixed = None
        adapter.enable_packed_mixed = True
        adapter.mixed_attention_policy = "fa3_hybrid"
        adapter.full_mixed_graph = False
        adapter.step_calls = []
        adapter.decisions = {}
        adapter.observer = None
        counts = []

        def forward(*_args, mixed_decode_count, mixed_attention_policy):
            counts.append(mixed_decode_count)
            self.assertEqual(mixed_attention_policy, "fa3_hybrid")
            return torch.zeros((5, 4))

        adapter.piecewise_prefill = SimpleNamespace(forward=forward)
        decode, prefill = inputs()
        adapter(*decode)
        adapter(*prefill)
        self.assertEqual(counts, [2])


if __name__ == "__main__":
    unittest.main()
