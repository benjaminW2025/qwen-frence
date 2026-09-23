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
                                 {"mixed_decode_count": 0})])
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
        adapter.step_calls = []
        adapter.decisions = {}
        adapter.observer = None
        counts = []

        def forward(*_args, mixed_decode_count):
            counts.append(mixed_decode_count)
            return torch.zeros((5, 4))

        adapter.piecewise_prefill = SimpleNamespace(forward=forward)
        decode, prefill = inputs()
        adapter(*decode)
        adapter(*prefill)
        self.assertEqual(counts, [2])


if __name__ == "__main__":
    unittest.main()
