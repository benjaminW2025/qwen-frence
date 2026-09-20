"""CPU-side correctness and validation for the fused output-head candidate."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import torch


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "custom_kernels/fused_lm_head.py"
SOURCE = PATH.read_text()
HAS_TRITON = importlib.util.find_spec("triton") is not None
if HAS_TRITON:
    SPEC = importlib.util.spec_from_file_location("fused_lm_head", PATH)
    MODULE = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(MODULE)
else:
    MODULE = None


class FusedOutputHeadSourceTests(unittest.TestCase):
    def test_kernel_uses_tiled_dot_and_two_stage_argmax(self):
        self.assertIn("tl.dot(hidden, weight)", SOURCE)
        self.assertIn("_reduce_partial_argmax", SOURCE)
        self.assertNotIn("torch.empty((rows, weight.shape[0])", SOURCE)

    def test_kernel_rounds_logits_and_prefers_lowest_tied_token(self):
        self.assertIn("accumulator.to(hidden_ptr.dtype.element_ty)", SOURCE)
        self.assertIn("tl.min(winning_indices", SOURCE)

    def test_global_correctness_suite_covers_b8_and_b64(self):
        source = (ROOT / "correctness/checks/check_custom_kernels.py").read_text()
        self.assertIn("check_fused_lm_head", source)
        self.assertIn("for batch in (8, 64)", source)


@unittest.skipUnless(HAS_TRITON, "Triton is not installed")
class FusedOutputHeadTests(unittest.TestCase):
    def test_chunked_control_matches_materialized_argmax_and_first_tie(self):
        hidden = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        weight = torch.tensor([[2.0, 0.0], [2.0, 0.0], [0.0, 3.0], [0.0, 1.0]])
        expected = torch.nn.functional.linear(hidden, weight).argmax(-1)
        actual = MODULE.chunked_lm_head_argmax(hidden, weight, chunk_size=2)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(actual.tolist(), [0, 2])

    def test_workspace_shape_uses_flattened_rows_and_vocab_tiles(self):
        hidden = torch.empty(8, 1, 1536)
        weight = torch.empty(151936, 1536)
        self.assertEqual(MODULE.workspace_shape(hidden, weight), (8, 1187))

    def test_fused_candidate_rejects_cpu_before_launch(self):
        with self.assertRaisesRegex(ValueError, "CUDA"):
            MODULE.fused_lm_head_argmax(
                torch.empty(8, 1536, dtype=torch.float16),
                torch.empty(128, 1536, dtype=torch.float16),
            )


if __name__ == "__main__":
    unittest.main()
