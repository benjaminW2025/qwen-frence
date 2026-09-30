"""CPU address/lifetime contracts for opt-in cross-segment buffer reuse."""
import importlib.util
from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "baseline"))
SPEC = importlib.util.spec_from_file_location(
    "piecewise_boundary_contract", ROOT / "engine/graph/piecewise_prefill.py")
MODULE = importlib.util.module_from_spec(SPEC)
with mock.patch.dict(sys.modules, {"ragged_prefill": SimpleNamespace(_rope_factors=None)}):
    SPEC.loader.exec_module(MODULE)


class BoundaryBuffersTests(unittest.TestCase):
    def test_capture_does_not_retain_autograd_intermediates(self):
        model = SimpleNamespace(layers=[], cfg=SimpleNamespace(),
                                embed=torch.nn.Embedding(16, 4))
        pool = SimpleNamespace(k_pool=[torch.zeros(1)])
        stream = mock.Mock()
        with torch.enable_grad(), \
                mock.patch.object(torch.cuda, "current_stream", return_value=stream), \
                mock.patch.object(torch.cuda, "Stream", return_value=stream), \
                mock.patch.object(torch.cuda, "stream", side_effect=lambda *a, **k: nullcontext()), \
                mock.patch.object(torch.cuda, "CUDAGraph"), \
                mock.patch.object(torch.cuda, "graph", side_effect=lambda *a, **k: nullcontext()):
            boundary = MODULE._AttentionBoundary(model, pool, 0, 8, None, None, None)
            self.assertFalse(boundary.result.requires_grad)
            self.assertIsNone(boundary.result.grad_fn)
            self.assertTrue(torch.is_grad_enabled())

    def boundary(self, bound=True):
        boundary = MODULE._AttentionBoundary.__new__(MODULE._AttentionBoundary)
        boundary.bound_residual = bound
        boundary.residual = torch.zeros(8, 4)
        boundary.attention = torch.full((8, 4), 123.)
        boundary.graph = mock.Mock()
        boundary.result = object()
        return boundary

    def test_aliases_skip_copies_on_repeated_padded_calls(self):
        boundary = self.boundary()
        for tokens in (3, 7, 1, 8):
            before = boundary.attention._version
            residual_before = boundary.residual._version
            result = boundary.run_from_attention(
                boundary.residual.view(8, 4), boundary.attention[:tokens], tokens)
            self.assertIs(result, boundary.result)
            self.assertEqual(boundary.attention._version, before)
            self.assertEqual(boundary.residual._version, residual_before)
        self.assertEqual(boundary.graph.replay.call_count, 4)

    def test_changed_residual_address_is_rejected(self):
        boundary = self.boundary()
        with self.assertRaisesRegex(ValueError, "address changed"):
            boundary.run_from_attention(boundary.residual.clone(), boundary.attention, 8)
        boundary.graph.replay.assert_not_called()

    def test_control_copies_only_live_attention_rows(self):
        boundary = self.boundary(False)
        boundary.run_from_attention(torch.ones(8, 4), torch.ones(3, 4), 3)
        torch.testing.assert_close(boundary.residual, torch.ones(8, 4))
        torch.testing.assert_close(boundary.attention[:3], torch.ones(3, 4))
        torch.testing.assert_close(boundary.attention[3:], torch.full((5, 4), 123.))

    def test_segments_keep_their_producer_residual(self):
        prefill = MODULE.PiecewisePrefill.__new__(MODULE.PiecewisePrefill)
        prefill.model = SimpleNamespace(layers=[object(), object()])
        prefill.pool = SimpleNamespace(k_pool=[torch.zeros(1)])
        prefill.buckets = (8, 16)
        prefill.max_capture_tokens, prefill.max_shapes = 16, 2
        prefill.shapes = {}
        prefill.captured_calls = prefill.eager_calls = 0
        prefill.enable_boundary_buffer_reuse = True
        def make_piece(model, pool, index, bucket, *args, **kwargs):
            return SimpleNamespace(result=(None, torch.zeros(bucket, 4)),
                                   residual=kwargs.get("residual_input"))
        with mock.patch.object(MODULE, "_AttentionBoundary", side_effect=make_piece):
            first = prefill.pieces(3)
            other = prefill.pieces(12)
            self.assertIs(prefill.pieces(7), first)
        for pieces in (first, other):
            self.assertIs(pieces[1].residual, pieces[0].result[1])
            self.assertIs(pieces[2].residual, pieces[1].result[1])
        self.assertIsNot(first[1].residual, other[1].residual)

    def test_copy_counters_prove_which_path_ran(self):
        bound, control = self.boundary(), self.boundary(False)
        for tokens in (3, 8):
            bound.run_from_attention(bound.residual, bound.attention[:tokens], tokens)
            control.run_from_attention(torch.ones(8, 4), torch.ones(tokens, 4), tokens)
        self.assertEqual((bound.residual_copies, bound.attention_copies), (0, 0))
        self.assertEqual((control.residual_copies, control.attention_copies), (2, 2))
        prefill = MODULE.PiecewisePrefill.__new__(MODULE.PiecewisePrefill)
        # Index 0 has no residual/attention input and is excluded.
        prefill.shapes = {8: [object(), bound, control]}
        self.assertEqual(prefill.boundary_copies(), {"residual": 2, "attention": 2})

    def test_shared_pool_reaches_every_segment_and_capture_memory_is_recorded(self):
        for shared in (True, False):
            prefill = MODULE.PiecewisePrefill.__new__(MODULE.PiecewisePrefill)
            prefill.model = SimpleNamespace(layers=[object(), object()])
            prefill.pool = SimpleNamespace(k_pool=[torch.zeros(1)])
            prefill.buckets = (8, 16)
            prefill.max_capture_tokens, prefill.max_shapes = 16, 2
            prefill.shapes, prefill.capture_memory = {}, {}
            prefill.captured_calls = prefill.eager_calls = 0
            prefill.graph_pool = object() if shared else None
            received = []
            def make_piece(model, pool, index, bucket, *args, **kwargs):
                received.append(kwargs.get("graph_pool"))
                return SimpleNamespace(result=(None, torch.zeros(bucket, 4)))
            reserved = iter((1000, 1600, 1600, 1900))
            with mock.patch.object(MODULE, "_AttentionBoundary", side_effect=make_piece), \
                    mock.patch.object(MODULE.torch.cuda, "memory_reserved", side_effect=lambda device: next(reserved)):
                prefill.pieces(3)
                prefill.pieces(12)
            self.assertEqual(len(received), 6)  # 3 segments x 2 buckets
            self.assertTrue(all(pool is prefill.graph_pool for pool in received))
            self.assertEqual(prefill.capture_memory, {8: 600, 16: 300})

    def test_fused_epilogue_flag_reaches_every_segment(self):
        for enabled in (True, False):
            prefill = MODULE.PiecewisePrefill.__new__(MODULE.PiecewisePrefill)
            prefill.model = SimpleNamespace(layers=[object(), object()])
            prefill.pool = SimpleNamespace(k_pool=[torch.zeros(1)])
            prefill.buckets = (8,)
            prefill.max_capture_tokens, prefill.max_shapes = 8, 1
            prefill.shapes, prefill.capture_memory = {}, {}
            prefill.captured_calls = prefill.eager_calls = 0
            prefill.enable_fused_gemm_epilogues = enabled
            received = []
            def make_piece(model, pool, index, bucket, *args, **kwargs):
                received.append(kwargs.get("enable_fused_gemm_epilogues", False))
                return SimpleNamespace(result=(None, torch.zeros(bucket, 4)))
            with mock.patch.object(MODULE, "_AttentionBoundary", side_effect=make_piece), \
                    mock.patch.object(MODULE.torch.cuda, "memory_reserved", return_value=0):
                prefill.pieces(5)
            self.assertEqual(received, [enabled] * 3)


if __name__ == "__main__":
    unittest.main()
