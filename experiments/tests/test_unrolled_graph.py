"""CPU-side protocol checks for K-step CUDA graph decode."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import types
import unittest
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT / "baseline", ROOT / "engine/kvcache", ROOT / "engine/graph",
                  ROOT / "experiments/decode"):
    sys.path.insert(0, str(directory))

UNROLLED_PATH = ROOT / "engine/graph/unrolled_graph_decoder.py"
SPEC = importlib.util.spec_from_file_location("unrolled_graph_decoder_test", UNROLLED_PATH)
UNROLLED = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(UNROLLED)

CONTROL_PATH = ROOT / "experiments/decode/benchmark_unrolled_graph.py"
CONTROL_SPEC = importlib.util.spec_from_file_location("benchmark_unrolled_graph", CONTROL_PATH)
CONTROL = importlib.util.module_from_spec(CONTROL_SPEC)
CONTROL_SPEC.loader.exec_module(CONTROL)


class UnrolledGraphTests(unittest.TestCase):
    def test_eos_scan_checks_every_position_and_uses_k_sentinel(self):
        tokens = torch.tensor([
            [99, 1, 1, 1],
            [2, 99, 2, 2],
            [3, 3, 3, 3],
            [4, 4, 4, 99],
        ])
        self.assertEqual(UNROLLED.first_eos_positions(tokens, 99).tolist(), [0, 1, 4, 3])
        self.assertEqual(UNROLLED.first_eos_positions(tokens, -1).tolist(), [4, 4, 4, 4])

    def test_exact_metadata_crosses_page_boundary(self):
        cfg = SimpleNamespace(n_layers=1, n_kv_heads=2, d_head=8, vocab=128,
                              max_seq_len=128)
        cache, metadata, ids, _ = CONTROL.stage_unrolled_case(
            torch, cfg, batch=2, context=15, max_steps=4,
            dtype=torch.float16, seed=7, device="cpu")
        self.assertEqual(ids.shape, (2, 1))
        self.assertEqual([int(row[0][0]) for row in metadata], [15, 16, 17, 18])
        self.assertEqual([int(row[1][0]) for row in metadata], [16, 17, 18, 19])
        slots = CONTROL.future_slots(metadata)
        self.assertEqual(slots[:, 0].tolist(), [15, 16, 17, 18])
        self.assertEqual(slots[:, 1].tolist(), [47, 48, 49, 50])
        self.assertEqual(cache.num_blocks, 4)

    def test_slot_snapshot_rejects_out_of_range_indices_before_index_select(self):
        cache = SimpleNamespace(num_blocks=2, block_size=16)
        with self.assertRaisesRegex(ValueError, "exceeds cache capacity 32"):
            CONTROL._validate_slots(cache, torch.tensor([0, 31, 32]))

    def test_comprehensive_logit_accuracy_retains_step_details(self):
        reference = (torch.zeros(2, 1, 4), torch.zeros(2, 1, 4))
        candidate = (reference[0].clone(), reference[1].clone())
        candidate[1][0, 0, 3] = 0.2
        reference_tokens = torch.stack([row.argmax(-1).reshape(-1) for row in reference])
        candidate_tokens = torch.stack([row.argmax(-1).reshape(-1) for row in candidate])
        result = CONTROL.trajectory_accuracy(
            torch, candidate_tokens, candidate, reference_tokens, reference, atol=.05)
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["logits_outside_atol"], 1)
        self.assertEqual(result["first_token_divergence_step"], 1)
        self.assertEqual(len(result["steps"]), 2)

    def test_validation_and_production_storage_contract_is_explicit(self):
        source = UNROLLED_PATH.read_text()
        self.assertIn("if self.retain_logits", source)
        self.assertIn("logits_kept.append(output)", source)
        self.assertIn("tokens = torch.stack(token_rows", source)

    def test_forward_retains_each_step_logits_only_for_validation(self):
        batch, steps, vocab = 2, 4, 7
        positions = [torch.full((batch,), step, dtype=torch.int32)
                     for step in range(steps)]
        metadata = tuple(
            (position, position + 1, torch.zeros(batch, 1, dtype=torch.int32),
             torch.full((batch,), step, dtype=torch.long))
            for step, position in enumerate(positions)
        )

        def fake_forward(_model, _cache, token, *_metadata, **_options):
            logits = torch.zeros(batch, 1, vocab)
            next_token = (token.reshape(batch) + 1) % vocab
            logits.scatter_(2, next_token[:, None, None], 1)
            return logits

        fake_module = types.SimpleNamespace(graph_decode_forward=fake_forward)
        with mock.patch.dict(sys.modules, {"paged_graph_decoder": fake_module}):
            validation = UNROLLED.UnrolledCUDAGraphDecoder(
                object(), object(), metadata, retain_logits=True)
            validation.s_first_ids.copy_(torch.tensor([[1], [3]]))
            validation_tokens, validation_logits = validation._forward()
            production = UNROLLED.UnrolledCUDAGraphDecoder(
                object(), object(), metadata, retain_logits=False)
            production.s_first_ids.copy_(torch.tensor([[1], [3]]))
            production_tokens, production_logits = production._forward()

        self.assertEqual(len(validation_logits), steps)
        self.assertEqual([tuple(item.shape) for item in validation_logits],
                         [(batch, 1, vocab)] * steps)
        self.assertEqual(production_logits, ())
        self.assertTrue(torch.equal(validation_tokens, production_tokens))
        self.assertEqual(validation_tokens.tolist(), [[2, 4], [3, 5], [4, 6], [5, 0]])

    def test_current_fusion_flags_reach_every_unrolled_step(self):
        batch, steps, vocab = 2, 2, 7
        metadata = tuple(
            (torch.full((batch,), step, dtype=torch.int32),
             torch.full((batch,), step + 1, dtype=torch.int32),
             torch.zeros(batch, 1, dtype=torch.int32),
             torch.full((batch,), step, dtype=torch.long))
            for step in range(steps)
        )
        seen = []

        def fake_forward(_model, _cache, token, *_metadata, **options):
            seen.append(options)
            logits = torch.zeros(batch, 1, vocab)
            logits.scatter_(2, token.reshape(batch, 1, 1), 1)
            return logits

        fake_module = types.SimpleNamespace(graph_decode_forward=fake_forward)
        with mock.patch.dict(sys.modules, {"paged_graph_decoder": fake_module}):
            decoder = UNROLLED.UnrolledCUDAGraphDecoder(
                object(), object(), metadata,
                decode_attention_policy="fa3",
                enable_residual_rmsnorm=True,
                enable_native_decode_qkv_postprocess=True,
            )
            decoder._forward()

        self.assertEqual(len(seen), steps)
        self.assertTrue(all(row["decode_attention_policy"] == "fa3" for row in seen))
        self.assertTrue(all(row["enable_residual_rmsnorm"] for row in seen))
        self.assertTrue(all(row["enable_native_decode_qkv_postprocess"] for row in seen))


if __name__ == "__main__":
    unittest.main()
