"""Focused CPU contracts for QKV and gate/up packing."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

import torch


BASELINE = Path(__file__).resolve().parents[1]
if str(BASELINE) not in sys.path:
    sys.path.insert(0, str(BASELINE))

from naive_forward import DecoderLayer, Model, Qwen2Config  # noqa: E402
from weight_loader import QwenWeightLoader  # noqa: E402


def tiny_config(**overrides):
    return Qwen2Config(
        vocab=32, d_model=16, d_ff=32, n_layers=2, n_heads=2,
        n_kv_heads=1, d_head=8, max_seq_len=32, **overrides,
    )


class PackedProjectionTests(unittest.TestCase):
    def test_packed_outputs_match_the_loaded_separate_weights(self):
        torch.manual_seed(7)
        layer = DecoderLayer(tiny_config()).eval()
        reference = deepcopy(layer).eval()
        hidden = torch.randn(3, 2, 16)
        expected_qkv = reference.q_proj(hidden), reference.k_proj(hidden), reference.v_proj(hidden)
        expected_gate_up = reference.gate_proj(hidden), reference.up_proj(hidden)

        layer.pack_projections_()
        observed_qkv = layer.project_qkv(hidden)
        observed_gate_up = layer.project_gate_up(hidden)
        for expected, observed in zip(expected_qkv + expected_gate_up, observed_qkv + observed_gate_up):
            torch.testing.assert_close(observed, expected)
        self.assertIsNotNone(layer.qkv_proj.bias)
        self.assertFalse(hasattr(layer, "q_proj"))
        self.assertFalse(hasattr(layer, "gate_proj"))

    def test_model_packs_every_layer_and_keeps_tied_output_weight(self):
        model = Model(tiny_config()).eval()
        self.assertIs(model.lm_head.weight, model.embed.weight)
        model.pack_projections_()
        self.assertTrue(all(layer.qkv_proj is not None for layer in model.layers))
        self.assertTrue(all(layer.gate_up_proj is not None for layer in model.layers))
        self.assertIs(model.lm_head.weight, model.embed.weight)

    def test_loader_packs_only_after_loading_hf_layout_weights(self):
        source = Model(tiny_config(pack_qkv=False, pack_gate_up=False)).eval()
        hf_state = {}
        for key, value in source.state_dict().items():
            if key == "embed.weight":
                hf_key = "model.embed_tokens.weight"
            elif key == "norm.weight":
                hf_key = "model.norm.weight"
            elif key == "lm_head.weight":
                hf_key = "lm_head.weight"
            else:
                _, layer, rest = key.split(".", 2)
                if rest.startswith(("q_proj.", "k_proj.", "v_proj.", "o_proj.")):
                    hf_key = f"model.layers.{layer}.self_attn.{rest}"
                elif rest.startswith(("gate_proj.", "up_proj.", "down_proj.")):
                    hf_key = f"model.layers.{layer}.mlp.{rest}"
                elif rest.startswith("input_norm."):
                    hf_key = f"model.layers.{layer}.input_layernorm.{rest.removeprefix('input_norm.')}"
                else:
                    hf_key = f"model.layers.{layer}.post_attention_layernorm.{rest.removeprefix('post_attn_norm.')}"
            hf_state[hf_key] = value.clone()

        class FakeHFModel:
            def state_dict(self):
                return hf_state

        loaded = QwenWeightLoader(tiny_config()).convert(FakeHFModel(), "cpu", torch.float32)
        hidden = torch.randn(2, 16)
        for expected, observed in zip(
            (source.layers[0].q_proj(hidden), source.layers[0].k_proj(hidden), source.layers[0].v_proj(hidden)),
            loaded.layers[0].project_qkv(hidden),
        ):
            torch.testing.assert_close(observed, expected)
        self.assertFalse(hasattr(loaded.layers[0], "q_proj"))

    def test_disabled_packing_keeps_the_reference_layout(self):
        layer = DecoderLayer(tiny_config(pack_qkv=False, pack_gate_up=False)).eval()
        layer.pack_projections_()
        self.assertIsNone(layer.qkv_proj)
        self.assertIsNone(layer.gate_up_proj)
        self.assertTrue(hasattr(layer, "q_proj"))
        self.assertTrue(hasattr(layer, "gate_proj"))


if __name__ == "__main__":
    unittest.main()
