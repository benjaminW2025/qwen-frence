"""
Load in hugging face weights into our model config
"""

import torch
from transformers import AutoModelForCausalLM

from naive_forward import Qwen2Config, Model


class QwenWeightLoader:
    # Exact HF key -> our key.
    _EXACT = {
        "model.embed_tokens.weight": "embed.weight",
        "model.norm.weight": "norm.weight",
    }
    # HF keys we deliberately skip (handled separately).
    _SKIP = {"lm_head.weight"}  # tied to the embedding
    # Sequential substring renames for the per-layer keys (order matters).
    _RENAMES = [
        ("model.layers.", "layers."),
        (".self_attn.", "."),
        (".mlp.", "."),
        (".input_layernorm.", ".input_norm."),
        (".post_attention_layernorm.", ".post_attn_norm."),
    ]

    def __init__(self, cfg=None):
        self.cfg = cfg or Qwen2Config()

    def map_key(self, hf_key):
        """Translate one HF state_dict key to our module naming (None -> skip)."""
        if hf_key in self._SKIP:
            return None
        if hf_key in self._EXACT:
            return self._EXACT[hf_key]
        key = hf_key
        for src, dst in self._RENAMES:
            key = key.replace(src, dst)
        return key

    def remap_state_dict(self, hf_sd):
        """Translate a full HF state_dict into our naming (+ the tied lm_head)."""
        our_sd = {}
        for k, v in hf_sd.items():
            nk = self.map_key(k)
            if nk is not None:
                our_sd[nk] = v
        if self.cfg.tie_embeddings:
            our_sd["lm_head.weight"] = hf_sd["model.embed_tokens.weight"]
        return our_sd

    def convert(self, hf_model, device="cuda", dtype=torch.float16, strict=True):
        """Build our Model and load weights from an already-instantiated HF model."""
        model = Model(self.cfg).to(device, dtype).eval()
        model.load_state_dict(self.remap_state_dict(hf_model.state_dict()), strict=strict)
        return model

    def load_pretrained(self, model_id, device="cuda", dtype=torch.float16,
                        attn_implementation="sdpa", strict=True):
        """Convenience: download the HF model and convert it to our Model."""
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, attn_implementation=attn_implementation
        )
        return self.convert(hf_model, device, dtype, strict=strict)
