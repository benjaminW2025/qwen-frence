"""
Naive Qwen2.5-1.5B decoder forward pass in pure PyTorch.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from kv_cache import KVCache

@dataclass
class Qwen2Config:
    """Qwen2.5-1.5B architecture hyperparameters."""
    vocab: int = 151936
    d_model: int = 1536
    d_ff: int = 8960
    n_layers: int = 28
    n_heads: int = 12
    n_kv_heads: int = 2          # grouped-query attention (GQA): 12 q heads share 2 kv heads
    d_head: int = 128            # n_heads * d_head = 1536 = d_model
    max_seq_len: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    attention_bias: bool = True  # Qwen2.5 puts a bias on q/k/v (but not o_proj)
    tie_embeddings: bool = True  # 1.5B ties lm_head to the embedding matrix
    use_custom_kernels: bool = False  # opt-in Triton RMSNorm + Qwen RoPE

    def __post_init__(self):
        assert self.n_heads * self.d_head == self.d_model, \
            "n_heads * d_head must equal d_model"


def rotate_half(x):
    # Split the last dim in half and rotate the pairs: [x1, x2] -> [-x2, x1].
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin):
    # RoPE: rotate each feature pair by its position's angle so q·k encodes RELATIVE position.
    # cos/sin are shaped to broadcast over (batch, heads, seq, head_dim). Qwen uses the
    # rotate-half convention (pairs are (i, i + d/2)), matching HF -- not the interleaved one.
    return (x * cos) + (rotate_half(x) * sin)


def apply_rope(x, cos, sin, cfg, positions):
    """Dispatch RoPE without importing Triton unless the custom path is enabled."""
    if cfg.use_custom_kernels:
        from kernel_dispatch import rope
        return rope(x, positions, cfg.rope_theta)
    return apply_rotary_pos_emb(x, cos, sin)


def apply_rms_norm(x, norm, cfg):
    """Dispatch RMSNorm while retaining nn.RMSNorm as the reference path."""
    if cfg.use_custom_kernels:
        from kernel_dispatch import rms_norm
        return rms_norm(x, norm.weight, cfg.rms_norm_eps)
    return norm(x)


SWIGLU_FUSION_ROW_THRESHOLD = 1408
ROPE_KV_FUSION_TOKEN_THRESHOLD = 64


def apply_swiglu(gate, up, cfg, *, enable_regime_fusions=False):
    """Apply the measured packed-row SwiGLU policy without device inspection."""
    rows = gate.numel() // gate.shape[-1]
    if (
        enable_regime_fusions
        and cfg.use_custom_kernels
        and rows > SWIGLU_FUSION_ROW_THRESHOLD
    ):
        from kernel_dispatch import swiglu

        return swiglu(gate, up, block_size=512, num_warps=4, num_stages=2)
    return F.silu(gate) * up


def apply_packed_rope_kv_write(
    q, k, v, positions, slot_mapping, k_pool, v_pool, cfg,
    *, enable_regime_fusions=False,
):
    """Return fused rotated Q, or ``None`` when the measured policy does not apply."""
    if not (
        enable_regime_fusions
        and cfg.use_custom_kernels
        and q.ndim == 4
        and q.shape[0] == 1
        and q.shape[2] >= ROPE_KV_FUSION_TOKEN_THRESHOLD
    ):
        return None
    from kernel_dispatch import rope_kv_write

    return rope_kv_write(
        q, k, v, positions, slot_mapping, k_pool, v_pool,
        base=cfg.rope_theta, num_warps=2, num_stages=2,
    )


class DecoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Attention (GQA): q has n_heads, k/v have n_kv_heads. Qwen2.5 biases q/k/v.
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head, bias=cfg.attention_bias)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.d_head, bias=cfg.attention_bias)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.d_head, bias=cfg.attention_bias)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.d_head, cfg.d_model, bias=False)

        # SwiGLU MLP.
        self.gate_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down_proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

        # Pre-norm: normalize before attention, and before the MLP.
        self.input_norm = nn.RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
        self.post_attn_norm = nn.RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)

    def forward(self, x, cos, sin, positions, cache: KVCache, layer, start, attn_mask=None):
        # NEED TO DO:
        # X Decide on design choice for attn_mask flag
        # X Adapt RoPE to decode layers
        # X Append KV to KVCache

        # x: (batch, seq_len, d_model). cos/sin: broadcastable to (batch, heads, seq_len, d_head).
        cfg = self.cfg
        B, S, _ = x.shape

        # Attention sub-layer
        residual = x
        h = apply_rms_norm(x, self.input_norm, cfg)

        # Project to q/k/v and split into heads. GQA: q gets n_heads, k/v get fewer (n_kv_heads).
        q = self.q_proj(h).view(B, S, cfg.n_heads, cfg.d_head).transpose(1, 2)
        k = self.k_proj(h).view(B, S, cfg.n_kv_heads, cfg.d_head).transpose(1, 2)
        v = self.v_proj(h).view(B, S, cfg.n_kv_heads, cfg.d_head).transpose(1, 2)

        # Inject position information into q and k by rotating them (RoPE).
        q = apply_rope(q, cos, sin, cfg, positions)
        k = apply_rope(k, cos, sin, cfg, positions)

        # SAVE TO KV CACHE
        cache.write(layer, start, k, v)
        # READ CACHE W/ NEW KV PAIR CURR TOKEN NEEDS TO ATTEND TO ITSELF
        k, v = cache.read(layer)

        # GQA: each kv head is shared by a group of query heads, so replicate kv heads to match.
        n_rep = cfg.n_heads // cfg.n_kv_heads
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)

        is_causal = q.shape[2] == k.shape[2]

        # Cache aware case
        if (attn_mask is None): 
            # number of query heads == entire seq_len
            # Occurs when query length = k length (instead of query length = 1)
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
        else:
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        # Causal scaled-dot-product attention (each token attends only to itself and the past).

        # Merge heads back to d_model and project out; add to the residual stream.
        attn = attn.transpose(1, 2).reshape(B, S, cfg.n_heads * cfg.d_head)
        x = residual + self.o_proj(attn)

        # --- MLP sub-layer (SwiGLU: the gate branch acts as a learned per-feature filter) ---
        residual = x
        h = apply_rms_norm(x, self.post_attn_norm, cfg)
        h = self.down_proj(apply_swiglu(self.gate_proj(h), self.up_proj(h), cfg))
        x = residual + h

        return x


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab, cfg.d_model)
        self.layers = nn.ModuleList(DecoderLayer(cfg) for _ in range(cfg.n_layers))
        self.norm = nn.RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab, bias=False)

        # Qwen2.5-1.5B ties the output projection to the input embedding matrix.
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

    def forward(self, input_ids, cache: KVCache):
        # input_ids: (batch, seq_len) token ids.
        cfg = self.cfg
        B, S = input_ids.shape

        start = cache.cur_len

        # Look up token embeddings -> the residual stream the layers refine.
        x = self.embed(input_ids)

        # Build RoPE cos/sin once for positions 0..S-1 (rotate-half layout, base=rope_theta).
        pos = torch.arange(start, start + S, device=x.device, dtype=torch.float32)
        positions = pos[None, :].expand(B, -1)
        if cfg.use_custom_kernels:
            cos = sin = None  # the Triton kernel computes angles in-register
        else:
            inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.d_head, 2, device=x.device,
                                                              dtype=torch.float32) / cfg.d_head))
            freqs = torch.outer(pos, inv_freq)          # (S, d_head/2)
            emb = torch.cat((freqs, freqs), dim=-1)     # half-frequencies duplicated
            cos = emb.cos()[None, None].to(x.dtype)     # broadcasts over batch/heads
            sin = emb.sin()[None, None].to(x.dtype)

        cache.cur_len += S

        # Run the decoder stack (causality dependent on prefill vs decode)
        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, positions, cache, i, start)

        # Final norm, then project to vocabulary logits
        # Only need to project the very last token position
        x = apply_rms_norm(x, self.norm, cfg) # x shape is (batch, seq_len, d_model)
        return self.lm_head(x[:, -1:, :])
