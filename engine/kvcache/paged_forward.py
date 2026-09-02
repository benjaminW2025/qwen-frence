"""Paged-KV forward pass for Qwen2.5-1.5B."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "baseline"))

import torch

from paged_decode_attention import paged_decode_attention, decode_attention_sdpa
from paged_prefill_attention import paged_prefill_attention
from naive_forward import apply_rms_norm, apply_rope

# Debug instrumentation (set PAGED_DEBUG=1). Off by default; safe to leave in place.
_DEBUG = bool(os.environ.get("PAGED_DEBUG"))

def build_rope(cfg, starts, S, device, dtype):
    """
    Per sequence RoPE
    """
    starts = torch.as_tensor(starts, device=device, dtype=torch.float32)     # (B,)
    offs = torch.arange(S, device=device, dtype=torch.float32)               # (S,)
    pos = starts[:, None] + offs[None, :]                                    # (B, S)

    inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.d_head, 2, device=device,
                                                      dtype=torch.float32) / cfg.d_head))
    freqs = pos[:, :, None] * inv_freq[None, None, :]        # (B, S, d_head/2)
    emb = torch.cat((freqs, freqs), dim=-1)                  # (B, S, d_head)
    cos = emb.cos()[:, None, :, :].to(dtype)                 # (B, 1, S, d_head)
    sin = emb.sin()[:, None, :, :].to(dtype)
    return cos, sin


def build_rope_from_positions(cfg, positions, dtype):
    """
    Graph-safe RoPE
    """
    device = positions.device
    pos = positions.to(torch.float32)                        # (B,)
    inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.d_head, 2, device=device,
                                                      dtype=torch.float32) / cfg.d_head))
    freqs = pos[:, None] * inv_freq[None, :]                 # (B, d_head/2)
    emb = torch.cat((freqs, freqs), dim=-1)                  # (B, d_head)
    cos = emb.cos()[:, None, None, :].to(dtype)             # (B, 1, 1, d_head)
    sin = emb.sin()[:, None, None, :].to(dtype)
    return cos, sin


def kernel_inputs(cache, device):
    """Pack the paged cache's Python block tables + lengths into the tensors the
    decode kernel expects: block_table (B, max_blocks) int32, seq_lens (B,) int32."""
    max_blocks = max(len(bt) for bt in cache.block_tables)
    padded = [bt + [0] * (max_blocks - len(bt)) for bt in cache.block_tables]
    block_table = torch.tensor(padded, device=device, dtype=torch.int32)
    seq_lens = torch.tensor(cache.cur_lens, device=device, dtype=torch.int32)
    return block_table, seq_lens

def paged_layer_forward(layer, x, cos, sin, positions, cache, layer_idx, starts,
                        block_table, seq_lens, is_prefill):
    """One decoder layer over the paged cache. `layer` is a baseline DecoderLayer
    (we call its submodules directly). Mirrors naive_forward's DecoderLayer.forward
    except for the write/attend section."""
    cfg = layer.cfg
    B, S, _ = x.shape
    group = cfg.n_heads // cfg.n_kv_heads

    # --- attention pre-norm + qkv projection + head split (same as baseline) ---
    residual = x
    h = apply_rms_norm(x, layer.input_norm, cfg)
    q = layer.q_proj(h).view(B, S, cfg.n_heads,    cfg.d_head).transpose(1, 2)   # (B, n_heads,    S, d)
    k = layer.k_proj(h).view(B, S, cfg.n_kv_heads, cfg.d_head).transpose(1, 2)   # (B, n_kv_heads, S, d)
    v = layer.v_proj(h).view(B, S, cfg.n_kv_heads, cfg.d_head).transpose(1, 2)

    # RoPE on q and k (v is not rotated).
    q = apply_rope(q, cos, sin, cfg, positions)
    k = apply_rope(k, cos, sin, cfg, positions)

    # Store k and v into KV cache
    cache.write(layer_idx, starts, k, v)

    # If prefill
    if (is_prefill):
        attn_out = paged_prefill_attention(q, k, v, group)
        # Run attention layer
    else:
        # First extract layer
        out = paged_decode_attention(q[:, :, 0, :], cache.k_pool[layer_idx],
                                     cache.v_pool[layer_idx],
                                     block_table, seq_lens) # (B, n_kv_heads, d_head)

        if _DEBUG:
            # Kernel vs an SDPA reference computed from the SAME pool -> isolates a
            # kernel bug (kernel NaN, ref finite) from bad cached data (both NaN).
            ref = decode_attention_sdpa(q[:, :, 0, :], cache.k_pool[layer_idx],
                                        cache.v_pool[layer_idx], block_table, seq_lens)
            k_fin = bool(torch.isfinite(out).all())
            r_fin = bool(torch.isfinite(ref).all())
            pool_fin = bool(torch.isfinite(cache.k_pool[layer_idx]).all()
                            and torch.isfinite(cache.v_pool[layer_idx]).all())
            err = (out.float() - ref.float()).abs().max().item() if (k_fin and r_fin) else float("nan")
            if (not k_fin) or (not r_fin) or (not pool_fin) or err > 5e-2:
                print(f"[decode L{layer_idx:>2}] kernel_finite={k_fin} ref_finite={r_fin} "
                      f"pool_finite={pool_fin} err={err:.4f} "
                      f"seq_lens={seq_lens.tolist()} block_table={block_table.tolist()}")

        attn_out = out[:, :, None, :] # (B, n_kv_heads, seq_len, d_head)

    # Merge heads
    attn = attn_out.transpose(1, 2).reshape(B, S, cfg.n_heads * cfg.d_head)
    x = residual + layer.o_proj(attn)

    # SwiGLU <- need to replace with Triton kernel
    residual = x
    h = apply_rms_norm(x, layer.post_attn_norm, cfg)
    h = layer.down_proj(torch.nn.functional.silu(layer.gate_proj(h)) * layer.up_proj(h))
    x = residual + h
    return x

# Wrap layerwise paged forward into model forward
@torch.no_grad()
def paged_forward(model, cache, input_ids):
    """input_ids: (B, S). Returns logits for the last position: (B, 1, vocab)."""
    cfg = model.cfg
    B, S = input_ids.shape
    device = input_ids.device
    is_prefill = S > 1

    # Positions BEFORE this step (per sequence) -- needed for RoPE and for write().
    starts = list(cache.cur_lens)                    # copy; allocate_block will advance cur_lens

    # Grow KV pool by max
    cache.allocate_block([S] * B)

    # Kernel inputs reflect the POST-allocate lengths (so decode attends to the new token too).
    block_table, seq_lens = kernel_inputs(cache, device)

    x = model.embed(input_ids)
    starts_tensor = torch.as_tensor(starts, device=device, dtype=torch.float32)
    positions = starts_tensor[:, None] + torch.arange(S, device=device, dtype=torch.float32)[None]
    cos, sin = ((None, None) if cfg.use_custom_kernels
                else build_rope(cfg, starts, S, device, x.dtype))

    for i, layer in enumerate(model.layers):
        x = paged_layer_forward(layer, x, cos, sin, positions, cache, i, starts,
                                block_table, seq_lens, is_prefill)

    x = apply_rms_norm(x, model.norm, cfg)
    return model.lm_head(x[:, -1:, :])
