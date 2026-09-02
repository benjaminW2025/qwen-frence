"""
CUDA-graph'd single-token decode.

Constraints of a graph:
- The graph captures fixed memory ADDRESSES
    - Inputs must be passed in through a fixed buffer (k_pool and v_pool already are)
    - Intermediaries can have variable memory addresses
    - Output must be passed out through a fixed buffer as well

Only decode is graphed
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "baseline"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "kvcache"))

import torch

from paged_decode_attention import (
    paged_decode_attention_dispatch,
    resolve_decode_attention_policy,
)
from paged_forward import build_rope_from_positions
from naive_forward import (
    apply_rms_norm,
    apply_rope,
    apply_swiglu,
)


class CUDAGraphDecoder:
    def __init__(self, model, cache, batch_size, max_blocks, device, dtype):
        self.model = model
        self.cache = cache
        self.B = batch_size
        self.max_blocks = max_blocks          # fixed block-table width baked into the graph
        self.device = device
        self.dtype = dtype

        # Static input buffers
        self.s_input_ids    = torch.zeros(batch_size, 1, dtype=torch.long,  device=device)
        self.s_positions    = torch.zeros(batch_size,    dtype=torch.int32, device=device)  # RoPE pos of the current token (= cached len before it)
        self.s_seq_lens     = torch.zeros(batch_size,    dtype=torch.int32, device=device)  # cached length AFTER this token (kernel reads 0..seq_len-1)
        self.s_block_table  = torch.zeros(batch_size, max_blocks, dtype=torch.int32, device=device)
        self.s_slot_mapping = torch.zeros(batch_size,    dtype=torch.long,  device=device)  # flat pool slot for the current token: pid*block_size + offset

        # Static output buffers
        self.s_logits = None
        self.graph = None

    def _step_forward(self):
        return graph_decode_forward(
            self.model, self.cache,
            self.s_input_ids, self.s_positions, self.s_seq_lens,
            self.s_block_table, self.s_slot_mapping,
        )

    def capture(self, warmup=3):
        """
        Warm up then record the graph
        """
        # Warm up on a side stream. The static buffers hold zeros here, so this writes
        # to pool slot 0 -- harmless as long as capture happens before real decoding.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup):
                self._step_forward()
        torch.cuda.current_stream().wait_stream(s)

        # Record. Everything launched inside is replayed verbatim by replay().
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.s_logits = self._step_forward()
        return self

    @torch.no_grad()
    def decode(self, input_ids, positions, seq_lens, block_table, slot_mapping):
        """
        Refresh the static buffers with this step's values, replay the
        graph, and return the logits buffer.
        """
        self.s_input_ids.copy_(input_ids)
        self.s_positions.copy_(positions)
        self.s_seq_lens.copy_(seq_lens)
        self.s_block_table.copy_(block_table)
        self.s_slot_mapping.copy_(slot_mapping)
        self.graph.replay()
        return self.s_logits        # (B, 1, vocab)


# Graph safe decode forwrad
def graph_decode_forward(model, cache, input_ids, positions, seq_lens,
                         block_table, slot_mapping, decode_attention_policy="production",
                         max_decode_context_length=None,
                         enable_regime_fusions=False):
    """
    Mirrors paged_forward's decode branch, with RoPE from a
    positions tensor and the KV write as a tensor scatter.
    """
    cfg = model.cfg
    B = input_ids.shape[0]
    S = 1
    n_kv, d_head = cfg.n_kv_heads, cfg.d_head

    x = model.embed(input_ids)                                       # (B, 1, d_model)
    cos, sin = ((None, None) if cfg.use_custom_kernels
                else build_rope_from_positions(cfg, positions, x.dtype))

    rope_positions = positions[:, None]
    effective_decode_attention_policy = resolve_decode_attention_policy(
        decode_attention_policy, max_decode_context_length
    )
    for i, layer in enumerate(model.layers):
        residual = x
        h = apply_rms_norm(x, layer.input_norm, cfg)
        q = layer.q_proj(h).view(B, S, cfg.n_heads, d_head).transpose(1, 2)   # (B, n_heads, 1, d)
        k = layer.k_proj(h).view(B, S, n_kv, d_head).transpose(1, 2)          # (B, n_kv,    1, d)
        v = layer.v_proj(h).view(B, S, n_kv, d_head).transpose(1, 2)
        # Native decode is (B, H, 1, D). The packed RoPE/KV fusion wins when its
        # inputs are already (1, H, T, D), as in mixed/resumed prefill, but adapting
        # decode required three materializing layout copies per layer. At short
        # contexts those copies outweighed the fused work by roughly 10%, so decode
        # deliberately retains its native-layout kernels.
        q = apply_rope(q, cos, sin, cfg, rope_positions)
        k = apply_rope(k, cos, sin, cfg, rope_positions)
        k_flat = cache.k_pool[i].view(-1, n_kv, d_head)
        v_flat = cache.v_pool[i].view(-1, n_kv, d_head)
        k_flat.index_copy_(0, slot_mapping, k[:, :, 0, :].contiguous())
        v_flat.index_copy_(0, slot_mapping, v[:, :, 0, :].contiguous())

        out = paged_decode_attention_dispatch(
            q[:, :, 0, :], cache.k_pool[i], cache.v_pool[i], block_table, seq_lens,
            policy=effective_decode_attention_policy,
            max_context_length=max_decode_context_length,
        )                                                                     # (B, n_heads, d)
        attn = out[:, :, None, :].transpose(1, 2).reshape(B, S, cfg.n_heads * d_head)
        x = residual + layer.o_proj(attn)

        residual = x
        h = apply_rms_norm(x, layer.post_attn_norm, cfg)
        h = layer.down_proj(
            apply_swiglu(
                layer.gate_proj(h),
                layer.up_proj(h),
                cfg,
                enable_regime_fusions=enable_regime_fusions,
            )
        )
        x = residual + h

    x = apply_rms_norm(x, model.norm, cfg)
    return model.lm_head(x[:, -1:, :])


def build_decode_step_inputs(cache, tokens, max_blocks, device):
    """
    Build the five per-step buffer values for a graphed decode step
    """
    B = cache.batch_size
    block_size = cache.block_size

    positions, seq_lens, slot_mapping = [], [], []
    for b in range(B):
        # Grab the length of each batch
        cur = cache.cur_lens[b]         
        p = cur - 1                             # absolute position of the current token
        positions.append(p) # Store for RoPE generation
        seq_lens.append(cur) # Keep track of sequence lengths
        block_id = cache.block_tables[b][p // block_size]   # which physical block holds it
        slot_mapping.append(block_id * block_size + (p % block_size))  # flat pool slot

    # block_table padded to the FIXED max_blocks
    padded = [bt + [0] * (max_blocks - len(bt)) for bt in cache.block_tables]

    input_ids    = torch.tensor(tokens,       device=device, dtype=torch.long).view(B, 1)
    positions    = torch.tensor(positions,    device=device, dtype=torch.int32)
    seq_lens     = torch.tensor(seq_lens,     device=device, dtype=torch.int32)
    block_table  = torch.tensor(padded,       device=device, dtype=torch.int32)
    slot_mapping = torch.tensor(slot_mapping, device=device, dtype=torch.long)
    return input_ids, positions, seq_lens, block_table, slot_mapping
