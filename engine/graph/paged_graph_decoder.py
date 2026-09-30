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
    def __init__(self, model, cache, batch_size, max_blocks, device, dtype,
                 decode_attention_policy="production", max_decode_context_length=None,
                 enable_residual_rmsnorm=False,
                 enable_native_decode_rope_kv=False,
                 enable_native_decode_qkv_postprocess=False,
                 enable_fused_qkv_rope_cache=False,
                 enable_packed_qkv_rope_cache=False,
                 output_head_policy="logits", output_head_config=None,
                 enable_stable_decode_table_cache=False,
                 enable_fused_gemm_epilogues=False, gemm_epilogue_intervention="all"):
        self.model = model
        self.cache = cache
        self.B = batch_size
        self.max_blocks = max_blocks          # fixed block-table width baked into the graph
        self.device = device
        self.dtype = dtype
        # A graph cannot re-decide anything on replay, so the attention policy is
        # resolved once here and its action is recorded into the captured kernels.
        # The context bound defaults to everything the fixed block table can address:
        # that is the longest step this graph will ever replay, and a policy resolved
        # for it stays valid for every shorter one.
        self.decode_attention_policy = decode_attention_policy
        if max_decode_context_length is None and decode_attention_policy != "production":
            max_decode_context_length = max_blocks * cache.block_size
        self.max_decode_context_length = max_decode_context_length
        self.enable_residual_rmsnorm = bool(enable_residual_rmsnorm)
        self.enable_native_decode_rope_kv = bool(enable_native_decode_rope_kv)
        self.enable_native_decode_qkv_postprocess = bool(
            enable_native_decode_qkv_postprocess
        )
        self.enable_fused_qkv_rope_cache = bool(enable_fused_qkv_rope_cache)
        self.enable_packed_qkv_rope_cache = bool(enable_packed_qkv_rope_cache)
        self.enable_stable_decode_table_cache = bool(enable_stable_decode_table_cache)
        self.enable_fused_gemm_epilogues = bool(enable_fused_gemm_epilogues)
        self.gemm_epilogue_intervention = gemm_epilogue_intervention
        self.output_head_policy = output_head_policy
        self.output_head_config = output_head_config
        if output_head_policy not in ("logits", "fused_argmax"):
            raise ValueError(f"unknown output-head policy: {output_head_policy}")
        postprocess_modes = sum((self.enable_native_decode_rope_kv,
                                 self.enable_native_decode_qkv_postprocess,
                                 self.enable_fused_qkv_rope_cache,
                                 self.enable_packed_qkv_rope_cache))
        if postprocess_modes > 1:
            raise ValueError("select one QKV postprocessing mode")

        # Static input buffers
        self.s_input_ids    = torch.zeros(batch_size, 1, dtype=torch.long,  device=device)
        self.s_positions    = torch.zeros(batch_size,    dtype=torch.int32, device=device)  # RoPE pos of the current token (= cached len before it)
        # Capture with one valid cached token. Some external attention backends do
        # not promise a useful launch for an all-empty batch; replay still replaces
        # this tensor with the live sequence lengths before executing the graph.
        self.s_seq_lens     = torch.ones(batch_size,     dtype=torch.int32, device=device)  # cached length AFTER this token (kernel reads 0..seq_len-1)
        self.s_block_table  = torch.zeros(batch_size, max_blocks, dtype=torch.int32, device=device)
        self.s_slot_mapping = torch.zeros(batch_size,    dtype=torch.long,  device=device)  # flat pool slot for the current token: pid*block_size + offset

        # Static output buffers
        self.s_logits = None
        self.graph = None
        self._block_table_source_signature = None

    def _step_forward(self):
        if self.enable_fused_gemm_epilogues and self.gemm_epilogue_intervention == "all":
            return fused_graph_decode_forward(
                self.model, self.cache,
                self.s_input_ids, self.s_positions, self.s_seq_lens,
                self.s_block_table, self.s_slot_mapping,
                decode_attention_policy=self.decode_attention_policy,
                max_decode_context_length=self.max_decode_context_length,
                output_head_policy=self.output_head_policy,
                output_head_config=self.output_head_config,
            )
        return graph_decode_forward(
            self.model, self.cache,
            self.s_input_ids, self.s_positions, self.s_seq_lens,
            self.s_block_table, self.s_slot_mapping,
            decode_attention_policy=self.decode_attention_policy,
            max_decode_context_length=self.max_decode_context_length,
            enable_residual_rmsnorm=self.enable_residual_rmsnorm,
            enable_native_decode_rope_kv=self.enable_native_decode_rope_kv,
            enable_native_decode_qkv_postprocess=(
                self.enable_native_decode_qkv_postprocess
            ),
            enable_fused_qkv_rope_cache=self.enable_fused_qkv_rope_cache,
            enable_packed_qkv_rope_cache=self.enable_packed_qkv_rope_cache,
            output_head_policy=self.output_head_policy,
            output_head_config=self.output_head_config,
            gemm_epilogue_intervention=(self.gemm_epilogue_intervention
                                       if self.enable_fused_gemm_epilogues else None),
        )

    @torch.no_grad()
    def capture(self, warmup=3):
        """
        Warm up then record the graph
        """
        # Warm up on a side stream. Static metadata describes one token in block zero
        # and writes pool slot zero -- harmless because capture precedes real decoding.
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
        # The C++ stable-cohort path retains one fixed-address, immutable full
        # page table. Copy it only when the source storage or contents change.
        # A reused metadata buffer can hold a different cohort after prefill;
        # tensor version detects an in-place rewrite at the same address.
        if self.enable_stable_decode_table_cache:
            table_signature = (
                block_table.data_ptr(), tuple(block_table.shape),
                tuple(block_table.stride()), block_table._version,
            )
        else:
            table_signature = None
        if not self.enable_stable_decode_table_cache or table_signature != self._block_table_source_signature:
            self.s_block_table.copy_(block_table)
            self._block_table_source_signature = table_signature
        self.s_slot_mapping.copy_(slot_mapping)
        self.graph.replay()
        return self.s_logits


# Graph safe decode forwrad
def graph_decode_forward(model, cache, input_ids, positions, seq_lens,
                         block_table, slot_mapping, decode_attention_policy="production",
                         max_decode_context_length=None,
                         enable_regime_fusions=False,
                         enable_native_decode_rope_kv=False,
                         enable_native_decode_qkv_postprocess=False,
                         enable_fused_qkv_rope_cache=False,
                         enable_packed_qkv_rope_cache=False,
                         enable_residual_rmsnorm=False,
                         output_head_policy="logits",
                         output_head_config=None,
                         layer_observer=None, gemm_epilogue_intervention=None):
    """
    Mirrors paged_forward's decode branch, with RoPE from a
    positions tensor and the KV write as a tensor scatter.
    """
    cfg = model.cfg
    fused = None
    ops = frozenset()
    gate_weights = None
    if gemm_epilogue_intervention is not None:
        from kernel_dispatch import _load
        fused = _load("fused_gemm")
        ops = fused.intervention_ops(gemm_epilogue_intervention)
        if "gate-up" in ops:
            gate_weights = fused.prepare_gate_only(model)
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
    normalized = (apply_rms_norm(x, model.layers[0].input_norm, cfg)
                  if enable_residual_rmsnorm else None)
    for i, layer in enumerate(model.layers):
        residual = x
        h = (normalized if enable_residual_rmsnorm
             else apply_rms_norm(x, layer.input_norm, cfg))
        direct_q = "qkv" in ops or enable_fused_qkv_rope_cache or enable_packed_qkv_rope_cache
        if "qkv" in ops:
            if layer.qkv_proj is None:
                raise ValueError("QKV epilogue requires packed QKV weights")
            q = fused.qkv_rope_cache(
                h.reshape(B, cfg.d_model), layer.qkv_proj.weight, layer.qkv_proj.bias, None,
                positions=positions.to(torch.long), slots=slot_mapping,
                k_pool=cache.k_pool[i], v_pool=cache.v_pool[i], theta=cfg.rope_theta)
        elif enable_fused_qkv_rope_cache:
            from kernel_dispatch import fused_qkv_rope_cache
            if layer.qkv_proj is None:
                raise ValueError("full QKV fusion requires packed QKV weights")
            q = fused_qkv_rope_cache(
                h, layer.qkv_proj.weight, layer.qkv_proj.bias,
                positions, slot_mapping, cache.k_pool[i], cache.v_pool[i],
                base=cfg.rope_theta,
            )
        elif enable_packed_qkv_rope_cache:
            from kernel_dispatch import packed_qkv_rope_cache
            if layer.qkv_proj is None:
                raise ValueError("packed QKV epilogue requires packed QKV weights")
            q = packed_qkv_rope_cache(
                layer.qkv_proj(h), positions, slot_mapping,
                cache.k_pool[i], cache.v_pool[i], base=cfg.rope_theta,
            )
        else:
            q, k, v = layer.project_qkv(h)
            q = q.view(B, S, cfg.n_heads, d_head).transpose(1, 2)
            k = k.view(B, S, n_kv, d_head).transpose(1, 2)
            v = v.view(B, S, n_kv, d_head).transpose(1, 2)
        if direct_q:
            pass
        elif enable_native_decode_qkv_postprocess:
            from kernel_dispatch import native_decode_rope_kv_write
            q = native_decode_rope_kv_write(
                q, k, v, positions, slot_mapping,
                cache.k_pool[i], cache.v_pool[i],
                base=cfg.rope_theta, rotate_q=True,
            )
        elif enable_native_decode_rope_kv:
            from kernel_dispatch import native_decode_rope_kv_write
            # The fused Q rotation changed one FP16 value at layer zero in the
            # B=8, L=512 model check; that difference amplified across layers.
            # Keep the exact production Q path while fusing K RoPE and KV writes.
            q = apply_rope(q, cos, sin, cfg, rope_positions)
            native_decode_rope_kv_write(
                q, k, v, positions, slot_mapping, cache.k_pool[i], cache.v_pool[i],
                base=cfg.rope_theta, rotate_q=False,
            )
        else:
            # Adapting native decode to the packed-layout fusion materialized
            # three copies per layer and regressed. Keep that path unchanged.
            q = apply_rope(q, cos, sin, cfg, rope_positions)
            k = apply_rope(k, cos, sin, cfg, rope_positions)
            k_flat = cache.k_pool[i].view(-1, n_kv, d_head)
            v_flat = cache.v_pool[i].view(-1, n_kv, d_head)
            k_flat.index_copy_(0, slot_mapping, k[:, :, 0, :].contiguous())
            v_flat.index_copy_(0, slot_mapping, v[:, :, 0, :].contiguous())

        if layer_observer is not None:
            observed_q = (q[:, :, None, :]
                          if direct_q else q)
            layer_observer(i, "rope_kv", observed_q,
                           cache.k_pool[i].view(-1, n_kv, d_head).index_select(0, slot_mapping),
                           cache.v_pool[i].view(-1, n_kv, d_head).index_select(0, slot_mapping))

        query = (q if direct_q
                 else q[:, :, 0, :])
        out = paged_decode_attention_dispatch(
            query, cache.k_pool[i], cache.v_pool[i], block_table, seq_lens,
            policy=effective_decode_attention_policy,
            max_context_length=max_decode_context_length,
        )                                                                     # (B, n_heads, d)
        attn = out[:, :, None, :].transpose(1, 2).reshape(B, S, cfg.n_heads * d_head)
        if "residual-o" in ops:
            x, _ = fused.residual_gemm(attn.reshape(B, cfg.d_model), layer.o_proj.weight,
                                       residual.reshape(B, cfg.d_model), partials=False)
            x = x.view(B, S, cfg.d_model)
            h = apply_rms_norm(x, layer.post_attn_norm, cfg)
        elif enable_residual_rmsnorm:
            from kernel_dispatch import residual_add_rms_norm
            projected = layer.o_proj(attn)
            x, h = residual_add_rms_norm(
                residual, projected, layer.post_attn_norm.weight,
                cfg.rms_norm_eps,
            )
        else:
            projected = layer.o_proj(attn)
            x = residual + projected
            h = apply_rms_norm(x, layer.post_attn_norm, cfg)

        residual = x
        if "gate-up" in ops:
            activation = fused.gate_up_swiglu(h.reshape(B, cfg.d_model), gate_weights[i]).view(B, S, cfg.d_ff)
        else:
            gate, up = layer.project_gate_up(h)
            activation = apply_swiglu(gate, up, cfg, enable_regime_fusions=enable_regime_fusions)
        if "residual-down" in ops:
            x, _ = fused.residual_gemm(activation.reshape(B, cfg.d_ff), layer.down_proj.weight,
                                       residual.reshape(B, cfg.d_model), partials=False)
            x = x.view(B, S, cfg.d_model)
            if enable_residual_rmsnorm:
                next_norm = (model.layers[i + 1].input_norm
                             if i + 1 < len(model.layers) else model.norm)
                normalized = apply_rms_norm(x, next_norm, cfg)
        elif enable_residual_rmsnorm:
            branch = layer.down_proj(activation)
            from kernel_dispatch import residual_add_rms_norm
            next_norm = (model.layers[i + 1].input_norm
                         if i + 1 < len(model.layers) else model.norm)
            x, normalized = residual_add_rms_norm(
                residual, branch, next_norm.weight, cfg.rms_norm_eps,
            )
        else:
            branch = layer.down_proj(activation)
            x = residual + branch
        if layer_observer is not None:
            layer_observer(i, "layer_output", x)

    x = normalized if enable_residual_rmsnorm else apply_rms_norm(x, model.norm, cfg)
    if output_head_policy == "fused_argmax":
        from kernel_dispatch import fused_lm_head_argmax
        return fused_lm_head_argmax(
            x[:, -1, :].contiguous(), model.lm_head.weight,
            **(output_head_config or {}),
        )
    if output_head_policy != "logits":
        raise ValueError(f"unknown output-head policy: {output_head_policy}")
    return model.lm_head(x[:, -1:, :])


def fused_graph_decode_forward(model, cache, input_ids, positions, seq_lens,
                               block_table, slot_mapping, decode_attention_policy="production",
                               max_decode_context_length=None, output_head_policy="logits",
                               output_head_config=None):
    """Decode with the fused GEMM epilogues: per layer, attention plus four GEMMs.

    QKV+bias+RoPE+cache write, o_proj+residual, gate/up+SwiGLU, down+residual;
    both RMSNorms are folded into the GEMMs (custom_kernels/fused_gemm.py). The
    QKV/postprocess flags of graph_decode_forward do not apply here.
    """
    from kernel_dispatch import _load
    fused = _load("fused_gemm")
    weights = fused.prepare_model(model)
    cfg = model.cfg
    batch = input_ids.shape[0]
    policy = resolve_decode_attention_policy(decode_attention_policy, max_decode_context_length)
    x = model.embed(input_ids.view(batch))                           # (B, d_model)
    partials = fused.row_square_partials(x)
    rope_positions = positions.to(torch.long)  # the epilogue reads int64 positions
    for i, layer in enumerate(weights):
        q = fused.attention_inputs(layer, x, partials, cfg, positions=rope_positions,
                                   slots=slot_mapping, k_pool=cache.k_pool[i],
                                   v_pool=cache.v_pool[i])            # (B, n_heads, d)
        out = paged_decode_attention_dispatch(
            q, cache.k_pool[i], cache.v_pool[i], block_table, seq_lens,
            policy=policy, max_context_length=max_decode_context_length,
        )
        x, partials = fused.layer_tail(
            layer, out.reshape(batch, cfg.n_heads * cfg.d_head).contiguous(), x, cfg)
    if output_head_policy == "fused_argmax":
        from kernel_dispatch import fused_lm_head_argmax
        # The final norm runs in the head's input load (ledger D16).
        return fused_lm_head_argmax(
            x, model.lm_head.weight, norm_weight=model.norm.weight,
            row_partials=partials, epsilon=cfg.rms_norm_eps,
            **(output_head_config or {}),
        )
    if output_head_policy != "logits":
        raise ValueError(f"unknown output-head policy: {output_head_policy}")
    return model.lm_head(apply_rms_norm(x, model.norm, cfg)[:, None, :])


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
