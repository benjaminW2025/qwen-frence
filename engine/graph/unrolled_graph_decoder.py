"""Fixed-regime multi-token CUDA graph decode.

This experimental decoder captures K autoregressive steps in one graph.  Its
validation mode retains every full-logit tensor; production mode retains only
the K sampled token rows.  It deliberately does not replace the flexible K1
decoder or scheduler path.
"""

from __future__ import annotations

import torch


def first_eos_positions(token_ids, eos_token_id):
    """Return the first EOS index per row, or K when EOS is absent.

    ``token_ids`` is ordered ``(K, B)``.  The function is intentionally pure so
    the C++ chunk-commit implementation can be checked against the same contract.
    """
    if token_ids.ndim != 2:
        raise ValueError("token IDs must have shape (steps, batch)")
    steps, batch = token_ids.shape
    if eos_token_id is None or eos_token_id < 0:
        return torch.full((batch,), steps, device=token_ids.device, dtype=torch.int64)
    matches = token_ids == eos_token_id
    indices = torch.arange(steps, device=token_ids.device, dtype=torch.int64)[:, None]
    sentinel = torch.full_like(indices, steps)
    return torch.where(matches, indices, sentinel).amin(dim=0)


class UnrolledCUDAGraphDecoder:
    """Capture K exact-metadata decode steps against one shared paged KV pool."""

    def __init__(
        self,
        model,
        cache,
        step_inputs,
        *,
        decode_attention_policy="splitk",
        max_decode_context_length=None,
        retain_logits=False,
        output_head_policy="logits",
        output_head_config=None,
        enable_residual_rmsnorm=False,
        enable_native_decode_rope_kv=False,
        enable_native_decode_qkv_postprocess=False,
        enable_packed_qkv_rope_cache=False,
    ):
        if not step_inputs:
            raise ValueError("unrolled decode requires at least one step")
        if retain_logits and output_head_policy != "logits":
            raise ValueError("logit-retaining validation requires the logits output head")
        batch = step_inputs[0][0].shape[0]
        if batch < 1:
            raise ValueError("unrolled decode requires a positive batch")
        normalized = []
        for positions, lengths, table, slots in step_inputs:
            if (positions.shape != (batch,) or lengths.shape != (batch,)
                    or slots.shape != (batch,) or table.ndim != 2
                    or table.shape[0] != batch):
                raise ValueError("every unrolled metadata row must match the fixed batch")
            normalized.append((positions, lengths, table, slots))
        self.model, self.cache = model, cache
        self.batch, self.steps = batch, len(normalized)
        self.decode_attention_policy = decode_attention_policy
        self.max_decode_context_length = max_decode_context_length
        self.retain_logits = retain_logits
        self.output_head_policy = output_head_policy
        self.output_head_config = output_head_config
        self.enable_residual_rmsnorm = bool(enable_residual_rmsnorm)
        self.enable_native_decode_rope_kv = bool(enable_native_decode_rope_kv)
        self.enable_native_decode_qkv_postprocess = bool(
            enable_native_decode_qkv_postprocess
        )
        self.enable_packed_qkv_rope_cache = bool(enable_packed_qkv_rope_cache)
        if sum((self.enable_native_decode_rope_kv,
                self.enable_native_decode_qkv_postprocess,
                self.enable_packed_qkv_rope_cache)) > 1:
            raise ValueError("select one QKV postprocessing mode")
        self.s_first_ids = torch.zeros((batch, 1), device=normalized[0][0].device,
                                       dtype=torch.long)
        # Clone metadata into graph-owned, fixed-address buffers.
        self.s_step_inputs = tuple(tuple(tensor.clone() for tensor in row)
                                   for row in normalized)
        self.graph = None
        self.s_tokens = None
        self.s_logits = None

    def _forward(self):
        from paged_graph_decoder import graph_decode_forward

        token = self.s_first_ids
        logits_kept = []
        token_rows = []
        for positions, lengths, table, slots in self.s_step_inputs:
            output = graph_decode_forward(
                self.model, self.cache, token, positions, lengths, table, slots,
                decode_attention_policy=self.decode_attention_policy,
                max_decode_context_length=self.max_decode_context_length,
                output_head_policy=self.output_head_policy,
                output_head_config=self.output_head_config,
                enable_residual_rmsnorm=self.enable_residual_rmsnorm,
                enable_native_decode_rope_kv=self.enable_native_decode_rope_kv,
                enable_native_decode_qkv_postprocess=(
                    self.enable_native_decode_qkv_postprocess
                ),
                enable_packed_qkv_rope_cache=self.enable_packed_qkv_rope_cache,
            )
            if self.output_head_policy == "logits":
                if self.retain_logits:
                    logits_kept.append(output)
                token = output.argmax(dim=-1).reshape(self.batch, 1)
            else:
                token = output.reshape(self.batch, 1)
            token_rows.append(token.reshape(self.batch))
        # Token storage is the production interface. Full logits are held only
        # by validation mode and never copied into an additional KxBxV tensor.
        tokens = torch.stack(token_rows, dim=0)
        return tokens, tuple(logits_kept)

    def capture(self, warmup=2):
        if warmup < 1:
            raise ValueError("unrolled graph warmup must be positive")
        current = torch.cuda.current_stream(self.s_first_ids.device)
        side = torch.cuda.Stream(device=self.s_first_ids.device)
        side.wait_stream(current)
        with torch.cuda.stream(side):
            for _ in range(warmup):
                self._forward()
        current.wait_stream(side)
        torch.cuda.synchronize(self.s_first_ids.device)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.s_tokens, self.s_logits = self._forward()
        return self

    @torch.no_grad()
    def replay(self, first_ids):
        if self.graph is None:
            raise RuntimeError("capture must run before replay")
        if first_ids.shape == (self.batch,):
            first_ids = first_ids[:, None]
        if first_ids.shape != (self.batch, 1):
            raise ValueError("first token IDs must match the fixed batch")
        self.s_first_ids.copy_(first_ids)
        self.graph.replay()
        return self.s_tokens, self.s_logits
