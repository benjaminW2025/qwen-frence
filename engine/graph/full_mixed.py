"""One fixed-shape CUDA graph for a packed mixed model step with paged FA3."""

from __future__ import annotations

import torch

from naive_forward import apply_rms_norm
from piecewise_prefill import _AttentionBoundary


class FullMixedGraph:
    """Capture model, KV writes, varlen attention, and head as one replay.

    The incoming tensors may change addresses between scheduler steps. Copies
    into fixed graph inputs happen before replay; no per-layer copies or graph
    replays are needed between attention and projection.
    """

    def __init__(self, model, pool, ids, positions, slots, cu, context,
                 table, max_query, *, enable_packed_qkv_rope_cache=False,
                 enable_residual_rmsnorm=False, enable_swiglu_fusion=False):
        if not ids.is_cuda or ids.numel() < 1 or context.numel() < 1:
            raise ValueError("full mixed graph requires nonempty CUDA metadata")
        self.model, self.pool = model, pool
        self.max_query = max_query
        self.ids = ids.clone()
        self.positions = positions.clone()
        self.slots = slots.clone()
        self.cu = cu.clone()
        self.context = context.clone()
        self.table = table.clone()
        self.valid_tokens = torch.full((), ids.numel(), device=ids.device,
                                       dtype=torch.int32)
        tokens = ids.numel()
        cfg = model.cfg
        boundaries = [
            _AttentionBoundary(
                model, pool, i, tokens, self.positions, self.slots, self.valid_tokens,
                enable_packed_qkv_rope_cache, enable_residual_rmsnorm,
                enable_swiglu_fusion, capture_graph=False)
            for i in range(len(model.layers) + 1)
        ]
        boundaries[0].ids = self.ids
        self.boundaries = boundaries

        def run_model():
            from kernel_dispatch import fa3_paged_varlen_attention

            q, residual = boundaries[0].run_segment()
            for layer in range(len(model.layers)):
                query = q[0].transpose(0, 1).contiguous()
                attention = fa3_paged_varlen_attention(
                    query, pool.k_pool[layer], pool.v_pool[layer],
                    self.cu, self.table, self.context,
                    max_query_len=max_query)
                boundary = boundaries[layer + 1]
                boundary.residual = residual
                boundary.attention = attention.reshape(tokens, cfg.n_heads * cfg.d_head)
                result = boundary.run_segment()
                if layer + 1 < len(model.layers):
                    q, residual = result
                else:
                    x = result
            if not enable_residual_rmsnorm:
                x = apply_rms_norm(x, model.norm, cfg)
            return model.lm_head(x.index_select(0, self.cu[1:].long() - 1))

        current = torch.cuda.current_stream(ids.device)
        side = torch.cuda.Stream(device=ids.device)
        side.wait_stream(current)
        with torch.cuda.stream(side):
            for _ in range(3):
                run_model()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=side):
                self.logits = run_model()
        current.wait_stream(side)

    def compatible(self, ids, positions, slots, cu, context, table, max_query):
        sources = (ids, positions, slots, cu, context, table)
        targets = (self.ids, self.positions, self.slots, self.cu,
                   self.context, self.table)
        return max_query == self.max_query and all(
            source.shape == target.shape and source.dtype == target.dtype
            and source.device == target.device
            for source, target in zip(sources, targets))

    @torch.no_grad()
    def forward(self, ids, positions, slots, cu, context, table, max_query):
        if not self.compatible(ids, positions, slots, cu, context, table, max_query):
            return None
        for destination, source in ((self.ids, ids), (self.positions, positions),
                                    (self.slots, slots), (self.cu, cu),
                                    (self.context, context), (self.table, table)):
            destination.copy_(source)
        self.graph.replay()
        return self.logits
