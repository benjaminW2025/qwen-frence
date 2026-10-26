"""Experimental FP8-stored decode cache with explicit FP16 FA3 staging.

This is NOT native FP8 attention. Persistent E4M3 cache values are dequantized
into the existing FP16 pools before each layer's FA3 call. The extra workspace,
conversion and traffic are deliberate and must be included in latency reports.
"""
from contextlib import contextmanager

import torch

from experiments.quant.projections import quantize_rows_reference


class StagedFP8KVCache:
    def __init__(self, pool, seed, mode="kv"):
        if mode not in ("k", "v", "kv"):
            raise ValueError("cache mode must be k, v or kv")
        self.entries = {}
        self.mode = mode
        self.expected_seed = []
        layers = len(pool.k_pool)
        tensors = pool.k_pool + pool.v_pool
        if len(seed) != len(tensors):
            raise ValueError("cache seed does not match pool")
        for index, (tensor, original) in enumerate(zip(tensors, seed)):
            selected = ("k" if index < layers else "v") in mode
            if selected:
                q, scale = quantize_rows_reference(original.reshape(-1, original.shape[-1]))
                entry = {"q": q.clone(), "scale": scale.clone(),
                         "seed_q": q, "seed_scale": scale,
                         "shape": tuple(tensor.shape)}
                self.entries[tensor.data_ptr()] = entry
                self.expected_seed.append(self.dequantized(entry))
            else:
                self.expected_seed.append(original)

    @staticmethod
    def dequantized(entry):
        return (entry["q"].float() * entry["scale"][:, None]).half().view(entry["shape"])

    def reset(self):
        for entry in self.entries.values():
            entry["q"].copy_(entry["seed_q"])
            entry["scale"].copy_(entry["seed_scale"])

    def storage_bytes(self):
        return sum(entry["q"].numel() * entry["q"].element_size()
                   + entry["scale"].numel() * entry["scale"].element_size()
                   for entry in self.entries.values())

    def stage(self, tensor, slots):
        entry = self.entries.get(tensor.data_ptr())
        if entry is not None:
            from experiments.quant.kernels.staged_cache import update_and_stage
            update_and_stage(tensor, slots, entry["q"], entry["scale"])

    @contextmanager
    def attention_hook(self):
        # Both eager dispatch and captured decode import this function through
        # kernel_dispatch. Graph capture records the conversion kernels too.
        import kernel_dispatch
        original = kernel_dispatch.fa3_paged_decode_attention

        def staged(q, k, v, table, lengths, **kwargs):
            position = lengths.to(torch.int64) - 1
            pages = table.gather(1, (position // k.shape[1])[:, None]).flatten().long()
            slots = pages * k.shape[1] + position % k.shape[1]
            self.stage(k, slots)
            self.stage(v, slots)
            return original(q, k, v, table, lengths, **kwargs)

        kernel_dispatch.fa3_paged_decode_attention = staged
        try:
            yield self
        finally:
            kernel_dispatch.fa3_paged_decode_attention = original
