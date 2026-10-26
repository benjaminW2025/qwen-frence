"""Native paged FA3 FP8 baseline; no full-cache FP16 staging.

Uses the existing external vLLM FA3 extension. Q/K/V all use E4M3; descales
are FP32 [sequence, KV head], as required by that interface. The compiled
backend must actually contain the FP8/page16 specializations; never fall back.
"""
from contextlib import contextmanager

import torch


def quantize_query(q, kv_heads):
    """One scale per GQA group, computed from this step's query only."""
    if q.ndim != 3 or q.shape[1] % kv_heads:
        raise ValueError("expected [batch, query heads, dim] with valid GQA")
    grouped = q.float().reshape(q.shape[0], kv_heads, q.shape[1] // kv_heads, q.shape[2])
    scales = (grouped.abs().amax((2, 3)) / 448.).clamp_min(1e-12)
    encoded = (grouped / scales[:, :, None, None]).clamp(-448., 448.)
    return encoded.reshape_as(q).to(torch.float8_e4m3fn).contiguous(), scales.contiguous()


def call_native(q8, k8, v8, table, lengths, qs, ks, vs, *, num_splits=0, scale=None):
    from custom_kernels.paged_decode_fa3 import _load_fa3, _query_offsets
    result = _load_fa3()(
        q=q8, k=k8, v=v8, cu_seqlens_q=_query_offsets(q8.shape[0], q8.device),
        max_seqlen_q=1, max_seqlen_k=table.shape[1] * k8.shape[1],
        seqused_k=lengths, block_table=table,
        softmax_scale=q8.shape[-1] ** -.5 if scale is None else scale,
        causal=False, num_splits=num_splits, fa_version=3,
        q_descale=qs, k_descale=ks, v_descale=vs)
    result = result[0] if isinstance(result, tuple) else result
    # The native FP8 FA3 implementation returns BF16; retain our FP16 contract.
    return result.to(torch.float16)


class NativeFP8Cache:
    def __init__(self, pool, seed, case, headroom=1.25):
        if case["kind"] != "decode" or headroom < 1:
            raise ValueError("native cache requires pure decode and headroom >= 1")
        self.entries = {}
        self.case = case
        self.headroom = headroom
        table = case["table"].cpu()
        owners = torch.empty(pool.k_pool[0].shape[0], dtype=torch.long)
        for row in range(table.shape[0]):
            owners[table[row].long()] = row
        owners = owners.repeat_interleave(pool.block_size).to(pool.k_pool[0].device)
        for tensor, original in zip(pool.k_pool + pool.v_pool, seed):
            flat = original.flatten(0, 1)
            scales = []
            for row, history in enumerate(case["history"]):
                positions = torch.arange(len(history), device=tensor.device)
                slots = case["table"][row].long()[positions // pool.block_size] * pool.block_size + positions % pool.block_size
                maximum = flat.index_select(0, slots).float().abs().amax((0, 2))
                scales.append((maximum * headroom / 448.).clamp_min(1e-12))
            scales = torch.stack(scales).contiguous()
            storage = (flat.float() / scales[owners, :, None]).clamp(-448., 448.)
            storage = storage.to(torch.float8_e4m3fn).view_as(tensor).contiguous()
            self.entries[tensor.data_ptr()] = {
                "q": storage.clone(), "seed_q": storage, "scales": scales,
                "owners": owners}

    def reset(self):
        for entry in self.entries.values():
            entry["q"].copy_(entry["seed_q"])

    def storage_bytes(self):
        return sum(e["q"].numel() + e["scales"].numel() * 4 for e in self.entries.values())

    def unchanged_prefix(self):
        for e in self.entries.values():
            stored = e["q"].view(torch.uint8).flatten(0, 1)
            initial = e["seed_q"].view(torch.uint8).flatten(0, 1)
            outside = torch.ones(stored.shape[0], device=stored.device, dtype=torch.bool)
            outside[self.case["slots"]] = False
            if not torch.equal(stored[outside], initial[outside]):
                return False
        return True

    def stored_writes(self, pool):
        slots = self.case["slots"]
        values = []
        for tensor in pool.k_pool + pool.v_pool:
            e = self.entries[tensor.data_ptr()]
            q = e["q"].view(torch.uint8).flatten(0, 1).index_select(0, slots).view(torch.float8_e4m3fn)
            scales = e["scales"][e["owners"][slots]]
            values.append((q.float() * scales[..., None]).half())
        return torch.stack(values)

    def saturation(self, pool):
        fractions = []
        for tensor in pool.k_pool + pool.v_pool:
            e = self.entries[tensor.data_ptr()]
            slots = self.case["slots"]
            raw = tensor.flatten(0, 1).index_select(0, slots).float()
            limits = e["scales"][e["owners"][slots], :, None] * 448.
            fractions.append(float((raw.abs() > limits).float().mean()))
        layers = len(pool.k_pool)
        return {"k_by_layer": fractions[:layers], "v_by_layer": fractions[layers:]}

    @contextmanager
    def attention_hook(self):
        import kernel_dispatch
        original = kernel_dispatch.fa3_paged_decode_attention

        def native(q, k, v, table, lengths, **kwargs):
            from experiments.quant.kernels.staged_cache import update_fp8_only
            position = lengths.long() - 1
            pages = table.gather(1, (position // k.shape[1])[:, None]).flatten().long()
            slots = pages * k.shape[1] + position % k.shape[1]
            ke, ve = self.entries[k.data_ptr()], self.entries[v.data_ptr()]
            update_fp8_only(k, slots, ke["q"], ke["scales"])
            update_fp8_only(v, slots, ve["q"], ve["scales"])
            q8, qs = quantize_query(q, k.shape[2])
            return call_native(q8, ke["q"], ve["q"], table, lengths,
                               qs, ke["scales"], ve["scales"], **kwargs)

        kernel_dispatch.fa3_paged_decode_attention = native
        try:
            yield self
        finally:
            kernel_dispatch.fa3_paged_decode_attention = original


@torch.no_grad()
def smoke_native(device):
    """Probe this installed binary's page16/GQA/FP8 combination before loading weights."""
    q = torch.ones(2, 12, 128, device=device, dtype=torch.float16)
    q8, qs = quantize_query(q, 2)
    cache = torch.ones(4, 16, 2, 128, device=device, dtype=torch.float16).to(torch.float8_e4m3fn)
    table = torch.tensor([[3, 1], [2, 0]], dtype=torch.int32, device=device)
    lengths = torch.tensor([17, 31], dtype=torch.int32, device=device)
    scales = torch.ones(2, 2, device=device, dtype=torch.float32)
    out = call_native(q8, cache, cache, table, lengths, qs, scales, scales)
    torch.testing.assert_close(out, torch.ones_like(out), atol=.05, rtol=.02)
    return {"passed": True, "page_size": 16, "query_heads": 12, "kv_heads": 2,
            "head_dim": 128, "note": "constant-value kernel canary, not model quality certification"}
