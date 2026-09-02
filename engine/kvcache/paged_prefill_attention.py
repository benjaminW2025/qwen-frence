"""Prefill attention for the paged KV-cache path."""

import torch.nn.functional as F


def paged_prefill_attention(q, k, v, group, scale=None):
    if scale is None:
        scale = 1.0 / (q.shape[-1] ** 0.5)

    # GQA: each kv head is shared by `group` query heads.
    k = k.repeat_interleave(group, dim=1)
    v = v.repeat_interleave(group, dim=1)

    # Causal attention
    return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
