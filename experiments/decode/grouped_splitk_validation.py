"""Independent PyTorch reference and adversarial inputs for the decode ablation."""

from __future__ import annotations


def make_inputs(lengths, *, page_size=16, head_dim=128, dtype=None, device="cuda",
                seed=0, strided=False, poison_padding=False):
    import torch

    dtype = torch.float16 if dtype is None else dtype
    generator = torch.Generator(device=device).manual_seed(seed)
    counts = [(length + page_size - 1) // page_size for length in lengths]
    max_pages = max(1, max(counts))
    pages = max(1, sum(counts))
    q = torch.randn(len(lengths), 12, head_dim, dtype=dtype, device=device, generator=generator)
    k = torch.randn(pages, page_size, 2, head_dim, dtype=dtype, device=device, generator=generator)
    v = torch.randn(k.shape, dtype=dtype, device=device, generator=generator)
    table = torch.full((len(lengths), max_pages), -1, dtype=torch.int32, device=device)
    page_ids = torch.randperm(pages, device=device, generator=generator)
    cursor = 0
    for row, (length, count) in enumerate(zip(lengths, counts)):
        table[row, :count] = page_ids[cursor:cursor + count].to(torch.int32)
        if poison_padding and length % page_size:
            last_page = page_ids[cursor + count - 1]
            k[last_page, length % page_size:] = float("nan")
            v[last_page, length % page_size:] = float("nan")
        cursor += count
    seq_lens = torch.tensor(lengths, dtype=torch.int32, device=device)
    if strided:
        # Independent layouts: K stays contiguous, V/Q and metadata have gaps.
        def with_gaps(tensor):
            shape = list(tensor.shape)
            shape[-1] *= 2
            storage = torch.empty(shape, dtype=tensor.dtype, device=tensor.device)
            storage.fill_(float("nan") if tensor.is_floating_point() else -1)
            view = storage[..., ::2]
            view.copy_(tensor)
            return view
        q, v, table, seq_lens = map(with_gaps, (q, v, table, seq_lens))
    return q, k, v, table, seq_lens


def attention_reference(q, k_pool, v_pool, block_table, seq_lens, *, scale=None):
    """Gather valid tokens and compute dense FP32 attention; empty rows are zero."""
    import torch

    scale = q.shape[-1] ** -0.5 if scale is None else scale
    group = q.shape[1] // k_pool.shape[2]
    output = torch.zeros_like(q)
    for row, length in enumerate(seq_lens.tolist()):
        if length == 0:
            continue
        count = (length + k_pool.shape[1] - 1) // k_pool.shape[1]
        pages = block_table[row, :count].long()
        keys = k_pool[pages].flatten(0, 1)[:length].float().repeat_interleave(group, dim=1)
        values = v_pool[pages].flatten(0, 1)[:length].float().repeat_interleave(group, dim=1)
        scores = torch.einsum("hd,thd->ht", q[row].float(), keys) * scale
        result = torch.einsum("ht,thd->hd", scores.softmax(dim=-1), values)
        output[row] = result.to(q.dtype)
    return output


def correctness_cases(*, page_size=16, dtype=None, device="cuda", seed=0):
    lengths = [0, 1, page_size - 1, page_size, page_size + 1, 2 * page_size + 3, 5 * page_size + 1]
    yield "ragged_strided", make_inputs(
        lengths, page_size=page_size, dtype=dtype, device=device, seed=seed,
        strided=True, poison_padding=True,
    ), {}
    tensors = make_inputs([0, 1, page_size + 1], page_size=page_size, dtype=dtype, device=device)
    tensors[0].fill_(1)
    tensors[1].fill_(-1)
    tensors[2].fill_(1)
    yield "negative_scores_empty_splits", tensors, {"scale": 1.0}
    tensors = make_inputs([1024], page_size=page_size, dtype=dtype, device=device)
    tensors[0].zero_()
    tensors[1].zero_()
    tensors[2].fill_(128)
    yield "unnormalized_overflow", tensors, {}


def check_output(actual, expected):
    import torch

    # Check dtype/shape as well as values: K=1 must have the same contract as K>1.
    if actual.dtype != expected.dtype or actual.shape != expected.shape:
        raise AssertionError(f"output contract mismatch: {actual.shape}/{actual.dtype}")
    tolerance = 2e-2 if actual.dtype == torch.bfloat16 else 2e-3
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    return (actual.float() - expected.float()).abs().max().item()
