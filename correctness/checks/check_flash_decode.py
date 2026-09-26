"""Numerical and live CUDA-graph checks for the independent attention candidate."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'custom_kernels'), str(ROOT / 'experiments/decode')]


def varlen_reference(q, k, v, cu, table, lengths):
    """Independent dense FP32 oracle with bottom-right causal alignment."""
    import torch
    result = torch.zeros_like(q)
    offsets = cu.tolist()
    for sequence, length in enumerate(lengths.tolist()):
        start, end = offsets[sequence:sequence + 2]
        if start == end or length == 0:
            continue
        pages = table[sequence, :(length + 15) // 16].long()
        keys = k[pages].flatten(0, 1)[:length].float().repeat_interleave(6, dim=1)
        values = v[pages].flatten(0, 1)[:length].float().repeat_interleave(6, dim=1)
        scores = torch.einsum('qhd,khd->hqk', q[start:end].float(), keys) * (128 ** -.5)
        query_positions = torch.arange(end - start, device=q.device) + length - (end - start)
        visible = torch.arange(length, device=q.device)[None, :] <= query_positions[:, None]
        scores.masked_fill_(~visible[None], float('-inf'))
        probabilities = scores.softmax(-1).nan_to_num(0.)
        result[start:end] = torch.einsum('hqk,khd->qhd', probabilities, values).to(q.dtype)
    return result


def variant_configs():
    """Every compiled tile/PV variant, scheduling mode and pipeline mode."""
    from itertools import product
    return [dict(tile_n=n, register_pv=registers, compact=compact,
                 overlap_qk=overlap, split_k=split)
            for n, registers, compact, overlap, split in product(
                (64, 128), (False, True), (False, True), (False, True), (1, 4))]


def check_worklist():
    import torch
    from paged_flash_decode import prepare_flash_worklist
    queries = [0, 1, 10, 11, 64, 255]
    cu = torch.tensor([0, *torch.tensor(queries).cumsum(0).tolist()], device='cuda', dtype=torch.int32)
    def expected_work(q):
        return [[seq, tile] for seq, length in enumerate(q) for tile in range((length * 6 + 63) // 64)]
    def check(work, q):
        expected = expected_work(q)
        rows = work.tolist()
        assert rows[:len(expected)] == expected, 'compact query tiles differ'
        assert all(row == [-1, -1] for row in rows[len(expected):]), 'invalid worklist padding'
    check(prepare_flash_worklist(cu, sum(queries)), queries)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        prepare_flash_worklist(cu, sum(queries))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            work = prepare_flash_worklist(cu, sum(queries))
    torch.cuda.current_stream().wait_stream(stream)
    changed = [0, 10, 1, 11, 64, 255]
    cu.copy_(torch.tensor([0, *torch.tensor(changed).cumsum(0).tolist()], device='cuda', dtype=torch.int32))
    graph.replay()
    check(work, changed)
    return dict(case='compact_worklist_mutated_offsets', tiles=len(expected_work(changed)))


def check_varlen():
    import torch
    from paged_flash_decode import flash_varlen, prepare_flash_worklist
    from grouped_splitk_validation import make_inputs, check_output
    lengths = [0, 4096, 21, 16, 64, 300, 2048]
    # Includes an all-masked query row (17 queries, 16 keys), empty sequence,
    # non-tile-aligned requests, resumed prefill and decode in one call.
    queries = [0, 1, 10, 17, 31, 256, 1]
    offsets = [0]
    for length in queries:
        offsets.append(offsets[-1] + length)
    rows = []
    for config in variant_configs():
        # Reset every candidate to nontrivial identical data; repeated mutations
        # must not underflow the cache to zero and make later checks vacuous.
        _, k, v, table, seq = make_inputs(lengths, seed=31, poison_padding=True)
        cu = torch.tensor(offsets, device='cuda', dtype=torch.int32)
        torch.manual_seed(29)
        q = torch.randn((sum(queries), 12, 128), device='cuda', dtype=torch.float16)
        reference = varlen_reference(q, k, v, cu, table, seq)
        def operation():
            return flash_varlen(q, k, v, cu, table, seq, max_query_len=max(queries), **config)
        check_output(operation(), reference)
        if config['compact']:
            # Also cover caller-managed reuse. Changed offsets below are tested
            # using the graph's rebuilding path, never this now-stale worklist.
            work = prepare_flash_worklist(cu, q.shape[0])
            check_output(flash_varlen(q, k, v, cu, table, seq, max_query_len=max(queries),
                                     worklist=work, **config), reference)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                operation()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = operation()
        torch.cuda.current_stream().wait_stream(side)
        # Change Q/K/V in place, preserving NaN poison in unused padding.
        q.mul_(-1)
        k.mul_(.75)
        v.mul_(-.5)
        # Relabel physical pages while retaining the same logical cache. This
        # changes the page table without bringing poisoned padding into live KV.
        k.copy_(k.flip(0))
        v.copy_(v.flip(0))
        table.copy_(torch.where(table >= 0, k.shape[0] - 1 - table, table))
        cu[3].add_(1)  # q=10 -> 11 crosses a packed M64 query-tile boundary.
        seq[1].sub_(1)
        reference = varlen_reference(q, k, v, cu, table, seq)
        graph.replay()
        error = check_output(output, reference)
        rows.append(dict(case='varlen_mutated_q_k_v_offsets_pages_lengths', config=config, max_abs=error))
    return rows


def run_checks():
    import torch
    from paged_flash_decode import flash_decode, flash_varlen
    from grouped_splitk_validation import attention_reference, correctness_cases, check_output, make_inputs

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; CPU import/compile checks are not kernel validation')
    rows = []
    for dtype in (torch.float16,):
        for name, tensors, options in correctness_cases(dtype=dtype, seed=20260914):
            if name == 'ragged_strided':
                try:
                    flash_decode(*tensors, split_k=1, **options)
                except ValueError as error:
                    if 'must be contiguous' not in str(error):
                        raise
                else:
                    raise AssertionError('strided KV/metadata must fail, not silently copy')
                # TMA requires contiguous KV; strided Q is supported directly.
                tensors = (tensors[0], *(t.contiguous() for t in tensors[1:]))
                name = 'ragged_strided_q_contiguous_kv'
            reference = attention_reference(*tensors, **options)
            for split in (1, 2, 4, 8, 16, 32):
                actual = flash_decode(*tensors, split_k=split, **options)
                error = check_output(actual, reference)
                rows.append(dict(case=name, dtype=str(dtype), split=split, max_abs=error))
        for batch in (8, 64):
            for context in (256, 2048, 4096):
                tensors = make_inputs([context] * batch, dtype=dtype, seed=batch + context)
                q, k, v, table, lengths = tensors
                expected = attention_reference(*tensors)
                check_output(flash_decode(*tensors), expected)
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(3):
                        flash_decode(*tensors)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        output = flash_decode(*tensors)
                torch.cuda.current_stream().wait_stream(side)
                graph.replay()
                check_output(output, expected)
                # Change every address-backed runtime input; do not recapture.
                q.mul_(-.75)
                k.add_(.125)
                v.mul_(-.5)
                table.copy_(table.roll(1, dims=0))
                lengths.sub_(torch.arange(batch, device=q.device, dtype=torch.int32) % 17)
                expected = attention_reference(*tensors)
                graph.replay()
                error = check_output(output, expected)
                rows.append(dict(case='mutated_graph_q_k_v_pages_lengths', dtype=str(dtype),
                                 batch=batch, context=context, max_abs=error))
    rows.extend(check_varlen())
    rows.append(check_worklist())
    print(f'independent attention: {len(rows)} numerical/graph checks passed', flush=True)
    return rows


if __name__ == '__main__':
    run_checks()
