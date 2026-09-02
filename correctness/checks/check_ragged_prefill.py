"""Correctness gates for packed ragged prefill and batched scheduler admission."""

from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace

import _bootstrap  # noqa: F401

import torch

from naive_forward import Qwen2Config
from packed_prefill_attention import packed_prefill_attention
from paged_engine import PagedEngine
from paged_kv_cache import PagedKVCache
from ragged_prefill import ragged_prefill
from scheduler import Request, Scheduler, Status
from iteration_plan import prefill_attention_pairs


DEVICE = "cuda"
DTYPE = torch.float16
LOGIT_ATOL = 5e-2
LOGIT_RTOL = 2e-3
CHUNK_EXECUTION_DRIFT_ATOL = 1e-1
# Layer 0 isolates projection, RoPE, and physical cache placement before attention
# differences can compound. Later layers use a separate sanity bound; end-to-end
# correctness is gated by logits/top-1 and the scheduler decode lifecycle below.
KV_PLACEMENT_ATOL = 3e-2
KV_DRIFT_ATOL = 1e-1
KV_RTOL = 2e-3
ATTENTION_CASES = (
    [1],
    [15, 16, 17],
    [31, 32, 33],
    [3, 65, 129, 257],
)


def synthetic_prompt(length, salt, vocab):
    return [((position * 104729 + salt * 8191) % (vocab - 1)) + 1 for position in range(length)]


def logical_kv(cache, layer, slot, length):
    table = cache.block_tables[slot]
    keys = cache.k_pool[layer][table].reshape(-1, cache.n_kv_heads, cache.d_head)[:length]
    values = cache.v_pool[layer][table].reshape(-1, cache.n_kv_heads, cache.d_head)[:length]
    return keys, values


def test_planner_contracts():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.block_size = 16
    scheduler.running = {}
    scheduler.free_slots = [0, 1, 2, 3]
    scheduler.max_prefill_batch_size = 4
    scheduler.waiting = deque(
        Request(index, [index + 1] * length, max_tokens=4)
        for index, length in enumerate((15, 16, 17, 33))
    )
    scheduler.cache = SimpleNamespace(free_blocks=list(range(12)))
    assert scheduler._planned_prefill_count() == 4

    scheduler.cache.free_blocks = list(range(6))
    assert scheduler._planned_prefill_count() == 3

    scheduler.max_prefill_batch_size = 2
    assert scheduler._planned_prefill_count() == 2

    scheduler.cache.free_blocks = []
    assert scheduler._planned_prefill_count() == 0

    try:
        Scheduler(None, None, 1, 1, 16, set(), device="cpu", max_prefill_batch_size=0)
    except ValueError:
        pass
    else:
        raise AssertionError("max_prefill_batch_size=0 must be rejected")

    try:
        Scheduler(None, None, 1, 1, 16, set(), device="cpu", max_prefill_chunk_size=0)
    except ValueError:
        pass
    else:
        raise AssertionError("max_prefill_chunk_size=0 must be rejected")

    try:
        Scheduler(
            None, None, 1, 1, 16, set(), device="cpu",
            max_prefill_attention_pairs=0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("max_prefill_attention_pairs=0 must be rejected")

    try:
        Scheduler(
            None, None, 4, 1, 16, set(), device="cpu",
            max_num_batched_tokens=3,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("the global token budget must fit every active decode")


def test_global_token_budget_planner():
    """Decode tokens are reserved first and prefills fill, but never exceed, the rest."""
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.block_size = 16
    scheduler.max_running = 4
    scheduler.max_num_batched_tokens = 10
    scheduler.max_prefill_batch_size = 4
    scheduler.max_prefill_chunk_size = 6
    scheduler.running = {
        0: Request(10, [1, 2], 8),
        1: Request(11, [3, 4], 8),
    }
    scheduler.prefilling = {2: Request(12, list(range(20)), 8)}
    scheduler.waiting = deque([Request(13, list(range(12)), 8)])
    scheduler.free_slots = [3]
    scheduler.cache = SimpleNamespace(
        free_blocks=list(range(16)),
        block_tables=[[0], [1], [], []],
    )
    scheduler.prefilling[2].slot = 2
    scheduler.prefilling[2].status = Status.PREFILLING

    slots, lengths = scheduler._plan_prefill(
        scheduler.max_num_batched_tokens - len(scheduler.running)
    )
    assert slots == [2, 3]
    assert lengths == [6, 2]
    assert sum(lengths) + len(scheduler.running) == scheduler.max_num_batched_tokens
    assert scheduler.prefilling[3].status is Status.PREFILLING
    assert not scheduler.waiting

    # A budget is a ceiling, not a fill target: dispatch all work if less is available.
    scheduler.waiting.clear()
    scheduler.prefilling = {2: Request(14, [1, 2, 3], 8)}
    scheduler.prefilling[2].slot = 2
    scheduler.prefilling[2].status = Status.PREFILLING
    slots, lengths = scheduler._plan_prefill(8)
    assert slots == [2]
    assert lengths == [3]


def test_prefix_aware_attention_work_planner():
    """Equal token chunks shrink as their cached prefix—and attention work—grows."""
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.block_size = 16
    scheduler.max_running = 2
    scheduler.max_num_batched_tokens = 10
    scheduler.max_prefill_batch_size = 2
    scheduler.max_prefill_chunk_size = None
    scheduler.max_prefill_attention_pairs = 75
    scheduler.running = {}
    scheduler.waiting = deque()
    scheduler.free_slots = []
    scheduler.cache = SimpleNamespace(free_blocks=[], block_tables=[[], []])

    fresh = Request(20, list(range(40)), 1)
    fresh.slot = 0
    fresh.status = Status.PREFILLING
    scheduler.prefilling = {0: fresh}
    assert scheduler._plan_prefill(10, 75) == ([0], [10])
    assert prefill_attention_pairs(0, 10) == 55

    resumed = Request(21, list(range(40)), 1)
    resumed.slot = 1
    resumed.status = Status.PREFILLING
    resumed.num_prompt_tokens_computed = 20
    scheduler.prefilling = {1: resumed}
    assert scheduler._plan_prefill(10, 75) == ([1], [3])
    assert prefill_attention_pairs(20, 3) == 66

    prefill_only = scheduler._plan_iteration()
    assert prefill_only.kind == "prefill_only"
    assert prefill_only.prefill_chunk_lengths == (10,)
    assert prefill_attention_pairs(20, 10) > scheduler.max_prefill_attention_pairs

    decode = Request(22, [1], 2)
    decode.slot = 0
    decode.status = Status.RUNNING
    decode.output_ids = [2]
    scheduler.running = {0: decode}
    plan = scheduler._plan_iteration()
    assert plan.decode_slots == (0,)
    assert plan.prefill_slots == (1,)
    assert plan.prefill_chunk_lengths == (3,)
    assert plan.total_tokens == 4


class FailingCache:
    """Minimal cache double for admission rollback contracts."""

    def __init__(self, slots, free_blocks=16):
        self.block_tables = [[] for _ in range(slots)]
        self.free_blocks = list(range(free_blocks))

    def occupy(self, slot):
        self.block_tables[slot].append(self.free_blocks.pop())

    def free(self, slot):
        self.free_blocks.extend(self.block_tables[slot])
        self.block_tables[slot] = []


def test_admission_rollback():
    """Model/kernel failures must leave queues, slots, and cache reusable."""
    requests = [Request(index, [index + 1] * 8, max_tokens=4) for index in range(2)]
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.block_size = 16
    scheduler.running = {}
    scheduler.prefilling = {}
    scheduler.free_slots = [0, 1]
    scheduler.max_prefill_batch_size = 2
    scheduler.waiting = deque(requests)
    scheduler.cache = FailingCache(2)
    scheduler.prefill_batch_sizes = []
    scheduler.prefill_batch_token_counts = []
    scheduler.max_prefill_chunk_size = None
    scheduler.eos_ids = set()

    def fail_batch(slots, _requests):
        for slot in slots:
            scheduler.cache.occupy(slot)
        raise RuntimeError("injected packed-prefill failure")

    scheduler._compute_prefill_batch = fail_batch
    try:
        scheduler._admit_batch()
    except RuntimeError:
        pass
    else:
        raise AssertionError("injected packed-prefill failure was not propagated")
    assert list(scheduler.waiting) == requests
    assert scheduler.free_slots == [0, 1]
    assert scheduler.running == {}
    assert scheduler.cache.block_tables == [[], []]
    assert all(req.status is Status.WAITING and req.slot is None for req in requests)

    scheduler.max_prefill_batch_size = 1

    def fail_serial(slots, _requests):
        scheduler.cache.occupy(slots[0])
        raise RuntimeError("injected serial-prefill failure")

    scheduler._compute_prefill_batch = fail_serial
    try:
        scheduler._admit()
    except RuntimeError:
        pass
    else:
        raise AssertionError("injected serial-prefill failure was not propagated")
    assert list(scheduler.waiting) == requests
    assert scheduler.free_slots == [0, 1]
    assert scheduler.running == {}
    assert scheduler.cache.block_tables == [[], []]
    assert all(req.status is Status.WAITING and req.slot is None for req in requests)


def test_resumable_request_lifecycle():
    """Intermediate chunks commit state but must never sample an output token."""
    requests = [Request(0, [1, 2, 3, 4, 5], 4), Request(1, [6, 7, 8], 4)]
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.prefilling = {0: requests[0], 1: requests[1]}
    scheduler.running = {}
    scheduler.finished = {}
    scheduler.free_slots = []
    scheduler.eos_ids = set()
    scheduler.max_prefill_chunk_size = 2
    scheduler.prefill_batch_sizes = []
    scheduler.prefill_batch_token_counts = []
    observed_starts = []
    for slot, req in scheduler.prefilling.items():
        req.slot = slot
        req.status = Status.PREFILLING

    def fake_compute(_slots, active_requests):
        starts = [req.num_prompt_tokens_computed for req in active_requests]
        lengths = [min(req.remaining_prompt_tokens, 2) for req in active_requests]
        observed_starts.append(starts)
        logits = torch.zeros(len(active_requests), 16)
        logits[:, 7] = 1
        return logits, lengths

    scheduler._compute_prefill_batch = fake_compute
    scheduler._prefill_region = lambda _name: nullcontext()

    scheduler._advance_prefill_batch()
    assert observed_starts == [[0, 0]]
    assert [req.num_prompt_tokens_computed for req in requests] == [2, 2]
    assert all(not req.output_ids for req in requests)
    assert not scheduler.running

    scheduler._advance_prefill_batch()
    assert observed_starts[-1] == [2, 2]
    assert requests[0].status is Status.PREFILLING
    assert requests[1].status is Status.RUNNING
    assert requests[0].output_ids == []
    assert requests[1].output_ids == [7]

    scheduler._advance_prefill_batch()
    assert observed_starts[-1] == [4]
    assert requests[0].status is Status.RUNNING
    assert requests[0].output_ids == [7]
    assert scheduler.prefilling == {}
    assert scheduler.prefill_batch_sizes == [2, 2, 1]
    assert scheduler.prefill_batch_token_counts == [4, 3, 1]


@torch.no_grad()
def check_packed_attention_case(lengths, dtype):
    """Compare one variable-length launch with the per-sequence SDPA reference."""
    torch.manual_seed(sum(lengths) + len(lengths) * 1009)
    total_tokens = sum(lengths)
    query_heads, kv_heads, d_head = 12, 2, 128
    group = query_heads // kv_heads

    # Match production's token-major projection followed by a head/token transpose.
    q = torch.randn(
        total_tokens, query_heads, d_head, device=DEVICE, dtype=dtype
    ).transpose(0, 1).unsqueeze(0)
    k = torch.randn(
        total_tokens, kv_heads, d_head, device=DEVICE, dtype=dtype
    ).transpose(0, 1).unsqueeze(0)
    v = torch.randn(
        total_tokens, kv_heads, d_head, device=DEVICE, dtype=dtype
    ).transpose(0, 1).unsqueeze(0)
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    cu_seqlens = torch.tensor(offsets, device=DEVICE, dtype=torch.int32)

    reference = packed_prefill_attention(
        q, k, v, cu_seqlens, max(lengths), group, backend="sdpa"
    )
    actual = packed_prefill_attention(
        q, k, v, cu_seqlens, max(lengths), group, backend="triton"
    )
    error = (actual.float() - reference.float()).abs().max().item()
    atol = 6e-2 if dtype == torch.float16 else 1.5e-1
    rtol = 2e-2 if dtype == torch.float16 else 4e-2
    torch.testing.assert_close(
        actual.float(), reference.float(), atol=atol, rtol=rtol
    )

    # Change every Q/K/V value in one sequence. Programs assigned to all other
    # sequences must remain bit-identical, proving offsets prevent leakage.
    if len(lengths) > 1:
        isolated_sequence = len(lengths) // 2
        start, end = offsets[isolated_sequence:isolated_sequence + 2]
        q_changed, k_changed, v_changed = q.clone(), k.clone(), v.clone()
        q_changed[:, :, start:end].add_(7)
        k_changed[:, :, start:end].mul_(-3)
        v_changed[:, :, start:end].add_(11)
        changed = packed_prefill_attention(
            q_changed,
            k_changed,
            v_changed,
            cu_seqlens,
            max(lengths),
            group,
            backend="triton",
        )
        outside = torch.ones(total_tokens, device=DEVICE, dtype=torch.bool)
        outside[start:end] = False
        assert torch.equal(actual[:, :, outside], changed[:, :, outside])

    return error


@torch.no_grad()
def reference_prefill(engine, prompt):
    cache = engine._new_cache(len(prompt))
    logits = engine.prefill(SimpleNamespace(prompt_ids=prompt), cache)
    return logits.clone(), cache


@torch.no_grad()
def check_case(engine, case_name, lengths, identical=False):
    prompts = [
        synthetic_prompt(length, 7 if identical else index + 1, engine.cfg.vocab)
        for index, length in enumerate(lengths)
    ]
    references = [reference_prefill(engine, prompt) for prompt in prompts]

    batch_size = len(prompts)
    slots = list(reversed(range(batch_size)))
    num_blocks = sum((length + engine.block_size - 1) // engine.block_size for length in lengths)
    cache = PagedKVCache(
        engine.cfg,
        batch_size,
        num_blocks,
        engine.block_size,
        DEVICE,
        DTYPE,
    )
    logits = ragged_prefill(engine.model, cache, slots, prompts)

    max_logit_error = 0.0
    max_kv_error = 0.0
    for row, (prompt, slot, (reference_logits, reference_cache)) in enumerate(
        zip(prompts, slots, references)
    ):
        logit_error = (logits[row].float() - reference_logits.float()).abs().max().item()
        max_logit_error = max(max_logit_error, logit_error)
        torch.testing.assert_close(
            logits[row].float(), reference_logits.float(), atol=LOGIT_ATOL, rtol=LOGIT_RTOL
        )
        assert logits[row].argmax().item() == reference_logits.argmax().item()

        for layer in range(engine.cfg.n_layers):
            got_k, got_v = logical_kv(cache, layer, slot, len(prompt))
            ref_k, ref_v = logical_kv(reference_cache, layer, 0, len(prompt))
            k_error = (got_k.float() - ref_k.float()).abs().max().item()
            v_error = (got_v.float() - ref_v.float()).abs().max().item()
            layer_error = max(k_error, v_error)
            max_kv_error = max(max_kv_error, layer_error)
            context = (
                f"case={case_name} row={row} slot={slot} prompt_length={len(prompt)} "
                f"layer={layer} k_max_abs={k_error:.6f} v_max_abs={v_error:.6f}. "
                "A layer-0 failure points to packed projection/RoPE or KV placement; "
                "a later-only failure can be accumulated attention/GEMM numerics."
            )
            tolerance = KV_PLACEMENT_ATOL if layer == 0 else KV_DRIFT_ATOL
            torch.testing.assert_close(
                got_k.float(), ref_k.float(), atol=tolerance, rtol=KV_RTOL, msg=context
            )
            torch.testing.assert_close(
                got_v.float(), ref_v.float(), atol=tolerance, rtol=KV_RTOL, msg=context
            )

    if identical:
        for row in range(1, batch_size):
            torch.testing.assert_close(logits[row], logits[0], atol=0, rtol=0)
        assert len({tuple(cache.block_tables[slot]) for slot in slots}) == batch_size
        # Identical sequences occupy distinct physical blocks but must produce
        # identical logical K/V at every layer. This catches scatter cross-talk
        # independently of packed-vs-serial FP16 execution-shape differences.
        reference_slot = slots[0]
        reference_length = len(prompts[0])
        for layer in range(engine.cfg.n_layers):
            expected_k, expected_v = logical_kv(
                cache, layer, reference_slot, reference_length
            )
            for slot in slots[1:]:
                actual_k, actual_v = logical_kv(cache, layer, slot, reference_length)
                torch.testing.assert_close(actual_k, expected_k, atol=0, rtol=0)
                torch.testing.assert_close(actual_v, expected_v, atol=0, rtol=0)

    return max_logit_error, max_kv_error


def _run_chunked_prefill(engine, prompts, slots, chunk_size, attention_backend):
    num_blocks = sum(
        (len(prompt) + engine.block_size - 1) // engine.block_size for prompt in prompts
    )
    cache = PagedKVCache(
        engine.cfg, len(prompts), num_blocks, engine.block_size, DEVICE, DTYPE
    )
    computed = [0] * len(prompts)
    final_logits = [None] * len(prompts)

    while any(done < len(prompt) for done, prompt in zip(computed, prompts)):
        rows = [
            row for row, (done, prompt) in enumerate(zip(computed, prompts))
            if done < len(prompt)
        ]
        active_slots = [slots[row] for row in rows]
        starts = [computed[row] for row in rows]
        chunks = [
            prompts[row][computed[row]:computed[row] + chunk_size] for row in rows
        ]
        logits = ragged_prefill(
            engine.model,
            cache,
            active_slots,
            chunks,
            prompt_starts=starts,
            attention_backend=attention_backend,
        )
        for result_row, request_row in enumerate(rows):
            computed[request_row] += len(chunks[result_row])
            if computed[request_row] == len(prompts[request_row]):
                final_logits[request_row] = logits[result_row]
    return final_logits, cache


@torch.no_grad()
def check_chunked_case(engine, lengths, chunk_size):
    """Separate chunk execution-shape drift from continuation-kernel parity."""
    prompts = [
        synthetic_prompt(length, index + 40, engine.cfg.vocab)
        for index, length in enumerate(lengths)
    ]
    references = [reference_prefill(engine, prompt) for prompt in prompts]
    slots = list(reversed(range(len(prompts))))
    chunked = {
        backend: _run_chunked_prefill(engine, prompts, slots, chunk_size, backend)
        for backend in ("sdpa", "triton")
    }

    execution_drift = {"sdpa": 0.0, "triton": 0.0}
    max_kv_error = 0.0
    for backend, (final_logits, cache) in chunked.items():
        for row, (prompt, slot, (reference_logits, reference_cache)) in enumerate(
            zip(prompts, slots, references)
        ):
            logits = final_logits[row]
            logit_error = (logits.float() - reference_logits.float()).abs().max().item()
            execution_drift[backend] = max(execution_drift[backend], logit_error)
            torch.testing.assert_close(
                logits.float(),
                reference_logits.float(),
                atol=CHUNK_EXECUTION_DRIFT_ATOL,
                rtol=LOGIT_RTOL,
                msg=(
                    f"backend={backend} row={row} prompt_length={len(prompt)}. "
                    "This comparison permits bounded FP16 drift from changing full-prompt "
                    "GEMMs into chunk-shaped GEMMs; top-1 and same-shape kernel parity "
                    "remain separate strict gates."
                ),
            )
            assert logits.argmax().item() == reference_logits.argmax().item()
            for layer in range(engine.cfg.n_layers):
                got_k, got_v = logical_kv(cache, layer, slot, len(prompt))
                ref_k, ref_v = logical_kv(reference_cache, layer, 0, len(prompt))
                layer_error = max(
                    (got_k.float() - ref_k.float()).abs().max().item(),
                    (got_v.float() - ref_v.float()).abs().max().item(),
                )
                max_kv_error = max(max_kv_error, layer_error)
                assert torch.isfinite(got_k).all() and torch.isfinite(got_v).all()
                if layer == 0:
                    context = (
                        f"full-vs-chunked layer-0 placement backend={backend} row={row} "
                        f"slot={slot} prompt_length={len(prompt)} "
                        f"k_max_abs={(got_k.float() - ref_k.float()).abs().max().item():.6f} "
                        f"v_max_abs={(got_v.float() - ref_v.float()).abs().max().item():.6f}. "
                        "Layer 0 independently gates QKV projection, absolute RoPE, and "
                        "logical-to-physical cache placement. Later-layer values are "
                        "diagnostic because chunk-shaped FP16 attention/GEMMs accumulate "
                        "different rounding from one-shot execution."
                    )
                    torch.testing.assert_close(
                        got_k.float(),
                        ref_k.float(),
                        atol=KV_PLACEMENT_ATOL,
                        rtol=KV_RTOL,
                        msg=context,
                    )
                    torch.testing.assert_close(
                        got_v.float(),
                        ref_v.float(),
                        atol=KV_PLACEMENT_ATOL,
                        rtol=KV_RTOL,
                        msg=context,
                    )

    sdpa_logits, sdpa_cache = chunked["sdpa"]
    triton_logits, triton_cache = chunked["triton"]
    kernel_logit_error = 0.0
    kernel_kv_error = 0.0
    for row, (prompt, slot) in enumerate(zip(prompts, slots)):
        logit_error = (
            triton_logits[row].float() - sdpa_logits[row].float()
        ).abs().max().item()
        kernel_logit_error = max(kernel_logit_error, logit_error)
        torch.testing.assert_close(
            triton_logits[row].float(),
            sdpa_logits[row].float(),
            atol=LOGIT_ATOL,
            rtol=LOGIT_RTOL,
            msg=f"same-chunk Triton-vs-SDPA logits row={row} length={len(prompt)}",
        )
        assert triton_logits[row].argmax().item() == sdpa_logits[row].argmax().item()
        for layer in range(engine.cfg.n_layers):
            triton_k, triton_v = logical_kv(triton_cache, layer, slot, len(prompt))
            sdpa_k, sdpa_v = logical_kv(sdpa_cache, layer, slot, len(prompt))
            layer_error = max(
                (triton_k.float() - sdpa_k.float()).abs().max().item(),
                (triton_v.float() - sdpa_v.float()).abs().max().item(),
            )
            kernel_kv_error = max(kernel_kv_error, layer_error)
            tolerance = KV_PLACEMENT_ATOL if layer == 0 else KV_DRIFT_ATOL
            context = (
                f"same-chunk Triton-vs-SDPA KV row={row} slot={slot} "
                f"prompt_length={len(prompt)} layer={layer} "
                f"k_max_abs={(triton_k.float() - sdpa_k.float()).abs().max().item():.6f} "
                f"v_max_abs={(triton_v.float() - sdpa_v.float()).abs().max().item():.6f}"
            )
            torch.testing.assert_close(
                triton_k.float(),
                sdpa_k.float(),
                atol=tolerance,
                rtol=KV_RTOL,
                msg=context,
            )
            torch.testing.assert_close(
                triton_v.float(),
                sdpa_v.float(),
                atol=tolerance,
                rtol=KV_RTOL,
                msg=context,
            )
    return execution_drift, max_kv_error, kernel_logit_error, kernel_kv_error


@torch.no_grad()
def test_scheduler_lifecycle(engine):
    prompts = [
        synthetic_prompt(length, index + 20, engine.cfg.vocab)
        for index, length in enumerate((31, 64, 127, 257))
    ]
    max_tokens = 4
    max_running = len(prompts)
    max_blocks = (max(len(prompt) for prompt in prompts) + max_tokens + 15) // 16
    num_blocks = max_running * max_blocks + max_running

    def run(
        prefill_batch_size,
        chunk_size=None,
        token_budget=4096,
        attention_pair_budget=None,
    ):
        scheduler = Scheduler(
            engine.model,
            engine.cfg,
            max_running,
            num_blocks,
            engine.block_size,
            set(),
            DEVICE,
            DTYPE,
            max_prefill_batch_size=prefill_batch_size,
            max_prefill_chunk_size=chunk_size,
            max_num_batched_tokens=token_budget,
            max_prefill_attention_pairs=attention_pair_budget,
        )
        assert not scheduler.use_graph, "mixed correctness must use eager execution"
        ids = [scheduler.add_request(prompt, max_tokens) for prompt in prompts]
        output = scheduler.run()
        return (
            [output[request_id] for request_id in ids],
            scheduler.prefill_batch_sizes,
            scheduler.prefill_batch_token_counts,
            scheduler.iteration_decode_token_counts,
            scheduler.iteration_prefill_token_counts,
            scheduler.iteration_prefill_attention_pairs,
            scheduler.iteration_token_counts,
            scheduler.iteration_kinds,
        )

    serial, serial_batches, _, _, _, _, _, _ = run(1)
    packed, packed_batches, _, _, _, _, _, _ = run(max_running)
    chunked, chunked_batches, chunked_tokens, _, _, _, _, _ = run(max_running, 64)
    mixed, _, mixed_prefill_tokens, mixed_decodes, mixed_prefills, _, mixed_totals, _ = run(
        max_running, token_budget=128
    )
    cost_aware, _, _, _, _, cost_pairs, _, cost_kinds = run(
        max_running, token_budget=4096, attention_pair_budget=4096
    )
    assert serial_batches == [1] * max_running, serial_batches
    assert packed_batches == [max_running], packed_batches
    assert all(len(tokens) == max_tokens for tokens in serial + packed + chunked)
    assert packed == serial, "packed prefill changed the generated lifecycle"
    assert chunked == serial, "chunked prefill changed the generated lifecycle"
    assert mixed == serial, "mixed prefill/decode changed the generated lifecycle"
    assert cost_aware == serial, "attention-work chunking changed generated tokens"
    assert sum(chunked_tokens) == sum(map(len, prompts))
    assert sum(mixed_prefill_tokens) == sum(map(len, prompts))
    assert all(total <= 128 for total in mixed_totals)
    assert any(decode and prefill for decode, prefill in zip(mixed_decodes, mixed_prefills))
    assert all(
        pairs <= 4096
        for kind, pairs in zip(cost_kinds, cost_pairs)
        if kind == "mixed"
    )
    return serial_batches, packed_batches, chunked_batches, chunked_tokens, mixed_totals


@torch.no_grad()
def test_staggered_arrival_lifecycle(engine):
    """A newly arrived prefill must share the next iteration with active decodes."""
    scheduler = Scheduler(
        engine.model,
        engine.cfg,
        max_running=2,
        num_blocks=8,
        block_size=engine.block_size,
        eos_ids=set(),
        device=DEVICE,
        dtype=DTYPE,
        max_num_batched_tokens=17,
        max_prefill_attention_pairs=75,
    )
    first_id = scheduler.add_request(synthetic_prompt(16, 90, engine.cfg.vocab), 3)
    scheduler.step()
    assert scheduler.iteration_kinds == ["prefill_only"]
    assert len(scheduler.running) == 1

    second_id = scheduler.add_request(synthetic_prompt(33, 91, engine.cfg.vocab), 3)
    scheduler.step()
    assert scheduler.iteration_kinds[-1] == "mixed"
    assert scheduler.iteration_decode_token_counts[-1] == 1
    assert scheduler.iteration_prefill_token_counts[-1] == 11
    assert scheduler.iteration_prefill_attention_pairs[-1] == 66
    assert scheduler.iteration_token_counts[-1] == 12
    second = next(req for req in scheduler.prefilling.values() if req.req_id == second_id)
    assert second.num_prompt_tokens_computed == 11
    assert second.next_kv_position == 11 and second.next_rope_position == 11

    outputs = scheduler.run()
    assert len(outputs[first_id]) == 3
    assert len(outputs[second_id]) == 3
    assert all(count <= 17 for count in scheduler.iteration_token_counts)


def main():
    print("prefill batch planner contracts...")
    test_planner_contracts()
    test_global_token_budget_planner()
    test_prefix_aware_attention_work_planner()
    test_admission_rollback()
    test_resumable_request_lifecycle()
    print("  PASS")

    assert torch.cuda.is_available(), "packed ragged prefill requires an NVIDIA GPU"
    print("packed variable-length attention vs per-sequence SDPA...")
    for dtype in (torch.float16, torch.bfloat16):
        for lengths in ATTENTION_CASES:
            error = check_packed_attention_case(lengths, dtype)
            print(f"  dtype={str(dtype):<14} lengths={str(lengths):<22} err={error:.6f}")
    print("  PASS")

    print("loading custom-kernel engine...")
    engine = PagedEngine(
        cfg=Qwen2Config(use_custom_kernels=True), device=DEVICE, dtype=DTYPE
    )

    eager_cases = [
        ("eager-rope-ragged", [15, 17, 33], False),
    ]
    custom_cases = [
        ("batch-one", [16], False),
        ("block-boundaries", [15, 16, 17], False),
        ("heterogeneous", [33, 128, 257, 512], False),
        ("slot-isolation", [64, 64, 64, 64], True),
    ]
    print("serial-reference logits and logical KV parity...")
    for use_custom_kernels, cases in ((False, eager_cases), (True, custom_cases)):
        engine.cfg.use_custom_kernels = use_custom_kernels
        for name, lengths, identical in cases:
            print(f"  checking {name:<18} lengths={lengths}...", flush=True)
            logit_error, kv_error = check_case(engine, name, lengths, identical)
            print(
                f"  {name:<18} lengths={str(lengths):<22} "
                f"logit_err={logit_error:.6f} kv_err={kv_error:.6f}"
            )
    print("  PASS")

    print("resumable packed prefill vs full serial reference...")
    execution_drift, kv_drift, kernel_logit_error, kernel_kv_error = (
        check_chunked_case(engine, [15, 33, 70], chunk_size=16)
    )
    print(
        "  chunks=16 lengths=[15, 33, 70] "
        f"full-vs-chunked-sdpa={execution_drift['sdpa']:.6f} "
        f"full-vs-chunked-triton={execution_drift['triton']:.6f}"
    )
    print(
        f"  same-chunk triton-vs-sdpa logits={kernel_logit_error:.6f} "
        f"kv={kernel_kv_error:.6f} full-kv-drift={kv_drift:.6f}"
    )
    print("  PASS")

    print("full serial-versus-packed scheduler lifecycle...")
    serial_batches, packed_batches, chunked_batches, chunked_tokens, mixed_totals = (
        test_scheduler_lifecycle(engine)
    )
    test_staggered_arrival_lifecycle(engine)
    print(f"  serial prefill batches: {serial_batches}")
    print(f"  packed prefill batches: {packed_batches}")
    print(f"  chunked prefill batches: {chunked_batches}")
    print(f"  chunked prefill tokens : {chunked_tokens}")
    print(f"  mixed iteration tokens : {mixed_totals}")
    print("  PASS")
    print("\nOVERALL: PASS")
    return True


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
