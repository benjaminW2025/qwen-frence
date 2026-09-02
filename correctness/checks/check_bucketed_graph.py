"""
Comprehensive correctness harness for bucketed CUDA-graph decode.

This checks three separate contracts:
  1. Bucket construction, selection, padding, trimming, and invalid inputs.
  2. Every active batch size against eager graph_decode_forward, using the exact
     padded tensors consumed by the selected CUDA graph (teacher-forced parity).
  3. Full Scheduler integration while requests finish at different times, forcing
     the active batch through every size and therefore every bucket transition.

Run directly on a CUDA machine:
    python3 inference-engine/correctness/checks/check_bucketed_graph.py

Assertions are intentional: any failure produces a non-zero process exit code.
"""

import _bootstrap  # noqa: F401

import torch

from bucketed_graph_decoder import BucketedGraphDecoder, default_buckets
from naive_forward import Qwen2Config
from paged_engine import PagedEngine
from paged_graph_decoder import graph_decode_forward
from scheduler import Scheduler


DEVICE = "cuda"
DTYPE = torch.float16
MAX_RUNNING = 4
LOGIT_ATOL = 1e-2
LOGIT_RTOL = 1e-3

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Once upon a time, in a land far away,",
    "The three laws of thermodynamics are",
]


class RecordingDecoder:
    """CPU stand-in that records the tensors dispatched to a captured graph."""

    def __init__(self, bucket, max_blocks):
        self.bucket = bucket
        self.max_blocks = max_blocks
        self.calls = []

    def decode(self, *inputs):
        self.calls.append(inputs)
        assert all(t.shape[0] == self.bucket for t in inputs)
        assert inputs[3].shape == (self.bucket, self.max_blocks)
        return torch.arange(self.bucket * 3).reshape(self.bucket, 1, 3)


def make_recording_bucket_decoder(max_running=7, max_blocks=5):
    """Construct without __init__ so this contract test does not capture CUDA graphs."""
    decoder = BucketedGraphDecoder.__new__(BucketedGraphDecoder)
    decoder.max_blocks = max_blocks
    decoder.buckets = default_buckets(max_running)
    decoder.decoders = {
        bucket: RecordingDecoder(bucket, max_blocks) for bucket in decoder.buckets
    }
    return decoder


def assert_raises(exc_type, message_fragment, fn):
    try:
        fn()
    except exc_type as exc:
        assert message_fragment in str(exc), (message_fragment, str(exc))
    else:
        raise AssertionError(f"expected {exc_type.__name__}: {message_fragment}")


def test_bucket_and_padding_contracts():
    expected = {
        1: [1],
        2: [1, 2],
        3: [1, 2, 3],
        4: [1, 2, 4],
        5: [1, 2, 4, 5],
        7: [1, 2, 4, 7],
        8: [1, 2, 4, 8],
    }
    for max_running, buckets in expected.items():
        assert default_buckets(max_running) == buckets

    decoder = make_recording_bucket_decoder()
    max_blocks = decoder.max_blocks

    # Exercise every active batch size, including exact buckets and in-between sizes,
    # at minimal, intermediate, and already-full block-table widths.
    for n in range(1, 8):
        expected_bucket = next(bucket for bucket in decoder.buckets if bucket >= n)
        assert decoder._pick_bucket(n) == expected_bucket

        for width in (1, 3, max_blocks):
            input_ids = torch.arange(10, 10 + n).view(n, 1)
            positions = torch.arange(20, 20 + n, dtype=torch.int32)
            seq_lens = torch.arange(30, 30 + n, dtype=torch.int32)
            block_table = torch.arange(1, n * width + 1, dtype=torch.int32).view(n, width)
            slot_mapping = torch.arange(40, 40 + n)

            selected = decoder.decoders[expected_bucket]
            calls_before = len(selected.calls)
            output = decoder.decode(
                input_ids, positions, seq_lens, block_table, slot_mapping
            )
            assert len(selected.calls) == calls_before + 1

            seen = selected.calls[-1]
            for original, padded in zip(
                (input_ids, positions, seq_lens, slot_mapping),
                (seen[0], seen[1], seen[2], seen[4]),
            ):
                torch.testing.assert_close(padded[:n], original)
                if expected_bucket > n:
                    expected_padding = original[:1].expand(
                        (expected_bucket - n,) + original.shape[1:]
                    )
                    torch.testing.assert_close(padded[n:], expected_padding)

            torch.testing.assert_close(seen[3][:n, :width], block_table)
            assert torch.count_nonzero(seen[3][:, width:]).item() == 0
            if expected_bucket > n:
                expected_row = seen[3][:1].expand(expected_bucket - n, max_blocks)
                torch.testing.assert_close(seen[3][n:], expected_row)

            expected_output = torch.arange(expected_bucket * 3).reshape(
                expected_bucket, 1, 3
            )[:n]
            torch.testing.assert_close(output, expected_output)

    empty = (
        torch.empty(0, 1, dtype=torch.long),
        torch.empty(0, dtype=torch.int32),
        torch.empty(0, dtype=torch.int32),
        torch.empty(0, 1, dtype=torch.int32),
        torch.empty(0, dtype=torch.long),
    )
    assert_raises(ValueError, "empty batch", lambda: decoder.decode(*empty))

    too_large = tuple(torch.zeros((8,) + t.shape[1:], dtype=t.dtype) for t in empty)
    assert_raises(ValueError, "exceeds largest bucket", lambda: decoder.decode(*too_large))

    valid = (
        torch.zeros(1, 1, dtype=torch.long),
        torch.zeros(1, dtype=torch.int32),
        torch.ones(1, dtype=torch.int32),
        torch.zeros(1, max_blocks + 1, dtype=torch.int32),
        torch.zeros(1, dtype=torch.long),
    )
    assert_raises(ValueError, "exceeds captured width", lambda: decoder.decode(*valid))


def cache_sizes(engine, prompt_ids, max_tokens):
    max_total = max(len(prompt) + limit for prompt, limit in zip(prompt_ids, max_tokens))
    max_blocks = (max_total + engine.block_size - 1) // engine.block_size
    # Generous pool: these tests isolate graph behavior rather than preemption.
    num_blocks = MAX_RUNNING * max_blocks + MAX_RUNNING
    return num_blocks, max_blocks


@torch.no_grad()
def test_cuda_replay_for_every_batch_size(engine, prompt_ids):
    """Compare replay to eager using the identical bucket-padded inputs."""
    max_tokens = [12] * MAX_RUNNING
    num_blocks, max_blocks = cache_sizes(engine, prompt_ids, max_tokens)
    scheduler = Scheduler(
        engine.model,
        engine.cfg,
        MAX_RUNNING,
        num_blocks,
        engine.block_size,
        set(),
        DEVICE,
        DTYPE,
        use_graph=True,
        graph_max_blocks=max_blocks,
    )
    for prompt in prompt_ids:
        scheduler.add_request(prompt, max_tokens=12)

    # Populate one request at a time: this test isolates every decode batch size;
    # packed admission itself has a dedicated correctness sweep.
    for expected_running in range(1, MAX_RUNNING + 1):
        scheduler._admit()
        assert len(scheduler.running) == expected_running

    results = []
    active_all = list(scheduler.running)
    for n in range(1, MAX_RUNNING + 1):
        active = active_all[:n]
        n_news = [1 if slot in active else 0 for slot in range(MAX_RUNNING)]
        scheduler.cache.allocate_block(n_news)
        tokens = [scheduler.running[slot].output_ids[-1] for slot in active]
        inputs = scheduler._build_step_inputs(active, tokens)

        bucket = scheduler.graph_decoder._pick_bucket(n)
        captured_decoder = scheduler.graph_decoder.decoders[bucket]
        real_decode = captured_decoder.decode
        recorded = {}

        def record_and_replay(*padded_inputs):
            recorded["inputs"] = padded_inputs
            return real_decode(*padded_inputs)

        captured_decoder.decode = record_and_replay
        try:
            graphed = scheduler.graph_decoder.decode(*inputs).clone()
        finally:
            captured_decoder.decode = real_decode

        assert "inputs" in recorded
        padded_inputs = recorded["inputs"]
        assert all(t.shape[0] == bucket for t in padded_inputs)
        assert padded_inputs[3].shape == (bucket, max_blocks)

        # Replay has already written the current KV values. Re-running the same forward
        # eagerly is idempotent and provides an identical-shape, identical-input oracle.
        eager = graph_decode_forward(
            engine.model, scheduler.cache, *padded_inputs
        )[:n]
        max_error = (graphed.float() - eager.float()).abs().max().item()
        torch.testing.assert_close(
            graphed.float(), eager.float(), atol=LOGIT_ATOL, rtol=LOGIT_RTOL
        )
        assert torch.equal(
            graphed[:, -1].argmax(dim=-1), eager[:, -1].argmax(dim=-1)
        )
        results.append((n, bucket, max_error))

    return results


def matched_prefix(a, b):
    for index, (left, right) in enumerate(zip(a, b)):
        if left != right:
            return index
    return min(len(a), len(b))


@torch.no_grad()
def run_scheduler(engine, prompt_ids, max_tokens, use_graph):
    num_blocks, max_blocks = cache_sizes(engine, prompt_ids, max_tokens)
    scheduler = Scheduler(
        engine.model,
        engine.cfg,
        MAX_RUNNING,
        num_blocks,
        engine.block_size,
        set(),  # Disable EOS so the requested lengths force deterministic batch transitions.
        DEVICE,
        DTYPE,
        use_graph=use_graph,
        graph_max_blocks=max_blocks,
    )
    request_ids = [
        scheduler.add_request(prompt, limit)
        for prompt, limit in zip(prompt_ids, max_tokens)
    ]

    active_sizes = []
    selected_buckets = []
    if use_graph:
        real_pick_bucket = scheduler.graph_decoder._pick_bucket

        def record_bucket(n):
            bucket = real_pick_bucket(n)
            active_sizes.append(n)
            selected_buckets.append(bucket)
            return bucket

        scheduler.graph_decoder._pick_bucket = record_bucket

    output = scheduler.run()
    return (
        [output[request_id] for request_id in request_ids],
        active_sizes,
        selected_buckets,
    )


@torch.no_grad()
def test_scheduler_integration(engine, prompt_ids):
    # One request finishes per decode iteration: active sizes must be 4 -> 3 -> 2 -> 1.
    max_tokens = [2, 3, 4, 5]
    eager, _, _ = run_scheduler(engine, prompt_ids, max_tokens, use_graph=False)
    graphed, active_sizes, selected_buckets = run_scheduler(
        engine, prompt_ids, max_tokens, use_graph=True
    )

    assert active_sizes == [4, 3, 2, 1], active_sizes
    assert selected_buckets == [4, 4, 2, 1], selected_buckets
    assert [len(tokens) for tokens in eager] == max_tokens
    assert [len(tokens) for tokens in graphed] == max_tokens

    # Prefill and the first decode occur at the same physical batch size in both paths.
    # Later generation can legitimately take a different fp16 branch because n=3 is
    # padded to 4. Exact padded-forward parity is already asserted above for every n.
    prefixes = [matched_prefix(got, ref) for got, ref in zip(graphed, eager)]
    assert all(prefix >= min(2, limit) for prefix, limit in zip(prefixes, max_tokens)), prefixes
    return prefixes, active_sizes, selected_buckets


def main():
    print("bucket construction/padding/error contracts...")
    test_bucket_and_padding_contracts()
    print("  PASS")

    assert torch.cuda.is_available(), "CUDA replay tests require an NVIDIA GPU"
    print("loading paged engine...")
    engine = PagedEngine(cfg=Qwen2Config(), device=DEVICE, dtype=DTYPE)
    prompt_ids = [
        engine.tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
        for prompt in PROMPTS
    ]

    print("teacher-forced eager parity for every active batch size...")
    replay_results = test_cuda_replay_for_every_batch_size(engine, prompt_ids)
    for n, bucket, max_error in replay_results:
        print(f"  active={n} -> bucket={bucket}: max logit error={max_error:.6f}")
    print("  PASS")

    print("full scheduler lifecycle and bucket transitions...")
    prefixes, active_sizes, selected_buckets = test_scheduler_integration(engine, prompt_ids)
    print(f"  active sizes    : {active_sizes}")
    print(f"  selected buckets: {selected_buckets}")
    print(f"  eager prefixes : {prefixes}")
    print("  PASS")

    print("\nOVERALL: PASS")
    return True


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
