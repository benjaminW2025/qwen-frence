"""Behavioral and logit-routing tests for the real C++ scheduler extension."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import types
import unittest
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[3]
CPP_DIR = ROOT / "engine" / "cpp"
for path in (
    CPP_DIR / "build",
    ROOT / "engine" / "scheduler",
):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

try:
    import inference_engine_cpp as cpp
except ImportError as error:  # pragma: no cover - exercised only before building
    cpp = None
    IMPORT_ERROR = error
else:
    IMPORT_ERROR = None

# The scheduler imports its GPU model runners at module-import time. These tests
# replace all three runners with deterministic CPU functions, so install small
# import stubs rather than requiring Triton merely to exercise control flow.
for module_name, function_name in (
    ("ragged_prefill", "ragged_prefill"),
    ("mixed_batch", "mixed_batch_forward"),
    ("paged_graph_decoder", "graph_decode_forward"),
):
    if module_name not in sys.modules:
        module = types.ModuleType(module_name)
        setattr(module, function_name, lambda *args, **kwargs: None)
        sys.modules[module_name] = module

import scheduler as python_scheduler


VOCAB_SIZE = 64


def deterministic_logits(source_tokens: torch.Tensor) -> torch.Tensor:
    """Return exact logits whose argmax is `(source_token + 1) % vocab`."""
    source_tokens = source_tokens.reshape(-1).to(torch.long)
    targets = (source_tokens + 1) % VOCAB_SIZE
    logits = torch.full(
        (source_tokens.numel(), VOCAB_SIZE),
        -8.0,
        dtype=torch.float32,
        device=source_tokens.device,
    )
    logits.scatter_(1, targets.unsqueeze(1), 8.0)
    return logits


def make_cpp_config(**overrides):
    config = cpp.SchedulerConfig()
    config.max_batch_size = 4
    config.max_prefill_tokens_per_iter = 64
    config.max_context_length = 64
    config.block_size = 4
    config.num_kv_heads = 1
    config.head_dim = 8
    config.eos_token_id = -1
    for name, value in overrides.items():
        setattr(config, name, value)
    return config


def cpp_forward(trace=None):
    def forward(
        input_ids,
        positions,
        slot_mapping,
        cu_seqlens,
        context_lens,
        block_table,
        max_query_length,
        is_decode,
    ):
        del positions, slot_mapping, context_lens, block_table, max_query_length
        if is_decode:
            source_tokens = input_ids
            phase = "decode"
        else:
            ends = cu_seqlens[1:].to(torch.long) - 1
            source_tokens = input_ids.index_select(0, ends)
            phase = "prefill"
        logits = deterministic_logits(source_tokens)
        if trace is not None:
            trace.append((phase, logits.detach().cpu().clone()))
        return logits

    return forward


def run_cpp_workload(prompts, output_lengths, *, trace=None):
    loop = cpp.IterationLoop(make_cpp_config(), torch.device("cpu"))
    for prompt, output_length in zip(prompts, output_lengths):
        loop.submit_request(prompt, output_length)
    while loop.num_pending() or loop.num_running():
        loop.step(cpp_forward(trace))
    return dict(loop.pop_completed())


def run_python_workload(prompts, output_lengths, *, trace=None):
    config = SimpleNamespace(
        n_layers=0,
        n_kv_heads=1,
        d_head=8,
        max_seq_len=64,
    )
    scheduler = python_scheduler.Scheduler(
        model=None,
        cfg=config,
        max_running=4,
        num_blocks=64,
        block_size=4,
        eos_ids=set(),
        device="cpu",
        dtype=torch.float32,
        max_num_batched_tokens=64,
    )

    def record(phase, source_tokens):
        logits = deterministic_logits(source_tokens)
        if trace is not None:
            trace.append((phase, logits.detach().cpu().clone()))
        return logits

    def fake_prefill(model, cache, slots, chunks, **kwargs):
        del model, kwargs
        new_tokens = [0] * cache.batch_size
        for slot, chunk in zip(slots, chunks):
            new_tokens[slot] = len(chunk)
        cache.allocate_block(new_tokens)
        source = torch.tensor([chunk[-1] for chunk in chunks], dtype=torch.long)
        return record("prefill", source)

    def fake_decode(model, cache, input_ids, *args, **kwargs):
        del model, cache, args, kwargs
        return record("decode", input_ids.reshape(-1)).unsqueeze(1)

    def fake_mixed(
        model,
        cache,
        *,
        decode_slots,
        decode_tokens,
        prefill_slots,
        prefill_chunks,
        **kwargs,
    ):
        del model, kwargs
        new_tokens = [0] * cache.batch_size
        for slot in decode_slots:
            new_tokens[slot] = 1
        for slot, chunk in zip(prefill_slots, prefill_chunks):
            new_tokens[slot] = len(chunk)
        cache.allocate_block(new_tokens)
        decode = record("decode", torch.tensor(decode_tokens, dtype=torch.long))
        prefill = record(
            "prefill",
            torch.tensor([chunk[-1] for chunk in prefill_chunks], dtype=torch.long),
        )
        return decode, prefill

    with (
        mock.patch.object(python_scheduler, "ragged_prefill", fake_prefill),
        mock.patch.object(python_scheduler, "graph_decode_forward", fake_decode),
        mock.patch.object(python_scheduler, "mixed_batch_forward", fake_mixed),
    ):
        for prompt, output_length in zip(prompts, output_lengths):
            scheduler.add_request(prompt, output_length)
        return scheduler.run()


@unittest.skipIf(cpp is None, f"build extension first: {IMPORT_ERROR}")
class IterationLoopTests(unittest.TestCase):
    def test_chunked_prefill_metadata_and_logits(self):
        config = make_cpp_config(max_prefill_tokens_per_iter=2)
        loop = cpp.IterationLoop(config, torch.device("cpu"))
        loop.submit_request([5, 6, 7], 2)
        calls = []

        def forward(ids, positions, slots, cu, context, blocks, max_query, is_decode):
            calls.append({
                "ids": ids.tolist(),
                "positions": positions.tolist(),
                "slots": slots.tolist(),
                "cu": cu.tolist(),
                "context": context.tolist(),
                "block_shape": tuple(blocks.shape),
                "max_query": max_query,
                "is_decode": is_decode,
            })
            if is_decode:
                targets = torch.tensor([10])
            elif len(calls) == 1:
                targets = torch.tensor([8])  # Partial-chunk prediction is ignored.
            else:
                targets = torch.tensor([9])
            logits = torch.full((1, 16), -1.0)
            logits.scatter_(1, targets.unsqueeze(1), 1.0)
            return logits

        while loop.num_pending() or loop.num_running():
            loop.step(forward)

        self.assertEqual(dict(loop.pop_completed()), {0: [9, 10]})
        self.assertEqual(calls[0]["ids"], [5, 6])
        self.assertEqual(calls[0]["positions"], [0, 1])
        self.assertEqual(calls[0]["cu"], [0, 2])
        self.assertEqual(calls[0]["context"], [2])
        self.assertEqual(calls[0]["max_query"], 2)
        self.assertFalse(calls[0]["is_decode"])
        self.assertEqual(calls[1]["ids"], [7])
        self.assertEqual(calls[1]["positions"], [2])
        self.assertEqual(calls[1]["cu"], [0, 1])
        self.assertEqual(calls[1]["context"], [3])
        self.assertTrue(calls[2]["is_decode"])
        self.assertEqual(calls[2]["ids"], [9])
        self.assertEqual(calls[2]["cu"], [])

    def test_mixed_logits_keep_decode_then_prefill_order(self):
        loop = cpp.IterationLoop(make_cpp_config(), torch.device("cpu"))
        loop.submit_request([1], 3)
        modes = []

        def forward(*args):
            modes.append(bool(args[-1]))
            return cpp_forward()(*args)

        loop.step(forward)
        loop.submit_request([5], 2)
        while loop.num_pending() or loop.num_running():
            loop.step(forward)

        self.assertEqual(dict(loop.pop_completed()), {0: [2, 3, 4], 1: [6, 7]})
        self.assertEqual(modes, [False, True, False, True])

    def test_eos_completes_and_releases_request(self):
        config = make_cpp_config(eos_token_id=3)
        loop = cpp.IterationLoop(config, torch.device("cpu"))
        loop.submit_request([1], 10)
        while loop.num_pending() or loop.num_running():
            loop.step(cpp_forward())
        self.assertEqual(dict(loop.pop_completed()), {0: [2, 3]})
        self.assertEqual(loop.num_running(), 0)

    def test_completed_blocks_are_reused(self):
        config = make_cpp_config(
            max_batch_size=1,
            max_context_length=2,
            block_size=2,
        )
        loop = cpp.IterationLoop(config, torch.device("cpu"))

        loop.submit_request([1], 1)
        loop.submit_request([5], 1)
        self.assertEqual(loop.step(cpp_forward()), 1)
        self.assertEqual(dict(loop.pop_completed()), {0: [2]})
        self.assertEqual(loop.num_pending(), 1)

        # max_batch_size admits only one request, and this scheduler owns only
        # one physical block. The pending request can now run only if completion
        # returned the first request's block.
        self.assertEqual(loop.step(cpp_forward()), 1)
        self.assertEqual(dict(loop.pop_completed()), {1: [6]})

    def test_invalid_inputs_and_logits_are_rejected(self):
        config = make_cpp_config()
        loop = cpp.IterationLoop(config, torch.device("cpu"))
        with self.assertRaisesRegex(ValueError, "at least one token"):
            loop.submit_request([], 1)
        with self.assertRaisesRegex(ValueError, "must be positive"):
            loop.submit_request([1], 0)
        with self.assertRaisesRegex(ValueError, "max_context_length"):
            loop.submit_request([1] * 64, 1)

        loop.submit_request([1], 1)
        with self.assertRaisesRegex(RuntimeError, "row count"):
            loop.step(lambda *args: torch.zeros((2, VOCAB_SIZE)))

        bad_config = make_cpp_config(max_batch_size=0)
        with self.assertRaisesRegex(ValueError, "max_batch_size"):
            cpp.IterationLoop(bad_config, torch.device("cpu"))

    def test_reused_metadata_clears_padding_and_keeps_storage(self):
        loop = cpp.IterationLoop(make_cpp_config(max_batch_size=1), torch.device("cpu"))
        pointers = {}
        widths = []

        def forward(*args):
            ids, positions, slots, cu, context, blocks, _, decode = args
            ptrs = tuple(t.data_ptr() for t in (ids, positions, slots, context, blocks))
            if decode in pointers:
                self.assertEqual(ptrs, pointers[decode])
            pointers[decode] = ptrs
            # Admission reserves blocks for the whole prompt + output budget.
            width = widths[-1]
            self.assertEqual(torch.count_nonzero(blocks[:, width:]).item(), 0)
            if decode:
                expected_slots = blocks.gather(
                    1, (positions // 4).reshape(-1, 1)
                ).reshape(-1).to(torch.long) * 4 + positions % 4
                torch.testing.assert_close(slots, expected_slots)
            return cpp_forward()(*args)

        for prompt in ([1] * 17, [5]):
            widths.append((len(prompt) + 3 + 3) // 4)
            loop.submit_request(prompt, 3)
            while loop.num_pending() or loop.num_running():
                loop.step(forward)
            loop.pop_completed()

    def test_logits_and_outputs_match_python_scheduler(self):
        prompts = ([1, 2], [5], [8, 9, 10])
        output_lengths = (3, 2, 1)
        cpp_trace = []
        python_trace = []

        cpp_outputs = run_cpp_workload(prompts, output_lengths, trace=cpp_trace)
        python_outputs = run_python_workload(
            prompts, output_lengths, trace=python_trace
        )

        self.assertEqual(cpp_outputs, python_outputs)
        self.assertEqual(
            [phase for phase, _ in cpp_trace],
            [phase for phase, _ in python_trace],
        )
        self.assertEqual(len(cpp_trace), len(python_trace))
        for (_, cpp_logits), (_, python_logits) in zip(cpp_trace, python_trace):
            torch.testing.assert_close(cpp_logits, python_logits, rtol=0, atol=0)


@unittest.skipUnless(cpp is not None and torch.cuda.is_available(), "requires CUDA extension runtime")
class PinnedMetadataCudaTests(unittest.TestCase):
    def test_metadata_and_outputs_match_cpu_across_streams(self):
        def run(device):
            loop = cpp.IterationLoop(
                make_cpp_config(max_prefill_tokens_per_iter=3), device
            )
            trace = []
            pointers = {}
            streams = [torch.cuda.Stream(device=device) for _ in range(2)] if device.type == "cuda" else []

            def forward(*args):
                tensors = args[:6]
                decode = args[-1]
                ptrs = tuple(t.data_ptr() for t in tensors if t.numel())
                if decode in pointers:
                    self.assertEqual(ptrs, pointers[decode])
                pointers[decode] = ptrs
                if streams:
                    self.assertEqual(torch.cuda.current_stream(device), streams[iteration % 2])
                    # Delay reads to exercise event ordering without an incidental
                    # .cpu()/.item() synchronization inside the callback.
                    torch.cuda._sleep(100_000)
                trace.append(([t.clone() for t in tensors], args[6:]))
                return cpp_forward()(*args)

            loop.submit_request([1, 2, 3, 4, 5], 4)
            iteration = 0
            while loop.num_pending() or loop.num_running():
                if iteration == 2:
                    loop.submit_request([11, 12], 2)
                if streams:
                    with torch.cuda.stream(streams[iteration % 2]):
                        loop.step(forward)
                        self.assertEqual(torch.cuda.current_stream(device), streams[iteration % 2])
                else:
                    loop.step(forward)
                iteration += 1
                self.assertLess(iteration, 30)
            return dict(loop.pop_completed()), [([t.cpu() for t in ts], flags) for ts, flags in trace]

        expected, cpu_trace = run(torch.device("cpu"))
        actual, gpu_trace = run(torch.device("cuda"))
        self.assertEqual(actual, expected)
        self.assertEqual(len(cpu_trace), len(gpu_trace))
        for (cpu_tensors, cpu_flags), (gpu_tensors, gpu_flags) in zip(cpu_trace, gpu_trace):
            self.assertEqual(cpu_flags, gpu_flags)
            for expected_tensor, actual_tensor in zip(cpu_tensors, gpu_tensors):
                torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)

    def test_delayed_consumers_survive_reuse_and_destruction(self):
        loop = cpp.IterationLoop(make_cpp_config(), torch.device("cuda"))
        streams = [torch.cuda.Stream() for _ in range(2)]
        snapshots = []

        def forward(*args):
            torch.cuda._sleep(1_000_000)
            snapshots.append(args[0].clone())
            # Deliberately return independent CPU logits: step's normal token
            # readback cannot accidentally fence these outstanding GPU reads.
            return deterministic_logits(torch.tensor([1]))

        for i in range(12):
            loop.submit_request([i + 1], 1)
            with torch.cuda.stream(streams[i % 2]):
                self.assertEqual(loop.step(forward), 1)
        del loop  # Must finish the last consumer before releasing its buffers.
        for i, snapshot in enumerate(snapshots):
            torch.testing.assert_close(snapshot.cpu(), torch.tensor([i + 1]))

    def test_failed_callback_drains_reads_and_restores_stream(self):
        loop = cpp.IterationLoop(make_cpp_config(), torch.device("cuda"))
        loop.submit_request([7, 8], 2)
        stream = torch.cuda.Stream()
        snapshots = []

        def failing_forward(*args):
            torch.cuda._sleep(1_000_000)
            snapshots.append(args[0].clone())
            raise RuntimeError("intentional callback failure")

        with torch.cuda.stream(stream):
            with self.assertRaisesRegex(RuntimeError, "intentional callback failure"):
                loop.step(failing_forward)
            self.assertEqual(torch.cuda.current_stream(), stream)
            self.assertTrue(stream.query())
        # Retry on another stream after changing the scheduled batch.
        loop.submit_request([20], 1)
        while loop.num_pending() or loop.num_running():
            loop.step(cpp_forward())
        torch.testing.assert_close(snapshots[0].cpu(), torch.tensor([7, 8]))
        self.assertEqual(dict(loop.pop_completed()), {0: [9, 10], 1: [21]})

    @unittest.skipUnless(torch.cuda.device_count() > 1, "requires two CUDA devices")
    def test_requested_device_and_caller_device_are_preserved(self):
        with torch.cuda.device(0):
            loop = cpp.IterationLoop(make_cpp_config(), torch.device("cuda:1"))
            self.assertEqual(torch.cuda.current_device(), 0)
            loop.submit_request([1], 1)

            def forward(*args):
                self.assertEqual(torch.cuda.current_device(), 1)
                self.assertEqual(args[0].device, torch.device("cuda:1"))
                return cpp_forward()(*args)

            loop.step(forward)
            self.assertEqual(torch.cuda.current_device(), 0)
            self.assertEqual(dict(loop.pop_completed()), {0: [2]})


if __name__ == "__main__":
    unittest.main()
