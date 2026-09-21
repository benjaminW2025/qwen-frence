"""Host-side contract for the capture-safe split-K decode policy.

Capture itself needs a GPU, but everything that decides *what* gets captured is
host code: which policy survives the context floor, which action the policy freezes,
and what context bound a graph commits to. Those are the parts that can silently
make a captured arm measure the production kernel, so they are tested here.
"""
from __future__ import annotations

from pathlib import Path
import json
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]


def _stub_triton():
    """Triton is a GPU dependency; the functions under test are pure host code."""
    if "triton" in sys.modules:
        return
    language = types.ModuleType("triton.language")
    language.constexpr = object
    triton = types.ModuleType("triton")
    triton.language = language
    triton.jit = lambda fn: fn
    triton.cdiv = lambda a, b: -(-a // b)
    triton.next_power_of_2 = lambda n: 1 << (n - 1).bit_length()
    sys.modules["triton"], sys.modules["triton.language"] = triton, language


_stub_triton()
for path in (ROOT / "engine/kvcache", ROOT / "engine/graph", ROOT / "baseline",
             ROOT / "experiments/decode"):
    sys.path.insert(0, str(path))
import paged_decode_attention as pda


class PolicyResolutionTests(unittest.TestCase):
    def test_production_and_adaptive_are_unchanged(self):
        self.assertEqual(pda.resolve_decode_attention_policy("production", None), "production")
        self.assertEqual(pda.resolve_decode_attention_policy("production", 99999), "production")
        # The adaptive floor and its behaviour must not move when splitk is added.
        floor = pda.MIN_ADAPTIVE_DECODE_CONTEXT_LENGTH
        self.assertEqual(pda.resolve_decode_attention_policy("adaptive", floor - 1), "production")
        self.assertEqual(pda.resolve_decode_attention_policy("adaptive", floor), "adaptive")

    def test_splitk_honours_its_own_floor(self):
        floor = pda.MIN_SPLITK_DECODE_CONTEXT_LENGTH
        self.assertEqual(pda.resolve_decode_attention_policy("splitk", floor - 1), "production")
        self.assertEqual(pda.resolve_decode_attention_policy("splitk", floor), "splitk")
        self.assertEqual(pda.resolve_decode_attention_policy("native_grouped", floor - 1),
                         "production")
        self.assertEqual(pda.resolve_decode_attention_policy("native_grouped", floor),
                         "native_grouped")
        self.assertEqual(pda.resolve_decode_attention_policy("fa3", 1), "fa3")

    def test_unknown_policy_and_missing_context_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be one of"):
            pda.resolve_decode_attention_policy("bogus", 4096)
        for policy in ("adaptive", "splitk", "native_grouped", "fa3"):
            with self.assertRaisesRegex(ValueError, "context length"):
                pda.resolve_decode_attention_policy(policy, None)

    def test_fa3_dispatch_uses_automatic_split_scheduler(self):
        sentinel = object()
        calls = []
        dispatch = types.ModuleType("kernel_dispatch")

        def fa3(*args, **kwargs):
            calls.append((args, kwargs))
            return sentinel

        dispatch.fa3_paged_decode_attention = fa3
        tensors = tuple(object() for _ in range(5))
        with mock.patch.dict(sys.modules, {"kernel_dispatch": dispatch}):
            result = pda.paged_decode_attention_dispatch(
                *tensors, policy="fa3", max_context_length=4096, scale=.125,
            )
        self.assertIs(result, sentinel)
        self.assertEqual(calls, [(tensors, {"scale": .125, "num_splits": 0})])

    def test_frozen_action_matches_the_studied_page_threshold(self):
        threshold = pda.SPLITK_PAGE_THRESHOLD
        short = pda.select_splitk_config(threshold * 16)
        long = pda.select_splitk_config(threshold * 16 + 1)
        self.assertEqual((short["split_k"], short["num_stages"]), (8, 2))
        self.assertEqual((long["split_k"], long["num_stages"]), (22, 3))
        # The study fixed head grouping at one and four warps; the engine must not
        # quietly select a configuration the policy was never fitted against.
        for config in (short, long):
            self.assertEqual(config["heads_per_program"], 1)
            self.assertEqual(config["num_warps"], 4)
        with self.assertRaisesRegex(ValueError, "positive host context length"):
            pda.select_splitk_config(0)

    def test_action_is_selected_on_pages_not_raw_tokens(self):
        # 80 pages of 16 covers 1265..1280 tokens; every one of them is the short
        # action, and the first token of page 81 flips it.
        self.assertEqual(pda.select_splitk_config(1265)["split_k"], 8)
        self.assertEqual(pda.select_splitk_config(1280)["split_k"], 8)
        self.assertEqual(pda.select_splitk_config(1281)["split_k"], 22)
        # A larger page size moves the boundary with it rather than staying at 1280.
        self.assertEqual(pda.select_splitk_config(1281, page_size=32)["split_k"], 8)


class CaptureBoundTests(unittest.TestCase):
    """A graph cannot re-decide on replay, so its committed bound must be the widest."""

    def _decoder(self, policy, max_blocks=64, block_size=16, bound=None, **kwargs):
        import torch
        from paged_graph_decoder import CUDAGraphDecoder
        cache = types.SimpleNamespace(block_size=block_size, k_pool=[], v_pool=[])
        return CUDAGraphDecoder(
            model=types.SimpleNamespace(cfg=None), cache=cache, batch_size=2,
            max_blocks=max_blocks, device=torch.device("cpu"), dtype=torch.float32,
            decode_attention_policy=policy, max_decode_context_length=bound,
            **kwargs,
        )

    def test_bound_defaults_to_the_whole_captured_block_table(self):
        decoder = self._decoder("splitk", max_blocks=64, block_size=16)
        # Anything the block table can address is a context this graph may replay.
        self.assertEqual(decoder.max_decode_context_length, 64 * 16)
        self.assertEqual(decoder.decode_attention_policy, "splitk")

    def test_explicit_bound_is_respected(self):
        decoder = self._decoder("splitk", max_blocks=64, bound=2048)
        self.assertEqual(decoder.max_decode_context_length, 2048)

    def test_capture_metadata_starts_with_one_valid_token(self):
        decoder = self._decoder("fa3", max_blocks=64, bound=1024)
        self.assertEqual(decoder.s_seq_lens.tolist(), [1, 1])
        self.assertFalse(decoder.s_block_table.any())

    def test_fusion_flags_are_explicit_and_mutually_exclusive(self):
        decoder = self._decoder(
            "fa3", enable_residual_rmsnorm=True,
            enable_fused_qkv_rope_cache=True,
        )
        self.assertTrue(decoder.enable_residual_rmsnorm)
        self.assertTrue(decoder.enable_fused_qkv_rope_cache)
        with self.assertRaisesRegex(ValueError, "one QKV"):
            self._decoder(
                "fa3", enable_native_decode_qkv_postprocess=True,
                enable_fused_qkv_rope_cache=True,
            )

    def test_production_needs_no_bound(self):
        decoder = self._decoder("production", max_blocks=64)
        self.assertIsNone(decoder.max_decode_context_length)

    def test_default_bound_selects_the_long_action_for_a_wide_table(self):
        # A graph whose table reaches past the page threshold must bake the long
        # action, even though its early steps are far shorter.
        decoder = self._decoder("splitk", max_blocks=1024, block_size=16)
        config = pda.select_splitk_config(decoder.max_decode_context_length)
        self.assertEqual(config["split_k"], 22)


class BenchmarkReportingTests(unittest.TestCase):
    def test_resolved_action_exposes_a_production_fallback(self):
        from benchmark_splitk_capture import resolved_action
        below = pda.MIN_SPLITK_DECODE_CONTEXT_LENGTH - 1
        fallback = resolved_action("splitk", below, 16)
        # A fallback arm is identical to arm A; reporting must not present that as a
        # measured split-K result.
        self.assertEqual(fallback, {"effective_policy": "production", "action": None})
        measured = resolved_action("splitk", 8192, 16)
        self.assertEqual(measured["effective_policy"], "splitk")
        self.assertEqual(measured["action"], "K22-S3")
        self.assertIsNone(resolved_action("production", 8192, 16)["action"])

    def test_cache_history_is_nonzero_reproducible_and_independent_per_layer(self):
        import torch
        from benchmark_splitk_capture import seed_cache_history

        def cache():
            return types.SimpleNamespace(
                device="cpu",
                k_pool=[torch.zeros(3, 16, 2, 8) for _ in range(2)],
                v_pool=[torch.zeros(3, 16, 2, 8) for _ in range(2)],
            )

        first, second = cache(), cache()
        seed_cache_history(first, 123)
        seed_cache_history(second, 123)
        for left, right in zip(first.k_pool + first.v_pool,
                               second.k_pool + second.v_pool):
            self.assertTrue(torch.equal(left, right))
            self.assertTrue(torch.isfinite(left).all())
            self.assertTrue(torch.any(left != 0))
        self.assertFalse(torch.equal(first.k_pool[0], first.k_pool[1]))
        self.assertFalse(torch.equal(first.k_pool[0], first.v_pool[0]))

    def test_main_passes_the_loaded_engine_to_cases(self):
        import benchmark_splitk_capture as benchmark

        engine = object()
        loader = types.ModuleType("run_phase_sweep")
        loader.make_engine = lambda *args: (engine, 1.25)
        metadata = types.ModuleType("run_benchmarks")
        metadata.system_metadata = lambda: {"test": True}
        seen = []

        def run_case(loaded, **kwargs):
            seen.append(loaded)
            return {"status": "incorrect", "batch_size": kwargs["batch_size"],
                    "context_length": kwargs["context_length"]}

        with tempfile.TemporaryDirectory() as directory:
            argv = ["benchmark_splitk_capture.py", "--batch-sizes", "1",
                    "--context-lengths", "1024", "--output-dir", directory]
            with (mock.patch.dict(sys.modules, {"run_phase_sweep": loader,
                                                "run_benchmarks": metadata}),
                  mock.patch.object(sys, "argv", argv),
                  mock.patch.object(benchmark, "run_case", side_effect=run_case)):
                with self.assertRaisesRegex(SystemExit, "1 failed cases"):
                    benchmark.main()
            manifest = json.loads((Path(directory) / "manifest.json").read_text())
            report = json.loads((Path(directory) / "report.json").read_text())
        self.assertEqual(seen, [engine])
        self.assertEqual(manifest["model_load_seconds"], 1.25)
        self.assertEqual(report["cases"][0]["status"], "incorrect")
        self.assertEqual(report["status"], "incomplete")


if __name__ == "__main__":
    unittest.main()
