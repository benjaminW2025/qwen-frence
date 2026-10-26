"""Native FA3 metadata/scaling contracts; optional real Hopper execution."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.quant import full_model, native_fa3


class NativeContracts(unittest.TestCase):
    def fixture(self):
        case = full_model.make_case("decode", 2, 1, 17, 100, 3, "cpu")
        pool = SimpleNamespace(block_size=16,
                               k_pool=[torch.randn(case["blocks"], 16, 2, 128).half()],
                               v_pool=[torch.randn(case["blocks"], 16, 2, 128).half()])
        seed = [x.clone() for x in pool.k_pool + pool.v_pool]
        return case, pool, seed

    def test_query_scales_are_per_gqa_group(self):
        q = torch.zeros(2, 12, 128).half()
        q[:, 6:] = 2
        quantized, scales = native_fa3.quantize_query(q, 2)
        self.assertEqual(scales.shape, (2, 2))
        self.assertEqual(quantized.dtype, torch.float8_e4m3fn)
        actual = quantized.float().reshape(2, 2, 6, 128) * scales[:, :, None, None]
        torch.testing.assert_close(actual.reshape_as(q), q.float(), atol=1e-6, rtol=1e-6)

    def test_prefix_only_calibration_and_saturation(self):
        case, pool, seed = self.fixture()
        for tensor in seed:
            tensor.flatten(0, 1)[case["slots"]] = 0
        cache = native_fa3.NativeFP8Cache(pool, seed, case, headroom=1.25)
        self.assertTrue(cache.unchanged_prefix())
        for entry in cache.entries.values():
            self.assertEqual(entry["scales"].shape, (2, 2))
            self.assertEqual(entry["q"].dtype, torch.float8_e4m3fn)
            self.assertEqual(entry["owners"].unique().tolist(), [0, 1])
        for tensor in pool.k_pool + pool.v_pool:
            tensor.flatten(0, 1)[case["slots"]] = 1000
        result = cache.saturation(pool)
        self.assertEqual(result["k_by_layer"], [1.])
        self.assertEqual(result["v_by_layer"], [1.])

    def test_byte_prefix_guard_and_reset(self):
        case, pool, seed = self.fixture()
        cache = native_fa3.NativeFP8Cache(pool, seed, case)
        entry = next(iter(cache.entries.values()))
        entry["q"].view(torch.uint8).flatten(0, 1)[case["history_slots"][0]] = 0
        self.assertFalse(cache.unchanged_prefix())
        cache.reset()
        self.assertTrue(cache.unchanged_prefix())
        self.assertEqual(cache.stored_writes(pool).shape, (2, 2, 2, 128))

    def test_native_call_passes_scales_and_forces_fa3(self):
        from custom_kernels import paged_decode_fa3
        q = torch.zeros(2, 12, 128).to(torch.float8_e4m3fn)
        k = torch.zeros(4, 16, 2, 128).to(torch.float8_e4m3fn)
        table = torch.zeros(2, 2, dtype=torch.int32)
        lengths = torch.ones(2, dtype=torch.int32)
        scales = torch.ones(2, 2)
        backend = mock.Mock(return_value=torch.ones(2, 12, 128).bfloat16())
        with mock.patch.object(paged_decode_fa3, "_load_fa3", return_value=backend):
            out = native_fa3.call_native(q, k, k, table, lengths, scales, scales, scales)
        self.assertEqual(out.dtype, torch.float16)
        self.assertEqual(backend.call_args.kwargs["fa_version"], 3)
        self.assertIs(backend.call_args.kwargs["k"], k)
        self.assertIs(backend.call_args.kwargs["k_descale"], scales)

    def test_empty_variants_enables_native_only_experiment(self):
        args = full_model.arguments(["--variants", "--kv-modes", "off", "--native-fa3",
                                     "--output", "/tmp/native-fp8-uncreated-report-contract.json"])
        self.assertEqual(args.variants, [])
        self.assertTrue(args.native_fa3)


HOPPER = torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


@unittest.skipUnless(HOPPER, "requires Hopper and native vLLM FA3")
class NativeGPU(unittest.TestCase):
    @torch.no_grad()
    def test_native_page16_kernel_canary(self):
        self.assertTrue(native_fa3.smoke_native("cuda")["passed"])

    @torch.no_grad()
    def test_native_full_model_graph_replay(self):
        from naive_forward import Model, Qwen2Config
        from model_adapter import allocate_pool
        from experiments.quant.projections import strict_accumulation
        cfg = Qwen2Config(vocab=128, d_model=256, d_ff=320, n_layers=2,
                          n_heads=2, n_kv_heads=1, use_custom_kernels=True)
        with strict_accumulation():
            model = Model(cfg).cuda().half().eval().pack_projections_()
            case = full_model.make_case("decode", 2, 1, 17, cfg.vocab, 8, "cuda")
            pool = allocate_pool(cfg, case["blocks"], "cuda")
            for tensor in full_model.cache_state(pool):
                tensor.zero_()
            full_model.prepare_prefix(model, pool, case)
            seed = [tensor.clone() for tensor in full_model.cache_state(pool)]
            cache = native_fa3.NativeFP8Cache(pool, seed, case)
            with cache.attention_hook():
                adapter, run = full_model.build_forward(model, pool, case)
                run()
                full_model.restore_cache(pool, seed)
                cache.reset()
                output = run().clone()
                self.assertTrue(torch.isfinite(output).all())
                self.assertEqual(output.shape, (2, cfg.vocab))
                self.assertTrue(cache.unchanged_prefix())
                full_model.restore_cache(pool, seed)
                cache.reset()
                torch.testing.assert_close(output, run(), atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
