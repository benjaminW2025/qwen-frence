"""Portable quantization/sensitivity contracts and real CUDA tile validation."""
import importlib.util
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location(
    "local_attention_fp8", ROOT / "experiments/quant/attention.py")
local = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(local)


class TileAttentionContracts(unittest.TestCase):
    def test_every_e4m3_byte_decodes_like_torch(self):
        bits = torch.arange(256, dtype=torch.int32).to(torch.uint8).reshape(1, 1, 2, 128)
        actual = local.dequantize(bits, torch.ones(1, 1, 2))
        expected = bits.view(torch.float8_e4m3fn).half()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0, equal_nan=True)

    def test_zero_and_scaling_granularity(self):
        x = torch.tensor([[[[0., 0.], [-1., 1.], [-100., 100.]]]])
        for mode, shape in (("token", (1, 1, 3)), ("head", (1, 1, 1))):
            bits, scales = local.quantize(x, mode)
            self.assertEqual(scales.shape, shape)
            self.assertEqual(bits.dtype, torch.uint8)
            result = local.dequantize(bits, scales)
            self.assertTrue(torch.isfinite(result).all())
            torch.testing.assert_close(result.float(), x, atol=.04, rtol=.04)

    def test_bottom_right_causality(self):
        q = torch.zeros(1, 1, 2, 32).half()
        k = torch.zeros(1, 1, 4, 32).half()
        v = torch.arange(1, 5).reshape(1, 1, 4, 1).expand_as(k).half()
        actual = local.oracle(q, k, v)
        torch.testing.assert_close(actual[:, :, 0], torch.full((1, 1, 32), 2.).half())
        torch.testing.assert_close(actual[:, :, 1], torch.full((1, 1, 32), 2.5).half())

    def test_selected_kv_head_affects_only_its_query_group(self):
        torch.manual_seed(10)
        q = torch.randn(1, 12, 3, 32).half()
        k, v = torch.randn(1, 2, 19, 32).half(), torch.randn(1, 2, 19, 32).half()
        k8, ks = local.quantize(k)
        v8, vs = local.quantize(v)
        for mode in ("k", "v", "kv"):
            ek, ev = local.effective_kv(k, v, k8, v8, ks, vs, torch.tensor([1, 0]), mode)
            ref, actual = local.oracle(q, k, v), local.oracle(q, ek, ev)
            torch.testing.assert_close(ref[:, 6:], actual[:, 6:], atol=0, rtol=0)
            self.assertGreater(float((ref[:, :6] - actual[:, :6]).abs().max()), 0)

    def test_sweep_and_finite_checks(self):
        self.assertEqual(len(local.variants(2, True)), 9)
        self.assertEqual(len(local.variants(1, True)), 3)
        with self.assertRaises(ValueError):
            local.quantize(torch.full((1, 1, 1, 32), float("nan")))
        self.assertFalse(local.metrics(torch.zeros(1, 1, 1, 32),
                                      torch.full((1, 1, 1, 32), float("nan")))["finite"])


CUDA_TILES = (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
              and importlib.util.find_spec("triton") is not None)


@unittest.skipUnless(CUDA_TILES, "requires Ampere-or-newer CUDA GPU and Triton")
class TileAttentionCUDA(unittest.TestCase):
    @torch.no_grad()
    def test_tiles_match_staged_inputs_and_fp32_oracle(self):
        from experiments.quant.kernels.attention import attention
        from experiments.quant.projections import strict_accumulation
        torch.manual_seed(18)
        with strict_accumulation():
            for queries, length in ((1, 17), (7, 65), (33, 33)):
                q = torch.randn(2, 12, queries, 128, device="cuda", dtype=torch.float16)
                k = torch.randn(2, 2, length, 128, device="cuda", dtype=torch.float16)
                v = torch.randn_like(k)
                for scaling in ("token", "head"):
                    k8, ks = local.quantize(k, scaling)
                    v8, vs = local.quantize(v, scaling)
                    for selected in ([1, 1], [1, 0], [0, 1]):
                        mask = torch.tensor(selected, device="cuda", dtype=torch.int32)
                        for mode in ("off", "k", "v", "kv"):
                            ek, ev = local.effective_kv(k, v, k8, v8, ks, vs, mask, mode)
                            expected = local.oracle(q, ek, ev)
                            actual = attention(q, k, v, k8, v8, ks, vs, mask, mode)
                            staged = attention(q, ek, ev, k8, v8, ks, vs, mask, "off")
                            torch.testing.assert_close(actual, expected, atol=.01, rtol=.02)
                            torch.testing.assert_close(actual, staged, atol=.002, rtol=.003)


if __name__ == "__main__":
    unittest.main()
