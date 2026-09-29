"""CPU contracts for the fused-epilogue GEMMs; not CUDA kernel validation.

Each test pins down math or index arithmetic the CUDA epilogues depend on,
against an independent float64 formulation of the unfused engine path.
"""
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('fused_gemm_contract', ROOT / 'custom_kernels/fused_gemm.py')
fused = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fused)

EPS = 1e-6


def rmsnorm(x, gamma):
    return gamma * x / torch.sqrt(x.square().mean(-1, keepdim=True) + EPS)


def row_factor(partials, hidden):
    return 1 / torch.sqrt(partials.sum(-1, keepdim=True) / hidden + EPS)


def rope(heads, positions, theta):
    # Rotate-half RoPE exactly as packed_qkv_rope_cache and the QKV epilogue compute it.
    d = torch.arange(64, dtype=torch.float64)
    angle = positions.double()[:, None, None] * torch.exp(d * (-math.log(theta) / 64))
    first, second = heads[..., :64], heads[..., 64:]
    return torch.cat((first * angle.cos() - second * angle.sin(),
                      first * angle.sin() + second * angle.cos()), dim=-1)


def thread_column(c0, j):
    # Accumulator geometry proven by validate_fragment at import.
    return c0 + j % 2 + 8 * (j // 4)


class FusedGemmContract(unittest.TestCase):
    def setUp(self):
        self.generator = torch.Generator().manual_seed(7)

    def randn(self, *shape, scale=1.0):
        return torch.randn(*shape, generator=self.generator, dtype=torch.float64) * scale

    def test_norm_fold_reproduces_rmsnorm_then_linear(self):
        x = self.randn(9, 512, scale=3)
        gamma = self.randn(512).abs() + .5
        weight = self.randn(40, 512, scale=.05)
        partials = fused.row_square_partials(x)
        self.assertEqual(partials.shape, (9, 2))
        torch.testing.assert_close(partials.sum(-1), x.square().sum(-1).float(), rtol=1e-6, atol=1e-4)
        folded = x @ (weight * gamma[None, :]).T * row_factor(partials.double(), 512)
        torch.testing.assert_close(folded, rmsnorm(x, gamma) @ weight.T, rtol=1e-6, atol=1e-6)

    def test_fold_norm_rounds_once_in_the_weight_dtype(self):
        weight = self.randn(8, 16).half()
        gamma = (self.randn(16).abs() + .5).half()
        folded = fused.fold_norm(weight, gamma)
        self.assertEqual(folded.dtype, torch.float16)
        torch.testing.assert_close(folded, (weight.float() * gamma.float()).half(), rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, 'input dimension'):
            fused.fold_norm(weight, gamma[:8])

    def test_interleave_places_each_up_column_eight_after_its_gate(self):
        gate, up = self.randn(48, 16), self.randn(48, 16)
        x = self.randn(5, 16)
        output = x @ fused.interleave_gate_up(gate, up).T
        for column in range(output.shape[1]):
            source = (column // 16) * 8 + column % 8
            expected = (x @ gate.T if column % 16 < 8 else x @ up.T)[:, source]
            torch.testing.assert_close(output[:, column], expected)
        split_gate, split_up = fused.split_interleaved(output)
        torch.testing.assert_close(split_gate, x @ gate.T)
        torch.testing.assert_close(split_up, x @ up.T)

    def test_swiglu_epilogue_with_row_scale_matches_norm_then_swiglu(self):
        x = self.randn(6, 256, scale=2)
        gamma = self.randn(256).abs() + .5
        gate, up = self.randn(32, 256, scale=.05), self.randn(32, 256, scale=.05)
        accumulators = x @ fused.interleave_gate_up(gate * gamma, up * gamma).T
        scale = row_factor(fused.row_square_partials(x).double(), 256)
        result = torch.empty(6, 32, dtype=torch.float64)
        for n in range(accumulators.shape[1]):
            if n % 16 < 8:  # gate column; its up partner is n + 8
                g, u = accumulators[:, n:n + 1] * scale, accumulators[:, n + 8:n + 9] * scale
                result[:, (n // 16) * 8 + n % 8] = (g * torch.sigmoid(g) * u)[:, 0]
        h = rmsnorm(x, gamma)
        torch.testing.assert_close(result, torch.nn.functional.silu(h @ gate.T) * (h @ up.T))

    def test_fragment_geometry_gives_every_pairing_the_epilogues_use(self):
        for c0 in (0, 2, 4, 6):
            for j in range(0, 128, 2):
                column = thread_column(c0, j)
                self.assertEqual(thread_column(c0, j + 1), column + 1)
                if (j // 4) % 2 == 0:                   # SwiGLU gate element -> up partner at j + 4
                    self.assertEqual(column % 16 < 8, True)
                    self.assertEqual(thread_column(c0, j + 4), column + 8)
                first_half = (j // 4) % 16 < 8          # the QKV epilogue's static test
                self.assertEqual(first_half, column % 128 < 64)
                if first_half:                          # RoPE partner at j + 32
                    self.assertEqual(thread_column(c0, j + 32), column + 64)

    def test_qkv_tiles_heads_and_cache_offsets(self):
        head_dim = 128
        # Prefill tiles are 256 columns (two heads), decode tiles 128 (one head).
        for tile_n in (256, 128):
            kinds = []
            for tile in range(fused.QKV_WIDTH // tile_n):
                # The QKV epilogue's classification of a tile index.
                is_q = tile < 12 * head_dim // tile_n
                is_v = tile >= 14 * head_dim // tile_n
                heads = {n // head_dim for n in range(tile * tile_n, (tile + 1) * tile_n)}
                expected = {h for h in heads}
                kinds.append('q' if is_q else 'v' if is_v else 'k')
                self.assertTrue(all(h < 12 for h in expected) if is_q else
                                all(h >= 14 for h in expected) if is_v else
                                all(12 <= h < 14 for h in expected), (tile_n, tile, heads))
                if not is_q:
                    self.assertTrue(all(h - (14 if is_v else 12) in (0, 1) for h in heads))
            self.assertEqual(kinds.count('q') * tile_n, 12 * head_dim)
            self.assertEqual(kinds.count('k') * tile_n, 2 * head_dim)
            self.assertEqual(kinds.count('v') * tile_n, 2 * head_dim)
        pool = self.randn(5, 16, 2, 128)
        flat = pool.reshape(-1)
        for slot in (0, 17, 79):
            for kv_head in (0, 1):
                for d in (0, 63, 64, 127):
                    self.assertEqual(flat[(slot * 2 + kv_head) * head_dim + d],
                                     pool[slot // 16, slot % 16, kv_head, d])

    def test_qkv_epilogue_model_matches_the_unfused_path(self):
        rows, hidden, theta = 7, 256, 1_000_000.0
        x = self.randn(rows, hidden, scale=2)
        gamma = self.randn(hidden).abs() + .5
        weight = self.randn(fused.QKV_WIDTH, hidden, scale=.05)
        bias = self.randn(fused.QKV_WIDTH, scale=.1)
        positions = torch.tensor([0, 1, 5, 31, 900, 4095, 8191])
        # Unfused: rmsnorm -> linear + bias -> split heads -> RoPE on Q and K.
        packed = rmsnorm(x, gamma) @ weight.T + bias
        heads = packed.reshape(rows, 16, 128)
        expected_q, expected_k = rope(heads[:, :12], positions, theta), rope(heads[:, 12:14], positions, theta)
        expected_v = heads[:, 14:]
        # Fused: folded weight, per-row scale then bias in the epilogue, RoPE on (d, d + 64) pairs.
        accumulators = x @ fused.prepare_qkv(weight, gamma).T
        values = (accumulators * row_factor(fused.row_square_partials(x).double(), hidden) + bias).reshape(rows, 16, 128)
        # Partials are FP32, as the kernel writes them, so r carries FP32 rounding.
        fp32 = dict(rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(rope(values[:, :12], positions, theta), expected_q, **fp32)
        torch.testing.assert_close(rope(values[:, 12:14], positions, theta), expected_k, **fp32)
        torch.testing.assert_close(values[:, 14:], expected_v, **fp32)

    def test_prepared_model_layer_matches_the_unfused_layer(self):
        # A float64 stand-in for the extension, written from the kernels' documented
        # contracts, so this checks prepare_model's gamma routing and the wiring of
        # attention_inputs/layer_tail, not CUDA.
        hidden, ff, rows, theta = 256, 32, 5, 1_000_000.0
        cfg = SimpleNamespace(rms_norm_eps=EPS, rope_theta=theta)

        class Extension:
            @staticmethod
            def residual_gemm(x, w, residual, partials, out, config):
                y = residual + x @ w.T
                return y, fused.row_square_partials(y).double() if partials else None

            @staticmethod
            def gate_up_swiglu(x, w, partials, hidden_size, eps, config):
                gate, up = fused.split_interleaved((x @ w.T) * row_factor(partials, hidden_size))
                return torch.nn.functional.silu(gate) * up

            @staticmethod
            def qkv_rope_cache(x, w, bias, partials, hidden_size, eps, positions, slots,
                               k_pool, v_pool, valid, theta_, config):
                heads = ((x @ w.T) * row_factor(partials, hidden_size) + bias).reshape(len(x), 16, 128)
                k_pool.view(-1, 2, 128)[slots] = rope(heads[:, 12:14], positions, theta_)
                v_pool.view(-1, 2, 128)[slots] = heads[:, 14:]
                return rope(heads[:, :12], positions, theta_)

        def linear(out_features, in_features, bias=False):
            return SimpleNamespace(weight=self.randn(out_features, in_features, scale=.05),
                                   bias=self.randn(out_features, scale=.1) if bias else None)

        def norm():
            return SimpleNamespace(weight=self.randn(hidden).abs() + .5)

        layer = SimpleNamespace(qkv_proj=linear(fused.QKV_WIDTH, hidden, True),
                                o_proj=linear(hidden, 1536), gate_up_proj=linear(2 * ff, hidden),
                                down_proj=linear(hidden, ff), input_norm=norm(), post_attn_norm=norm())
        model = SimpleNamespace(layers=[layer])
        weights = fused.prepare_model(model)
        self.assertIs(fused.prepare_model(model), weights)  # idempotent: prepared once

        x = self.randn(rows, hidden, scale=2)
        positions, slots = torch.tensor([0, 3, 17, 40, 41]), torch.tensor([0, 3, 17, 40, 41])
        pools = [torch.zeros(4, 16, 2, 128, dtype=torch.float64) for _ in range(4)]
        attention = self.randn(rows, 1536)
        with mock.patch.object(fused, '_extension', return_value=Extension):
            q = fused.attention_inputs(weights[0], x, fused.row_square_partials(x).double(), cfg,
                                       positions=positions, slots=slots, k_pool=pools[0], v_pool=pools[1])
            out, partials = fused.layer_tail(weights[0], attention, x, cfg)

        heads = (rmsnorm(x, layer.input_norm.weight) @ layer.qkv_proj.weight.T
                 + layer.qkv_proj.bias).reshape(rows, 16, 128)
        fp32 = dict(rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(q, rope(heads[:, :12], positions, theta), **fp32)
        torch.testing.assert_close(pools[0].view(-1, 2, 128)[slots], rope(heads[:, 12:14], positions, theta), **fp32)
        torch.testing.assert_close(pools[1].view(-1, 2, 128)[slots], heads[:, 14:], **fp32)
        middle = x + attention @ layer.o_proj.weight.T
        gate, up = (rmsnorm(middle, layer.post_attn_norm.weight) @ layer.gate_up_proj.weight.T).chunk(2, -1)
        expected = middle + (torch.nn.functional.silu(gate) * up) @ layer.down_proj.weight.T
        torch.testing.assert_close(out, expected, **fp32)
        torch.testing.assert_close(partials.sum(-1), expected.square().sum(-1), rtol=1e-5, atol=1e-3)

    def test_prepare_model_requires_packed_projections(self):
        layer = SimpleNamespace(qkv_proj=None, gate_up_proj=None)
        with self.assertRaisesRegex(ValueError, 'packed QKV and gate/up'):
            fused.prepare_model(SimpleNamespace(layers=[layer]))

    def test_prepare_rejects_wrong_shapes(self):
        with self.assertRaisesRegex(ValueError, '2048 rows'):
            fused.prepare_qkv(self.randn(2040, 16), self.randn(16))
        with self.assertRaisesRegex(ValueError, 'multiple of 8'):
            fused.interleave_gate_up(self.randn(44, 16), self.randn(44, 16))
        with self.assertRaisesRegex(ValueError, 'multiple of 256'):
            fused.row_square_partials(self.randn(3, 300))

    def test_stale_binary_is_rejected_before_fragment_validation(self):
        validate = mock.Mock()
        binary = mock.Mock(__file__='/tmp/stale.so', abi_version=1, validate_fragment=validate)
        fused._extension.cache_clear()
        with mock.patch.dict('sys.modules', {'inference_gemm_epilogue': binary}):
            with self.assertRaisesRegex(RuntimeError, 'Stale GEMM epilogue extension'):
                fused._extension()
            binary.abi_version = fused.ABI_VERSION
            binary.source_sha256 = 'old'
            with self.assertRaisesRegex(RuntimeError, 'source hash differs'):
                fused._extension()
        validate.assert_not_called()
        fused._extension.cache_clear()


if __name__ == '__main__':
    unittest.main()
