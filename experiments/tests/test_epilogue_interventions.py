"""CPU mathematical/routing checks; the actual CUDA epilogues still require the GPU gates."""
from contextlib import ExitStack
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from experiments.integration import ab_gemm_epilogues
from naive_forward import Model, Qwen2Config
import kernel_dispatch
with mock.patch.dict("sys.modules", {
    "paged_decode_attention": SimpleNamespace(paged_decode_attention_dispatch=None,
                                               resolve_decode_attention_policy=lambda p, *a: p),
    "paged_forward": SimpleNamespace(build_rope_from_positions=None),
}):
    import paged_graph_decoder as decode

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("fused_epilogue_cpu", ROOT / "custom_kernels/fused_gemm.py")
fused = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fused)
SPEC = importlib.util.spec_from_file_location("piecewise_epilogue_cpu", ROOT / "engine/graph/piecewise_prefill.py")
prefill = importlib.util.module_from_spec(SPEC)
with mock.patch.dict("sys.modules", {"ragged_prefill": SimpleNamespace(_rope_factors=lambda *a: (None, None))}):
    SPEC.loader.exec_module(prefill)


class SelectiveEpilogues(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.cfg = Qwen2Config(vocab=32, d_model=16, d_ff=32, n_layers=2,
                              n_heads=2, n_kv_heads=1, d_head=8,
                              max_seq_len=64, use_custom_kernels=False)
        self.model = Model(self.cfg).eval()
        self.model.pack_projections_()
        self.calls = []
        self.extension = SimpleNamespace(intervention_ops=fused.intervention_ops,
                                        prepare_gate_only=fused.prepare_gate_only,
                                        residual_gemm=self.residual, gate_up_swiglu=self.gate,
                                        qkv_rope_cache=self.qkv)

    def pool(self):
        return SimpleNamespace(k_pool=[torch.zeros(1, 16, 1, 8) for _ in range(2)],
                               v_pool=[torch.zeros(1, 16, 1, 8) for _ in range(2)])

    def residual(self, x, weight, residual, **kwargs):
        self.calls.append("residual")
        self.assertFalse(kwargs["partials"])
        return residual + x @ weight.T, None

    def gate(self, x, weight):
        self.calls.append("gate-up")
        gate, up = fused.split_interleaved(x @ weight.T)
        return torch.nn.functional.silu(gate) * up

    def qkv(self, x, weight, bias, partials, *, positions, slots, k_pool, v_pool,
            valid_tokens=None, **kwargs):
        self.calls.append("qkv")
        self.assertIsNone(partials)
        packed = torch.nn.functional.linear(x, weight, bias)
        q, k, v = packed.split((16, 8, 8), dim=-1)
        live = x.shape[0] if valid_tokens is None else int(valid_tokens)
        k_pool.view(-1, 1, 8)[slots[:live]] = k[:live].view(live, 1, 8)
        v_pool.view(-1, 1, 8)[slots[:live]] = v[:live].view(live, 1, 8)
        q = q.view(x.shape[0], 2, 8)
        q[live:] = 0
        return q

    def write(self, k, v, slots, valid, k_pool, v_pool):
        live = int(valid)
        k_pool.view(-1, 1, 8)[slots[:live]] = k[0, :, :live].transpose(0, 1)
        v_pool.view(-1, 1, 8)[slots[:live]] = v[0, :, :live].transpose(0, 1)

    def patches(self):
        stack = ExitStack()
        stack.enter_context(torch.no_grad())
        stack.enter_context(mock.patch.object(kernel_dispatch, "_load", return_value=self.extension))
        stack.enter_context(mock.patch.object(prefill, "apply_rope", side_effect=lambda x, *a: x))
        stack.enter_context(mock.patch.object(decode, "apply_rope", side_effect=lambda x, *a: x))
        stack.enter_context(mock.patch.object(decode, "build_rope_from_positions", return_value=(None, None)))
        stack.enter_context(mock.patch.object(decode, "paged_decode_attention_dispatch", side_effect=lambda q, *a, **k: q * .5))
        stack.enter_context(mock.patch.object(kernel_dispatch, "masked_kv_write", side_effect=self.write))
        def residual_norm(residual, branch, gamma, epsilon):
            x = residual + branch
            norm = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + epsilon) * gamma).to(x.dtype)
            return x, norm
        stack.enter_context(mock.patch.object(kernel_dispatch, "residual_add_rms_norm", side_effect=residual_norm))
        return stack

    def check_calls(self, intervention):
        ops = fused.intervention_ops(intervention)
        self.assertEqual(self.calls.count("gate-up"), 2 if "gate-up" in ops else 0)
        self.assertEqual(self.calls.count("qkv"), 2 if "qkv" in ops else 0)
        self.assertEqual(self.calls.count("residual"), 2 * len(ops & {"residual-o", "residual-down"}))

    def test_decode_preserves_nonselected_operations_and_output_shape(self):
        ids = torch.tensor([[1], [2], [3]])
        positions = torch.zeros(3, dtype=torch.int32)
        lengths = torch.ones(3, dtype=torch.int32)
        table = torch.zeros(3, 1, dtype=torch.int32)
        slots = torch.arange(3)
        with self.patches():
            control_pool = self.pool()
            control = decode.graph_decode_forward(self.model, control_pool, ids, positions,
                                                   lengths, table, slots, enable_residual_rmsnorm=True)
            for intervention in ab_gemm_epilogues.INTERVENTIONS[1:]:
                with self.subTest(intervention=intervention):
                    self.calls.clear()
                    pool = self.pool()
                    actual = decode.graph_decode_forward(self.model, pool, ids, positions,
                                                          lengths, table, slots,
                                                          enable_residual_rmsnorm=True,
                                                          gemm_epilogue_intervention=intervention)
                    torch.testing.assert_close(actual, control)
                    for a, b in zip(pool.k_pool + pool.v_pool, control_pool.k_pool + control_pool.v_pool):
                        torch.testing.assert_close(a, b)
                    self.check_calls(intervention)

    def run_prefill(self, intervention):
        pool = self.pool()
        tokens = 3
        positions, slots = torch.zeros(tokens, dtype=torch.long), torch.arange(tokens)
        valid = torch.tensor(tokens, dtype=torch.int32)
        result = None
        for index in range(3):
            boundary = prefill._AttentionBoundary(self.model, pool, index, tokens,
                positions, slots, valid, capture_graph=False,
                enable_fused_gemm_epilogues=intervention is not None,
                enable_residual_rmsnorm=True,
                gemm_epilogue_intervention=intervention or "all")
            if index == 0:
                boundary.ids.copy_(torch.tensor([1, 2, 3]))
            else:
                q, residual = result
                boundary.residual.copy_(residual)
                boundary.attention.copy_(q[0].transpose(0, 1).reshape(tokens, 16) * .5)
            result = boundary.run_segment()
        return result, pool

    def test_prefill_preserves_rmsnorm_and_matches_decode_interventions(self):
        with self.patches():
            control, control_pool = self.run_prefill(None)
            for intervention in ab_gemm_epilogues.INTERVENTIONS[1:]:
                with self.subTest(intervention=intervention):
                    self.calls.clear()
                    actual, pool = self.run_prefill(intervention)
                    torch.testing.assert_close(actual, control)
                    for a, b in zip(pool.k_pool + pool.v_pool, control_pool.k_pool + control_pool.v_pool):
                        torch.testing.assert_close(a, b)
                    self.check_calls(intervention)


if __name__ == "__main__":
    unittest.main()
