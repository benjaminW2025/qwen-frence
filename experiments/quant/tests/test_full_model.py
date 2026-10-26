"""CPU experiment contracts plus explicitly gated real CUDA kernel checks."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.quant.projections import (FP8Linear, parse_groups,
    quantize_rows_reference, replace_projections, strict_accumulation)
from experiments.quant.staged_cache import StagedFP8KVCache

SPEC = importlib.util.spec_from_file_location(
    "fp8_accuracy_experiment", ROOT / "experiments/quant/full_model.py")
experiment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(experiment)


class FP8AccuracyContracts(unittest.TestCase):
    def test_quantization_scales_rows_independently_and_preserves_zero(self):
        x = torch.tensor([[0., 0., 0.], [-2., 0., 2.], [-200., 0., 200.]])
        q, scales = quantize_rows_reference(x)
        self.assertEqual(q.dtype, torch.float8_e4m3fn)
        self.assertEqual(scales.dtype, torch.float32)
        self.assertTrue(torch.isfinite(scales).all())
        torch.testing.assert_close(q.float() * scales[:, None], x, rtol=1e-6, atol=1e-6)
        self.assertEqual(float(q[0].float().abs().sum()), 0)

    def test_nonfinite_weights_rejected(self):
        for value in (float("inf"), float("nan")):
            with self.assertRaisesRegex(ValueError, "finite"):
                quantize_rows_reference(torch.tensor([[value]]))

    def test_weight_preparation_keeps_bias_and_no_fp16_weight(self):
        layer = torch.nn.Linear(16, 8, dtype=torch.float16)
        replacement = FP8Linear(layer)
        self.assertEqual(replacement.weight_fp8.shape, layer.weight.shape)
        self.assertEqual(replacement.weight_scales.shape, (8,))
        torch.testing.assert_close(replacement.bias, layer.bias)
        self.assertFalse(hasattr(replacement, "weight"))

    def test_selection_and_restoration_after_exception(self):
        model = SimpleNamespace(layers=[SimpleNamespace(
            gate_up_proj=torch.nn.Linear(16, 32, dtype=torch.float16),
            down_proj=torch.nn.Linear(16, 16, dtype=torch.float16)) for _ in range(2)])
        originals = [layer.gate_up_proj for layer in model.layers]
        down = [layer.down_proj for layer in model.layers]
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with replace_projections(model, ("gate-up",)):
                self.assertTrue(all(isinstance(layer.gate_up_proj, FP8Linear) for layer in model.layers))
                self.assertEqual([layer.down_proj for layer in model.layers], down)
                raise RuntimeError("intentional")
        self.assertEqual([layer.gate_up_proj for layer in model.layers], originals)

    def test_precision_flags_restored_on_exception(self):
        backend = torch.backends.cuda.matmul
        old = backend.allow_fp16_reduced_precision_reduction
        with self.assertRaises(RuntimeError):
            with strict_accumulation() as settings:
                self.assertTrue(all(value is False for value in settings.values()))
                raise RuntimeError()
        self.assertEqual(backend.allow_fp16_reduced_precision_reduction, old)

    def test_invalid_group_selection(self):
        for value in ("", "gate-up,gate-up", "lm-head"):
            with self.assertRaises(ValueError):
                parse_groups(value)

    def test_case_metadata_and_noncontiguous_pages(self):
        for kind, chunks, contexts in (("prefill", [19] * 4, [19] * 4),
                                      ("decode", [1] * 4, [18] * 4),
                                      ("mixed", [1, 1, 19, 19], [18, 18, 19, 19])):
            case = experiment.make_case(kind, 4, 19, 17, 100, 1, "cpu")
            self.assertEqual(case["chunks"], chunks)
            self.assertEqual(case["contexts"].tolist(), contexts)
            self.assertEqual(case["cu"].diff().tolist(), chunks)
            self.assertEqual(case["slots"].unique().numel(), sum(chunks))
            for row in range(4):
                start, end = case["cu"][row:row+2].tolist()
                for index in range(start, end):
                    pos = int(case["positions"][index])
                    expected = int(case["table"][row, pos // 16]) * 16 + pos % 16
                    self.assertEqual(int(case["slots"][index]), expected)
            self.assertEqual(case["table"].unique().numel(), case["blocks"])

    def test_missing_threshold_is_unassessed_not_pass(self):
        logits = torch.tensor([[1., 2., 3.], [3., 1., 0.]])
        result = experiment.compare_logits(logits, logits + .1)
        self.assertEqual(result["status"], "unassessed")
        self.assertEqual(result["top1_agreement"], 1.)
        self.assertEqual(result["reference_top2_margins"], [1., 2.])
        self.assertEqual(experiment.compare_logits(logits, logits + .1, .01, 0.)["status"], "fail")

    def test_token_agreement_and_nonfinite_failures(self):
        ref = torch.tensor([[1., 2.]])
        self.assertEqual(experiment.compare_logits(ref, ref.flip(-1), min_agreement=1)["status"], "fail")
        self.assertEqual(experiment.compare_logits(ref, ref * float("nan"))["status"], "fail")

    def test_cache_guard_detects_overwrite_outside_current_slots(self):
        pool = SimpleNamespace(k_pool=[torch.zeros(2, 16, 1, 2)],
                               v_pool=[torch.zeros(2, 16, 1, 2)])
        seed = [x.clone() for x in experiment.cache_state(pool)]
        case = {"slots": torch.tensor([17])}
        pool.k_pool[0].view(32, 1, 2)[17] = 1
        _, preserved = experiment.capture_writes(pool, case, seed)
        self.assertTrue(preserved)
        pool.k_pool[0].view(32, 1, 2)[0] = 1
        _, preserved = experiment.capture_writes(pool, case, seed)
        self.assertFalse(preserved)
        experiment.restore_cache(pool, seed)
        self.assertEqual(float(pool.k_pool[0].sum()), 0)

    def test_latency_summary_validates_samples(self):
        self.assertEqual(experiment.latency_summary([3., 1., 2.])["median_ms"], 2.)
        for values in ([], [0.], [float("nan")]):
            with self.assertRaises(ValueError):
                experiment.latency_summary(values)

    def test_context_plan_varies_decode_history_without_duplicating_prefill(self):
        args = SimpleNamespace(scenarios=["prefill", "decode", "mixed"],
                               prefixes=[256, 2048, 8192], samples=3, seed=19)
        plan = experiment.case_plan(args)
        self.assertEqual(len(plan), 21)
        self.assertEqual(sum(row["kind"] == "prefill" for row in plan), 3)
        for prefix in args.prefixes:
            rows = [row for row in plan if row["kind"] == "decode" and row["prefix"] == prefix]
            self.assertEqual([row["seed"] for row in rows], [19, 20, 21])

    def test_accuracy_aggregation_keeps_contexts_separate(self):
        def row(prefix, logits):
            return {"scenario": "decode", "cached_prefix_tokens": prefix,
                    "variant": "fp8-cache-kv", "logits": logits,
                    "speedup_vs_opening_control": {"synchronized_wall": .8}}
        ref = torch.tensor([[1., 2.]])
        same = experiment.compare_logits(ref, ref)
        different = experiment.compare_logits(ref, ref.flip(-1))
        result = experiment.aggregate_accuracy([row(256, same), row(256, different), row(2048, same)])
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["top1_agreement"], .5)
        self.assertEqual(result[0]["compared_requests"], 2)
        self.assertEqual(result[1]["top1_agreement"], 1.)

    def test_cache_modes_scaling_storage_and_reset(self):
        pool = SimpleNamespace(k_pool=[torch.randn(2, 16, 2, 128).half()],
                               v_pool=[torch.randn(2, 16, 2, 128).half()])
        seed = [x.clone() for x in experiment.cache_state(pool)]
        for mode in ("k", "v", "kv"):
            cache = StagedFP8KVCache(pool, seed, mode)
            self.assertEqual(len(cache.entries), len(mode))
            self.assertEqual(cache.storage_bytes(), len(mode) * (8192 + 64 * 4))
            for index, actual in enumerate(cache.expected_seed):
                selected = ("k" if index == 0 else "v") in mode
                if selected:
                    q, scale = quantize_rows_reference(seed[index].reshape(-1, 128))
                    expected = (q.float() * scale[:, None]).half().view_as(actual)
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                else:
                    self.assertIs(actual, seed[index])
            for entry in cache.entries.values():
                entry["q"].copy_(torch.zeros_like(entry["q"]))
                entry["scale"].zero_()
            cache.reset()
            for entry in cache.entries.values():
                torch.testing.assert_close(entry["q"].float(), entry["seed_q"].float(), atol=0, rtol=0)
                torch.testing.assert_close(entry["scale"], entry["seed_scale"], atol=0, rtol=0)

    def test_cache_hook_uses_live_page_mapping_and_restores_dispatch(self):
        import kernel_dispatch
        pool = SimpleNamespace(k_pool=[torch.zeros(4, 16, 1, 128).half()],
                               v_pool=[torch.zeros(4, 16, 1, 128).half()])
        seed = [x.clone() for x in experiment.cache_state(pool)]
        cache = StagedFP8KVCache(pool, seed)
        original = kernel_dispatch.fa3_paged_decode_attention
        table = torch.tensor([[0, 3], [2, 1]], dtype=torch.int32)
        with mock.patch.object(kernel_dispatch, "fa3_paged_decode_attention", return_value="attention") as attention:
            with mock.patch.object(cache, "stage") as stage:
                with self.assertRaisesRegex(RuntimeError, "intentional"):
                    with cache.attention_hook():
                        result = kernel_dispatch.fa3_paged_decode_attention(
                            None, pool.k_pool[0], pool.v_pool[0], table, torch.tensor([17, 32]))
                        self.assertEqual(result, "attention")
                        self.assertEqual(stage.call_args_list[0].args[1].tolist(), [48, 31])
                        self.assertEqual(stage.call_count, 2)
                        raise RuntimeError("intentional")
                self.assertIs(kernel_dispatch.fa3_paged_decode_attention, attention)
        self.assertIs(kernel_dispatch.fa3_paged_decode_attention, original)


def hopper_available():
    return (torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)
            and importlib.util.find_spec("triton") is not None)


@unittest.skipUnless(hopper_available(), "requires Hopper CUDA and Triton")
class FP8CUDAKernels(unittest.TestCase):
    @torch.no_grad()
    def test_quantization_gemm_and_graph_replay_with_changed_inputs(self):
        from experiments.quant.kernels.projections import quantize_activations
        with strict_accumulation():
            for m, k, n in ((1, 33, 17), (17, 160, 96), (8, 1536, 2048), (3, 8960, 1536)):
                layer = torch.nn.Linear(k, n, bias=(n != 17), device="cuda", dtype=torch.float16)
                projection = FP8Linear(layer)
                x = torch.randn(m, k, device="cuda", dtype=torch.float16) * .1
                x[0].zero_()
                q, scales = quantize_activations(x)
                rq, rs = quantize_rows_reference(x)
                torch.testing.assert_close(q.float(), rq.float(), atol=0, rtol=0)
                torch.testing.assert_close(scales, rs, atol=0, rtol=0)
                w = projection.weight_fp8.float() * projection.weight_scales[:, None]
                expected = (q.float() * scales[:, None]) @ w.T
                if projection.bias is not None:
                    expected += projection.bias.float()
                torch.testing.assert_close(projection(x), expected.half(), atol=.003, rtol=.005)
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    projection(x)
                torch.cuda.current_stream().wait_stream(side)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = projection(x)
                for factor in (0., -.5, 2.):
                    x.copy_(torch.randn_like(x) * factor)
                    eager = projection(x)
                    graph.replay()
                    torch.testing.assert_close(captured, eager, atol=0, rtol=0)

    @unittest.skipUnless(importlib.util.find_spec("vllm") is not None, "requires bundled FA3")
    @torch.no_grad()
    def test_real_forward_prefix_reset_and_selected_projections(self):
        from naive_forward import Model, Qwen2Config
        from model_adapter import allocate_pool
        cfg = Qwen2Config(vocab=128, d_model=256, d_ff=320, n_layers=2,
                          n_heads=2, n_kv_heads=1, use_custom_kernels=True)
        model = Model(cfg).cuda().half().eval()
        model.pack_projections_()
        with strict_accumulation():
            for kind in ("prefill", "decode", "mixed"):
                case = experiment.make_case(kind, 2, 5, 17, cfg.vocab, 5, "cuda")
                pool = allocate_pool(cfg, case["blocks"], "cuda")
                for tensor in experiment.cache_state(pool):
                    tensor.zero_()
                experiment.prepare_prefix(model, pool, case)
                seed = [x.clone() for x in experiment.cache_state(pool)]
                adapter, run = experiment.build_forward(model, pool, case)
                run()
                experiment.restore_cache(pool, seed)
                reference = run().clone()
                self.assertEqual(reference.shape, (2, cfg.vocab))
                del adapter, run
                for group in ("gate-up", "down", "output", "qkv"):
                    with replace_projections(model, (group,)):
                        adapter, run = experiment.build_forward(model, pool, case)
                        run()
                        experiment.restore_cache(pool, seed)
                        actual = run().clone()
                        self.assertTrue(torch.isfinite(actual).all())
                        _, preserved = experiment.capture_writes(pool, case, seed)
                        self.assertTrue(preserved)
                        experiment.restore_cache(pool, seed)
                        torch.testing.assert_close(actual, run(), atol=0, rtol=0)
                        del adapter, run
                if kind == "decode":
                    for mode in ("k", "v", "kv"):
                        shadow = StagedFP8KVCache(pool, seed, mode)
                        with shadow.attention_hook():
                            adapter, run = experiment.build_forward(model, pool, case)
                            run()
                            experiment.restore_cache(pool, seed)
                            shadow.reset()
                            actual = run().clone()
                            self.assertTrue(torch.isfinite(actual).all())
                            _, preserved = experiment.capture_writes(pool, case, shadow.expected_seed)
                            self.assertTrue(preserved)
                            for tensor in experiment.cache_state(pool):
                                entry = shadow.entries.get(tensor.data_ptr())
                                if entry is not None:
                                    torch.testing.assert_close(tensor, shadow.dequantized(entry), atol=0, rtol=0)
                            experiment.restore_cache(pool, seed)
                            shadow.reset()
                            torch.testing.assert_close(actual, run(), atol=0, rtol=0)
                            del adapter, run


if __name__ == "__main__":
    unittest.main()
