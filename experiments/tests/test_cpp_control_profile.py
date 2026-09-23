"""CPU-only checks for the focused C++/Python control-plane profiler."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "engine/cpp/build"))
PATH = ROOT / "experiments/integration/profile_cpp_control.py"
SPEC = importlib.util.spec_from_file_location("profile_cpp_control", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
try:
    import inference_engine_cpp as cpp
except ImportError:
    cpp = None


class ProfileDesignTests(unittest.TestCase):
    def test_case_and_target_selection(self):
        args = MODULE.build_parser().parse_args([])
        MODULE.validate_args(args)
        self.assertEqual(MODULE.select_target(
            [{"kind": "prefill"}, {"kind": "decode"}, {"kind": "decode"}],
            "decode", 1), 2)
        with self.assertRaisesRegex(ValueError, "occurrence"):
            MODULE.select_target([{"kind": "prefill"}], "decode", 0)

    def test_invalid_case_and_negative_repetitions_fail(self):
        args = MODULE.build_parser().parse_args(["--case-id", "missing"])
        with self.assertRaisesRegex(ValueError, "case ID"):
            MODULE.validate_args(args)
        args = MODULE.build_parser().parse_args(["--repetitions", "0"])
        with self.assertRaisesRegex(ValueError, "repetitions"):
            MODULE.validate_args(args)
        args = MODULE.build_parser().parse_args(["--prefill-budget", "0"])
        with self.assertRaisesRegex(ValueError, "prefill-budget"):
            MODULE.validate_args(args)

    def test_prefill_fusion_profile_requires_piecewise_adapter(self):
        args = MODULE.build_parser().parse_args([
            "--preset", "fixed", "--case-id", "fixed-b64-l2048-o128",
            "--kind", "prefill", "--prefill-budget", "8192",
            "--decode-attention-policy", "fa3",
            "--prefill-fusion-mode", "swiglu",
        ])
        MODULE.validate_args(args)
        args.adapter = "eager-prefill"
        with self.assertRaisesRegex(ValueError, "piecewise-prefill"):
            MODULE.validate_args(args)

    def test_cuda_activity_summary_counts_leaf_events_once(self):
        events = [
            SimpleNamespace(name="decode_attention_kernel", device_type="DeviceType.CUDA",
                            self_device_time_total=120.0),
            SimpleNamespace(name="fused_rms_norm", device_type="DeviceType.CUDA",
                            self_device_time_total=20.0),
            SimpleNamespace(name="nvjet_gemm", device_type="DeviceType.CUDA",
                            self_device_time_total=60.0),
            SimpleNamespace(name="python/parent", device_type="DeviceType.CPU",
                            self_device_time_total=200.0),
            SimpleNamespace(name="python/model_callback_decode",
                            device_type="DeviceType.CUDA",
                            self_device_time_total=200.0),
        ]
        summary = MODULE.cuda_activity_summary(events)
        self.assertEqual(summary["summed_cuda_activity_us"], 200.0)
        self.assertEqual(summary["activity_count"], 3)
        categories = {row["category"]: row for row in summary["categories"]}
        self.assertEqual(categories["attention"]["percent_of_cuda_activity"], 60.0)
        self.assertEqual(categories["gemm"]["total_us"], 60.0)

    def test_flash_attention_is_not_misbucketed_as_gemm(self):
        self.assertEqual(MODULE.cuda_kernel_category(
            "void cutlass::device_kernel<flash::FlashAttnFwdSm90>()"), "attention")
        self.assertEqual(MODULE.cuda_kernel_category(
            "void flash::flash_fwd_splitkv_kernel<cutlass::half_t>()"), "attention")
        self.assertEqual(MODULE.cuda_kernel_category(
            "vllm::reshape_and_cache_flash_kernel"), "kv_write")

    def test_profile_callback_defers_context_device_copy(self):
        class Adapter:
            def __call__(self, *args):
                return "logits"

        callback = MODULE.ProfileCallback(Adapter())
        context = torch.tensor([513, 1025])
        args = (torch.tensor([1, 2]), None, None, None, context, None, 1, True)
        self.assertEqual(callback(*args), "logits")
        self.assertEqual(callback.calls[0]["context"].data_ptr(), context.data_ptr())
        self.assertEqual(callback.materialize()[0]["context_lengths"], [513, 1025])

    def test_fixed_profile_selects_real_full_batch_decode_and_packed_prefill(self):
        from fixed_regime import get_fixed_case

        case = get_fixed_case("probe-b64-l4096-o256")
        args = MODULE.build_parser().parse_args(
            ["--preset", "fixed", "--case-id", case["id"],
             "--decode-attention-policy", "fa3", "--qkv-mode", "native"])
        MODULE.validate_args(args)
        steps = [{"kind": "prefill", "calls": [(False, 2048, 1, 2048)]},
                 {"kind": "decode", "calls": [(True, 1, 1, 1)]},
                 {"kind": "decode", "calls": [(True, 64, 64, 1)]}]
        self.assertEqual(MODULE.select_fixed_target(steps, case, "prefill", 0), 0)
        self.assertEqual(MODULE.select_fixed_target(steps, case, "decode", 0), 2)
        with self.assertRaisesRegex(ValueError, "eligible decode"):
            MODULE.select_fixed_target(steps, case, "decode", 1)


@unittest.skipIf(cpp is None, "build C++ extension first")
class CppRangeTests(unittest.TestCase):
    def test_every_fixed_row_has_targetable_decode_and_prefill_steps(self):
        from benchmark_integrated_graph import dry_schedule
        from fixed_regime import FIXED_SHAPES, get_fixed_case, verify_fixed_result

        for shape in FIXED_SHAPES:
            case = get_fixed_case(shape["id"])
            result = dry_schedule(torch, cpp, case, 20260914)
            verify_fixed_result(result, shape["id"])
            for kind in ("decode", "prefill"):
                target = MODULE.select_fixed_target(result["steps"], case, kind, 0)
                self.assertEqual(result["steps"][target]["kind"], kind)

    def test_scheduler_ranges_appear_in_torch_profiler(self):
        from torch.profiler import ProfilerActivity, profile

        config = cpp.SchedulerConfig()
        config.max_batch_size = 1
        config.max_prefill_tokens_per_iter = 4
        config.max_context_length = 4
        config.block_size = 2
        config.num_kv_heads = 1
        config.head_dim = 4
        config.eos_token_id = -1
        loop = cpp.IterationLoop(config, torch.device("cpu"))
        loop.submit_request([1, 2], 1)

        def forward(ids, positions, slots, cu, context, table, max_query, decode):
            return torch.zeros((context.numel(), 8))

        with profile(activities=[ProfilerActivity.CPU]) as prof:
            loop.step(forward)
        names = {row["name"] for row in MODULE.stage_summary(prof)}
        self.assertTrue({"cpp/step", "cpp/schedule", "cpp/build_prefill_batch",
                         "cpp/callback_prefill", "cpp/sample_argmax",
                         "cpp/sample_device_to_host"}.issubset(names))


if __name__ == "__main__":
    unittest.main()
