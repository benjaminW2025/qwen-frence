"""CPU-only checks for the focused C++/Python control-plane profiler."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

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


@unittest.skipIf(cpp is None, "build C++ extension first")
class CppRangeTests(unittest.TestCase):
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
