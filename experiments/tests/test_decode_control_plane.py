"""CPU-only contract checks for the FA3 K-step/control-plane experiment."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "experiments/integration/benchmark_decode_control_plane.py"
SPEC = importlib.util.spec_from_file_location("benchmark_decode_control_plane", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DecodeControlPlaneTests(unittest.TestCase):
    def test_fixed_regime_output_head_configs_reuse_measured_winners(self):
        b8 = MODULE.output_head_config(8)
        self.assertEqual((b8["block_m"], b8["block_n"], b8["block_k"]),
                         (16, 256, 64))
        self.assertEqual((b8["num_warps"], b8["num_stages"]), (8, 3))

        b64 = MODULE.output_head_config(64)
        self.assertEqual((b64["block_m"], b64["block_n"], b64["block_k"]),
                         (64, 64, 64))
        self.assertEqual((b64["num_warps"], b64["num_stages"]), (4, 3))

    def test_plan_defaults_to_accepted_fusions(self):
        args = MODULE.build_parser().parse_args(["plan"])
        MODULE.validate(args)
        self.assertEqual(args.qkv_mode, "native")
        self.assertTrue(args.residual_rmsnorm)
        self.assertEqual(MODULE.STEPS, (2, 4, 8))
        self.assertEqual(MODULE.GRAPH_STEPS, (1, 2, 4, 8))

    def test_invalid_output_head_batch_fails_before_gpu_work(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            MODULE.output_head_config(0)


if __name__ == "__main__":
    unittest.main()
