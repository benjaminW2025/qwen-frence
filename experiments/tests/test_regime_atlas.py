"""CPU-only controls for the detailed regime-atlas runner."""

import importlib.util
from pathlib import Path
import sys
import unittest


PATH = Path(__file__).resolve().parents[1] / "profiling" / "run_regime_atlas.py"
SPEC = importlib.util.spec_from_file_location("run_regime_atlas", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class RegimeAtlasTests(unittest.TestCase):
    def test_default_regimes_cover_distinct_failure_modes(self):
        args = MODULE.build_parser().parse_args([])
        self.assertEqual(len(args.regimes), 8)
        self.assertIn("launch_bound", args.regimes)
        self.assertIn("decode_low_concurrency", args.regimes)
        self.assertIn("long_fresh_prefill", args.regimes)
        self.assertIn("dual_heavy", args.regimes)

    def test_command_preserves_regime_shape(self):
        args = MODULE.build_parser().parse_args([])
        regime = next(item for item in MODULE.REGIMES if item.name == "balanced_resumed")
        command = MODULE.command_for(args, regime, Path("/tmp/atlas"))
        self.assertEqual(command[command.index("--decode-requests") + 1], "32")
        self.assertEqual(command[command.index("--prefill-prefix-length") + 1], "4096")
        self.assertEqual(command[command.index("--prefill-chunk-size") + 1], "2048")


if __name__ == "__main__":
    unittest.main()
