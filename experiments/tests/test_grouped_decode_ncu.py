"""Host-side CLI contract for the matched grouped-decode NCU capture."""

from pathlib import Path
import importlib.util
import unittest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "experiments" / "decode" / "profile_grouped_decode_ncu.py"
SPEC = importlib.util.spec_from_file_location("profile_grouped_decode_ncu", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ProfileContractTests(unittest.TestCase):
    def test_defaults_match_the_failed_speedup_cell(self):
        args = MODULE.build_parser().parse_args(["--arm", "native"])
        self.assertEqual((args.batch_size, args.context_length, args.split_k),
                         (64, 4096, 22))

    def test_arm_is_required_and_matched(self):
        with self.assertRaises(SystemExit):
            MODULE.build_parser().parse_args([])
        self.assertEqual(MODULE.build_parser().parse_args(
            ["--arm", "current"]).arm, "current")


if __name__ == "__main__":
    unittest.main()
