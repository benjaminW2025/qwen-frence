import argparse
import unittest

import _bootstrap  # noqa: F401

from profile_prefill import build_parser, validate_args


class PrefillProfilerControlPlaneTests(unittest.TestCase):
    def test_defaults_target_one_real_2k_prefill(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.implementation, "custom-kernels")
        self.assertEqual(args.prompt_length, 2048)
        self.assertEqual(args.requests, 1)
        self.assertEqual(args.attention_backend, "triton")
        self.assertFalse(args.nvtx_only)
        validate_args(args)

    def test_invalid_counts_are_rejected(self):
        args = build_parser().parse_args(["--profile-repetitions", "0"])
        with self.assertRaisesRegex(ValueError, "profile-repetitions"):
            validate_args(args)

    def test_model_context_limit_is_enforced(self):
        args = build_parser().parse_args(["--prompt-length", "32769"])
        with self.assertRaisesRegex(ValueError, "32768"):
            validate_args(args)


if __name__ == "__main__":
    unittest.main()
