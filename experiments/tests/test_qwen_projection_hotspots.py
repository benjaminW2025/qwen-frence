"""CPU-only argument and protocol tests for the Qwen projection profiler."""

import argparse
import importlib.util
from pathlib import Path
import unittest

import _bootstrap  # noqa: F401


PATH = Path(__file__).resolve().parents[1] / "model" / "profile_qwen_projection_hotspots.py"
SPEC = importlib.util.spec_from_file_location("profile_qwen_projection_hotspots", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class QwenProjectionProfilerTests(unittest.TestCase):
    def test_decode_batch_defaults_and_parser(self):
        args = MODULE.build_parser().parse_args(["bench"])
        MODULE.validate_args(args)
        self.assertEqual(args.batches, (1, 2, 4, 8, 16, 32, 64, 96, 128, 256))

    def test_trace_batch_must_be_measured(self):
        args = MODULE.build_parser().parse_args(["trace", "--batches", "1,8", "--trace-batch", "32"])
        with self.assertRaisesRegex(ValueError, "included"):
            MODULE.validate_args(args)

    def test_cuda_capture_is_trace_only(self):
        args = MODULE.build_parser().parse_args(["bench", "--cuda-profiler-range"])
        with self.assertRaisesRegex(ValueError, "trace command"):
            MODULE.validate_args(args)

    def test_batch_parser_rejects_duplicates(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            MODULE.batch_sizes("1,1")


if __name__ == "__main__":
    unittest.main()
