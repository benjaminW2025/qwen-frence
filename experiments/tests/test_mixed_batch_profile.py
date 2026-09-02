"""CPU-only controls for the mixed decode/prefill profiler."""

import argparse
import importlib.util
from pathlib import Path
import unittest

import _bootstrap  # noqa: F401


PATH = Path(__file__).resolve().parents[1] / "scheduler" / "profile_mixed_batch.py"
SPEC = importlib.util.spec_from_file_location("profile_mixed_batch", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
SCHEDULER_SOURCE = (
    Path(__file__).resolve().parents[2] / "engine" / "scheduler" / "scheduler.py"
).read_text()


class MixedBatchProfilerTests(unittest.TestCase):
    def test_defaults_describe_a_real_mixed_iteration(self):
        args = MODULE.build_parser().parse_args([])
        MODULE.validate_args(args)
        self.assertGreater(args.decode_requests, 0)
        self.assertGreater(args.prefill_requests, 0)
        self.assertGreater(args.prefill_chunk_size, 1)

    def test_context_limit_and_profiler_staging_contracts(self):
        base = vars(MODULE.build_parser().parse_args([]))
        with self.assertRaisesRegex(ValueError, "context limit"):
            MODULE.validate_args(argparse.Namespace(**(base | {
                "prefill_prefix_length": 32768,
                "prefill_chunk_size": 1,
            })))
        with self.assertRaisesRegex(ValueError, "one target iteration"):
            MODULE.validate_args(argparse.Namespace(**(base | {
                "profile_repetitions": 2,
            })))

    def test_cuda_profiler_range_requires_nvtx_mode(self):
        args = MODULE.build_parser().parse_args(["--cuda-profiler-range"])
        with self.assertRaisesRegex(ValueError, "requires --nvtx-only"):
            MODULE.validate_args(args)

    def test_decode_sampling_is_one_batched_reduction_and_host_transfer(self):
        method = SCHEDULER_SOURCE.split(
            "    def _commit_decode_logits", 1
        )[1].split("\n    def ", 1)[0]
        self.assertIn("logits.argmax(dim=-1).tolist()", method)
        self.assertNotIn("logits[row].argmax()", method)
        self.assertIn('"decode/sample_batch"', method)


if __name__ == "__main__":
    unittest.main()
