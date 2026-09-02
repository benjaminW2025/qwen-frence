"""CPU-only tests for phase-sweep planning and aggregation."""

import argparse
import unittest

import _bootstrap  # noqa: F401

from run_phase_sweep import (
    implementation_flags,
    parse_implementations,
    parse_phases,
    planned_cases,
    summarize_times,
    validate_sweep,
)


class PhaseSweepTests(unittest.TestCase):
    def test_parsers_and_implementation_flags(self):
        self.assertEqual(parse_phases("prefill,decode"), ["prefill", "decode"])
        self.assertEqual(parse_implementations("custom-kernels"), ["custom-kernels"])
        self.assertEqual(implementation_flags("continuous-batching"), (False, False))
        self.assertEqual(implementation_flags("bucketed-cuda-graphs"), (True, False))
        self.assertEqual(implementation_flags("custom-kernels"), (False, True))
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_phases("unknown")

    def test_case_planning(self):
        cases = planned_cases(
            ["prefill", "decode"],
            ["custom-kernels"],
            [1, 8],
            [128, 512],
            [1024, 32736],
        )
        self.assertEqual(len(cases), 8)
        self.assertEqual(cases[0], ("prefill", "custom-kernels", 1, 128))
        self.assertEqual(cases[-1], ("decode", "custom-kernels", 8, 32736))

    def test_sweep_validation_includes_decode_headroom(self):
        validate_sweep(
            batch_sizes=[1, 8],
            prefill_lengths=[128, 4096],
            context_lengths=[128, 32736],
            decode_steps=32,
            max_seq_len=32768,
        )
        with self.assertRaisesRegex(ValueError, "exceed"):
            validate_sweep(
                batch_sizes=[1],
                prefill_lengths=[128],
                context_lengths=[32737],
                decode_steps=32,
                max_seq_len=32768,
            )

    def test_timing_summary(self):
        summary = summarize_times([10.0, 20.0, 30.0], tokens=200, batch_size=4)
        self.assertEqual(summary["median_ms"], 20.0)
        self.assertEqual(summary["tokens_per_second"], 10_000.0)
        self.assertEqual(summary["per_sequence_tokens_per_second"], 2_500.0)
        with self.assertRaises(ValueError):
            summarize_times([], tokens=1, batch_size=1)


if __name__ == "__main__":
    unittest.main()
