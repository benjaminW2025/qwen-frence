import argparse
import unittest

import _bootstrap  # noqa: F401

from benchmark_ragged_prefill import case_lengths, parse_patterns


class RaggedPrefillBenchmarkTests(unittest.TestCase):
    def test_uniform_lengths(self):
        self.assertEqual(case_lengths("uniform", 4, 128), [128, 128, 128, 128])

    def test_ramp_lengths(self):
        self.assertEqual(case_lengths("ramp", 4, 128), [32, 64, 96, 128])
        self.assertEqual(case_lengths("ramp", 1, 7), [7])

    def test_pattern_parser(self):
        self.assertEqual(parse_patterns("uniform,ramp"), ["uniform", "ramp"])
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_patterns("unknown")


if __name__ == "__main__":
    unittest.main()
