"""CPU-only controls for the resumed paged-prefill tile sweep."""

import argparse
from pathlib import Path
import unittest

import _bootstrap  # noqa: F401

from benchmark_paged_prefill_tiles import (
    DEFAULT_CONFIGS,
    PRODUCTION_CONFIG,
    build_parser,
    parse_kernel_configs,
    summarize_configs,
    validate_args,
)


class PagedPrefillTileSweepTests(unittest.TestCase):
    def test_default_plan_contains_production_control(self):
        args = build_parser().parse_args([])
        self.assertIn(PRODUCTION_CONFIG, args.configs)
        self.assertEqual(args.configs, parse_kernel_configs(DEFAULT_CONFIGS))

    def test_parser_rejects_invalid_or_duplicate_configs(self):
        for value in ("", "64x32", "48x32x4x2", "64x32x3x2", "64x32x4x0",
                      "64x32x4x2,64x32x4x2"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                parse_kernel_configs(value)

    def test_context_limit_is_enforced(self):
        args = build_parser().parse_args([])
        with self.assertRaisesRegex(ValueError, "context limit"):
            validate_args(args, [1], [2048], [32768])

    def test_summary_quantifies_static_and_oracle_gap(self):
        rows = []
        configs = (PRODUCTION_CONFIG, (32, 32, 4, 2))
        timings = {
            (1, 64, 0): (10.0, 5.0),
            (1, 64, 128): (10.0, 20.0),
        }
        for shape, values in timings.items():
            for config, latency in zip(configs, values):
                rows.append({
                    "batch_size": shape[0],
                    "query_length": shape[1],
                    "prefix_length": shape[2],
                    "block_m": config[0],
                    "block_n": config[1],
                    "num_warps": config[2],
                    "num_stages": config[3],
                    "triton_median_ms": latency,
                    "speedup_vs_production": 10.0 / latency,
                    "status": "ok",
                })
        summary = summarize_configs(rows)
        self.assertEqual(summary["best_static_config"], list(PRODUCTION_CONFIG))
        self.assertAlmostEqual(summary["oracle_speedup_vs_best_static"], 20.0 / 15.0)

    def test_runtime_rule_records_query_and_fresh_prefix_boundaries(self):
        source = (
            Path(__file__).resolve().parents[2]
            / "custom_kernels"
            / "packed_paged_prefill_attention.py"
        ).read_text()
        self.assertIn("ADAPTIVE_QUERY_TOKEN_THRESHOLD = 1280", source)
        self.assertIn("FRESH_QUERY_TOKEN_THRESHOLD = 2048", source)
        self.assertIn("max_prefix_length == 0", source)


if __name__ == "__main__":
    unittest.main()
