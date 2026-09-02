import argparse
import unittest

import _bootstrap  # noqa: F401

from benchmark_packed_attention_tiles import (
    parse_tile_configurations,
    tile_configurations,
    validate_tile_sizes,
)


class PackedAttentionTileSweepTests(unittest.TestCase):
    def test_cartesian_tile_plan(self):
        self.assertEqual(
            tile_configurations([32, 64], [32, 128]),
            [(32, 32), (32, 128), (64, 32), (64, 128)],
        )

    def test_tile_sizes_must_be_supported_powers_of_two(self):
        validate_tile_sizes([16, 32, 64, 128], "blocks")
        for values in ([], [8], [48]):
            with self.assertRaises(ValueError):
                validate_tile_sizes(values, "blocks")

    def test_parse_explicit_tile_plan_preserves_order(self):
        self.assertEqual(
            parse_tile_configurations("32x32,64X32,64x64,128x64"),
            [(32, 32), (64, 32), (64, 64), (128, 64)],
        )

    def test_explicit_tile_plan_rejects_invalid_or_duplicate_pairs(self):
        for value in ("", "64", "64xnope", "48x32", "64x32,64x32"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                parse_tile_configurations(value)


if __name__ == "__main__":
    unittest.main()
