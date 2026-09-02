"""CPU-only contracts for the matched mixed execution-surface profiler."""

import importlib.util
from pathlib import Path
import unittest

import _bootstrap  # noqa: F401


PATH = Path(__file__).resolve().parents[1] / "scheduler" / "profile_mixed_surface.py"
SPEC = importlib.util.spec_from_file_location("profile_mixed_surface", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MixedSurfaceProfilerTests(unittest.TestCase):
    def test_default_surface_contains_matched_axes(self):
        args = MODULE.build_parser().parse_args([])
        decodes = MODULE.parse_int_list(args.decode_requests)
        contexts = MODULE.parse_int_list(args.decode_context_lengths)
        prefills = MODULE.parse_int_list(args.prefill_tokens)
        prefixes = MODULE.parse_nonnegative_int_list(args.prefill_prefix_lengths)
        MODULE.validate_surface(
            decode_requests=decodes,
            decode_context_lengths=contexts,
            prefill_tokens=prefills,
            prefill_prefix_lengths=prefixes,
            prefill_requests=args.prefill_requests,
            block_size=args.block_size,
            warmups=args.warmups,
            repetitions=args.repetitions,
        )
        cases = MODULE.planned_mixed_cases(decodes, contexts, prefills, prefixes)
        self.assertEqual(len(cases), len(decodes) * len(contexts) * len(prefills) * len(prefixes))
        self.assertIn((32, 2048, 4096, 4096), cases)

    def test_surface_rejects_indivisible_or_oversized_chunks(self):
        common = dict(
            decode_requests=[1],
            decode_context_lengths=[2048],
            prefill_prefix_lengths=[0],
            block_size=16,
            warmups=1,
            repetitions=3,
        )
        with self.assertRaisesRegex(ValueError, "not divisible"):
            MODULE.validate_surface(
                **common, prefill_tokens=[513], prefill_requests=2
            )
        with self.assertRaisesRegex(ValueError, "max_seq_len"):
            MODULE.validate_surface(
                **(common | {"prefill_prefix_lengths": [32768]}),
                prefill_tokens=[2],
                prefill_requests=2,
            )

    def test_tradeoff_metrics_use_matched_controls(self):
        metrics = MODULE.derive_tradeoff(
            mixed_ms=46.0, decode_ms=12.0, prefill_ms=39.0
        )
        self.assertEqual(metrics["incremental_prefill_ms"], 34.0)
        self.assertEqual(metrics["incremental_decode_ms"], 7.0)
        self.assertEqual(metrics["separate_sum_ms"], 51.0)
        self.assertEqual(metrics["packing_benefit_ms"], 5.0)
        self.assertAlmostEqual(metrics["packing_speedup"], 51.0 / 46.0)
        self.assertAlmostEqual(metrics["decode_stretch"], 46.0 / 12.0)


if __name__ == "__main__":
    unittest.main()
