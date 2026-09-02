"""CPU-only controls for the grouped-GQA decode sweep."""

from pathlib import Path
import unittest


PATH = Path(__file__).resolve().parents[1] / "decode" / "benchmark_grouped_gqa_decode.py"
SOURCE = PATH.read_text()


class GroupedGQADecodeSweepTests(unittest.TestCase):
    def test_default_axes_cover_the_profiled_decode_knee(self):
        self.assertIn('default="1,8,32,64"', SOURCE)
        self.assertIn('default="128,2048,8192,16384"', SOURCE)
        self.assertIn('default="1,2,3,6"', SOURCE)

    def test_candidate_kernel_exposes_gqa_head_grouping(self):
        kernel = (
            PATH.parents[2] / "custom_kernels" / "paged_decode_candidate.py"
        ).read_text()
        self.assertIn("HEADS_PER_PROGRAM", kernel)
        self.assertIn("values[None, :, :]", kernel)

    def test_dispatch_rule_uses_the_measured_batch_and_context_boundaries(self):
        kernel = (
            PATH.parents[2] / "custom_kernels" / "paged_decode_candidate.py"
        ).read_text()
        self.assertIn("DECODE_LOW_BATCH_THRESHOLD = 16", kernel)
        self.assertIn("DECODE_HIGH_BATCH_THRESHOLD = 64", kernel)
        self.assertIn("DECODE_LOW_BATCH_KV_TOKEN_THRESHOLD = 32768", kernel)
        self.assertIn("DECODE_HIGH_BATCH_KV_TOKEN_THRESHOLD = 16384", kernel)
        self.assertIn("8 if batch_size < DECODE_LOW_BATCH_THRESHOLD else 4", kernel)
        self.assertIn('"heads_per_program": 1', kernel)

    def test_sweep_has_independent_ragged_correctness_gate_and_summary(self):
        self.assertIn("decode_attention_reference", SOURCE)
        self.assertIn("[1, 15, 16, 17, 31, 33, 70]", SOURCE)
        self.assertIn("shape_summaries", SOURCE)
        self.assertIn("estimated_read_amplification", SOURCE)


if __name__ == "__main__":
    unittest.main()
