"""CPU-only controls for the fused SwiGLU experiment."""

from pathlib import Path
import unittest


EXPERIMENTS = Path(__file__).resolve().parents[1]
SCRIPT = EXPERIMENTS / "mlp" / "benchmark_swiglu_fusion.py"
SOURCE = SCRIPT.read_text()
KERNEL = EXPERIMENTS.parent / "custom_kernels" / "swiglu.py"
KERNEL_SOURCE = KERNEL.read_text()


class SwiGLUFusionTests(unittest.TestCase):
    def test_default_rows_include_regime_atlas_shapes(self):
        self.assertIn('default="1,8,32,64,520,2048,4128,8200,16384"', SOURCE)
        self.assertIn('default=8960', SOURCE)

    def test_experiment_preserves_raw_samples_and_traffic_model(self):
        self.assertIn('"baseline_raw_ms"', SOURCE)
        self.assertIn('"fused_raw_ms"', SOURCE)
        self.assertIn('"traffic_model"', SOURCE)
        self.assertIn("5 * elements * element_size", SOURCE)
        self.assertIn("3 * elements * element_size", SOURCE)

    def test_kernel_fuses_activation_and_multiply(self):
        self.assertIn("gate * tl.sigmoid(gate)", KERNEL_SOURCE)
        self.assertIn("* up", KERNEL_SOURCE)
        self.assertIn("torch.empty_like(gate)", KERNEL_SOURCE)

    def test_correctness_preflight_is_independent_of_benchmark_shapes(self):
        self.assertIn("torch.testing.assert_close", SOURCE)
        self.assertIn("17, 257", SOURCE)
        self.assertIn('"correctness_preflight"', SOURCE)


if __name__ == "__main__":
    unittest.main()
