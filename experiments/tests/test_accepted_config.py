"""The shipping preset must not acquire unqualified candidates by accident."""
import unittest

from engine.accepted_config import accepted_variant_options, retained_fusion_options
from experiments.integration.benchmark_current_8_vs_vllm import (
    adapter_options as burst_options,
)
from experiments.integration.benchmark_current_mixed_8_vs_vllm import (
    adapter_options as mixed_options,
)


class AcceptedConfigTests(unittest.TestCase):
    def test_retained_fusions_reach_both_runtime_paths(self):
        for factory in (burst_options, mixed_options):
            for batch in (8, 64):
                case = {"max_running": batch, "lengths": [256] * batch,
                        "outputs": [128] * batch}
                options = factory(case, [2048], "fa3", accepted_variant_options())
                self.assertEqual(options["decode_attention_policy"], "fa3")
                self.assertEqual(options["prefill_attention_policy"], "fa3_varlen")
                self.assertEqual(options["decode_buckets"], [batch])
                for key, value in retained_fusion_options().items():
                    self.assertEqual(options[key], value)
                for key, value in accepted_variant_options().items():
                    self.assertEqual(options[key], value)

    def test_options_cannot_mutate_the_preset(self):
        changed = accepted_variant_options()
        changed["enable_decode_fused_gemm_epilogues"] = True
        self.assertFalse(accepted_variant_options()["enable_decode_fused_gemm_epilogues"])


if __name__ == "__main__":
    unittest.main()
