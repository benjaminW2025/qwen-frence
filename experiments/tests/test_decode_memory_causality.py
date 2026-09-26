"""CPU-visible contracts for cache-causality and FA3 oracle experiments."""

from pathlib import Path
import importlib.util
import sys
import unittest
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FA3 = load("paged_decode_fa3_test", ROOT / "custom_kernels" / "paged_decode_fa3.py")
BENCHMARK = load(
    "benchmark_decode_memory_causality_test",
    ROOT / "experiments" / "decode" / "benchmark_decode_memory_causality.py",
)


class FA3AdapterTests(unittest.TestCase):
    def tensors(self):
        return (
            torch.zeros(2, 12, 128, dtype=torch.float16),
            torch.zeros(2, 16, 2, 128, dtype=torch.float16),
            torch.zeros(2, 16, 2, 128, dtype=torch.float16),
            torch.zeros(2, 1, dtype=torch.int32),
            torch.full((2,), 16, dtype=torch.int32),
        )

    def test_exact_layout_is_forwarded_without_cache_copy(self):
        seen = {}

        def fake(**kwargs):
            seen.update(kwargs)
            return kwargs["q"]

        tensors = self.tensors()
        with mock.patch.object(FA3, "_load_fa3", return_value=fake):
            output = FA3.fa3_paged_decode_attention(*tensors, num_splits=22)
        self.assertEqual(output.shape, tensors[0].shape)
        self.assertIs(seen["q"], tensors[0])
        self.assertIs(seen["k"], tensors[1])
        self.assertIs(seen["v"], tensors[2])
        self.assertIs(seen["seqused_k"], tensors[4])
        self.assertEqual(seen["cu_seqlens_q"].tolist(), [0, 1, 2])
        self.assertEqual(seen["max_seqlen_q"], 1)
        self.assertEqual(seen["max_seqlen_k"], 16)
        self.assertIs(seen["block_table"], tensors[3])
        self.assertEqual(seen["fa_version"], 3)
        self.assertEqual(seen["num_splits"], 22)

    def test_rejects_metadata_dtype_before_importing_vllm(self):
        tensors = list(self.tensors())
        tensors[3] = tensors[3].long()
        with self.assertRaisesRegex(ValueError, "must be int32"):
            FA3.fa3_paged_decode_attention(*tensors)


class CausalityCLIContracts(unittest.TestCase):
    def test_defaults_target_observed_failure(self):
        args = BENCHMARK.build_parser().parse_args([])
        self.assertEqual((args.batch_size, args.context_length, args.split_k),
                         (64, 4096, 22))
        self.assertFalse(args.skip_fa3)


if __name__ == "__main__":
    unittest.main()
