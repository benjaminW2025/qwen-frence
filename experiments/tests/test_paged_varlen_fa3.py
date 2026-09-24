"""CPU-side API contract for the optional packed paged FA3 call."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "custom_kernels/paged_varlen_fa3.py"
SPEC = importlib.util.spec_from_file_location("paged_varlen_fa3", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeTensor:
    def __init__(self, shape, dtype=torch.float16):
        self.shape = shape
        self.ndim = len(shape)
        self.dtype = dtype
        self.device = torch.device("cuda:0")
        self.is_cuda = True

    def numel(self):
        result = 1
        for dimension in self.shape:
            result *= dimension
        return result

    def contiguous(self):
        return self


class PackedVarlenFa3Tests(unittest.TestCase):
    def inputs(self):
        return (FakeTensor((1028, 12, 128)), FakeTensor((100, 16, 2, 128)),
                FakeTensor((100, 16, 2, 128)), FakeTensor((9,), torch.int32),
                FakeTensor((8, 18), torch.int32), FakeTensor((8,), torch.int32))

    def test_mixed_rows_use_one_causal_paged_varlen_call(self):
        recorded = []
        inputs = self.inputs()
        with patch.object(MODULE, "_load_fa3_varlen",
                          return_value=lambda **kwargs: recorded.append(kwargs) or "ok"):
            result = MODULE.fa3_paged_varlen_attention(
                *inputs, max_query_len=256)
        self.assertEqual(result, "ok")
        self.assertEqual(len(recorded), 1)
        self.assertTrue(recorded[0]["causal"])
        self.assertEqual(recorded[0]["max_seqlen_q"], 256)
        self.assertEqual(recorded[0]["max_seqlen_k"], 288)
        self.assertEqual(recorded[0]["fa_version"], 3)
        self.assertIs(recorded[0]["seqused_k"], inputs[5])
        self.assertIs(recorded[0]["block_table"], inputs[4])

    def test_rejects_non_int32_offsets(self):
        inputs = list(self.inputs())
        inputs[3] = FakeTensor((9,), torch.int64)
        with self.assertRaisesRegex(ValueError, "int32"):
            MODULE.fa3_paged_varlen_attention(*inputs, max_query_len=256)


if __name__ == "__main__":
    unittest.main()
