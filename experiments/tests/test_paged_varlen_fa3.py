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

    def stride(self, dimension=None):
        strides = tuple(self.numel() // self.shape[0] if i == 0 else
                        self.shape[-1] if i == 1 else 1 for i in range(self.ndim))
        return strides if dimension is None else strides[dimension]

    def is_contiguous(self):
        return True

    def data_ptr(self):
        return id(self)


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

    def test_direct_output_and_strided_query_are_forwarded(self):
        inputs = self.inputs()
        out = FakeTensor(inputs[0].shape)
        with patch.object(inputs[0], "contiguous", side_effect=AssertionError("unexpected copy")), \
             patch.object(MODULE, "_load_fa3_varlen") as loader:
            loader.return_value.return_value = (out, None)
            result = MODULE.fa3_paged_varlen_attention(
                *inputs, max_query_len=256, out=out, allow_strided_q=True)
        self.assertIs(result, out)
        self.assertIs(loader.return_value.call_args.kwargs["q"], inputs[0])
        self.assertIs(loader.return_value.call_args.kwargs["out"], out)

    def test_rejects_backend_ignoring_output_buffer(self):
        inputs = self.inputs()
        with patch.object(MODULE, "_load_fa3_varlen") as loader:
            loader.return_value.return_value = FakeTensor(inputs[0].shape)
            with self.assertRaisesRegex(RuntimeError, "did not honor"):
                MODULE.fa3_paged_varlen_attention(
                    *inputs, max_query_len=256, out=FakeTensor(inputs[0].shape))

    def test_rejects_invalid_output_shape(self):
        with self.assertRaisesRegex(ValueError, "output must"):
            MODULE.fa3_paged_varlen_attention(
                *self.inputs(), max_query_len=256, out=FakeTensor((1, 12, 128)))


if __name__ == "__main__":
    unittest.main()
