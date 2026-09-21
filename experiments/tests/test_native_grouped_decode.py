"""CPU-only contract tests for the native grouped-decode experiment."""

from pathlib import Path
import importlib.util
import sys
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
KERNEL_DIR = ROOT / "custom_kernels"
DECODE_DIR = ROOT / "experiments" / "decode"
for path in (KERNEL_DIR, DECODE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


class FakeTensor:
    def __init__(self, shape, *, dtype="fp16", device="cuda"):
        self.shape = tuple(shape)
        self.ndim = len(shape)
        self.dtype = dtype
        self.device = device


class NativeGroupedContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = KERNEL_DIR / "paged_decode_native_grouped.py"
        spec = importlib.util.spec_from_file_location("native_grouped_contract", path)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def inputs(self, group=6, head_dim=128, page_size=16):
        return (
            FakeTensor((8, 2 * group, head_dim)),
            FakeTensor((2048, page_size, 2, head_dim)),
            FakeTensor((2048, page_size, 2, head_dim)),
            FakeTensor((8, 256), dtype="int32"),
            FakeTensor((8,), dtype="int32"),
        )

    def test_rejects_unsupported_group(self):
        with self.assertRaisesRegex(ValueError, "six query heads"):
            self.module.native_grouped_splitk_decode_attention(*self.inputs(group=4), split_k=8)

    def test_rejects_unsupported_layout(self):
        with self.assertRaisesRegex(ValueError, "head_dim=128"):
            self.module.native_grouped_splitk_decode_attention(
                *self.inputs(head_dim=64), split_k=8
            )

    def test_requires_real_split(self):
        with self.assertRaisesRegex(ValueError, "split_k >= 2"):
            self.module.native_grouped_splitk_decode_attention(*self.inputs(), split_k=1)

    def test_missing_extension_has_one_build_command(self):
        with mock.patch.object(self.module.importlib, "import_module", side_effect=ImportError):
            with self.assertRaisesRegex(RuntimeError, "setup.py build_ext --inplace"):
                self.module._extension()


if __name__ == "__main__":
    unittest.main()
