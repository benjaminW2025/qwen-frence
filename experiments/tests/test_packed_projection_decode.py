import argparse
import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1] / "model" / "benchmark_packed_projection_decode.py"
SPEC = importlib.util.spec_from_file_location("packed_projection_decode", PATH)
MODULE = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(MODULE)

class Tests(unittest.TestCase):
    def test_defaults_cover_all_layouts(self):
        args = MODULE.parser().parse_args([])
        self.assertEqual(MODULE.validate(args), ("warm", "evict"))
        self.assertEqual([x[0] for x in MODULE.LAYOUTS], ["separate", "qkv_packed", "qkv_gate_up_packed"])
    def test_invalid_cache_mode_rejected(self):
        args = MODULE.parser().parse_args(["--cache-modes", "cold"])
        with self.assertRaises(ValueError): MODULE.validate(args)
    def test_int_list_rejects_duplicates(self):
        with self.assertRaises(argparse.ArgumentTypeError): MODULE.int_list("1,1")

if __name__ == "__main__": unittest.main()
