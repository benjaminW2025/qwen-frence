import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import torch

PATH = Path(__file__).resolve().parents[1] / "model" / "benchmark_packed_projection_decode.py"
SPEC = importlib.util.spec_from_file_location("packed_projection_decode", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Tests(unittest.TestCase):
    def test_defaults_cover_all_layouts(self):
        args = MODULE.parser().parse_args([])
        self.assertEqual(MODULE.validate(args), ("warm", "evict"))
        self.assertEqual([x[0] for x in MODULE.LAYOUTS], ["separate", "qkv_packed", "qkv_gate_up_packed"])
    def test_invalid_cache_mode_rejected(self):
        args = MODULE.parser().parse_args(["--cache-modes", "cold"])
        with self.assertRaises(ValueError): MODULE.validate(args)
    def test_int_list_rejects_duplicates(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            MODULE.int_list("1,1")

    def test_report_serializes_output_path_and_resumes_complete_case(self):
        with tempfile.TemporaryDirectory() as directory:
            args = MODULE.parser().parse_args(["--output-dir", directory, "--trials", "1", "--samples", "2"])
            configuration = MODULE.design(args, (1,), (16,), ("warm",))
            rows = [
                {"batch": 1, "context": 16, "trial": 0, "cache": "warm", "layout": name,
                 "samples_ms": [1.0, 1.2], "median_ms": 1.1, "speedup_vs_separate": 1.0}
                for name, _, _ in MODULE.LAYOUTS
            ]
            checkpoint = MODULE.checkpoint_path(args.output_dir, 1, 16)
            MODULE.atomic_json(checkpoint, {
                "status": "complete", "fingerprint": "same", "batch": 1, "context": 16,
                "records": rows,
            })
            loaded = MODULE.validate_checkpoint(
                json.loads(checkpoint.read_text()), fingerprint="same", batch=1, context=16,
                modes=("warm",), trials=1, samples=2,
            )
            report = args.output_dir / "decode-ablation-results.json"
            MODULE.atomic_json(report, MODULE.report_payload(configuration, "same", {}, loaded))
            self.assertEqual(json.loads(report.read_text())["configuration"]["output_dir"], directory)
            self.assertEqual(len(json.loads(report.read_text())["records"]), 3)

    def test_staged_page_and_slot_geometry(self):
        from naive_forward import Qwen2Config

        cfg = Qwen2Config(vocab=32, d_model=16, d_ff=32, n_layers=1, n_heads=2,
                          n_kv_heads=1, d_head=8, max_seq_len=64)
        cache, (ids, positions, lengths, table, slots) = MODULE.stage_case(
            torch, cfg, 2, 16, torch.float32, 7, "cpu",
        )
        self.assertEqual(tuple(ids.shape), (2, 1))
        self.assertEqual(positions.tolist(), [16, 16])
        self.assertEqual(lengths.tolist(), [17, 17])
        self.assertEqual(table.tolist(), [[0, 1], [2, 3]])
        self.assertEqual(slots.tolist(), [16, 48])
        self.assertEqual(cache.cur_lens, [17, 17])


if __name__ == "__main__":
    unittest.main()
