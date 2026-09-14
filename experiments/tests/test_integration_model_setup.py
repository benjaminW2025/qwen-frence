"""Startup guards for synthetic-token integration runs; no GPU or Hub access."""

from __future__ import annotations

import importlib.util
import io
import os
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest import mock
import warnings
import torch  # Keep PyTorch loaded across the isolated sys.modules patches.

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments/integration"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "baseline"))
import model_setup


class ModelSetupTests(unittest.TestCase):
    def test_missing_hf_transfer_disables_stale_flag_before_hub_import(self):
        modules = {name: value for name, value in sys.modules.items()
                   if name != "huggingface_hub.constants"}
        with mock.patch.dict(os.environ, {"HF_HUB_ENABLE_HF_TRANSFER": "1"}):
            with mock.patch.object(model_setup.sys, "modules", modules):
                with mock.patch.object(model_setup.importlib.util, "find_spec", return_value=None):
                    with warnings.catch_warnings(record=True) as caught:
                        mode = model_setup.prepare_hub_transfer()
            self.assertEqual(os.environ["HF_HUB_ENABLE_HF_TRANSFER"], "0")
        self.assertEqual(mode, "standard_fallback_missing_hf_transfer")
        self.assertIn("standard Hugging Face download path", str(caught[0].message))

    def test_already_imported_hub_fails_with_actionable_error(self):
        with mock.patch.dict(os.environ, {"HF_HUB_ENABLE_HF_TRANSFER": "1"}):
            with mock.patch.object(model_setup.sys, "modules", {"huggingface_hub.constants": object()}):
                with mock.patch.object(model_setup.importlib.util, "find_spec", return_value=None):
                    with self.assertRaisesRegex(RuntimeError, "Restart with"):
                        model_setup.prepare_hub_transfer()

    def test_model_only_loader_does_not_construct_tokenizer(self):
        calls = []
        fake = ModuleType("weight_loader")

        class Loader:
            def __init__(self, cfg):
                calls.append(("cfg", cfg.use_custom_kernels))

            def load_pretrained(self, model_id, device, dtype):
                calls.append((model_id, device, str(dtype)))
                return object()

        fake.QwenWeightLoader = Loader
        with mock.patch.dict(sys.modules, {"weight_loader": fake}):
            engine, seconds, mode = model_setup.load_model_only(
                "synthetic/model", "cpu", "float16", hub_transfer="checked")
        self.assertTrue(engine.cfg.use_custom_kernels)
        self.assertEqual(engine.device, "cpu")
        self.assertEqual(mode, "checked")
        self.assertGreaterEqual(seconds, 0)
        self.assertEqual(calls[0], ("cfg", True))
        self.assertEqual(calls[1][:2], ("synthetic/model", "cpu"))

    def test_all_new_experiments_check_setup_before_weight_loading(self):
        for filename in ("profile_cpp_control.py", "benchmark_cpp_graph.py",
                         "benchmark_piecewise_prefill.py"):
            with self.subTest(filename=filename):
                path = HERE / filename
                spec = importlib.util.spec_from_file_location(filename.removesuffix(".py") + "_preflight", path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                with mock.patch.object(sys, "argv", [str(path), "--check-setup"]):
                    with mock.patch.object(module, "check_startup", return_value={"ok": True}):
                        with mock.patch.object(module, "load_model_only", side_effect=AssertionError("loaded weights")):
                            with mock.patch("sys.stdout", new_callable=io.StringIO) as output:
                                module.main()
                self.assertIn('"ok": true', output.getvalue())


if __name__ == "__main__":
    unittest.main()
