"""Suite reports missing GPU coverage explicitly and routes subcommand help."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from experiments.quant import attention, suite


class SuiteContracts(unittest.TestCase):
    def test_reference_run_does_not_claim_full_model_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "results"
            with mock.patch.object(attention, "main", return_value=0), mock.patch("torch.cuda.is_available", return_value=False):
                result = suite.main(["--reference-only", "--output-dir", str(output)])
            report = json.loads((output / "suite.json").read_text())
            self.assertEqual(result, 0)
            self.assertEqual(report["status"], "reference-only-complete")
            self.assertEqual([row["status"] for row in report["stages"]], ["complete", "not-run", "not-run"])

    def test_automatic_cpu_execution_is_partial(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "results"
            with mock.patch.object(attention, "main", return_value=0), mock.patch("torch.cuda.is_available", return_value=False):
                result = suite.main(["--output-dir", str(output)])
            self.assertEqual(result, 2)
            self.assertEqual(json.loads((output / "suite.json").read_text())["status"], "partial")

    def test_subcommand_help_is_forwarded(self):
        root = Path(__file__).resolve().parents[3]
        spec = importlib.util.spec_from_file_location("quant_entry_contract", root / "quant.py")
        entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entry)
        main = mock.Mock(return_value=0)
        with mock.patch.object(entry.importlib, "import_module", return_value=SimpleNamespace(main=main)):
            self.assertEqual(entry.main(["full-model", "--help"]), 0)
        main.assert_called_once_with(["--help"])


if __name__ == "__main__":
    unittest.main()
