"""CPU-only contracts for the paid-GPU setup confirmation runner."""

from pathlib import Path
import unittest

import _bootstrap  # noqa: F401

from run_setup_checks import smoke_command, version_contract


class SetupChecksTests(unittest.TestCase):
    def test_vllm_contract_rejects_unbounded_transformers_upgrade(self):
        results = version_contract("vllm", {
            "torch": "2.8.0",
            "transformers": "5.0.0",
            "triton": "3.4.0",
            "vllm": "0.10.2",
        })
        contract = next(r for r in results if r.name == "vllm-version-contract")
        self.assertFalse(contract.passed)
        self.assertIn("transformers=5.0.0", contract.detail)

    def test_vllm_contract_accepts_pinned_pair(self):
        results = version_contract("vllm", {
            "torch": "2.8.0",
            "transformers": "4.55.2",
            "triton": "3.4.0",
            "vllm": "0.10.2",
        })
        self.assertTrue(all(result.passed for result in results))

    def test_smoke_command_is_intentionally_tiny_and_strict(self):
        command = smoke_command("vllm", "model", Path("results"))
        self.assertEqual(command[command.index("--backends") + 1], "vllm")
        self.assertEqual(command[command.index("--num-requests") + 1], "1")
        self.assertEqual(command[command.index("--prompt-lengths") + 1], "16")
        self.assertEqual(command[command.index("--output-lengths") + 1], "2")
        self.assertIn("--strict-backends", command)

    def test_local_smoke_covers_control_and_candidate(self):
        command = smoke_command("local", "model", Path("results"))
        self.assertEqual(
            command[command.index("--backends") + 1],
            "custom-kernels,regime-dispatched",
        )


if __name__ == "__main__":
    unittest.main()
