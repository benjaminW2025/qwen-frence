"""CPU-only tests for cross-backend result validation and diagnostics."""

import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import subprocess

import _bootstrap  # noqa: F401

from benchmark_core import BackendRun, RequestSpec, RequestTrace, Workload
from run_benchmarks import (
    attach_baseline_diagnostics,
    attach_performance_comparisons,
    load_comparison_results,
    repository_metadata,
    validate_run,
    validate_comparison_contract,
    workload_fingerprint,
)


def request_dict(request_id, output_ids):
    return {"request_id": request_id, "output_ids": output_ids}


class BenchmarkRunnerTests(unittest.TestCase):
    def test_repository_provenance_records_commit_and_dirty_state(self):
        completed = [
            subprocess.CompletedProcess([], 0, stdout="abc123\n", stderr=""),
            subprocess.CompletedProcess([], 0, stdout=" M README.md\n", stderr=""),
        ]
        with patch("run_benchmarks.subprocess.run", side_effect=completed):
            self.assertEqual(
                repository_metadata(), {"commit": "abc123", "dirty": True}
            )

    def test_cross_backend_prefix_diagnostics(self):
        results = {
            "pytorch-baseline": {
                "runs": [{"requests": [
                    request_dict("a", [1, 2]),
                    request_dict("b", [3, 4]),
                ]}]
            },
            "paged-kv": {
                "runs": [{"requests": [
                    request_dict("a", [1, 2]),
                    request_dict("b", [3, 9]),
                ]}]
            },
        }
        attach_baseline_diagnostics(results)
        self.assertEqual(
            results["paged-kv"]["correctness_vs_pytorch_baseline"],
            {
                "exact_requests": 1,
                "total_requests": 2,
                "exact_request_fraction": 0.5,
                "min_matched_prefix_tokens": 1,
                "mean_matched_prefix_tokens": 1.5,
            },
        )

    def test_run_validation_checks_saved_tokens(self):
        workload = Workload("one", [RequestSpec("a", [1], 2)])
        valid = RequestTrace(
            request_id="a",
            prompt_tokens=1,
            requested_output_tokens=2,
            arrival_s=0,
            submitted_s=0,
            token_times_s=[0.1, 0.2],
            finished_s=0.2,
            output_tokens=2,
            output_ids=[4, 5],
        )
        validate_run(BackendRun("fake", 0.2, [valid]), workload)

        valid.output_ids = [4]
        with self.assertRaisesRegex(AssertionError, "saved token IDs"):
            validate_run(BackendRun("fake", 0.2, [valid]), workload)

    def test_relative_throughput(self):
        results = {
            "custom-kernels": {"summary": {"output_throughput_tok_s": 100.0}},
            "vllm": {"summary": {"output_throughput_tok_s": 250.0}},
        }
        attach_performance_comparisons(results)
        self.assertEqual(
            results["vllm"]["relative_output_throughput"]["vs_custom_kernels"],
            2.5,
        )

    def test_comparison_import_requires_exact_workload_and_config(self):
        workload = Workload("one", [RequestSpec("a", [1, 2], 3)])
        configuration = {
            "model": "model",
            "dtype": "float16",
            "block_size": 16,
            "max_running": 8,
        }
        payload = {
            "schema_version": 1,
            "created_at": "now",
            "system": {"gpu": "test"},
            "workload": workload.to_dict(),
            "configuration": configuration,
            "backends": {"custom-kernels": {"summary": {}, "runs": []}},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps(payload))
            results, sources = load_comparison_results(
                [path], workload, configuration
            )
            self.assertIn("custom-kernels", results)
            self.assertEqual(
                sources[0]["workload_sha256"], workload_fingerprint(workload)
            )

            mismatched = dict(configuration, max_running=4)
            with self.assertRaisesRegex(ValueError, "max_running mismatch"):
                load_comparison_results([path], workload, mismatched)

    def test_unmatched_vllm_result_is_rejected(self):
        results = {
            "vllm": {
                "runs": [{"metadata": {
                    "max_num_seqs": 256,
                    "max_model_len": 131072,
                    "kv_cache_mode": "vllm-native",
                }}]
            }
        }
        with self.assertRaisesRegex(ValueError, "max_num_seqs"):
            validate_comparison_contract(
                results,
                max_running=8,
                max_model_len=1152,
                vllm_kv_cache_mode="matched",
            )


if __name__ == "__main__":
    unittest.main()
