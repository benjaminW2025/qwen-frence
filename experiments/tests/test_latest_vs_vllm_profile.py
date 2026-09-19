"""CPU-only contract tests for the matched local/vLLM decode profiler."""

from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/integration"))
PATH = ROOT / "experiments/integration/profile_latest_vs_vllm.py"
SPEC = importlib.util.spec_from_file_location("profile_latest_vs_vllm", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeEngine:
    def __init__(self):
        self.requests = {}

    def add_request(self, request_id, prompt, params):
        self.requests[request_id] = {"progress": 0, "maximum": params.max_tokens,
                                     "output_kind": params.output_kind}

    def has_unfinished_requests(self):
        return any(row["progress"] < row["maximum"] for row in self.requests.values())

    def step(self):
        outputs = []
        for request_id, row in self.requests.items():
            if row["progress"] >= row["maximum"]:
                continue
            row["progress"] += 1
            finished = row["progress"] == row["maximum"]
            if row["output_kind"] == "final_only" and not finished:
                continue
            outputs.append(SimpleNamespace(
                request_id=request_id,
                outputs=[SimpleNamespace(token_ids=list(range(row["progress"])))],
                finished=finished,
            ))
        return outputs


class FakeLLM:
    def __init__(self):
        self.llm_engine = FakeEngine()
        self.profile_calls = []

    def start_profile(self):
        self.profile_calls.append("start")

    def stop_profile(self):
        self.profile_calls.append("stop")


class MatchedProfileTests(unittest.TestCase):
    def fake_vllm_modules(self):
        vllm = ModuleType("vllm")
        sampling = ModuleType("vllm.sampling_params")

        class SamplingParams:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class RequestOutputKind:
            CUMULATIVE = "cumulative"
            FINAL_ONLY = "final_only"

        vllm.SamplingParams = SamplingParams
        sampling.RequestOutputKind = RequestOutputKind
        return {"vllm": vllm, "vllm.sampling_params": sampling}

    def test_vllm_target_is_exact_full_cohort_occurrence(self):
        llm = FakeLLM()
        requests = [{"id": 0, "prompt": [1, 2], "output": 6},
                    {"id": 1, "prompt": [3, 4], "output": 6}]
        with patch.dict(sys.modules, self.fake_vllm_modules()):
            result = MODULE.discover_vllm_target(
                llm, requests, occurrence=2, run_id="test")
        self.assertEqual(set(result["target_progress_before"].values()), {3})
        self.assertEqual(set(result["target_progress_after"].values()), {4})
        self.assertEqual(set(result["target_context_lengths"].values()), {5})
        self.assertEqual(result["pure_full_decode_steps"], 5)
        self.assertEqual(result["target_engine_step_index"], 3)

    def test_timed_vllm_replay_profiles_only_discovered_step(self):
        llm = FakeLLM()
        requests = [{"id": 0, "prompt": [1, 2], "output": 6},
                    {"id": 1, "prompt": [3, 4], "output": 6}]
        with patch.dict(sys.modules, self.fake_vllm_modules()):
            result = MODULE.measure_vllm_target(
                llm, requests, target_step_index=3, run_id="timed", profile=True)
        self.assertGreater(result["target_wall_ms"], 0)
        self.assertEqual(result["total_engine_steps"], 6)
        self.assertEqual(llm.profile_calls, ["start", "stop"])

    def test_chrome_trace_parser_counts_only_cuda_leaf_activities(self):
        payload = {"traceEvents": [
            {"ph": "X", "cat": "kernel", "name": "paged_attention_kernel", "dur": 120},
            {"ph": "X", "cat": "kernel", "name": "cutlass_gemm", "dur": 80},
            {"ph": "X", "cat": "cpu_op", "name": "parent", "dur": 999},
            {"ph": "i", "cat": "kernel", "name": "marker", "dur": 50},
        ]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.pt.trace.json.gz"
            with gzip.open(path, "wt") as stream:
                json.dump(payload, stream)
            summary = MODULE.summarize_chrome_trace(path)
        self.assertEqual(summary["summed_cuda_activity_us"], 200)
        self.assertEqual(summary["activity_count"], 2)
        categories = {row["category"]: row for row in summary["categories"]}
        self.assertEqual(categories["attention"]["total_us"], 120)
        self.assertEqual(categories["gemm"]["total_us"], 80)

    def test_category_comparison_uses_union_and_reports_ratios(self):
        local = {"categories": [
            {"category": "attention", "total_us": 120, "percent_of_cuda_activity": 60},
            {"category": "gemm", "total_us": 80, "percent_of_cuda_activity": 40},
        ]}
        vllm = {"categories": [
            {"category": "attention", "total_us": 60, "percent_of_cuda_activity": 75},
            {"category": "rope", "total_us": 20, "percent_of_cuda_activity": 25},
        ]}
        rows = {row["category"]: row for row in MODULE.compare_categories(local, vllm)}
        self.assertEqual(rows["attention"]["local_over_vllm"], 2)
        self.assertIsNone(rows["gemm"]["local_over_vllm"])
        self.assertEqual(rows["rope"]["local_us"], 0)

    def test_partial_output_requires_explicit_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "vllm"
            target.mkdir()
            (target / "partial.txt").write_text("interrupted")
            with self.assertRaisesRegex(ValueError, "retry-failed"):
                MODULE.prepare_destination(target, MODULE.complete_vllm, False)
            self.assertTrue(MODULE.prepare_destination(target, MODULE.complete_vllm, True))
            self.assertTrue(target.is_dir())
            self.assertEqual(len(list(Path(directory).glob("vllm-failed-*"))), 1)


if __name__ == "__main__":
    unittest.main()
