"""CPU-only contracts for the matched prefill/mixed step experiment."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import torch


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "experiments/integration/profile_prefill_mixed_vs_vllm.py"
SPEC = importlib.util.spec_from_file_location("profile_prefill_mixed_vs_vllm", PATH)
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
                request_id=request_id, finished=finished,
                outputs=[SimpleNamespace(token_ids=list(range(row["progress"])))],
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


def fake_vllm_modules():
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


class PhaseProfileTests(unittest.TestCase):
    def test_target_kv_snapshot_waits_for_candidate_observer(self):
        pool = SimpleNamespace(
            k_pool=[torch.arange(8.).reshape(2, 2, 1, 2)],
            v_pool=[torch.arange(8., 16.).reshape(2, 2, 1, 2)])
        observer = SimpleNamespace(rows=[])
        with self.assertRaisesRegex(AssertionError, "observer missed"):
            MODULE.snapshot_observed_target_kv(torch, pool, observer, 2)
        observer.rows = [
            {"metadata": (None, None, torch.tensor([1]))},
            {"metadata": (None, None, torch.tensor([3]))}]
        slots, values = MODULE.snapshot_observed_target_kv(torch, pool, observer, 2)
        self.assertEqual(slots.tolist(), [1, 3])
        self.assertEqual(values[0].flatten().tolist(), [2., 3., 6., 7.])
        self.assertEqual(values[1].flatten().tolist(), [10., 11., 14., 15.])

    def test_target_kv_check_reports_corruption(self):
        baseline = [torch.tensor([[[1., 2.]]])]
        close = [torch.tensor([[[1.001, 2.]]])]
        corrupt = [torch.tensor([[[1., float("nan")]]])]
        self.assertEqual(MODULE.compare_target_kv(torch, baseline, close)["status"],
                         "pass")
        result = MODULE.compare_target_kv(torch, baseline, corrupt)
        self.assertEqual(result["status"], "numerical_difference")
        self.assertEqual(result["outside_tolerance"], 1)

    def test_ladder_reuses_completed_arms_and_reports_each_rung(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "reference"
            (reference / "vllm").mkdir(parents=True)
            (reference / "vllm/report.json").write_text(
                '{"model": "cached-model", "unprofiled_median_wall_ms": 7.5}')
            shape = "fixed-b8-l256-o128"
            arms = ("packed", "packed-exact", "packed-exact-qkv")
            for arm, wall in zip(arms, (11.0, 8.0, 7.8)):
                directory = (root / "experiments/results/matched-phase-profile"
                             / f"mixed-{arm}" / shape)
                directory.mkdir(parents=True)
                comparison = {
                    "status": "complete", "work": {
                        "shape_id": shape, "local_mixed_policy": arm,
                        "shared_prefill_budget": 2048},
                    "local_wall_ms": wall, "vllm_wall_ms": 7.5,
                    "local_over_vllm": wall / 7.5,
                    "same_history_target_correctness": {"status": "pass"},
                    "same_history_exact_control_correctness": (
                        {"status": "pass"} if arm == "packed-exact-qkv" else None),
                    "local_cuda_activity": {"categories": [
                        {"category": "gemm", "total_us": wall * 500}]}}
                (directory / "comparison.json").write_text(json.dumps(comparison))
            args = SimpleNamespace(kind="mixed", vllm_reference_dir=reference,
                                   shape_id=shape, suite_dir=root / "suite",
                                   model="cached-model", device="cuda:0", seed=20260914,
                                   arrival_step=5, output_tokens=16, prefill_budget=2048,
                                   local_policy="fa3", warmups=1, repetitions=3,
                                   retry_failed=False, local_mixed_policy="separate")
            with (patch.object(MODULE, "ROOT", root),
                  patch.object(MODULE, "complete_vllm", return_value=True),
                  patch.object(MODULE, "complete_local", return_value=True),
                  patch.object(MODULE, "resolve_model_source", return_value="cached-model"),
                  patch.object(MODULE, "verify_vllm_target"),
                  patch.object(MODULE.subprocess, "run") as launch):
                MODULE.run_ladder(args, {"lengths": [256] * 8}, 4, 4)
            launch.assert_not_called()
            report = json.loads((root / "experiments/results/matched-phase-profile"
                                 / "mixed-ladder" / shape / "ladder.json").read_text())
            self.assertEqual(report["status"], "complete")
            self.assertEqual([row["capture_bucket_tokens"] for row in report["rows"]],
                             [2048, 1028, 1028])
            self.assertAlmostEqual(report["rows"][1]["speedup_vs_broad"], 11 / 8)

    def test_exact_mixed_capture_keeps_scheduler_budget(self):
        case = {"lengths": [256] * 8}
        broad = SimpleNamespace(local_mixed_policy="packed", prefill_budget=2048)
        exact = SimpleNamespace(local_mixed_policy="packed-exact", prefill_budget=2048)
        fused = SimpleNamespace(local_mixed_policy="packed-exact-qkv", prefill_budget=2048)
        self.assertEqual(MODULE.capture_configuration(broad, case, 4, 4), (2048, False))
        self.assertEqual(MODULE.capture_configuration(exact, case, 4, 4), (1028, False))
        self.assertEqual(MODULE.capture_configuration(fused, case, 4, 4), (1028, True))
        self.assertEqual(MODULE.capture_configuration(
            SimpleNamespace(local_mixed_policy="packed-exact", prefill_budget=8192),
            {"lengths": [256] * 64}, 16, 16), (4112, False))

    def test_exact_capture_rejects_bucket_past_scheduler_budget(self):
        args = SimpleNamespace(local_mixed_policy="packed-exact", prefill_budget=1024)
        with self.assertRaisesRegex(ValueError, "capture bucket"):
            MODULE.capture_configuration(args, {"lengths": [256]}, 4, 4)

    def test_target_logit_check_reports_difference_without_dropping_timing(self):
        metadata = (torch.tensor([1]), torch.tensor([0]), torch.tensor([2]),
                    torch.empty(0, dtype=torch.int32), torch.tensor([1]),
                    torch.tensor([[3]], dtype=torch.int32))
        baseline = [{"metadata": metadata, "max_query": 1, "decode": True,
                     "logits": torch.tensor([[0., 1.]])}]
        candidate = [{"metadata": metadata, "max_query": 1, "decode": True,
                      "logits": torch.tensor([[1., 0.]])}]
        result = MODULE.compare_target_logits(torch, baseline, candidate)
        self.assertEqual(result["status"], "numerical_difference")
        self.assertEqual(result["rows"][0]["argmax_differences"], 1)

    def test_vllm_step_mode_is_set_before_import(self):
        with patch.dict(os.environ, {"VLLM_ENABLE_V1_MULTIPROCESSING": "1"}):
            MODULE.configure_vllm_step_mode()
            self.assertEqual(os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"], "0")

    def test_prefill_target_advances_every_new_request_once(self):
        llm = FakeLLM()
        requests = [{"id": i, "prompt": [1] * 256, "output": 12}
                    for i in range(4)]
        with patch.dict(sys.modules, fake_vllm_modules()):
            result = MODULE.vllm_run(llm, requests, 4, 0, "prefill", cumulative=True)
        MODULE.verify_vllm_target(result, 4, 0, "prefill")
        self.assertEqual(set(result["target_progress_before"].values()), {0})
        self.assertEqual(set(result["target_progress_after"].values()), {1})

    def test_mixed_target_advances_decode_and_new_prefill(self):
        llm = FakeLLM()
        requests = [{"id": i, "prompt": [1] * 256, "output": 12}
                    for i in range(8)]
        with patch.dict(sys.modules, fake_vllm_modules()):
            result = MODULE.vllm_run(llm, requests, 4, 5, "mixed", cumulative=True)
        MODULE.verify_vllm_target(result, 4, 4, "mixed")
        self.assertEqual(list(result["target_progress_before"].values()),
                         [5] * 4 + [0] * 4)
        self.assertEqual(list(result["target_progress_after"].values()),
                         [6] * 4 + [1] * 4)

    def test_mixed_target_rejects_prefill_without_decode(self):
        result = {"target_progress_before": {"a": 5, "b": 0},
                  "target_progress_after": {"a": 5, "b": 1}}
        with self.assertRaisesRegex(AssertionError, "did not execute mixed"):
            MODULE.verify_vllm_target(result, 1, 1, "mixed")


if __name__ == "__main__":
    unittest.main()
