"""CPU-only contracts for the matched prefill/mixed step experiment."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


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
