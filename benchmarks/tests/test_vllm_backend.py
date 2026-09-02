"""CPU-only contract tests for the optional vLLM adapter."""

from types import ModuleType, SimpleNamespace
from unittest import TestCase
from unittest.mock import patch
import sys

import _bootstrap  # noqa: F401

from benchmark_backends import create_backend
from benchmark_core import RequestSpec, Workload


class FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeLLM:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.generate_calls = []
        self.__class__.instances.append(self)

    def generate(self, inputs, params, use_tqdm):
        self.generate_calls.append((inputs, params, use_tqdm))
        return [
            SimpleNamespace(
                outputs=[SimpleNamespace(token_ids=[7] * param.kwargs["max_tokens"])],
                metrics=None,
            )
            for param in params
        ]


class VLLMBackendContractTests(TestCase):
    def test_workload_bounds_and_sampling_are_matched(self):
        fake_vllm = ModuleType("vllm")
        fake_vllm.LLM = FakeLLM
        fake_vllm.SamplingParams = FakeSamplingParams
        fake_transformers = ModuleType("transformers")
        fake_transformers.AutoConfig = SimpleNamespace(
            from_pretrained=lambda _model: SimpleNamespace(
                num_hidden_layers=2,
                num_key_value_heads=2,
                hidden_size=16,
                num_attention_heads=4,
            )
        )
        workload = Workload(
            "matched",
            [
                RequestSpec("a", [1] * 8, 4),
                RequestSpec("b", [2] * 20, 7),
            ],
        )

        with patch.dict(
            sys.modules,
            {"vllm": fake_vllm, "transformers": fake_transformers},
        ):
            backend = create_backend(
                "vllm",
                workload=workload,
                model_id="model",
                device="cuda",
                dtype="float16",
                block_size=16,
                max_running=8,
                num_blocks=None,
                seed=3,
                vllm_gpu_memory_utilization=0.85,
                vllm_kv_cache_mode="matched",
            )
            run = backend.run(workload)

        llm = FakeLLM.instances[-1]
        self.assertEqual(llm.kwargs["max_num_seqs"], 8)
        self.assertEqual(llm.kwargs["max_num_batched_tokens"], 4096)
        self.assertTrue(llm.kwargs["enable_chunked_prefill"])
        self.assertEqual(llm.kwargs["max_model_len"], 27)
        self.assertEqual(llm.kwargs["block_size"], 16)
        # 2 layers * K/V * 2 KV heads * head_dim 4 * fp16 * 16 tokens * 24 blocks.
        self.assertEqual(llm.kwargs["kv_cache_memory_bytes"], 24576)
        self.assertNotIn("gpu_memory_utilization", llm.kwargs)
        self.assertFalse(llm.kwargs["enable_prefix_caching"])
        self.assertFalse(llm.kwargs["enforce_eager"])
        self.assertEqual(llm.kwargs["generation_config"], "vllm")
        self.assertEqual(
            llm.generate_calls[0][0],
            [{"prompt_token_ids": [1] * 8}, {"prompt_token_ids": [2] * 20}],
        )
        for params in llm.generate_calls[0][1]:
            self.assertEqual(params.kwargs["temperature"], 0.0)
            self.assertTrue(params.kwargs["ignore_eos"])
            self.assertFalse(params.kwargs["detokenize"])
        self.assertEqual(run.metadata["max_running"], 8)
        self.assertEqual(run.metadata["matched_num_blocks"], 24)
        self.assertEqual(run.metadata["kv_cache_mode"], "matched-local-pool")
        self.assertFalse(run.metadata["per_request_metrics_available"])

    def test_staggered_workload_is_rejected_before_vllm_initialization(self):
        workload = Workload(
            "poisson",
            [RequestSpec("a", [1], 1, arrival_s=0.5)],
            arrival_pattern="poisson",
        )
        with self.assertRaisesRegex(Exception, "only supports burst"):
            create_backend(
                "vllm",
                workload=workload,
                model_id="model",
                device="cuda",
                dtype="float16",
                block_size=16,
                max_running=8,
                num_blocks=None,
                seed=3,
                vllm_gpu_memory_utilization=0.85,
                vllm_kv_cache_mode="matched",
            )


if __name__ == "__main__":
    import unittest

    unittest.main()
