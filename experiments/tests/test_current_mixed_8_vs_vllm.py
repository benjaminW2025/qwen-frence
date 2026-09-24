"""CPU-only gates for staged mixed workloads and vLLM step accounting."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import unittest

from experiments.integration import benchmark_current_mixed_8_vs_vllm as mixed


SUITE = Path(__file__).resolve().parents[2] / (
    "experiments/results/full-checkpoint-20260916T033540Z")


class MixedEightTests(unittest.TestCase):
    def test_every_shape_has_cpu_verified_mixed_step_and_graph_bucket(self):
        args = SimpleNamespace(suite_dir=SUITE, seed=20260914)
        expected = ((4, [1024, 1028]), (4, [1024, 1028]),
                    (7, [2048, 2055]), (7, [2048, 2055]),
                    (7, [2048, 2104]), (7, [2048, 2104]),
                    (35, [2048, 2111]), (35, [2048, 2111]))
        for shape_id, (arrival, buckets) in zip(mixed.SHAPES, expected):
            with self.subTest(shape_id=shape_id):
                case, requests, first, step, selected, digest, blocks = mixed.mixed_plan(
                    args, shape_id)
                self.assertEqual(step, arrival)
                self.assertEqual(selected, buckets)
                self.assertEqual(first * 2, case["max_running"])
                self.assertEqual(sum(row["arrival"] == step for row in requests), first)
                self.assertEqual(len(digest), 64)
                self.assertGreater(blocks, 0)

    def test_in_process_vllm_accounting_drains_two_waves(self):
        class Engine:
            def __init__(self):
                self.remaining = {}
                self.progress = {}

            def has_unfinished_requests(self):
                return bool(self.remaining)

            def step(self):
                rows = []
                for key, length in list(self.remaining.items()):
                    self.progress[key] += 1
                    count = self.progress[key]
                    finished = count == length
                    rows.append(SimpleNamespace(
                        request_id=key,
                        outputs=[SimpleNamespace(token_ids=[1] * count)],
                        finished=finished))
                    if finished:
                        del self.remaining[key]
                return rows

        engine = Engine()

        def add_requests(llm, rows, run_id, *, cumulative):
            self.assertTrue(cumulative)
            expected = {}
            for row in rows:
                key = f"{run_id}:{row['id']}"
                engine.remaining[key] = row["output"]
                engine.progress[key] = 0
                expected[key] = row["output"]
            return expected

        requests = [dict(id=index, prompt=[1, 2], output=8) for index in range(4)]
        with (mock.patch("torch.cuda.synchronize"),
              mock.patch("profile_latest_vs_vllm.add_vllm_requests",
                         side_effect=add_requests)):
            row = mixed.vllm_once(SimpleNamespace(llm_engine=engine), requests,
                                  first=2, arrival=3, run_id="test")
        self.assertEqual(len(row["outputs"]), 4)
        self.assertTrue(row["first_wave_advanced_on_injection"])
        self.assertEqual(row["total_steps"], 11)


if __name__ == "__main__":
    unittest.main()
