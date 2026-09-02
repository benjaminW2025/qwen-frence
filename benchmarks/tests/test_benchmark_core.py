"""CPU-only tests for the shared benchmark workload and metrics."""

import unittest

import _bootstrap  # noqa: F401

from benchmark_core import (
    BackendRun,
    RequestTrace,
    Workload,
    make_synthetic_workload,
    median_aggregate,
    percentile,
)


class BenchmarkCoreTests(unittest.TestCase):
    def test_workload_round_trip_and_determinism(self):
        kwargs = dict(
            name="test",
            num_requests=4,
            prompt_lengths=[3, 5],
            output_lengths=[2, 4],
            vocab_size=100,
            seed=7,
            request_rate=2.0,
        )
        first = make_synthetic_workload(**kwargs)
        second = make_synthetic_workload(**kwargs)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(Workload.from_dict(first.to_dict()).to_dict(), first.to_dict())
        self.assertTrue(first.has_staggered_arrivals)
        self.assertEqual([len(r.prompt_ids) for r in first.requests], [3, 5, 3, 5])
        self.assertEqual([r.max_tokens for r in first.requests], [2, 4, 2, 4])

    def test_request_and_aggregate_metrics(self):
        traces = [
            RequestTrace(
                request_id="a",
                prompt_tokens=4,
                requested_output_tokens=3,
                arrival_s=0.0,
                submitted_s=0.01,
                token_times_s=[0.10, 0.14, 0.18],
                finished_s=0.18,
                output_tokens=3,
            ),
            RequestTrace(
                request_id="b",
                prompt_tokens=6,
                requested_output_tokens=1,
                arrival_s=0.05,
                submitted_s=0.06,
                token_times_s=[0.20],
                finished_s=0.20,
                output_tokens=1,
            ),
        ]
        run = BackendRun(backend="fake", wall_time_s=0.2, traces=traces)
        metrics = run.aggregate()
        self.assertEqual(metrics["completed_requests"], 2)
        self.assertEqual(metrics["input_tokens"], 10)
        self.assertEqual(metrics["output_tokens"], 4)
        self.assertAlmostEqual(metrics["request_throughput_rps"], 10)
        self.assertAlmostEqual(metrics["output_throughput_tok_s"], 20)
        self.assertAlmostEqual(traces[0].ttft_ms(), 100)
        self.assertAlmostEqual(traces[0].tpot_ms(), 40)
        self.assertEqual(len(traces[0].itl_ms()), 2)
        self.assertIsNone(traces[1].tpot_ms())

    def test_percentile_and_repetition_summary(self):
        self.assertEqual(percentile([0, 10, 20], 50), 10)
        self.assertEqual(percentile([0, 10], 95), 9.5)
        trace = RequestTrace(
            request_id="a",
            prompt_tokens=1,
            requested_output_tokens=1,
            arrival_s=0,
            submitted_s=0,
            token_times_s=[0.1],
            finished_s=0.1,
            output_tokens=1,
        )
        summary = median_aggregate([
            BackendRun("fake", 1.0, [trace]),
            BackendRun("fake", 3.0, [trace]),
        ])
        self.assertEqual(summary["repetitions"], 2)
        self.assertEqual(summary["wall_time_s"], 2)

    def test_unavailable_latency_is_not_fabricated(self):
        trace = RequestTrace(
            request_id="vllm-v1",
            prompt_tokens=4,
            requested_output_tokens=2,
            arrival_s=0,
            submitted_s=0,
            token_times_s=[],
            finished_s=1.0,
            output_tokens=2,
            latency_available=False,
        )
        metrics = BackendRun("vllm", 1.0, [trace]).aggregate()
        self.assertEqual(metrics["completed_requests"], 1)
        self.assertIsNone(metrics["ttft_ms"]["median"])
        self.assertIsNone(metrics["tpot_ms"]["median"])
        self.assertIsNone(metrics["e2e_ms"]["median"])


if __name__ == "__main__":
    unittest.main()
