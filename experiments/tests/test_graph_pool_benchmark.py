"""Case isolation and failure persistence without allocating CUDA memory."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from experiments.prefill import benchmark_graph_pool as benchmark


class GraphPoolCases(unittest.TestCase):
    def args(self, directory):
        return SimpleNamespace(output=Path(directory) / "report.json",
                               cases=[(8, 256), (8, 2048)], model="model",
                               device="cuda:0", attention="fa3", private_limit=8192,
                               rounds=6, seed=23, gemm_epilogues=False)

    def test_each_case_runs_in_a_separate_process_and_is_aggregated(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(directory)
            calls = []
            def run(command, **kwargs):
                calls.append(command)
                output = Path(command[command.index("--output") + 1])
                case = command[command.index("--cases") + 1]
                sequences, length = map(int, case.split("x"))
                output.write_text(json.dumps({"status": "pass", "rows": [{
                    "sequences": sequences, "length": length,
                    "logits_bitwise_equal": True, "kv_bitwise_equal": True}]}))
                return SimpleNamespace(returncode=0)
            with mock.patch.object(benchmark.subprocess, "run", side_effect=run):
                self.assertEqual(benchmark.run_cases(args), 0)
            self.assertEqual(len(calls), 2)
            self.assertTrue(all("--worker" in command for command in calls))
            report = json.loads(args.output.read_text())
            self.assertEqual(report["status"], "pass")
            self.assertEqual(len(report["rows"]), 2)

    def test_capture_crash_writes_failed_report_and_stops_further_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(directory)
            with mock.patch.object(benchmark.subprocess, "run",
                                   return_value=SimpleNamespace(returncode=1)) as run:
                self.assertEqual(benchmark.run_cases(args), 1)
            self.assertEqual(run.call_count, 1)
            report = json.loads(args.output.read_text())
            self.assertEqual(report["status"], "fail")
            self.assertEqual(report["rows"][0]["exit_code"], 1)


if __name__ == "__main__":
    unittest.main()
