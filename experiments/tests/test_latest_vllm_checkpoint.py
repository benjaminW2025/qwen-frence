"""CPU-only checks for the frozen integrated-engine/vLLM checkpoint."""

from __future__ import annotations

import importlib.util
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/integration"))
PATH = ROOT / "experiments/integration/benchmark_latest_vs_vllm.py"
SPEC = importlib.util.spec_from_file_location("benchmark_latest_vs_vllm", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CheckpointContractTests(unittest.TestCase):
    def test_table_actions_default_to_eight_factorial_cells(self):
        args = MODULE.build_parser().parse_args(["plan-table", "--output-dir", "/tmp/suite"])
        self.assertEqual(len(MODULE.table_shapes(args)), 8)
        self.assertTrue(all(shape["role"] == "factorial" for shape in MODULE.table_shapes(args)))
        args.include_context_probes = True
        self.assertEqual(len(MODULE.table_shapes(args)), 10)

    def test_table_plan_accounts_for_every_backend_workload(self):
        args = MODULE.build_parser().parse_args(
            ["plan-table", "--output-dir", "/tmp/suite", "--trials", "3",
             "--samples", "2", "--repetitions", "4"])
        output = io.StringIO()
        with redirect_stdout(output):
            MODULE.plan_table(args)
        plan = json.loads(output.getvalue())
        self.assertEqual(len(plan["rows"]), 8)
        self.assertTrue(all(row["ablation_workloads"] == 24 for row in plan["rows"]))
        self.assertTrue(all(row["reference_workloads"] == 8 for row in plan["rows"]))

    def test_table_runner_visits_all_eight_cells_and_both_measured_processes(self):
        with tempfile.TemporaryDirectory() as directory:
            args = MODULE.build_parser().parse_args(
                ["run-table", "--output-dir", directory])
            empty = {"ablation_complete": False, "reference_complete": False}
            with (patch.object(MODULE, "resolve_model_source", return_value="/model"),
                  patch.object(MODULE.importlib.util, "find_spec", return_value=object()),
                  patch.object(MODULE, "prepare_or_validate", return_value="created"),
                  patch.object(MODULE, "check_schedule") as schedule,
                  patch.object(MODULE, "validate_resumed_measurements", return_value=empty),
                  patch.object(MODULE, "run_ablation") as ablation,
                  patch.object(MODULE, "run_reference") as reference,
                  patch.object(MODULE, "analyze") as analyze,
                  patch.object(MODULE, "aggregate_table") as aggregate,
                  redirect_stdout(io.StringIO())):
                MODULE.run_table(args)
            self.assertEqual(schedule.call_count, 8)
            self.assertEqual(ablation.call_count, 8)
            self.assertEqual(reference.call_count, 8)
            self.assertEqual(analyze.call_count, 8)
            aggregate.assert_called_once_with(args)
            self.assertEqual(
                [call.args[0].shape_id for call in ablation.call_args_list],
                [shape["id"] for shape in MODULE.FACTORIAL_SHAPES])

    def test_exact_frozen_prompt_workloads_reach_all_ten_cpp_shapes(self):
        import torch
        try:
            import inference_engine_cpp as cpp
        except ImportError:
            self.skipTest("build C++ extension first")
        from benchmark_integrated_graph import dry_schedule
        from fixed_regime import FIXED_SHAPES, verify_fixed_result

        for shape in FIXED_SHAPES:
            case = MODULE.get_fixed_case(shape["id"])
            _, requests = MODULE.planned_workload(case, 20260914)
            result = dry_schedule(torch, cpp, case, 20260914, requests)
            verify_fixed_result(result, shape["id"])

    def test_splitk_contract_matches_production_policy_without_importing_triton(self):
        from fixed_regime import SPLITK_MIN_CONTEXT

        policy_source = (ROOT / "engine/kvcache/paged_decode_attention.py").read_text()
        self.assertIn(f"MIN_SPLITK_DECODE_CONTEXT_LENGTH = {SPLITK_MIN_CONTEXT}",
                      policy_source)

    def test_fixed_case_and_kv_capacity(self):
        args = MODULE.build_parser().parse_args(["plan"])
        case, blocks = MODULE.contract(args)
        self.assertEqual(case["id"], "fixed-b8-l256-o128")
        self.assertEqual(blocks, 200)
        workload, requests = MODULE.planned_workload(case, args.seed)
        self.assertEqual([len(row["prompt"]) for row in requests], [256] * 8)
        self.assertEqual([row["output"] for row in requests], [128] * 8)
        self.assertEqual([row.request_id for row in workload.requests],
                         [str(i) for i in range(8)])

    def test_all_shape_rows_derive_expected_cohort_and_kv_bounds(self):
        from fixed_regime import FACTORIAL_SHAPES, CONTEXT_PROBES, FIXED_SHAPES, shape_summary

        self.assertEqual(len(FACTORIAL_SHAPES), 8)
        self.assertEqual(len(CONTEXT_PROBES), 2)
        self.assertEqual(len(FIXED_SHAPES), 10)
        for shape in FIXED_SHAPES:
            args = MODULE.build_parser().parse_args(
                ["plan", "--shape-id", shape["id"]])
            case, blocks = MODULE.contract(args)
            summary = shape_summary(shape)
            self.assertEqual(case["max_running"], shape["batch"])
            self.assertEqual(summary["total_prompt_tokens"],
                             shape["prompt_length"] * shape["batch"])
            self.assertEqual(summary["total_output_tokens"],
                             shape["output_length"] * shape["batch"])
            self.assertEqual(summary["max_context_tokens_per_request"],
                             shape["prompt_length"] + shape["output_length"])
            self.assertEqual(blocks, shape["batch"] *
                             ((summary["max_context_tokens_per_request"] + 15) // 16 + 1))

    def test_frozen_workload_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args = MODULE.build_parser().parse_args(
                ["prepare", "--output-dir", directory])
            case, _ = MODULE.contract(args)
            workload, _ = MODULE.planned_workload(case, args.seed)
            workload.requests[0].prompt_ids[0] += 1
            MODULE.atomic_json(Path(directory) / "workload.json", workload.to_dict())
            with self.assertRaisesRegex(ValueError, "frozen workload differs"):
                MODULE.load_frozen(args, case)

    def test_model_cache_gate_checks_tokenizer_and_all_weight_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}")
            (root / "tokenizer_config.json").write_text("{}")
            (root / "tokenizer.json").write_text("{}")
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"a": "model-00001.safetensors",
                                           "b": "model-00002.safetensors"}}))
            (root / "model-00001.safetensors").write_bytes(b"test")
            with self.assertRaisesRegex(ValueError, "missing weight shards"):
                MODULE.validate_model_files(root)
            (root / "model-00002.safetensors").write_bytes(b"test")
            MODULE.validate_model_files(root)

    def test_stage_model_cache_completes_and_validates_pinned_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / MODULE.MODEL_REVISION
            snapshot.mkdir()
            (snapshot / "config.json").write_text("{}")
            (snapshot / "tokenizer_config.json").write_text("{}")
            (snapshot / "tokenizer.json").write_text("{}")
            (snapshot / "model.safetensors").write_bytes(b"test")
            args = MODULE.build_parser().parse_args(["stage-model-cache"])
            with (patch("model_setup.prepare_hub_transfer") as transfer,
                  patch("huggingface_hub.snapshot_download", return_value=str(snapshot)) as download,
                  redirect_stdout(io.StringIO())):
                MODULE.stage_model_cache(args)
            transfer.assert_called_once_with()
            download.assert_called_once_with(repo_id=MODULE.MODEL,
                                             revision=MODULE.MODEL_REVISION)

    def test_incomplete_default_cache_error_names_staging_command(self):
        args = MODULE.build_parser().parse_args(["check-model-cache"])
        with patch("huggingface_hub.snapshot_download", side_effect=FileNotFoundError()):
            with self.assertRaisesRegex(ValueError, "stage-model-cache"):
                MODULE.resolve_model_source(args)

    def test_analyzer_reports_net_gain_and_remaining_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            args = MODULE.build_parser().parse_args(
                ["analyze", "--output-dir", directory])
            case, blocks = MODULE.contract(args)
            workload, requests = MODULE.planned_workload(case, args.seed)
            root = Path(directory)
            model_source = str(root / "snapshots" / MODULE.MODEL_REVISION)
            outputs = {str(i): [0] * 128 for i in range(8)}
            run_requests = [{"request_id": str(i), "output_ids": [0] * 128}
                            for i in range(8)]
            MODULE.atomic_json(root / "workload.json", workload.to_dict())
            MODULE.atomic_json(root / "ablation/manifest.json", {
                "model": model_source, "plan": {
                    "requests_sha256": MODULE.requests_fingerprint(requests)}})
            MODULE.atomic_json(root / "ablation/report.json", {
                "status": "ok", "case_id": case["id"],
                "actual_work": {"max_actual_decode_batch": 8,
                                "pure_full_decode_steps_at_target_batch": 127,
                                "prefill_token_counts": [2048]},
                "trial_medians_ms": {"wall_ms": {
                    "eager": [120, 120, 120], "decode_graph": [100, 100, 100],
                    "piecewise": [80, 80, 80], "piecewise_splitk": [64, 64, 64]}},
                "capture": {"piecewise": {"timed_prefill_capture_calls": 1},
                            "piecewise_splitk": {"timed_prefill_capture_calls": 1}},
                "checks": {"piecewise_splitk": {"decisions": {"production": 60}}},
                "output_ids": outputs,
                "splitk_decision": {"choice": "inactive_short_context"}, "effects": {}})
            MODULE.atomic_json(root / "reference/reference.json", {
                "workload": workload.to_dict(), "configuration": {
                    "model": model_source, "dtype": "float16", "block_size": 16,
                    "max_running": 8, "max_num_batched_tokens": 2048,
                    "num_blocks": blocks, "vllm_kv_cache_mode": "matched"},
                "comparison_contract": {"sampling": "greedy-temperature-0-ignore-eos",
                                        "max_concurrent_sequences": 8,
                                        "identical_prompt_token_ids": True,
                                        "identical_requested_output_lengths": True},
                "backends": {
                    "regime-dispatched": {"summary": {"output_throughput_tok_s": 8000},
                                          "runs": [{"requests": run_requests}]},
                    "vllm": {"summary": {"output_throughput_tok_s": 20000},
                             "runs": [{"metadata": {"max_num_seqs": 8,
                                                    "max_num_batched_tokens": 2048,
                                                    "max_model_len": 384,
                                                    "kv_cache_mode": "matched-local-pool",
                                                    "prefix_caching": False,
                                                    "cuda_graphs": True},
                                       "requests": run_requests}]}}})
            with redirect_stdout(io.StringIO()):
                MODULE.analyze(args, case, blocks)
            summary = json.loads((root / "summary.json").read_text())
            self.assertAlmostEqual(summary["integrated_production_vs_old"], 1.6)
            self.assertAlmostEqual(summary["integrated_splitk_vs_old"], 2.0)
            self.assertAlmostEqual(summary["vllm_vs_integrated_splitk"], 1.25)
            self.assertTrue(summary["output_agreement"]["old_vs_integrated"])
            broken = json.loads((root / "reference/reference.json").read_text())
            broken["comparison_contract"]["sampling"] = "temperature-1"
            MODULE.atomic_json(root / "reference/reference.json", broken)
            with self.assertRaisesRegex(ValueError, "sampling or workload contract"):
                MODULE.analyze(args, case, blocks)
            fixed = json.loads((root / "ablation/report.json").read_text())
            fixed["checks"]["piecewise_splitk"]["decisions"] = {"H1-K8-S2": 60}
            MODULE.atomic_json(root / "ablation/report.json", fixed)
            with self.assertRaisesRegex(ValueError, "split-K execution"):
                MODULE.analyze(args, case, blocks)

    def test_gpu_commands_parse_with_frozen_contract_without_running(self):
        from benchmark_integrated_graph import build_parser as integrated_parser
        from run_benchmarks import build_parser as reference_parser

        with tempfile.TemporaryDirectory() as directory:
            args = MODULE.build_parser().parse_args(
                ["plan", "--output-dir", directory])
            case, blocks = MODULE.contract(args)
            workload, _ = MODULE.planned_workload(case, args.seed)
            MODULE.atomic_json(Path(directory) / "workload.json", workload.to_dict())
            model_source = str(Path(directory) / "snapshots" / MODULE.MODEL_REVISION)
            with (patch.object(MODULE.subprocess, "run") as called,
                  patch.object(MODULE, "resolve_model_source", return_value=model_source)):
                MODULE.run_ablation(args, case)
            command = called.call_args.args[0]
            parsed = integrated_parser().parse_args(command[2:])
            self.assertEqual(parsed.preset, "fixed")
            self.assertEqual(parsed.expected_max_decode_batch, 8)
            self.assertEqual(parsed.min_full_decode_steps, 64)
            self.assertEqual(parsed.expected_prefill_tokens, 2048)
            self.assertEqual(parsed.prefill_buckets, [2048])
            self.assertEqual(parsed.workload_in, Path(directory) / "workload.json")
            self.assertEqual(parsed.model, model_source)
            with (patch.object(MODULE.subprocess, "run") as called,
                  patch.object(MODULE.importlib.util, "find_spec", return_value=object()),
                  patch.object(MODULE, "resolve_model_source", return_value=model_source)):
                MODULE.run_reference(args, case, blocks)
            command = called.call_args.args[0]
            parsed = reference_parser().parse_args(command[2:])
            self.assertEqual(parsed.backends, ["regime-dispatched", "vllm"])
            self.assertEqual(parsed.max_num_batched_tokens, 2048)
            self.assertEqual(parsed.num_blocks, 200)
            self.assertEqual(parsed.vllm_kv_cache_mode, "matched")
            self.assertEqual(parsed.workload_in, Path(directory) / "workload.json")
            self.assertEqual(parsed.model, model_source)

    def test_profile_command_uses_same_shape_prompt_ids_and_pinned_model(self):
        from profile_cpp_control import build_parser as profile_parser

        with tempfile.TemporaryDirectory() as directory:
            args = MODULE.build_parser().parse_args(
                ["run-profile", "--shape-id", "probe-b64-l4096-o256",
                 "--output-dir", directory, "--profile-kind", "prefill",
                 "--profile-occurrence", "1", "--profile-decode-policy", "splitk"])
            case, _ = MODULE.contract(args)
            workload, _ = MODULE.planned_workload(case, args.seed)
            MODULE.atomic_json(Path(directory) / "workload.json", workload.to_dict())
            with (patch.object(MODULE.subprocess, "run") as called,
                  patch.object(MODULE, "resolve_model_source", return_value="/pinned-model")):
                MODULE.run_profile(args, case)
            parsed = profile_parser().parse_args(called.call_args.args[0][2:])
            self.assertEqual(parsed.preset, "fixed")
            self.assertEqual(parsed.case_id, "probe-b64-l4096-o256")
            self.assertEqual(parsed.model, "/pinned-model")
            self.assertEqual(parsed.workload_in, Path(directory) / "workload.json")
            self.assertEqual(parsed.kind, "prefill")
            self.assertEqual(parsed.occurrence, 1)
            self.assertEqual(parsed.decode_attention_policy, "splitk")
            self.assertEqual(parsed.repetitions, 1)
            self.assertEqual(parsed.warmups, 0)

    def test_large_shape_commands_use_their_own_batch_and_pool(self):
        from benchmark_integrated_graph import build_parser as integrated_parser
        from run_benchmarks import build_parser as reference_parser

        from fixed_regime import FIXED_SHAPES, shape_summary

        with tempfile.TemporaryDirectory() as directory:
            for shape in FIXED_SHAPES:
                shape_id = shape["id"]
                batch = shape["batch"]
                result_dir = Path(directory) / shape_id
                args = MODULE.build_parser().parse_args(
                    ["plan", "--shape-id", shape_id, "--output-dir", str(result_dir)])
                case, blocks = MODULE.contract(args)
                workload, _ = MODULE.planned_workload(case, args.seed)
                MODULE.atomic_json(result_dir / "workload.json", workload.to_dict())
                with (patch.object(MODULE.subprocess, "run") as called,
                      patch.object(MODULE, "resolve_model_source", return_value="/model")):
                    MODULE.run_ablation(args, case)
                parsed = integrated_parser().parse_args(called.call_args.args[0][2:])
                self.assertEqual(parsed.case_id, shape_id)
                self.assertEqual(parsed.expected_max_decode_batch, batch)
                self.assertEqual(parsed.min_full_decode_steps, 64)
                with (patch.object(MODULE.subprocess, "run") as called,
                      patch.object(MODULE.importlib.util, "find_spec", return_value=object()),
                      patch.object(MODULE, "resolve_model_source", return_value="/model")):
                    MODULE.run_reference(args, case, blocks)
                parsed = reference_parser().parse_args(called.call_args.args[0][2:])
                self.assertEqual(parsed.max_running, batch)
                self.assertEqual(parsed.num_blocks,
                                 shape_summary(shape)["logical_kv_blocks_with_headroom"])


if __name__ == "__main__":
    unittest.main()
