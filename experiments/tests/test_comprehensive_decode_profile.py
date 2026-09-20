"""CPU-only protocol checks for the comprehensive decode diagnostic."""

from __future__ import annotations

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/integration"))
PATH = ROOT / "experiments/integration/profile_decode_comprehensive.py"
SPEC = importlib.util.spec_from_file_location("profile_decode_comprehensive", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ComprehensiveDecodeTests(unittest.TestCase):
    def args(self):
        return MODULE.build_parser().parse_args([
            "plan", "--suite-dir", "/tmp/suite", "--output-dir", "/tmp/output",
        ])

    def test_surface_is_two_batches_by_three_contexts(self):
        shapes = [MODULE.get_fixed_shape(shape) for shape in MODULE.SLOPE_SHAPES]
        self.assertEqual({shape["batch"] for shape in shapes}, {8, 64})
        self.assertEqual({shape["prompt_length"] for shape in shapes}, {256, 2048, 4096})
        self.assertEqual(len(shapes), 6)

    def test_plan_exposes_gpu_cost_before_running(self):
        args = self.args()
        output = io.StringIO()
        with redirect_stdout(output):
            MODULE.plan(args)
        plan = json.loads(output.getvalue())
        self.assertEqual(plan["local_model_loads"], 1)
        self.assertEqual(plan["vllm_model_loads"], 1)
        self.assertEqual(plan["local_measured_arms"], 10)
        self.assertEqual(plan["local_full_workloads_total"], 51)
        self.assertEqual(plan["vllm_full_workloads_total"], 31)
        self.assertEqual(plan["heavy_traces"], {"local": 1, "vllm": 1})
        self.assertFalse(plan["decode_graph_captures"]["power_of_two_ladder"])
        self.assertIn("exact per-cell maximum context", plan[
            "decode_graph_captures"]["full_model"])
        self.assertTrue(plan["output_head_interventions"]["full_model_splitk_ab"])
        self.assertEqual(len(plan["output_head_interventions"][
            "fused_tiled_projection_argmax"]), 3)
        self.assertEqual(plan["unrolled_decode"]["steps"], [2, 4, 8])
        self.assertTrue(plan["unrolled_decode"]["validation_retains_every_logit"])

    def test_context_slope_has_time_per_token_units(self):
        slope_ms_per_token = MODULE.linear_slope([(256, 1.0), (2048, 2.0), (4096, 3.0)])
        self.assertGreater(slope_ms_per_token, 0)
        self.assertAlmostEqual(MODULE.linear_slope([(1, 2), (2, 4)]), 2.0)

    def test_atomic_json_never_leaves_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested/report.json"
            MODULE.atomic_json(path, {"status": "complete"})
            self.assertEqual(json.loads(path.read_text()), {"status": "complete"})
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_selected_prefill_budgets_match_regime_decision(self):
        self.assertEqual(MODULE.local_budget("fixed-b8-l2048-o128"), 4096)
        self.assertEqual(MODULE.local_budget("fixed-b64-l2048-o128"), 8192)

    def test_analyzer_requires_and_combines_complete_surface(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args()
            args.output_dir = Path(directory)
            fingerprint, protocol = MODULE.protocol_fingerprint(args)
            cuda = {"summed_cuda_activity_us": 1000.0, "activity_count": 10,
                    "categories": [{"category": "attention", "calls": 1,
                                    "total_us": 1000.0,
                                    "percent_of_cuda_activity": 100.0}],
                    "kernels": []}
            local_rows, vllm_rows = [], []
            for shape_id in MODULE.SLOPE_SHAPES:
                shape = MODULE.get_fixed_shape(shape_id)
                context = shape["prompt_length"] + 33
                splitk = {"policy": "splitk", "median_wall_ms": context / 1000,
                          "context": {"median": context},
                          "cuda_activity": cuda if shape_id == args.trace_shape else None}
                production = {"policy": "production",
                              "median_wall_ms": context / 900,
                              "context": {"median": context}, "cuda_activity": None}
                arms = [splitk] if shape["prompt_length"] < 1024 else [production, splitk]
                local_rows.append({"shape_id": shape_id, "batch": shape["batch"],
                                   "prompt": shape["prompt_length"], "arms": arms})
                vllm_rows.append({"shape_id": shape_id, "median_wall_ms": context / 2000,
                                  "context": {"median": context},
                                  "cuda_activity": cuda if shape_id == args.trace_shape else None})
            MODULE.atomic_json(args.output_dir / "local/report.json", {
                "status": "complete", "fingerprint": fingerprint, "protocol": protocol,
                "model": "/model", "slope": local_rows,
                "attention": [], "fusion": [], "rope_kv_fusion": [],
                "gemm": [{
                    "batch": batch,
                    "current_route_estimates": {
                        "graph_cache_evicted": {
                            "projected_28_layer_plus_lm_head_ms": 0.1,
                        },
                    },
                } for batch in MODULE.DECODE_BATCHES],
                "output_head": [], "output_head_full_model": [],
                "unrolled_decode": [],
            })
            MODULE.atomic_json(args.output_dir / "vllm/report.json", {
                "status": "complete", "fingerprint": fingerprint, "protocol": protocol,
                "model": "/model", "vllm_version": MODULE.PINNED_VLLM,
                "slope": vllm_rows,
            })
            with redirect_stdout(io.StringIO()):
                MODULE.analyze(args)
            summary = json.loads((args.output_dir / "summary.json").read_text())
            self.assertEqual(len(summary["rows"]), 6)
            self.assertEqual(summary["rows"][0]["local_splitk_over_vllm"], 2.0)
            self.assertAlmostEqual(summary["decode_context_slopes"][0][
                "local_us_per_context_token"], 1.0)
            self.assertEqual([row["batch"] for row in summary["gemm"]], [8, 64])
            self.assertGreater(summary["gemm"][0]["derived"][
                "projected_cache_evicted_gemm_fraction_of_local_step"], 0)


if __name__ == "__main__":
    unittest.main()
