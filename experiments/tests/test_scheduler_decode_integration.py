"""Matched scheduling, factorial accounting, and frozen-policy gate tests."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "experiments/integration", ROOT / "engine/cpp/build", ROOT / "baseline",
             ROOT / "engine/model_runner", ROOT / "engine/kvcache", ROOT / "engine/graph"):
    sys.path.insert(0, str(path))
import math
from design import (POLICY_MAX_CONTEXT, POLICY_SPLIT_CONTEXT, load_policy, longctx_plan,
                    make_plan, make_requests, select_action, stable_hash, summarize,
                    validate_checkpoint, validate_tree, variant_order)
import torch
from python_control import PythonControl
try:
    import inference_engine_cpp as cpp
except ImportError:
    cpp = None


class GraphAdapterTests(unittest.TestCase):
    def test_varlen_attention_uses_only_live_rows_from_padded_graph(self):
        import piecewise_prefill

        tokens, bucket = 258, 512
        cfg = SimpleNamespace(n_heads=2, d_head=2)
        model = SimpleNamespace(cfg=cfg, layers=[object()], lm_head=lambda x: x)
        pool = SimpleNamespace(k_pool=[torch.zeros(1)], v_pool=[torch.zeros(1)])
        prefill = piecewise_prefill.PiecewisePrefill.__new__(
            piecewise_prefill.PiecewisePrefill)
        prefill.model, prefill.pool = model, pool
        prefill.enable_residual_rmsnorm = True
        prefill.graph_replays = 0
        initial = SimpleNamespace(
            positions=torch.zeros(bucket, dtype=torch.long),
            slots=torch.zeros(bucket, dtype=torch.long),
            valid_tokens=torch.zeros((), dtype=torch.int32),
            run_initial=lambda ids, count: (
                torch.zeros((1, cfg.n_heads, bucket, cfg.d_head)),
                torch.zeros((bucket, cfg.n_heads * cfg.d_head))),
        )

        def from_attention(residual, attention, count):
            self.assertEqual(attention.shape, (tokens, cfg.n_heads * cfg.d_head))
            return torch.zeros((bucket, cfg.n_heads * cfg.d_head))

        prefill.pieces = lambda count: [initial, SimpleNamespace(
            run_from_attention=from_attention)]

        def varlen_attention(query, *args, **kwargs):
            self.assertEqual(query.shape, (tokens, cfg.n_heads, cfg.d_head))
            return torch.zeros_like(query)

        with mock.patch("kernel_dispatch.fa3_paged_varlen_attention",
                        side_effect=varlen_attention):
            logits = prefill.forward(
                torch.zeros(tokens, dtype=torch.long),
                torch.arange(tokens), torch.arange(tokens),
                torch.tensor([0, 1, tokens], dtype=torch.int32),
                torch.tensor([1, tokens - 1], dtype=torch.int32),
                torch.zeros((2, 1), dtype=torch.int32), tokens - 1,
                mixed_attention_policy="fa3_varlen")
        self.assertEqual(logits.shape, (2, cfg.n_heads * cfg.d_head))
        self.assertEqual(prefill.graph_replays, 2)

    def test_piecewise_prefill_shape_limit_is_explicit(self):
        import piecewise_prefill

        class FakePiece:
            def __init__(self, *args):
                self.args = args

        cfg = SimpleNamespace()
        model = SimpleNamespace(cfg=cfg, layers=[object(), object()])
        pool = SimpleNamespace(k_pool=[torch.zeros(1), torch.zeros(1)],
                               v_pool=[torch.zeros(1), torch.zeros(1)])
        prefill = piecewise_prefill.PiecewisePrefill.__new__(piecewise_prefill.PiecewisePrefill)
        prefill.model, prefill.pool = model, pool
        prefill.max_capture_tokens, prefill.max_shapes = 128, 1
        prefill.buckets = (64, 128)
        prefill.shapes, prefill.captured_calls, prefill.eager_calls = {}, 0, 0
        with mock.patch.object(piecewise_prefill, "_AttentionBoundary", FakePiece):
            first = prefill.pieces(64)
            self.assertIs(first, prefill.pieces(64))
            self.assertIsNone(prefill.pieces(65))
            self.assertIsNone(prefill.pieces(129))
        self.assertEqual(len(first), 3)  # Two layers need three attention-boundary graphs.
        self.assertIs(first[0].args[4], first[1].args[4])
        self.assertIs(first[0].args[5], first[1].args[5])
        self.assertIs(first[0].args[6], first[1].args[6])
        self.assertEqual(sorted(prefill.shapes), [64])
        self.assertEqual((prefill.captured_calls, prefill.eager_calls), (2, 2))

    def test_prefill_bucket_selection_pads_without_creating_new_shapes(self):
        from piecewise_prefill import PiecewisePrefill

        prefill = PiecewisePrefill.__new__(PiecewisePrefill)
        prefill.buckets = (128, 256, 512, 1024, 2048)
        prefill.max_shapes = 2
        prefill.model = SimpleNamespace(layers=[object()])
        prefill.pool = SimpleNamespace(k_pool=[torch.zeros(1)])
        prefill.shapes = {}
        prefill.captured_calls = prefill.eager_calls = 0
        with mock.patch("piecewise_prefill._AttentionBoundary", side_effect=lambda *a: object()):
            self.assertIs(prefill.pieces(127), prefill.pieces(128))
            self.assertIs(prefill.pieces(255), prefill.pieces(256))
            self.assertIsNone(prefill.pieces(257))  # New bucket exceeds shape limit.
            self.assertIsNone(prefill.pieces(2049))
        self.assertEqual(sorted(prefill.shapes), [128, 256])
        self.assertEqual((prefill.captured_calls, prefill.eager_calls), (4, 2))

    def test_prefill_fusion_flags_reach_every_captured_boundary(self):
        import piecewise_prefill

        captured = []

        class FakePiece:
            def __init__(self, *args):
                captured.append(args)

        prefill = piecewise_prefill.PiecewisePrefill.__new__(
            piecewise_prefill.PiecewisePrefill
        )
        prefill.model = SimpleNamespace(layers=[object(), object()])
        prefill.pool = SimpleNamespace(k_pool=[torch.zeros(1), torch.zeros(1)])
        prefill.buckets, prefill.max_shapes = (64,), 1
        prefill.shapes = {}
        prefill.captured_calls = prefill.eager_calls = 0
        prefill.enable_packed_qkv_rope_cache = True
        prefill.enable_residual_rmsnorm = True
        prefill.enable_swiglu_fusion = True
        with mock.patch.object(piecewise_prefill, "_AttentionBoundary", FakePiece):
            prefill.pieces(63)
        self.assertEqual(len(captured), 3)
        self.assertTrue(all(args[7:] == (True, True, True) for args in captured))

    def test_piecewise_adapter_prefill_dispatch_records_once(self):
        from model_adapter import PiecewiseGraphModelAdapter

        adapter = PiecewiseGraphModelAdapter.__new__(PiecewiseGraphModelAdapter)
        calls = []
        adapter.piecewise_prefill = SimpleNamespace(
            forward=lambda *args, **kwargs: calls.append(args) or torch.ones(2, 7))
        adapter.step_calls = []
        adapter.observer = lambda args, logits: calls.append((args, logits))
        ids = torch.tensor([1, 2, 3])
        positions = torch.tensor([0, 1, 2])
        slots = torch.tensor([0, 1, 2])
        cu = torch.tensor([0, 2, 3])
        context = torch.tensor([2, 1])
        table = torch.zeros(2, 1)
        logits = adapter(ids, positions, slots, cu, context, table, 2, False)
        self.assertEqual(tuple(logits.shape), (2, 7))
        self.assertEqual(adapter.step_calls, [(False, 3, 2, 2)])
        self.assertEqual(len(calls), 2)
        self.assertIs(calls[1][1], logits)

    def test_cpp_decode_metadata_reaches_graph_and_returns_logits_rows(self):
        from model_adapter import GraphModelAdapter

        created = []
        graphs = type(sys)("bucketed_graph_decoder")

        class FakeGraph:
            def __init__(self, *args, **kwargs):
                created.append((args, kwargs))

            def decode(self, ids, positions, context, table, slots):
                self.inputs = (ids, positions, context, table, slots)
                return torch.ones(ids.shape[0], 1, 7)

        graphs.BucketedGraphDecoder = FakeGraph
        attention = type(sys)("paged_decode_attention")
        attention.resolve_decode_attention_policy = (
            lambda policy, bound: "production" if bound < 1024 else policy
        )
        attention.select_splitk_config = lambda bound, page_size: {"split_k": 8, "num_stages": 2}
        pool = SimpleNamespace(block_size=16, k_pool=[torch.zeros(2, 16, 2, 8)],
                               v_pool=[torch.zeros(2, 16, 2, 8)])

        with mock.patch.dict(sys.modules, {"bucketed_graph_decoder": graphs,
                                          "paged_decode_attention": attention}):
            adapter = GraphModelAdapter(SimpleNamespace(), pool, SimpleNamespace(),
                                        max_running=4, max_context_length=1023,
                                        decode_attention_policy="splitk")
            ids = torch.tensor([2, 3])
            positions = torch.tensor([20, 30])
            slots = torch.tensor([1, 2])
            context = torch.tensor([21, 31], dtype=torch.int32)
            table = torch.tensor([[1, 0], [0, 1]], dtype=torch.int32)
            logits = adapter(ids, positions, slots, torch.empty(0), context, table, 1, True)
            splitk_adapter = GraphModelAdapter(SimpleNamespace(), pool, SimpleNamespace(),
                                               max_running=4, max_context_length=2048,
                                               decode_attention_policy="splitk")
            splitk_adapter(ids, positions, slots, torch.empty(0), context, table, 1, True)

        self.assertEqual(tuple(logits.shape), (2, 7))
        self.assertEqual(adapter.decisions, {"production": 1})
        self.assertEqual(splitk_adapter.decisions, {"H1-K8-S2": 1})
        self.assertEqual(adapter.step_calls, [(True, 2, 2, 1)])
        self.assertEqual(created[0][1]["max_decode_context_length"], 1023)
        self.assertEqual(created[0][0][3], 64)
        graph_ids, graph_pos, graph_context, graph_table, graph_slots = adapter.graph_decoder.inputs
        self.assertEqual(tuple(graph_ids.shape), (2, 1))
        for actual, expected in ((graph_pos, positions), (graph_context, context),
                                 (graph_table, table), (graph_slots, slots)):
            self.assertTrue(torch.equal(actual, expected))


class DesignTests(unittest.TestCase):
    def test_model_revision_survives_config_serialization(self):
        from benchmark_scheduler_decode import resolved_model_revision
        config = SimpleNamespace(_commit_hash="immutable-revision", to_dict=lambda: {})
        self.assertNotIn("_commit_hash", config.to_dict())
        self.assertEqual(resolved_model_revision(config), "immutable-revision")
        with self.assertRaisesRegex(ValueError, "immutable"):
            resolved_model_revision(SimpleNamespace())

    def test_longctx_preset_brackets_the_policy_split_and_stays_in_range(self):
        plan = longctx_plan()
        self.assertEqual(plan, make_plan("longctx"))
        peaks = {c["id"]: max(n + o for n, o in zip(c["lengths"], c["outputs"])) for c in plan}
        # Every case must stay inside the frozen policy's page range, or both arms
        # silently run production attention and the effect is vacuously 1.0.
        self.assertLessEqual(max(peaks.values()), POLICY_MAX_CONTEXT)
        # The split must be bracketed from both sides, which neither existing
        # preset does: smoke tops out at 49 pages, full jumps 512 -> 8192.
        self.assertTrue(any(p <= POLICY_SPLIT_CONTEXT for p in peaks.values()))
        self.assertTrue(any(p > POLICY_SPLIT_CONTEXT for p in peaks.values()))
        self.assertEqual(peaks["uniform-b4-l1216"], POLICY_SPLIT_CONTEXT)
        self.assertGreater(peaks["uniform-b4-l1280"], POLICY_SPLIT_CONTEXT)
        # A case pushed past the page range must fail loudly rather than measure nothing.
        with self.assertRaisesRegex(ValueError, "page range"):
            longctx_plan(output=POLICY_MAX_CONTEXT)

    def test_longctx_batch_axis_holds_the_action_fixed(self):
        plan = {c["id"]: c for c in longctx_plan()}
        sweep = [plan[f"saturated-b{b}-l512"] for b in (32, 64, 96, 128)]
        # The batch crossover must be read at a fixed action, or a change of batch
        # and a change of K/stages arrive together and neither is attributable.
        self.assertTrue(all(max(n + o for n, o in zip(c["lengths"], c["outputs"]))
                            <= POLICY_SPLIT_CONTEXT for c in sweep))
        self.assertEqual([c["max_running"] for c in sweep], [32, 64, 96, 128])
        # Declared batch must actually form; the runner enforces this for "saturated".
        self.assertTrue(all(c["kind"] == "saturated" for c in sweep))
        for case in sweep:
            admit = math.ceil(case["max_running"] * 512 / case["prefill_budget"])
            self.assertGreaterEqual(max(case["outputs"]) - admit, 64)

    def test_dispatch_coverage_separates_fallback_from_no_effect(self):
        from benchmark_scheduler_decode import dispatch_coverage
        records = [{"case_id": "a", "variant": "python-selected", "decisions": {"H1-K22-S3": 7}},
                   {"case_id": "a", "variant": "cpp-selected", "decisions": {"H1-K22-S3": 7}},
                   {"case_id": "b", "variant": "python-selected", "decisions": {"production": 7}},
                   {"case_id": "c", "variant": "python-production", "decisions": {"production": 7}}]
        coverage = dispatch_coverage(records)
        self.assertEqual(coverage["a"], {"actions": {"H1-K22-S3": 14},
                                         "production_fallback": False,
                                         "actions_measured": ["H1-K22-S3"]})
        self.assertTrue(coverage["b"]["production_fallback"])
        self.assertEqual(coverage["b"]["actions_measured"], [])
        self.assertNotIn("c", coverage)  # production arms carry no policy decision

    def test_deterministic_workloads_and_paired_orders(self):
        for preset in ("smoke", "full", "longctx"):
            plan = make_plan(preset)
            self.assertEqual(len(plan), len({c["id"] for c in plan}))
            self.assertTrue(any(c["kind"] == "staggered" for c in plan))
            for case in plan:
                a = make_requests(case, 4, 128)
                self.assertEqual(a, make_requests(case, 4, 128))
                self.assertNotEqual(a, make_requests(case, 5, 128))
                self.assertLessEqual(max(n + o for n, o in zip(case["lengths"], case["outputs"])), 32768)
        self.assertEqual(len(variant_order("scheduler", 1)), 2)
        self.assertEqual(set(variant_order("combined", 1)), set(variant_order("combined", 2)))

    def test_factorial_effects_and_incomplete_rejection(self):
        values = {"python-production": 100, "cpp-production": 50,
                  "python-selected": 80, "cpp-selected": 20}
        records = [{"case_id": "a", "trial": t, "sample": s, "variant": name,
                    "wall_ms": value * (1 + t / 10)}
                   for t in range(3) for s in range(2) for name, value in values.items()]
        result = summarize(records, "combined", 3, 2)[0]["effects"]
        self.assertEqual(result["scheduler"]["ratio"], 2)
        self.assertAlmostEqual(result["combined"]["ratio"], 5)
        self.assertEqual(result["interaction"]["ratio"], 2)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            summarize(records[:-1], "combined", 3, 2)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            summarize(records + records[:1], "combined", 3, 2)

    def test_analysis_keeps_partial_cases_pending(self):
        from benchmark_scheduler_decode import analyze
        from contextlib import redirect_stdout
        import io
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            manifest = {"fingerprint": "test", "phase": "scheduler", "trials": 2,
                        "samples": 1, "seed": 0, "plan": [{"id": "a"}, {"id": "b"}]}
            (directory / "manifest.json").write_text(json.dumps(manifest))
            (directory / "trials").mkdir()
            def write(trial):
                records = [{"case_id": "a", "trial": trial, "sample": 0, "variant": name,
                            "wall_ms": value, "decode_wall_ms": value / 2, "prefill_wall_ms": value / 2, "mixed_wall_ms": 0}
                           for name, value in (("python-production", 10), ("cpp-production", 5))]
                (directory / "trials" / f"a-{trial}.json").write_text(json.dumps(
                    {"status": "complete", "fingerprint": "test", "records": records, "case_id": "a", "trial": trial,
                     "checks": {name: {"metadata_equal": True, "outputs_equal": True} for name in ("python-production", "cpp-production")}}))
            write(0)
            with redirect_stdout(io.StringIO()):
                analyze(SimpleNamespace(output_dir=directory))
            report = json.loads((directory / "report.json").read_text())
            self.assertEqual(report["cases"], [])
            self.assertEqual(report["pending_observations"]["a"], 2)
            write(1)
            with redirect_stdout(io.StringIO()):
                analyze(SimpleNamespace(output_dir=directory))
            report = json.loads((directory / "report.json").read_text())
            self.assertEqual(report["status"], "partial")
            self.assertEqual(len(report["cases"]), 1)
            self.assertEqual(report["cases"][0]["effects"]["scheduler"]["ratio"], 2)
            self.assertEqual(len(report["phase_effects"]["decode"]), 1)
            self.assertEqual(report["phase_effects"]["mixed"], [])

    def test_complete_checkpoint_rejects_missing_pairs_and_nonfinite_times(self):
        manifest = {"fingerprint": "test", "phase": "scheduler", "trials": 1, "samples": 1,
                    "plan": [{"id": "a"}]}
        names = ("python-production", "cpp-production")
        payload = {"status": "complete", "fingerprint": "test", "case_id": "a", "trial": 0,
                   "checks": {n: {"metadata_equal": True, "outputs_equal": True} for n in names},
                   "records": [{"case_id": "a", "trial": 0, "sample": 0, "variant": n,
                                "wall_ms": 2., "decode_wall_ms": 1., "prefill_wall_ms": 1., "mixed_wall_ms": 0.}
                               for n in names]}
        validate_checkpoint(payload, manifest)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate_checkpoint({**payload, "records": payload["records"][:1]}, manifest)
        bad = copy.deepcopy(payload)
        bad["records"][0]["mixed_wall_ms"] = float("nan")
        with self.assertRaisesRegex(ValueError, "timing"):
            validate_checkpoint(bad, manifest)
        with self.assertRaisesRegex(ValueError, "correctness"):
            validate_checkpoint({**payload, "checks": {}}, manifest)

    def test_dispatch_fallback_and_tree_validation(self):
        policy = {"tree": {"feature": "max_pages", "threshold": 16,
                            "left": {"action": [4, 2]}, "right": {"action": [16, 3]}},
                  "batch_range": [1, 128], "pages_range": [8, 1024]}
        self.assertEqual(select_action(policy, 8, 256), (4, 2))
        self.assertEqual(select_action(policy, 8, 257), (16, 3))
        self.assertIsNone(select_action(policy, 129, 256))
        self.assertIsNone(select_action(policy, 8, 1))
        for tree in ({"action": [0, 2]}, {"action": [2, 9]}, {"feature": "bad"}):
            with self.assertRaises(ValueError):
                validate_tree(tree)

    def test_policy_requires_gate_source_identity_and_hardware(self):
        source = ROOT / "experiments/decode/paged_decode_grouped_splitk_pipelined.py"
        spec = {"hardware": {"name": "H100"}, "source_hashes": {source.name: stable_hash(source.read_text()),
                    "../../engine/kvcache/paged_decode_attention.py": stable_hash((ROOT / "engine/kvcache/paged_decode_attention.py").read_text())},
                "plan": [{"suite": "train", "batch": 1, "features": {"max_pages": 8}},
                         {"suite": "test", "batch": 128, "features": {"max_pages": 1024}}]}
        fingerprint = stable_hash(spec)
        report = {"status": "microbenchmark_candidate", "fingerprint": fingerprint,
                  "policy": {"selected": {"tree": {"action": [16, 3]}}}}
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "manifest.json").write_text(json.dumps({**spec, "fingerprint": fingerprint}))
            path = directory / "policy-report.json"
            path.write_text(json.dumps(report))
            self.assertEqual(load_policy(directory)["tree"], {"action": [16, 3]})
            with self.assertRaisesRegex(ValueError, "hardware/software"):
                load_policy(directory, {"name": "A100"})
            report["status"] = "needs_more_work"
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "gate"):
                load_policy(directory)
            candidate = load_policy(directory, allow_candidate=True)
            self.assertTrue(candidate["experimental_override"])
            self.assertEqual(candidate["policy_status"], "needs_more_work")


@unittest.skipIf(cpp is None, "build C++ extension first")
class MatchedSchedulerTests(unittest.TestCase):
    def test_full_plan_drains_and_saturation_cases_reach_target_batch(self):
        from benchmark_scheduler_decode import execute, make_config
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None))
        class Adapter:
            def __init__(self):
                self.decisions = {}
                self.step_calls = []
            def __call__(self, ids, pos, slots, cu, context, blocks, maximum, decode):
                self.step_calls.append((decode, ids.numel(), context.numel(), maximum))
                return torch.zeros((context.numel(), 2))
        for preset in ("full", "longctx"):
            for case in make_plan(preset):
                requests = [{"id": i, "prompt": [1] * length, "output": case["outputs"][i],
                             "arrival": case["arrivals"][i]} for i, length in enumerate(case["lengths"])]
                result = execute(fake_torch, cpp.IterationLoop(make_config(cpp, case), torch.device("cpu")), Adapter(), requests,
                                 synchronize_steps=True)
                self.assertEqual([len(result["outputs"][i]) for i in range(len(requests))], case["outputs"])
                if case["kind"] == "saturated":
                    self.assertEqual(result["max_actual_decode_batch"], case["max_running"])
                    self.assertGreaterEqual(result["decode_batch_histogram"][str(case["max_running"])], 64)
                if preset == "longctx":
                    # The decode-phase effect is the point of this preset, and
                    # `analyze` drops any case whose decode_wall_ms is zero, so
                    # every case must produce pure-decode iterations.
                    self.assertTrue(any(step["kind"] == "decode" for step in result["steps"]),
                                    f"{case['id']} scheduled no pure-decode iteration")

    def test_metadata_page_reuse_eos_and_mixed_schedule(self):
        for eos in (-1, 9):
            config = cpp.SchedulerConfig()
            config.max_batch_size = 2
            config.max_context_length = 32
            config.max_prefill_tokens_per_iter = 3
            config.block_size = 16
            config.eos_token_id = eos
            loops = [PythonControl(config, "cpu"), cpp.IterationLoop(config, torch.device("cpu"))]
            config.overlap_prefill_build = False
            loops.append(cpp.IterationLoop(config, torch.device("cpu")))
            def run(loop):
                trace, outputs = [], {}
                # Repeat on the same allocator, with both queued and later arrivals.
                for _ in range(2):
                    loop.submit_request([4, 5, 6, 7], 4)
                    loop.submit_request([8], 2)
                    loop.submit_request([20] * 17, 2)
                    step = 0
                    def forward(*args):
                        trace.append(([t.tolist() for t in args[:6]], args[6:], loop.max_decode_context_length()))
                        source = args[0] if args[-1] else args[0].index_select(0, args[3][1:].long() - 1)
                        logits = torch.full((len(source), 64), -1.)
                        return logits.scatter_(1, ((source + 1) % 64)[:, None], 1.)
                    while loop.num_pending() or loop.num_running():
                        if step == 2:
                            loop.submit_request([11, 12], 3)
                        loop.step(forward)
                        outputs.update(loop.pop_completed())
                        step += 1
                        self.assertLess(step, 40)
                return trace, outputs
            reference = run(loops[0])
            for loop in loops[1:]:
                self.assertEqual(reference, run(loop))


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/Triton")
class AdapterCudaTests(unittest.TestCase):
    def test_prefill_fusions_replay_padded_bucket_with_matching_logits_and_kv(self):
        from model_adapter import ModelAdapter, allocate_pool
        from piecewise_prefill import PiecewisePrefill
        from naive_forward import Qwen2Config, Model

        torch.manual_seed(43)
        cfg = Qwen2Config(vocab=128, n_layers=1, d_ff=256,
                          use_custom_kernels=True)
        model = Model(cfg).to("cuda", torch.float16).eval().pack_projections_()
        cases = [
            ([5, 6, 7], [0, 1, 2], [16, 17, 18], [0, 3], [3], [[1, 0]], 3),
            ([8, 9, 10], [3, 4, 5], [19, 20, 21], [0, 3], [6], [[1, 0]], 3),
        ]
        for qkv, residual in ((True, False), (False, True), (True, True)):
            eager_pool = allocate_pool(cfg, 4, "cuda")
            graph_pool = allocate_pool(cfg, 4, "cuda")
            for pool in (eager_pool, graph_pool):
                for tensor in pool.k_pool + pool.v_pool:
                    tensor.fill_(float("nan"))
            eager = ModelAdapter(model, eager_pool, None)
            graph = PiecewisePrefill(
                model, graph_pool, max_capture_tokens=4,
                enable_packed_qkv_rope_cache=qkv,
                enable_residual_rmsnorm=residual,
            )
            with torch.no_grad():
                for raw_ids, raw_pos, raw_slots, raw_cu, raw_ctx, raw_table, max_query in cases:
                    ids = torch.tensor(raw_ids, device="cuda")
                    positions = torch.tensor(raw_pos, device="cuda")
                    slots = torch.tensor(raw_slots, device="cuda")
                    cu = torch.tensor(raw_cu, device="cuda", dtype=torch.int32)
                    context = torch.tensor(raw_ctx, device="cuda", dtype=torch.int32)
                    table = torch.tensor(raw_table, device="cuda", dtype=torch.int32)
                    expected = eager(ids, positions, slots, cu, context, table,
                                     max_query, False)
                    actual = graph.forward(ids, positions, slots, cu, context, table,
                                           max_query)
                    torch.testing.assert_close(actual, expected, atol=.05, rtol=.01)
                    for eager_cache, graph_cache in zip(
                            eager_pool.k_pool + eager_pool.v_pool,
                            graph_pool.k_pool + graph_pool.v_pool):
                        flat_a = eager_cache.view(-1, cfg.n_kv_heads, cfg.d_head)
                        flat_b = graph_cache.view(-1, cfg.n_kv_heads, cfg.d_head)
                        torch.testing.assert_close(flat_b[slots], flat_a[slots],
                                                   atol=.05, rtol=.01)
            for tensor in graph_pool.k_pool + graph_pool.v_pool:
                self.assertTrue(torch.isnan(tensor[0]).all())

    def test_piecewise_prefill_replays_new_positions_slots_and_ragged_layout(self):
        from model_adapter import ModelAdapter, allocate_pool
        from piecewise_prefill import PiecewisePrefill
        from naive_forward import Qwen2Config, Model

        torch.manual_seed(42)
        cfg = Qwen2Config(vocab=128, n_layers=1, d_ff=256, use_custom_kernels=True)
        model = Model(cfg).to("cuda", torch.float16).eval()
        eager_pool = allocate_pool(cfg, 4, "cuda")
        graph_pool = allocate_pool(cfg, 4, "cuda")
        for pool in (eager_pool, graph_pool):
            for tensor in pool.k_pool + pool.v_pool:
                tensor.fill_(float("nan"))
        eager = ModelAdapter(model, eager_pool, None)
        graph = PiecewisePrefill(model, graph_pool, max_capture_tokens=4)
        cases = [
            ([5, 6, 7], [0, 1, 2], [16, 17, 18], [0, 3], [3], [[1, 0]], 3),
            ([8, 9, 10], [3, 4, 5], [19, 20, 21], [0, 3], [6], [[1, 0]], 3),
            ([11, 12, 13], [6, 0, 1], [22, 32, 33], [0, 1, 3], [7, 2],
             [[1, 0], [2, 0]], 2),
        ]
        with torch.no_grad():
            for raw_ids, raw_pos, raw_slots, raw_cu, raw_ctx, raw_table, max_query in cases:
                ids = torch.tensor(raw_ids, device="cuda")
                positions = torch.tensor(raw_pos, device="cuda")
                slots = torch.tensor(raw_slots, device="cuda")
                cu = torch.tensor(raw_cu, device="cuda", dtype=torch.int32)
                context = torch.tensor(raw_ctx, device="cuda", dtype=torch.int32)
                table = torch.tensor(raw_table, device="cuda", dtype=torch.int32)
                expected = eager(ids, positions, slots, cu, context, table, max_query, False)
                actual = graph.forward(ids, positions, slots, cu, context, table, max_query)
                torch.testing.assert_close(actual, expected, atol=.05, rtol=.01)
                for eager_cache, graph_cache in zip(eager_pool.k_pool + eager_pool.v_pool,
                                                    graph_pool.k_pool + graph_pool.v_pool):
                    flat_a = eager_cache.view(-1, cfg.n_kv_heads, cfg.d_head)
                    flat_b = graph_cache.view(-1, cfg.n_kv_heads, cfg.d_head)
                    torch.testing.assert_close(flat_b[slots], flat_a[slots], atol=.05, rtol=.01)
        self.assertEqual(sorted(graph.shapes), [4])
        self.assertEqual((graph.captured_calls, graph.eager_calls), (3, 0))
        self.assertEqual(graph.graph_replays, 6)  # One layer: two graphs per call.
        for tensor in graph_pool.k_pool + graph_pool.v_pool:
            self.assertTrue(torch.isnan(tensor[0]).all())  # Padding never touches slot zero.

    def test_real_adapter_against_dense_attention(self):
        from benchmark_scheduler_decode import independent_check
        from naive_forward import Qwen2Config, Model
        torch.manual_seed(41)
        cfg = Qwen2Config(vocab=128, n_layers=1, d_ff=256, use_custom_kernels=True)
        model = Model(cfg).to("cuda", torch.float16).eval()
        independent_check(torch, model, "cuda")


if __name__ == "__main__":
    unittest.main()
