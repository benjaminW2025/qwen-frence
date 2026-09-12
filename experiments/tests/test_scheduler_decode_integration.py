"""Matched scheduling, factorial accounting, and frozen-policy gate tests."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "experiments/integration", ROOT / "engine/cpp/build", ROOT / "baseline",
             ROOT / "engine/model_runner", ROOT / "engine/kvcache"):
    sys.path.insert(0, str(path))
from design import (load_policy, make_plan, make_requests, select_action, stable_hash,
                    summarize, validate_checkpoint, validate_tree, variant_order)
import torch
from python_control import PythonControl
try:
    import inference_engine_cpp as cpp
except ImportError:
    cpp = None


class DesignTests(unittest.TestCase):
    def test_deterministic_workloads_and_paired_orders(self):
        for preset in ("smoke", "full"):
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
        for case in make_plan("full"):
            requests = [{"id": i, "prompt": [1] * length, "output": case["outputs"][i],
                         "arrival": case["arrivals"][i]} for i, length in enumerate(case["lengths"])]
            result = execute(fake_torch, cpp.IterationLoop(make_config(cpp, case), torch.device("cpu")), Adapter(), requests)
            self.assertEqual([len(result["outputs"][i]) for i in range(len(requests))], case["outputs"])
            if case["kind"] == "saturated":
                self.assertEqual(result["max_actual_decode_batch"], case["max_running"])
                self.assertGreaterEqual(result["decode_batch_histogram"][str(case["max_running"])], 64)

    def test_metadata_page_reuse_eos_and_mixed_schedule(self):
        for eos in (-1, 9):
            config = cpp.SchedulerConfig()
            config.max_batch_size = 2
            config.max_context_length = 32
            config.max_prefill_tokens_per_iter = 3
            config.block_size = 16
            config.eos_token_id = eos
            loops = [PythonControl(config, "cpu"), cpp.IterationLoop(config, torch.device("cpu"))]
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
            self.assertEqual(run(loops[0]), run(loops[1]))


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/Triton")
class AdapterCudaTests(unittest.TestCase):
    def test_real_adapter_against_dense_attention(self):
        from benchmark_scheduler_decode import independent_check
        from naive_forward import Qwen2Config, Model
        torch.manual_seed(41)
        cfg = Qwen2Config(vocab=128, n_layers=1, d_ff=256, use_custom_kernels=True)
        model = Model(cfg).to("cuda", torch.float16).eval()
        independent_check(torch, model, "cuda")


if __name__ == "__main__":
    unittest.main()
