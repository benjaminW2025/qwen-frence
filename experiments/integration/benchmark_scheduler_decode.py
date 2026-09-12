#!/usr/bin/env python3
"""Real-model matched scheduler × frozen decode-policy experiment. See README.md."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (HERE, ROOT / "baseline", ROOT / "engine/model_runner", ROOT / "engine/kvcache",
             ROOT / "engine/cpp/build", ROOT / "experiments/decode"):
    sys.path.insert(0, str(path))
from design import VARIANTS, load_policy, make_plan, make_requests, stable_hash, summarize, validate_checkpoint, variant_order


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_hashes():
    paths = []
    for directory in ("baseline", "engine/cpp/src", "engine/cpp/include", "engine/model_runner",
                      "engine/kvcache", "custom_kernels", "experiments/integration", "experiments/decode"):
        paths.extend(p for p in (ROOT / directory).rglob("*") if p.suffix in (".py", ".cpp", ".hpp"))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def hardware(torch, triton):
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    from benchmark_decode_stage_policy import driver_version
    return {"name": props.name, "sms": props.multi_processor_count,
            "capability": list(torch.cuda.get_device_capability()), "memory": props.total_memory,
            "uuid": str(getattr(props, "uuid", "unknown")), "torch": torch.__version__,
            "triton": triton.__version__, "cuda": torch.version.cuda, "driver": driver_version(),
            "python": platform.python_version()}


def make_config(cpp, case):
    config = cpp.SchedulerConfig()
    config.max_batch_size = case["max_running"]
    config.max_prefill_tokens_per_iter = case["prefill_budget"]
    config.max_context_length = max(n + output for n, output in zip(case["lengths"], case["outputs"]))
    config.block_size, config.num_kv_heads, config.head_dim, config.eos_token_id = 16, 2, 128, -1
    return config


class TraceCheck:
    """Compare all metadata; full logits for the first occurrence of each shape."""
    def __init__(self, torch, reference=None):
        self.torch, self.reference = torch, reference
        self.rows, self.seen, self.cursor = [], set(), 0
        self.max_logit_error = 0.0

    def __call__(self, args, logits):
        digest = hashlib.sha256()
        for tensor in args[:6]:
            cpu = tensor.detach().cpu().contiguous()
            digest.update(str((tuple(cpu.shape), str(cpu.dtype))).encode())
            digest.update(cpu.numpy().tobytes())
        key = (bool(args[-1]), args[0].numel(), args[4].numel(), int(args[-2]))
        row = {"shape": key, "metadata": digest.hexdigest()}
        if self.reference is None:
            row["logits"] = logits.detach().cpu() if key not in self.seen else None
            self.rows.append(row)
            self.seen.add(key)
        else:
            if self.cursor >= len(self.reference):
                raise AssertionError("extra callback in candidate")
            expected = self.reference[self.cursor]
            if any(row[k] != expected[k] for k in ("shape", "metadata")):
                raise AssertionError("scheduler plans or metadata differ")
            if expected["logits"] is not None:
                actual = logits.detach().cpu()
                self.torch.testing.assert_close(actual, expected["logits"], atol=.05, rtol=.01)
                self.max_logit_error = max(self.max_logit_error, float((actual.float() - expected["logits"].float()).abs().max()))
        self.cursor += 1

    def finish(self):
        if self.reference is not None and self.cursor != len(self.reference):
            raise AssertionError("missing callback in candidate")


def execute(torch, loop, adapter, requests):
    """One fully drained workload. Arrivals use iteration indices, not wall time."""
    pending = sorted(requests, key=lambda r: (r["arrival"], r["id"]))
    cursor, iteration = 0, 0
    mapping, outputs, steps = {}, {}, []
    iteration_limit = sum(len(r["prompt"]) + r["output"] for r in requests) + max(r["arrival"] for r in requests) + 1
    adapter.decisions.clear()
    torch.cuda.synchronize()
    start = time.perf_counter()
    while cursor < len(pending) or loop.num_pending() or loop.num_running():
        if not loop.num_pending() and not loop.num_running() and cursor < len(pending):
            iteration = max(iteration, pending[cursor]["arrival"])
        while cursor < len(pending) and pending[cursor]["arrival"] <= iteration:
            request = pending[cursor]
            mapping[loop.submit_request(request["prompt"], request["output"])] = request["id"]
            cursor += 1
        adapter.step_calls = []
        step_start = time.perf_counter()
        completed = loop.step(adapter)
        elapsed = (time.perf_counter() - step_start) * 1000
        if not adapter.step_calls:
            raise RuntimeError("scheduler made no progress")
        kinds = {call[0] for call in adapter.step_calls}
        steps.append({"kind": "mixed" if len(kinds) == 2 else "decode" if True in kinds else "prefill",
                      "wall_ms": elapsed, "calls": list(adapter.step_calls), "completed": completed})
        for request_id, tokens in loop.pop_completed():
            outputs[mapping[request_id]] = tokens
        iteration += 1
        if iteration > iteration_limit:
            raise RuntimeError("iteration limit exceeded")
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - start) * 1000
    decode_batches = [call[1] for step in steps for call in step["calls"] if call[0]]
    return {"decode_batch_histogram": {str(b): decode_batches.count(b) for b in sorted(set(decode_batches))},
            "max_actual_decode_batch": max(decode_batches, default=0), "wall_ms": wall_ms, "output_tokens_per_s": sum(map(len, outputs.values())) * 1000 / wall_ms,
            "steps": steps, "decisions": dict(adapter.decisions), "outputs": outputs,
            **{f"{kind}_wall_ms": sum(step["wall_ms"] for step in steps if step["kind"] == kind)
               for kind in ("decode", "prefill", "mixed")}}


def independent_check(torch, model, device):
    """Chunked prefill and decode against the existing dense SDPA model path."""
    from kv_cache import KVCache
    from model_adapter import ModelAdapter, allocate_pool
    from types import SimpleNamespace
    pool = allocate_pool(model.cfg, 2, device)
    for tensor in pool.k_pool + pool.v_pool:
        tensor.fill_(float("nan"))
    loop = SimpleNamespace(max_decode_context_length=lambda: 5)
    adapter = ModelAdapter(model, pool, loop)
    prompt = [5, 6, 7, 8, 9]
    table = torch.tensor([[1, 0]], device=device, dtype=torch.int32)
    previous = 0
    with torch.no_grad():
        for end, decode in ((2, False), (4, False), (5, True)):
            ids = torch.tensor(prompt[previous:end], device=device)
            positions = torch.arange(previous, end, device=device)
            cu = torch.tensor([] if decode else [0, end - previous], device=device, dtype=torch.int32)
            actual = adapter(ids, positions, positions + 16, cu,
                             torch.tensor([end], device=device, dtype=torch.int32), table, end - previous, decode)
            cache = KVCache(model.cfg, 1, device, torch.float16, max_seq_len=end)
            custom = model.cfg.use_custom_kernels
            try:
                model.cfg.use_custom_kernels = False
                expected = model(torch.tensor([prompt[:end]], device=device), cache)[:, -1]
            finally:
                model.cfg.use_custom_kernels = custom
            torch.testing.assert_close(actual, expected, atol=.05, rtol=.01)
            previous = end
    torch.cuda.synchronize()


def run(args):
    import torch
    import triton
    import inference_engine_cpp as cpp
    from naive_forward import Qwen2Config
    from weight_loader import QwenWeightLoader
    from model_adapter import ModelAdapter, allocate_pool
    from python_control import PythonControl

    if not hasattr(cpp.IterationLoop, "max_decode_context_length"):
        raise RuntimeError("rebuild extension with make cpp-scheduler-build")
    extension = Path(cpp.__file__)
    sources = list((ROOT / "engine/cpp/src").glob("*.cpp")) + list((ROOT / "engine/cpp/include").glob("*.hpp"))
    if any(p.stat().st_mtime_ns > extension.stat().st_mtime_ns for p in sources):
        raise RuntimeError("C++ extension is older than source; rebuild it")
    torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    hw = hardware(torch, triton)
    policy = load_policy(args.policy_dir, hw) if args.phase == "combined" else None
    plan = make_plan(args.preset)
    selected = [c for c in plan if args.case_id is None or c["id"] == args.case_id]
    if not selected:
        raise ValueError("unknown case ID")
    cfg = Qwen2Config(use_custom_kernels=True)
    spec = {"schema_version": 1, "phase": args.phase, "preset": args.preset, "plan": plan,
            "trials": args.trials, "samples": args.samples, "warmups": args.warmups, "seed": args.seed,
            "hardware": hw, "model": args.model, "model_config": asdict(cfg), "dtype": "float16",
            "sources": source_hashes(), "extension_sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
            "policy": policy, "execution": "eager model; separate decode/prefill forwards; EOS disabled",
            "control": "matched Python reference, not the production Python scheduler"}
    # Resolve immutable model identity before accepting or creating checkpoints.
    from transformers import AutoConfig
    hf_config = AutoConfig.from_pretrained(args.model, revision=args.revision)
    for hf_name, our_name in (("vocab_size", "vocab"), ("hidden_size", "d_model"),
                              ("intermediate_size", "d_ff"), ("num_hidden_layers", "n_layers"),
                              ("num_attention_heads", "n_heads"), ("num_key_value_heads", "n_kv_heads"),
                              ("rope_theta", "rope_theta"), ("rms_norm_eps", "rms_norm_eps"),
                              ("tie_word_embeddings", "tie_embeddings")):
        if getattr(hf_config, hf_name, None) != getattr(cfg, our_name):
            raise ValueError(f"model is outside the fixed Qwen2.5-1.5B scope: {hf_name}")
    revision = getattr(hf_config, "_commit_hash", None)
    if not revision:
        raise ValueError("use a Hub model with an immutable resolved revision")
    spec["model_revision"] = revision
    identity = stable_hash(spec)
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text())["fingerprint"] != identity:
        raise ValueError("resume identity changed; use a new output directory")
    manifest = {**spec, "fingerprint": identity}
    atomic_json(manifest_path, manifest)
    from transformers import AutoModelForCausalLM
    hf = AutoModelForCausalLM.from_pretrained(args.model, revision=revision, dtype=torch.float16, attn_implementation="sdpa")
    model = QwenWeightLoader(cfg).convert(hf, args.device, torch.float16)
    del hf
    independent_check(torch, model, args.device)
    names = list(VARIANTS) if policy else list(VARIANTS)[:2]
    for case in selected:
        pending_trials = []
        for trial in range(args.trials):
            checkpoint = args.output_dir / "trials" / f"{case['id']}-t{trial}.json"
            if checkpoint.exists():
                validate_checkpoint(json.loads(checkpoint.read_text()), manifest, case["id"], trial)
            else:
                pending_trials.append(trial)
        if not pending_trials:
            continue
        # Release unused allocator reservations before the physical-free-memory
        # check. This is outside timing and does not evict live model weights.
        torch.cuda.empty_cache()
        config = make_config(cpp, case)
        blocks = config.max_batch_size * ((config.max_context_length + 15) // 16)
        pool_bytes = blocks * 16 * 2 * 128 * 2 * cfg.n_layers * 2
        if pool_bytes > torch.cuda.mem_get_info()[0] * .70:
            raise RuntimeError(f"{case['id']}: KV pool needs {pool_bytes / 2**30:.2f} GiB; insufficient workspace headroom")
        pool = allocate_pool(cfg, blocks, args.device)
        for trial in pending_trials:
            checkpoint = args.output_dir / "trials" / f"{case['id']}-t{trial}.json"
            from benchmark_decode_stage_policy import telemetry
            telemetry_before = telemetry()
            trial_started = time.perf_counter()
            requests = make_requests(case, args.seed + trial * 100003 + int(stable_hash(case)[:8], 16), cfg.vocab)
            reference = None
            expected = None
            expected_steps = None
            checks = {}
            runners = {}
            for name in names:
                scheduler, decode = VARIANTS[name]
                loop = cpp.IterationLoop(config, torch.device(args.device)) if scheduler == "cpp" else PythonControl(config, args.device)
                adapter = ModelAdapter(model, pool, loop, policy if decode == "selected" else None)
                runners[name] = (loop, adapter)
                for tensor in pool.k_pool + pool.v_pool:
                    tensor.fill_(float("nan"))
                del tensor
                checker = TraceCheck(torch, reference)
                adapter.observer = checker
                validation = execute(torch, loop, adapter, requests)
                checker.finish()
                adapter.observer = None
                signature = [(r["kind"], r["calls"], r["completed"]) for r in validation["steps"]]
                if case["kind"] == "saturated" and validation["max_actual_decode_batch"] != case["max_running"]:
                    raise AssertionError("saturation case never reached its declared decode batch")
                if reference is None:
                    reference, expected, expected_steps = checker.rows, validation["outputs"], signature
                elif validation["outputs"] != expected or signature != expected_steps:
                    raise AssertionError(f"{name}: generated tokens or scheduled work differ; no timing accepted")
                checks[name] = {"callbacks": checker.cursor, "max_logit_error": checker.max_logit_error,
                                "outputs_equal": True, "metadata_equal": True,
                                "decisions": validation["decisions"]}
            del reference, checker
            for _ in range(args.warmups):
                for name in variant_order(args.phase, args.seed + trial):
                    execute(torch, *runners[name], requests)
            torch.cuda.reset_peak_memory_stats()
            records, order = [], []
            for sample in range(args.samples):
                for name in variant_order(args.phase, args.seed + trial * 1009 + sample):
                    order.append(name)
                    measured = execute(torch, *runners[name], requests)
                    if measured.pop("outputs") != expected:
                        raise AssertionError("timed generation changed after preflight")
                    signature = [(r["kind"], r["calls"], r["completed"]) for r in measured["steps"]]
                    if signature != expected_steps:
                        raise AssertionError("timed scheduling differs from preflight")
                    records.append({"case_id": case["id"], "trial": trial, "sample": sample,
                                    "variant": name, **measured})
            atomic_json(checkpoint, {"status": "complete", "fingerprint": identity, "case_id": case["id"],
                                    "trial": trial, "checks": checks, "order": order, "records": records,
                                    "requests_sha256": stable_hash(requests),
                                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                    "telemetry_before": telemetry_before, "telemetry_after": telemetry(),
                                    "trial_wall_seconds": time.perf_counter() - trial_started})
            print(f"complete: {case['id']} trial {trial + 1}/{args.trials}", flush=True)
            del runners, adapter, loop
        del pool


def analyze(args):
    manifest = json.loads((args.output_dir / "manifest.json").read_text())
    # Invalidate an older successful report before inspecting new/corrupt data.
    atomic_json(args.output_dir / "report.json", {"status": "validating", "production_ready": False,
                                                "fingerprint": manifest["fingerprint"]})
    records = []
    for path in sorted((args.output_dir / "trials").glob("*.json")):
        trial = json.loads(path.read_text())
        records.extend(validate_checkpoint(trial, manifest))
    names = list(VARIANTS) if manifest["phase"] == "combined" else list(VARIANTS)[:2]
    planned = {c["id"] for c in manifest["plan"]}
    observed = {r["case_id"] for r in records}
    if not observed <= planned:
        raise ValueError("unplanned case in results")
    complete_cases = set()
    pending = {}
    for case in planned:
        wanted = {(t, sample, name) for t in range(manifest["trials"])
                  for sample in range(manifest["samples"]) for name in names}
        seen = [(r["trial"], r["sample"], r["variant"]) for r in records if r["case_id"] == case]
        if len(seen) != len(set(seen)) or not set(seen) <= wanted:
            raise ValueError("duplicate or unplanned observation")
        if set(seen) == wanted:
            complete_cases.add(case)
        else:
            pending[case] = len(wanted - set(seen))
    complete = [r for r in records if r["case_id"] in complete_cases]
    phase_effects = {}
    for kind in ("decode", "prefill", "mixed"):
        metric = f"{kind}_wall_ms"
        available = {case for case in complete_cases
                     if all(r.get(metric, 0) > 0 for r in complete if r["case_id"] == case)}
        phase_effects[kind] = summarize([r for r in complete if r["case_id"] in available],
            manifest["phase"], manifest["trials"], manifest["samples"], manifest["seed"], metric=metric)
    result = {"status": "complete" if complete_cases == planned else "partial",
              "production_ready": False, "fingerprint": manifest["fingerprint"],
              "pending_observations": pending,
              "cases": summarize(complete, manifest["phase"], manifest["trials"], manifest["samples"], manifest["seed"]),
              "phase_effects": phase_effects,
              "scope": "matched implementation comparison; eager real model; step-index arrivals; no serving TTFT claim"}
    atomic_json(args.output_dir / "report.json", result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run", "analyze"))
    parser.add_argument("--phase", choices=("scheduler", "combined"), default="scheduler")
    parser.add_argument("--preset", choices=("smoke", "full"), default="full")
    parser.add_argument("--policy-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/results/scheduler-decode")
    parser.add_argument("--case-id")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()
    if args.trials < 1 or args.samples < 1 or args.warmups < 1:
        parser.error("trials, samples, and warmups must be positive")
    if args.command == "analyze":
        analyze(args)
    elif args.command == "plan":
        cases = make_plan(args.preset)
        print(json.dumps({"cases": cases, "variants": variant_order(args.phase, 0),
                          "timed_workloads": len(cases) * args.trials * args.samples * (4 if args.phase == "combined" else 2),
                          "note": "preflight and warmup workloads are additional"}, indent=2))
    else:
        if args.phase == "combined" and args.policy_dir is None:
            parser.error("combined phase requires --policy-dir from the completed decode sweep")
        run(args)


if __name__ == "__main__":
    main()
