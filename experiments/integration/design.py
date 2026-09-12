"""CPU-only design, frozen-policy validation, and paired factorial analysis."""
from __future__ import annotations

import json
import math
from pathlib import Path
import random
import statistics
import sys

DECODE = Path(__file__).resolve().parents[1] / "decode"
sys.path.insert(0, str(DECODE))
from decode_stage_policy import paired_interval, predict, stable_hash

VARIANTS = {
    "python-production": ("python", "production"),
    "cpp-production": ("cpp", "production"),
    "python-selected": ("python", "selected"),
    "cpp-selected": ("cpp", "selected"),
}


def make_plan(preset="full"):
    if preset not in ("smoke", "full"):
        raise ValueError("unknown preset")
    shapes = [(2, 257), (4, 513)] if preset == "smoke" else [
        (b, length) for b in (1, 8, 32, 128) for length in (512, 8192)
    ]
    output = 4 if preset == "smoke" else 64
    cases = []
    for batch, length in shapes:
        cases.append({"id": f"uniform-b{batch}-l{length}", "kind": "uniform",
                      "lengths": [length] * batch, "arrivals": [0] * batch,
                      "outputs": [output] * batch, "max_running": batch,
                      "prefill_budget": 128 if preset == "smoke" else 2048})
    if preset == "full":
        # With a 2048-token prefill budget, long prompts admit slowly: 64
        # generated tokens only sustain ~16 concurrent decode requests. Add
        # explicit saturation cases that keep early arrivals alive long enough
        # to reach the declared batch and then decode together for >=64 steps.
        for batch in (32, 128):
            length, budget = 8192, 2048
            limit = math.ceil(batch * length / budget) + output
            cases.append({"id": f"saturated-b{batch}-l{length}", "kind": "saturated",
                          "lengths": [length] * batch, "arrivals": [0] * batch,
                          "outputs": [limit] * batch, "max_running": batch,
                          "prefill_budget": budget})
    batch, length = (4, 769) if preset == "smoke" else (32, 6145)
    for kind in ("ragged", "staggered"):
        lengths = [max(1, length // (1 + i % 4)) for i in range(batch)]
        cases.append({"id": f"{kind}-b{batch}-l{length}", "kind": kind,
                      "lengths": lengths,
                      "arrivals": [0 if kind == "ragged" else (i % 4) * 2 for i in range(batch)],
                      "outputs": [output + i % 3 for i in range(batch)],
                      "max_running": batch // 2, "prefill_budget": 128 if preset == "smoke" else 2048})
    return cases


def make_requests(case, seed, vocab):
    rng = random.Random(seed)
    return [{"id": i, "prompt": [rng.randrange(vocab) for _ in range(n)],
             "output": case["outputs"][i], "arrival": case["arrivals"][i]}
            for i, n in enumerate(case["lengths"])]


def variant_order(phase, seed):
    variants = list(VARIANTS) if phase == "combined" else list(VARIANTS)[:2]
    random.Random(seed).shuffle(variants)
    return variants


def validate_tree(tree):
    if not isinstance(tree, dict):
        raise ValueError("invalid policy tree")
    if "action" in tree:
        action = tree["action"]
        if (not isinstance(action, list) or len(action) != 2
                or any(type(x) is not int for x in action)
                or not 1 <= action[0] <= 4096 or action[1] not in (1, 2, 3, 4)):
            raise ValueError("invalid K/stages action")
        return
    if (tree.get("feature") not in ("batch", "max_pages")
            or not isinstance(tree.get("threshold"), (int, float))
            or not math.isfinite(tree["threshold"])):
        raise ValueError("invalid policy split")
    validate_tree(tree.get("left"))
    validate_tree(tree.get("right"))


def load_policy(directory, hardware=None):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    report = json.loads((directory / "policy-report.json").read_text())
    if report.get("status") != "microbenchmark_candidate":
        raise ValueError("decode policy has not passed the declared microbenchmark gate")
    if report.get("fingerprint") != manifest.get("fingerprint"):
        raise ValueError("policy/report manifest mismatch")
    spec = {k: v for k, v in manifest.items() if k not in ("fingerprint", "created_at")}
    if stable_hash(spec) != manifest["fingerprint"]:
        raise ValueError("decode manifest fingerprint is invalid")
    required = {"paged_decode_grouped_splitk_pipelined.py", "../../engine/kvcache/paged_decode_attention.py"}
    if not required <= set(manifest["source_hashes"]):
        raise ValueError("decode sweep did not fingerprint both candidate and production kernels")
    for name, digest in manifest["source_hashes"].items():
        if stable_hash((DECODE / name).read_text()) != digest:
            raise ValueError(f"decode source changed since sweep: {name}; use a new sweep directory")
    if hardware is not None:
        for key in ("name", "sms", "capability", "memory", "uuid", "torch", "triton", "cuda", "driver", "python"):
            if hardware.get(key) != manifest["hardware"].get(key):
                raise ValueError(f"decode policy hardware/software mismatch: {key}")
    tree = report["policy"]["selected"]["tree"]
    validate_tree(tree)
    cases = [c for c in manifest["plan"] if c["suite"] != "mechanism"]
    return {"tree": tree, "report_hash": stable_hash(report),
            "sweep_fingerprint": manifest["fingerprint"],
            "batch_range": [min(c["batch"] for c in cases), max(c["batch"] for c in cases)],
            "pages_range": [min(c["features"]["max_pages"] for c in cases),
                            max(c["features"]["max_pages"] for c in cases)],
            "heads_per_program": 1, "num_warps": 4}


def select_action(policy, batch, context):
    pages = (context + 15) // 16
    if (not policy["batch_range"][0] <= batch <= policy["batch_range"][1]
            or not policy["pages_range"][0] <= pages <= policy["pages_range"][1]):
        return None  # Explicit production fallback; never extrapolate silently.
    return predict(policy["tree"], {"batch": batch, "max_pages": pages})


def validate_checkpoint(payload, manifest, case_id=None, trial=None):
    """A complete trial must contain every paired arm/sample, not just a flag."""
    if payload.get("status") != "complete" or payload.get("fingerprint") != manifest["fingerprint"]:
        raise ValueError("invalid checkpoint status/fingerprint")
    case_id = payload.get("case_id") if case_id is None else case_id
    trial = payload.get("trial") if trial is None else trial
    if (payload.get("case_id") != case_id or payload.get("trial") != trial
            or case_id not in {c["id"] for c in manifest["plan"]}
            or type(trial) is not int or not 0 <= trial < manifest["trials"]):
        raise ValueError("checkpoint case/trial mismatch")
    names = list(VARIANTS) if manifest["phase"] == "combined" else list(VARIANTS)[:2]
    wanted = {(sample, name) for sample in range(manifest["samples"]) for name in names}
    seen = set()
    for record in payload.get("records", []):
        key = (record.get("sample"), record.get("variant"))
        if (record.get("case_id") != case_id or record.get("trial") != trial
                or type(record.get("sample")) is not int or key not in wanted or key in seen):
            raise ValueError("duplicate/unplanned checkpoint observation")
        for metric in ("wall_ms", "decode_wall_ms", "prefill_wall_ms", "mixed_wall_ms"):
            value = record.get(metric)
            if (not isinstance(value, (int, float)) or not math.isfinite(value)
                    or value < 0 or (metric == "wall_ms" and value == 0)):
                raise ValueError(f"invalid checkpoint timing: {metric}")
        seen.add(key)
    if seen != wanted:
        raise ValueError("incomplete checkpoint observations")
    if set(payload.get("checks", {})) != set(names):
        raise ValueError("missing checkpoint correctness checks")
    for check in payload["checks"].values():
        if check.get("outputs_equal") is not True or check.get("metadata_equal") is not True:
            raise ValueError("checkpoint correctness checks did not pass")
    return payload["records"]


def summarize(records, phase, expected_trials, expected_samples, seed=0, metric="wall_ms"):
    names = list(VARIANTS) if phase == "combined" else list(VARIANTS)[:2]
    index = {}
    for record in records:
        key = (record["case_id"], record["trial"], record["sample"], record["variant"])
        if key in index:
            raise ValueError(f"duplicate measurement: {key}")
        if (record["variant"] not in names or not 0 <= record["trial"] < expected_trials
                or not 0 <= record["sample"] < expected_samples
                or not math.isfinite(record[metric]) or record[metric] <= 0):
            raise ValueError("invalid measurement")
        index[key] = record
    result = []
    for case in sorted({r["case_id"] for r in records}):
        medians = {name: [] for name in names}
        for trial in range(expected_trials):
            for name in names:
                keys = [(case, trial, sample, name) for sample in range(expected_samples)]
                if any(key not in index for key in keys):
                    raise ValueError(f"incomplete paired measurements: {case}")
                medians[name].append(statistics.median(index[key][metric] for key in keys))
        comparisons = {"scheduler": ("python-production", "cpp-production")}
        if phase == "combined":
            comparisons.update({"decode_on_python": ("python-production", "python-selected"),
                                "decode_on_cpp": ("cpp-production", "cpp-selected"),
                                "scheduler_with_decode": ("python-selected", "cpp-selected"),
                                "combined": ("python-production", "cpp-selected")})
        effects = {label: paired_interval([a / b for a, b in zip(medians[left], medians[right])], seed=seed)
                   for label, (left, right) in comparisons.items()}
        if phase == "combined":
            # >1: combined gain exceeds the product of the isolated gains.
            effects["interaction"] = paired_interval([
                medians["cpp-production"][t] * medians["python-selected"][t]
                / (medians["python-production"][t] * medians["cpp-selected"][t])
                for t in range(expected_trials)], seed=seed)
        result.append({"case_id": case, "trial_medians_ms": medians, "effects": effects})
    return result
