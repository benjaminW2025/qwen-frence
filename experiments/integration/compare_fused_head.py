"""Compare completed accepted/fused-head runs; do not promote from timing alone."""
import hashlib
import json
import math
from pathlib import Path

from fixed_regime import FACTORIAL_SHAPES

SHAPES = tuple(row["id"] for row in FACTORIAL_SHAPES)
CONTRACT_KEYS = ("model", "suite_dir", "seed", "warmups", "repetitions",
                 "vllm_version", "vllm_budget", "local_graph_pool", "budgets")


def load(path):
    return json.loads(path.read_text())


def validate_baseline(directory, candidate_config):
    baseline = load(Path(directory) / "summary.json")
    config = baseline["configuration"]
    if baseline.get("status") != "complete" or config.get("fused_greedy_output", False):
        raise ValueError("baseline must be a completed accepted-engine run without fused head")
    if any(config.get(key) != candidate_config.get(key) for key in CONTRACT_KEYS):
        raise ValueError("baseline/candidate model, budgets, workload or repetitions differ")
    if len(baseline["rows"]) != len(SHAPES) or {row["shape_id"] for row in baseline["rows"]} != set(SHAPES):
        raise ValueError("baseline must contain all eight cells")
    return baseline


def reference_hashes(root, shape):
    files = sorted((root / shape / "vllm").glob("*.json"))
    if len(files) != 1:
        raise ValueError(f"{shape}: exactly one burst reference required")
    return [hashlib.sha256(path.read_bytes()).hexdigest() for path in (
        files[0], root / "mixed" / shape / "vllm.json", root / "phases" / shape / "vllm.json")]


def compare(baseline_dir, candidate_dir, minimum_gain=.01):
    baseline_dir, candidate_dir = Path(baseline_dir), Path(candidate_dir)
    candidate = load(candidate_dir / "summary.json")
    if candidate.get("status") != "complete" or not candidate["configuration"].get("fused_greedy_output"):
        raise ValueError("candidate must be a completed fused-head run")
    baseline = validate_baseline(baseline_dir, candidate["configuration"])
    if len(candidate["rows"]) != len(SHAPES) or {row["shape_id"] for row in candidate["rows"]} != set(SHAPES):
        raise ValueError("candidate must contain all eight cells")
    base_rows = {row["shape_id"]: row for row in baseline["rows"]}
    rows, exact = [], True
    for row in candidate["rows"]:
        shape = row["shape_id"]
        if reference_hashes(baseline_dir, shape) != reference_hashes(candidate_dir, shape):
            raise ValueError(f"{shape}: A/B did not use the same saved vLLM reference")
        record = {"shape_id": shape}
        for kind, prefix in (("burst", Path()), ("mixed", Path("mixed"))):
            left = load(baseline_dir / prefix / shape / "local.json")
            right = load(candidate_dir / prefix / shape / "local.json")
            expected = dict(left["engine_flags"], fused_greedy_output=True)
            if left["engine_flags"].get("fused_greedy_output") or right["engine_flags"] != expected:
                raise ValueError(f"{shape}/{kind}: A/B changed more than the fused head")
            count = candidate["configuration"]["repetitions"]
            if len(left["runs"]) != count or len(right["runs"]) != count:
                raise ValueError(f"{shape}/{kind}: repetitions missing")
            matches = all(a["outputs"] == b["outputs"] for a, b in zip(left["runs"], right["runs"]))
            record[f"{kind}_tokens_exact"] = matches
            exact = exact and matches
            base_rate = base_rows[shape][kind]["local_output_tokens_per_s"]
            head_rate = row[kind]["local_output_tokens_per_s"]
            if any(not math.isfinite(rate) or rate <= 0 for rate in (base_rate, head_rate)):
                raise ValueError(f"{shape}/{kind}: invalid throughput")
            record[f"{kind}_speedup"] = head_rate / base_rate
        left_phase = load(baseline_dir / "phases" / shape / "local.json")
        right_phase = load(candidate_dir / "phases" / shape / "local.json")
        if right_phase["engine_flags"] != dict(left_phase["engine_flags"], fused_greedy_output=True):
            raise ValueError(f"{shape}/phases: A/B changed more than the fused head")
        phase_exact = all(left_phase[key] == right_phase[key]
                          for key in ("burst_outputs_sha256", "mixed_outputs_sha256"))
        record["phase_tokens_exact"] = phase_exact
        exact = exact and phase_exact
        record["phase_speedups"] = {}
        for kind in ("prefill", "decode", "mixed"):
            times = [item["phases"]["phases"][kind]["local"]["median_step_wall_ms"]
                     for item in (base_rows[shape], row)]
            if any(not math.isfinite(value) or value <= 0 for value in times):
                raise ValueError(f"{shape}/{kind}: invalid phase timing")
            record["phase_speedups"][kind] = times[0] / times[1]
        rows.append(record)
    rates = {kind: math.exp(sum(math.log(row[f"{kind}_speedup"]) for row in rows) / len(rows))
             for kind in ("burst", "mixed")}
    selected = "fused-head" if exact and all(rate > 1 + minimum_gain for rate in rates.values()) else "accepted"
    return {"status": "complete", "selected": selected, "exact_baseline_tokens": exact,
            "geomean_burst_speedup": rates["burst"], "geomean_mixed_speedup": rates["mixed"],
            "rows": rows, "selected_runtime_flags": {"fused_greedy_output": selected == "fused-head"},
            "policy": "Exact baseline tokens in burst/mixed/phases; >1% geomean gain in both burst and mixed.",
            "note": "Three-run medians are not a statistical confidence interval. Per-cell regressions remain visible."}
