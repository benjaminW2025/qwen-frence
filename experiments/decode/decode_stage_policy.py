"""CPU-only design, statistics, and held-out policy fitting for decode stages.

Policy loss is mean log slowdown relative to the measured per-shape oracle.
Only training data fit tree leaves/splits; validation selects depth; test data
are used exclusively for reporting. No production dispatch is modified.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import statistics


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def auto_k(batch, pages, sms):
    return min(max(1, math.ceil(4 * sms / (12 * batch))), max(1, pages // 2))


def make_plan(preset="full", sms=132):
    if preset not in ("full", "smoke") or sms < 1:
        raise ValueError("preset must be full/smoke and SM count positive")
    smoke = preset == "smoke"
    ks = [1, 2, 4] if smoke else [1, 2, 4, 8, 16, 32, 64]
    stages = [1, 2, 3, 4]
    cases = []

    def add(suite, batch, length, shape="uniform", target_k=None, pages_per_program=None):
        if shape == "uniform":
            lengths = [length] * batch
        elif shape == "ramp":
            lengths = [max(1, length * (i + 1) // batch) for i in range(batch)]
        else:
            lengths = [length if i % 4 == 0 else 1 for i in range(batch)]
        case = {"suite": suite, "batch": batch, "context": length, "shape": shape,
                "lengths": lengths, "target_k": target_k, "target_pages_per_program": pages_per_program}
        case["id"] = f"{suite}-b{batch}-l{length}-{shape}" + (f"-k{target_k}" if target_k else "")
        cases.append(case)

    for b in ([1, 8] if smoke else [1, 8, 32, 128]):
        for k in ks:
            for pages in ([1, 2, 8] if smoke else [1, 2, 3, 4, 8, 16, 32, 64]):
                length = pages * k * 16
                if length <= 32768:
                    add("mechanism", b, length, target_k=k, pages_per_program=pages)
    axes = {
        "train": ([1, 8], [128, 512]) if smoke else ([1, 8, 32, 128], [128, 512, 2048, 8192, 16384]),
        "validation": ([4], [256]) if smoke else ([4, 16, 64], [256, 1024, 4096, 12288]),
        "test": ([2], [257]) if smoke else ([2, 12, 48, 96, 256], [257, 1537, 6145, 16383]),
    }
    for suite, (batches, lengths) in axes.items():
        for b in batches:
            for length in lengths:
                add(suite, b, length)
    for b in ([2] if smoke else [12, 48, 256]):
        for length in ([257] if smoke else [1537, 6145, 16383]):
            for shape in ("ramp", "skew"):
                add("ragged", b, length, shape)
    # Same action set across dispatch shapes, including all heuristic controls.
    candidates = sorted(set(ks) | {auto_k(c["batch"], math.ceil(c["context"] / 16), sms)
                                  for c in cases if c["suite"] != "mechanism"})
    for case in cases:
        case["actions"] = [[k, s] for k in ([case["target_k"]] if case["target_k"] else candidates)
                           for s in stages]
        case["features"] = {"batch": case["batch"], "max_pages": math.ceil(max(case["lengths"]) / 16)}
    return cases


def work_features(case, k, sms):
    counts = [math.ceil(length / 16) for length in case["lengths"]]
    iterations = [((i + 1) * pages) // k - (i * pages) // k for pages in counts for i in range(k)]
    return {"pages_per_program_min": min(iterations), "pages_per_program_max": max(iterations),
            "pages_per_program_mean": statistics.mean(iterations),
            "empty_program_fraction": iterations.count(0) / len(iterations),
            "launched_programs_per_sm": case["batch"] * 12 * k / sms,
            "active_programs": 12 * sum(min(k, p) for p in counts),
            "kv_bytes": sum(counts) * 16 * 2 * 128 * 2 * 2}


def paired_interval(ratios, seed=0, draws=2000):
    """Bootstrap whole trial pairs, not correlated event samples within a trial."""
    if not ratios or any(not math.isfinite(x) or x <= 0 for x in ratios):
        raise ValueError("paired ratios must be finite and positive")
    logs = [math.log(x) for x in ratios]
    center = math.exp(statistics.mean(logs))
    if len(logs) < 3:
        return {"ratio": center, "ci95": None, "trials": len(logs)}
    rng = random.Random(seed)
    bootstrap = sorted(math.exp(sum(rng.choices(logs, k=len(logs))) / len(logs)) for _ in range(draws))
    return {"ratio": center, "ci95": [bootstrap[int(.025 * draws)], bootstrap[int(.975 * draws)]],
            "trials": len(logs)}


def aggregate_cases(plan, records, trials, cache):
    index = {}
    for record in records:
        if record["cache"] == cache and record["role"] == "full" and record["action"] != "production":
            key = (record["case_id"], tuple(record["action"]), record["trial"])
            if key in index:
                raise ValueError(f"duplicate observation: {key}")
            index[key] = statistics.median(record["samples_ms"])
    result = []
    for case in plan:
        if case["suite"] == "mechanism":
            continue
        costs = {}
        for action in case["actions"]:
            key = tuple(action)
            observations = [index.get((case["id"], key, trial)) for trial in range(trials)]
            if any(x is None for x in observations):
                raise ValueError(f"incomplete data: {case['id']} {action} {cache}")
            costs[key] = statistics.median(observations)
        result.append({"id": case["id"], "suite": case["suite"], "features": case["features"], "costs": costs})
    return result


def leaf_for(data):
    if not data:
        raise ValueError("empty training set")
    actions = set(data[0]["costs"])
    if any(set(row["costs"]) != actions for row in data):
        raise ValueError("dispatch cases must share the same measured actions")
    def loss(action):
        return sum(math.log(row["costs"][action] / min(row["costs"].values())) for row in data)
    # Stable tie-break: fewer stages, then smaller K.
    action = min(actions, key=lambda a: (loss(a), a[1], a[0]))
    return {"action": list(action)}, loss(action)


def fit_tree(data, depth, min_leaf=3):
    node, best_loss = leaf_for(data)
    if depth == 0 or len(data) < 2 * min_leaf:
        return node
    best_split = None
    for feature in ("batch", "max_pages"):
        values = sorted({r["features"][feature] for r in data})
        for a, b in zip(values, values[1:]):
            threshold = (a + b) / 2
            left = [r for r in data if r["features"][feature] <= threshold]
            right = [r for r in data if r["features"][feature] > threshold]
            if min(len(left), len(right)) < min_leaf:
                continue
            loss = leaf_for(left)[1] + leaf_for(right)[1]
            if loss < best_loss - 1e-12:
                best_loss, best_split = loss, (feature, threshold, left, right)
    if best_split is None:
        return node
    feature, threshold, left, right = best_split
    return {"feature": feature, "threshold": threshold,
            "left": fit_tree(left, depth - 1, min_leaf), "right": fit_tree(right, depth - 1, min_leaf)}


def predict(tree, features):
    while "action" not in tree:
        tree = tree["left"] if features[tree["feature"]] <= tree["threshold"] else tree["right"]
    return tuple(tree["action"])


def evaluate(data, select):
    rows = [{"case_id": r["id"], "action": list(select(r)),
             "regret": r["costs"][select(r)] / min(r["costs"].values())} for r in data]
    if not rows:
        raise ValueError("empty evaluation set")
    regrets = sorted(row["regret"] for row in rows)
    return {"geomean_regret": math.exp(statistics.mean(math.log(x) for x in regrets)),
            "p95_regret": regrets[math.ceil(.95 * len(regrets)) - 1], "max_regret": max(regrets), "rows": rows}


def select_policy(data):
    train = [r for r in data if r["suite"] == "train"]
    validation = [r for r in data if r["suite"] == "validation"]
    candidates = []
    for depth in (0, 1, 2):
        tree = fit_tree(train, depth)
        score = evaluate(validation, lambda r: predict(tree, r["features"]))
        candidates.append({"depth": depth, "tree": tree, "validation": score})
    best_score = min(c["validation"]["geomean_regret"] for c in candidates)
    # Prefer a simpler tree if its validation loss is within 1% of the best.
    selected = next(c for c in candidates if c["validation"]["geomean_regret"] <= best_score * 1.01)
    return {"selected": selected, "candidates": candidates, "fixed_baseline": candidates[0]["tree"]}
