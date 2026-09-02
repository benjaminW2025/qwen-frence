#!/usr/bin/env python3
"""Fit tiny latency-aware dispatch trees from isolated H100 sweep results."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics


EXPERIMENTS = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Shape:
    component: str
    key: tuple[int, ...]
    features: dict[str, int]
    latencies: dict[str, float]
    baseline_action: str


COMPONENT_FEATURES = {
    "decode": ("batch_size", "context_length", "total_kv_tokens"),
    "swiglu": ("rows",),
    "rope-kv": ("tokens",),
    "paged-prefill": (
        "batch_size", "query_length", "prefix_length", "total_query_tokens",
        "context_length", "attention_pairs",
    ),
}

DEFAULT_ACTIONS = {
    "decode": {
        "production-default",
        "candidate-h1-w4-s2",
        "candidate-h1-w8-s2",
    },
}

EVALUATION_FIELDS = (
    "component", "split", "shape", "features", "baseline_action",
    "selected_action", "oracle_action", "baseline_ms", "selected_ms",
    "oracle_ms", "speedup_vs_baseline", "oracle_regret",
    "oracle_margin", "robust_winner",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decode-json", type=Path)
    parser.add_argument("--swiglu-json", type=Path)
    parser.add_argument("--rope-kv-json", type=Path)
    parser.add_argument("--paged-prefill-json", type=Path)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--min-leaf-shapes", type=int, default=2)
    parser.add_argument("--min-split-improvement", type=float, default=0.003)
    parser.add_argument(
        "--max-training-regression", type=float, default=0.02,
        help="reject a leaf action if it regresses any training shape beyond this fraction",
    )
    parser.add_argument("--holdout-modulus", type=int, default=5)
    parser.add_argument("--winner-margin", type=float, default=1.02)
    parser.add_argument(
        "--output-dir", type=Path,
        default=EXPERIMENTS / "results" / "dispatch-policy",
    )
    return parser


def _load(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def _set_latency(target, action, latency):
    latency = float(latency)
    previous = target.get(action)
    if previous is None or latency < previous:
        target[action] = latency


def load_decode_shapes(path: Path) -> list[Shape]:
    grouped = {}
    for row in _load(path)["rows"]:
        if row.get("status") != "ok" or row.get("median_ms") is None:
            continue
        action = str(row["variant"])
        if action not in DEFAULT_ACTIONS["decode"]:
            continue
        key = (int(row["batch_size"]), int(row["context_length"]))
        grouped.setdefault(key, {})[action] = float(row["median_ms"])
    shapes = []
    for (batch, context), latencies in sorted(grouped.items()):
        if "production-default" not in latencies:
            raise ValueError(f"decode shape {(batch, context)} lacks production control")
        shapes.append(Shape(
            "decode", (batch, context),
            {
                "batch_size": batch,
                "context_length": context,
                "total_kv_tokens": batch * context,
            },
            latencies, "production-default",
        ))
    return shapes


def load_swiglu_shapes(path: Path) -> list[Shape]:
    grouped = {}
    for row in _load(path)["rows"]:
        if row.get("status") != "ok" or row.get("fused_median_ms") is None:
            continue
        rows = int(row["rows"])
        latencies = grouped.setdefault(rows, {})
        _set_latency(latencies, "baseline", row["baseline_median_ms"])
        action = f"fused-b{int(row['block_size'])}-w{int(row['num_warps'])}"
        _set_latency(latencies, action, row["fused_median_ms"])
    shapes = []
    for rows, latencies in sorted(grouped.items()):
        shapes.append(Shape(
            "swiglu", (rows,), {"rows": rows}, latencies, "baseline",
        ))
    return shapes


def load_rope_kv_shapes(path: Path) -> list[Shape]:
    grouped = {}
    for row in _load(path)["rows"]:
        if row.get("status") != "ok" or row.get("fused_median_ms") is None:
            continue
        tokens = int(row["tokens"])
        latencies = grouped.setdefault(tokens, {})
        _set_latency(latencies, "baseline", row["baseline_median_ms"])
        action = f"fused-w{int(row['num_warps'])}"
        _set_latency(latencies, action, row["fused_median_ms"])
    return [
        Shape("rope-kv", (tokens,), {"tokens": tokens}, latencies, "baseline")
        for tokens, latencies in sorted(grouped.items())
    ]


def load_paged_prefill_shapes(path: Path) -> list[Shape]:
    grouped = {}
    for row in _load(path)["rows"]:
        if row.get("status") != "ok" or row.get("triton_median_ms") is None:
            continue
        batch = int(row["batch_size"])
        query = int(row["query_length"])
        prefix = int(row["prefix_length"])
        key = (batch, query, prefix)
        action = "tile-" + "x".join(str(int(row[name])) for name in (
            "block_m", "block_n", "num_warps", "num_stages",
        ))
        _set_latency(grouped.setdefault(key, {}), action, row["triton_median_ms"])
    baseline = "tile-64x32x4x2"
    shapes = []
    for (batch, query, prefix), latencies in sorted(grouped.items()):
        if baseline not in latencies:
            raise ValueError(
                f"paged-prefill shape {(batch, query, prefix)} lacks production control"
            )
        shapes.append(Shape(
            "paged-prefill", (batch, query, prefix),
            {
                "batch_size": batch,
                "query_length": query,
                "prefix_length": prefix,
                "total_query_tokens": batch * query,
                "context_length": prefix + query,
                "attention_pairs": batch * (
                    query * prefix + query * (query + 1) // 2
                ),
            },
            latencies, baseline,
        ))
    return shapes


LOADERS = {
    "decode": load_decode_shapes,
    "swiglu": load_swiglu_shapes,
    "rope-kv": load_rope_kv_shapes,
    "paged-prefill": load_paged_prefill_shapes,
}


def validate_shapes(shapes: list[Shape]) -> None:
    if len(shapes) < 2:
        raise ValueError("each dispatch fit needs at least two successful shapes")
    component = shapes[0].component
    for shape in shapes:
        if shape.component != component:
            raise ValueError("a policy fit cannot mix components")
        if shape.baseline_action not in shape.latencies:
            raise ValueError(f"shape {shape.key} lacks baseline action")
        if not shape.latencies or any(value <= 0 for value in shape.latencies.values()):
            raise ValueError(f"shape {shape.key} has invalid latency data")


def deterministic_holdout(shape: Shape, modulus: int) -> bool:
    digest = hashlib.sha256(
        f"{shape.component}:{','.join(map(str, shape.key))}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") % modulus == 0


def split_train_holdout(shapes: list[Shape], modulus: int):
    if modulus < 2:
        raise ValueError("holdout modulus must be at least two")
    train = [shape for shape in shapes if not deterministic_holdout(shape, modulus)]
    holdout = [shape for shape in shapes if deterministic_holdout(shape, modulus)]
    if not train or not holdout:
        ordered = sorted(shapes, key=lambda shape: shape.key)
        holdout = ordered[::modulus] or ordered[-1:]
        holdout_keys = {shape.key for shape in holdout}
        train = [shape for shape in ordered if shape.key not in holdout_keys]
    if not train:
        raise ValueError("holdout split left no training shapes")
    return train, holdout


def _leaf_choice(shapes: list[Shape], max_training_regression: float):
    """Choose the action with the lowest mean per-shape oracle regret.

    Raw latency sums let expensive long-context shapes overwhelm short shapes and can
    intentionally regress launch-bound work. Normalizing each shape by its own oracle
    gives every dispatch point equal weight while still optimizing measured latency.
    """
    actions = sorted({action for shape in shapes for action in shape.latencies})
    candidates = []
    for action in actions:
        if all(action in shape.latencies for shape in shapes):
            if any(
                shape.latencies[action]
                > shape.latencies[shape.baseline_action] * (1 + max_training_regression)
                for shape in shapes
            ):
                continue
            total_regret = sum(
                shape.latencies[action] / min(shape.latencies.values())
                for shape in shapes
            )
            candidates.append((total_regret, action))
    if not candidates:
        raise ValueError("no action succeeded on every shape in a policy leaf")
    return min(candidates)


def fit_tree(
    shapes: list[Shape],
    features: tuple[str, ...],
    *,
    max_depth: int,
    min_leaf_shapes: int,
    min_split_improvement: float,
    max_training_regression: float,
    depth: int = 0,
) -> dict:
    leaf_cost, leaf_action = _leaf_choice(shapes, max_training_regression)
    leaf = {
        "kind": "leaf",
        "action": leaf_action,
        "training_shapes": len(shapes),
        "training_regret_sum": leaf_cost,
    }
    if depth >= max_depth or len(shapes) < 2 * min_leaf_shapes:
        return leaf

    best = None
    for feature in features:
        values = sorted({shape.features[feature] for shape in shapes})
        for low, high in zip(values, values[1:]):
            threshold = (low + high) / 2
            left = [shape for shape in shapes if shape.features[feature] <= threshold]
            right = [shape for shape in shapes if shape.features[feature] > threshold]
            if len(left) < min_leaf_shapes or len(right) < min_leaf_shapes:
                continue
            left_cost, _ = _leaf_choice(left, max_training_regression)
            right_cost, _ = _leaf_choice(right, max_training_regression)
            split_cost = left_cost + right_cost
            candidate = (split_cost, feature, threshold, left, right)
            if best is None or candidate[:3] < best[:3]:
                best = candidate
    if best is None:
        return leaf
    split_cost, feature, threshold, left, right = best
    improvement = (leaf_cost - split_cost) / leaf_cost
    if improvement < min_split_improvement:
        return leaf
    return {
        "kind": "split",
        "feature": feature,
        "threshold": threshold,
        "training_shapes": len(shapes),
        "leaf_action_without_split": leaf_action,
        "relative_training_improvement": improvement,
        "left": fit_tree(
            left, features, max_depth=max_depth,
            min_leaf_shapes=min_leaf_shapes,
            min_split_improvement=min_split_improvement,
            max_training_regression=max_training_regression, depth=depth + 1,
        ),
        "right": fit_tree(
            right, features, max_depth=max_depth,
            min_leaf_shapes=min_leaf_shapes,
            min_split_improvement=min_split_improvement,
            max_training_regression=max_training_regression, depth=depth + 1,
        ),
    }


def select_action(tree: dict, features: dict[str, int]) -> str:
    node = tree
    while node["kind"] == "split":
        branch = "left" if features[node["feature"]] <= node["threshold"] else "right"
        node = node[branch]
    return node["action"]


def readable_rules(tree: dict) -> list[str]:
    rules = []

    def visit(node, conditions):
        if node["kind"] == "leaf":
            rules.append(
                f"{' and '.join(conditions) if conditions else 'always'} -> {node['action']}"
            )
            return
        threshold = node["threshold"]
        rendered = str(int(threshold)) if float(threshold).is_integer() else f"{threshold:g}"
        visit(node["left"], conditions + [f"{node['feature']} <= {rendered}"])
        visit(node["right"], conditions + [f"{node['feature']} > {rendered}"])

    visit(tree, [])
    return rules


def _percentile(values, probability):
    ordered = sorted(values)
    index = max(0, math.ceil(probability * len(ordered)) - 1)
    return ordered[index]


def evaluate(tree: dict, shapes: list[Shape], split: str, winner_margin: float):
    rows = []
    for shape in shapes:
        selected_action = select_action(tree, shape.features)
        if selected_action not in shape.latencies:
            selected_action = shape.baseline_action
        ordered = sorted((latency, action) for action, latency in shape.latencies.items())
        oracle_ms, oracle_action = ordered[0]
        runner_up_ms = ordered[1][0] if len(ordered) > 1 else oracle_ms
        baseline_ms = shape.latencies[shape.baseline_action]
        selected_ms = shape.latencies[selected_action]
        margin = runner_up_ms / oracle_ms
        rows.append({
            "component": shape.component,
            "split": split,
            "shape": "x".join(map(str, shape.key)),
            "features": json.dumps(shape.features, sort_keys=True),
            "baseline_action": shape.baseline_action,
            "selected_action": selected_action,
            "oracle_action": oracle_action,
            "baseline_ms": baseline_ms,
            "selected_ms": selected_ms,
            "oracle_ms": oracle_ms,
            "speedup_vs_baseline": baseline_ms / selected_ms,
            "oracle_regret": selected_ms / oracle_ms,
            "oracle_margin": margin,
            "robust_winner": margin >= winner_margin,
        })
    return rows


def summarize(rows: list[dict]) -> dict:
    baseline = sum(row["baseline_ms"] for row in rows)
    selected = sum(row["selected_ms"] for row in rows)
    oracle = sum(row["oracle_ms"] for row in rows)
    regrets = [row["oracle_regret"] for row in rows]
    return {
        "shape_count": len(rows),
        "aggregate_speedup_vs_baseline": baseline / selected,
        "aggregate_oracle_regret": selected / oracle,
        "geomean_oracle_regret": math.exp(
            statistics.fmean(math.log(value) for value in regrets)
        ),
        "p95_oracle_regret": _percentile(regrets, 0.95),
        "max_oracle_regret": max(regrets),
        "oracle_action_agreement": statistics.fmean(
            row["selected_action"] == row["oracle_action"] for row in rows
        ),
        "robust_winner_fraction": statistics.fmean(
            bool(row["robust_winner"]) for row in rows
        ),
    }


def fit_component(shapes: list[Shape], args) -> tuple[dict, list[dict]]:
    validate_shapes(shapes)
    train, holdout = split_train_holdout(shapes, args.holdout_modulus)
    component = shapes[0].component
    tree = fit_tree(
        train, COMPONENT_FEATURES[component],
        max_depth=args.max_depth,
        min_leaf_shapes=args.min_leaf_shapes,
        min_split_improvement=args.min_split_improvement,
        max_training_regression=args.max_training_regression,
    )
    rows = (
        evaluate(tree, train, "train", args.winner_margin)
        + evaluate(tree, holdout, "holdout", args.winner_margin)
    )
    policy = {
        "component": component,
        "features": list(COMPONENT_FEATURES[component]),
        "tree": tree,
        "rules": readable_rules(tree),
        "train": summarize([row for row in rows if row["split"] == "train"]),
        "holdout": summarize([row for row in rows if row["split"] == "holdout"]),
        "all": summarize(rows),
    }
    return policy, rows


def main() -> int:
    args = build_parser().parse_args()
    if args.max_depth < 0 or args.min_leaf_shapes < 1:
        raise ValueError("max depth must be non-negative and min leaf positive")
    if not 0 <= args.min_split_improvement < 1:
        raise ValueError("min split improvement must be in [0, 1)")
    if not 0 <= args.max_training_regression < 1:
        raise ValueError("max training regression must be in [0, 1)")
    if args.winner_margin < 1:
        raise ValueError("winner margin must be at least one")
    inputs = {
        "decode": args.decode_json,
        "swiglu": args.swiglu_json,
        "rope-kv": args.rope_kv_json,
        "paged-prefill": args.paged_prefill_json,
    }
    selected = {name: path for name, path in inputs.items() if path is not None}
    if not selected:
        raise ValueError("provide at least one component result JSON")

    policies = []
    evaluation_rows = []
    for component, path in selected.items():
        policy, rows = fit_component(LOADERS[component](path), args)
        policy["source"] = str(path)
        policies.append(policy)
        evaluation_rows.extend(rows)
        print(f"\n[{component}]")
        for rule in policy["rules"]:
            print(f"  {rule}")
        held = policy["holdout"]
        print(
            f"  held-out: {held['shape_count']} shapes, "
            f"speedup={held['aggregate_speedup_vs_baseline']:.3f}x, "
            f"oracle_regret={held['aggregate_oracle_regret']:.4f}x, "
            f"p95_regret={held['p95_oracle_regret']:.4f}x"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    stem = args.output_dir / f"dispatch-policy-{stamp}"
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "max_depth": args.max_depth,
            "min_leaf_shapes": args.min_leaf_shapes,
            "min_split_improvement": args.min_split_improvement,
            "max_training_regression": args.max_training_regression,
            "holdout_modulus": args.holdout_modulus,
            "winner_margin": args.winner_margin,
        },
        "policies": policies,
        "evaluation_rows": evaluation_rows,
    }
    json_path = stem.with_suffix(".json")
    csv_path = stem.with_suffix(".csv")
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVALUATION_FIELDS)
        writer.writeheader()
        writer.writerows(
            {field: row.get(field) for field in EVALUATION_FIELDS}
            for row in evaluation_rows
        )
    print(f"\njson: {json_path}")
    print(f"csv: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
