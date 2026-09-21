#!/usr/bin/env python3
"""Gate FA3 paged decode for CUDA-graph replay before full-model timing.

This test deliberately changes Q, sequence lengths, and page tables after capture.
Passing therefore proves that replay reads the engine's live static buffers rather
than accidentally reusing the values present while the graph was recorded.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (HERE, ROOT / "custom_kernels"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def comma_separated_ints(value, *, allow_zero=False):
    try:
        result = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    lower_bound = 0 if allow_zero else 1
    if not result or any(item < lower_bound for item in result):
        requirement = "nonnegative" if allow_zero else "positive"
        raise argparse.ArgumentTypeError(f"values must be {requirement}")
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=comma_separated_ints, default=[8, 64])
    parser.add_argument("--context-lengths", type=comma_separated_ints, default=[2048, 4096])
    parser.add_argument("--num-splits",
                        type=lambda value: comma_separated_ints(value, allow_zero=True),
                        default=[0, 22],
                        help="FA3 split counts; 0 selects automatic scheduling")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "experiments/results/fa3-graph-capture")
    return parser


def error_report(torch, actual, expected):
    difference = (actual.float() - expected.float()).abs()
    tolerance = 2e-2 if expected.dtype == torch.bfloat16 else 2e-3
    outside = difference > tolerance + tolerance * expected.float().abs()
    return {
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "mismatched": int(outside.sum()),
        "elements": outside.numel(),
        "atol": tolerance,
        "rtol": tolerance,
        "allclose": not bool(outside.any()),
    }


def measure(torch, operation, warmups, repetitions):
    for _ in range(warmups):
        operation()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repetitions):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        operation()
        end.record()
        end.synchronize()
        samples.append(float(begin.elapsed_time(end)))
    return {"median_ms": statistics.median(samples), "samples_ms": samples}


def run_shape(torch, operation, make_inputs, *, batch_size, context_length,
              num_splits, dtype, seed, warmups, repetitions):
    q, k_pool, v_pool, table, seq_lens = make_inputs(
        [context_length] * batch_size, dtype=dtype, seed=seed,
    )
    static_q = q.clone()
    static_table = table.clone()
    static_lens = seq_lens.clone()

    def static_operation():
        return operation(static_q, k_pool, v_pool, static_table, static_lens,
                         num_splits=num_splits)

    # Allocator/kernel initialization must happen before capture and on a side
    # stream, matching the engine's graph decoder.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            static_operation()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = static_operation()
    graph.replay()
    torch.cuda.synchronize()
    captured_check = error_report(torch, graph_output, operation(
        q, k_pool, v_pool, table, seq_lens, num_splits=num_splits,
    ))

    # Change every runtime input that should be dynamic. Rolling the page table
    # makes stale metadata obvious because each sequence owns distinct pages.
    generator = torch.Generator(device=q.device).manual_seed(seed + 1)
    changed_q = torch.randn(q.shape, device=q.device, dtype=q.dtype, generator=generator)
    offsets = torch.arange(batch_size, device=q.device, dtype=torch.int32) % 13
    changed_lens = seq_lens - offsets
    changed_table = table.roll(1, dims=0)
    expected = operation(changed_q, k_pool, v_pool, changed_table, changed_lens,
                         num_splits=num_splits)
    static_q.copy_(changed_q)
    static_table.copy_(changed_table)
    static_lens.copy_(changed_lens)
    graph.replay()
    torch.cuda.synchronize()
    replay_check = error_report(torch, graph_output, expected)
    changed = not torch.equal(graph_output, operation(
        q, k_pool, v_pool, table, seq_lens, num_splits=num_splits,
    ))

    def copy_and_replay():
        static_q.copy_(changed_q)
        static_table.copy_(changed_table)
        static_lens.copy_(changed_lens)
        graph.replay()

    eager = measure(
        torch,
        lambda: operation(changed_q, k_pool, v_pool, changed_table, changed_lens,
                          num_splits=num_splits),
        warmups, repetitions,
    )
    replay = measure(torch, graph.replay, warmups, repetitions)
    replay_with_copies = measure(torch, copy_and_replay, warmups, repetitions)
    passed = captured_check["allclose"] and replay_check["allclose"] and changed
    return {
        "batch_size": batch_size,
        "context_length": context_length,
        "num_splits": num_splits,
        "scheduler": "auto" if num_splits == 0 else f"fixed-{num_splits}",
        "captured_input_check": captured_check,
        "mutated_replay_check": replay_check,
        "output_changed_after_mutation": changed,
        "graph_safe": passed,
        "eager": eager,
        "graph_replay": replay,
        "graph_replay_with_input_copies": replay_with_copies,
        "speedup_replay_vs_eager": eager["median_ms"] / replay["median_ms"],
        "speedup_copy_replay_vs_eager": (
            eager["median_ms"] / replay_with_copies["median_ms"]
        ),
    }


def main():
    args = build_parser().parse_args()
    if args.warmups < 0 or args.repetitions < 1:
        raise ValueError("warmups must be nonnegative and repetitions positive")
    import torch
    from grouped_splitk_validation import make_inputs
    from paged_decode_fa3 import fa3_paged_decode_attention

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    dtype = getattr(torch, args.dtype)
    split_counts = args.num_splits
    if len(set(split_counts)) != len(split_counts):
        raise ValueError("num-splits values must be distinct")

    rows = []
    for batch_size in args.batch_sizes:
        for context_length in args.context_lengths:
            for num_splits in split_counts:
                label = "auto" if num_splits == 0 else str(num_splits)
                print(f"FA3 graph gate B={batch_size} C={context_length} splits={label}",
                      flush=True)
                row = run_shape(
                    torch, fa3_paged_decode_attention, make_inputs,
                    batch_size=batch_size, context_length=context_length,
                    num_splits=num_splits, dtype=dtype,
                    seed=args.seed + batch_size + context_length + num_splits,
                    warmups=args.warmups, repetitions=args.repetitions,
                )
                rows.append(row)
                print(
                    f"  graph_safe={row['graph_safe']} "
                    f"eager={row['eager']['median_ms']:.4f} ms "
                    f"copy+replay={row['graph_replay_with_input_copies']['median_ms']:.4f} ms",
                    flush=True,
                )

    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "configuration": {**vars(args), "output_dir": str(args.output_dir),
                          "resolved_num_splits": split_counts},
        "rows": rows,
        "all_graph_safe": all(row["graph_safe"] for row in rows),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_dir / f"fa3-graph-capture-{stamp}.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {output}")
    if not report["all_graph_safe"]:
        raise SystemExit("FA3 CUDA-graph safety gate failed; do not run full-model timing")


if __name__ == "__main__":
    main()
