#!/usr/bin/env python3
"""Private vs shared CUDA graph memory pools for the piecewise prefill graphs.

The shared pool changes where segment scratch lives, not what runs, so logits
and every live K/V value must be bitwise identical. Reports the reserved memory
each capture added and the median packed forward time per arm. Buckets above
--private-limit run the shared arm only (private capture needs ~1 GB of
scratch per segment there).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INTEGRATION = ROOT / "experiments/integration"
for directory in (INTEGRATION, ROOT / "baseline", ROOT / "benchmarks", ROOT / "engine/graph",
                  ROOT / "engine/model_runner", ROOT / "engine/kvcache", ROOT / "engine/cpp/build"):
    sys.path.insert(0, str(directory))

from ab_gemm_epilogues import POLICIES, Layout, alternate, parse_case, prefill_inputs  # noqa: E402


def run_cases(args):
    """Isolate allocator/capture failures and persist every completed case."""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for sequences, length in args.cases:
        case_output = args.output.parent / f"case-{sequences}x{length}.json"
        command = [sys.executable, str(Path(__file__).resolve()),
                   "--worker", "--output", str(case_output), "--model", args.model,
                   "--device", args.device, "--attention", args.attention,
                   "--cases", f"{sequences}x{length}",
                   "--private-limit", str(args.private_limit),
                   "--rounds", str(args.rounds), "--seed", str(args.seed)]
        if args.gemm_epilogues:
            command.append("--gemm-epilogues")
        if not case_output.exists():
            result = subprocess.run(command, cwd=ROOT)
            if result.returncode and not case_output.exists():
                case_output.write_text(json.dumps({"status": "fail", "rows": [{
                    "sequences": sequences, "length": length,
                    "tokens": sequences * length, "error": "case subprocess failed",
                    "exit_code": result.returncode}]}, indent=2) + "\n")
        case = json.loads(case_output.read_text())
        rows.extend(case["rows"])
        if case["status"] != "pass":
            break
    passed = len(rows) == len(args.cases) and all(
        "error" not in row and row.get("logits_bitwise_equal", True)
        and row.get("kv_bitwise_equal", True) for row in rows)
    args.output.write_text(json.dumps({
        "status": "pass" if passed else "fail", "attention": args.attention,
        "gemm_epilogues": args.gemm_epilogues, "rows": rows,
        "case_isolation": "subprocess"}, indent=2) + "\n")
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="report JSON path")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attention", choices=tuple(POLICIES), default="fa3")
    parser.add_argument("--cases", type=parse_case, nargs="+", default=[(8, 256), (1, 2048), (8, 2048)],
                        help="SEQUENCESxLENGTH; the bucket is their product")
    parser.add_argument("--private-limit", type=int, default=8192)
    parser.add_argument("--gemm-epilogues", action="store_true",
                        help="capture the fused-epilogue segments instead of the accepted ones")
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    if not args.worker:
        sys.exit(run_cases(args))
    if len(args.cases) != 1:
        parser.error("each worker must execute exactly one case")

    from benchmark_latest_vs_vllm import resolve_model_source
    from model_setup import check_startup, load_model_only
    import torch
    torch.set_grad_enabled(False)
    from model_adapter import allocate_pool
    from piecewise_prefill import PiecewisePrefill

    setup = check_startup(args.device)
    engine, _, _ = load_model_only(resolve_model_source(args), args.device, "float16",
                                   hub_transfer=setup["hub_transfer"])
    model, cfg, device = engine.model, engine.cfg, engine.device
    if args.gemm_epilogues:
        from kernel_dispatch import _load
        _load("fused_gemm").prepare_model(model)
    policy = POLICIES[args.attention][0]
    generator = torch.Generator().manual_seed(args.seed)
    rows = []
    for sequences, length in args.cases:
        tokens = sequences * length
        arms = ("private", "shared") if tokens <= args.private_limit else ("shared",)
        # Shared-only large cases do not need a second KV pool.
        pools = {arm: allocate_pool(cfg, tokens // 16 + sequences + 1, device)
                 for arm in arms}
        prompts = [torch.randint(0, cfg.vocab, (length,), generator=generator) for _ in range(sequences)]
        inputs = prefill_inputs(torch, device, prompts, Layout(torch, device, sequences, length),
                                list(range(sequences)))
        graphs, logits = {}, {}
        for arm in arms:
            torch.cuda.synchronize()
            graphs[arm] = PiecewisePrefill(model, pools[arm], max_capture_tokens=tokens, max_shapes=1,
                                           token_buckets=[tokens], enable_residual_rmsnorm=True,
                                           enable_swiglu_fusion=True, share_graph_pool=arm == "shared",
                                           enable_fused_gemm_epilogues=args.gemm_epilogues)
            logits[arm] = graphs[arm].forward(*inputs, mixed_attention_policy=policy)
        row = {"sequences": sequences, "length": length, "tokens": tokens, "arms": list(arms),
               "capture_reserved_gib": {arm: graphs[arm].capture_memory[tokens] / 2**30 for arm in arms}}
        if len(arms) == 2:
            live = inputs[2]
            row["logits_bitwise_equal"] = torch.equal(logits["private"], logits["shared"])
            row["kv_bitwise_equal"] = all(
                torch.equal(a.view(-1, 2, 128)[live], b.view(-1, 2, 128)[live])
                for a, b in zip(pools["private"].k_pool + pools["private"].v_pool,
                                pools["shared"].k_pool + pools["shared"].v_pool))
        timed, _ = alternate(torch, {arm: (lambda graph=graph: graph.forward(
            *inputs, mixed_attention_policy=policy)) for arm, graph in graphs.items()},
            repetitions=5, rounds=args.rounds)
        row["median_forward_ms"] = timed
        rows.append(row)
        print(f"{sequences}x{length}: reserved " + ", ".join(
            f"{arm} {value:.2f} GiB" for arm, value in row["capture_reserved_gib"].items())
            + "; forward " + ", ".join(f"{arm} {value:.3f} ms" for arm, value in timed.items())
            + (f"; bitwise equal logits={row['logits_bitwise_equal']} kv={row['kv_bitwise_equal']}"
               if len(arms) == 2 else ""), flush=True)
    passed = all(row.get("logits_bitwise_equal", True) and row.get("kv_bitwise_equal", True) for row in rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"status": "pass" if passed else "fail", "attention": args.attention,
                                       "gemm_epilogues": args.gemm_epilogues, "rows": rows}, indent=2) + "\n")
    print(f"status={'pass' if passed else 'fail'}; report={args.output}")
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
