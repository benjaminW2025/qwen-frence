"""
Correctness: CUDA-graph decode vs naive (baseline) and eager-paged, token-for-token.

Teacher-forced logit parity (same discipline as check_paged_vs_baseline): prefill all
three, then feed EVERY engine the same next token (baseline's greedy pick) and compare
logits per step. Identical inputs -> any gap is a real graph discrepancy, not drift.

Three streams from two models:
  - baseline : contiguous KVCache + SDPA  (BaselineEngine)
  - paged    : PagedKVCache + eager paged_forward  (PagedGraphEngine's inherited decode_step)
  - graph    : PagedKVCache + captured CUDA graph   (CUDAGraphDecoder.decode)
The paged + graph streams share one PagedGraphEngine (same weights), so only baseline
loads a second copy.

PASS: graph's top-1 agrees with baseline at every step (and with eager-paged), max abs
logit err within fp16 tolerance. The graph forward is numerically the same as eager
paged, so err vs paged should be ~0; err vs baseline ~ the usual cross-impl fp16 noise.
"""

import _bootstrap  # noqa: F401

from types import SimpleNamespace

import torch

from naive_forward import Qwen2Config
from kv_cache import KVCache
from baseline_engine import BaselineEngine

from paged_graph_engine import PagedGraphEngine
from paged_graph_decoder import CUDAGraphDecoder, build_decode_step_inputs

DEVICE = "cuda"
DTYPE = torch.float16

PROMPTS = [
    "The capital of France is",
    "Once upon a time, in a distant kingdom,",
    "def fibonacci(n):",
]
N_STEPS = 32
LOGIT_TOL = 5e-2
EAGER_REPLAY_TOL = 1e-2


def _prefill(engine, cache, prompt_ids):
    return engine.prefill(SimpleNamespace(prompt_ids=prompt_ids), cache)


def compare_prompt(baseline, paged, prompt, n_steps):
    tok_ids = baseline.tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
    max_blocks = (len(tok_ids) + n_steps + paged.block_size - 1) // paged.block_size

    # Three independent caches, same content.
    b_cache = KVCache(baseline.cfg, 1, DEVICE, DTYPE)
    p_cache = paged._new_cache(len(tok_ids) + n_steps)   # eager-paged reference
    g_cache = paged._new_cache(len(tok_ids) + n_steps)   # graphed path

    b_log = _prefill(baseline, b_cache, tok_ids)
    p_log = _prefill(paged, p_cache, tok_ids)
    _prefill(paged, g_cache, tok_ids)                    # advance g_cache; capture reads its state

    # Capture the graph AFTER prefill (throwaway write goes to an unused pool slot).
    decoder = CUDAGraphDecoder(paged.model, g_cache, batch_size=1,
                               max_blocks=max_blocks, device=DEVICE, dtype=DTYPE)
    decoder.capture()

    def graph_step(tok):
        g_cache.allocate_block([1])
        inputs = build_decode_step_inputs(g_cache, [tok], max_blocks, DEVICE)
        return decoder.decode(*inputs)[0, -1]

    max_err_gb = max_err_gp = 0.0
    agree_gb = agree_gp = True
    first_div = None
    rows = []

    # The graph path is decode-only (no "prefill logits"), so we compare from step 1 on.
    # baseline's greedy pick drives all three engines in lockstep.
    tok = int(b_log.argmax())
    for step in range(1, n_steps + 1):
        b_log = baseline.decode_step(b_cache, tok)
        p_log = paged.decode_step(p_cache, tok)
        g_log = graph_step(tok)

        err_gb = (g_log.float() - b_log.float()).abs().max().item()
        err_gp = (g_log.float() - p_log.float()).abs().max().item()
        a_gb = int(g_log.argmax()) == int(b_log.argmax())
        a_gp = int(g_log.argmax()) == int(p_log.argmax())
        max_err_gb = max(max_err_gb, err_gb)
        max_err_gp = max(max_err_gp, err_gp)
        if not a_gb and first_div is None:
            first_div = step
        agree_gb &= a_gb
        agree_gp &= a_gp
        rows.append((step, err_gb, err_gp, a_gb, a_gp))
        tok = int(b_log.argmax())

    return dict(max_err_gb=max_err_gb, max_err_gp=max_err_gp,
                agree_gb=agree_gb, agree_gp=agree_gp, first_div=first_div, rows=rows)


def main():
    assert torch.cuda.is_available(), "needs the GPU"
    cfg = Qwen2Config()
    print("loading baseline engine...")
    baseline = BaselineEngine(cfg=cfg, device=DEVICE, dtype=DTYPE)
    print("loading paged graph engine...")
    paged = PagedGraphEngine(cfg=cfg, device=DEVICE, dtype=DTYPE)

    overall = True
    for prompt in PROMPTS:
        r = compare_prompt(baseline, paged, prompt, N_STEPS)
        ok = (
            r["agree_gb"]
            and r["agree_gp"]
            and r["max_err_gb"] <= LOGIT_TOL
            and r["max_err_gp"] <= EAGER_REPLAY_TOL
        )
        overall &= ok

        print("\n" + "=" * 70)
        print(f"prompt: {prompt!r}")
        print(f"  graph vs baseline : top-1 always={r['agree_gb']}  max_err={r['max_err_gb']:.4f}"
              + ("" if r["agree_gb"] else f"  (first div step {r['first_div']})"))
        print(f"  graph vs eager-paged: top-1 always={r['agree_gp']}  max_err={r['max_err_gp']:.4f}"
              f"   (tol {EAGER_REPLAY_TOL}; same forward, replayed)")
        print(f"  -> {'PASS' if ok else 'FAIL'}")
        if not (r["agree_gb"] and r["agree_gp"]):
            print("  step | err_g_vs_b | err_g_vs_p | agree_gb | agree_gp")
            for row in r["rows"][1:]:
                step, egb, egp, agb, agp = row
                flag = "" if (agb and agp) else "  <--"
                print(f"  {step:>4} | {egb:>10.4f} | {egp:>10.4f} | {str(agb):>8} | {str(agp):>8}{flag}")

    print("\n" + "=" * 70)
    print("OVERALL:", "PASS ✅" if overall else "FAIL ❌")
    return overall


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
