"""
Correctness: paged forward vs the contiguous baseline, token-for-token.
"""

import _bootstrap  # noqa: F401

from types import SimpleNamespace

import torch

from naive_forward import Qwen2Config
from kv_cache import KVCache
from baseline_engine import BaselineEngine
from paged_engine import PagedEngine

DEVICE = "cuda"
DTYPE = torch.float16

PROMPTS = [
    "The capital of France is",
    "Once upon a time, in a distant kingdom,",
    "def fibonacci(n):",
]
N_STEPS = 32          # decode steps to compare after prefill
LOGIT_TOL = 5e-2      # fp16 over 28 layers + different attention implementations


def _prefill(engine, cache, prompt_ids):
    # Both engines' prefill only reads seq.prompt_ids, so a tiny stand-in works.
    seq = SimpleNamespace(prompt_ids=prompt_ids)
    return engine.prefill(seq, cache)          # (vocab,)


def compare_prompt(baseline, paged, prompt, n_steps):
    tok_ids = baseline.tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()

    # Independent caches, same content. Baseline reserves max_seq_len; paged sizes to req.
    b_cache = KVCache(baseline.cfg, 1, DEVICE, DTYPE)
    p_cache = paged._new_cache(len(tok_ids) + n_steps)

    b_log = _prefill(baseline, b_cache, tok_ids)
    p_log = _prefill(paged, p_cache, tok_ids)

    max_err = 0.0
    all_agree = True
    first_div = None
    rows = []

    def record(step, bl, pl):
        nonlocal max_err, all_agree, first_div
        err = (bl.float() - pl.float()).abs().max().item()
        b_top1, p_top1 = int(bl.argmax()), int(pl.argmax())
        agree = b_top1 == p_top1
        top2 = bl.float().topk(2).values
        margin = (top2[0] - top2[1]).item()      # how close the baseline's top-1 vs top-2 is
        max_err = max(max_err, err)
        if not agree and first_div is None:
            first_div = step
        all_agree &= agree
        rows.append((step, err, agree, margin, b_top1, p_top1))

    record(0, b_log, p_log)                      # prefill's last-position logits
    tok = int(b_log.argmax())                    # drive BOTH with the baseline's greedy pick
    for step in range(1, n_steps + 1):
        b_log = baseline.decode_step(b_cache, tok)
        p_log = paged.decode_step(p_cache, tok)
        record(step, b_log, p_log)
        tok = int(b_log.argmax())

    return max_err, all_agree, first_div, rows


def main():
    assert torch.cuda.is_available(), "needs the GPU"
    cfg = Qwen2Config()
    print("loading baseline engine...")
    baseline = BaselineEngine(cfg=cfg, device=DEVICE, dtype=DTYPE)
    print("loading paged engine...")
    paged = PagedEngine(cfg=cfg, device=DEVICE, dtype=DTYPE)

    overall_pass = True
    for prompt in PROMPTS:
        max_err, all_agree, first_div, rows = compare_prompt(baseline, paged, prompt, N_STEPS)
        ok = all_agree and max_err <= LOGIT_TOL
        overall_pass &= ok

        print("\n" + "=" * 70)
        print(f"prompt: {prompt!r}")
        print(f"  steps compared     : {len(rows)} (prefill + {N_STEPS} decode)")
        print(f"  top-1 agrees always: {all_agree}"
              + ("" if all_agree else f"  (first divergence at step {first_div})"))
        print(f"  max abs logit err  : {max_err:.4f}  (tol {LOGIT_TOL})")
        print(f"  -> {'PASS' if ok else 'FAIL'}")

        # If something diverged, show the neighborhood so a real bug is separable from
        # a benign near-tie (tiny margin + tiny err = fp16 coin-flip, not a bug).
        if not all_agree:
            print("  step |  logit_err | agree | top1_margin | b_top1 | p_top1")
            for step, err, agree, margin, bt, pt in rows:
                flag = "" if agree else "  <-- DIVERGE"
                print(f"  {step:>4} | {err:>10.4f} | {str(agree):>5} | "
                      f"{margin:>11.4f} | {bt:>6} | {pt:>6}{flag}")

    print("\n" + "=" * 70)
    print("OVERALL:", "PASS ✅" if overall_pass else "FAIL ❌")
    return overall_pass


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
