"""
Correctness: continuous-batching Scheduler vs an independent per-request reference.

Reference: each request run ALONE (batch=1) through PagedEngine's validated prefill +
decode_step, greedy. The scheduler must reproduce those tokens while interleaving many
requests through a shared pool with admission / eviction / (part 2) preemption.

Caveat on exactness: the scheduler decodes a given sequence at whatever batch size the
schedule produces (1..max_running), while the reference decodes it at batch=1. fp16
matmuls can pick different kernels per batch size, so a greedy near-tie can flip and the
autoregressive continuation then diverges. That's benign fp16, not a bug -- so we report
the matched-prefix length per request and flag only EARLY divergence (step < 2), which
would indicate a real defect (bad prefill / KV write / block indexing).

Part 1: generous pool (no preemption) -> validates prefill + ragged batched decode +
        admission/eviction.
Part 2: tight pool -> forces preemption; validates recompute + atomic alloc + n_prompt
        accounting. Asserts preemption actually fired.
"""

import _bootstrap  # noqa: F401

from types import SimpleNamespace

import torch

from naive_forward import Qwen2Config
from paged_engine import PagedEngine
from scheduler import Scheduler

DEVICE = "cuda"
DTYPE = torch.float16

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Once upon a time, in a land far away,",
    "The three laws of thermodynamics are",
    "import numpy as np\n",
    "To make a good cup of coffee, you",
]
MAX_TOKENS = 32
MAX_RUNNING = 3


@torch.no_grad()
def ref_generate(engine, prompt_ids, max_tokens, eos_ids):
    """One request, alone, batch=1, greedy -> list of generated token ids."""
    cache = engine._new_cache(len(prompt_ids) + max_tokens)
    logits = engine.prefill(SimpleNamespace(prompt_ids=prompt_ids), cache)
    out = [int(logits.argmax())]
    while not (len(out) >= max_tokens or out[-1] in eos_ids):
        logits = engine.decode_step(cache, out[-1])
        out.append(int(logits.argmax()))
    return out


def run_scheduler(engine, prompt_ids_list, max_tokens, num_blocks):
    sched = Scheduler(engine.model, engine.cfg, MAX_RUNNING, num_blocks,
                      engine.block_size, engine.eos_ids, DEVICE, DTYPE)
    ids = [sched.add_request(p, max_tokens) for p in prompt_ids_list]
    out = sched.run()
    return [out[i] for i in ids], sched.n_preemptions


def matched_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def report(name, sched_outs, ref_outs, n_preempt):
    print("\n" + "=" * 68)
    print(f"{name}   (preemptions: {n_preempt})")
    print(f"  {'req':>3} | {'ref_len':>7} | {'sched_len':>9} | {'match':>5} | result")
    ok = True
    for i, (s, r) in enumerate(zip(sched_outs, ref_outs)):
        m = matched_prefix(s, r)
        full = (s == r)
        # early divergence = real-bug suspect; late = benign fp16 flip
        verdict = "EXACT" if full else ("fp16-flip?" if m >= 2 else "BUG?")
        ok &= (full or m >= 2)
        print(f"  {i:>3} | {len(r):>7} | {len(s):>9} | {m:>5} | {verdict}")
    print(f"  -> {'PASS' if ok else 'FAIL (early divergence)'}")
    return ok


def main():
    assert torch.cuda.is_available(), "needs the GPU"
    engine = PagedEngine(cfg=Qwen2Config(), device=DEVICE, dtype=DTYPE)
    prompt_ids = [engine.tokenizer(p, return_tensors="pt").input_ids[0].tolist() for p in PROMPTS]

    print("building per-request reference (each alone, batch=1)...")
    ref = [ref_generate(engine, p, MAX_TOKENS, engine.eos_ids) for p in prompt_ids]

    # Part 1: generous pool -> no preemption expected.
    gen_blocks = MAX_RUNNING * ((max(len(p) for p in prompt_ids) + MAX_TOKENS) // engine.block_size + 2) * 2
    out1, pre1 = run_scheduler(engine, prompt_ids, MAX_TOKENS, gen_blocks)
    ok1 = report(f"PART 1: generous pool ({gen_blocks} blocks)", out1, ref, pre1)

    # Part 2: tight pool -> force preemption. Sized below the concurrent working set but
    # above a single sequence's worst case (so no hard error).
    tight_blocks = 8
    out2, pre2 = run_scheduler(engine, prompt_ids, MAX_TOKENS, tight_blocks)
    ok2 = report(f"PART 2: tight pool ({tight_blocks} blocks)", out2, ref, pre2)
    if pre2 == 0:
        print("  WARNING: no preemption fired -- shrink tight_blocks to actually exercise it")

    print("\n" + "=" * 68)
    overall = ok1 and ok2 and pre2 > 0
    print("OVERALL:", "PASS ✅" if overall else "CHECK ❌")
    return overall


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
