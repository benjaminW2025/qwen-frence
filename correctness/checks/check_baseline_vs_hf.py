"""
Correctness test for naive forward pass implementation
"""

import _bootstrap  # noqa: F401

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from naive_forward import Qwen2Config
from kv_cache import KVCache
from weight_loader import QwenWeightLoader

MODEL_ID = "Qwen/Qwen2.5-1.5B"
DEVICE = "cuda"
DTYPE = torch.float16
N_PAIRS = 20  # sampled logits shown for the worst-diff prompt

# Diverse prompts (factual / code / narrative / math / QA) to stress different
# token distributions. Each is run independently at batch=1 (no padding needed).
PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Once upon a time, in a distant land,",
    "The meaning of life is",
    "2 + 2 =",
    "In the beginning God created the heavens",
    "Q: What is the speed of light? A:",
    "The quick brown fox jumps over",
]

LOGIT_TOL = 5e-2


@torch.no_grad()
def eval_prompt(model, hf_model, tok, prompt):
    """Compare the final-position logits produced from the same complete prompt."""
    input_ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
    cache = KVCache(
        model.cfg,
        batch_size=1,
        device=DEVICE,
        dtype=DTYPE,
        max_seq_len=input_ids.shape[1],
    )
    ours = model(input_ids, cache)                    # (1, 1, vocab)
    ref = hf_model(input_ids).logits[:, -1:, :]       # (1, 1, vocab)

    diff = (ours.float() - ref.float()).abs()
    # Agreement at the final prompt position (the first generated token).
    top1_match = (ours.argmax(-1) == ref.argmax(-1)).float().mean().item()
    return {
        "prompt": prompt,
        "seq": input_ids.shape[1],
        "max_diff": diff.max().item(),
        "mean_diff": diff.mean().item(),
        "top1_match": top1_match,
        "ours": ours,
        "ref": ref,
    }


@torch.no_grad()
def main():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    # HF reference. Force SDPA so its attention path matches ours (both flash/SDPA).
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=DTYPE, attn_implementation="sdpa"
    ).to(DEVICE).eval()

    cfg = Qwen2Config()
    model = QwenWeightLoader(cfg).convert(hf_model, DEVICE, DTYPE)

    results = [eval_prompt(model, hf_model, tok, p) for p in PROMPTS]

    # Per-prompt summary table.
    print(f"{'prompt':<42} {'seq':>4} {'max_diff':>10} {'mean_diff':>10} {'top1':>7}")
    print("-" * 78)
    for r in results:
        p = (r["prompt"][:39] + "...") if len(r["prompt"]) > 42 else r["prompt"]
        print(f"{p:<42} {r['seq']:>4} {r['max_diff']:>10.5f} "
              f"{r['mean_diff']:>10.6f} {r['top1_match'] * 100:>6.1f}%")

    # Aggregate across all prompts.
    overall_max = max(r["max_diff"] for r in results)
    mean_top1 = sum(r["top1_match"] for r in results) / len(results)
    print("-" * 78)
    print(f"overall max abs diff : {overall_max:.6f}")
    print(f"mean top-1 agreement : {mean_top1 * 100:.1f}%")

    # Sample N_PAIRS logits from the worst-diff prompt to eyeball where it drifts.
    worst = max(results, key=lambda r: r["max_diff"])
    ours, ref = worst["ours"], worst["ref"]
    S, V = ours.shape[1], ours.shape[2]
    g = torch.Generator().manual_seed(0)
    positions = torch.randint(0, S, (N_PAIRS,), generator=g)
    tokens = torch.randint(0, V, (N_PAIRS,), generator=g)

    print(f"\nworst prompt: {worst['prompt']!r}  (sampled {N_PAIRS} logits)")
    print(f"{'pos':>4} {'vocab':>7} | {'ours':>10} {'hf':>10} | {'abs_diff':>10}")
    print("-" * 50)
    for pos, t in zip(positions.tolist(), tokens.tolist()):
        o = ours[0, pos, t].item()
        r = ref[0, pos, t].item()
        print(f"{pos:>4} {t:>7} | {o:>10.4f} {r:>10.4f} | {abs(o - r):>10.6f}")

    all_top1 = all(result["top1_match"] == 1.0 for result in results)
    ok = overall_max <= LOGIT_TOL and all_top1
    print(f"\noverall max abs diff vs tolerance {LOGIT_TOL}  ->  "
          f"{'PASS' if overall_max <= LOGIT_TOL else 'FAIL'}")
    print(f"top-1 agreement at every position             ->  "
          f"{'PASS' if all_top1 else 'FAIL'}")
    print("OVERALL:", "PASS ✅" if ok else "FAIL ❌")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
