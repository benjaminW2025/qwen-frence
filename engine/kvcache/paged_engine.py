"""Paged-KV autoregressive inference engine for Qwen2.5-1.5B."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "baseline"))

import time
from dataclasses import dataclass, field

import torch
from transformers import AutoTokenizer

from naive_forward import Qwen2Config
from weight_loader import QwenWeightLoader

from paged_kv_cache import PagedKVCache
from paged_forward import paged_forward

MODEL_ID = "Qwen/Qwen2.5-1.5B"


@dataclass
class Sequence:
    prompt_ids: list[int]
    output_ids: list[int] = field(default_factory=list)
    max_tokens: int = 64
    finished: bool = False

    def is_finished(self, eos_ids) -> bool:
        return (self.finished
                or len(self.output_ids) >= self.max_tokens
                or (self.output_ids and self.output_ids[-1] in eos_ids))


class PagedEngine:
    def __init__(self, model_id=MODEL_ID, cfg=None, block_size=16,
                 device="cuda", dtype=torch.float16):
        self.cfg = cfg or Qwen2Config()
        self.device = device
        self.dtype = dtype
        self.block_size = block_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = QwenWeightLoader(self.cfg).load_pretrained(model_id, device, dtype)
        eos = self.tokenizer.eos_token_id
        self.eos_ids = {eos} if isinstance(eos, int) else set(eos or [])

    def _new_cache(self, max_total_tokens, batch_size=1):
        # Paging's whole point: size the pool to THIS request, not max_seq_len.
        # (A real server keeps one shared fixed pool + a scheduler; here a single
        # request owns the pool, so we allocate exactly the blocks it can need.)
        num_blocks = (max_total_tokens + self.block_size - 1) // self.block_size
        return PagedKVCache(self.cfg, batch_size, num_blocks,
                            self.block_size, self.device, self.dtype)

    def prefill(self, seq, cache):
        input_ids = torch.tensor([seq.prompt_ids], device=self.device)
        logits = paged_forward(self.model, cache, input_ids)   # (B, 1, vocab)
        return logits[0, -1]                                   # (vocab,)

    def decode_step(self, cache, token):
        input_ids = torch.tensor([[token]], device=self.device)
        logits = paged_forward(self.model, cache, input_ids)   # (B, 1, vocab)
        return logits[0, -1]

    def sample(self, logits):
        """Greedy -- matches BaselineEngine so outputs are directly comparable."""
        return int(logits.argmax(dim=-1).item())

    @torch.no_grad()
    def generate(self, prompt, max_tokens=64, verbose=True):
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
        seq = Sequence(prompt_ids=prompt_ids, max_tokens=max_tokens)

        # Fresh pool sized to prompt + generation
        cache = self._new_cache(len(prompt_ids) + max_tokens)

        # Prefill
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = self.prefill(seq, cache)
        torch.cuda.synchronize()
        prefill_s = time.perf_counter() - t0

        # Decode
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        while True:
            token = self.sample(logits)
            seq.output_ids.append(token)
            if seq.is_finished(self.eos_ids):
                break
            logits = self.decode_step(cache, token)   # token sits at position seq.pos - 1
        torch.cuda.synchronize()
        decode_s = time.perf_counter() - t0

        text = self.tokenizer.decode(seq.output_ids, skip_special_tokens=True)

        if verbose:
            n_decode = len(seq.output_ids) - 1        # first token came from prefill
            print(f"prefill: {len(prompt_ids)} tok in {prefill_s * 1e3:.1f} ms")
            if n_decode > 0 and decode_s > 0:
                print(f"decode : {n_decode} tok in {decode_s * 1e3:.1f} ms "
                      f"({n_decode / decode_s:.1f} tok/s)")
        return text


if __name__ == "__main__":
    engine = PagedEngine()
    print(engine.generate("The capital of France is", max_tokens=32))
