"""
Naive autoregressive inference engine for Qwen2.5-1.5B.
  - KVCache            : allocation + write/read (contiguous, per-layer)
  - prefill / decode_step : the cache-aware forward (threading cache + positions)
  - sample             : sampling strategy
"""

import time
from dataclasses import dataclass, field

import torch
from transformers import AutoTokenizer
from kv_cache import KVCache

from naive_forward import Qwen2Config
from weight_loader import QwenWeightLoader

MODEL_ID = "Qwen/Qwen2.5-1.5B"


@dataclass
class Sequence:
    prompt_ids: list[int]
    output_ids: list[int] = field(default_factory=list)
    max_tokens: int = 64
    finished: bool = False

    @property
    def pos(self) -> int:
        # Absolute position of the NEXT token = number of tokens seen so far.
        return len(self.prompt_ids) + len(self.output_ids)

    def is_finished(self, eos_ids) -> bool:
        return (self.finished
                or len(self.output_ids) >= self.max_tokens
                or (self.output_ids and self.output_ids[-1] in eos_ids))

class BaselineEngine:
    def __init__(self, model_id=MODEL_ID, cfg=None, batch_size=1, device="cuda", dtype=torch.float16):
        self.cfg = cfg or Qwen2Config()
        self.device = device
        self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = QwenWeightLoader(self.cfg).load_pretrained(model_id, device, dtype)
        eos = self.tokenizer.eos_token_id
        self.eos_ids = {eos} if isinstance(eos, int) else set(eos or [])

        # Persistent KV pool: allocated once (sized to cfg.max_seq_len), reset per request.
        self.cache = KVCache(self.cfg, batch_size, self.device, self.dtype)

    def prefill(self, seq, cache):
        # Wrap into PyTorch tensor
        input_ids = torch.tensor([seq.prompt_ids], device=self.device)
        # Cache aware forward implemented in Model class
        prefill_logits = self.model.forward(input_ids, cache)
        return prefill_logits[0, -1]

    def decode_step(self, cache, token):
        input_ids = torch.tensor([[token]], device=self.device)
        # Cache aware forward implemented in Model class
        decode_logits = self.model.forward(input_ids, cache)
        return decode_logits[0, -1]

    def sample(self, logits):
        """Greedy by default. TODO (you): temperature / top-k / top-p."""
        return int(logits.argmax(dim=-1).item())

    # ------------------------------------------------------------------ #
    # Orchestration (provided boilerplate).                              #
    # MAKE SURE THAT DECODE IS PROPERLY HANDLING THE FACT THE FIRST      #
    # IS ALREADY PREDICTED                                               #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def generate(self, prompt, max_tokens=64, verbose=True):
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
        seq = Sequence(prompt_ids=prompt_ids, max_tokens=max_tokens)

        # Reuse the persistent pool for this request (clears cur_len to 0).
        self.cache.reset()
        cache = self.cache

        # --- Prefill: prompt -> logits for the first generated token ---
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = self.prefill(seq, cache)
        torch.cuda.synchronize()
        prefill_s = time.perf_counter() - t0

        # --- Decode: sample -> append -> feed back, one token per step ---
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        while True:
            token = self.sample(logits)
            seq.output_ids.append(token)
            if seq.is_finished(self.eos_ids):
                break
            logits = self.decode_step(cache, token)  # token is at position seq.pos - 1
        torch.cuda.synchronize()
        decode_s = time.perf_counter() - t0

        text = self.tokenizer.decode(seq.output_ids, skip_special_tokens=True)

        if verbose:
            n_decode = len(seq.output_ids) - 1  # first token came from prefill
            print(f"prefill: {len(prompt_ids)} tok in {prefill_s * 1e3:.1f} ms")
            if n_decode > 0 and decode_s > 0:
                print(f"decode : {n_decode} tok in {decode_s * 1e3:.1f} ms "
                      f"({n_decode / decode_s:.1f} tok/s)")
        return text


if __name__ == "__main__":
    engine = BaselineEngine()
    print(engine.generate("The capital of France is", max_tokens=32))
