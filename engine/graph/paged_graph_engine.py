"""
Paged engine with a CUDA-graph'd decode loop. Key difference from
the normal paged engine is that we perform a singular CUDA graph
capture and then loop decode on that.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "baseline"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "kvcache"))

import time

import torch

from paged_engine import PagedEngine, Sequence
from paged_graph_decoder import CUDAGraphDecoder, build_decode_step_inputs


class PagedGraphEngine(PagedEngine):

    @torch.no_grad()
    def generate(self, prompt, max_tokens=64, verbose=True):
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
        seq = Sequence(prompt_ids=prompt_ids, max_tokens=max_tokens)

        total = len(prompt_ids) + max_tokens
        cache = self._new_cache(total)
        # Fixed block-table width baked into the graph = the hard length cap for this run.
        max_blocks = (total + self.block_size - 1) // self.block_size

        # Prefill
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = self.prefill(seq, cache)
        torch.cuda.synchronize()
        prefill_s = time.perf_counter() - t0

        # Build + capture the decode graph
        decoder = CUDAGraphDecoder(self.model, cache, batch_size=1,
                                      max_blocks=max_blocks, device=self.device, dtype=self.dtype)
        decoder.capture() # Capture the forward pass

        # Graphed decode loop
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        while True:
            token = self.sample(logits)
            seq.output_ids.append(token)
            if seq.is_finished(self.eos_ids):
                break

            cache.allocate_block([1]) # Start with B=1 case
            inputs = build_decode_step_inputs(cache, [token], max_blocks, device=self.device)
            logits = decoder.decode(*inputs)
            logits = logits[0, -1]

        torch.cuda.synchronize()
        decode_s = time.perf_counter() - t0

        text = self.tokenizer.decode(seq.output_ids, skip_special_tokens=True)

        if verbose:
            n_decode = len(seq.output_ids) - 1
            print(f"prefill: {len(prompt_ids)} tok in {prefill_s * 1e3:.1f} ms")
            if n_decode > 0 and decode_s > 0:
                print(f"decode : {n_decode} tok in {decode_s * 1e3:.1f} ms "
                      f"({n_decode / decode_s:.1f} tok/s)  [CUDA graph]")
        return text


if __name__ == "__main__":
    engine = PagedGraphEngine()
    print(engine.generate("The capital of France is", max_tokens=32))
