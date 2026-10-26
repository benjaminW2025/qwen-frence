#!/usr/bin/env python3
"""Capture post-RoPE Q/K/V from a real FP16 Qwen forward without FA3.

Uses the repository's PyTorch baseline and SDPA, with Triton disabled. Supply
real text via --text-file or a saved int64 [B,L] tensor via --token-ids. These
teacher-forced captures can probe final single-token or chunk queries against
the same full history; they do not model accumulated quantization error.
"""
import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "baseline", ROOT / "benchmarks", ROOT / "experiments/integration"):
    sys.path.insert(0, str(directory))


def rotate_queries(projected, cfg, length):
    from naive_forward import apply_rotary_pos_emb
    q = projected[..., :cfg.n_heads * cfg.d_head]
    q = q.reshape(projected.shape[0], length, cfg.n_heads, cfg.d_head).transpose(1, 2)
    pos = torch.arange(length, device=q.device, dtype=torch.float32)
    inv = 1. / cfg.rope_theta ** (torch.arange(0, cfg.d_head, 2, device=q.device).float() / cfg.d_head)
    angles = torch.outer(pos, inv)
    angles = torch.cat((angles, angles), -1)
    return apply_rotary_pos_emb(q, angles.cos()[None, None].to(q.dtype),
                               angles.sin()[None, None].to(q.dtype))


@torch.no_grad()
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--device", default="cuda:0")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text-file", type=Path)
    source.add_argument("--token-ids", type=Path)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--queries", nargs="+", type=int, default=[1, 32])
    parser.add_argument("--layers", nargs="+", type=int, default=[0, 7, 14, 21, 27])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("use a new capture path")
    if not 1 <= args.max_tokens <= 32768 or min(args.queries) < 1 or any(not 0 <= i < 28 for i in args.layers):
        parser.error("invalid context, query lengths or layers")
    from benchmark_latest_vs_vllm import resolve_model_source
    from model_setup import prepare_hub_transfer
    prepare_hub_transfer()
    from naive_forward import Qwen2Config
    from weight_loader import QwenWeightLoader
    from kv_cache import KVCache
    from experiments.quant.projections import strict_accumulation
    model_source = resolve_model_source(args)
    if args.token_ids:
        ids = torch.load(args.token_ids, map_location="cpu", weights_only=True)
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=True)
        ids = tokenizer(args.text_file.read_text(), return_tensors="pt", truncation=True,
                        max_length=args.max_tokens).input_ids
    if not isinstance(ids, torch.Tensor) or ids.dtype != torch.int64 or ids.ndim != 2 or not ids.numel():
        raise ValueError("token IDs must be a nonempty int64 [B,L] tensor")
    ids = ids[:, :args.max_tokens].to(args.device)
    cfg = Qwen2Config(use_custom_kernels=False)
    if int(ids.min()) < 0 or int(ids.max()) >= cfg.vocab:
        raise ValueError("token ID outside model vocabulary")
    with strict_accumulation():
        model = QwenWeightLoader(cfg).load_pretrained(model_source, args.device, torch.float16)
        cache = KVCache(cfg, ids.shape[0], args.device, torch.float16, max_seq_len=ids.shape[1])
        queries, hooks = {}, []
        for layer in sorted(set(args.layers)):
            def save(_module, _inputs, output, index=layer):
                queries[index] = rotate_queries(output, cfg, ids.shape[1]).cpu()
            hooks.append(model.layers[layer].qkv_proj.register_forward_hook(save))
        try:
            model(ids, cache)
        finally:
            for hook in hooks:
                hook.remove()
        captures = []
        for layer in sorted(queries):
            k, v = cache.k_cache[layer].cpu(), cache.v_cache[layer].cpu()
            for count in args.queries:
                if count <= ids.shape[1]:
                    captures.append({"layer": layer, "model_source": model_source,
                                     "input_ids": ids.cpu(), "q": queries[layer][:, :, -count:].contiguous(),
                                     "k": k, "v": v})
        if not captures:
            raise ValueError("no query lengths fit the supplied text")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(captures, args.output)
        print(f"Saved {len(captures)} probes from {ids.shape[1]} real input tokens to {args.output}")


if __name__ == "__main__":
    main()
