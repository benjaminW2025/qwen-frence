"""Focused segment-layout check for the packed QKV post-GEMM epilogue."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[2]
KERNEL_DIR = ROOT / "custom_kernels"
sys.path.insert(0, str(KERNEL_DIR))
HAS_TRITON = importlib.util.find_spec("triton") is not None


@unittest.skipUnless(HAS_TRITON and torch.cuda.is_available(), "requires CUDA and Triton")
class PackedQKVEpilogueTests(unittest.TestCase):
    def test_q_k_and_v_segments_restart_their_head_indices(self):
        from packed_qkv_rope_cache import packed_qkv_rope_cache

        # A zero position makes RoPE the identity, so every destination must
        # equal its corresponding packed 128-column segment exactly.
        packed = torch.arange(2048, device="cuda", dtype=torch.float32).to(torch.float16)
        packed = packed.view(1, 2048)
        positions = torch.zeros(1, device="cuda", dtype=torch.int32)
        slots = torch.tensor([3], device="cuda", dtype=torch.long)
        k_pool = torch.full((1, 16, 2, 128), -1, device="cuda", dtype=torch.float16)
        v_pool = torch.full_like(k_pool, -2)

        q = packed_qkv_rope_cache(packed, positions, slots, k_pool, v_pool)
        torch.cuda.synchronize()

        segments = packed.view(16, 128)
        self.assertTrue(torch.equal(q[0], segments[:12]))
        self.assertTrue(torch.equal(k_pool.view(-1, 2, 128)[3], segments[12:14]))
        self.assertTrue(torch.equal(v_pool.view(-1, 2, 128)[3], segments[14:16]))

    def test_bucket_padding_never_writes_kv_and_returns_zero_queries(self):
        from packed_qkv_rope_cache import packed_qkv_rope_cache

        packed = torch.randn(3, 2048, device="cuda", dtype=torch.float16)
        positions = torch.zeros(3, device="cuda", dtype=torch.int32)
        slots = torch.tensor([3, 4, 0], device="cuda", dtype=torch.long)
        valid_tokens = torch.tensor(2, device="cuda", dtype=torch.int32)
        k_pool = torch.full((1, 16, 2, 128), -1, device="cuda", dtype=torch.float16)
        v_pool = torch.full_like(k_pool, -2)

        q = packed_qkv_rope_cache(
            packed, positions, slots, k_pool, v_pool, valid_tokens=valid_tokens,
        )
        torch.cuda.synchronize()

        segments = packed.view(3, 16, 128)
        self.assertTrue(torch.equal(q[:2], segments[:2, :12]))
        self.assertTrue(torch.equal(q[2], torch.zeros_like(q[2])))
        self.assertTrue(torch.equal(k_pool.view(-1, 2, 128)[3:5],
                                    segments[:2, 12:14]))
        self.assertTrue(torch.equal(v_pool.view(-1, 2, 128)[3:5],
                                    segments[:2, 14:16]))
        self.assertTrue(torch.equal(k_pool.view(-1, 2, 128)[0],
                                    torch.full_like(k_pool[0, 0], -1)))
        self.assertTrue(torch.equal(v_pool.view(-1, 2, 128)[0],
                                    torch.full_like(v_pool[0, 0], -2)))


if __name__ == "__main__":
    unittest.main()
