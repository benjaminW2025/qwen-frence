"""Sweep the paged decode-attention Triton kernel across cache boundaries."""

import math
import _bootstrap  # noqa: F401

import torch

from paged_decode_attention import check_correctness


LOGIT_TOL = 5e-2
CASES = (
    # block size, exact ragged lengths. These deliberately straddle boundaries.
    (16, [1]),
    (16, [15, 16]),
    (16, [16, 17]),
    (16, [1, 15, 16, 17]),
    (16, [31, 32, 33, 100]),
    (32, [31, 32, 33, 129]),
)


def main():
    assert torch.cuda.is_available(), "paged attention correctness requires an NVIDIA GPU"
    errors = []
    for block_size, seq_lens in CASES:
        print("\n" + "=" * 72)
        print(f"batch={len(seq_lens)} block_size={block_size} seq_lens={seq_lens}")
        error = check_correctness(
            n_heads=12,
            n_kv_heads=2,
            d_head=128,
            block_size=block_size,
            dtype=torch.float16,
            seq_lens_override=seq_lens,
        )
        errors.append(error)

    max_error = max(errors)
    ok = math.isfinite(max_error) and max_error <= LOGIT_TOL
    print("\n" + "=" * 72)
    print(f"maximum error: {max_error:.6f} (tolerance {LOGIT_TOL})")
    print("OVERALL:", "PASS ✅" if ok else "FAIL ❌")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
