#!/usr/bin/env python3
"""Small, untimed H100 correctness probes; never launches a tuning sweep."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "custom_kernels"), str(Path(__file__).parent)]


def main():
    import torch
    from grouped_splitk_validation import attention_reference, make_inputs
    from paged_flash_decode import flash_decode

    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("this diagnostic requires an H100")

    cases = [
        (1, False, 64, False, "uniform"),
        (16, False, 64, False, "uniform"),
        (17, False, 64, False, "uniform"),
        (64, False, 64, False, "uniform"),
        (16, False, 64, False, "random"),
        (17, False, 64, False, "random"),
        (64, False, 64, False, "random"),
        (65, False, 64, False, "random"),
        (129, False, 64, False, "random"),
        (1, True, 64, False, "uniform"),
        (17, True, 64, False, "random"),
        (1, False, 128, False, "uniform"),
        (129, False, 128, False, "random"),
        (129, False, 64, True, "random"),
    ]
    failed = False
    for context, register_pv, tile_n, overlap, mode in cases:
        q, k, v, table, lengths = make_inputs([context], seed=103 + context)
        # Zero Q/K gives exactly uniform scores. A failure here isolates the
        # softmax/PV path from the QK GEMM itself.
        if mode == "uniform":
            q.zero_()
            k.zero_()
        expected = attention_reference(q, k, v, table, lengths)
        actual = flash_decode(q, k, v, table, lengths, split_k=1,
                              tile_n=tile_n, register_pv=register_pv,
                              overlap_qk=overlap)
        torch.cuda.synchronize()
        maximum = (actual.float() - expected.float()).abs().max().item()
        close = bool(torch.allclose(actual, expected, atol=0.002, rtol=0.002))
        failed |= not close
        print(f"C={context:3} mode={mode:7} N={tile_n:3} register_pv={int(register_pv)} "
              f"overlap={int(overlap)} close={close} max_abs={maximum:.6f}", flush=True)
    if failed:
        raise SystemExit("Hopper diagnostic found numerical errors; do not benchmark")


if __name__ == "__main__":
    main()
