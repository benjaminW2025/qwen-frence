# Custom kernels

Production Triton kernels used by the inference engine:

- `fused_rms.py`: fused RMSNorm over flattened token rows
- `rope.py`: Qwen rotate-half RoPE with explicit absolute positions
- `packed_prefill_attention.py`: packed variable-length causal attention
- `packed_paged_prefill_attention.py`: resumable packed attention over paged K/V

Experimental kernels used only by isolated intervention harnesses until their ship
criteria pass:

- `paged_decode_candidate.py`: alternate single/grouped-head paged decode layout
- `swiglu.py`: fused SiLU and gate product
- `rope_kv_write.py`: fused Q/K rotation and direct paged K/V placement

The inference engine loads these lazily through `baseline/kernel_dispatch.py`, so
the PyTorch baseline and CPU-only infrastructure do not require Triton imports.
Triton uses its normal cache configuration; this package does not set a cache path.
