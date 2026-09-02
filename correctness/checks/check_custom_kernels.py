"""Correctness sweep for the integrated Triton RMSNorm and Qwen RoPE paths."""

import _bootstrap  # noqa: F401

import torch

from kernel_dispatch import rms_norm, rope, rope_kv_write, swiglu
from kv_cache import KVCache
from naive_forward import Qwen2Config
from weight_loader import QwenWeightLoader


MODEL_ID = "Qwen/Qwen2.5-1.5B"
DEVICE = "cuda"
DTYPE = torch.float16
KERNEL_TOL = 3e-2
MODEL_TOL = 1e-1


def check_rms_norm():
    print("isolated Triton RMSNorm parity...")
    torch.manual_seed(0)
    ok = True
    for rows in (1, 3, 17, 128):
        x = torch.randn(rows, 1536, device=DEVICE, dtype=DTYPE)
        weight = torch.randn(1536, device=DEVICE, dtype=DTYPE)
        out = rms_norm(x, weight, 1e-6)
        ref = torch.nn.functional.rms_norm(x, (1536,), weight, 1e-6)
        error = (out.float() - ref.float()).abs().max().item()
        passed = bool(torch.isfinite(out).all()) and error <= KERNEL_TOL
        ok &= passed
        print(f"  rows={rows:<3}: max error={error:.6f}  {'PASS' if passed else 'FAIL'}")
    return ok


def _rope_reference(x, positions, theta):
    half = x.shape[-1] // 2
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, x.shape[-1], 2, device=x.device, dtype=torch.float32)
                  / x.shape[-1])
    )
    freqs = positions.float()[:, :, None] * inv_freq[None, None]
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos()[:, None].to(x.dtype), emb.sin()[:, None].to(x.dtype)
    first, second = x[..., :half], x[..., half:]
    return x * cos + torch.cat((-second, first), dim=-1) * sin


def check_rope():
    print("isolated Triton Qwen RoPE parity...")
    torch.manual_seed(1)
    cases = (
        (1, 12, torch.tensor([[0]], device=DEVICE, dtype=torch.int32)),
        (3, 2, torch.tensor([[1], [16], [32767]], device=DEVICE, dtype=torch.int32)),
        (2, 12, torch.tensor(
            [[0, 1, 2, 3, 4, 5, 6], [127, 128, 129, 130, 131, 132, 133]],
            device=DEVICE, dtype=torch.int32,
        )),
        # Regression for packed batches whose flattened token dimension crosses
        # CUDA's 65,535 grid-Y/grid-Z limit.
        (1, 1, torch.arange(65536, device=DEVICE, dtype=torch.int32).unsqueeze(0)),
    )
    ok = True
    for batch, heads, positions in cases:
        seq_len = positions.shape[1]
        # Matches the non-contiguous head layout produced by q/k projection.
        x = torch.randn(batch, seq_len, heads, 128, device=DEVICE, dtype=DTYPE).transpose(1, 2)
        out = rope(x, positions, 1_000_000.0)
        ref = _rope_reference(x, positions, 1_000_000.0)
        error = (out.float() - ref.float()).abs().max().item()
        passed = bool(torch.isfinite(out).all()) and error <= KERNEL_TOL
        ok &= passed
        print(
            f"  shape={tuple(x.shape)}, positions=[{positions.min().item()},"
            f" {positions.max().item()}]: max error={error:.6f}  "
            f"{'PASS' if passed else 'FAIL'}"
        )
    return ok


def check_swiglu():
    print("isolated fused SwiGLU parity...")
    torch.manual_seed(3)
    ok = True
    for rows in (64, 1409, 2048):
        gate = torch.randn(rows, 8960, device=DEVICE, dtype=DTYPE)
        up = torch.randn_like(gate)
        out = swiglu(gate, up, block_size=512, num_warps=4, num_stages=2)
        ref = torch.nn.functional.silu(gate.float()) * up.float()
        error = (out.float() - ref).abs().max().item()
        passed = bool(torch.isfinite(out).all()) and error <= KERNEL_TOL
        ok &= passed
        print(f"  rows={rows:<4}: max error={error:.6f}  {'PASS' if passed else 'FAIL'}")
    return ok


def check_rope_kv_write():
    print("isolated fused RoPE plus paged-KV placement parity...")
    torch.manual_seed(4)
    tokens, page_size, n_heads, n_kv_heads, d_head = 129, 16, 12, 2, 128
    q = torch.randn(1, tokens, n_heads, d_head, device=DEVICE, dtype=DTYPE).transpose(1, 2)
    k = torch.randn(1, tokens, n_kv_heads, d_head, device=DEVICE, dtype=DTYPE).transpose(1, 2)
    v = torch.randn_like(k)
    positions = torch.arange(4096, 4096 + tokens, device=DEVICE, dtype=torch.int32)[None]
    num_slots = ((tokens + page_size - 1) // page_size + 2) * page_size
    slot_mapping = torch.randperm(num_slots, device=DEVICE)[:tokens]
    k_pool = torch.full(
        (num_slots // page_size, page_size, n_kv_heads, d_head),
        float("nan"), device=DEVICE, dtype=DTYPE,
    )
    v_pool = torch.full_like(k_pool, float("nan"))

    q_out = rope_kv_write(
        q, k, v, positions, slot_mapping, k_pool, v_pool,
        base=1_000_000.0, num_warps=2, num_stages=2,
    )
    q_ref = _rope_reference(q, positions, 1_000_000.0)
    k_ref = _rope_reference(k, positions, 1_000_000.0)[0].transpose(0, 1)
    v_ref = v[0].transpose(0, 1)
    flat_k = k_pool.view(-1, n_kv_heads, d_head).index_select(0, slot_mapping)
    flat_v = v_pool.view(-1, n_kv_heads, d_head).index_select(0, slot_mapping)
    q_error = (q_out.float() - q_ref.float()).abs().max().item()
    k_error = (flat_k.float() - k_ref.float()).abs().max().item()
    v_error = (flat_v.float() - v_ref.float()).abs().max().item()
    passed = (
        bool(torch.isfinite(q_out).all())
        and bool(torch.isfinite(flat_k).all())
        and bool(torch.isfinite(flat_v).all())
        and max(q_error, k_error) <= KERNEL_TOL
        and v_error == 0.0
    )
    print(
        f"  tokens={tokens}: q error={q_error:.6f}, k error={k_error:.6f}, "
        f"v error={v_error:.6f}  {'PASS' if passed else 'FAIL'}"
    )
    return passed


@torch.no_grad()
def check_full_model():
    print("full-model PyTorch versus custom-kernel prefill/decode parity...")
    cfg = Qwen2Config()
    model = QwenWeightLoader(cfg).load_pretrained(MODEL_ID, DEVICE, DTYPE)
    generator = torch.Generator(device=DEVICE).manual_seed(2)
    prompt = torch.randint(0, cfg.vocab, (1, 17), generator=generator, device=DEVICE)
    decode_token = torch.randint(0, cfg.vocab, (1, 1), generator=generator, device=DEVICE)

    def run(use_custom):
        cfg.use_custom_kernels = use_custom
        cache = KVCache(cfg, 1, DEVICE, DTYPE, max_seq_len=18)
        return model(prompt, cache), model(decode_token, cache)

    reference = run(False)
    custom = run(True)
    ok = True
    for label, out, ref in zip(("prefill", "decode"), custom, reference):
        error = (out.float() - ref.float()).abs().max().item()
        top1 = int(out[0, -1].argmax()) == int(ref[0, -1].argmax())
        passed = bool(torch.isfinite(out).all()) and error <= MODEL_TOL and top1
        ok &= passed
        print(
            f"  {label:<7}: max logit error={error:.6f}, "
            f"top-1={'match' if top1 else 'MISMATCH'}  {'PASS' if passed else 'FAIL'}"
        )
    return ok


def main():
    ok = check_rms_norm()
    ok = check_rope() and ok
    ok = check_swiglu() and ok
    ok = check_rope_kv_write() and ok
    ok = check_full_model() and ok
    print("OVERALL:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
