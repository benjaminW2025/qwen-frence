"""Write new decode vectors to FP8 storage and stage the pool for FP16 FA3."""
import triton
import triton.language as tl


@triton.jit
def _write(X, SLOTS, Q, S, HEADS: tl.constexpr, DIM: tl.constexpr,
           BLOCK: tl.constexpr):
    request, head = tl.program_id(0), tl.program_id(1)
    slot = tl.load(SLOTS + request)
    row = slot * HEADS + head
    col = tl.arange(0, BLOCK)
    value = tl.load(X + row * DIM + col, col < DIM, 0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(value), 0) / 448., 1e-12)
    quantized = tl.minimum(tl.maximum(value / scale, -448.), 448.)
    tl.store(Q + row * DIM + col, quantized, col < DIM)
    tl.store(S + row, scale)


@triton.jit
def _stage(Q, S, X, ELEMENTS: tl.constexpr, DIM: tl.constexpr,
           BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(Q + index, index < ELEMENTS, 0).to(tl.float32)
    scale = tl.load(S + index // DIM, index < ELEMENTS, 0)
    tl.store(X + index, value * scale, index < ELEMENTS)


def update_and_stage(workspace, slots, quantized, scales):
    heads, dim = workspace.shape[-2:]
    _write[(slots.numel(), heads)](workspace, slots, quantized, scales,
                                   heads, dim, triton.next_power_of_2(dim))
    _stage[(triton.cdiv(workspace.numel(), 256),)](
        quantized, scales, workspace, workspace.numel(), dim, 256)


@triton.jit
def _write_fixed_scale(X, SLOTS, Q, S, HEADS: tl.constexpr, DIM: tl.constexpr,
                       BLOCK: tl.constexpr):
    request, head = tl.program_id(0), tl.program_id(1)
    slot = tl.load(SLOTS + request)
    scale = tl.load(S + request * HEADS + head)
    features = tl.arange(0, BLOCK)
    offset = (slot * HEADS + head) * DIM + features
    value = tl.load(X + offset, features < DIM, 0).to(tl.float32)
    encoded = tl.minimum(tl.maximum(value / scale, -448.), 448.)
    tl.store(Q + offset, encoded, features < DIM)


def update_fp8_only(workspace, slots, quantized, scales):
    """Write only new KV vectors; native FA3 reads the FP8 pool directly."""
    heads, dim = workspace.shape[-2:]
    _write_fixed_scale[(slots.numel(), heads)](
        workspace, slots, quantized, scales, heads, dim, triton.next_power_of_2(dim))
