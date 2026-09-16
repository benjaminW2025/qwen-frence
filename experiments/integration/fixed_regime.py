"""Frozen input-shape table for the C++/graph and vLLM checkpoint.

Do not tune these rows from benchmark results. Changing a row creates a new
regime revision, not an in-place reinterpretation of previous measurements.
"""

from __future__ import annotations

SHORT_PROMPT_LENGTH = 256
LONG_PROMPT_LENGTH = 2048
SHORT_OUTPUT_LENGTH = 128
LONG_OUTPUT_LENGTH = 256
PREFILL_TOKENS_PER_STEP = 2048
MIN_FULL_DECODE_STEPS = 64
PAGE_SIZE = 16
# Mirrors engine/kvcache/paged_decode_attention.py without importing Triton in
# CPU-only checkpoint analysis. Recheck this contract if the production policy moves.
SPLITK_MIN_CONTEXT = 1024

FACTORIAL_SHAPES = tuple(
    {"id": f"fixed-b{batch}-l{prompt}-o{output}", "batch": batch,
     "prompt_length": prompt, "output_length": output,
     "role": "factorial",
     "expected_full_decode_steps": output - batch * prompt // PREFILL_TOKENS_PER_STEP}
    for batch in (8, 64)
    for prompt in (SHORT_PROMPT_LENGTH, LONG_PROMPT_LENGTH)
    for output in (SHORT_OUTPUT_LENGTH, LONG_OUTPUT_LENGTH)
)
CONTEXT_PROBES = (
    {"id": "probe-b8-l4096-o256", "batch": 8, "prompt_length": 4096,
     "output_length": 256, "role": "longer_context_probe",
     "expected_full_decode_steps": 241},
    {"id": "probe-b64-l4096-o256", "batch": 64, "prompt_length": 4096,
     "output_length": 256, "role": "longer_context_probe",
     "expected_full_decode_steps": 129},
)
FIXED_SHAPES = FACTORIAL_SHAPES + CONTEXT_PROBES


def fixed_plan():
    rows = []
    for shape in FIXED_SHAPES:
        batch = shape["batch"]
        rows.append({
            "id": shape["id"], "kind": "fixed-uniform",
            "lengths": [shape["prompt_length"]] * batch,
            "arrivals": [0] * batch,
            "outputs": [shape["output_length"]] * batch,
            "max_running": batch,
            "prefill_budget": PREFILL_TOKENS_PER_STEP,
        })
    return rows


def get_fixed_shape(shape_id):
    matches = [row for row in FIXED_SHAPES if row["id"] == shape_id]
    if len(matches) != 1:
        raise ValueError("shape ID is not in the frozen input table")
    return matches[0]


def get_fixed_case(shape_id):
    return next(row for row in fixed_plan() if row["id"] == get_fixed_shape(shape_id)["id"])


def verify_fixed_result(result, shape_id):
    """Require the actual scheduler work to match a table row before timing."""
    shape = get_fixed_shape(shape_id)
    batches = [call[1] for step in result["steps"] for call in step["calls"] if call[0]]
    prefill = [call[1] for step in result["steps"] for call in step["calls"] if not call[0]]
    full_steps = sum(step["kind"] == "decode" and
                     any(call[0] and call[1] == shape["batch"] for call in step["calls"])
                     for step in result["steps"])
    if max(batches, default=0) != shape["batch"]:
        raise AssertionError(f"{shape_id}: actual max decode batch "
                             f"{max(batches, default=0)} != {shape['batch']}")
    if PREFILL_TOKENS_PER_STEP not in prefill:
        raise AssertionError(f"{shape_id}: no {PREFILL_TOKENS_PER_STEP}-token packed prefill")
    if full_steps != shape["expected_full_decode_steps"] or full_steps < MIN_FULL_DECODE_STEPS:
        raise AssertionError(f"{shape_id}: {full_steps} pure full-batch decode steps != "
                             f"frozen {shape['expected_full_decode_steps']}")
    return {"max_actual_decode_batch": max(batches),
            "max_packed_prefill_tokens": max(prefill),
            "pure_full_decode_steps": full_steps}


def shape_summary(shape):
    batch = shape["batch"]
    prompt = shape["prompt_length"]
    output = shape["output_length"]
    context = prompt + output
    return {"id": shape["id"], "role": shape["role"], "batch": batch,
            "prompt_tokens_per_request": prompt,
            "output_tokens_per_request": output,
            "max_context_tokens_per_request": context,
            "total_prompt_tokens": batch * prompt,
            "total_output_tokens": batch * output,
            "total_request_tokens": batch * context,
            "max_packed_prefill_tokens_per_step": PREFILL_TOKENS_PER_STEP,
            "graph_prefill_bucket": PREFILL_TOKENS_PER_STEP,
            "logical_kv_blocks_with_headroom": batch * ((context + PAGE_SIZE - 1) // PAGE_SIZE + 1),
            "expected_cpu_dry_full_decode_steps": shape["expected_full_decode_steps"],
            "minimum_accepted_full_decode_steps": MIN_FULL_DECODE_STEPS}
