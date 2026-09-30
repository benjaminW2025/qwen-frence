"""Retained runtime fusions from the completed FA3 engine measurements.

Keep this separate from candidate selection: unfinished experiments must not
silently change the accepted engine. Attention and graph shapes are configured
by the caller; the accepted measured path uses external FA3 and exact buckets.
"""


def retained_fusion_options():
    """Fresh constructor kwargs shared by burst, mixed and phase execution."""
    return {
        "enable_residual_rmsnorm": True,
        "enable_native_decode_qkv_postprocess": True,
        "enable_prefill_swiglu_fusion": True,
    }


def accepted_variant_options():
    """Freeze the established baseline; no promotion from incomplete runs."""
    return {
        "enable_prefill_shared_graph_pool": False,
        "enable_prefill_boundary_buffer_reuse": False,
        "enable_stable_decode_table_cache": False,
        "enable_prefill_fused_gemm_epilogues": False,
        "enable_decode_fused_gemm_epilogues": False,
        "output_head_policy": "logits",
    }
