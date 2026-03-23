"""Quantization-related kernels (e.g. Hadamard + int4 KV fusion)."""

from sglang.QuantKernel.fused_hadamard_int4_kv import (
    MAX_HADAMARD_ORDER,
    quantized_set_kv_int4_hadamard_fused_triton,
    validate_hadamard_order_for_kv_fuse,
)

__all__ = [
    "MAX_HADAMARD_ORDER",
    "quantized_set_kv_int4_hadamard_fused_triton",
    "validate_hadamard_order_for_kv_fuse",
]
