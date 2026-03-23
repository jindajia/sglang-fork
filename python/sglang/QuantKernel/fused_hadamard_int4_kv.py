"""
Fused Hadamard (FWHT) + int4 KV cache write (Triton).

Intended as a faster single-kernel substitute for the ``memory_pool`` int4 path that does
Hadamard then ``quantized_set_kv_int4_triton``. Scaling is ``input_bf16.to(fp32) / sqrt(order)``
then in-kernel Sylvester FWHT (same butterfly as ``fast_hadamard_transform`` for power-of-2
last dim), bf16 round-trip, then int4 pack. That scaling differs slightly from
``(bf16_tensor / sqrt(order))`` before CUDA Hadamard, so packed bytes can differ in rare
cases from the unfused path.

``hadamard_order`` must be a **power of two** with ``2 <= hadamard_order <= MAX_ORDER`` and
``head_dim % hadamard_order == 0`` (e.g. 8 … 512 and beyond, up to the compile limit below).

``MAX_HADAMARD_ORDER`` caps the FWHT segment size (``tl.arange(0, order)`` per program); raise
it if your Triton build successfully compiles larger blocks.

FWHT runs on in-kernel tensors (``tl.gather`` butterfly); no global scratch buffers.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.mem_cache.kv_quant_kernels import _quantized_set_kv_int4_kernel

# Upper bound for a single FWHT block (power-of-two only). Increase if your GPU/Triton version
# compiles larger ``tl.arange(0, head_dim_)`` in blocked FWHT.
MAX_HADAMARD_ORDER = 4096


def validate_hadamard_order_for_kv_fuse(hadamard_order: int, head_dim: int) -> None:
    """Raise ``ValueError`` if ``hadamard_order`` / ``head_dim`` are invalid for the fused kernel."""
    _validate_hadamard_order_impl(hadamard_order, head_dim)


def _validate_hadamard_order_impl(hadamard_order: int, head_dim: int) -> None:
    if hadamard_order < 2:
        raise ValueError(f"hadamard_order must be >= 2, got {hadamard_order}")
    if hadamard_order & (hadamard_order - 1):
        raise ValueError(
            f"hadamard_order must be a power of two, got {hadamard_order}"
        )
    if hadamard_order > MAX_HADAMARD_ORDER:
        raise ValueError(
            f"hadamard_order must be <= {MAX_HADAMARD_ORDER} (FWHT segment size), got {hadamard_order}"
        )
    if head_dim % hadamard_order:
        raise ValueError(
            f"head_dim ({head_dim}) must be divisible by hadamard_order ({hadamard_order})"
        )


@triton.jit
def _fwht_blocked_segments_tensor(x, head_dim_: tl.constexpr, LOG: tl.constexpr):
    """FWHT on each contiguous block of size ``2**LOG`` tiling ``head_dim_`` (vectorized).

    For ``p = g * 2**LOG + loc`` and ``stride = 2**s`` with ``s < LOG``,
    ``(p ^ stride) == g * 2**LOG + (loc ^ stride)``, so one global butterfly applies
    all segment transforms in parallel (no loop over groups).
    """
    i = tl.arange(0, head_dim_)
    for s in tl.static_range(0, LOG):
        stride = 1 << s
        partner = i ^ stride
        lo = tl.minimum(i, partner)
        hi = tl.maximum(i, partner)
        u0 = tl.gather(x, lo, 0)
        v0 = tl.gather(x, hi, 0)
        x = tl.where(i == lo, u0 + v0, u0 - v0)
    return x


def _make_fused_kernel(head_dim: int, hadamard_order: int):
    _validate_hadamard_order_impl(hadamard_order, head_dim)
    log_n = int(math.log2(hadamard_order))
    block_half = triton.next_power_of_2(head_dim // 2)
    pre_scale = 1.0 / math.sqrt(float(hadamard_order))

    @triton.jit
    def _fused_hadamard_int4_set_kv_kernel(
        input_ptr,
        loc_ptr,
        cache_ptr,
        scales_zeros_ptr,
        num_tokens,
        num_heads,
        head_dim_: tl.constexpr,
        input_stride_token,
        input_stride_head,
        input_stride_dim,
        cache_stride_loc,
        cache_stride_head,
        cache_stride_dim,
        sz_stride_loc,
        sz_stride_head,
        sz_stride_dim,
        LOG: tl.constexpr,
        PRE_SCALE: tl.constexpr,
        BLOCK_HALF: tl.constexpr,
    ):
        token_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        if token_idx >= num_tokens or head_idx >= num_heads:
            return

        cache_loc = tl.load(loc_ptr + token_idx)

        dim_full = tl.arange(0, head_dim_)
        input_off_base = (
            token_idx * input_stride_token + head_idx * input_stride_head
        )
        x_bf16 = tl.load(
            input_ptr + input_off_base + dim_full * input_stride_dim,
            mask=dim_full < head_dim_,
            other=0.0,
        ).to(tl.float32)
        x_scaled = x_bf16 * PRE_SCALE
        acc = _fwht_blocked_segments_tensor(x_scaled, head_dim_, LOG)

        half_dim = head_dim_ // 2
        dim_offsets = tl.arange(0, BLOCK_HALF)
        dim_mask = dim_offsets < half_dim
        safe_off1 = tl.where(dim_mask, dim_offsets, 0)
        safe_off2 = tl.where(dim_mask, dim_offsets + half_dim, 0)
        # Match separate hadamard_transform (bf16) + int4: rotate in fp32, then round to
        # bf16 before quant, same as storing Hadamard output and reloading in int4 kernel.
        vals1 = (
            tl.where(dim_mask, tl.gather(acc, safe_off1, 0), 0.0)
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        vals2 = (
            tl.where(dim_mask, tl.gather(acc, safe_off2, 0), 0.0)
            .to(tl.bfloat16)
            .to(tl.float32)
        )

        val_min_1 = tl.min(vals1, axis=0)
        val_min_2 = tl.min(vals2, axis=0)
        val_max_1 = tl.max(vals1, axis=0)
        val_max_2 = tl.max(vals2, axis=0)
        val_min = tl.minimum(val_min_1, val_min_2)
        val_max = tl.maximum(val_max_1, val_max_2)
        val_range = tl.maximum(val_max - val_min, 1e-8)
        scale = val_range / 15.0
        zero = -val_min / scale
        q_vals1 = (vals1 / scale + zero + 0.5).to(tl.uint8)
        q_vals2 = (vals2 / scale + zero + 0.5).to(tl.uint8)
        packed = q_vals1 | (q_vals2 << 4)
        cache_offset = (
            cache_loc * cache_stride_loc
            + head_idx * cache_stride_head
            + dim_offsets * cache_stride_dim
        )
        tl.store(cache_ptr + cache_offset, packed, mask=dim_mask)
        sz_offset_base = cache_loc * sz_stride_loc + head_idx * sz_stride_head
        tl.store(scales_zeros_ptr + sz_offset_base + 0 * sz_stride_dim, scale)
        tl.store(scales_zeros_ptr + sz_offset_base + 1 * sz_stride_dim, zero)

    return _fused_hadamard_int4_set_kv_kernel, {
        "head_dim_": head_dim,
        "LOG": log_n,
        "PRE_SCALE": pre_scale,
        "BLOCK_HALF": block_half,
    }


_KERNEL_CACHE: Dict[Tuple[int, int, int], Tuple] = {}
_KERNEL_REV = 10  # bump when JIT changes (invalidates cache)

def _get_kernel(head_dim: int, hadamard_order: int):
    k = (head_dim, hadamard_order, _KERNEL_REV)
    if k not in _KERNEL_CACHE:
        fn, cfg = _make_fused_kernel(head_dim, hadamard_order)
        _KERNEL_CACHE[k] = (fn, cfg)
    return _KERNEL_CACHE[k]


def quantized_set_kv_int4_hadamard_fused_triton(
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    loc: torch.Tensor,
    k_cache_buffer: torch.Tensor,
    v_cache_buffer: torch.Tensor,
    k_scales_zeros_buffer: torch.Tensor,
    v_scales_zeros_buffer: torch.Tensor,
    hadamard_order: int,
    work_k: torch.Tensor | None = None,
    work_v: torch.Tensor | None = None,
    rotate_v: bool = True,
) -> None:
    """
    Fused Hadamard along ``hadamard_order``-sized blocks + int4 pack.

    ``work_k`` / ``work_v`` are ignored (kept for backward compatibility; no scratch buffers).
    """
    _ = (work_k, work_v)
    num_tokens, num_heads, head_dim = cache_k.shape
    assert cache_v.shape == cache_k.shape
    assert head_dim % 2 == 0
    _validate_hadamard_order_impl(hadamard_order, head_dim)

    kernel, cfg = _get_kernel(head_dim, hadamard_order)
    fused_grid = (num_tokens, num_heads)

    kernel[fused_grid](
        cache_k,
        loc,
        k_cache_buffer,
        k_scales_zeros_buffer,
        num_tokens,
        num_heads,
        cfg["head_dim_"],
        cache_k.stride(0),
        cache_k.stride(1),
        cache_k.stride(2),
        k_cache_buffer.stride(0),
        k_cache_buffer.stride(1),
        k_cache_buffer.stride(2),
        k_scales_zeros_buffer.stride(0),
        k_scales_zeros_buffer.stride(1),
        k_scales_zeros_buffer.stride(2),
        LOG=cfg["LOG"],
        PRE_SCALE=cfg["PRE_SCALE"],
        BLOCK_HALF=cfg["BLOCK_HALF"],
        num_warps=1,
        num_stages=1,
    )

    if rotate_v:
        kernel[fused_grid](
            cache_v,
            loc,
            v_cache_buffer,
            v_scales_zeros_buffer,
            num_tokens,
            num_heads,
            cfg["head_dim_"],
            cache_v.stride(0),
            cache_v.stride(1),
            cache_v.stride(2),
            v_cache_buffer.stride(0),
            v_cache_buffer.stride(1),
            v_cache_buffer.stride(2),
            v_scales_zeros_buffer.stride(0),
            v_scales_zeros_buffer.stride(1),
            v_scales_zeros_buffer.stride(2),
            LOG=cfg["LOG"],
            PRE_SCALE=cfg["PRE_SCALE"],
            BLOCK_HALF=cfg["BLOCK_HALF"],
            num_warps=1,
            num_stages=1,
        )
    else:
        BLOCK_SIZE_DIM = triton.next_power_of_2(head_dim // 2)
        _quantized_set_kv_int4_kernel[fused_grid](
            cache_v,
            loc,
            v_cache_buffer,
            v_scales_zeros_buffer,
            num_tokens,
            num_heads,
            head_dim,
            cache_v.stride(0),
            cache_v.stride(1),
            cache_v.stride(2),
            v_cache_buffer.stride(0),
            v_cache_buffer.stride(1),
            v_cache_buffer.stride(2),
            v_scales_zeros_buffer.stride(0),
            v_scales_zeros_buffer.stride(1),
            v_scales_zeros_buffer.stride(2),
            BLOCK_SIZE_DIM=BLOCK_SIZE_DIM,
            num_warps=1,
            num_stages=1,
        )
