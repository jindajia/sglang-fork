# Copyright (c) OpenMMLab. All rights reserved.
# Adapted from LMDeploy for SGLang
"""
Efficient flatten + dequantize KV cache kernel.

This kernel flattens paged KV cache to contiguous memory while dequantizing
only the tokens that are actually used (based on page table and sequence lengths).

Key benefits:
- Only dequantizes tokens referenced in page_table (not entire buffer)
- Produces contiguous output suitable for flash_attn_varlen_func
- Combines scatter-gather + dequantization in one kernel
"""

from typing import Literal, Tuple, Union

import torch
import triton
import triton.language as tl

# Removed _dequant_int4 - INT4 unpacking is now done inline following decode_attention.py logic

# FP4 E2M1 values lookup table (matches kvfp4_tensor.py)
E2M1_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


@triton.jit
def _flatten_kv_cache_quant_sglang_fp4(
    # Input pointers
    kc_ptr,  # Quantized K cache [total_slots, num_heads, head_dim//2] (packed fp4)
    vc_ptr,  # Quantized V cache
    ksf_ptr,  # K scale factors [total_slots, (num_heads * head_dim) // 16] (uint8)
    vsf_ptr,  # V scale factors
    page_table_ptr,  # Page table [batch_size, max_slots_per_seq] - contains slot indices
    cache_seqlens_ptr,  # Valid sequence lengths [batch_size]
    cu_seqlens_k_ptr,  # Cumulative seq lengths [batch_size + 1]
    # Output pointers
    ko_ptr,  # Flattened dequantized K output [total_tokens, num_heads, head_dim]
    vo_ptr,  # Flattened dequantized V output
    # Strides for K cache (input) - 3D [slots, heads, dim//2]
    stride_kc_slot: tl.constexpr,
    stride_kch: tl.constexpr,
    stride_kcd: tl.constexpr,
    # Strides for V cache (input)
    stride_vc_slot: tl.constexpr,
    stride_vch: tl.constexpr,
    stride_vcd: tl.constexpr,
    # Strides for scale factors - 2D [slots, (num_heads * head_dim) // 16]
    stride_ksf_slot: tl.constexpr,
    stride_ksf_dim: tl.constexpr,
    stride_vsf_slot: tl.constexpr,
    stride_vsf_dim: tl.constexpr,
    # Strides for output
    stride_kos: tl.constexpr,
    stride_koh: tl.constexpr,
    stride_kod: tl.constexpr,
    stride_vos: tl.constexpr,
    stride_voh: tl.constexpr,
    stride_vod: tl.constexpr,
    # Page table stride
    stride_page_table,
    # Config
    TOTAL_SLOTS,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM_K: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_DK: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    SCALE_BLOCK_SIZE: tl.constexpr,  # 16 for fp4
):
    """
    Flatten slot-based KV cache with on-demand FP4 dequantization for SGLang.
    
    FP4 format:
    - Packed storage: 2 fp4 values per uint8 byte
    - Block-based scaling: 16 values per block, scale stored as uint8 (scale_exp + 127)
    - E2M1 format: 1 sign bit, 2 exponent bits, 1 mantissa bit
    """
    block_id = tl.program_id(0)
    batch_id = tl.program_id(1)
    head_id = tl.program_id(2)

    seqlen = tl.load(cache_seqlens_ptr + batch_id)
    start_loc = tl.load(cu_seqlens_k_ptr + batch_id)

    if block_id * BLOCK_SIZE >= seqlen:
        return

    # Token offsets within this block
    offs_tok = tl.arange(0, BLOCK_SIZE)
    
    # FP4 is packed: head_dim stored as head_dim // 2
    BLOCK_DK_PACKED: tl.constexpr = BLOCK_DK // 2
    BLOCK_DV_PACKED: tl.constexpr = BLOCK_DV // 2
    HALF_HDK: tl.constexpr = HEAD_DIM_K // 2
    HALF_HDV: tl.constexpr = HEAD_DIM_V // 2
    
    offs_dk = tl.arange(0, BLOCK_DK_PACKED)
    offs_dv = tl.arange(0, BLOCK_DV_PACKED)
    mask_dk_packed = offs_dk < HALF_HDK
    mask_dv_packed = offs_dv < HALF_HDV

    # Output token offsets
    offs_out_tok = block_id * BLOCK_SIZE + offs_tok
    mask_tok = offs_out_tok < seqlen

    # Full dimension offsets for output
    offs_dok = tl.arange(0, BLOCK_DK)
    offs_dov = tl.arange(0, BLOCK_DV)
    mask_dok = offs_dok < HEAD_DIM_K
    mask_dov = offs_dov < HEAD_DIM_V

    # Get slot indices
    global_tok_positions = block_id * BLOCK_SIZE + offs_tok
    page_indices_in_seq = global_tok_positions // PAGE_SIZE
    offsets_within_page = global_tok_positions % PAGE_SIZE

    page_table_ptrs = (
        page_table_ptr + batch_id * stride_page_table + page_indices_in_seq
    )
    page_indices = tl.load(page_table_ptrs, mask=mask_tok, other=0)
    slot_indices = page_indices * PAGE_SIZE + offsets_within_page
    slot_indices = tl.where(slot_indices < TOTAL_SLOTS, slot_indices, 0)
    slot_indices = tl.where(slot_indices >= 0, slot_indices, 0)

    out_positions = start_loc + offs_out_tok

    # ==== Process K cache ====
    # Load packed fp4 values
    kc_ptrs = (
        kc_ptr
        + slot_indices[:, None] * stride_kc_slot
        + head_id * stride_kch
        + offs_dk[None, :] * stride_kcd
    )
    kc_packed = tl.load(
        kc_ptrs, mask=mask_tok[:, None] & mask_dk_packed[None, :], other=0
    )

    # Unpack lower and upper 4 bits
    k_lower_fp4 = kc_packed & 0x0F
    k_upper_fp4 = (kc_packed >> 4) & 0x0F

    # Extract sign and magnitude for lower half
    k_lower_sign = (k_lower_fp4 & 0x08) != 0
    k_lower_mag = k_lower_fp4 & 0x07
    
    # Extract sign and magnitude for upper half
    k_upper_sign = (k_upper_fp4 & 0x08) != 0
    k_upper_mag = k_upper_fp4 & 0x07

    # Convert to float using E2M1 lookup
    # E2M1_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    k_lower_vals = tl.where(
        k_lower_mag == 0, 0.0,
        tl.where(k_lower_mag == 1, 0.5,
        tl.where(k_lower_mag == 2, 1.0,
        tl.where(k_lower_mag == 3, 1.5,
        tl.where(k_lower_mag == 4, 2.0,
        tl.where(k_lower_mag == 5, 3.0,
        tl.where(k_lower_mag == 6, 4.0, 6.0)))))))
    k_lower_vals = tl.where(k_lower_sign, -k_lower_vals, k_lower_vals)

    k_upper_vals = tl.where(
        k_upper_mag == 0, 0.0,
        tl.where(k_upper_mag == 1, 0.5,
        tl.where(k_upper_mag == 2, 1.0,
        tl.where(k_upper_mag == 3, 1.5,
        tl.where(k_upper_mag == 4, 2.0,
        tl.where(k_upper_mag == 5, 3.0,
        tl.where(k_upper_mag == 6, 4.0, 6.0)))))))
    k_upper_vals = tl.where(k_upper_sign, -k_upper_vals, k_upper_vals)


    # Load scale factors for each token and block
    # Scale factors are stored as uint8: scale_exp = scale_factor - 127
    # packed[i] contains dim 2*i (lower) and dim 2*i+1 (upper)
    # So for lower: dim_offset = offs_dk * 2
    # For upper: dim_offset = offs_dk * 2 + 1
    head_dim_offset = head_id * HEAD_DIM_K
    k_lower_dim_offsets = offs_dk[None, :] * 2
    k_upper_dim_offsets = offs_dk[None, :] * 2 + 1
    k_lower_block_indices = (head_dim_offset + k_lower_dim_offsets) // SCALE_BLOCK_SIZE
    k_upper_block_indices = (head_dim_offset + k_upper_dim_offsets) // SCALE_BLOCK_SIZE

    k_lower_sf_ptrs = (
        ksf_ptr
        + slot_indices[:, None] * stride_ksf_slot
        + k_lower_block_indices * stride_ksf_dim
    )
    k_lower_sf = tl.load(
        k_lower_sf_ptrs, mask=mask_tok[:, None] & mask_dk_packed[None, :], other=127
    ).to(tl.float32) - 127.0
    k_lower_scale = tl.exp2(k_lower_sf)

    k_upper_sf_ptrs = (
        ksf_ptr
        + slot_indices[:, None] * stride_ksf_slot
        + k_upper_block_indices * stride_ksf_dim
    )
    k_upper_sf = tl.load(
        k_upper_sf_ptrs, mask=mask_tok[:, None] & mask_dk_packed[None, :], other=127
    ).to(tl.float32) - 127.0
    k_upper_scale = tl.exp2(k_upper_sf)

    # Apply scaling
    k_lower = (k_lower_vals * k_lower_scale).to(ko_ptr.dtype.element_ty)
    k_upper = (k_upper_vals * k_upper_scale).to(ko_ptr.dtype.element_ty)

    # Store: packed[i] contains dim 2*i (lower) and dim 2*i+1 (upper)
    k_lower_dim_offsets_out = offs_dk[None, :] * 2
    k_upper_dim_offsets_out = offs_dk[None, :] * 2 + 1
    mask_k_lower_out = k_lower_dim_offsets_out < HEAD_DIM_K
    mask_k_upper_out = k_upper_dim_offsets_out < HEAD_DIM_K

    ko_ptrs_lower = (
        ko_ptr
        + head_id * stride_koh
        + out_positions[:, None] * stride_kos
        + k_lower_dim_offsets_out * stride_kod
    )
    tl.store(ko_ptrs_lower, k_lower, mask=mask_tok[:, None] & mask_k_lower_out)

    ko_ptrs_upper = (
        ko_ptr
        + head_id * stride_koh
        + out_positions[:, None] * stride_kos
        + k_upper_dim_offsets_out * stride_kod
    )
    tl.store(ko_ptrs_upper, k_upper, mask=mask_tok[:, None] & mask_k_upper_out)

    # ==== Process V cache ====
    # Similar logic for V
    vc_ptrs = (
        vc_ptr
        + slot_indices[:, None] * stride_vc_slot
        + head_id * stride_vch
        + offs_dv[None, :] * stride_vcd
    )
    vc_packed = tl.load(
        vc_ptrs, mask=mask_tok[:, None] & mask_dv_packed[None, :], other=0
    )

    v_lower_fp4 = vc_packed & 0x0F
    v_upper_fp4 = (vc_packed >> 4) & 0x0F

    v_lower_sign = (v_lower_fp4 & 0x08) != 0
    v_lower_mag = v_lower_fp4 & 0x07
    v_upper_sign = (v_upper_fp4 & 0x08) != 0
    v_upper_mag = v_upper_fp4 & 0x07

    v_lower_vals = tl.where(
        v_lower_mag == 0, 0.0,
        tl.where(v_lower_mag == 1, 0.5,
        tl.where(v_lower_mag == 2, 1.0,
        tl.where(v_lower_mag == 3, 1.5,
        tl.where(v_lower_mag == 4, 2.0,
        tl.where(v_lower_mag == 5, 3.0,
        tl.where(v_lower_mag == 6, 4.0, 6.0)))))))
    v_lower_vals = tl.where(v_lower_sign, -v_lower_vals, v_lower_vals)

    v_upper_vals = tl.where(
        v_upper_mag == 0, 0.0,
        tl.where(v_upper_mag == 1, 0.5,
        tl.where(v_upper_mag == 2, 1.0,
        tl.where(v_upper_mag == 3, 1.5,
        tl.where(v_upper_mag == 4, 2.0,
        tl.where(v_upper_mag == 5, 3.0,
        tl.where(v_upper_mag == 6, 4.0, 6.0)))))))
    v_upper_vals = tl.where(v_upper_sign, -v_upper_vals, v_upper_vals)

    # Load scale factors for V (similar to K)
    head_dim_offset_v = head_id * HEAD_DIM_V
    v_lower_dim_offsets = offs_dv[None, :] * 2
    v_upper_dim_offsets = offs_dv[None, :] * 2 + 1
    v_lower_block_indices = (head_dim_offset_v + v_lower_dim_offsets) // SCALE_BLOCK_SIZE
    v_upper_block_indices = (head_dim_offset_v + v_upper_dim_offsets) // SCALE_BLOCK_SIZE

    v_lower_sf_ptrs = (
        vsf_ptr
        + slot_indices[:, None] * stride_vsf_slot
        + v_lower_block_indices * stride_vsf_dim
    )
    v_lower_sf = tl.load(
        v_lower_sf_ptrs, mask=mask_tok[:, None] & mask_dv_packed[None, :], other=127
    ).to(tl.float32) - 127.0
    v_lower_scale = tl.exp2(v_lower_sf)

    v_upper_sf_ptrs = (
        vsf_ptr
        + slot_indices[:, None] * stride_vsf_slot
        + v_upper_block_indices * stride_vsf_dim
    )
    v_upper_sf = tl.load(
        v_upper_sf_ptrs, mask=mask_tok[:, None] & mask_dv_packed[None, :], other=127
    ).to(tl.float32) - 127.0
    v_upper_scale = tl.exp2(v_upper_sf)

    v_lower = (v_lower_vals * v_lower_scale).to(vo_ptr.dtype.element_ty)
    v_upper = (v_upper_vals * v_upper_scale).to(vo_ptr.dtype.element_ty)

    # Store: packed[i] contains dim 2*i (lower) and dim 2*i+1 (upper)
    v_lower_dim_offsets_out = offs_dv[None, :] * 2
    v_upper_dim_offsets_out = offs_dv[None, :] * 2 + 1
    mask_v_lower_out = v_lower_dim_offsets_out < HEAD_DIM_V
    mask_v_upper_out = v_upper_dim_offsets_out < HEAD_DIM_V

    vo_ptrs_lower = (
        vo_ptr
        + head_id * stride_voh
        + out_positions[:, None] * stride_vos
        + v_lower_dim_offsets_out * stride_vod
    )
    tl.store(vo_ptrs_lower, v_lower, mask=mask_tok[:, None] & mask_v_lower_out)

    vo_ptrs_upper = (
        vo_ptr
        + head_id * stride_voh
        + out_positions[:, None] * stride_vos
        + v_upper_dim_offsets_out * stride_vod
    )
    tl.store(vo_ptrs_upper, v_upper, mask=mask_tok[:, None] & mask_v_upper_out)


@triton.jit
def _flatten_kv_cache_quant_sglang(
    # Input pointers
    kc_ptr,  # Quantized K cache [total_slots, num_heads, head_dim]
    vc_ptr,  # Quantized V cache
    ksz_ptr,  # K scales/zeros [total_slots, num_heads, 2]
    vsz_ptr,  # V scales/zeros
    page_table_ptr,  # Page table [batch_size, max_slots_per_seq] - contains slot indices
    cache_seqlens_ptr,  # Valid sequence lengths [batch_size]
    cu_seqlens_k_ptr,  # Cumulative seq lengths [batch_size + 1]
    # Output pointers
    ko_ptr,  # Flattened dequantized K output [total_tokens, num_heads, head_dim]
    vo_ptr,  # Flattened dequantized V output
    # Strides for K cache (input) - 3D [slots, heads, dim]
    stride_kc_slot: tl.constexpr,  # Stride across slots
    stride_kch: tl.constexpr,  # Stride across heads
    stride_kcd: tl.constexpr,  # Stride across dimensions
    # Strides for V cache (input)
    stride_vc_slot: tl.constexpr,
    stride_vch: tl.constexpr,
    stride_vcd: tl.constexpr,
    # Strides for scales/zeros - 3D [slots, heads, 2]
    stride_ksz_slot: tl.constexpr,
    stride_kszh: tl.constexpr,
    stride_kszd: tl.constexpr,  # =2, for [scale, zero]
    stride_vsz_slot: tl.constexpr,
    stride_vszh: tl.constexpr,
    stride_vszd: tl.constexpr,
    # Strides for output
    stride_kos: tl.constexpr,  # Output K seq/token stride
    stride_koh: tl.constexpr,  # Output K head stride
    stride_kod: tl.constexpr,  # Output K dim stride
    stride_vos: tl.constexpr,
    stride_voh: tl.constexpr,
    stride_vod: tl.constexpr,
    # Page table stride
    stride_page_table,  # Stride for page table indexing
    # Config
    quant_policy: tl.constexpr,  # 4=int4, 8=int8
    TOTAL_SLOTS,  # Total number of slots in cache (for bounds checking)
    PAGE_SIZE: tl.constexpr,  # Page size (tokens per page)
    HEAD_DIM_K: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,  # Number of tokens to process per kernel block
    BLOCK_DK: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    """
    Flatten slot-based KV cache with on-demand dequantization for SGLang.

    Grid: (num_blocks_per_seq, batch_size, num_heads)
    Each thread block processes BLOCK_SIZE tokens for one head of one sequence.
    """
    block_id = tl.program_id(0)  # Which block of tokens within the sequence
    batch_id = tl.program_id(1)  # Which sequence in batch
    head_id = tl.program_id(2)  # Which head

    seqlen = tl.load(cache_seqlens_ptr + batch_id)
    start_loc = tl.load(cu_seqlens_k_ptr + batch_id)

    if block_id * BLOCK_SIZE >= seqlen:
        return

    # Token offsets within this block
    offs_tok = tl.arange(0, BLOCK_SIZE)

    # Dimension offsets
    if quant_policy == 4:
        # int4: stored dimensions are half the actual dimensions
        # Use packed dimension size for offsets (matching decode_attention.py)
        BLOCK_DK_PACKED: tl.constexpr = BLOCK_DK // 2
        BLOCK_DV_PACKED: tl.constexpr = BLOCK_DV // 2
        HALF_HDK: tl.constexpr = HEAD_DIM_K // 2
        HALF_HDV: tl.constexpr = HEAD_DIM_V // 2
        offs_dk = tl.arange(0, BLOCK_DK_PACKED)
        offs_dv = tl.arange(0, BLOCK_DV_PACKED)
        mask_dk_packed = offs_dk < HALF_HDK
        mask_dv_packed = offs_dv < HALF_HDV
    else:
        offs_dk = tl.arange(0, BLOCK_DK)
        offs_dv = tl.arange(0, BLOCK_DV)
        mask_dk_packed = offs_dk < HEAD_DIM_K
        mask_dv_packed = offs_dv < HEAD_DIM_V

    # Output token offsets (in flattened space)
    offs_out_tok = block_id * BLOCK_SIZE + offs_tok
    mask_tok = offs_out_tok < seqlen

    # Full dimension offsets for output (after dequantization)
    offs_dok = tl.arange(0, BLOCK_DK)
    offs_dov = tl.arange(0, BLOCK_DV)
    mask_dok = offs_dok < HEAD_DIM_K
    mask_dov = offs_dov < HEAD_DIM_V

    # Get slot indices for all tokens in this block
    # SGLang uses paged structure: page_table contains page indices
    # Slot index = page_index * PAGE_SIZE + offset_within_page

    # Global token positions for this block
    global_tok_positions = block_id * BLOCK_SIZE + offs_tok

    # Which page does each token belong to?
    page_indices_in_seq = global_tok_positions // PAGE_SIZE
    offsets_within_page = global_tok_positions % PAGE_SIZE

    # Load page indices from page table
    page_table_ptrs = (
        page_table_ptr + batch_id * stride_page_table + page_indices_in_seq
    )
    page_indices = tl.load(page_table_ptrs, mask=mask_tok, other=0)

    # Compute actual slot indices: page_index * PAGE_SIZE + offset
    slot_indices = page_indices * PAGE_SIZE + offsets_within_page

    # Clamp slot indices to valid range [0, TOTAL_SLOTS-1] to prevent illegal memory access
    slot_indices = tl.where(slot_indices < TOTAL_SLOTS, slot_indices, 0)
    slot_indices = tl.where(slot_indices >= 0, slot_indices, 0)

    # ==== Process K cache ====
    # Load scales and zeros for each token (needed for both INT4 and INT8)
    ksz_ptrs = ksz_ptr + slot_indices * stride_ksz_slot + head_id * stride_kszh
    ks = tl.load(ksz_ptrs, mask=mask_tok, other=1.0)  # Scale
    kz = tl.load(ksz_ptrs + stride_kszd, mask=mask_tok, other=0.0)  # Zero point

    # Output token positions in flattened space
    out_positions = start_loc + offs_out_tok

    if quant_policy == 4:
        # INT4: Load packed data (head_dim // 2)
        kc_ptrs = (
            kc_ptr
            + slot_indices[:, None] * stride_kc_slot
            + head_id * stride_kch
            + offs_dk[None, :] * stride_kcd
        )
        kc_packed = tl.load(
            kc_ptrs, mask=mask_tok[:, None] & mask_dk_packed[None, :], other=0
        )

        # Unpack lower 4 bits (first half of dimension)
        # kc_packed shape: [BLOCK_SIZE, BLOCK_DK_PACKED]
        k_lower = ((kc_packed & 0x0F).to(tl.float32) - kz[:, None]) * ks[:, None]
        k_lower = k_lower.to(ko_ptr.dtype.element_ty)

        # Unpack upper 4 bits (second half of dimension)
        k_upper = (((kc_packed >> 4) & 0x0F).to(tl.float32) - kz[:, None]) * ks[:, None]
        k_upper = k_upper.to(ko_ptr.dtype.element_ty)

        # For storing: offs_dk now has correct size (BLOCK_DK_PACKED)
        # Store lower half to [0, HEAD_DIM_K//2)
        ko_ptrs_lower = (
            ko_ptr
            + head_id * stride_koh
            + out_positions[:, None] * stride_kos
            + offs_dk[None, :] * stride_kod
        )
        tl.store(
            ko_ptrs_lower, k_lower, mask=mask_tok[:, None] & mask_dk_packed[None, :]
        )

        # Store upper half to [HEAD_DIM_K//2, HEAD_DIM_K)
        offs_dk_upper = offs_dk + (HEAD_DIM_K // 2)
        mask_dk_upper = offs_dk_upper < HEAD_DIM_K
        ko_ptrs_upper = (
            ko_ptr
            + head_id * stride_koh
            + out_positions[:, None] * stride_kos
            + offs_dk_upper[None, :] * stride_kod
        )
        tl.store(ko_ptrs_upper, k_upper, mask=mask_tok[:, None] & mask_dk_upper)
    else:
        # INT8: Standard loading and dequantization
        kc_ptrs = (
            kc_ptr
            + slot_indices[:, None] * stride_kc_slot
            + head_id * stride_kch
            + offs_dk[None, :] * stride_kcd
        )
        kc = tl.load(kc_ptrs, mask=mask_tok[:, None], other=0)

        # Apply dequantization: x_dequant = (x_quant - zero) * scale
        kq = ((kc.to(tl.float32) - kz[:, None]) * ks[:, None]).to(
            ko_ptr.dtype.element_ty
        )

        # Store to flattened output
        ko_ptrs = (
            ko_ptr
            + head_id * stride_koh
            + out_positions[:, None] * stride_kos
            + offs_dok[None, :] * stride_kod
        )
        tl.store(ko_ptrs, kq, mask=mask_tok[:, None] & mask_dok[None, :])

    # ==== Process V cache ====
    # Load scales and zeros for each token (needed for both INT4 and INT8)
    vsz_ptrs = vsz_ptr + slot_indices * stride_vsz_slot + head_id * stride_vszh
    vs = tl.load(vsz_ptrs, mask=mask_tok, other=1.0)  # Scale
    vz = tl.load(vsz_ptrs + stride_vszd, mask=mask_tok, other=0.0)  # Zero point

    if quant_policy == 4:
        # INT4: Load packed data (head_dim // 2), unpack lower/upper 4 bits separately
        # Following the correct logic from decode_attention.py
        # offs_dv has size BLOCK_DV_PACKED (not BLOCK_DV!) to avoid duplicate loads
        vc_ptrs = (
            vc_ptr
            + slot_indices[:, None] * stride_vc_slot
            + head_id * stride_vch
            + offs_dv[None, :] * stride_vcd
        )
        vc_packed = tl.load(
            vc_ptrs, mask=mask_tok[:, None] & mask_dv_packed[None, :], other=0
        )

        # Unpack lower 4 bits (first half of dimension)
        # vc_packed shape: [BLOCK_SIZE, BLOCK_DV_PACKED]
        v_lower = ((vc_packed & 0x0F).to(tl.float32) - vz[:, None]) * vs[:, None]
        v_lower = v_lower.to(vo_ptr.dtype.element_ty)

        # Unpack upper 4 bits (second half of dimension)
        v_upper = (((vc_packed >> 4) & 0x0F).to(tl.float32) - vz[:, None]) * vs[:, None]
        v_upper = v_upper.to(vo_ptr.dtype.element_ty)

        # For storing: offs_dv now has correct size (BLOCK_DV_PACKED)
        # Store lower half to [0, HEAD_DIM_V//2)
        vo_ptrs_lower = (
            vo_ptr
            + head_id * stride_voh
            + out_positions[:, None] * stride_vos
            + offs_dv[None, :] * stride_vod
        )
        tl.store(
            vo_ptrs_lower, v_lower, mask=mask_tok[:, None] & mask_dv_packed[None, :]
        )

        # Store upper half to [HEAD_DIM_V//2, HEAD_DIM_V)
        offs_dv_upper = offs_dv + (HEAD_DIM_V // 2)
        mask_dv_upper = offs_dv_upper < HEAD_DIM_V
        vo_ptrs_upper = (
            vo_ptr
            + head_id * stride_voh
            + out_positions[:, None] * stride_vos
            + offs_dv_upper[None, :] * stride_vod
        )
        tl.store(vo_ptrs_upper, v_upper, mask=mask_tok[:, None] & mask_dv_upper)
    else:
        # INT8: Standard loading and dequantization
        vc_ptrs = (
            vc_ptr
            + slot_indices[:, None] * stride_vc_slot
            + head_id * stride_vch
            + offs_dv[None, :] * stride_vcd
        )
        vc = tl.load(vc_ptrs, mask=mask_tok[:, None], other=0)

        # Apply dequantization: x_dequant = (x_quant - zero) * scale
        vq = ((vc.to(tl.float32) - vz[:, None]) * vs[:, None]).to(
            vo_ptr.dtype.element_ty
        )

        # Store to flattened output
        vo_ptrs = (
            vo_ptr
            + head_id * stride_voh
            + out_positions[:, None] * stride_vos
            + offs_dov[None, :] * stride_vod
        )
        tl.store(vo_ptrs, vq, mask=mask_tok[:, None] & mask_dov[None, :])


def flatten_kv_cache_sglang(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scales_zeros: Union[torch.Tensor, None],
    v_scales_zeros: Union[torch.Tensor, None],
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    page_size: int,
    num_heads: int,
    head_dim_k: int,
    head_dim_v: int,
    quant_policy: Union[Literal[4, 8], Literal["fp4"]],
    output_dtype: torch.dtype,
    max_seq_len_k: int,
    out_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Flatten paged KV cache with on-demand dequantization for SGLang.

    Args:
        k_cache: Quantized K cache [total_slots, num_heads, head_dim_k_quant] (3D, will be reshaped)
        v_cache: Quantized V cache [total_slots, num_heads, head_dim_v_quant]
        k_scales_zeros: K scales/zeros [total_slots, num_heads, 2] (will be reshaped)
        v_scales_zeros: V scales/zeros [total_slots, num_heads, 2]
        page_table: Page table [batch_size, max_num_pages]
        cache_seqlens: Valid sequence lengths [batch_size]
        cu_seqlens_k: Cumulative sequence lengths [batch_size + 1]
        page_size: Number of tokens per page
        num_heads: Number of attention heads
        head_dim_k: Head dimension for K (full, not quantized size)
        head_dim_v: Head dimension for V
        quant_policy: 4 for int4, 8 for int8, "fp4" for fp4
        output_dtype: Output dtype (typically fp16 or bf16)
        max_seq_len_k: Maximum sequence length in batch (must be provided to avoid D2H copy)
        out_size: Total number of output tokens (must be provided to avoid D2H copy)

    Returns:
        Tuple of (flattened_k, flattened_v)
        - flattened_k: [total_tokens, num_heads, head_dim_k]
        - flattened_v: [total_tokens, num_heads, head_dim_v]
    """
    batch_size = page_table.size(0)
    max_slots_per_seq = page_table.size(1)

    # SGLang stores cache as [total_slots, num_heads, head_dim] - no reshaping needed!
    # Sanity checks
    assert (
        k_cache.dim() == 3
    ), f"Expected 3D k_cache, got {k_cache.dim()}D with shape {k_cache.shape}"
    assert (
        v_cache.dim() == 3
    ), f"Expected 3D v_cache, got {v_cache.dim()}D with shape {v_cache.shape}"
    assert (
        k_cache.size(1) == num_heads
    ), f"k_cache heads {k_cache.size(1)} != num_heads {num_heads}"
    assert (
        v_cache.size(1) == num_heads
    ), f"v_cache heads {v_cache.size(1)} != num_heads {num_heads}"

    # Page table contains page indices, each page holds page_size tokens
    # So max_slots_per_seq is the number of pages, not tokens
    num_pages_needed = (max_seq_len_k + page_size - 1) // page_size
    if num_pages_needed > max_slots_per_seq:
        raise ValueError(
            f"Max sequence length {max_seq_len_k} requires {num_pages_needed} pages "
            f"but page table only has {max_slots_per_seq} entries. "
            f"Page table shape: {page_table.shape}, cache_seqlens: {cache_seqlens}"
        )

    # Adjust head dims for quantized storage
    head_dim_k_stored = k_cache.size(-1)
    head_dim_v_stored = v_cache.size(-1)

    if quant_policy == 4:
        # int4: each byte stores 2 values
        assert head_dim_k == head_dim_k_stored * 2
        assert head_dim_v == head_dim_v_stored * 2
    elif quant_policy == "fp4":
        # fp4: each byte stores 2 fp4 values
        assert head_dim_k == head_dim_k_stored * 2
        assert head_dim_v == head_dim_v_stored * 2
        assert k_scales_zeros is not None, "k_scales_zeros required for fp4"
        assert v_scales_zeros is not None, "v_scales_zeros required for fp4"

    # Block sizes (power of 2 for Triton)
    BLOCK_DK = triton.next_power_of_2(head_dim_k)
    BLOCK_DV = triton.next_power_of_2(head_dim_v)
    BLOCK_SIZE = 32  # Process 32 tokens per block (tunable)

    # Allocate flattened output
    # Use layout: [num_heads, total_tokens, head_dim] for better memory access
    k_flattened = torch.empty(
        (out_size, num_heads, head_dim_k), dtype=output_dtype, device=k_cache.device
    )
    v_flattened = torch.empty(
        (out_size, num_heads, head_dim_v), dtype=output_dtype, device=v_cache.device
    )

    # Calculate grid size using pre-computed max_seq_len_k (avoids D2H copy)
    num_blocks = (max_seq_len_k + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_blocks, batch_size, num_heads)

    # Get total slots for bounds checking
    total_slots = k_cache.size(0)

    if quant_policy == "fp4":
        # Use FP4 kernel
        SCALE_BLOCK_SIZE = 16
        _flatten_kv_cache_quant_sglang_fp4[grid](
            # Input buffers
            k_cache,
            v_cache,
            k_scales_zeros,
            v_scales_zeros,
            page_table,
            cache_seqlens,
            cu_seqlens_k,
            # Output buffers
            k_flattened,
            v_flattened,
            # K cache strides (3D: [slots, heads, dim//2])
            stride_kc_slot=k_cache.stride(0),
            stride_kch=k_cache.stride(1),
            stride_kcd=k_cache.stride(2),
            # V cache strides
            stride_vc_slot=v_cache.stride(0),
            stride_vch=v_cache.stride(1),
            stride_vcd=v_cache.stride(2),
            # Scale factors strides (2D: [slots, (num_heads * head_dim) // 16])
            stride_ksf_slot=k_scales_zeros.stride(0),
            stride_ksf_dim=k_scales_zeros.stride(1),
            stride_vsf_slot=v_scales_zeros.stride(0),
            stride_vsf_dim=v_scales_zeros.stride(1),
            # Output strides
            stride_kos=k_flattened.stride(0),
            stride_koh=k_flattened.stride(1),
            stride_kod=k_flattened.stride(2),
            stride_vos=v_flattened.stride(0),
            stride_voh=v_flattened.stride(1),
            stride_vod=v_flattened.stride(2),
            # Page table stride
            stride_page_table=page_table.stride(0),
            # Config
            TOTAL_SLOTS=total_slots,
            PAGE_SIZE=page_size,
            HEAD_DIM_K=head_dim_k,
            HEAD_DIM_V=head_dim_v,
            BLOCK_SIZE=BLOCK_SIZE,
            BLOCK_DK=BLOCK_DK,
            BLOCK_DV=BLOCK_DV,
            NUM_HEADS=num_heads,
            SCALE_BLOCK_SIZE=SCALE_BLOCK_SIZE,
        )
    else:
        # Use INT4/INT8 kernel
        assert k_scales_zeros is not None, "k_scales_zeros required for int4/int8"
        assert v_scales_zeros is not None, "v_scales_zeros required for int4/int8"
        _flatten_kv_cache_quant_sglang[grid](
            # Input buffers
            k_cache,
            v_cache,
            k_scales_zeros,
            v_scales_zeros,
            page_table,
            cache_seqlens,
            cu_seqlens_k,
            # Output buffers
            k_flattened,
            v_flattened,
            # K cache strides (3D: [slots, heads, dim])
            stride_kc_slot=k_cache.stride(0),
            stride_kch=k_cache.stride(1),
            stride_kcd=k_cache.stride(2),
            # V cache strides
            stride_vc_slot=v_cache.stride(0),
            stride_vch=v_cache.stride(1),
            stride_vcd=v_cache.stride(2),
            # Scales/zeros strides (3D: [slots, heads, 2])
            stride_ksz_slot=k_scales_zeros.stride(0),
            stride_kszh=k_scales_zeros.stride(1),
            stride_kszd=k_scales_zeros.stride(2),
            stride_vsz_slot=v_scales_zeros.stride(0),
            stride_vszh=v_scales_zeros.stride(1),
            stride_vszd=v_scales_zeros.stride(2),
            # Output strides
            stride_kos=k_flattened.stride(0),
            stride_koh=k_flattened.stride(1),
            stride_kod=k_flattened.stride(2),
            stride_vos=v_flattened.stride(0),
            stride_voh=v_flattened.stride(1),
            stride_vod=v_flattened.stride(2),
            # Page table stride
            stride_page_table=page_table.stride(0),
            # Config
            quant_policy=quant_policy,
            TOTAL_SLOTS=total_slots,
            PAGE_SIZE=page_size,
            HEAD_DIM_K=head_dim_k,
            HEAD_DIM_V=head_dim_v,
            BLOCK_SIZE=BLOCK_SIZE,
            BLOCK_DK=BLOCK_DK,
            BLOCK_DV=BLOCK_DV,
        )

    return k_flattened, v_flattened
