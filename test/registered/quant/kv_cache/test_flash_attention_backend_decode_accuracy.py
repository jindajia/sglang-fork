"""
Unit tests for Flash Attention backend decode with int8/int4/fp4 quantized KV cache.
Tests decode attention with quantized KV cache against non-quantized baseline.
"""

import unittest

import torch

from sgl_kernel.flash_attn import flash_attn_varlen_func
from sglang.srt.layers.attention.kernels.flatten_kv_cache import flatten_kv_cache_sglang
from sglang.srt.mem_cache.kv_quant_kernels import (
    quantized_set_kv_fp4_torch,
    quantized_set_kv_int4_triton,
    quantized_set_kv_int8_triton,
)
from sglang.srt.utils import get_device, get_device_sm
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=60, suite="stage-b-test-small-1-gpu-amd")


@unittest.skipIf(
    get_device_sm() < 90,
    "Flash Attention backend requires CUDA SM 90 or higher (Hopper/Blackwell)",
)
class TestFlashAttentionBackendDecodeAccuracy(CustomTestCase):
    """Test Flash Attention backend decode with int8/int4/fp4 quantized KV cache."""

    def _set_all_seeds(self, seed):
        """Set all random seeds for reproducibility."""
        import random

        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def setUp(self):
        # Set seeds before each test method
        self._set_all_seeds(42)

    def _test_flash_attention_decode_quant_once(
        self, B, H_Q, H_KV, D, S, kv_dtype, page_size=16, cache_size=None
    ):
        """
        Test Flash Attention backend decode with quantized KV cache against non-quantized baseline.

        Args:
            B: Batch size
            H_Q: Number of query heads
            H_KV: Number of KV heads
            D: Head dimension
            S: Sequence length (number of tokens in KV cache)
            kv_dtype: "int4", "int8", or "fp4"
            page_size: Page size for paged KV cache (default 16)
            cache_size: Size of cache buffer (defaults to total_tokens)
        """
        device = get_device()
        dtype = torch.bfloat16
        total_tokens = B * S
        # Calculate required cache size based on page table
        max_pages = (S + page_size - 1) // page_size
        total_slots_needed = B * max_pages * page_size
        if cache_size is None:
            cache_size = total_slots_needed
        sm_scale = 1.0 / (D**0.5)

        # Create query (one per batch item, decode step)
        q = torch.randn(B, H_Q, D, dtype=dtype, device=device)

        # Create non-quantized KV cache (baseline)
        k_buffer_fp = torch.randn(total_tokens, H_KV, D, dtype=dtype, device=device)
        v_buffer_fp = torch.randn(total_tokens, H_KV, D, dtype=dtype, device=device)

        # Create cumulative sequence lengths for query (decode: each query has length 1)
        cu_seqlens_q = torch.arange(0, B + 1, dtype=torch.int32, device=device)

        # Create cumulative sequence lengths for key (each batch item has S tokens)
        cu_seqlens_k = torch.zeros(B + 1, dtype=torch.int32, device=device)
        cu_seqlens_k[1:] = torch.cumsum(
            torch.full((B,), S, dtype=torch.int32, device=device), dim=0
        )

        # Create cache sequence lengths
        cache_seqlens = torch.full((B,), S, dtype=torch.int32, device=device)

        # Create page table for paged KV cache
        # Each batch item needs (S + page_size - 1) // page_size pages
        max_pages = (S + page_size - 1) // page_size
        page_table = torch.zeros(B, max_pages, dtype=torch.int32, device=device)
        for b in range(B):
            num_pages = (S + page_size - 1) // page_size
            # Assign sequential page indices for each batch item
            page_table[b, :num_pages] = torch.arange(
                b * max_pages, b * max_pages + num_pages, device=device, dtype=torch.int32
            )

        # Run non-quantized baseline using flash_attn_varlen_func
        # For baseline, we need to reshape KV cache to match flash_attn format
        # flash_attn expects [total_tokens, num_heads, head_dim]
        # Use ver=4 for Blackwell (sm100) GPUs
        fa_ver = 4 if get_device_sm() >= 100 else None
        baseline_kwargs = {"ver": fa_ver} if fa_ver is not None else {}
        o_baseline = flash_attn_varlen_func(
            q=q.contiguous().view(-1, H_Q, D),
            k=k_buffer_fp.contiguous().view(-1, H_KV, D),
            v=v_buffer_fp.contiguous().view(-1, H_KV, D),
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=1,  # Decode: each query has length 1
            max_seqlen_k=S,
            softmax_scale=sm_scale,
            causal=True,
            **baseline_kwargs,
        )

        # Create quantized KV cache
        if kv_dtype == "int4":
            head_dim_stored = D // 2
            assert D % 2 == 0, "head_dim must be even for int4"
        elif kv_dtype == "fp4":
            head_dim_stored = D // 2
            assert D % 16 == 0, "head_dim must be divisible by 16 for fp4"
        else:
            head_dim_stored = D

        # Create cache buffers
        k_cache_buffer = torch.zeros(
            cache_size, H_KV, head_dim_stored, device=device, dtype=torch.uint8
        )
        v_cache_buffer = torch.zeros(
            cache_size, H_KV, head_dim_stored, device=device, dtype=torch.uint8
        )

        if kv_dtype == "fp4":
            # FP4 uses scale factors: [cache_size, (num_heads * head_dim) // 16]
            k_scales_zeros = torch.zeros(
                cache_size, (H_KV * D) // 16, device=device, dtype=torch.uint8
            )
            v_scales_zeros = torch.zeros(
                cache_size, (H_KV * D) // 16, device=device, dtype=torch.uint8
            )
        else:
            # INT4/INT8 use scales/zeros: [cache_size, num_heads, 2]
            k_scales_zeros = torch.zeros(
                cache_size, H_KV, 2, device=device, dtype=torch.float32
            )
            v_scales_zeros = torch.zeros(
                cache_size, H_KV, 2, device=device, dtype=torch.float32
            )

        # Quantize KV cache
        # Map tokens to cache locations according to page table (must match flatten_kv_cache_sglang)
        cache_loc_list = []
        for b in range(B):
            seq_start = cu_seqlens_k[b].item()
            seq_end = cu_seqlens_k[b + 1].item()
            for token_idx in range(seq_start, seq_end):
                token_in_seq = token_idx - seq_start
                page_idx = token_in_seq // page_size
                offset_in_page = token_in_seq % page_size
                # Get the page ID from page table
                page_id = page_table[b, page_idx].item()
                # Compute slot index: page_id * page_size + offset_in_page
                # This matches the formula in flatten_kv_cache: slot = page_index * PAGE_SIZE + offset
                slot_id = page_id * page_size + offset_in_page
                cache_loc_list.append(slot_id)
        cache_loc = torch.tensor(cache_loc_list, device=device, dtype=torch.int32)
        if kv_dtype == "int4":
            quantized_set_kv_int4_triton(
                k_buffer_fp,
                v_buffer_fp,
                cache_loc,
                k_cache_buffer,
                v_cache_buffer,
                k_scales_zeros,
                v_scales_zeros,
            )
        elif kv_dtype == "int8":
            quantized_set_kv_int8_triton(
                k_buffer_fp,
                v_buffer_fp,
                cache_loc,
                k_cache_buffer,
                v_cache_buffer,
                k_scales_zeros,
                v_scales_zeros,
            )
        elif kv_dtype == "fp4":
            quantized_set_kv_fp4_torch(
                k_buffer_fp,
                v_buffer_fp,
                cache_loc,
                k_cache_buffer,
                v_cache_buffer,
                k_scales_zeros,
                v_scales_zeros,
            )

        # Flatten and dequantize KV cache using flatten_kv_cache_sglang
        if kv_dtype == "fp4":
            quant_policy = "fp4"
        else:
            quant_policy = 4 if kv_dtype == "int4" else 8

        flatten_k, flatten_v = flatten_kv_cache_sglang(
            k_cache=k_cache_buffer,
            v_cache=v_cache_buffer,
            k_scales_zeros=k_scales_zeros,
            v_scales_zeros=v_scales_zeros,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_k=cu_seqlens_k,
            page_size=page_size,
            num_heads=H_KV,
            head_dim_k=D,
            head_dim_v=D,
            quant_policy=quant_policy,
            output_dtype=dtype,
            max_seq_len_k=S,
            out_size=total_tokens,
        )

        # Run quantized decode attention using flash_attn_varlen_func
        # Use ver=4 for Blackwell (sm100) GPUs
        quant_kwargs = {"ver": fa_ver} if fa_ver is not None else {}
        o_quant = flash_attn_varlen_func(
            q=q.contiguous().view(-1, H_Q, D),
            k=flatten_k.contiguous().view(-1, H_KV, D),
            v=flatten_v.contiguous().view(-1, H_KV, D),
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=1,  # Decode: each query has length 1
            max_seqlen_k=S,
            softmax_scale=sm_scale,
            causal=True,
            **quant_kwargs,
        )

        # Compare outputs
        diff = o_quant - o_baseline
        norm_diff = torch.norm(diff).item()
        norm_baseline = torch.norm(o_baseline).item()
        relative_error = norm_diff / (norm_baseline + 1e-8)

        # Calculate max absolute difference
        max_diff = torch.abs(diff).max().item()

        # Print error metrics for debugging
        print(
            f"\nConfig: B={B}, H_Q={H_Q}, H_KV={H_KV}, D={D}, S={S}, dtype={kv_dtype}, page_size={page_size}"
        )
        print(f"Relative error: {relative_error:.6f}")
        print(f"Max absolute diff: {max_diff:.6f}")
        print(f"Norm(baseline): {norm_baseline:.6f}, Norm(diff): {norm_diff:.6f}")

        # Set thresholds based on quantization type
        if kv_dtype == "int4":
            # Allow up to 20% relative error for int4
            max_rel_error_threshold = 0.20
            max_abs_error_threshold = 0.4
        elif kv_dtype == "fp4":
            # Allow up to 20% relative error for fp4
            max_rel_error_threshold = 0.20
            max_abs_error_threshold = 1.0
        else:
            # For int8, quantization error is smaller
            max_rel_error_threshold = 0.02
            max_abs_error_threshold = 0.1

        self.assertTrue(
            relative_error < max_rel_error_threshold,
            f"Relative error {relative_error:.6f} exceeds {max_rel_error_threshold} for {kv_dtype}",
        )
        self.assertTrue(
            max_diff < max_abs_error_threshold,
            f"Max absolute diff {max_diff:.6f} exceeds {max_abs_error_threshold} for {kv_dtype}",
        )

    def test_flash_attention_decode_int8_basic(self):
        """Test basic int8 Flash Attention decode."""
        self._test_flash_attention_decode_quant_once(
            B=2, H_Q=4, H_KV=4, D=64, S=10, kv_dtype="int8"
        )

    def test_flash_attention_decode_int4_basic(self):
        """Test basic int4 Flash Attention decode."""
        self._test_flash_attention_decode_quant_once(
            B=2, H_Q=4, H_KV=4, D=64, S=10, kv_dtype="int4"
        )

    def test_flash_attention_decode_fp4_basic(self):
        """Test basic fp4 Flash Attention decode."""
        self._test_flash_attention_decode_quant_once(
            B=2, H_Q=4, H_KV=4, D=128, S=10, kv_dtype="fp4"
        )

    def test_flash_attention_decode_int8_grouped(self):
        """Test int8 Flash Attention decode with grouped attention (GQA)."""
        self._test_flash_attention_decode_quant_once(
            B=2, H_Q=16, H_KV=4, D=128, S=20, kv_dtype="int8"
        )

    def test_flash_attention_decode_int4_grouped(self):
        """Test int4 Flash Attention decode with grouped attention (GQA)."""
        self._test_flash_attention_decode_quant_once(
            B=2, H_Q=16, H_KV=4, D=128, S=20, kv_dtype="int4"
        )

    def test_flash_attention_decode_fp4_grouped(self):
        """Test fp4 Flash Attention decode with grouped attention (GQA)."""
        self._test_flash_attention_decode_quant_once(
            B=2, H_Q=16, H_KV=4, D=128, S=20, kv_dtype="fp4"
        )

    def test_flash_attention_decode_int8_different_head_dims(self):
        """Test int8 Flash Attention decode with different head dimensions."""
        # FA4 (ver=4) on Blackwell doesn't support head_dim=256, so skip it
        head_dims = [64, 128] if get_device_sm() >= 100 else [64, 128, 256]
        for head_dim in head_dims:
            self._test_flash_attention_decode_quant_once(
                B=2, H_Q=8, H_KV=8, D=head_dim, S=10, kv_dtype="int8"
            )

    def test_flash_attention_decode_int4_different_head_dims(self):
        """Test int4 Flash Attention decode with different head dimensions."""
        # FA4 (ver=4) on Blackwell doesn't support head_dim=256, so skip it
        head_dims = [64, 128] if get_device_sm() >= 100 else [64, 128, 256]
        for head_dim in head_dims:
            self._test_flash_attention_decode_quant_once(
                B=2, H_Q=8, H_KV=8, D=head_dim, S=10, kv_dtype="int4"
            )

    def test_flash_attention_decode_fp4_different_head_dims(self):
        """Test fp4 Flash Attention decode with different head dimensions."""
        # FA4 (ver=4) on Blackwell doesn't support head_dim=256, so skip it
        # fp4 requires head_dim divisible by 16
        head_dims = [128, 256] if get_device_sm() >= 100 else [128, 256]
        for head_dim in head_dims:
            self._test_flash_attention_decode_quant_once(
                B=2, H_Q=8, H_KV=8, D=head_dim, S=10, kv_dtype="fp4"
            )

    def test_flash_attention_decode_int8_different_seq_lens(self):
        """Test int8 Flash Attention decode with different sequence lengths."""
        for seq_len in [5, 20, 50, 100]:
            self._test_flash_attention_decode_quant_once(
                B=2, H_Q=4, H_KV=4, D=64, S=seq_len, kv_dtype="int8"
            )

    def test_flash_attention_decode_int4_different_seq_lens(self):
        """Test int4 Flash Attention decode with different sequence lengths."""
        for seq_len in [5, 20, 50, 100]:
            self._test_flash_attention_decode_quant_once(
                B=2, H_Q=4, H_KV=4, D=64, S=seq_len, kv_dtype="int4"
            )

    def test_flash_attention_decode_fp4_different_seq_lens(self):
        """Test fp4 Flash Attention decode with different sequence lengths."""
        for seq_len in [5, 20, 50, 100]:
            self._test_flash_attention_decode_quant_once(
                B=2, H_Q=4, H_KV=4, D=128, S=seq_len, kv_dtype="fp4"
            )

    def test_flash_attention_decode_int8_large_batch(self):
        """Test int8 Flash Attention decode with large batch."""
        self._test_flash_attention_decode_quant_once(
            B=8, H_Q=16, H_KV=16, D=128, S=50, kv_dtype="int8"
        )

    def test_flash_attention_decode_int4_large_batch(self):
        """Test int4 Flash Attention decode with large batch."""
        self._test_flash_attention_decode_quant_once(
            B=8, H_Q=16, H_KV=16, D=128, S=50, kv_dtype="int4"
        )

    def test_flash_attention_decode_fp4_large_batch(self):
        """Test fp4 Flash Attention decode with large batch."""
        self._test_flash_attention_decode_quant_once(
            B=8, H_Q=16, H_KV=16, D=128, S=50, kv_dtype="fp4"
        )

    def test_flash_attention_decode_int8_different_page_sizes(self):
        """Test int8 Flash Attention decode with different page sizes."""
        for page_size in [8, 16, 32]:
            self._test_flash_attention_decode_quant_once(
                B=2, H_Q=4, H_KV=4, D=64, S=40, kv_dtype="int8", page_size=page_size
            )

    def test_flash_attention_decode_int4_different_page_sizes(self):
        """Test int4 Flash Attention decode with different page sizes."""
        for page_size in [8, 16, 32]:
            self._test_flash_attention_decode_quant_once(
                B=2, H_Q=4, H_KV=4, D=64, S=40, kv_dtype="int4", page_size=page_size
            )

    def test_flash_attention_decode_fp4_different_page_sizes(self):
        """Test fp4 Flash Attention decode with different page sizes."""
        for page_size in [8, 16, 32]:
            self._test_flash_attention_decode_quant_once(
                B=2, H_Q=4, H_KV=4, D=128, S=40, kv_dtype="fp4", page_size=page_size
            )


if __name__ == "__main__":
    unittest.main()
