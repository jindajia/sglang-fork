"""
Sanity: fused Hadamard + int4 KV write matches numpy Hadamard (float32 FWHT) +
bf16 round-trip + quantized_set_kv_int4_triton (same as separate steps in memory_pool).
"""

import math
import unittest

import numpy as np
import torch

from sglang.QuantKernel.fused_hadamard_int4_kv import (
    quantized_set_kv_int4_hadamard_fused_triton,
)
from sglang.srt.mem_cache.kv_quant_kernels import quantized_set_kv_int4_triton

try:
    from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
    from sglang.test.test_utils import CustomTestCase

    register_cuda_ci(est_time=120, suite="stage-b-test-small-1-gpu")
    register_amd_ci(est_time=120, suite="stage-b-test-small-1-gpu-amd")
except ImportError:
    CustomTestCase = unittest.TestCase


def _numpy_fwht_f32(a: np.ndarray) -> np.ndarray:
    a2 = np.array(a, dtype=np.float32, copy=True)
    n = len(a2)
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(h):
                u = a2[i + j]
                v = a2[i + j + h]
                a2[i + j] = u + v
                a2[i + j + h] = u - v
        h *= 2
    return a2


def _ref_hadamard_bf16(x: torch.Tensor, order: int) -> torch.Tensor:
    """Same scaling + FWHT as the fused kernel: ``x.float() * (1/sqrt(order))`` then FWHT."""
    o = x.shape[-1]
    shaped = x.view(*x.shape[:-1], o // order, order)
    flat = shaped.float().reshape(-1, order) * (1.0 / math.sqrt(order))
    out = np.stack([_numpy_fwht_f32(flat[i].cpu().numpy()) for i in range(flat.shape[0])])
    y = torch.from_numpy(out).to(x.device).to(torch.bfloat16).reshape_as(shaped)
    return y.reshape_as(x)


class TestFusedHadamardInt4KV(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")

    def _run_case(
        self,
        num_tokens: int,
        num_heads: int,
        head_dim: int,
        cache_size: int,
        order: int,
        rotate_v: bool,
    ):
        device = torch.device("cuda")
        dtype = torch.bfloat16
        torch.manual_seed(0)

        cache_k = torch.randn(
            num_tokens, num_heads, head_dim, device=device, dtype=dtype
        )
        cache_v = torch.randn(
            num_tokens, num_heads, head_dim, device=device, dtype=dtype
        )
        loc = torch.randperm(cache_size, device=device, dtype=torch.int32)[
            :num_tokens
        ]

        k_f = torch.zeros(
            cache_size, num_heads, head_dim // 2, device=device, dtype=torch.uint8
        )
        v_f = torch.zeros_like(k_f)
        ks_f = torch.zeros(
            cache_size, num_heads, 2, device=device, dtype=torch.float32
        )
        vs_f = torch.zeros_like(ks_f)

        quantized_set_kv_int4_hadamard_fused_triton(
            cache_k,
            cache_v,
            loc,
            k_f,
            v_f,
            ks_f,
            vs_f,
            order,
            rotate_v=rotate_v,
        )

        k_r = _ref_hadamard_bf16(cache_k, order)
        v_r = _ref_hadamard_bf16(cache_v, order) if rotate_v else cache_v

        k_s = torch.zeros_like(k_f)
        v_s = torch.zeros_like(v_f)
        ks_s = torch.zeros_like(ks_f)
        vs_s = torch.zeros_like(vs_f)
        quantized_set_kv_int4_triton(
            k_r, v_r, loc, k_s, v_s, ks_s, vs_s
        )

        self.assertTrue(
            torch.equal(k_f, k_s), "K packed mismatch vs reference pipeline"
        )
        self.assertTrue(
            torch.equal(v_f, v_s), "V packed mismatch vs reference pipeline"
        )
        self.assertTrue(
            torch.equal(ks_f, ks_s), "K scales mismatch vs reference pipeline"
        )
        self.assertTrue(
            torch.equal(vs_f, vs_s), "V scales mismatch vs reference pipeline"
        )

    def test_order16_rotate_both(self):
        self._run_case(5, 4, 128, 2048, 16, True)

    def test_order8_rotate_both(self):
        self._run_case(3, 2, 64, 512, 8, True)

    def test_order16_k_only(self):
        self._run_case(4, 2, 128, 1024, 16, False)


if __name__ == "__main__":
    unittest.main()
