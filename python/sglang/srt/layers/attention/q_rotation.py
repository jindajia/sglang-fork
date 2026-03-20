from __future__ import annotations

import logging
import os
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def _parse_compute_dtype(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in ("float64", "fp64", "double"):
        return torch.float64
    if normalized in ("float32", "fp32", "single"):
        return torch.float32
    if normalized in ("bfloat16", "bf16"):
        return torch.bfloat16
    raise ValueError(
        "Unsupported SGLANG_Q_ROTATION_COMPUTE_DTYPE "
        f"'{name}'. Expected float64, float32, or bfloat16."
    )


class QRotationManager:
    def __init__(self) -> None:
        self.path = os.environ.get("SGLANG_Q_ROTATION_PATH")
        self.compute_dtype = _parse_compute_dtype(
            os.environ.get("SGLANG_Q_ROTATION_COMPUTE_DTYPE", "float32")
        )
        self.enabled = bool(self.path)
        self.grouping: Optional[str] = None
        self._cpu_layers: dict[int, torch.Tensor] = {}
        self._device_cache: dict[tuple[int, str], torch.Tensor] = {}
        self._loaded = False

    def _load_if_needed(self) -> None:
        if not self.enabled or self._loaded:
            return

        state = torch.load(self.path, map_location="cpu")
        layers = state.get("layers")
        if not isinstance(layers, dict) or not layers:
            raise ValueError(
                f"Invalid Q rotation file at {self.path}: missing non-empty 'layers'"
            )

        self.grouping = state.get("source_grouping", state.get("grouping", "layer"))
        for layer_id, layer_data in layers.items():
            rotation = layer_data["rotation"].to(dtype=self.compute_dtype)
            self._cpu_layers[int(layer_id)] = rotation.contiguous()

        self._loaded = True
        if os.environ.get("HADAMARD", "0") in ("1", "true", "True"):
            logger.warning(
                "Both SGLANG_Q_ROTATION_PATH and HADAMARD are enabled. "
                "This composes two rotations; set HADAMARD=0 if you want a clean comparison."
            )
        logger.info(
            "Loaded Q rotation from %s with grouping=%s, compute_dtype=%s",
            self.path,
            self.grouping,
            self.compute_dtype,
        )

    def get_rotation(self, layer_id: int, device: torch.device) -> tuple[torch.Tensor, str]:
        self._load_if_needed()
        rotation = self._cpu_layers.get(layer_id)
        if rotation is None:
            raise KeyError(
                f"Layer {layer_id} not found in Q rotation file {self.path}"
            )

        cache_key = (layer_id, str(device))
        if cache_key not in self._device_cache:
            self._device_cache[cache_key] = rotation.to(device=device, copy=True)
        return self._device_cache[cache_key], self.grouping or "layer"


_Q_ROTATION_MANAGER = QRotationManager()


def _apply_layer_rotation(
    q: torch.Tensor,
    k: torch.Tensor,
    rotation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.matmul(q, rotation), torch.matmul(k, rotation)


def _apply_head_rotation(
    q: torch.Tensor,
    k: torch.Tensor,
    rotation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if rotation.shape[0] != q.shape[1]:
        raise ValueError(
            "Head-wise Q rotation expects rotation.shape[0] == num_q_heads, "
            f"got {rotation.shape[0]} and {q.shape[1]}"
        )
    if q.shape[1] != k.shape[1]:
        raise ValueError(
            "Head-wise Q rotation requires q and k to have the same number of heads. "
            f"Got q_heads={q.shape[1]}, k_heads={k.shape[1]}"
        )
    return (
        torch.einsum("thd,hdf->thf", q, rotation),
        torch.einsum("thd,hdf->thf", k, rotation),
    )


def _apply_kv_group_rotation(
    q: torch.Tensor,
    k: torch.Tensor,
    rotation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_groups = rotation.shape[0]
    if k.shape[1] != num_groups:
        raise ValueError(
            "KV-group Q rotation expects rotation.shape[0] == num_kv_heads, "
            f"got {rotation.shape[0]} and {k.shape[1]}"
        )
    if q.shape[1] % num_groups != 0:
        raise ValueError(
            f"num_q_heads ({q.shape[1]}) must be divisible by num_groups ({num_groups})"
        )

    group_size = q.shape[1] // num_groups
    q_grouped = q.reshape(q.shape[0], num_groups, group_size, q.shape[-1])
    q_rotated = torch.einsum("tghd,gdf->tghf", q_grouped, rotation).reshape_as(q)
    k_rotated = torch.einsum("tgd,gdf->tgf", k, rotation)
    return q_rotated, k_rotated


@torch._dynamo.disable()
def maybe_apply_qk_rotation(
    q: torch.Tensor,
    k: Optional[torch.Tensor],
    *,
    layer_id: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    if not _Q_ROTATION_MANAGER.enabled or k is None:
        return q, k

    rotation, grouping = _Q_ROTATION_MANAGER.get_rotation(layer_id, q.device)

    q_shape = q.shape
    k_shape = k.shape
    q_work = q.reshape(-1, num_q_heads, head_dim).to(dtype=rotation.dtype)
    k_work = k.reshape(-1, num_kv_heads, head_dim).to(dtype=rotation.dtype)

    if grouping == "layer":
        q_rotated, k_rotated = _apply_layer_rotation(q_work, k_work, rotation)
    elif grouping == "head":
        q_rotated, k_rotated = _apply_head_rotation(q_work, k_work, rotation)
    elif grouping == "kv_group":
        q_rotated, k_rotated = _apply_kv_group_rotation(q_work, k_work, rotation)
    else:
        raise ValueError(
            f"Unsupported Q rotation grouping '{grouping}' in {_Q_ROTATION_MANAGER.path}"
        )

    q_rotated = q_rotated.to(dtype=q.dtype).reshape(q_shape)
    k_rotated = k_rotated.to(dtype=k.dtype).reshape(k_shape)
    return q_rotated, k_rotated
