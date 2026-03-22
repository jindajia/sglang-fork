#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any, Optional

import torch


LAYER_DIR_RE = re.compile(r"layer_(\d+)$")
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute high-precision Q/K rotation matrices either from precomputed "
            "Q statistics or from raw dumped Q/K chunks with exact "
            "Hadamard+INT4-aware surrogate objectives."
        )
    )
    parser.add_argument(
        "--objective",
        choices=("matrix", "q_weighted_logit", "teacher_weighted_logit"),
        default="matrix",
        help=(
            "matrix: diagonalize a precomputed PSD matrix from q_statistics_*.pt. "
            "q_weighted_logit: build a new PSD matrix from raw dumped q/k chunks. "
            "teacher_weighted_logit: same as q_weighted_logit but multiply the "
            "sampled query losses by external teacher/task weights."
        ),
    )
    parser.add_argument(
        "--statistics-path",
        type=Path,
        default=None,
        help="Path to q_statistics_*.pt generated from dumped Q chunks.",
    )
    parser.add_argument(
        "--tensor-path",
        type=Path,
        default=None,
        help=(
            "Root dump directory containing aligned layer_*/q/*.pt and "
            "layer_*/k/*.pt chunks."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help=(
            "Path to save the rotation file. Defaults to q_rotation_<grouping>.pt "
            "for matrix mode and q_rotation_<grouping>_<objective>.pt for "
            "q_weighted_logit mode."
        ),
    )
    parser.add_argument(
        "--save-objective-matrix-path",
        type=Path,
        default=None,
        help="Optional path to save the intermediate objective-matrix state.",
    )
    parser.add_argument(
        "--grouping",
        choices=("layer", "head", "kv_group"),
        default=None,
        help=(
            "Grouping of the output rotation. Defaults to the grouping stored in "
            "q_statistics_*.pt for matrix mode and to layer for q_weighted_logit mode."
        ),
    )
    parser.add_argument(
        "--matrix-key",
        choices=("covariance", "second_moment", "objective_matrix"),
        default="covariance",
        help=(
            "Which PSD matrix to diagonalize in matrix mode. Use covariance for "
            "centered Q, second_moment for raw E[q q^T], or objective_matrix for "
            "a saved q_weighted_logit builder state."
        ),
    )
    parser.add_argument(
        "--damp-ratio",
        type=float,
        default=0.0,
        help=(
            "Optional CARE/TransMLA-style diagonal damping ratio. A value like "
            "0.01 matches their PCA helpers, while 0.0 avoids perturbing the input "
            "matrix."
        ),
    )
    parser.add_argument(
        "--save-dtype",
        choices=("float64", "float32"),
        default="float64",
        help=(
            "Dtype used to save rotation matrices. Default float64 keeps the "
            "highest precision for later experimentation."
        ),
    )
    parser.add_argument(
        "--accum-dtype",
        choices=("float64", "float32"),
        default="float64",
        help=(
            "Accumulation dtype for q_weighted_logit mode. float64 is slower but "
            "best for stable offline accumulation."
        ),
    )
    parser.add_argument(
        "--teacher-weight-path",
        type=Path,
        default=None,
        help=(
            "Optional root directory containing aligned layer_*/teacher/*.pt chunks. "
            "Each chunk can be shaped [T], [T, num_kv_heads], or [T, num_q_heads] "
            "and is used by --objective=teacher_weighted_logit, or by selection "
            "when you want to score candidates with task-aware query weights."
        ),
    )
    parser.add_argument(
        "--teacher-weight-normalization",
        choices=("none", "mean1", "sum1"),
        default="mean1",
        help=(
            "How to normalize each loaded teacher-weight chunk before it is applied "
            "to sampled query losses."
        ),
    )
    parser.add_argument(
        "--init-rotation-path",
        type=Path,
        default=None,
        help=(
            "Optional rotation file used as the reference/init for q_weighted_logit "
            "mode. This is typically the current best second_moment+damp baseline."
        ),
    )
    parser.add_argument(
        "--hadamard-order",
        type=int,
        default=16,
        help="Block Hadamard order used by the runtime. Default matches HADAMARD_ORDER=16.",
    )
    parser.add_argument(
        "--importance-mode",
        choices=("uniform", "softmax", "topk"),
        default="softmax",
        help=(
            "How to weight causal query-key pairs inside the q_weighted_logit "
            "surrogate and exact offline loss."
        ),
    )
    parser.add_argument(
        "--softmax-temperature",
        type=float,
        default=1.0,
        help="Temperature used when --importance-mode=softmax.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=8,
        help="Top-k used when --importance-mode=topk.",
    )
    parser.add_argument(
        "--max-query-samples-per-head",
        type=int,
        default=128,
        help=(
            "Maximum number of query positions sampled per head within each chunk "
            "for q_weighted_logit accumulation and optional selection."
        ),
    )
    parser.add_argument(
        "--max-chunks-per-layer",
        type=int,
        default=None,
        help="Optional cap on how many chunk ids to read per layer in q_weighted_logit mode.",
    )
    parser.add_argument(
        "--chunk-stride",
        type=int,
        default=1,
        help="Read every Nth chunk when building the q_weighted_logit objective.",
    )
    parser.add_argument(
        "--causal-window",
        type=int,
        default=0,
        help=(
            "If >0, only keys in the last causal-window positions are considered "
            "for each sampled query."
        ),
    )
    parser.add_argument(
        "--rotation-strength",
        type=float,
        default=1.0,
        help=(
            "Orthogonal interpolation strength between identity and the learned "
            "rotation. 1.0 keeps the full rotation; 0.0 becomes identity."
        ),
    )
    parser.add_argument(
        "--rotation-structure",
        choices=(
            "dense",
            "block_diagonal",
            "permuted_block_diagonal",
            "signed_diagonal",
        ),
        default="dense",
        help=(
            "Optional structural constraint imposed on the learned orthogonal "
            "rotation before export."
        ),
    )
    parser.add_argument(
        "--rotation-block-size",
        type=int,
        default=16,
        help=(
            "Block size used by the block-diagonal structured rotation families. "
            "Default 16 matches the current H16 grouping."
        ),
    )
    parser.add_argument(
        "--permutation-mode",
        choices=("diag_round_robin",),
        default="diag_round_robin",
        help=(
            "Permutation strategy used by --rotation-structure=permuted_block_diagonal."
        ),
    )
    parser.add_argument(
        "--selection-mode",
        choices=("off", "if_better"),
        default="off",
        help=(
            "Optional ablation that keeps only layers/groups whose exact offline "
            "q_weighted_logit loss improves over identity."
        ),
    )
    parser.add_argument(
        "--min-relative-improvement",
        type=float,
        default=0.0,
        help=(
            "Minimum relative improvement required by --selection-mode=if_better. "
            "For example, 0.01 requires at least a 1%% loss reduction."
        ),
    )
    args = parser.parse_args()

    if args.damp_ratio < 0.0:
        raise ValueError("--damp-ratio must be >= 0")
    if args.rotation_strength < 0.0 or args.rotation_strength > 1.0:
        raise ValueError("--rotation-strength must be in [0, 1]")
    if args.softmax_temperature <= 0.0:
        raise ValueError("--softmax-temperature must be > 0")
    if args.topk < 1:
        raise ValueError("--topk must be >= 1")
    if args.max_query_samples_per_head < 1:
        raise ValueError("--max-query-samples-per-head must be >= 1")
    if args.chunk_stride < 1:
        raise ValueError("--chunk-stride must be >= 1")
    if args.causal_window < 0:
        raise ValueError("--causal-window must be >= 0")
    if args.min_relative_improvement < 0.0:
        raise ValueError("--min-relative-improvement must be >= 0")
    if args.rotation_block_size < 1:
        raise ValueError("--rotation-block-size must be >= 1")

    if args.objective == "matrix":
        if args.statistics_path is None:
            raise ValueError("--statistics-path is required when --objective=matrix")
    else:
        if args.tensor_path is None:
            raise ValueError(
                f"--tensor-path is required when --objective={args.objective}"
            )
        if args.objective == "teacher_weighted_logit" and args.teacher_weight_path is None:
            raise ValueError(
                "--teacher-weight-path is required when "
                "--objective=teacher_weighted_logit"
            )
        if args.grouping is None:
            args.grouping = "layer"

    return args


def infer_output_path(
    *,
    objective: str,
    grouping: str,
    statistics_path: Optional[Path],
    tensor_path: Optional[Path],
) -> Path:
    if objective == "matrix":
        assert statistics_path is not None
        return statistics_path.with_name(f"q_rotation_{grouping}.pt")
    assert tensor_path is not None
    return tensor_path / f"q_rotation_{grouping}_{objective}.pt"


def canonicalize_eigenvector_signs(eigenvectors: torch.Tensor) -> torch.Tensor:
    """Make eigenvector signs deterministic across runs."""
    vecs = eigenvectors.clone()
    for col in range(vecs.shape[1]):
        column = vecs[:, col]
        pivot = torch.argmax(column.abs()).item()
        if column[pivot] < 0:
            vecs[:, col] = -vecs[:, col]
    return vecs


def decompose_psd_matrix(
    matrix: torch.Tensor,
    damp_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    x = matrix.detach().to(device="cpu", dtype=torch.float64).contiguous()
    symmetry_error = (x - x.mT).abs().max().item()
    x = 0.5 * (x + x.mT)

    damp_value = 0.0
    if damp_ratio != 0.0:
        mean_diag = torch.diag(x).mean().item()
        damp_value = damp_ratio * mean_diag
        diag = torch.arange(x.shape[-1], device=x.device)
        x[diag, diag] += damp_value

    evals, evecs = torch.linalg.eigh(x)
    order = torch.argsort(evals, descending=True)
    evals = evals[order].contiguous()
    evecs = evecs[:, order].contiguous()
    evecs = canonicalize_eigenvector_signs(evecs)

    identity = torch.eye(evecs.shape[1], dtype=evecs.dtype)
    orthogonality_error = (evecs.mT @ evecs - identity).abs().max().item()
    reconstruction = evecs @ torch.diag(evals) @ evecs.mT
    denom = max(x.abs().max().item(), 1.0)
    reconstruction_error = (reconstruction - x).abs().max().item() / denom

    metrics = {
        "symmetry_error_before_symmetrize": symmetry_error,
        "damp_value": damp_value,
        "orthogonality_error": orthogonality_error,
        "relative_reconstruction_error": reconstruction_error,
    }
    return evals, evecs, metrics


def build_balanced_block_permutation(
    scores: torch.Tensor,
    block_size: int,
    permutation_mode: str,
) -> torch.Tensor:
    dim = scores.numel()
    if dim % block_size != 0:
        raise ValueError(
            f"rotation-block-size ({block_size}) must divide dimension ({dim})"
        )
    if permutation_mode != "diag_round_robin":
        raise ValueError(f"Unsupported permutation-mode: {permutation_mode}")

    num_blocks = dim // block_size
    order = torch.argsort(scores, descending=True).tolist()
    buckets = [[] for _ in range(num_blocks)]
    for idx, dim_idx in enumerate(order):
        buckets[idx % num_blocks].append(dim_idx)
    permutation = [dim_idx for bucket in buckets for dim_idx in bucket]
    return torch.tensor(permutation, dtype=torch.long)


def _unpermute_rotation(rotation: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
    restored = torch.empty_like(rotation)
    restored[permutation[:, None], permutation[None, :]] = rotation
    return restored


def decompose_block_psd_matrix(
    matrix: torch.Tensor,
    damp_ratio: float,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    dim = matrix.shape[0]
    if dim % block_size != 0:
        raise ValueError(
            f"rotation-block-size ({block_size}) must divide dimension ({dim})"
        )

    rotation = torch.zeros_like(matrix, dtype=torch.float64)
    evals: list[torch.Tensor] = []
    block_metrics = []
    for block_start in range(0, dim, block_size):
        block_end = block_start + block_size
        block = matrix[block_start:block_end, block_start:block_end]
        block_evals, block_evecs, metrics = decompose_psd_matrix(block, damp_ratio)
        rotation[block_start:block_end, block_start:block_end] = block_evecs
        evals.append(block_evals)
        block_metrics.append(metrics)

    return (
        torch.cat(evals, dim=0).contiguous(),
        rotation.contiguous(),
        {
            "structure": "block_diagonal",
            "block_size": block_size,
            "blocks": block_metrics,
        },
    )


def decompose_structured_psd_matrix(
    matrix: torch.Tensor,
    *,
    damp_ratio: float,
    rotation_structure: str,
    rotation_block_size: int,
    permutation_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if rotation_structure == "dense":
        evals, evecs, metrics = decompose_psd_matrix(matrix, damp_ratio)
        metrics["structure"] = "dense"
        return evals, evecs, metrics

    if rotation_structure == "block_diagonal":
        return decompose_block_psd_matrix(matrix, damp_ratio, rotation_block_size)

    if rotation_structure == "permuted_block_diagonal":
        sym = 0.5 * (matrix + matrix.mT)
        scores = torch.diag(sym).abs()
        permutation = build_balanced_block_permutation(
            scores,
            rotation_block_size,
            permutation_mode,
        )
        permuted = sym.index_select(0, permutation).index_select(1, permutation)
        evals, block_rotation, metrics = decompose_block_psd_matrix(
            permuted,
            damp_ratio,
            rotation_block_size,
        )
        metrics["structure"] = "permuted_block_diagonal"
        metrics["permutation_mode"] = permutation_mode
        metrics["permutation"] = permutation.tolist()
        return evals, _unpermute_rotation(block_rotation, permutation), metrics

    if rotation_structure == "signed_diagonal":
        evals, dense_rotation, metrics = decompose_psd_matrix(matrix, damp_ratio)
        signs = torch.sign(torch.diagonal(dense_rotation))
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        rotation = torch.diag(signs)
        metrics["structure"] = "signed_diagonal"
        metrics["positive_sign_fraction"] = float((signs > 0).to(torch.float64).mean().item())
        return evals, rotation.contiguous(), metrics

    raise ValueError(f"Unsupported rotation-structure: {rotation_structure}")


def process_layer_matrix(
    matrix: torch.Tensor,
    *,
    damp_ratio: float,
    rotation_structure: str,
    rotation_block_size: int,
    permutation_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if matrix.ndim == 2:
        return decompose_structured_psd_matrix(
            matrix,
            damp_ratio=damp_ratio,
            rotation_structure=rotation_structure,
            rotation_block_size=rotation_block_size,
            permutation_mode=permutation_mode,
        )

    if matrix.ndim != 3:
        raise ValueError(
            f"Expected matrix rank 2 or 3, got shape {tuple(matrix.shape)}"
        )

    eval_list = []
    vec_list = []
    metric_list = []
    for idx in range(matrix.shape[0]):
        evals, evecs, metrics = decompose_structured_psd_matrix(
            matrix[idx],
            damp_ratio=damp_ratio,
            rotation_structure=rotation_structure,
            rotation_block_size=rotation_block_size,
            permutation_mode=permutation_mode,
        )
        eval_list.append(evals)
        vec_list.append(evecs)
        metric_list.append(metrics)

    return (
        torch.stack(eval_list, dim=0).contiguous(),
        torch.stack(vec_list, dim=0).contiguous(),
        {"groups": metric_list, "structure": rotation_structure},
    )


def build_normalized_hadamard(order: int) -> torch.Tensor:
    if order < 1 or order & (order - 1):
        raise ValueError(f"Hadamard order must be a positive power of two, got {order}")
    if order == 1:
        return torch.ones((1, 1), dtype=torch.float64)
    half = build_normalized_hadamard(order // 2)
    top = torch.cat([half, half], dim=1)
    bottom = torch.cat([half, -half], dim=1)
    return torch.cat([top, bottom], dim=0) / math.sqrt(2.0)


def build_block_hadamard(head_dim: int, hadamard_order: int) -> torch.Tensor:
    if head_dim % hadamard_order != 0:
        raise ValueError(
            f"head_dim ({head_dim}) must be divisible by hadamard_order ({hadamard_order})"
        )
    block = build_normalized_hadamard(hadamard_order)
    num_blocks = head_dim // hadamard_order
    return torch.block_diag(*([block] * num_blocks)).contiguous()


def apply_block_hadamard(x: torch.Tensor, block_hadamard: torch.Tensor) -> torch.Tensor:
    original_shape = x.shape
    flat = x.reshape(-1, original_shape[-1]).to(dtype=block_hadamard.dtype)
    return (flat @ block_hadamard).reshape(original_shape)


def quantize_dequantize_int4(x: torch.Tensor) -> torch.Tensor:
    flat = x.reshape(-1, x.shape[-1])
    x_min = flat.amin(dim=-1, keepdim=True)
    x_max = flat.amax(dim=-1, keepdim=True)
    scale = (x_max - x_min).clamp_min(1e-8) / 15.0
    zero = -x_min / scale
    q = torch.round(flat / scale + zero).clamp_(0.0, 15.0)
    dq = (q - zero) * scale
    return dq.reshape_as(x)


def load_rotation_state(path: Path) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu")
    grouping = state.get("source_grouping", state.get("grouping", "layer"))
    layers = state.get("layers")
    if not isinstance(layers, dict) or not layers:
        raise ValueError(f"Rotation file does not contain a non-empty 'layers': {path}")
    result_layers: dict[int, torch.Tensor] = {}
    for layer_id, layer_data in layers.items():
        rotation = layer_data.get("rotation")
        if rotation is None:
            raise ValueError(f"Rotation file is missing layer {layer_id} rotation: {path}")
        result_layers[int(layer_id)] = rotation.to(dtype=torch.float64).contiguous()
    return {"path": str(path), "grouping": grouping, "layers": result_layers}


def identity_rotation(grouping: str, num_q_heads: int, num_kv_heads: int, head_dim: int) -> torch.Tensor:
    eye = torch.eye(head_dim, dtype=torch.float64)
    if grouping == "layer":
        return eye
    if grouping == "head":
        if num_q_heads != num_kv_heads:
            raise ValueError(
                "head grouping requires num_q_heads == num_kv_heads, got "
                f"{num_q_heads} and {num_kv_heads}"
            )
        return eye.expand(num_q_heads, -1, -1).clone()
    if grouping == "kv_group":
        return eye.expand(num_kv_heads, -1, -1).clone()
    raise ValueError(f"Unsupported grouping: {grouping}")


def _apply_layer_rotation(x: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    return torch.matmul(x, rotation)


def _apply_head_rotation(x: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    if rotation.shape[0] != x.shape[1]:
        raise ValueError(
            f"Head rotation expects rotation.shape[0] == num_heads, got {rotation.shape[0]} and {x.shape[1]}"
        )
    return torch.einsum("thd,hdf->thf", x, rotation)


def _apply_q_kv_group_rotation(
    q: torch.Tensor,
    rotation: torch.Tensor,
    num_kv_heads: int,
) -> torch.Tensor:
    num_groups = rotation.shape[0]
    if num_groups != num_kv_heads:
        raise ValueError(
            "KV-group rotation expects rotation.shape[0] == num_kv_heads, "
            f"got {rotation.shape[0]} and {num_kv_heads}"
        )
    if q.shape[1] % num_groups != 0:
        raise ValueError(
            f"num_q_heads ({q.shape[1]}) must be divisible by num_groups ({num_groups})"
        )
    group_size = q.shape[1] // num_groups
    q_grouped = q.reshape(q.shape[0], num_groups, group_size, q.shape[-1])
    return torch.einsum("tghd,gdf->tghf", q_grouped, rotation).reshape_as(q)


def _apply_k_kv_group_rotation(
    k: torch.Tensor,
    rotation: torch.Tensor,
    num_kv_heads: int,
) -> torch.Tensor:
    num_groups = rotation.shape[0]
    if num_groups != num_kv_heads:
        raise ValueError(
            "KV-group rotation expects rotation.shape[0] == num_kv_heads, "
            f"got {rotation.shape[0]} and {num_kv_heads}"
        )
    if k.shape[1] != num_groups:
        raise ValueError(
            f"num_kv_heads ({k.shape[1]}) must match num_groups ({num_groups})"
        )
    return torch.einsum("tgd,gdf->tgf", k, rotation)


def apply_rotation_to_q_chunk(
    q: torch.Tensor,
    rotation: Optional[torch.Tensor],
    grouping: str,
    num_kv_heads: int,
) -> torch.Tensor:
    if rotation is None:
        return q
    if grouping == "layer":
        return _apply_layer_rotation(q, rotation)
    if grouping == "head":
        return _apply_head_rotation(q, rotation)
    if grouping == "kv_group":
        return _apply_q_kv_group_rotation(q, rotation, num_kv_heads)
    raise ValueError(f"Unsupported grouping: {grouping}")


def apply_rotation_to_k_chunk(
    k: torch.Tensor,
    rotation: Optional[torch.Tensor],
    grouping: str,
    num_kv_heads: int,
) -> torch.Tensor:
    if rotation is None:
        return k
    if grouping == "layer":
        return _apply_layer_rotation(k, rotation)
    if grouping == "head":
        return _apply_head_rotation(k, rotation)
    if grouping == "kv_group":
        return _apply_k_kv_group_rotation(k, rotation, num_kv_heads)
    raise ValueError(f"Unsupported grouping: {grouping}")


def discover_layer_ids(tensor_path: Path) -> list[int]:
    layer_ids: list[int] = []
    for child in tensor_path.iterdir():
        if not child.is_dir():
            continue
        match = LAYER_DIR_RE.match(child.name)
        if match:
            layer_ids.append(int(match.group(1)))
    return sorted(layer_ids)


def list_aligned_chunk_ids(layer_dir: Path) -> list[int]:
    q_dir = layer_dir / "q"
    k_dir = layer_dir / "k"
    if not q_dir.is_dir() or not k_dir.is_dir():
        return []

    q_ids = {int(path.stem) for path in q_dir.iterdir() if path.suffix == ".pt" and path.stem.isdigit()}
    k_ids = {int(path.stem) for path in k_dir.iterdir() if path.suffix == ".pt" and path.stem.isdigit()}
    return sorted(q_ids & k_ids)


def infer_teacher_weight_granularity(
    weights: torch.Tensor,
    *,
    num_tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
) -> str:
    if weights.ndim == 1 and weights.shape[0] == num_tokens:
        return "token"
    if weights.ndim == 2 and weights.shape == (num_tokens, num_kv_heads):
        return "kv_group"
    if weights.ndim == 2 and weights.shape == (num_tokens, num_q_heads):
        return "head"
    raise ValueError(
        "Teacher weights must be shaped [T], [T, num_kv_heads], or [T, num_q_heads], "
        f"got {tuple(weights.shape)}"
    )


def normalize_teacher_weights(
    weights: torch.Tensor,
    normalization: str,
) -> torch.Tensor:
    if normalization == "none":
        return weights
    if normalization == "mean1":
        return weights / weights.mean().clamp_min(EPS)
    if normalization == "sum1":
        return weights / weights.sum().clamp_min(EPS) * weights.numel()
    raise ValueError(f"Unsupported teacher-weight-normalization: {normalization}")


def load_teacher_weight_chunk(
    teacher_weight_root: Path,
    *,
    layer_id: int,
    chunk_id: int,
    num_tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    normalization: str,
) -> tuple[torch.Tensor, str]:
    teacher_path = teacher_weight_root / f"layer_{layer_id}" / "teacher" / f"{chunk_id}.pt"
    if not teacher_path.is_file():
        raise FileNotFoundError(f"Teacher weight chunk not found: {teacher_path}")

    payload = torch.load(teacher_path, map_location="cpu")
    if isinstance(payload, dict):
        weights = payload.get("weights")
        granularity = payload.get("granularity")
        if weights is None:
            raise ValueError(f"Teacher weight chunk is missing 'weights': {teacher_path}")
    else:
        weights = payload
        granularity = None

    if not isinstance(weights, torch.Tensor):
        raise ValueError(f"Teacher weight chunk must contain a tensor: {teacher_path}")
    weights = weights.to(dtype=torch.float64)
    inferred = infer_teacher_weight_granularity(
        weights,
        num_tokens=num_tokens,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
    )
    if granularity is None:
        granularity = inferred
    elif granularity != inferred:
        raise ValueError(
            f"Teacher weight chunk granularity mismatch for {teacher_path}: "
            f"declared {granularity}, inferred {inferred}"
        )
    return normalize_teacher_weights(weights, normalization), granularity


def get_sampled_teacher_query_weights(
    weights: torch.Tensor,
    *,
    granularity: str,
    query_positions: torch.Tensor,
    kv_idx: int,
    start: int,
    end: int,
) -> torch.Tensor:
    sampled = weights.index_select(0, query_positions)
    if granularity == "token":
        return sampled.unsqueeze(-1)
    if granularity == "kv_group":
        return sampled[:, kv_idx].unsqueeze(-1)
    if granularity == "head":
        return sampled[:, start:end]
    raise ValueError(f"Unsupported teacher weight granularity: {granularity}")


def sample_query_indices(num_tokens: int, max_samples: int) -> torch.Tensor:
    if num_tokens <= max_samples:
        return torch.arange(num_tokens, dtype=torch.long)
    return (
        torch.linspace(0, num_tokens - 1, steps=max_samples, dtype=torch.float64)
        .round()
        .to(dtype=torch.long)
        .unique(sorted=True)
    )


def build_causal_mask(
    query_positions: torch.Tensor,
    num_keys: int,
    causal_window: int,
) -> torch.Tensor:
    key_positions = torch.arange(num_keys, dtype=query_positions.dtype)
    mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
    if causal_window > 0:
        start = (query_positions - causal_window + 1).clamp_min(0)
        mask &= key_positions.unsqueeze(0) >= start.unsqueeze(1)
    return mask


def compute_importance_weights(
    logits: torch.Tensor,
    mask: torch.Tensor,
    *,
    importance_mode: str,
    softmax_temperature: float,
    topk: int,
) -> torch.Tensor:
    dtype = logits.dtype
    if importance_mode == "uniform":
        weights = mask.to(dtype=dtype)
    elif importance_mode == "softmax":
        neg_inf = torch.finfo(dtype).min
        masked_logits = (logits / softmax_temperature).masked_fill(~mask, neg_inf)
        weights = torch.softmax(masked_logits, dim=-1)
        weights = torch.where(mask, weights, torch.zeros_like(weights))
    elif importance_mode == "topk":
        neg_inf = torch.finfo(dtype).min
        masked_logits = logits.masked_fill(~mask, neg_inf)
        k = min(topk, logits.shape[-1])
        topk_idx = torch.topk(masked_logits, k=k, dim=-1).indices
        weights = torch.zeros_like(logits)
        weights.scatter_(-1, topk_idx, 1.0)
        weights = weights * mask.to(dtype=dtype)
    else:
        raise ValueError(f"Unsupported importance_mode: {importance_mode}")

    denom = weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return weights / denom


def compute_weighted_query_losses(
    q_rot_sampled: torch.Tensor,
    k_rot: torch.Tensor,
    residual: torch.Tensor,
    query_positions: torch.Tensor,
    *,
    importance_mode: str,
    softmax_temperature: float,
    topk: int,
    causal_window: int,
) -> torch.Tensor:
    head_dim = q_rot_sampled.shape[-1]
    logits = torch.einsum("shd,td->sht", q_rot_sampled, k_rot) / math.sqrt(head_dim)
    logit_error_sq = (
        torch.einsum("shd,td->sht", q_rot_sampled, residual).square() / head_dim
    )
    mask = build_causal_mask(query_positions, k_rot.shape[0], causal_window)
    mask = mask.unsqueeze(1).expand_as(logits)
    importance = compute_importance_weights(
        logits,
        mask,
        importance_mode=importance_mode,
        softmax_temperature=softmax_temperature,
        topk=topk,
    )
    return (importance * logit_error_sq).sum(dim=-1)


def init_grouped_accumulators(
    grouping: str,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    accum_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if grouping == "layer":
        matrix = torch.zeros((head_dim, head_dim), dtype=accum_dtype)
        weight_sum = torch.zeros((), dtype=accum_dtype)
        query_count = torch.zeros((), dtype=torch.int64)
        return matrix, weight_sum, query_count
    if grouping == "head":
        if num_q_heads != num_kv_heads:
            raise ValueError(
                "head grouping requires num_q_heads == num_kv_heads, got "
                f"{num_q_heads} and {num_kv_heads}"
            )
        matrix = torch.zeros((num_q_heads, head_dim, head_dim), dtype=accum_dtype)
        weight_sum = torch.zeros((num_q_heads,), dtype=accum_dtype)
        query_count = torch.zeros((num_q_heads,), dtype=torch.int64)
        return matrix, weight_sum, query_count
    if grouping == "kv_group":
        matrix = torch.zeros((num_kv_heads, head_dim, head_dim), dtype=accum_dtype)
        weight_sum = torch.zeros((num_kv_heads,), dtype=accum_dtype)
        query_count = torch.zeros((num_kv_heads,), dtype=torch.int64)
        return matrix, weight_sum, query_count
    raise ValueError(f"Unsupported grouping: {grouping}")


def normalize_grouped_matrix(
    matrix: torch.Tensor,
    weight_sum: torch.Tensor,
) -> torch.Tensor:
    if matrix.ndim == 2:
        if weight_sum.item() <= 0:
            raise ValueError("q_weighted_logit produced zero total weight")
        normalized = matrix / weight_sum
        return 0.5 * (normalized + normalized.mT)

    if torch.any(weight_sum <= 0):
        raise ValueError("q_weighted_logit produced at least one zero-weight group")
    normalized = matrix / weight_sum.view(-1, 1, 1)
    return 0.5 * (normalized + normalized.transpose(-1, -2))


def get_layer_rotation(
    rotation_state: Optional[dict[str, Any]],
    *,
    layer_id: int,
) -> tuple[Optional[torch.Tensor], str]:
    if rotation_state is None:
        return None, "layer"
    return rotation_state["layers"].get(layer_id), rotation_state["grouping"]


def build_weighted_logit_matrix_state(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], str]:
    tensor_path = args.tensor_path.expanduser().resolve()
    if not tensor_path.is_dir():
        raise FileNotFoundError(f"tensor-path does not exist: {tensor_path}")
    objective_name = args.objective
    use_teacher_weights = objective_name == "teacher_weighted_logit"
    teacher_weight_root = (
        args.teacher_weight_path.expanduser().resolve()
        if args.teacher_weight_path is not None
        else None
    )
    if use_teacher_weights:
        assert teacher_weight_root is not None
        if not teacher_weight_root.is_dir():
            raise FileNotFoundError(
                f"teacher-weight-path does not exist: {teacher_weight_root}"
            )

    init_rotation_state = None
    if args.init_rotation_path is not None:
        init_rotation_state = load_rotation_state(args.init_rotation_path.expanduser().resolve())

    accum_dtype = getattr(torch, args.accum_dtype)
    layer_ids = discover_layer_ids(tensor_path)
    if not layer_ids:
        raise RuntimeError(f"No layer_* directories found under {tensor_path}")

    result: dict[str, Any] = {
        "format_version": 1,
        "objective": objective_name,
        "tensor_path": str(tensor_path),
        "grouping": args.grouping,
        "hadamard_order": args.hadamard_order,
        "importance_mode": args.importance_mode,
        "softmax_temperature": args.softmax_temperature,
        "topk": args.topk,
        "max_query_samples_per_head": args.max_query_samples_per_head,
        "causal_window": args.causal_window,
        "chunk_stride": args.chunk_stride,
        "accum_dtype": args.accum_dtype,
        "init_rotation_path": (
            str(args.init_rotation_path.expanduser().resolve())
            if args.init_rotation_path is not None
            else None
        ),
        "init_rotation_grouping": (
            init_rotation_state["grouping"] if init_rotation_state is not None else None
        ),
        "teacher_weight_path": str(teacher_weight_root) if teacher_weight_root is not None else None,
        "teacher_weight_normalization": args.teacher_weight_normalization,
        "layers": {},
    }

    block_hadamard: Optional[torch.Tensor] = None
    total_chunks = 0
    for layer_id in layer_ids:
        layer_dir = tensor_path / f"layer_{layer_id}"
        chunk_ids = list_aligned_chunk_ids(layer_dir)[:: args.chunk_stride]
        if args.max_chunks_per_layer is not None:
            chunk_ids = chunk_ids[: args.max_chunks_per_layer]
        if not chunk_ids:
            print(f"layer {layer_id}: no aligned q/k chunk files found, skip")
            continue

        first_q = torch.load(layer_dir / "q" / f"{chunk_ids[0]}.pt", map_location="cpu")
        first_k = torch.load(layer_dir / "k" / f"{chunk_ids[0]}.pt", map_location="cpu")
        if first_q.ndim != 3 or first_k.ndim != 3:
            raise ValueError(
                f"Expected q/k chunk rank 3, got {tuple(first_q.shape)} and {tuple(first_k.shape)}"
            )
        if first_q.shape[0] != first_k.shape[0] or first_q.shape[-1] != first_k.shape[-1]:
            raise ValueError(
                f"Layer {layer_id} q/k chunk shape mismatch: {tuple(first_q.shape)} vs {tuple(first_k.shape)}"
            )

        _, num_q_heads, head_dim = first_q.shape
        _, num_kv_heads, k_head_dim = first_k.shape
        if head_dim != k_head_dim:
            raise ValueError(
                f"Layer {layer_id} q/k head_dim mismatch: {head_dim} vs {k_head_dim}"
            )
        if num_q_heads % num_kv_heads != 0:
            raise ValueError(
                f"Layer {layer_id}: num_q_heads ({num_q_heads}) must be divisible by "
                f"num_kv_heads ({num_kv_heads})"
            )

        if block_hadamard is None or block_hadamard.shape[0] != head_dim:
            block_hadamard = build_block_hadamard(head_dim, args.hadamard_order)

        matrix_accum, weight_sum, query_count = init_grouped_accumulators(
            grouping=args.grouping,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            accum_dtype=accum_dtype,
        )
        group_size = num_q_heads // num_kv_heads
        ref_rotation, ref_grouping = get_layer_rotation(init_rotation_state, layer_id=layer_id)

        for chunk_id in chunk_ids:
            q_chunk = torch.load(layer_dir / "q" / f"{chunk_id}.pt", map_location="cpu")
            k_chunk = torch.load(layer_dir / "k" / f"{chunk_id}.pt", map_location="cpu")
            if q_chunk.ndim != 3 or q_chunk.shape[1:] != (num_q_heads, head_dim):
                raise ValueError(
                    f"Layer {layer_id} chunk {chunk_id} q shape mismatch: "
                    f"expected (*, {num_q_heads}, {head_dim}), got {tuple(q_chunk.shape)}"
                )
            if k_chunk.ndim != 3 or k_chunk.shape[1:] != (num_kv_heads, head_dim):
                raise ValueError(
                    f"Layer {layer_id} chunk {chunk_id} k shape mismatch: "
                    f"expected (*, {num_kv_heads}, {head_dim}), got {tuple(k_chunk.shape)}"
                )
            if q_chunk.shape[0] != k_chunk.shape[0]:
                raise ValueError(
                    f"Layer {layer_id} chunk {chunk_id} token count mismatch: "
                    f"{q_chunk.shape[0]} vs {k_chunk.shape[0]}"
                )

            q_h = apply_block_hadamard(q_chunk, block_hadamard).to(dtype=accum_dtype)
            k_h = apply_block_hadamard(k_chunk, block_hadamard).to(dtype=accum_dtype)
            q_ref = apply_rotation_to_q_chunk(q_h, ref_rotation, ref_grouping, num_kv_heads)
            k_ref = apply_rotation_to_k_chunk(k_h, ref_rotation, ref_grouping, num_kv_heads)
            k_dequant = quantize_dequantize_int4(k_ref)
            residual = k_dequant - k_ref

            num_tokens = q_chunk.shape[0]
            query_positions = sample_query_indices(num_tokens, args.max_query_samples_per_head)
            teacher_weights = None
            teacher_granularity = None
            if teacher_weight_root is not None:
                teacher_weights, teacher_granularity = load_teacher_weight_chunk(
                    teacher_weight_root,
                    layer_id=layer_id,
                    chunk_id=chunk_id,
                    num_tokens=num_tokens,
                    num_q_heads=num_q_heads,
                    num_kv_heads=num_kv_heads,
                    normalization=args.teacher_weight_normalization,
                )
            for kv_idx in range(num_kv_heads):
                start = kv_idx * group_size
                end = start + group_size
                q_pre_sampled = q_h[query_positions, start:end, :]
                q_ref_sampled = q_ref[query_positions, start:end, :]
                q_losses = compute_weighted_query_losses(
                    q_ref_sampled,
                    k_ref[:, kv_idx, :],
                    residual[:, kv_idx, :],
                    query_positions,
                    importance_mode=args.importance_mode,
                    softmax_temperature=args.softmax_temperature,
                    topk=args.topk,
                    causal_window=args.causal_window,
                )
                if teacher_weights is not None and teacher_granularity is not None:
                    q_losses = q_losses * get_sampled_teacher_query_weights(
                        teacher_weights,
                        granularity=teacher_granularity,
                        query_positions=query_positions,
                        kv_idx=kv_idx,
                        start=start,
                        end=end,
                    ).to(dtype=q_losses.dtype)

                if args.grouping == "layer":
                    q_flat = q_pre_sampled.reshape(-1, head_dim)
                    w_flat = q_losses.reshape(-1)
                    matrix_accum += torch.einsum("n,nd,nf->df", w_flat, q_flat, q_flat)
                    weight_sum += w_flat.sum()
                    query_count += w_flat.numel()
                elif args.grouping == "kv_group":
                    q_flat = q_pre_sampled.reshape(-1, head_dim)
                    w_flat = q_losses.reshape(-1)
                    matrix_accum[kv_idx] += torch.einsum("n,nd,nf->df", w_flat, q_flat, q_flat)
                    weight_sum[kv_idx] += w_flat.sum()
                    query_count[kv_idx] += w_flat.numel()
                else:
                    for local_idx in range(group_size):
                        head_idx = start + local_idx
                        q_head = q_pre_sampled[:, local_idx, :]
                        w_head = q_losses[:, local_idx]
                        matrix_accum[head_idx] += torch.einsum(
                            "n,nd,nf->df", w_head, q_head, q_head
                        )
                        weight_sum[head_idx] += w_head.sum()
                        query_count[head_idx] += w_head.numel()

        objective_matrix = normalize_grouped_matrix(matrix_accum, weight_sum)
        layer_result = {
            "layer_id": layer_id,
            "objective_matrix": objective_matrix.contiguous(),
            "weight_sum": weight_sum.clone(),
            "query_count": query_count.clone(),
            "head_dim": head_dim,
            "num_q_heads": num_q_heads,
            "num_kv_heads": num_kv_heads,
            "num_chunks": len(chunk_ids),
            "chunk_ids": chunk_ids,
        }
        result["layers"][layer_id] = layer_result
        total_chunks += len(chunk_ids)
        print(
            f"layer {layer_id}: chunks={len(chunk_ids)}, q_shape=(*, {num_q_heads}, {head_dim}), "
            f"k_shape=(*, {num_kv_heads}, {head_dim}), objective_shape={tuple(objective_matrix.shape)}"
        )

    if not result["layers"]:
        raise RuntimeError(f"No valid {objective_name} layers found under {tensor_path}")

    print(
        f"Built {objective_name} objective for {len(result['layers'])} layers "
        f"from {total_chunks} aligned q/k chunks"
    )
    return result, "objective_matrix"


def load_matrix_state_from_statistics(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], str]:
    statistics_path = args.statistics_path.expanduser().resolve()
    if not statistics_path.is_file():
        raise FileNotFoundError(f"statistics file not found: {statistics_path}")

    stats = torch.load(statistics_path, map_location="cpu")
    grouping = args.grouping or stats.get("grouping", "layer")
    layers = stats.get("layers")
    if not isinstance(layers, dict) or not layers:
        raise ValueError(
            "statistics file does not contain a non-empty 'layers' dictionary"
        )

    matrix_state = {
        "format_version": 1,
        "objective": "matrix",
        "source_statistics_path": str(statistics_path),
        "grouping": grouping,
        "matrix_key": args.matrix_key,
        "layers": {},
    }
    for layer_id, layer_data in layers.items():
        if args.matrix_key not in layer_data:
            raise KeyError(
                f"Layer {layer_id} does not contain matrix key '{args.matrix_key}'"
            )
        matrix_state["layers"][int(layer_id)] = dict(layer_data)
    return matrix_state, args.matrix_key


def identity_like_rotation(rotation: torch.Tensor) -> torch.Tensor:
    eye = torch.eye(rotation.shape[-1], dtype=rotation.dtype)
    if rotation.ndim == 2:
        return eye
    return eye.expand(rotation.shape[0], -1, -1).clone()


def _blend_rotation_single(rotation: torch.Tensor, strength: float) -> torch.Tensor:
    if strength == 1.0:
        return rotation
    if strength == 0.0:
        return torch.eye(rotation.shape[-1], dtype=rotation.dtype)
    eye = torch.eye(rotation.shape[-1], dtype=rotation.dtype)
    mixed = (1.0 - strength) * eye + strength * rotation
    u, _, vh = torch.linalg.svd(mixed)
    blended = u @ vh
    return canonicalize_eigenvector_signs(blended)


def apply_rotation_strength(rotation: torch.Tensor, strength: float) -> torch.Tensor:
    if strength == 1.0:
        return rotation
    if rotation.ndim == 2:
        return _blend_rotation_single(rotation, strength)
    return torch.stack(
        [_blend_rotation_single(rotation[idx], strength) for idx in range(rotation.shape[0])],
        dim=0,
    )


def evaluate_rotation_losses(
    *,
    tensor_path: Path,
    grouping: str,
    rotation_layers: dict[int, torch.Tensor],
    hadamard_order: int,
    importance_mode: str,
    softmax_temperature: float,
    topk: int,
    max_query_samples_per_head: int,
    causal_window: int,
    chunk_stride: int,
    max_chunks_per_layer: Optional[int],
    teacher_weight_path: Optional[Path],
    teacher_weight_normalization: str,
) -> dict[int, torch.Tensor]:
    layer_ids = discover_layer_ids(tensor_path)
    block_hadamard: Optional[torch.Tensor] = None
    losses_by_layer: dict[int, torch.Tensor] = {}

    for layer_id in layer_ids:
        layer_dir = tensor_path / f"layer_{layer_id}"
        chunk_ids = list_aligned_chunk_ids(layer_dir)[::chunk_stride]
        if max_chunks_per_layer is not None:
            chunk_ids = chunk_ids[:max_chunks_per_layer]
        if not chunk_ids:
            continue

        first_q = torch.load(layer_dir / "q" / f"{chunk_ids[0]}.pt", map_location="cpu")
        first_k = torch.load(layer_dir / "k" / f"{chunk_ids[0]}.pt", map_location="cpu")
        _, num_q_heads, head_dim = first_q.shape
        _, num_kv_heads, _ = first_k.shape
        group_size = num_q_heads // num_kv_heads

        if block_hadamard is None or block_hadamard.shape[0] != head_dim:
            block_hadamard = build_block_hadamard(head_dim, hadamard_order)

        rotation = rotation_layers.get(layer_id)
        if rotation is None:
            rotation = identity_rotation(grouping, num_q_heads, num_kv_heads, head_dim)
        rotation = rotation.to(dtype=torch.float64)

        if grouping == "layer":
            loss_sum = torch.zeros((), dtype=torch.float64)
            count = torch.zeros((), dtype=torch.int64)
        elif grouping == "kv_group":
            loss_sum = torch.zeros((num_kv_heads,), dtype=torch.float64)
            count = torch.zeros((num_kv_heads,), dtype=torch.int64)
        else:
            loss_sum = torch.zeros((num_q_heads,), dtype=torch.float64)
            count = torch.zeros((num_q_heads,), dtype=torch.int64)

        for chunk_id in chunk_ids:
            q_chunk = torch.load(layer_dir / "q" / f"{chunk_id}.pt", map_location="cpu")
            k_chunk = torch.load(layer_dir / "k" / f"{chunk_id}.pt", map_location="cpu")
            if q_chunk.ndim != 3 or q_chunk.shape[1:] != (num_q_heads, head_dim):
                raise ValueError(
                    f"Layer {layer_id} chunk {chunk_id} q shape mismatch: "
                    f"expected (*, {num_q_heads}, {head_dim}), got {tuple(q_chunk.shape)}"
                )
            if k_chunk.ndim != 3 or k_chunk.shape[1:] != (num_kv_heads, head_dim):
                raise ValueError(
                    f"Layer {layer_id} chunk {chunk_id} k shape mismatch: "
                    f"expected (*, {num_kv_heads}, {head_dim}), got {tuple(k_chunk.shape)}"
                )
            if q_chunk.shape[0] != k_chunk.shape[0]:
                raise ValueError(
                    f"Layer {layer_id} chunk {chunk_id} token count mismatch: "
                    f"{q_chunk.shape[0]} vs {k_chunk.shape[0]}"
                )
            q_h = apply_block_hadamard(q_chunk, block_hadamard).to(dtype=torch.float64)
            k_h = apply_block_hadamard(k_chunk, block_hadamard).to(dtype=torch.float64)
            q_rot = apply_rotation_to_q_chunk(q_h, rotation, grouping, num_kv_heads)
            k_rot = apply_rotation_to_k_chunk(k_h, rotation, grouping, num_kv_heads)
            residual = quantize_dequantize_int4(k_rot) - k_rot
            num_tokens = q_chunk.shape[0]
            query_positions = sample_query_indices(num_tokens, max_query_samples_per_head)
            teacher_weights = None
            teacher_granularity = None
            if teacher_weight_path is not None:
                teacher_weights, teacher_granularity = load_teacher_weight_chunk(
                    teacher_weight_path,
                    layer_id=layer_id,
                    chunk_id=chunk_id,
                    num_tokens=num_tokens,
                    num_q_heads=num_q_heads,
                    num_kv_heads=num_kv_heads,
                    normalization=teacher_weight_normalization,
                )

            for kv_idx in range(num_kv_heads):
                start = kv_idx * group_size
                end = start + group_size
                q_rot_sampled = q_rot[query_positions, start:end, :]
                q_losses = compute_weighted_query_losses(
                    q_rot_sampled,
                    k_rot[:, kv_idx, :],
                    residual[:, kv_idx, :],
                    query_positions,
                    importance_mode=importance_mode,
                    softmax_temperature=softmax_temperature,
                    topk=topk,
                    causal_window=causal_window,
                )
                if teacher_weights is not None and teacher_granularity is not None:
                    q_losses = q_losses * get_sampled_teacher_query_weights(
                        teacher_weights,
                        granularity=teacher_granularity,
                        query_positions=query_positions,
                        kv_idx=kv_idx,
                        start=start,
                        end=end,
                    ).to(dtype=q_losses.dtype)

                if grouping == "layer":
                    loss_sum += q_losses.sum()
                    count += q_losses.numel()
                elif grouping == "kv_group":
                    loss_sum[kv_idx] += q_losses.sum()
                    count[kv_idx] += q_losses.numel()
                else:
                    for local_idx in range(group_size):
                        head_idx = start + local_idx
                        loss_sum[head_idx] += q_losses[:, local_idx].sum()
                        count[head_idx] += q_losses.shape[0]

        if loss_sum.ndim == 0:
            losses_by_layer[layer_id] = (loss_sum / count.clamp_min(1)).clone()
        else:
            losses_by_layer[layer_id] = (loss_sum / count.clamp_min(1)).clone()

    return losses_by_layer


def apply_group_selection(
    *,
    tensor_path: Path,
    grouping: str,
    candidate_layers: dict[int, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[dict[int, torch.Tensor], dict[int, dict[str, Any]]]:
    identity_layers = {
        layer_id: identity_like_rotation(rotation) for layer_id, rotation in candidate_layers.items()
    }
    candidate_losses = evaluate_rotation_losses(
        tensor_path=tensor_path,
        grouping=grouping,
        rotation_layers=candidate_layers,
        hadamard_order=args.hadamard_order,
        importance_mode=args.importance_mode,
        softmax_temperature=args.softmax_temperature,
        topk=args.topk,
        max_query_samples_per_head=args.max_query_samples_per_head,
        causal_window=args.causal_window,
        chunk_stride=args.chunk_stride,
        max_chunks_per_layer=args.max_chunks_per_layer,
        teacher_weight_path=(
            args.teacher_weight_path.expanduser().resolve()
            if args.teacher_weight_path is not None
            else None
        ),
        teacher_weight_normalization=args.teacher_weight_normalization,
    )
    identity_losses = evaluate_rotation_losses(
        tensor_path=tensor_path,
        grouping=grouping,
        rotation_layers=identity_layers,
        hadamard_order=args.hadamard_order,
        importance_mode=args.importance_mode,
        softmax_temperature=args.softmax_temperature,
        topk=args.topk,
        max_query_samples_per_head=args.max_query_samples_per_head,
        causal_window=args.causal_window,
        chunk_stride=args.chunk_stride,
        max_chunks_per_layer=args.max_chunks_per_layer,
        teacher_weight_path=(
            args.teacher_weight_path.expanduser().resolve()
            if args.teacher_weight_path is not None
            else None
        ),
        teacher_weight_normalization=args.teacher_weight_normalization,
    )

    selected_layers: dict[int, torch.Tensor] = {}
    selection_metrics: dict[int, dict[str, Any]] = {}
    threshold = 1.0 - args.min_relative_improvement
    for layer_id, rotation in candidate_layers.items():
        candidate_loss = candidate_losses[layer_id]
        identity_loss = identity_losses[layer_id]
        if candidate_loss.ndim == 0:
            keep = bool((candidate_loss < identity_loss * threshold).item())
            selected_layers[layer_id] = rotation if keep else identity_like_rotation(rotation)
            selection_metrics[layer_id] = {
                "candidate_loss": float(candidate_loss.item()),
                "identity_loss": float(identity_loss.item()),
                "keep": keep,
            }
        else:
            keep_mask = candidate_loss < identity_loss * threshold
            selected = rotation.clone()
            selected[~keep_mask] = identity_like_rotation(rotation)[~keep_mask]
            selected_layers[layer_id] = selected
            selection_metrics[layer_id] = {
                "candidate_loss": candidate_loss.tolist(),
                "identity_loss": identity_loss.tolist(),
                "keep_mask": keep_mask.tolist(),
            }
    return selected_layers, selection_metrics


def build_rotation_result(
    *,
    args: argparse.Namespace,
    matrix_state: dict[str, Any],
    matrix_key: str,
    grouping: str,
) -> dict[str, Any]:
    save_dtype = getattr(torch, args.save_dtype)
    result: dict[str, Any] = {
        "format_version": 1,
        "objective": matrix_state.get("objective", args.objective),
        "source_grouping": grouping,
        "matrix_key": matrix_key,
        "damp_ratio": args.damp_ratio,
        "rotation_dtype": args.save_dtype,
        "rotation_strength": args.rotation_strength,
        "rotation_structure": args.rotation_structure,
        "rotation_block_size": args.rotation_block_size,
        "permutation_mode": args.permutation_mode,
        "selection_mode": args.selection_mode,
        "layers": {},
    }
    if "source_statistics_path" in matrix_state:
        result["source_statistics_path"] = matrix_state["source_statistics_path"]
    if "tensor_path" in matrix_state:
        result["tensor_path"] = matrix_state["tensor_path"]
    if "init_rotation_path" in matrix_state:
        result["init_rotation_path"] = matrix_state["init_rotation_path"]
    if "importance_mode" in matrix_state:
        result["importance_mode"] = matrix_state["importance_mode"]
        result["softmax_temperature"] = matrix_state["softmax_temperature"]
        result["topk"] = matrix_state["topk"]
        result["hadamard_order"] = matrix_state["hadamard_order"]
        result["max_query_samples_per_head"] = matrix_state["max_query_samples_per_head"]
        result["causal_window"] = matrix_state["causal_window"]
    if "teacher_weight_path" in matrix_state:
        result["teacher_weight_path"] = matrix_state["teacher_weight_path"]
        result["teacher_weight_normalization"] = matrix_state.get(
            "teacher_weight_normalization"
        )
    elif args.teacher_weight_path is not None:
        result["teacher_weight_path"] = str(args.teacher_weight_path.expanduser().resolve())
        result["teacher_weight_normalization"] = args.teacher_weight_normalization

    candidate_layers: dict[int, torch.Tensor] = {}
    candidate_metrics: dict[int, dict[str, Any]] = {}
    candidate_evals: dict[int, torch.Tensor] = {}
    layer_metadata: dict[int, dict[str, Any]] = {}

    for layer_id in sorted(matrix_state["layers"].keys()):
        layer_data = matrix_state["layers"][layer_id]
        if matrix_key not in layer_data:
            raise KeyError(
                f"Layer {layer_id} does not contain matrix key '{matrix_key}'"
            )
        matrix = layer_data[matrix_key]
        evals, rotation, metrics = process_layer_matrix(
            matrix,
            damp_ratio=args.damp_ratio,
            rotation_structure=args.rotation_structure,
            rotation_block_size=args.rotation_block_size,
            permutation_mode=args.permutation_mode,
        )
        if args.rotation_strength != 1.0:
            rotation = apply_rotation_strength(rotation, args.rotation_strength)
            metrics["rotation_strength_applied"] = args.rotation_strength
        candidate_layers[int(layer_id)] = rotation.to(dtype=torch.float64)
        candidate_metrics[int(layer_id)] = metrics
        candidate_evals[int(layer_id)] = evals.to(dtype=torch.float64)
        layer_metadata[int(layer_id)] = layer_data

    selection_metrics: dict[int, dict[str, Any]] = {}
    final_layers = candidate_layers
    if args.selection_mode != "off":
        if args.tensor_path is None:
            raise ValueError("--selection-mode requires --tensor-path")
        final_layers, selection_metrics = apply_group_selection(
            tensor_path=args.tensor_path.expanduser().resolve(),
            grouping=grouping,
            candidate_layers=candidate_layers,
            args=args,
        )

    for layer_id in sorted(final_layers.keys()):
        layer_result = {
            "layer_id": layer_id,
            "rotation": final_layers[layer_id].to(dtype=save_dtype),
            "eigenvalues": candidate_evals[layer_id].to(dtype=save_dtype),
            "metrics": candidate_metrics[layer_id],
        }
        if layer_id in selection_metrics:
            layer_result["selection"] = selection_metrics[layer_id]

        layer_data = layer_metadata[layer_id]
        for key in (
            "count",
            "head_dim",
            "num_q_heads",
            "num_kv_heads",
            "num_chunks",
            "chunk_ids",
            "weight_sum",
            "query_count",
        ):
            if key in layer_data:
                layer_result[key] = layer_data[key]

        result["layers"][layer_id] = layer_result
        print(
            f"layer {layer_id}: matrix_shape={tuple(layer_data[matrix_key].shape)}, "
            f"rotation_shape={tuple(layer_result['rotation'].shape)}, "
            f"eigenvalues_shape={tuple(layer_result['eigenvalues'].shape)}"
        )

    return result


def maybe_save_objective_matrix(
    *,
    path: Optional[Path],
    matrix_state: dict[str, Any],
) -> None:
    if path is None:
        return
    output_path = path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(matrix_state, output_path)
    print(f"Saved objective-matrix state to {output_path}")


def main() -> None:
    args = parse_args()

    if args.objective == "matrix":
        matrix_state, matrix_key = load_matrix_state_from_statistics(args)
        grouping = matrix_state["grouping"]
    else:
        matrix_state, matrix_key = build_weighted_logit_matrix_state(args)
        grouping = matrix_state["grouping"]

    maybe_save_objective_matrix(
        path=args.save_objective_matrix_path,
        matrix_state=matrix_state,
    )

    output_path = (
        args.output_path.expanduser().resolve()
        if args.output_path is not None
        else infer_output_path(
            objective=args.objective,
            grouping=grouping,
            statistics_path=(
                args.statistics_path.expanduser().resolve()
                if args.statistics_path is not None
                else None
            ),
            tensor_path=(
                args.tensor_path.expanduser().resolve()
                if args.tensor_path is not None
                else None
            ),
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = build_rotation_result(
        args=args,
        matrix_state=matrix_state,
        matrix_key=matrix_key,
        grouping=grouping,
    )

    torch.save(result, output_path)
    print(f"Saved Q rotation file to {output_path}")


if __name__ == "__main__":
    main()
