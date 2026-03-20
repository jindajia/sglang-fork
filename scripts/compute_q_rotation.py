#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute high-precision Q rotation matrices from previously dumped "
            "Q covariance statistics."
        )
    )
    parser.add_argument(
        "--statistics-path",
        type=Path,
        required=True,
        help="Path to q_statistics_*.pt generated from dumped Q chunks.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help=(
            "Path to save the rotation file. Defaults to a sibling file named "
            "q_rotation_<grouping>.pt."
        ),
    )
    parser.add_argument(
        "--matrix-key",
        choices=("covariance", "second_moment"),
        default="covariance",
        help=(
            "Which PSD matrix to diagonalize. Use covariance for centered Q, "
            "or second_moment for raw E[q q^T]."
        ),
    )
    parser.add_argument(
        "--damp-ratio",
        type=float,
        default=0.0,
        help=(
            "Optional CARE/TransMLA-style diagonal damping ratio. "
            "A value like 0.01 matches their PCA helpers, while 0.0 avoids "
            "perturbing the input matrix."
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
    return parser.parse_args()


def infer_output_path(statistics_path: Path, grouping: str) -> Path:
    return statistics_path.with_name(f"q_rotation_{grouping}.pt")


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


def process_layer_matrix(
    matrix: torch.Tensor,
    damp_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if matrix.ndim == 2:
        evals, evecs, metrics = decompose_psd_matrix(matrix, damp_ratio)
        return evals, evecs, metrics

    if matrix.ndim != 3:
        raise ValueError(
            f"Expected matrix rank 2 or 3, got shape {tuple(matrix.shape)}"
        )

    eval_list = []
    vec_list = []
    metric_list = []
    for idx in range(matrix.shape[0]):
        evals, evecs, metrics = decompose_psd_matrix(matrix[idx], damp_ratio)
        eval_list.append(evals)
        vec_list.append(evecs)
        metric_list.append(metrics)

    return (
        torch.stack(eval_list, dim=0).contiguous(),
        torch.stack(vec_list, dim=0).contiguous(),
        {"groups": metric_list},
    )


def main() -> None:
    args = parse_args()
    statistics_path = args.statistics_path.expanduser().resolve()
    if not statistics_path.is_file():
        raise FileNotFoundError(f"statistics file not found: {statistics_path}")

    stats = torch.load(statistics_path, map_location="cpu")
    grouping = stats.get("grouping", "layer")
    layers = stats.get("layers")
    if not isinstance(layers, dict) or not layers:
        raise ValueError(
            "statistics file does not contain a non-empty 'layers' dictionary"
        )

    output_path = (
        args.output_path.expanduser().resolve()
        if args.output_path is not None
        else infer_output_path(statistics_path, grouping)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_dtype = getattr(torch, args.save_dtype)
    result: dict[str, Any] = {
        "format_version": 1,
        "source_statistics_path": str(statistics_path),
        "source_grouping": grouping,
        "matrix_key": args.matrix_key,
        "damp_ratio": args.damp_ratio,
        "rotation_dtype": args.save_dtype,
        "layers": {},
    }

    for layer_id in sorted(layers.keys()):
        layer_data = layers[layer_id]
        if args.matrix_key not in layer_data:
            raise KeyError(
                f"Layer {layer_id} does not contain matrix key '{args.matrix_key}'"
            )

        matrix = layer_data[args.matrix_key]
        evals, rotation, metrics = process_layer_matrix(matrix, args.damp_ratio)

        layer_result = {
            "layer_id": layer_id,
            "rotation": rotation.to(dtype=save_dtype),
            "eigenvalues": evals.to(dtype=save_dtype),
            "metrics": metrics,
        }

        for key in ("count", "head_dim", "num_q_heads", "num_chunks", "chunk_ids"):
            if key in layer_data:
                layer_result[key] = layer_data[key]

        result["layers"][layer_id] = layer_result

        print(
            f"layer {layer_id}: matrix_shape={tuple(matrix.shape)}, "
            f"rotation_shape={tuple(layer_result['rotation'].shape)}, "
            f"eigenvalues_shape={tuple(layer_result['eigenvalues'].shape)}"
        )

    torch.save(result, output_path)
    print(f"Saved Q rotation file to {output_path}")


if __name__ == "__main__":
    main()
