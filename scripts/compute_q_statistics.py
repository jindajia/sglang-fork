#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any

import torch


LAYER_DIR_RE = re.compile(r"layer_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute high-precision Q second-moment and covariance statistics from "
            "dumped Q chunks for the Q*H*R runtime."
        )
    )
    parser.add_argument(
        "--tensor-path",
        type=Path,
        required=True,
        help="Root dump directory containing layer_*/q/*.pt chunks.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help=(
            "Optional output path. Defaults to q_statistics_<grouping>.pt under "
            "--tensor-path."
        ),
    )
    parser.add_argument(
        "--grouping",
        choices=("layer", "kv_group", "head"),
        default="layer",
        help="How to accumulate Q statistics before eigendecomposition.",
    )
    parser.add_argument(
        "--hadamard-order",
        type=int,
        default=16,
        help="Block Hadamard order applied before accumulation. Default 16.",
    )
    parser.add_argument(
        "--accum-dtype",
        choices=("float64", "float32"),
        default="float64",
        help="Accumulation dtype. float64 is recommended.",
    )
    parser.add_argument(
        "--chunk-stride",
        type=int,
        default=1,
        help="Read every Nth chunk.",
    )
    parser.add_argument(
        "--max-chunks-per-layer",
        type=int,
        default=None,
        help="Optional cap on chunk count per layer.",
    )
    args = parser.parse_args()
    if args.chunk_stride < 1:
        raise ValueError("--chunk-stride must be >= 1")
    if args.max_chunks_per_layer is not None and args.max_chunks_per_layer < 1:
        raise ValueError("--max-chunks-per-layer must be >= 1")
    return args


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


def discover_layer_ids(tensor_path: Path) -> list[int]:
    layer_ids: list[int] = []
    for child in tensor_path.iterdir():
        if not child.is_dir():
            continue
        match = LAYER_DIR_RE.match(child.name)
        if match:
            layer_ids.append(int(match.group(1)))
    return sorted(layer_ids)


def list_chunk_ids(layer_dir: Path) -> list[int]:
    q_dir = layer_dir / "q"
    if not q_dir.is_dir():
        return []
    return sorted(
        int(path.stem)
        for path in q_dir.iterdir()
        if path.suffix == ".pt" and path.stem.isdigit()
    )


def init_accumulators(
    grouping: str,
    *,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    accum_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if grouping == "layer":
        return (
            torch.zeros((head_dim, head_dim), dtype=accum_dtype),
            torch.zeros((head_dim,), dtype=accum_dtype),
            torch.zeros((), dtype=torch.int64),
        )
    if grouping == "kv_group":
        return (
            torch.zeros((num_kv_heads, head_dim, head_dim), dtype=accum_dtype),
            torch.zeros((num_kv_heads, head_dim), dtype=accum_dtype),
            torch.zeros((num_kv_heads,), dtype=torch.int64),
        )
    if grouping == "head":
        if num_q_heads != num_kv_heads:
            raise ValueError(
                "head grouping requires num_q_heads == num_kv_heads, got "
                f"{num_q_heads} and {num_kv_heads}"
            )
        return (
            torch.zeros((num_q_heads, head_dim, head_dim), dtype=accum_dtype),
            torch.zeros((num_q_heads, head_dim), dtype=accum_dtype),
            torch.zeros((num_q_heads,), dtype=torch.int64),
        )
    raise ValueError(f"Unsupported grouping: {grouping}")


def accumulate_chunk(
    q_h: torch.Tensor,
    *,
    grouping: str,
    num_kv_heads: int,
    sum_outer: torch.Tensor,
    sum_vec: torch.Tensor,
    count: torch.Tensor,
) -> None:
    head_dim = q_h.shape[-1]
    if grouping == "layer":
        q_flat = q_h.reshape(-1, head_dim)
        sum_outer += torch.einsum("nd,nf->df", q_flat, q_flat)
        sum_vec += q_flat.sum(dim=0)
        count += q_flat.shape[0]
        return

    if grouping == "kv_group":
        group_size = q_h.shape[1] // num_kv_heads
        q_grouped = q_h.reshape(q_h.shape[0], num_kv_heads, group_size, head_dim)
        for kv_idx in range(num_kv_heads):
            q_flat = q_grouped[:, kv_idx, :, :].reshape(-1, head_dim)
            sum_outer[kv_idx] += torch.einsum("nd,nf->df", q_flat, q_flat)
            sum_vec[kv_idx] += q_flat.sum(dim=0)
            count[kv_idx] += q_flat.shape[0]
        return

    if grouping == "head":
        for head_idx in range(q_h.shape[1]):
            q_flat = q_h[:, head_idx, :]
            sum_outer[head_idx] += torch.einsum("nd,nf->df", q_flat, q_flat)
            sum_vec[head_idx] += q_flat.sum(dim=0)
            count[head_idx] += q_flat.shape[0]
        return

    raise ValueError(f"Unsupported grouping: {grouping}")


def finalize_statistics(
    *,
    sum_outer: torch.Tensor,
    sum_vec: torch.Tensor,
    count: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if torch.any(count <= 0):
        raise ValueError("At least one statistics group has zero count")

    if sum_outer.ndim == 2:
        count_fp = count.to(dtype=sum_outer.dtype)
        second_moment = sum_outer / count_fp
        mean = sum_vec / count_fp
        covariance = second_moment - torch.outer(mean, mean)
        second_moment = 0.5 * (second_moment + second_moment.mT)
        covariance = 0.5 * (covariance + covariance.mT)
        return second_moment.contiguous(), covariance.contiguous(), mean.contiguous()

    count_fp = count.to(dtype=sum_outer.dtype).view(-1, 1, 1)
    second_moment = sum_outer / count_fp
    mean = sum_vec / count.to(dtype=sum_vec.dtype).view(-1, 1)
    covariance = second_moment - mean.unsqueeze(-1) * mean.unsqueeze(-2)
    second_moment = 0.5 * (second_moment + second_moment.transpose(-1, -2))
    covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
    return second_moment.contiguous(), covariance.contiguous(), mean.contiguous()


def infer_output_path(tensor_path: Path, grouping: str) -> Path:
    return tensor_path / f"q_statistics_{grouping}.pt"


def main() -> None:
    args = parse_args()
    tensor_path = args.tensor_path.expanduser().resolve()
    if not tensor_path.is_dir():
        raise FileNotFoundError(f"tensor-path does not exist: {tensor_path}")

    output_path = (
        args.output_path.expanduser().resolve()
        if args.output_path is not None
        else infer_output_path(tensor_path, args.grouping)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    accum_dtype = getattr(torch, args.accum_dtype)
    layer_ids = discover_layer_ids(tensor_path)
    if not layer_ids:
        raise RuntimeError(f"No layer_* directories found under {tensor_path}")

    result: dict[str, Any] = {
        "format_version": 1,
        "tensor_path": str(tensor_path),
        "grouping": args.grouping,
        "hadamard_order": args.hadamard_order,
        "accum_dtype": args.accum_dtype,
        "chunk_stride": args.chunk_stride,
        "layers": {},
    }

    block_hadamard: torch.Tensor | None = None
    for layer_id in layer_ids:
        layer_dir = tensor_path / f"layer_{layer_id}"
        chunk_ids = list_chunk_ids(layer_dir)[:: args.chunk_stride]
        if args.max_chunks_per_layer is not None:
            chunk_ids = chunk_ids[: args.max_chunks_per_layer]
        if not chunk_ids:
            print(f"layer {layer_id}: no q chunks found, skip")
            continue

        first_q = torch.load(layer_dir / "q" / f"{chunk_ids[0]}.pt", map_location="cpu")
        if first_q.ndim != 3:
            raise ValueError(
                f"Expected q chunk rank 3, got shape {tuple(first_q.shape)}"
            )
        _, num_q_heads, head_dim = first_q.shape

        k_dir = layer_dir / "k"
        if k_dir.is_dir():
            first_k = torch.load(k_dir / f"{chunk_ids[0]}.pt", map_location="cpu")
            num_kv_heads = first_k.shape[1]
        else:
            num_kv_heads = num_q_heads
        if num_q_heads % num_kv_heads != 0:
            raise ValueError(
                f"Layer {layer_id}: num_q_heads ({num_q_heads}) must be divisible by "
                f"num_kv_heads ({num_kv_heads})"
            )

        if block_hadamard is None or block_hadamard.shape[0] != head_dim:
            block_hadamard = build_block_hadamard(head_dim, args.hadamard_order)

        sum_outer, sum_vec, count = init_accumulators(
            args.grouping,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            accum_dtype=accum_dtype,
        )

        for chunk_id in chunk_ids:
            q_chunk = torch.load(layer_dir / "q" / f"{chunk_id}.pt", map_location="cpu")
            if q_chunk.ndim != 3 or q_chunk.shape[1:] != (num_q_heads, head_dim):
                raise ValueError(
                    f"Layer {layer_id} chunk {chunk_id} q shape mismatch: "
                    f"expected (*, {num_q_heads}, {head_dim}), got {tuple(q_chunk.shape)}"
                )
            q_h = apply_block_hadamard(q_chunk, block_hadamard).to(dtype=accum_dtype)
            accumulate_chunk(
                q_h,
                grouping=args.grouping,
                num_kv_heads=num_kv_heads,
                sum_outer=sum_outer,
                sum_vec=sum_vec,
                count=count,
            )

        second_moment, covariance, mean = finalize_statistics(
            sum_outer=sum_outer,
            sum_vec=sum_vec,
            count=count,
        )
        result["layers"][layer_id] = {
            "layer_id": layer_id,
            "second_moment": second_moment,
            "covariance": covariance,
            "mean": mean,
            "count": count.clone(),
            "head_dim": head_dim,
            "num_q_heads": num_q_heads,
            "num_kv_heads": num_kv_heads,
            "num_chunks": len(chunk_ids),
            "chunk_ids": chunk_ids,
        }
        print(
            f"layer {layer_id}: grouping={args.grouping}, chunks={len(chunk_ids)}, "
            f"q_shape=(*, {num_q_heads}, {head_dim}), second_moment_shape={tuple(second_moment.shape)}"
        )

    if not result["layers"]:
        raise RuntimeError(f"No valid Q statistics layers found under {tensor_path}")

    torch.save(result, output_path)
    print(f"Saved Q statistics file to {output_path}")


if __name__ == "__main__":
    main()
