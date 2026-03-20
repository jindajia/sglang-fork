#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import torch


LAYER_DIR_RE = re.compile(r"layer_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute Q covariance from chunked SGLang Q dumps without first "
            "materializing concatenated full tensors."
        )
    )
    parser.add_argument(
        "--tensor-path",
        type=Path,
        required=True,
        help="Root dump directory, e.g. /data/shared/charlie/sglangfork",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help=(
            "Output .pt path. Defaults to <tensor-path>/q_statistics_<grouping>.pt"
        ),
    )
    parser.add_argument(
        "--grouping",
        choices=("layer", "head", "kv_group"),
        default="layer",
        help=(
            "How to aggregate Q samples before computing covariance. "
            "'layer' pools all Q heads in a layer, 'head' computes one covariance "
            "per Q head, and 'kv_group' pools Q heads by KV-sharing group."
        ),
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help=(
            "Optional number of layers to scan. If omitted, layer directories are "
            "auto-discovered from tensor-path."
        ),
    )
    parser.add_argument(
        "--num-kv-heads",
        type=int,
        default=None,
        help="Required when grouping=kv_group.",
    )
    parser.add_argument(
        "--accum-dtype",
        choices=("float32", "float64"),
        default="float64",
        help="Accumulation dtype used for sum(Q) and sum(QQ^T).",
    )
    return parser.parse_args()


def discover_layer_ids(tensor_path: Path, num_layers: int | None) -> list[int]:
    if num_layers is not None:
        return list(range(num_layers))

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

    chunk_ids: list[int] = []
    for path in q_dir.iterdir():
        if path.suffix != ".pt":
            continue
        if path.stem.isdigit():
            chunk_ids.append(int(path.stem))
    return sorted(chunk_ids)


def init_accumulators(
    grouping: str,
    num_q_heads: int,
    head_dim: int,
    num_kv_heads: int | None,
    accum_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if grouping == "layer":
        count = torch.zeros((), dtype=torch.int64)
        sum_q = torch.zeros((head_dim,), dtype=accum_dtype)
        sum_qqt = torch.zeros((head_dim, head_dim), dtype=accum_dtype)
        return count, sum_q, sum_qqt

    if grouping == "head":
        count = torch.zeros((num_q_heads,), dtype=torch.int64)
        sum_q = torch.zeros((num_q_heads, head_dim), dtype=accum_dtype)
        sum_qqt = torch.zeros((num_q_heads, head_dim, head_dim), dtype=accum_dtype)
        return count, sum_q, sum_qqt

    if num_kv_heads is None:
        raise ValueError("--num-kv-heads is required when grouping=kv_group")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )

    count = torch.zeros((num_kv_heads,), dtype=torch.int64)
    sum_q = torch.zeros((num_kv_heads, head_dim), dtype=accum_dtype)
    sum_qqt = torch.zeros((num_kv_heads, head_dim, head_dim), dtype=accum_dtype)
    return count, sum_q, sum_qqt


def update_accumulators(
    q_chunk: torch.Tensor,
    grouping: str,
    num_kv_heads: int | None,
    count: torch.Tensor,
    sum_q: torch.Tensor,
    sum_qqt: torch.Tensor,
    accum_dtype: torch.dtype,
) -> None:
    q_chunk = q_chunk.to(dtype=accum_dtype, device="cpu", copy=False)

    if grouping == "layer":
        flat = q_chunk.reshape(-1, q_chunk.shape[-1]).contiguous()
        count += flat.shape[0]
        sum_q += flat.sum(dim=0)
        sum_qqt += flat.T @ flat
        return

    if grouping == "head":
        token_count = q_chunk.shape[0]
        count += token_count
        sum_q += q_chunk.sum(dim=0)
        sum_qqt += torch.einsum("thd,thf->hdf", q_chunk, q_chunk)
        return

    assert num_kv_heads is not None
    token_count, num_q_heads, head_dim = q_chunk.shape
    group_size = num_q_heads // num_kv_heads
    grouped = q_chunk.view(token_count, num_kv_heads, group_size, head_dim)
    grouped = grouped.permute(1, 0, 2, 3).reshape(num_kv_heads, -1, head_dim)
    count += grouped.shape[1]
    sum_q += grouped.sum(dim=1)
    sum_qqt += torch.einsum("gnd,gnf->gdf", grouped, grouped)


def finalize_statistics(
    count: torch.Tensor,
    sum_q: torch.Tensor,
    sum_qqt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if count.ndim == 0:
        if count.item() == 0:
            raise ValueError("No Q samples were accumulated for this layer")
        count_f = count.to(dtype=sum_q.dtype)
        mean = sum_q / count_f
        second_moment = sum_qqt / count_f
        covariance = second_moment - torch.outer(mean, mean)
        covariance = 0.5 * (covariance + covariance.T)
        return mean.contiguous(), second_moment.contiguous(), covariance.contiguous()

    if torch.any(count == 0):
        raise ValueError("At least one group has zero Q samples")

    count_f = count.to(dtype=sum_q.dtype).unsqueeze(-1)
    mean = sum_q / count_f
    second_moment = sum_qqt / count.to(dtype=sum_q.dtype).view(-1, 1, 1)
    covariance = second_moment - torch.einsum("gd,gf->gdf", mean, mean)
    covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
    return mean.contiguous(), second_moment.contiguous(), covariance.contiguous()


def iter_q_chunks(layer_dir: Path, chunk_ids: Iterable[int]) -> Iterable[tuple[int, torch.Tensor]]:
    for chunk_id in chunk_ids:
        q_path = layer_dir / "q" / f"{chunk_id}.pt"
        yield chunk_id, torch.load(q_path, map_location="cpu")


def default_output_path(tensor_path: Path, grouping: str) -> Path:
    return tensor_path / f"q_statistics_{grouping}.pt"


def main() -> None:
    args = parse_args()
    tensor_path = args.tensor_path.expanduser().resolve()
    if not tensor_path.is_dir():
        raise FileNotFoundError(f"tensor-path does not exist: {tensor_path}")

    layer_ids = discover_layer_ids(tensor_path, args.num_layers)
    if not layer_ids:
        raise RuntimeError(f"No layer_* directories found under {tensor_path}")

    accum_dtype = getattr(torch, args.accum_dtype)
    output_path = (
        args.output_path.expanduser().resolve()
        if args.output_path is not None
        else default_output_path(tensor_path, args.grouping)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result: dict[str, object] = {
        "format_version": 1,
        "tensor_path": str(tensor_path),
        "grouping": args.grouping,
        "accum_dtype": args.accum_dtype,
        "num_kv_heads": args.num_kv_heads,
        "layers": {},
    }

    total_chunk_count = 0
    for layer_id in layer_ids:
        layer_dir = tensor_path / f"layer_{layer_id}"
        chunk_ids = list_chunk_ids(layer_dir)
        if not chunk_ids:
            print(f"layer {layer_id}: no q chunk files found, skip")
            continue

        first_chunk = torch.load(
            layer_dir / "q" / f"{chunk_ids[0]}.pt", map_location="cpu"
        )
        if first_chunk.ndim != 3:
            raise ValueError(
                f"Expected q chunk [T, H, D], got {tuple(first_chunk.shape)} for layer {layer_id}"
            )

        token_count, num_q_heads, head_dim = first_chunk.shape
        count, sum_q, sum_qqt = init_accumulators(
            grouping=args.grouping,
            num_q_heads=num_q_heads,
            head_dim=head_dim,
            num_kv_heads=args.num_kv_heads,
            accum_dtype=accum_dtype,
        )

        update_accumulators(
            q_chunk=first_chunk,
            grouping=args.grouping,
            num_kv_heads=args.num_kv_heads,
            count=count,
            sum_q=sum_q,
            sum_qqt=sum_qqt,
            accum_dtype=accum_dtype,
        )

        for chunk_id, q_chunk in iter_q_chunks(layer_dir, chunk_ids[1:]):
            if q_chunk.shape[1:] != (num_q_heads, head_dim):
                raise ValueError(
                    f"Layer {layer_id} chunk {chunk_id} shape mismatch: "
                    f"expected (*, {num_q_heads}, {head_dim}), got {tuple(q_chunk.shape)}"
                )
            update_accumulators(
                q_chunk=q_chunk,
                grouping=args.grouping,
                num_kv_heads=args.num_kv_heads,
                count=count,
                sum_q=sum_q,
                sum_qqt=sum_qqt,
                accum_dtype=accum_dtype,
            )

        mean, second_moment, covariance = finalize_statistics(count, sum_q, sum_qqt)

        layer_result = {
            "layer_id": layer_id,
            "num_q_heads": num_q_heads,
            "head_dim": head_dim,
            "num_chunks": len(chunk_ids),
            "chunk_ids": chunk_ids,
            "count": count.clone(),
            "mean": mean,
            "second_moment": second_moment,
            "covariance": covariance,
        }
        result["layers"][layer_id] = layer_result
        total_chunk_count += len(chunk_ids)

        print(
            f"layer {layer_id}: chunks={len(chunk_ids)}, "
            f"sample_shape=(*, {num_q_heads}, {head_dim}), "
            f"count_shape={tuple(count.shape) if count.ndim else '()'}, "
            f"cov_shape={tuple(covariance.shape)}"
        )

    if not result["layers"]:
        raise RuntimeError(f"No valid q chunk files found under {tensor_path}")

    torch.save(result, output_path)
    print(
        f"Saved Q statistics for {len(result['layers'])} layers "
        f"from {total_chunk_count} chunks to {output_path}"
    )


if __name__ == "__main__":
    main()
