#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build or execute a curated menu of rotation-learning experiments "
            "for the matrix, q_weighted, structured, and teacher-weighted families."
        )
    )
    parser.add_argument(
        "--family",
        choices=("matrix", "qweighted", "structured", "teacher", "all"),
        default="all",
        help="Which experiment family to emit.",
    )
    parser.add_argument(
        "--python-bin",
        default="python",
        help="Python executable used to invoke scripts/compute_q_rotation.py.",
    )
    parser.add_argument(
        "--statistics-path",
        type=Path,
        default=None,
        help=(
            "Default path to q_statistics_*.pt. This is typically the layer-level "
            "statistics file."
        ),
    )
    parser.add_argument(
        "--kv-group-statistics-path",
        type=Path,
        default=None,
        help="Optional q_statistics_*.pt built with --grouping=kv_group.",
    )
    parser.add_argument(
        "--head-statistics-path",
        type=Path,
        default=None,
        help="Optional q_statistics_*.pt built with --grouping=head.",
    )
    parser.add_argument(
        "--tensor-path",
        type=Path,
        default=None,
        help="Root dump directory with aligned layer_*/q and layer_*/k chunks.",
    )
    parser.add_argument(
        "--base-rotation-path",
        type=Path,
        default=Path("/data/shared/charlie/sglangfork/q_rotation_layer_second_moment_damp01.pt"),
        help="Baseline rotation used as the warm-start for q_weighted and teacher families.",
    )
    parser.add_argument(
        "--teacher-weight-path",
        type=Path,
        default=None,
        help="Root directory with aligned layer_*/teacher/*.pt chunks.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory where generated artifacts should be written.",
    )
    parser.add_argument(
        "--hadamard-order",
        type=int,
        default=16,
        help="Hadamard block order passed through to compute_q_rotation.py.",
    )
    parser.add_argument(
        "--save-dtype",
        choices=("float64", "float32"),
        default="float64",
        help="Rotation save dtype.",
    )
    parser.add_argument(
        "--accum-dtype",
        choices=("float64", "float32"),
        default="float64",
        help="Accumulation dtype for q_weighted and teacher objectives.",
    )
    parser.add_argument(
        "--max-query-samples-per-head",
        type=int,
        default=128,
        help="Sampling budget forwarded to compute_q_rotation.py.",
    )
    parser.add_argument(
        "--max-chunks-per-layer",
        type=int,
        default=None,
        help="Optional per-layer chunk cap.",
    )
    parser.add_argument(
        "--chunk-stride",
        type=int,
        default=1,
        help="Use every Nth chunk.",
    )
    parser.add_argument(
        "--causal-window",
        type=int,
        default=0,
        help="Optional causal window for q_weighted and teacher objectives.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run each generated command sequentially instead of only printing it.",
    )
    return parser.parse_args()


def require_path(path: Path | None, *, flag: str, family: str) -> Path:
    if path is None:
        raise ValueError(f"{flag} is required for the {family} family")
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{flag} does not exist: {resolved}")
    return resolved


def maybe_append_arg(command: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    command.extend([flag, str(value)])


def base_command(args: argparse.Namespace) -> list[str]:
    return [
        args.python_bin,
        "scripts/compute_q_rotation.py",
        "--hadamard-order",
        str(args.hadamard_order),
        "--save-dtype",
        args.save_dtype,
        "--accum-dtype",
        args.accum_dtype,
        "--max-query-samples-per-head",
        str(args.max_query_samples_per_head),
        "--chunk-stride",
        str(args.chunk_stride),
        "--causal-window",
        str(args.causal_window),
    ]


def matrix_family(args: argparse.Namespace) -> list[dict[str, Any]]:
    layer_statistics_path = require_path(
        args.statistics_path, flag="--statistics-path", family="matrix"
    )
    kv_group_statistics_path = require_path(
        args.kv_group_statistics_path or args.statistics_path,
        flag="--kv-group-statistics-path",
        family="matrix",
    )
    tensor_path = require_path(args.tensor_path, flag="--tensor-path", family="matrix")
    output_root = args.output_root / "matrix"
    experiments: list[dict[str, Any]] = []

    def add(name: str, statistics_path: Path, extra: list[str]) -> None:
        output_path = output_root / f"{name}.pt"
        command = base_command(args) + [
            "--objective",
            "matrix",
            "--statistics-path",
            str(statistics_path),
            "--tensor-path",
            str(tensor_path),
            "--output-path",
            str(output_path),
        ]
        maybe_append_arg(command, "--max-chunks-per-layer", args.max_chunks_per_layer)
        command.extend(extra)
        experiments.append({"family": "matrix", "name": name, "output_path": str(output_path), "command": command})

    add(
        "second_moment_damp01_layer",
        layer_statistics_path,
        ["--matrix-key", "second_moment", "--damp-ratio", "0.01", "--grouping", "layer"],
    )
    add(
        "second_moment_damp005_layer",
        layer_statistics_path,
        ["--matrix-key", "second_moment", "--damp-ratio", "0.005", "--grouping", "layer"],
    )
    add(
        "second_moment_damp01_kv_group",
        kv_group_statistics_path,
        ["--matrix-key", "second_moment", "--damp-ratio", "0.01", "--grouping", "kv_group"],
    )
    add(
        "second_moment_damp01_layer_strength075",
        layer_statistics_path,
        [
            "--matrix-key",
            "second_moment",
            "--damp-ratio",
            "0.01",
            "--grouping",
            "layer",
            "--rotation-strength",
            "0.75",
        ],
    )
    add(
        "second_moment_damp01_layer_select",
        layer_statistics_path,
        [
            "--matrix-key",
            "second_moment",
            "--damp-ratio",
            "0.01",
            "--grouping",
            "layer",
            "--selection-mode",
            "if_better",
            "--min-relative-improvement",
            "0.01",
        ],
    )
    add(
        "covariance_damp01_layer",
        layer_statistics_path,
        ["--matrix-key", "covariance", "--damp-ratio", "0.01", "--grouping", "layer"],
    )
    return experiments


def qweighted_family(args: argparse.Namespace) -> list[dict[str, Any]]:
    tensor_path = require_path(args.tensor_path, flag="--tensor-path", family="qweighted")
    base_rotation_path = require_path(
        args.base_rotation_path,
        flag="--base-rotation-path",
        family="qweighted",
    )
    output_root = args.output_root / "qweighted"
    experiments: list[dict[str, Any]] = []

    def add(name: str, extra: list[str]) -> None:
        output_path = output_root / f"{name}.pt"
        command = base_command(args) + [
            "--objective",
            "q_weighted_logit",
            "--tensor-path",
            str(tensor_path),
            "--init-rotation-path",
            str(base_rotation_path),
            "--output-path",
            str(output_path),
        ]
        maybe_append_arg(command, "--max-chunks-per-layer", args.max_chunks_per_layer)
        command.extend(extra)
        experiments.append({"family": "qweighted", "name": name, "output_path": str(output_path), "command": command})

    add(
        "softmax_layer_init",
        ["--importance-mode", "softmax", "--grouping", "layer"],
    )
    add(
        "uniform_layer_init",
        ["--importance-mode", "uniform", "--grouping", "layer"],
    )
    add(
        "topk_layer_init",
        ["--importance-mode", "topk", "--topk", "8", "--grouping", "layer"],
    )
    add(
        "softmax_kv_group_init",
        ["--importance-mode", "softmax", "--grouping", "kv_group"],
    )
    add(
        "softmax_kv_group_select_init",
        [
            "--importance-mode",
            "softmax",
            "--grouping",
            "kv_group",
            "--selection-mode",
            "if_better",
            "--min-relative-improvement",
            "0.01",
        ],
    )
    return experiments


def structured_family(args: argparse.Namespace) -> list[dict[str, Any]]:
    layer_statistics_path = require_path(
        args.statistics_path, flag="--statistics-path", family="structured"
    )
    tensor_path = require_path(
        args.tensor_path, flag="--tensor-path", family="structured"
    )
    base_rotation_path = require_path(
        args.base_rotation_path,
        flag="--base-rotation-path",
        family="structured",
    )
    output_root = args.output_root / "structured"
    experiments: list[dict[str, Any]] = []

    def add(name: str, extra: list[str], *, objective: str) -> None:
        output_path = output_root / f"{name}.pt"
        command = base_command(args) + [
            "--objective",
            objective,
            "--output-path",
            str(output_path),
        ]
        if objective == "matrix":
            command.extend(["--statistics-path", str(layer_statistics_path), "--tensor-path", str(tensor_path)])
        else:
            command.extend(
                [
                    "--tensor-path",
                    str(tensor_path),
                    "--init-rotation-path",
                    str(base_rotation_path),
                ]
            )
        maybe_append_arg(command, "--max-chunks-per-layer", args.max_chunks_per_layer)
        command.extend(extra)
        experiments.append({"family": "structured", "name": name, "output_path": str(output_path), "command": command})

    add(
        "second_moment_block16_layer",
        [
            "--matrix-key",
            "second_moment",
            "--damp-ratio",
            "0.01",
            "--grouping",
            "layer",
            "--rotation-structure",
            "block_diagonal",
            "--rotation-block-size",
            "16",
        ],
        objective="matrix",
    )
    add(
        "second_moment_permblock16_layer",
        [
            "--matrix-key",
            "second_moment",
            "--damp-ratio",
            "0.01",
            "--grouping",
            "layer",
            "--rotation-structure",
            "permuted_block_diagonal",
            "--rotation-block-size",
            "16",
        ],
        objective="matrix",
    )
    add(
        "second_moment_signeddiag_layer",
        [
            "--matrix-key",
            "second_moment",
            "--damp-ratio",
            "0.01",
            "--grouping",
            "layer",
            "--rotation-structure",
            "signed_diagonal",
        ],
        objective="matrix",
    )
    add(
        "qweighted_permblock16_kv_group_init",
        [
            "--importance-mode",
            "softmax",
            "--grouping",
            "kv_group",
            "--rotation-structure",
            "permuted_block_diagonal",
            "--rotation-block-size",
            "16",
        ],
        objective="q_weighted_logit",
    )
    return experiments


def teacher_family(args: argparse.Namespace) -> list[dict[str, Any]]:
    tensor_path = require_path(args.tensor_path, flag="--tensor-path", family="teacher")
    base_rotation_path = require_path(
        args.base_rotation_path,
        flag="--base-rotation-path",
        family="teacher",
    )
    teacher_weight_path = require_path(
        args.teacher_weight_path,
        flag="--teacher-weight-path",
        family="teacher",
    )
    output_root = args.output_root / "teacher"
    experiments: list[dict[str, Any]] = []

    def add(name: str, extra: list[str]) -> None:
        output_path = output_root / f"{name}.pt"
        command = base_command(args) + [
            "--objective",
            "teacher_weighted_logit",
            "--tensor-path",
            str(tensor_path),
            "--teacher-weight-path",
            str(teacher_weight_path),
            "--init-rotation-path",
            str(base_rotation_path),
            "--output-path",
            str(output_path),
        ]
        maybe_append_arg(command, "--max-chunks-per-layer", args.max_chunks_per_layer)
        command.extend(extra)
        experiments.append({"family": "teacher", "name": name, "output_path": str(output_path), "command": command})

    add(
        "softmax_layer_init",
        ["--importance-mode", "softmax", "--grouping", "layer"],
    )
    add(
        "softmax_kv_group_init",
        ["--importance-mode", "softmax", "--grouping", "kv_group"],
    )
    add(
        "softmax_kv_group_select_init",
        [
            "--importance-mode",
            "softmax",
            "--grouping",
            "kv_group",
            "--selection-mode",
            "if_better",
            "--min-relative-improvement",
            "0.01",
        ],
    )
    add(
        "softmax_permblock16_kv_group_init",
        [
            "--importance-mode",
            "softmax",
            "--grouping",
            "kv_group",
            "--rotation-structure",
            "permuted_block_diagonal",
            "--rotation-block-size",
            "16",
        ],
    )
    return experiments


def build_experiments(args: argparse.Namespace) -> list[dict[str, Any]]:
    families = []
    if args.family == "all":
        families = ["matrix", "qweighted", "structured", "teacher"]
    else:
        families = [args.family]

    experiments: list[dict[str, Any]] = []
    for family in families:
        if family == "matrix":
            experiments.extend(matrix_family(args))
        elif family == "qweighted":
            experiments.extend(qweighted_family(args))
        elif family == "structured":
            experiments.extend(structured_family(args))
        elif family == "teacher":
            experiments.extend(teacher_family(args))
        else:
            raise ValueError(f"Unsupported family: {family}")
    return experiments


def run_experiment(experiment: dict[str, Any]) -> int:
    return subprocess.run(experiment["command"], check=False).returncode


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    experiments = build_experiments(args)

    for experiment in experiments:
        rendered = " ".join(shlex.quote(part) for part in experiment["command"])
        print(f"[{experiment['family']}] {experiment['name']}")
        print(rendered)
        print()

    if not args.execute:
        return

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "rotation_scheme_menu_manifest.json"
    executed: list[dict[str, Any]] = []
    for experiment in experiments:
        record = dict(experiment)
        record["return_code"] = run_experiment(experiment)
        executed.append(record)
        if record["return_code"] != 0:
            raise SystemExit(
                f"Experiment {experiment['name']} failed with exit code {record['return_code']}"
            )

    manifest_path.write_text(json.dumps(executed, indent=2))
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
