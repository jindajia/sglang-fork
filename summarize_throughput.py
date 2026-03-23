#!/usr/bin/env python3
"""
summarize_throughput.py
Extract OTPS (user_tps_mean) and TTFT (ttft_mean) from tore_speed_eval CSV files
and render a Markdown report.

Usage:
    python3 summarize_throughput.py [--results_dir PATH] [--output FILE]

Defaults:
    --results_dir  throughput_results/Qwen3-4B-Thinking-2507
    --output       throughput_summary.md
"""

import argparse
import csv
import os
import sys
from pathlib import Path

# Config order and display names — matches rot_suffix values in throughput_test.sh
# rot_suffix format: baseline_{dtype} | rotation_{dtype}_{rk}_{rv}_{ho} | rotation_qr_{dtype}_{rk}_{rv}_{ho}
CONFIGS = [
    ("baseline_bf16",                   "BF16 baseline"),
    ("baseline_int4",                   "INT4 baseline"),
    ("rotation_int4_1_0_16",            "INT4 + Rotation (k=1,v=0,H=16)"),
    ("rotation_int4_1_0_64",            "INT4 + Rotation (k=1,v=0,H=64)"),
    ("rotation_int4_1_0_128",           "INT4 + Rotation (k=1,v=0,H=128)"),
    ("rotation_qr_int4_1_0_16",         "INT4 + Rotation_QR (k=1,v=0,H=16)"),
    ("rotation_qr_int4_1_0_64",         "INT4 + Rotation_QR (k=1,v=0,H=64)"),
    ("rotation_qr_int4_1_0_128",        "INT4 + Rotation_QR (k=1,v=0,H=128)"),
]

BATCH_SIZES = [1, 8, 16, 32]
INPUT_LENS  = [8192, 16384, 32768]


def in_label(n: int) -> str:
    return f"{n // 1024}k"


def load_metrics(csv_path: Path):
    """Return (otps, otps_std, ttft, ttft_std) from first data row, or Nones."""
    if not csv_path.exists():
        return None, None, None, None
    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                otps     = float(row["user_tps_mean"])  if row.get("user_tps_mean")  else None
                otps_std = float(row["user_tps_stdev"]) if row.get("user_tps_stdev") else None
                ttft     = float(row["ttft_mean"])       if row.get("ttft_mean")       else None
                ttft_std = float(row["ttft_stdev"])      if row.get("ttft_stdev")      else None
                return otps, otps_std, ttft, ttft_std
    except Exception as exc:
        print(f"  WARNING: cannot read {csv_path}: {exc}", file=sys.stderr)
    return None, None, None, None


def fmt_mean_std(mean, std, precision: int = 2) -> str:
    if mean is None:
        return "—"
    if std is None:
        return f"{mean:.{precision}f}"
    return f"{mean:.{precision}f} ± {std:.{precision}f}"


def render_config_table(results_dir: Path, rot_suffix: str) -> str:
    """Build a Markdown table for one config across all BS x input_len combos."""
    header = "| BS   | Input Len | OTPS (tok/s)          | TTFT (ms)              |"
    sep    = "|------|-----------|-----------------------|------------------------|"
    rows   = [header, sep]

    for bs in BATCH_SIZES:
        for input_len in INPUT_LENS:
            label    = in_label(input_len)
            csv_path = results_dir / rot_suffix / f"bs{bs}_in{label}.csv"
            otps, otps_std, ttft, ttft_std = load_metrics(csv_path)
            otps_str = fmt_mean_std(otps, otps_std, precision=2)
            ttft_str = fmt_mean_std(ttft, ttft_std, precision=2)
            rows.append(f"| {bs:<4} | {label:<9} | {otps_str:<21} | {ttft_str:<22} |")

    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description="Summarize throughput results to Markdown")
    parser.add_argument(
        "--results_dir",
        default="throughput_results/Qwen3-4B-Thinking-2507",
        help="Path to the model's results directory (default: throughput_results/Qwen3-4B-Thinking-2507)",
    )
    parser.add_argument(
        "--output",
        default="throughput_summary.md",
        help="Output Markdown file (default: throughput_summary.md)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"ERROR: results_dir not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    lines = [
        "# Throughput Benchmark — Qwen3-4B-Thinking-2507",
        "",
        "> **OTPS**: per-request output tokens/s (`user_tps_mean`)  ",
        "> **TTFT**: time to first token in ms (`ttft_mean`)  ",
        f"> Output fixed to {1024} tokens · {96} examples per run",
        "",
    ]

    for rot_suffix, display_name in CONFIGS:
        config_dir = results_dir / rot_suffix
        if not config_dir.exists():
            print(f"  SKIP {display_name}: directory not found ({config_dir})", file=sys.stderr)
            continue

        lines.append(f"## {display_name}")
        lines.append("")
        lines.append(render_config_table(results_dir, rot_suffix))
        lines.append("")

    md = "\n".join(lines)

    output_path = Path(args.output)
    with open(output_path, "w") as f:
        f.write(md)

    print(f"Written → {output_path.resolve()}")
    print()
    print(md)


if __name__ == "__main__":
    main()
