#!/usr/bin/env python3
"""
summarize_throughput.py
Auto-discover throughput result CSVs and render a Markdown report.

Usage:
    python3 summarize_throughput.py [--results_dir PATH] [--output FILE]

Defaults:
    --results_dir  throughput_results      (scans all model subdirs)
    --output       throughput_summary.md
"""

import argparse
import csv
import re
import sys
from pathlib import Path


def _f(row: dict, key: str):
    """Read a float from a CSV row, return None if missing or empty."""
    v = row.get(key, "")
    return float(v) if v else None


def load_metrics(csv_path: Path):
    """Return a dict of metrics from first data row, or None on failure."""
    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                return {
                    "otps_mean":  _f(row, "user_tps_mean"),
                    "otps_std":   _f(row, "user_tps_stdev"),
                    "otps_p05":   _f(row, "user_tps_p05"),
                    "otps_p50":   _f(row, "user_tps_p50"),
                    "otps_p80":   _f(row, "user_tps_p80"),
                    "otps_p95":   _f(row, "user_tps_p95"),
                    "otps_p99":   _f(row, "user_tps_p99"),
                    "total_tps":  _f(row, "summary_job_level_tps"),
                    "ttft_mean":  _f(row, "ttft_mean"),
                    "ttft_std":   _f(row, "ttft_stdev"),
                }
    except Exception as exc:
        print(f"  WARNING: cannot read {csv_path}: {exc}", file=sys.stderr)
    return None


def fmt(mean, std=None, precision: int = 1) -> str:
    if mean is None:
        return "—"
    if std is None:
        return f"{mean:.{precision}f}"
    return f"{mean:.{precision}f}±{std:.{precision}f}"


def fv(val, precision: int = 1) -> str:
    return "—" if val is None else f"{val:.{precision}f}"


# Matches filenames like bs1_in8k_out1k.csv
_CSV_RE = re.compile(r"^bs(\d+)_(.+)\.csv$")

_BS_ORDER = [1, 8, 16, 32]


def render_config_table(config_dir: Path) -> str:
    """Build a Markdown table from all CSVs found in a config directory."""
    rows_data = []
    for csv_path in sorted(config_dir.glob("bs*.csv")):
        m = _CSV_RE.match(csv_path.name)
        if not m:
            continue
        bs    = int(m.group(1))
        shape = m.group(2)
        metrics = load_metrics(csv_path)
        rows_data.append((bs, shape, metrics))

    if not rows_data:
        return "_No CSV files found._"

    # Sort by canonical BS order, then shape
    def sort_key(item):
        bs, shape, _ = item
        try:
            return (_BS_ORDER.index(bs), shape)
        except ValueError:
            return (len(_BS_ORDER) + bs, shape)

    rows_data.sort(key=sort_key)

    header = ("| BS   | Shape      "
              "| OTPS mean±std     | TOTAL_TPS  "
              "| p05    | p50    | p80    | p95    | p99    "
              "| TTFT mean±std      |")
    sep    = ("|------|------------"
              "|-------------------|------------"
              "|--------|--------|--------|--------|--------"
              "|--------------------|")
    lines  = [header, sep]

    for bs, shape, m in rows_data:
        if m is None:
            m = {}
        lines.append(
            f"| {bs:<4} | {shape:<10} "
            f"| {fmt(m.get('otps_mean'), m.get('otps_std')):<17} "
            f"| {fv(m.get('total_tps')):<10} "
            f"| {fv(m.get('otps_p05')):<6} "
            f"| {fv(m.get('otps_p50')):<6} "
            f"| {fv(m.get('otps_p80')):<6} "
            f"| {fv(m.get('otps_p95')):<6} "
            f"| {fv(m.get('otps_p99')):<6} "
            f"| {fmt(m.get('ttft_mean'), m.get('ttft_std')):<18} |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Summarize throughput results to Markdown")
    parser.add_argument(
        "--results_dir",
        default="throughput_results",
        help="Base results dir or a specific model dir (default: throughput_results)",
    )
    parser.add_argument(
        "--output",
        default="throughput_summary.md",
        help="Output Markdown file (default: throughput_summary.md)",
    )
    args = parser.parse_args()

    base = Path(args.results_dir)
    if not base.exists():
        print(f"ERROR: results_dir not found: {base}", file=sys.stderr)
        sys.exit(1)

    # Determine model directories to process:
    #   - if base contains config dirs directly (has csv files one level down), treat as single model
    #   - otherwise, treat each subdir as a model
    def is_model_dir(d: Path) -> bool:
        return any(d.iterdir()) and all(
            not p.is_dir() or any(p.glob("*.csv")) for p in d.iterdir() if p.is_dir()
        )

    subdirs = sorted(p for p in base.iterdir() if p.is_dir())
    # Check if base itself is a model dir (its children are config dirs with CSVs)
    if subdirs and any(any(p.glob("*.csv")) for p in subdirs):
        # base/config_dir/bs*.csv  →  base is a single model dir
        model_dirs = [base]
    else:
        # base/model/config_dir/bs*.csv  →  scan all model subdirs
        model_dirs = subdirs

    if not model_dirs:
        print(f"ERROR: no model subdirectories found under {base}", file=sys.stderr)
        sys.exit(1)

    lines = [
        "# Throughput Benchmark",
        "",
        "> **OTPS**: per-request output tokens/s (`user_tps_mean`)  ",
        "> **TTFT**: time to first token in ms (`ttft_mean`)  ",
        "",
    ]

    for model_dir in model_dirs:
        lines.append(f"## Model: {model_dir.name}")
        lines.append("")

        config_dirs = sorted(p for p in model_dir.iterdir() if p.is_dir())
        if not config_dirs:
            lines.append("_No config subdirectories found._")
            lines.append("")
            continue

        for config_dir in config_dirs:
            lines.append(f"### {config_dir.name}")
            lines.append("")
            lines.append(render_config_table(config_dir))
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
