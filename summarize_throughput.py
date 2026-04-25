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
                    "otps_mean":          _f(row, "user_tps_mean"),
                    "otps_std":           _f(row, "user_tps_stdev"),
                    "otps_p05":           _f(row, "user_tps_p05"),
                    "otps_p50":           _f(row, "user_tps_p50"),
                    "otps_p80":           _f(row, "user_tps_p80"),
                    "otps_p95":           _f(row, "user_tps_p95"),
                    "otps_p99":           _f(row, "user_tps_p99"),
                    "hit_mean":           _f(row, "cache_hit_ratio_mean"),
                    "hit_std":            _f(row, "cache_hit_ratio_stdev"),
                    "hit_p05":            _f(row, "cache_hit_ratio_p05"),
                    "hit_p50":            _f(row, "cache_hit_ratio_p50"),
                    "hit_p80":            _f(row, "cache_hit_ratio_p80"),
                    "hit_p95":            _f(row, "cache_hit_ratio_p95"),
                    "hit_p99":            _f(row, "cache_hit_ratio_p99"),
                    "ttft_mean":          _f(row, "ttft_mean"),
                    "ttft_std":           _f(row, "ttft_stdev"),
                    "elapsed_s":          _f(row, "summary_total_elapsed_time_s"),
                    "job_level_tps":      _f(row, "summary_job_level_tps"),
                    "actual_qps":         _f(row, "summary_actual_qps"),
                    "per_gpu_num_gpus":   _f(row, "per_gpu_num_gpus"),
                    "per_gpu_tps_mean":   _f(row, "per_gpu_tps_mean"),
                    "per_gpu_tps_stdev":  _f(row, "per_gpu_tps_stdev"),
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


def render_config_table(config_dir: Path, cache_hit: bool = False) -> str:
    """Build a Markdown table from all CSVs found in a config directory.

    If cache_hit=True, replace the OTPS percentile columns with cache-hit-ratio
    percentile columns (useful for Mode 2 prefix-cache benchmarks).
    """
    rows_data = []
    for csv_path in sorted(config_dir.glob("bs*.csv")):
        m = _CSV_RE.match(csv_path.name)
        if not m:
            continue
        bs      = int(m.group(1))
        shape   = m.group(2)
        metrics = load_metrics(csv_path)
        rows_data.append((bs, shape, metrics))

    if not rows_data:
        return "_No CSV files found._"

    # Sort:
    #   cache_hit mode (Mode 2): primary key = run number extracted from shape
    #     (e.g. 'nqa100k_run1' → 1), so all run1 rows come before run2 rows,
    #     with BS ordered canonically within each run group.
    #   default (Mode 1): BS first, then shape.
    def extract_run(shape: str) -> int:
        m = re.search(r"run(\d+)", shape)
        return int(m.group(1)) if m else 0

    def sort_key(item):
        bs, shape, _ = item
        try:
            bs_key = _BS_ORDER.index(bs)
        except ValueError:
            bs_key = len(_BS_ORDER) + bs
        if cache_hit:
            return (extract_run(shape), bs_key, shape)
        return (bs_key, shape)

    rows_data.sort(key=sort_key)

    if cache_hit:
        header = ("| BS | Shape "
                  "| OTPS±stdev "
                  "| HIT±stdev | HIT_p05 | HIT_p50 | HIT_p80 | HIT_p95 | HIT_p99 "
                  "| TTFT±stdev "
                  "| elapsed_s | job_tps | actual_qps "
                  "| gpus | per_gpu_tps±stdev |")
        sep    = ("|----|-------"
                  "|------------"
                  "|-----------|---------|---------|---------|---------|---------"
                  "|------------"
                  "|-----------|---------|------------"
                  "|------|-------------------|")
    else:
        header = ("| BS | Shape "
                  "| OTPS±stdev | OTPS_p05 | OTPS_p50 | OTPS_p80 | OTPS_p95 | OTPS_p99 "
                  "| TTFT±stdev "
                  "| elapsed_s | job_tps | actual_qps "
                  "| gpus | per_gpu_tps±stdev |")
        sep    = ("|----|-------"
                  "|------------|----------|----------|----------|----------|----------"
                  "|------------"
                  "|-----------|---------|------------"
                  "|------|-------------------|")
    lines  = [header, sep]

    for bs, shape, m in rows_data:
        if m is None:
            m = {}
        gpus = int(m["per_gpu_num_gpus"]) if m.get("per_gpu_num_gpus") is not None else None
        if cache_hit:
            middle_cols = (
                f"| {fmt(m.get('otps_mean'), m.get('otps_std'))} "
                f"| {fmt(m.get('hit_mean'), m.get('hit_std'), precision=3)} "
                f"| {fv(m.get('hit_p05'), precision=3)} "
                f"| {fv(m.get('hit_p50'), precision=3)} "
                f"| {fv(m.get('hit_p80'), precision=3)} "
                f"| {fv(m.get('hit_p95'), precision=3)} "
                f"| {fv(m.get('hit_p99'), precision=3)} "
            )
        else:
            middle_cols = (
                f"| {fmt(m.get('otps_mean'), m.get('otps_std'))} "
                f"| {fv(m.get('otps_p05'))} "
                f"| {fv(m.get('otps_p50'))} "
                f"| {fv(m.get('otps_p80'))} "
                f"| {fv(m.get('otps_p95'))} "
                f"| {fv(m.get('otps_p99'))} "
            )
        lines.append(
            f"| {bs} | {shape} "
            f"{middle_cols}"
            f"| {fmt(m.get('ttft_mean'), m.get('ttft_std'))} "
            f"| {fv(m.get('elapsed_s'))} "
            f"| {fv(m.get('job_level_tps'))} "
            f"| {fv(m.get('actual_qps'))} "
            f"| {'—' if gpus is None else gpus} "
            f"| {fmt(m.get('per_gpu_tps_mean'), m.get('per_gpu_tps_stdev'))} |"
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
    parser.add_argument(
        "--cache-hit",
        action="store_true",
        help="Replace OTPS percentile columns with cache-hit-ratio percentile columns "
             "(for Mode 2 prefix-cache benchmarks).",
    )
    args = parser.parse_args()

    base = Path(args.results_dir)
    if not base.exists():
        print(f"ERROR: results_dir not found: {base}", file=sys.stderr)
        sys.exit(1)

    subdirs = sorted(p for p in base.iterdir() if p.is_dir())
    # Check if base itself is a model dir (its children are config dirs with CSVs)
    if subdirs and any(any(p.glob("*.csv")) for p in subdirs):
        model_dirs = [base]
    else:
        model_dirs = subdirs

    if not model_dirs:
        print(f"ERROR: no model subdirectories found under {base}", file=sys.stderr)
        sys.exit(1)

    lines = [
        "# Throughput Benchmark",
        "",
        "> **OTPS**: per-request output tokens/s (`user_tps_mean`)  ",
        "> **TTFT**: time to first token in ms (`ttft_mean`)  ",
        "> **job_tps**: total output tokens / total elapsed time  ",
        "> **per_gpu_tps**: job_tps normalized per GPU  ",
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
            lines.append(render_config_table(config_dir, cache_hit=args.cache_hit))
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
