#!/usr/bin/env python3
"""
summarize_throughput.py — Summarize throughput results from throughput_results CSVs into Markdown.

Usage:
    python summarize_throughput.py [--results-dir throughput_results] [--model MODEL] [--output FILE]
"""

import argparse
import csv
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="throughput_results")
    p.add_argument("--model", default=None, help="Filter by model name (e.g. GLM-4.7-FP8)")
    p.add_argument("--output", default="throughput_summary.md")
    return p.parse_args()


def read_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else None


def fmt(val, decimals=1):
    try:
        return f"{float(val):.{decimals}f}"
    except (ValueError, TypeError):
        return "—"



def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"ERROR: {results_dir} not found", file=sys.stderr)
        sys.exit(1)

    # Collect all rows
    rows = []
    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        if args.model and args.model not in model_dir.name:
            continue
        for config_dir in sorted(model_dir.iterdir()):
            if not config_dir.is_dir():
                continue
            for csv_path in sorted(config_dir.glob("bs*_in8k_out1k.csv")):
                stem = csv_path.stem  # e.g. bs16_in8k_out1k
                parts = stem.split("_")
                bs        = parts[0][2:]
                in_label  = parts[1][2:]

                row = read_csv(csv_path)
                if row is None:
                    continue

                rows.append({
                    "model":           model_dir.name,
                    "config":          config_dir.name,
                    "bs":              int(bs),
                    "input":           in_label,
                    "otps_mean":       row.get("user_tps_mean", ""),
                    "otps_stdev":      row.get("user_tps_stdev", ""),
                    "otps_p05":        row.get("user_tps_p05", ""),
                    "otps_p50":        row.get("user_tps_p50", ""),
                    "otps_p80":        row.get("user_tps_p80", ""),
                    "otps_p95":        row.get("user_tps_p95", ""),
                    "otps_p99":        row.get("user_tps_p99", ""),
                    "ttft_mean":       row.get("ttft_mean", ""),
                    "ttft_stdev":      row.get("ttft_stdev", ""),
                    "elapsed":         row.get("summary_total_elapsed_time_s", ""),
                    "total_tps":       row.get("summary_job_level_tps", ""),
                    "actual_qps":      row.get("summary_actual_qps", ""),
                    "per_gpu_num":     row.get("per_gpu_num_gpus", ""),
                    "per_gpu_mean":    row.get("per_gpu_tps_mean", ""),
                    "per_gpu_stdev":   row.get("per_gpu_tps_stdev", ""),
                })

    if not rows:
        print("No CSV results found.")
        return

    rows.sort(key=lambda r: (r["model"], r["config"], r["bs"]))

    # Group by model → config, render one table per config
    lines = ["# Throughput Benchmark Summary", ""]
    lines += [
        "> **OTPS**: per-request output tokens/s (`user_tps_mean ± stdev`)  ",
        "> **TOTAL_TPS**: job-level throughput (`summary_job_level_tps`)  ",
        "> **TTFT**: time to first token in ms  ",
        "",
    ]

    md_header = "| BS | OTPS ± stdev | OTPS_p05 | OTPS_p50 | OTPS_p80 | OTPS_p95 | OTPS_p99 | TTFT mean ± stdev (ms) | elapsed_s | TOTAL_TPS | actual_QPS | GPUs | per_GPU_TPS ± stdev |"
    md_sep    = "|----|-------------|----------|----------|----------|----------|----------|------------------------|-----------|-----------|------------|------|---------------------|"

    current_model = None
    current_config = None
    table_rows = []

    def flush_table():
        if table_rows:
            lines.append(md_header)
            lines.append(md_sep)
            lines.extend(table_rows)
            lines.append("")

    for r in rows:
        if r["model"] != current_model:
            flush_table()
            table_rows = []
            current_config = None
            current_model = r["model"]
            lines.append(f"## {r['model']}")
            lines.append("")

        if r["config"] != current_config:
            flush_table()
            table_rows = []
            current_config = r["config"]
            lines.append(f"### {r['config']}")
            lines.append("")

        otps_str     = f"{fmt(r['otps_mean'])} ± {fmt(r['otps_stdev'])}"
        ttft_str     = f"{fmt(r['ttft_mean'])} ± {fmt(r['ttft_stdev'])}"
        per_gpu_str  = f"{fmt(r['per_gpu_mean'])} ± {fmt(r['per_gpu_stdev'])}"
        table_rows.append(
            f"| {r['bs']} | {otps_str} | {fmt(r['otps_p05'])} | {fmt(r['otps_p50'])} |"
            f" {fmt(r['otps_p80'])} | {fmt(r['otps_p95'])} | {fmt(r['otps_p99'])} |"
            f" {ttft_str} | {fmt(r['elapsed'])} | {fmt(r['total_tps'])} |"
            f" {fmt(r['actual_qps'], 3)} | {fmt(r['per_gpu_num'], 0)} | {per_gpu_str} |"
        )

    flush_table()

    md = "\n".join(lines)
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        f.write(md)

    print(f"Written → {output_path.resolve()}")
    print()
    print(md)


if __name__ == "__main__":
    main()
