#!/usr/bin/env python3
"""
summarize_ttft.py — Summarize TTFT / OTPS / TPS from ttft_results CSVs into Markdown.

Usage:
    python summarize_ttft.py [--results-dir ttft_results] [--model MODEL] [--output FILE]
"""

import argparse
import csv
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="ttft_results")
    p.add_argument("--model", default=None, help="Filter by model name (e.g. Qwen3-8B)")
    p.add_argument("--output", default="ttft_summary.md")
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
            for csv_path in sorted(config_dir.glob("bs*_in*_out*.csv")):
                stem = csv_path.stem  # e.g. bs4_in8k_out10
                parts = stem.split("_")
                bs        = parts[0][2:]
                in_label  = parts[1][2:]
                out_label = parts[2][3:]

                row = read_csv(csv_path)
                if row is None:
                    continue

                rows.append({
                    "model":      model_dir.name,
                    "config":     config_dir.name,
                    "bs":         int(bs),
                    "input":      in_label,
                    "output":     out_label,
                    "otps_mean":  row.get("user_tps_mean", ""),
                    "otps_stdev": row.get("user_tps_stdev", ""),
                    "total_tps":  row.get("summary_job_level_tps", ""),
                    "ttft_mean":  row.get("ttft_mean", ""),
                    "ttft_stdev": row.get("ttft_stdev", ""),
                    "ttft_p05":   row.get("ttft_p05", ""),
                    "ttft_p50":   row.get("ttft_p50", ""),
                    "ttft_p80":   row.get("ttft_p80", ""),
                    "ttft_p95":   row.get("ttft_p95", ""),
                    "ttft_p99":   row.get("ttft_p99", ""),
                })

    if not rows:
        print("No CSV results found.")
        return

    def input_sort_key(s):
        # Convert label like "8k", "16k", "32k", "8192" to numeric for sorting
        s = s.lower()
        if s.endswith("k"):
            return int(s[:-1]) * 1024
        return int(s)

    rows.sort(key=lambda r: (r["model"], r["config"], r["bs"], input_sort_key(r["input"])))

    # Group by model → config, render one table per config
    lines = ["# TTFT Benchmark Summary", ""]
    lines += [
        "> **OTPS**: per-request output tokens/s (`user_tps_mean ± stdev`)  ",
        "> **TOTAL_TPS**: job-level throughput (`summary_job_level_tps`)  ",
        "> **TTFT**: time to first token in ms  ",
        "",
    ]

    md_header = "| BS | Input | OTPS ± stdev | TOTAL_TPS | TTFT mean ± stdev (ms) | TTFT_p05 | TTFT_p50 | TTFT_p80 | TTFT_p95 | TTFT_p99 |"
    md_sep    = "|----|-------|-------------|-----------|------------------------|----------|----------|----------|----------|----------|"

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

        otps_str = f"{fmt(r['otps_mean'])} ± {fmt(r['otps_stdev'])}"
        ttft_str = f"{fmt(r['ttft_mean'])} ± {fmt(r['ttft_stdev'])}"
        table_rows.append(
            f"| {r['bs']} | {r['input']} | {otps_str} | {fmt(r['total_tps'])} |"
            f" {ttft_str} | {fmt(r['ttft_p05'])} | {fmt(r['ttft_p50'])} |"
            f" {fmt(r['ttft_p80'])} | {fmt(r['ttft_p95'])} | {fmt(r['ttft_p99'])} |"
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
