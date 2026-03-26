#!/usr/bin/env python3
"""
summarize_throughput.py
Auto-discover configs / batch-sizes / input-output lengths from the results
directory and render a Markdown report with all available metrics.

Usage:
    python3 summarize_throughput.py [--results_dir PATH] [--output FILE]

Defaults:
    --results_dir  throughput_results/Qwen3-8B
    --output       throughput_summary.md
"""

import argparse
import csv
import re
import sys
from pathlib import Path

# Config order and display names.
# Only configs whose subdirectory exists will be rendered.
CONFIGS = [
    ("baseline_bf16",       "BF16"),
    ("baseline_int4",       "INT4"),
    ("quant_int4_1_0_16",   "INT4 + R16 (k)"),
    ("quant_int4_1_0_64",   "INT4 + R64 (k)"),
    ("quant_int4_1_0_128",  "INT4 + R128 (k)"),
    ("quant_int4_1_0_512",  "INT4 + R512 (k)"),
    ("quant_int4_1_0_1024", "INT4 + R1024 (k)"),
    ("quant_int4_1_1_16",   "INT4 + R16 (kv)"),
    ("quant_int4_1_1_64",   "INT4 + R64 (kv)"),
    ("quant_int4_1_1_128",  "INT4 + R128 (kv)"),
]

# Metric families to report, in display order.
# Each entry: (family_key, display_name, unit, stat_suffixes)
# Only stats that actually exist in the CSV will be shown.
METRIC_FAMILIES = [
    ("user_tps", "OTPS",  "tok/s", ["mean"]),
    ("ttft",     "TTFT",  "ms",    ["mean", "p05", "p50", "p80", "p95", "p99"]),
]


def load_row(csv_path: Path) -> dict:
    """Return the first data row as a dict, or {}."""
    if not csv_path.exists():
        return {}
    try:
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                return {k: v for k, v in row.items() if v.strip() != ""}
    except Exception as exc:
        print(f"  WARNING: cannot read {csv_path}: {exc}", file=sys.stderr)
    return {}


def fval(row: dict, key: str) -> str:
    v = row.get(key)
    if v is None:
        return "—"
    try:
        f = float(v)
        return f"{f:.1f}"
    except ValueError:
        return "—"


def discover_results(results_dir: Path):
    """
    Returns:
        run_keys  : sorted list of (bs, in_label, out_label) tuples
        csv_map   : dict[(rot_suffix, bs, in_label, out_label)] -> Path
    Scans only the CONFIGS subdirectories.
    """
    csv_map = {}
    run_keys_set = set()

    pattern = re.compile(r"^bs(\d+)_(in\w+)_(out\w+)\.csv$")
    for rot_suffix, _ in CONFIGS:
        config_dir = results_dir / rot_suffix
        if not config_dir.is_dir():
            continue
        for f in config_dir.glob("bs*.csv"):
            m = pattern.match(f.name)
            if not m:
                continue
            bs        = int(m.group(1))
            in_label  = m.group(2)
            out_label = m.group(3)
            csv_map[(rot_suffix, bs, in_label, out_label)] = f
            run_keys_set.add((bs, in_label, out_label))

    def sort_key(t):
        bs, in_l, out_l = t
        in_num  = int(re.sub(r"\D", "", in_l)  or 0)
        out_num = int(re.sub(r"\D", "", out_l) or 0)
        return (in_num, bs, out_num)

    run_keys = sorted(run_keys_set, key=sort_key)
    return run_keys, csv_map


def detect_active_columns(run_keys, csv_map):
    """
    Sample all available CSVs to find which metric columns actually have data.
    Returns list of (col_header, csv_key) tuples to use as table columns.
    """
    all_keys = set()
    for rot_suffix, _ in CONFIGS:
        for (bs, in_l, out_l) in run_keys:
            path = csv_map.get((rot_suffix, bs, in_l, out_l))
            if path:
                row = load_row(path)
                all_keys.update(row.keys())

    active = []
    for family_key, display_name, unit, stats in METRIC_FAMILIES:
        for stat in stats:
            csv_key = f"{family_key}_{stat}"
            if csv_key in all_keys:
                label = f"{display_name}_{stat}"
                if unit:
                    label += f" ({unit})"
                active.append((label, csv_key))
    return active


def render_config_table(rot_suffix: str, run_keys, csv_map, active_cols) -> str:
    col_headers = ["BS", "Input", "Output"] + [h for h, _ in active_cols]
    widths = [max(4, len(h)) for h in col_headers]

    rows_data = []
    for (bs, in_l, out_l) in run_keys:
        path = csv_map.get((rot_suffix, bs, in_l, out_l))
        row = load_row(path) if path else {}
        cells = [str(bs), in_l, out_l] + [fval(row, k) for _, k in active_cols]
        rows_data.append(cells)
        for i, c in enumerate(cells):
            widths[i] = max(widths[i], len(c))

    def fmt_row(cells):
        return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |"

    sep = "|-" + "-|-".join("-" * w for w in widths) + "-|"
    lines = [fmt_row(col_headers), sep]
    for cells in rows_data:
        lines.append(fmt_row(cells))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Summarize throughput results to Markdown")
    parser.add_argument(
        "--results_dir",
        default="throughput_results/Qwen3-8B",
        help="Path to the model's results directory",
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

    run_keys, csv_map = discover_results(results_dir)

    if not run_keys:
        print(f"ERROR: no CSV files matching bs*_in*_out*.csv found under {results_dir}", file=sys.stderr)
        sys.exit(1)

    active_cols = detect_active_columns(run_keys, csv_map)

    model_name = results_dir.name
    present = [s for s, _ in CONFIGS if (results_dir / s).is_dir()]
    lines = [
        f"# Throughput Benchmark — {model_name}",
        "",
        f"Configs: {', '.join(present)}  ",
        f"Run combinations: {len(run_keys)}  ",
        "",
    ]

    for rot_suffix, display_name in CONFIGS:
        config_dir = results_dir / rot_suffix
        if not config_dir.is_dir():
            continue
        lines.append(f"## {display_name}")
        lines.append(f"`{rot_suffix}`")
        lines.append("")
        lines.append(render_config_table(rot_suffix, run_keys, csv_map, active_cols))
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
