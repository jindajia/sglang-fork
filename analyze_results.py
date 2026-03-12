#!/usr/bin/env python3
"""
analyze_results.py — Summarize eval_results into a Markdown report.
Only processes tasks that have aggregated.json.
Output: eval_results/summary.md
"""

import json
import os
from pathlib import Path
from collections import defaultdict

# =============================================================================
# Config
# =============================================================================

RESULTS_DIR = Path(__file__).parent / "eval_results"
OUTPUT_FILE = RESULTS_DIR / "summary.md"

# Config display order and labels
CONFIG_ORDER = [
    ("baseline_bf16",     "BF16"),
    ("baseline_int4",     "INT4"),
    ("quant_int4_1_0_16", "INT4 + R16 (k)"),
    ("quant_int4_1_1_16", "INT4 + R16 (k&v)"),
    ("quant_int4_1_0_64", "INT4 + R64 (k)"),
    ("quant_int4_1_1_64", "INT4 + R64 (k&v)"),
    ("quant_int4_1_0_128","INT4 + R128 (k)"),
    ("quant_int4_1_1_128","INT4 + R128 (k&v)"),
]
CONFIG_KEY_TO_LABEL = {k: v for k, v in CONFIG_ORDER}
CONFIG_KEYS_ORDERED = [k for k, _ in CONFIG_ORDER]

# Primary metric per task (first match wins)
TASK_PRIMARY_METRIC = {
    "gpqa_think":                       "gpqa/score",
    "humaneval_think":                  "humaneval/score",
    "customized_livecodebench_think":   "livecodebench-codegeneration(2025-01-05-2025-04-07)/Pass@1",
    "aime25_think":                     "aime25/avg",
    "math_500_think":                   "math_500/avg",
}

TASK_DISPLAY_NAMES = {
    "gpqa_think":                       "GPQA",
    "humaneval_think":                  "HumanEval",
    "customized_livecodebench_think":   "LiveCodeBench",
    "aime25_think":                     "AIME25",
    "math_500_think":                   "MATH-500",
}

# =============================================================================
# Helpers
# =============================================================================

def pct(x):
    """Convert raw score to percentage, rounded to 2 decimal places."""
    return round(x * 100, 2)

def avg_of_rounded(values):
    """Average of already-rounded percentage values, rounded to 2 decimal places."""
    if not values:
        return None
    return round(sum(values) / len(values), 2)

def fmt(x):
    """Format a percentage value for display."""
    if x is None:
        return "—"
    return f"{x:.2f}"

# =============================================================================
# Data loading
# =============================================================================

# Structure: data[model][task][config_key] = {"values": [...pct...], "avg": pct}
data = defaultdict(lambda: defaultdict(dict))

for agg_file in sorted(RESULTS_DIR.rglob("aggregated.json")):
    # Path: eval_results/{model}/{model}_{task}_{config}/aggregated.json
    config_dir = agg_file.parent
    model_dir  = config_dir.parent
    model_name = model_dir.name

    # Extract task and config from directory name: {model}_{task}_{config}
    dir_name = config_dir.name
    prefix = model_name + "_"
    if not dir_name.startswith(prefix):
        continue
    rest = dir_name[len(prefix):]  # e.g. "gpqa_think_baseline_bf16"

    # Match known tasks
    task_key = None
    config_key = None
    for t in TASK_PRIMARY_METRIC:
        if rest.startswith(t + "_"):
            task_key = t
            config_key = rest[len(t) + 1:]
            break
    if task_key is None or config_key is None:
        continue
    if config_key not in CONFIG_KEY_TO_LABEL:
        continue

    metric = TASK_PRIMARY_METRIC[task_key]
    agg = json.loads(agg_file.read_text())
    if metric not in agg:
        continue

    raw_values = agg[metric]["values"]
    pct_values = [pct(v) for v in raw_values]
    data[model_name][task_key][config_key] = {
        "values": pct_values,
        "avg":    avg_of_rounded(pct_values),
    }

# =============================================================================
# Markdown generation
# =============================================================================

lines = ["# KV Rotation Evaluation Results", ""]

for model_name in sorted(data):
    lines.append(f"## {model_name}")
    lines.append("")

    for task_key, task_label in TASK_DISPLAY_NAMES.items():
        task_data = data[model_name].get(task_key)
        if not task_data:
            continue

        # Determine max number of runs across configs
        max_runs = max(
            len(task_data[c]["values"])
            for c in task_data
            if c in task_data
        )

        lines.append(f"### {task_label}")
        lines.append("")

        # Table header
        run_headers = " | ".join(f"Run {i+1}" for i in range(max_runs))
        lines.append(f"| Config | {run_headers} | Avg |")
        lines.append(f"|{':---'} | {' | '.join(['---:'] * max_runs)} | ---:|")

        for config_key in CONFIG_KEYS_ORDERED:
            label = CONFIG_KEY_TO_LABEL[config_key]
            if config_key not in task_data:
                dashes = " | ".join("—" for _ in range(max_runs))
                lines.append(f"| {label} | {dashes} | — |")
                continue

            entry = task_data[config_key]
            values = entry["values"]
            avg    = entry["avg"]

            # Pad with — if fewer runs than max
            cells = [fmt(v) for v in values]
            while len(cells) < max_runs:
                cells.append("—")

            row = " | ".join(cells)
            lines.append(f"| {label} | {row} | {fmt(avg)} |")

        lines.append("")

OUTPUT_FILE.write_text("\n".join(lines))
print(f"✓ Written to {OUTPUT_FILE}")
