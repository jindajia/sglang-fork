#!/usr/bin/env python3
"""
analyze_results.py — Summarize eval_results into a Markdown report.
Uses aggregated.json if present; otherwise collects values from runN/results.jsonl.
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

# Hardcoded display order for known config keys.
# Kmeans configs are generated dynamically below.
_HADAMARD_ORDERS = [16, 64, 128]
_CLUSTERS        = [1, 16, 256, 2048]

_BASE_CONFIGS = [
    ("baseline_bf16",      "BF16"),
    ("baseline_int4",      "INT4"),
    ("quant_int4_1_0_16",  "INT4 + R16 (k)"),
    ("quant_int4_1_1_16",  "INT4 + R16 (k&v)"),
    ("quant_int4_1_0_64",  "INT4 + R64 (k)"),
    ("quant_int4_1_1_64",  "INT4 + R64 (k&v)"),
    ("quant_int4_1_0_128", "INT4 + R128 (k)"),
    ("quant_int4_1_1_128", "INT4 + R128 (k&v)"),
]

_KMEANS_QUANT_CONFIGS = [
    (f"kmeans_quant_{c}_int4_1_0_{ho}", f"INT4 + R{ho} (k) + Centroids={c}")
    for ho in _HADAMARD_ORDERS
    for c  in _CLUSTERS
]

_KMEANS_ONLY_CONFIGS = [
    (f"kmeans_{c}", f"INT4 + Centroids={c}")
    for c in _CLUSTERS
]

CONFIG_ORDER = _BASE_CONFIGS + _KMEANS_QUANT_CONFIGS + _KMEANS_ONLY_CONFIGS
CONFIG_KEY_TO_LABEL   = {k: v for k, v in CONFIG_ORDER}
CONFIG_KEYS_ORDERED   = [k for k, _ in CONFIG_ORDER]

# Primary metric per task (first match wins)
TASK_PRIMARY_METRIC = {
    "gpqa_think":                     "gpqa/score",
    "humaneval_think":                "humaneval/score",
    "customized_livecodebench_think": "livecodebench-codegeneration(2025-01-05-2025-04-07)/Pass@1",
    "aime25_think":                   "aime25/avg",
    "math_500_think":                 "math_500/avg",
}

TASK_DISPLAY_NAMES = {
    "gpqa_think":                     "GPQA",
    "humaneval_think":                "HumanEval",
    "customized_livecodebench_think": "LiveCodeBench",
    "aime25_think":                   "AIME25",
    "math_500_think":                 "MATH-500",
}

# =============================================================================
# Helpers
# =============================================================================

def pct(x):
    return round(x * 100, 2)

def avg_of_rounded(values):
    if not values:
        return None
    return round(sum(values) / len(values), 2)

def fmt(x):
    if x is None:
        return "—"
    return f"{x:.2f}"

# =============================================================================
# Data loading
# =============================================================================

def parse_config_dir(config_dir):
    """Return (model_name, task_key, config_key) for known layouts, or None."""
    # Layout A: eval_results/{model}/{task}/{config}/
    task_key   = config_dir.parent.name
    config_key = config_dir.name
    model_name = config_dir.parent.parent.name
    if task_key in TASK_PRIMARY_METRIC:
        return model_name, task_key, config_key

    # Layout B (old): eval_results/{model}/{model}_{task}_{config}/
    model_name = config_dir.parent.name
    dir_name   = config_dir.name
    prefix = model_name + "_"
    if dir_name.startswith(prefix):
        rest = dir_name[len(prefix):]
        for t in TASK_PRIMARY_METRIC:
            if rest.startswith(t + "_"):
                config_key = rest[len(t) + 1:]
                return model_name, t, config_key

    # Layout C: eval_results/{model}/{prefix}_{task}/{config}/
    # e.g. eval_results/GLM-4.7-FP8/GLM_aime25_think/quant_int4_1_0_64/
    model_name = config_dir.parent.parent.name
    task_dir   = config_dir.parent.name
    config_key = config_dir.name
    for t in TASK_PRIMARY_METRIC:
        if task_dir == t or task_dir.endswith("_" + t):
            return model_name, t, config_key

    return None


# Structure: data[model][task][config_key] = {"values": [...pct...], "avg": pct}
data = defaultdict(lambda: defaultdict(dict))

# --- Pass 1: aggregated.json ---
for agg_file in sorted(RESULTS_DIR.rglob("aggregated.json")):
    config_dir = agg_file.parent
    parsed = parse_config_dir(config_dir)
    if parsed is None:
        continue
    model_name, task_key, config_key = parsed

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

# --- Pass 2: fallback to runN/results.jsonl ---
for results_file in sorted(RESULTS_DIR.rglob("results.jsonl")):
    run_dir    = results_file.parent
    config_dir = run_dir.parent
    parsed = parse_config_dir(config_dir)
    if parsed is None:
        continue
    model_name, task_key, config_key = parsed

    if config_key in data[model_name][task_key]:
        continue

    metric = TASK_PRIMARY_METRIC[task_key]
    try:
        obj = json.loads(results_file.read_text().strip().splitlines()[0])
        metrics = obj.get("metrics", {})
        if metric not in metrics:
            continue
        raw_value = metrics[metric]
    except Exception:
        continue

    bucket = data[model_name][task_key].setdefault(config_key, {"values": [], "avg": None})
    bucket["values"].append(pct(raw_value))

# Finalize avg for fallback entries
for model_name in data:
    for task_key in data[model_name]:
        for config_key, entry in data[model_name][task_key].items():
            if entry["avg"] is None and entry["values"]:
                entry["avg"] = avg_of_rounded(entry["values"])

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

        # Ordered known configs that have data, then any unknown configs at the end
        known   = [k for k in CONFIG_KEYS_ORDERED if k in task_data]
        unknown = sorted(k for k in task_data if k not in CONFIG_KEY_TO_LABEL)
        all_configs = known + unknown

        if not all_configs:
            continue

        max_runs = max(len(task_data[c]["values"]) for c in all_configs)

        lines.append(f"### {task_label}")
        lines.append("")

        run_headers = " | ".join(f"Run {i+1}" for i in range(max_runs))
        lines.append(f"| Config | {run_headers} | Avg |")
        lines.append(f"|:--- | {' | '.join(['---:'] * max_runs)} | ---:|")

        for config_key in all_configs:
            label = CONFIG_KEY_TO_LABEL.get(config_key, config_key)
            entry = task_data[config_key]
            values = entry["values"]
            avg    = entry["avg"]

            cells = [fmt(v) for v in values]
            while len(cells) < max_runs:
                cells.append("—")

            row = " | ".join(cells)
            lines.append(f"| {label} | {row} | {fmt(avg)} |")

        lines.append("")

OUTPUT_FILE.write_text("\n".join(lines))
print(f"✓ Written to {OUTPUT_FILE}")
