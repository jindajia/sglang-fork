#!/usr/bin/env python3
"""Merge all CSVs under throughput_results_0326/Qwen3-8B into one file."""

import pandas as pd
from pathlib import Path
import re

ROOT = Path(__file__).parent / "throughput_results_0326" / "Qwen3-8B"
OUTPUT = ROOT / "merged.csv"

records = []
for csv_path in sorted(ROOT.rglob("*.csv")):
    if csv_path.name == "merged.csv":
        continue
    config = csv_path.parent.name  # e.g. baseline_int4, quant_int4_1_0_128
    filename = csv_path.stem        # e.g. bs16_in8k_out1k

    # Parse bs / input / output from filename
    m = re.match(r"bs(\d+)_in(\w+)_out(\w+)", filename)
    bs = int(m.group(1)) if m else None
    in_label = m.group(2) if m else None
    out_label = m.group(3) if m else None

    df = pd.read_csv(csv_path)
    df.insert(0, "config", config)
    df.insert(1, "batch_size", bs)
    df.insert(2, "input_len", in_label)
    df.insert(3, "output_len", out_label)
    records.append(df)

merged = pd.concat(records, ignore_index=True)
merged.to_csv(OUTPUT, index=False)
print(f"Merged {len(records)} files -> {OUTPUT}")
print(merged[["config", "batch_size", "input_len", "output_len",
              "summary_job_level_tps", "per_gpu_tps_mean", "ttft_mean"]].to_string(index=False))
