#!/bin/bash
# prepare_datasets.sh — Download and convert datasets required by eval scripts.
#
# Usage: bash prepare_datasets.sh <python_executable> <script_dir>

set -eo pipefail

PYTHON="${1:-python3}"
SCRIPT_DIR="${2:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

# Use shared HuggingFace cache if HF_HOME is not already set
if [ -z "$HF_HOME" ]; then
    export HF_HOME="/data/shared/huggingface"
fi

# Read HF token from the standard cache location if not already set
if [ -z "$HUGGING_FACE_HUB_TOKEN" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
    export HUGGING_FACE_HUB_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi
if [ -z "$HF_TOKEN" ]; then
    export HF_TOKEN="$HUGGING_FACE_HUB_TOKEN"
fi

mkdir -p "$SCRIPT_DIR/datasets"

# =============================================================================
# aime25 (math-ai/aime25 -> simple_math schema, saved as DatasetDict{"test":...})
# =============================================================================
if [ ! -d "$SCRIPT_DIR/datasets/aime25" ]; then
    echo "Preparing aime25 dataset (math-ai/aime25)..."
    "$PYTHON" - "$SCRIPT_DIR/datasets/aime25" <<'PYEOF'
import sys
from datasets import load_dataset, DatasetDict

out_path = sys.argv[1]
ds = load_dataset("math-ai/aime25", split="test")
converted = ds.map(
    lambda x: {
        "prompt": x["problem"] + "\nPlease reason step by step, and put your final answer within \\boxed{}.",
        "ground_truth": str(x["answer"]),
        "data_source": "aime25",
        "extra_info": {},
    },
    remove_columns=ds.column_names,
)
DatasetDict({"test": converted}).save_to_disk(out_path)
print(f"✓ aime25 saved ({len(converted)} examples) -> {out_path}")
PYEOF
else
    echo "✓ aime25 already present"
fi

# =============================================================================
# math_problems (togethercomputer/math_problems, eval subset)
# =============================================================================
if [ ! -d "$SCRIPT_DIR/datasets/math_problems" ]; then
    echo "Preparing math_problems dataset (togethercomputer/math_problems)..."
    "$PYTHON" - "$SCRIPT_DIR/datasets/math_problems" <<'PYEOF'
import sys
from datasets import load_dataset

out_path = sys.argv[1]
ds = load_dataset("togethercomputer/math_problems", "eval")
ds.save_to_disk(out_path)
print(f"✓ math_problems saved ({ds['test'].num_rows} examples) -> {out_path}")
PYEOF
else
    echo "✓ math_problems already present"
fi
