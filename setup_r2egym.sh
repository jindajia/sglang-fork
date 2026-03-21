#!/bin/bash
# setup_r2egym.sh — One-shot environment setup for R2E-Gym eval (r2egym conda env)
#
# Creates conda env 'r2egym' with:
#   - Python 3.12 (required by OpenHands >= 3.12)
#   - r2egym from /data/jisenli2/R2E-Gym (editable)
#   - OpenHands from togethercomputer fork (needed for tool schemas and prompts)
#   - gymnasium (required by r2egym.agenthub.environment.env)
#   - ipykernel (for Jupyter notebook kernel selection)
#
# Usage:
#   bash setup_r2egym.sh
#
# To recreate from scratch:
#   /data/jisenli2/miniconda/bin/conda env remove -n r2egym -y
#   bash setup_r2egym.sh

set -eo pipefail

CONDA_BASE="/data/$USER/miniconda"
CONDA_ENV_NAME="r2egym"
CONDA_ENV_DIR="$CONDA_BASE/envs/$CONDA_ENV_NAME"
CONDA="$CONDA_BASE/bin/conda"
PYTHON="$CONDA_ENV_DIR/bin/python3"
PIP="$CONDA_ENV_DIR/bin/pip"
R2EGYM_DIR="/data/$USER/R2E-Gym"

if [ ! -f "$CONDA" ]; then
    echo "ERROR: conda not found at $CONDA_BASE. Please install miniconda first."
    exit 1
fi

if [ ! -d "$R2EGYM_DIR" ]; then
    echo "ERROR: R2E-Gym not found at $R2EGYM_DIR."
    exit 1
fi

if [ -d "$CONDA_ENV_DIR" ]; then
    echo "Conda environment '$CONDA_ENV_NAME' already exists at $CONDA_ENV_DIR."
    echo "To recreate it, run: $CONDA env remove -n $CONDA_ENV_NAME -y"
    exit 0
fi

echo "=== Creating conda environment '$CONDA_ENV_NAME' (python 3.12) ==="
"$CONDA" create -y -n "$CONDA_ENV_NAME" python=3.12 -q

echo "=== Installing r2egym (editable) ==="
"$PIP" install -e "$R2EGYM_DIR" -q

echo "=== Installing gymnasium ==="
"$PIP" install gymnasium -q

echo "=== Installing OpenHands (togethercomputer fork, no-deps + ignore python version) ==="
# OpenHands requires Python >=3.12; install its deps manually below.
"$PIP" install "git+https://github.com/togethercomputer/OpenHands.git" \
    --no-deps --ignore-requires-python -q
# Required transitive deps for openhands.agenthub imports
"$PIP" install browsergym-core playwright -q
"$PIP" install "browsergym[core]" -q --no-deps || true

echo "=== Fixing huggingface_hub compatibility (HfFolder removed in >=1.0) ==="
# r2egym/agenthub/utils/utils.py imports HfFolder which was removed;
# that import has been patched out. This note is kept for reference.

echo "=== Installing ipykernel (for Jupyter notebook kernel) ==="
"$PIP" install ipykernel -q
"$PYTHON" -m ipykernel install --user --name r2egym --display-name "Python (r2egym)"

echo ""
echo "✓ Environment '$CONDA_ENV_NAME' ready at $CONDA_ENV_DIR"
echo "  Activate with: conda activate $CONDA_ENV_NAME"
echo "  Python:        $PYTHON"
echo "  Jupyter kernel: 'Python (r2egym)' (restart Jupyter to pick it up)"
