#!/bin/bash
# setup_env.sh — One-shot environment setup for kv_rotation (sglang_env)
#
# Creates conda env 'sglang_env' with:
#   - sglang from this repo (editable)
#   - fast-hadamard-transform (compiled with cuda-nvcc 12.8)
#   - flash-kmeans (for K-means centroid computation)
#   - lm_eval (for Stage 1 KV cache dumping in kmeansworkflow.sh)
#   - tore-eval (editable, for eval scripts)

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BASE="/data/jisenli2/miniconda"
CONDA_ENV_NAME="sglang_env"
CONDA_ENV_DIR="$CONDA_BASE/envs/$CONDA_ENV_NAME"
TORE_EVAL_DIR="$SCRIPT_DIR/tore-eval"

CONDA="$CONDA_BASE/bin/conda"

if [ ! -f "$CONDA" ]; then
    echo "ERROR: conda not found at $CONDA_BASE. Please install miniconda first."
    exit 1
fi

if [ -d "$CONDA_ENV_DIR" ]; then
    echo "Conda environment '$CONDA_ENV_NAME' already exists at $CONDA_ENV_DIR."
    echo "To recreate it, run: $CONDA env remove -n $CONDA_ENV_NAME"
    exit 0
fi

echo "=== Creating conda environment '$CONDA_ENV_NAME' (python 3.10) ==="
"$CONDA" create -y -n "$CONDA_ENV_NAME" python=3.10 -q

echo "=== Installing cuda-nvcc 12.8 (needed to compile fast-hadamard-transform) ==="
"$CONDA" install -y -n "$CONDA_ENV_NAME" \
    -c "nvidia/label/cuda-12.8.0" -c nvidia -c conda-forge \
    "cuda-nvcc=12.8.*" -q

# Build a self-contained CUDA_HOME layout from the scattered conda install paths
mkdir -p "$CONDA_ENV_DIR/cuda-home/bin"
ln -sfn "$CONDA_ENV_DIR/bin/nvcc"                          "$CONDA_ENV_DIR/cuda-home/bin/nvcc"
ln -sfn "$CONDA_ENV_DIR/bin/crt"                           "$CONDA_ENV_DIR/cuda-home/bin/crt"
ln -sfn "$CONDA_ENV_DIR/targets/x86_64-linux/include"     "$CONDA_ENV_DIR/cuda-home/include"
ln -sfn "$CONDA_ENV_DIR/targets/x86_64-linux/lib"         "$CONDA_ENV_DIR/cuda-home/lib64"

echo "=== Installing base build dependencies ==="
"$CONDA_ENV_DIR/bin/pip" install grpcio-tools numpy packaging ninja -q

echo "=== Installing requirements-eval.txt (ray, word2number, flash-kmeans, lm_eval, ...) ==="
"$CONDA_ENV_DIR/bin/pip" install -r "$SCRIPT_DIR/requirements-eval.txt" -q

echo "=== Installing sglang from this repo (editable) ==="
"$CONDA_ENV_DIR/bin/pip" install -e "$SCRIPT_DIR/python" --no-build-isolation -q

echo "=== Installing fast-hadamard-transform (compiled) ==="
CUDA_HOME="$CONDA_ENV_DIR/cuda-home" \
PATH="$CONDA_ENV_DIR/nvvm/bin:$CONDA_ENV_DIR/bin:/usr/bin:$PATH" \
    "$CONDA_ENV_DIR/bin/pip" install \
    "git+https://github.com/Dao-AILab/fast-hadamard-transform.git" \
    --no-build-isolation -q

echo "=== Installing tore-eval (editable) ==="
if [ ! -f "$TORE_EVAL_DIR/setup.py" ] && [ ! -f "$TORE_EVAL_DIR/pyproject.toml" ]; then
    echo "ERROR: tore-eval submodule not initialized. Run: git submodule update --init --recursive"
    exit 1
fi
rm -rf "$TORE_EVAL_DIR/src/tore_eval.egg-info" 2>/dev/null || true
"$CONDA_ENV_DIR/bin/pip" install -e "$TORE_EVAL_DIR" -q

echo ""
echo "✓ Environment '$CONDA_ENV_NAME' ready at $CONDA_ENV_DIR"
echo "  Activate with: conda activate $CONDA_ENV_NAME"
echo "  Python:        $CONDA_ENV_DIR/bin/python3"
