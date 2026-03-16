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
CONDA_BASE="/data/$USER/miniconda"
CONDA_ENV_NAME="sglang_env"
CONDA_ENV_DIR="$CONDA_BASE/envs/$CONDA_ENV_NAME"
TORE_EVAL_DIR="$SCRIPT_DIR/tore-eval"

CONDA="$CONDA_BASE/bin/conda"

if [ ! -f "$CONDA" ]; then
    echo "ERROR: conda not found at $CONDA_BASE. Please install miniconda first."
    exit 1
fi

echo "=== Accepting Anaconda Terms of Service ==="
"$CONDA" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
"$CONDA" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

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

# Expose libcuda.so stub where flashinfer JIT linker expects it
mkdir -p "$CONDA_ENV_DIR/lib64/stubs"
ln -sfn "$CONDA_ENV_DIR/targets/x86_64-linux/lib/stubs/libcuda.so" \
    "$CONDA_ENV_DIR/lib64/stubs/libcuda.so"

echo "=== Installing base build dependencies ==="
"$CONDA_ENV_DIR/bin/pip" install grpcio-tools numpy packaging ninja ipykernel -q

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

echo "=== Installing flash-kmeans (jindajia fork, required for _euclid_iter_compiled) ==="
"$CONDA_ENV_DIR/bin/pip" install --force-reinstall --no-deps \
    "git+https://github.com/jindajia/flash-kmeans" -q

echo "=== Initializing tore-eval submodule ==="
git -C "$SCRIPT_DIR" submodule update --init --recursive
TORE_EVAL_BRANCH="jisen/kv_rotation_eval"
current_branch=$(git -C "$TORE_EVAL_DIR" symbolic-ref --short HEAD 2>/dev/null || true)
if [ "$current_branch" != "$TORE_EVAL_BRANCH" ]; then
    echo "  Checking out tore-eval branch '$TORE_EVAL_BRANCH'..."
    git -C "$TORE_EVAL_DIR" checkout "$TORE_EVAL_BRANCH"
fi
echo "  tore-eval branch: $(git -C "$TORE_EVAL_DIR" symbolic-ref --short HEAD)"

echo "=== Installing tore-eval (editable) ==="
if [ ! -f "$TORE_EVAL_DIR/setup.py" ] && [ ! -f "$TORE_EVAL_DIR/pyproject.toml" ]; then
    echo "ERROR: tore-eval pyproject.toml not found after submodule init."
    exit 1
fi
rm -rf "$TORE_EVAL_DIR/src/tore_eval.egg-info" 2>/dev/null || true
"$CONDA_ENV_DIR/bin/pip" install -e "$TORE_EVAL_DIR" -q

echo ""
echo "✓ Environment '$CONDA_ENV_NAME' ready at $CONDA_ENV_DIR"
echo "  Activate with: conda activate $CONDA_ENV_NAME"
echo "  Python:        $CONDA_ENV_DIR/bin/python3"
