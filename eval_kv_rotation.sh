#!/bin/bash
# eval_kv_rotation.sh — Local SGLang evaluation script (non-Docker)
#
# Hadamard rotation settings are configured per-entry in MODEL_CONFIGS.

set -eo pipefail

# =============================================================================
# Model Configuration
# =============================================================================
# Format: "mode|hadamard|rotate_v|hadamard_order|kv_dtype|model_name|tp_size|ep_size|dp_size|gpu_devices|tasks"
#   mode           : BASE or QUANT
#                    BASE  — baseline (forces hadamard=0, rotate_v=0; kv_dtype still applies)
#                    QUANT — quantized run with rotation (uses all columns as configured)
#   hadamard       : 0 or 1 — enable Hadamard rotation on K/Q (ignored for BASE)
#   rotate_v       : 0 or 1 — also rotate V and de-rotate attention output (ignored for BASE)
#   hadamard_order : block size for block-Hadamard (e.g. 16; ignored for BASE)
#   kv_dtype       : BF16 (--kv-cache-dtype auto) or INT4 (--kv-cache-dtype int4)
#   model_name     : full HuggingFace model ID
#   tp_size        : tensor parallel size
#   ep_size        : expert parallel size
#   dp_size        : data parallel size
#   gpu_devices    : comma-separated GPU IDs passed to CUDA_VISIBLE_DEVICES
#   tasks          : comma-separated preset names with optional :N repeat count
#
# Available preset names:
#   gpqa_think
#   humaneval_think
#   customized_livecodebench_think
#   aime25_think
#   math_500_think
# eg: aime25_think:5,math_500_think (without space after comma)
TASKS_ALL="gpqa_think:5,humaneval_think:5,customized_livecodebench_think:5,aime25_think:5,math_500_think:5"
TASKS_ONCE="gpqa_think:1,humaneval_think:1,customized_livecodebench_think:1,aime25_think:1,math_500_think:1"
MODEL_CONFIGS=(
    "BASE|0|0|0|BF16|Qwen/Qwen3-8B|1|1|1|0|${TASKS_ALL}"
    "BASE|0|0|0|INT4|Qwen/Qwen3-8B|1|1|1|1|${TASKS_ONCE}"
    "QUANT|1|0|16|INT4|Qwen/Qwen3-8B|1|1|1|2|${TASKS_ALL}"
    "QUANT|1|1|16|INT4|Qwen/Qwen3-8B|1|1|1|3|${TASKS_ALL}"
    "QUANT|1|0|64|INT4|Qwen/Qwen3-8B|1|1|1|4|${TASKS_ALL}"
    "QUANT|1|1|64|INT4|Qwen/Qwen3-8B|1|1|1|5|${TASKS_ALL}"
    "QUANT|1|0|128|INT4|Qwen/Qwen3-8B|1|1|1|6|${TASKS_ALL}"
    "QUANT|1|1|128|INT4|Qwen/Qwen3-8B|1|1|1|7|${TASKS_ALL}"
)

# =============================================================================
# Server & Eval Config
# =============================================================================
BASE_PORT=30001
NUM_WORKERS=64

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TORE_EVAL_DIR="$SCRIPT_DIR/tore-eval"
RESULTS_DIR="$SCRIPT_DIR/eval_results"
LOGS_DIR="$SCRIPT_DIR/eval_logs"
CONDA_BASE="/data/jisenli2/miniconda"
CONDA_ENV_NAME="sglang_eval"
CONDA_ENV_DIR="$CONDA_BASE/envs/$CONDA_ENV_NAME"

mkdir -p "$RESULTS_DIR" "$LOGS_DIR/inference_logs" "$LOGS_DIR/batch_logs"

# =============================================================================
# Conda Environment Setup
# =============================================================================

# Install miniconda if not present
if [ ! -f "$CONDA_BASE/bin/conda" ]; then
    echo "Miniconda not found, installing to $CONDA_BASE ..."
    MINICONDA_INSTALLER=$(mktemp --suffix=.sh)
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o "$MINICONDA_INSTALLER"
    bash "$MINICONDA_INSTALLER" -b -p "$CONDA_BASE"
    rm -f "$MINICONDA_INSTALLER"
    "$CONDA_BASE/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
    "$CONDA_BASE/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
    echo "✓ Miniconda installed"
fi

CONDA="$CONDA_BASE/bin/conda"

# Accept Anaconda TOS (idempotent, required for non-interactive use)
"$CONDA" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
"$CONDA" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

if [ ! -d "$CONDA_ENV_DIR" ]; then
    echo "Creating conda environment '$CONDA_ENV_NAME' ..."
    "$CONDA" create -y -n "$CONDA_ENV_NAME" python=3.10 -q
    echo "Installing cuda-nvcc 12.8 (for fast-hadamard-transform compilation) ..."
    # cuda-nvcc needs fallback channels to resolve cuda-version dependency
    "$CONDA" install -y -n "$CONDA_ENV_NAME" \
        -c "nvidia/label/cuda-12.8.0" -c nvidia -c conda-forge \
        "cuda-nvcc=12.8.*" -q
    # Build a self-contained CUDA_HOME layout from the scattered conda install paths
    mkdir -p "$CONDA_ENV_DIR/cuda-home/bin"
    ln -sfn "$CONDA_ENV_DIR/bin/nvcc"                          "$CONDA_ENV_DIR/cuda-home/bin/nvcc"
    ln -sfn "$CONDA_ENV_DIR/bin/crt"                           "$CONDA_ENV_DIR/cuda-home/bin/crt"
    ln -sfn "$CONDA_ENV_DIR/targets/x86_64-linux/include"     "$CONDA_ENV_DIR/cuda-home/include"
    ln -sfn "$CONDA_ENV_DIR/targets/x86_64-linux/lib"         "$CONDA_ENV_DIR/cuda-home/lib64"
    echo "Installing base build dependencies ..."
    "$CONDA_ENV_DIR/bin/pip" install grpcio-tools numpy packaging ninja -q
    echo "Installing extra dependencies (requirements-eval.txt) ..."
    "$CONDA_ENV_DIR/bin/pip" install -r "$SCRIPT_DIR/requirements-eval.txt" -q
    echo "Installing sglang-fork (editable) ..."
    "$CONDA_ENV_DIR/bin/pip" install -e "$SCRIPT_DIR/python" --no-build-isolation -q
    echo "Installing fast-hadamard-transform ..."
    CUDA_HOME="$CONDA_ENV_DIR/cuda-home" \
    PATH="$CONDA_ENV_DIR/nvvm/bin:$CONDA_ENV_DIR/bin:/usr/bin:$PATH" \
        "$CONDA_ENV_DIR/bin/pip" install \
        "git+https://github.com/Dao-AILab/fast-hadamard-transform.git" \
        --no-build-isolation -q
    echo "Installing tore-eval (editable) ..."
    rm -rf "$TORE_EVAL_DIR/src/tore_eval.egg-info" 2>/dev/null || true
    "$CONDA_ENV_DIR/bin/pip" install -e "$TORE_EVAL_DIR" -q
    echo "✓ Conda environment ready"
fi

PYTHON="$CONDA_ENV_DIR/bin/python3"

# =============================================================================
# Dataset Preparation
# =============================================================================
bash "$SCRIPT_DIR/prepare_datasets.sh" "$PYTHON" "$SCRIPT_DIR"

# =============================================================================
# Helper Functions
# =============================================================================

extract_model_short_name() {
    basename "$1"
}

# Return a path that doesn't exist yet: if base doesn't exist return base,
# otherwise return base-1, base-2, ... (without extension awareness needed)
unique_log_path() {
    local base="$1"
    if [ ! -e "$base" ]; then
        echo "$base"
        return
    fi
    local i=1
    while [ -e "${base}-${i}" ]; do
        i=$((i + 1))
    done
    echo "${base}-${i}"
}

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$BATCH_LOG_FILE"
}

# =============================================================================
# eval_single_model
# =============================================================================

eval_single_model() {
    local mode="$1"
    local hadamard="$2"
    local rotate_v="$3"
    local hadamard_order="$4"
    local kv_dtype="$5"
    local model_name="$6"
    local tp_size="$7"
    local ep_size="$8"
    local dp_size="$9"
    local gpu_devices="${10}"
    local tasks="${11}"
    local server_port="${12}"
    local model_short_name
    model_short_name=$(extract_model_short_name "$model_name")

    # Validate mode
    if [[ "$mode" != "BASE" && "$mode" != "QUANT" ]]; then
        echo "ERROR: mode must be BASE or QUANT, got: '$mode'"
        return 1
    fi

    # For BASE mode, force no rotation (kv_dtype still applies)
    if [[ "$mode" == "BASE" ]]; then
        hadamard=0
        rotate_v=0
    fi

    # Validate kv_dtype
    if [[ "$kv_dtype" != "BF16" && "$kv_dtype" != "INT4" ]]; then
        echo "ERROR: kv_dtype must be BF16 or INT4, got: '$kv_dtype'"
        return 1
    fi
    local kv_cache_dtype
    if [[ "$kv_dtype" == "BF16" ]]; then
        kv_cache_dtype="auto"
    else
        kv_cache_dtype="int4"
    fi

    local kv_dtype_lower="${kv_dtype,,}"
    local rot_suffix
    if [[ "$mode" == "BASE" ]]; then
        rot_suffix="baseline_${kv_dtype_lower}"
    else
        rot_suffix="quant_${kv_dtype_lower}_${hadamard}_${rotate_v}_${hadamard_order}"
    fi
    mkdir -p "$LOGS_DIR/batch_logs/${model_short_name}" "$LOGS_DIR/inference_logs/${model_short_name}"
    BATCH_LOG_FILE=$(unique_log_path "$LOGS_DIR/batch_logs/${model_short_name}/${rot_suffix}.log")

    log_message "=========================================="
    log_message "Mode:     $mode"
    log_message "Model:    $model_name"
    log_message "TP/EP/DP: $tp_size/$ep_size/$dp_size"
    log_message "GPUs:     $gpu_devices"
    log_message "Tasks:    $tasks"
    log_message "HADAMARD=$hadamard  ROTATE_V=$rotate_v  HADAMARD_ORDER=$hadamard_order  KV_DTYPE=$kv_dtype"
    log_message "=========================================="

    # ------------------------------------------------------------------
    # Step 1: Start SGLang server
    # ------------------------------------------------------------------
    log_message "Starting SGLang server on port $server_port..."

    SERVER_LOG=$(unique_log_path "$LOGS_DIR/inference_logs/${model_short_name}/${rot_suffix}_server.log")

    HADAMARD=$hadamard \
    ROTATE_V=$rotate_v \
    HADAMARD_ORDER=$hadamard_order \
    CUDA_VISIBLE_DEVICES=$gpu_devices \
    PATH="$(dirname "$PYTHON"):$PATH" \
    LIBRARY_PATH="/usr/local/cuda/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$PYTHON" -m sglang.launch_server \
        --model-path "$model_name" \
        --max-running-requests 32 \
        --max-queued-requests 64 \
        --page-size 128 \
        --chunked-prefill-size 4096 \
        --mem-fraction-static 0.8 \
        --pp-max-micro-batch-size 32 \
        --kv-cache-dtype "$kv_cache_dtype" \
        --prefill-attention-backend fa3 \
        --decode-attention-backend triton \
        --sampling-backend flashinfer \
        --tensor-parallel-size "$tp_size" \
        --data-parallel-size "$dp_size" \
        --host 0.0.0.0 \
        --port "$server_port" \
        > "$SERVER_LOG" 2>&1 &

    SERVER_PID=$!
    log_message "Server started (PID: $SERVER_PID)"

    # ------------------------------------------------------------------
    # Step 2: Wait for server to be ready
    # ------------------------------------------------------------------
    log_message "Waiting for server to be ready..."
    MAX_WAIT=1800
    ELAPSED=0
    while [ $ELAPSED -lt $MAX_WAIT ]; do
        if curl -s "http://localhost:${server_port}/health" > /dev/null 2>&1; then
            log_message "✓ Server ready (${ELAPSED}s)"
            break
        fi
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            log_message "✗ Server process died"
            tail -50 "$SERVER_LOG"
            return 1
        fi
        if [ $((ELAPSED % 30)) -eq 0 ] && [ $ELAPSED -gt 0 ]; then
            log_message "  Still waiting... ${ELAPSED}s"
        fi
        sleep 5
        ELAPSED=$((ELAPSED + 5))
    done

    if [ $ELAPSED -ge $MAX_WAIT ]; then
        log_message "✗ Server timeout after ${MAX_WAIT}s"
        kill $SERVER_PID 2>/dev/null || true
        return 1
    fi

    # ------------------------------------------------------------------
    # Step 3: Run evaluations
    # ------------------------------------------------------------------
    local overall_exit=0
    IFS=',' read -ra TASK_LIST <<< "$tasks"
    for TASK_WITH_REPEAT in "${TASK_LIST[@]}"; do
        if [[ "$TASK_WITH_REPEAT" == *":"* ]]; then
            TASK_NAME="${TASK_WITH_REPEAT%%:*}"
            REPEAT="${TASK_WITH_REPEAT##*:}"
        else
            TASK_NAME="$TASK_WITH_REPEAT"
            REPEAT=1
        fi

        log_message "=========================================="
        log_message "Task: $TASK_NAME (repeat x${REPEAT})"
        log_message "=========================================="

        for RUN_IDX in $(seq 1 $REPEAT); do
            RUN_DIR="$RESULTS_DIR/${model_short_name}/${model_short_name}_${TASK_NAME}_${rot_suffix}/run${RUN_IDX}"
            mkdir -p "$RUN_DIR"
            log_message "  Run ${RUN_IDX}/${REPEAT} -> $RUN_DIR"

            cd "$SCRIPT_DIR"
            set +e
            "$PYTHON" -m tore_eval.eval \
                --framework preset \
                --preset_name "$TASK_NAME" \
                --model_name_or_path "$model_name" \
                --provider custom \
                --base_url "http://localhost:${server_port}/v1" \
                --api_key "" \
                --num_workers "$NUM_WORKERS" \
                --log_file "${RUN_DIR}/samples.jsonl" \
                --loggers "{\"local\": {\"output_dir\": \"${RUN_DIR}\"}}" \
                2>&1 | tee -a "$BATCH_LOG_FILE"
            TASK_EXIT=${PIPESTATUS[0]}
            set -e

            if [ $TASK_EXIT -ne 0 ]; then
                log_message "  ✗ Run ${RUN_IDX} failed (exit: $TASK_EXIT)"
                overall_exit=$TASK_EXIT
            else
                log_message "  ✓ Run ${RUN_IDX} completed"
            fi
        done

        # Aggregate results across runs (only if repeat > 1)
        if [ $REPEAT -gt 1 ]; then
            TASK_DIR="$RESULTS_DIR/${model_short_name}/${model_short_name}_${TASK_NAME}_${rot_suffix}"
            log_message "Aggregating ${REPEAT} runs for $TASK_NAME..."
            "$PYTHON" - <<PYEOF
import json, math, os

task_dir = "${TASK_DIR}"
n_runs = ${REPEAT}

all_metrics = []
for run_idx in range(1, n_runs + 1):
    results_file = os.path.join(task_dir, f"run{run_idx}", "results.jsonl")
    if not os.path.exists(results_file):
        print(f"Warning: {results_file} not found, skipping run {run_idx}")
        continue
    with open(results_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                metrics = obj.get("metrics", {})
                if metrics:
                    all_metrics.append(metrics)
                    break
            except json.JSONDecodeError:
                continue

if not all_metrics:
    print("No metrics found across runs, skipping aggregation")
else:
    all_keys = set()
    for m in all_metrics:
        all_keys.update(m.keys())

    aggregated = {}
    for key in sorted(all_keys):
        values = [m[key] for m in all_metrics if key in m and m[key] is not None]
        if not values:
            continue
        n = len(values)
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / n if n > 1 else 0.0
        std = math.sqrt(variance)
        half_range = max(max(values) - mean, mean - min(values))
        aggregated[key] = {
            "mean": mean,
            "std": std,
            "half_range": half_range,
            "values": values,
            "n_runs": n,
        }

    out_file = os.path.join(task_dir, "aggregated.json")
    with open(out_file, "w") as f:
        json.dump(aggregated, f, indent=4)
    print(f"✓ Aggregated {len(aggregated)} metrics from {len(all_metrics)} runs -> {out_file}")
    for key, stats in aggregated.items():
        print(f"  {key}: mean={stats['mean']:.4f}  std={stats['std']:.4f}  half_range={stats['half_range']:.4f}")
PYEOF
        fi
    done

    # ------------------------------------------------------------------
    # Step 4: Stop server
    # ------------------------------------------------------------------
    log_message "Stopping server (PID: $SERVER_PID)..."
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    log_message "✓ Server stopped"

    return $overall_exit
}

# =============================================================================
# Main
# =============================================================================

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] =========================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] SGLang KV Rotation Evaluation"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Configs: ${#MODEL_CONFIGS[@]} entry(s)"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] =========================================="
echo ""

declare -a PIDS
declare -A CONFIG_LABELS

for i in "${!MODEL_CONFIGS[@]}"; do
    config="${MODEL_CONFIGS[$i]}"
    port=$((BASE_PORT + i))
    IFS='|' read -r mode hadamard rotate_v hadamard_order kv_dtype model_name tp_size ep_size dp_size gpu_devices tasks <<< "$config"
    model_short_name=$(extract_model_short_name "$model_name")
    # Compute label (BASE forces hadamard/rotate_v=0, kv_dtype unchanged)
    local_kv_dtype_lower="${kv_dtype,,}"
    if [[ "$mode" == "BASE" ]]; then
        rot_suffix="baseline_${local_kv_dtype_lower}"
    else
        rot_suffix="quant_${local_kv_dtype_lower}_${hadamard}_${rotate_v}_${hadamard_order}"
    fi
    CONFIG_LABELS[$i]="${model_short_name}_${rot_suffix} (port=$port)"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching: ${CONFIG_LABELS[$i]}"
    eval_single_model "$mode" "$hadamard" "$rotate_v" "$hadamard_order" "$kv_dtype" "$model_name" "$tp_size" "$ep_size" "$dp_size" "$gpu_devices" "$tasks" "$port" &
    PIDS[$i]=$!
done

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All configs launched, waiting for completion..."
echo ""

OVERALL_EXIT=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ ${CONFIG_LABELS[$i]}"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ ${CONFIG_LABELS[$i]}"
        OVERALL_EXIT=1
    fi
done

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All done. Exit code: $OVERALL_EXIT"
exit $OVERALL_EXIT
