#!/bin/bash
#SBATCH --job-name=livecodebench-eval
#SBATCH --output=logs/slurm_logs/slurm_%j.out
#SBATCH --error=logs/slurm_logs/slurm_%j.out
#SBATCH --time=08:00:00
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64

# All-in-One Docker Model Evaluation Script
# Works with both SLURM (sbatch) and direct bash execution
#
# Usage:
#   sbatch eval_kv_rotation.sh
#   bash eval_kv_rotation.sh
#
# Temperature / top_p:
#   These cannot be overridden via CLI when using --framework preset.
#   Edit the corresponding preset YAML file in:
#   tore-eval/src/tore_eval/evaluators/preset/configs/<preset_name>.yaml

set -eo pipefail

# =============================================================================
# Configuration
# =============================================================================

# Model configurations
# Format: "model_name|tp_size|ep_size|dp_size|gpu_devices|tasks"
#   model_name  : full HuggingFace model ID (e.g. Qwen/Qwen3-8B)
#   gpu_devices : comma-separated GPU IDs, e.g. "0,1" or "4,5,6,7"
#   tasks       : comma-separated preset names, each with optional :N repeat count
#                 e.g. "customized_livecodebench_think:3,gpqa_think:1"
#                 default repeat = 1 if :N is omitted
#
# Available preset names:
#   gpqa_think
#   humaneval_think
#   customized_livecodebench_think   (release_v6, 2025-01-05 ~ 2025-04-07)
#   aime24_think
#   math_500_think
#
# Models run in parallel; tasks within each model run sequentially.
# Repeat runs for a task are also sequential.
# Output: eval_results/{model_short_name}_{task}/run{1..N}/
# After all runs: eval_results/{model_short_name}_{task}/aggregated.json
MODEL_CONFIGS=(
    "Qwen/Qwen3-8B|1|1|1|0,1,2,3,4|gpqa_think:5,humaneval_think:5,customized_livecodebench_think:5,aime24_think:5,math_500_think:5"
)

# Server config
BASE_PORT=8081
CONTAINER_PORT=8000
NUM_WORKERS=64

# Docker config
DOCKER_IMAGE="lmsysorg/sglang:dev"

# Paths
if [ ! -z "$SLURM_SUBMIT_DIR" ]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
TORE_EVAL_DIR="$SCRIPT_DIR/tore-eval"
RESULTS_DIR="$SCRIPT_DIR/eval_results"
LOGS_DIR="$SCRIPT_DIR/eval_logs"
HF_CACHE_DIR="/data/shared/huggingface/hub"
DG_CACHE_DIR="/data/shared/deep_gemm_cache"

# =============================================================================
# Setup
# =============================================================================

mkdir -p "$DG_CACHE_DIR"
mkdir -p "$LOGS_DIR/inference_logs"
mkdir -p "$LOGS_DIR/slurm_logs"
mkdir -p "$LOGS_DIR/batch_logs"
mkdir -p "$RESULTS_DIR"

# Get HuggingFace token
if [ -z "$HUGGING_FACE_HUB_TOKEN" ]; then
    if [ -f ~/.cache/huggingface/token ]; then
        export HUGGING_FACE_HUB_TOKEN=$(cat ~/.cache/huggingface/token)
    fi
fi

if [ -z "$HUGGING_FACE_HUB_TOKEN" ]; then
    echo "ERROR: HUGGING_FACE_HUB_TOKEN not set"
    echo "Please set it or create ~/.cache/huggingface/token"
    exit 1
fi

# =============================================================================
# Helper Functions
# =============================================================================

log_message() {
    local message="$1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $message" | tee -a "$BATCH_LOG_FILE"
}

extract_model_short_name() {
    # "org/model-name" -> "model-name"
    local model_name="$1"
    basename "$model_name"
}

cleanup_container() {
    local container_name="$1"
    if sudo docker ps -a --format '{{.Names}}' | grep -q "^${container_name}$"; then
        log_message "  Cleaning up container: $container_name"
        sudo docker stop "$container_name" >/dev/null 2>&1 || true
        sudo docker rm "$container_name" >/dev/null 2>&1 || true
    fi
}

eval_single_model() {
    local model_name="$1"
    local port="$2"
    local tp_size="$3"
    local ep_size="$4"
    local dp_size="$5"
    local gpu_devices="$6"
    local tasks="$7"        # comma-separated, each may have :N repeat suffix
    local model_short_name=$(extract_model_short_name "$model_name")
    local container_name="sglang-eval-${port}"
    BATCH_LOG_FILE="$LOGS_DIR/batch_logs/${model_short_name}_${port}.log"

    log_message "=========================================="
    log_message "Evaluating: $model_name"
    log_message "=========================================="
    log_message "Container: $container_name"
    log_message "Port: $port"
    log_message "TP/EP/DP: $tp_size/$ep_size/$dp_size"
    log_message "Tasks: $tasks"

    cleanup_container "$container_name"

    log_message ""
    log_message "Starting Docker container..."

    # Script executed inside the container
    local container_script=$(cat <<'EOF_SCRIPT'
#!/bin/bash
set -e

MODEL_NAME="$1"
MODEL_SHORT_NAME="$2"
PORT="$3"
TP_SIZE="$4"
EP_SIZE="$5"
DP_SIZE="$6"
NUM_WORKERS="$7"
TASKS="$8"   # comma-separated, each may have :N repeat suffix
HOST_PORT="$9"  # host-side port, used for log filenames only

echo "=== Container Execution Started ==="
echo "Model: $MODEL_NAME"
echo "Port: $PORT"
echo "TP/EP/DP: $TP_SIZE/$EP_SIZE/$DP_SIZE"
echo "Tasks: $TASKS"
echo ""

# Step 1: Install tore_eval from mounted directory
echo "Step 1: Installing tore_eval..."
cd /workspace/tore-eval
# Remove stale egg-info that may be owned by a different user (e.g. root from a previous run)
rm -rf src/tore_eval.egg-info 2>/dev/null || true
pip install -e . -q --user --break-system-packages --no-cache-dir --no-warn-script-location
pip install ray word2number -q --user --break-system-packages --no-cache-dir --no-warn-script-location
echo "✓ tore_eval installed"
echo ""

# Step 2: Download model from HuggingFace
echo "Step 2: Downloading model from HuggingFace..."
MODEL_PATH=$(python3 -c "
from huggingface_hub import snapshot_download
import os
path = snapshot_download(
    repo_id='${MODEL_NAME}',
    cache_dir='/root/.cache/huggingface',
    token=os.getenv('HUGGING_FACE_HUB_TOKEN')
)
print(path)
")
echo "✓ Model downloaded to: $MODEL_PATH"
echo ""

# Step 3: Start SGLang server in background
echo "Step 3: Starting SGLang server..."
python3 -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 119}' \
    --disable-flashinfer-autotune \
    --trust-remote-code \
    --log-requests \
    --log-requests-level 0 \
    --mem-fraction-static 0.85 \
    --tp "$TP_SIZE" \
    --ep "$EP_SIZE" \
    --dp "$DP_SIZE" \
    > /workspace/logs/inference_logs/${MODEL_SHORT_NAME}_${HOST_PORT}_server.log 2>&1 &

SERVER_PID=$!
echo "✓ Server started (PID: $SERVER_PID)"
echo ""

# Step 4: Wait for server to be ready
echo "Step 4: Waiting for server to be ready..."
MAX_WAIT=1800
ELAPSED=0
while [ $ELAPSED -lt $MAX_WAIT ]; do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
        echo "✓ Server is ready (after ${ELAPSED}s)"
        break
    fi

    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "✗ Server process died"
        tail -50 /workspace/logs/inference_logs/${MODEL_SHORT_NAME}_${HOST_PORT}_server.log
        exit 1
    fi

    if [ $((ELAPSED % 10)) -eq 0 ]; then
        echo "  Still waiting... ${ELAPSED}s elapsed"
    fi

    sleep 2
    ELAPSED=$((ELAPSED + 2))
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo "✗ Server did not become ready within ${MAX_WAIT}s"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi
echo ""

# Step 5: Run evaluations (sequentially per task, with optional repeats)
OVERALL_EXIT_CODE=0
IFS=',' read -ra TASK_LIST <<< "$TASKS"
for TASK_WITH_REPEAT in "${TASK_LIST[@]}"; do
    # Parse task name and repeat count: "task_name:N" or just "task_name"
    if [[ "$TASK_WITH_REPEAT" == *":"* ]]; then
        TASK_NAME="${TASK_WITH_REPEAT%%:*}"
        REPEAT="${TASK_WITH_REPEAT##*:}"
    else
        TASK_NAME="$TASK_WITH_REPEAT"
        REPEAT=1
    fi

    echo "=========================================="
    echo "Task: $TASK_NAME (repeat x${REPEAT})"
    echo "=========================================="

    for RUN_IDX in $(seq 1 $REPEAT); do
        RUN_DIR="/workspace/results/${MODEL_SHORT_NAME}_${TASK_NAME}/run${RUN_IDX}"
        mkdir -p "$RUN_DIR"

        echo "------------------------------------------"
        echo "  Run ${RUN_IDX}/${REPEAT}: $TASK_NAME -> $RUN_DIR"
        echo "------------------------------------------"

        cd /workspace
        python3 -m tore_eval.eval \
            --framework preset \
            --preset_name "$TASK_NAME" \
            --model_name_or_path "$MODEL_NAME" \
            --provider custom \
            --base_url "http://localhost:${PORT}/v1" \
            --api_key "" \
            --num_workers "$NUM_WORKERS" \
            --log_file "${RUN_DIR}/samples.jsonl" \
            --loggers "{\"local\": {\"output_dir\": \"${RUN_DIR}\"}}"

        TASK_EXIT_CODE=$?
        if [ $TASK_EXIT_CODE -ne 0 ]; then
            echo "✗ Run ${RUN_IDX} of task $TASK_NAME failed (exit code: $TASK_EXIT_CODE)"
            OVERALL_EXIT_CODE=$TASK_EXIT_CODE
        else
            echo "✓ Run ${RUN_IDX} of task $TASK_NAME completed"
        fi
        echo ""
    done

    # Aggregate results across runs (if more than 1 run)
    if [ $REPEAT -gt 1 ]; then
        TASK_RESULTS_DIR="/workspace/results/${MODEL_SHORT_NAME}_${TASK_NAME}"
        echo "Aggregating ${REPEAT} runs for $TASK_NAME..."
        python3 - <<PYEOF
import json, math, os

task_dir = "${TASK_RESULTS_DIR}"
n_runs = ${REPEAT}

# Collect metrics from each run's results.jsonl
all_metrics = []
for run_idx in range(1, n_runs + 1):
    run_dir = os.path.join(task_dir, f"run{run_idx}")
    results_file = os.path.join(run_dir, "results.jsonl")
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
                    break  # one entry per run file
            except json.JSONDecodeError:
                continue

if not all_metrics:
    print("No metrics found across runs, skipping aggregation")
else:
    # Collect all metric keys
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
        max_val = max(values)
        min_val = min(values)
        half_range = max(max_val - mean, mean - min_val)
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
        print(f"  {key}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, half_range={stats['half_range']:.4f}")
PYEOF
        echo ""
    fi

done

# Cleanup
echo "Cleaning up..."
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
echo "✓ Server stopped"
echo ""

if [ $OVERALL_EXIT_CODE -eq 0 ]; then
    echo "=== All Tasks Completed Successfully ==="
else
    echo "=== Some Tasks Failed (last exit code: $OVERALL_EXIT_CODE) ==="
fi

exit $OVERALL_EXIT_CODE
EOF_SCRIPT
)

    local start_time=$(date +%s)

    set +e
    echo "$container_script" | sudo docker run -i --rm \
        --name "$container_name" \
        --gpus "\"device=$gpu_devices\"" \
        -v "$TORE_EVAL_DIR:/workspace/tore-eval" \
        -v "$RESULTS_DIR:/workspace/results" \
        -v "$LOGS_DIR:/workspace/logs" \
        -v "$HF_CACHE_DIR:/root/.cache/huggingface" \
        -v "$DG_CACHE_DIR:/root/.cache/deep_gemm" \
        -e HUGGING_FACE_HUB_TOKEN="$HUGGING_FACE_HUB_TOKEN" \
        -e SGLANG_DISABLE_CUDNN_CHECK=1 \
        -p "${port}:${CONTAINER_PORT}" \
        --ipc=host \
        --shm-size=32g \
        "$DOCKER_IMAGE" \
        bash -s "$model_name" "$model_short_name" "$CONTAINER_PORT" "$tp_size" "$ep_size" "$dp_size" "$NUM_WORKERS" "$tasks" "$port" \
        2>&1 | tee -a "$BATCH_LOG_FILE"

    local exit_code=$?
    set -e

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    if [ $exit_code -eq 0 ]; then
        log_message "✓ Evaluation completed successfully (${duration}s)"
        return 0
    else
        log_message "✗ Evaluation failed with exit code $exit_code (${duration}s)"
        return 1
    fi
}

# =============================================================================
# Main Execution
# =============================================================================

declare -A RESULTS
declare -A PIDS
declare -A MODEL_NAMES
SUCCESS_COUNT=0
FAILURE_COUNT=0
TOTAL_MODELS=${#MODEL_CONFIGS[@]}

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting $TOTAL_MODELS models in parallel..."
echo ""

for i in "${!MODEL_CONFIGS[@]}"; do
    config="${MODEL_CONFIGS[$i]}"
    port=$((BASE_PORT + i))

    # Parse config: model_name|tp_size|ep_size|dp_size|gpu_devices|tasks
    IFS='|' read -r model_name tp_size ep_size dp_size gpu_devices tasks <<< "$config"
    model_short_name=$(extract_model_short_name "$model_name")
    MODEL_NAMES[$i]="$model_short_name"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching: $model_name (port=$port, GPUs=$gpu_devices, tasks=$tasks)"

    eval_single_model "$model_name" "$port" "$tp_size" "$ep_size" "$dp_size" "$gpu_devices" "$tasks" &
    PIDS[$i]=$!
done

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All models launched, waiting for completion..."
echo ""

for i in "${!PIDS[@]}"; do
    model_short_name="${MODEL_NAMES[$i]}"
    if wait ${PIDS[$i]}; then
        RESULTS[$model_short_name]="SUCCESS"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ $model_short_name: SUCCESS"
    else
        RESULTS[$model_short_name]="FAILED"
        FAILURE_COUNT=$((FAILURE_COUNT + 1))
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ $model_short_name: FAILED"
    fi
done

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "=========================================="
echo "Batch Evaluation Summary"
echo "=========================================="
echo "Total: $TOTAL_MODELS | Success: $SUCCESS_COUNT | Failed: $FAILURE_COUNT"
for model_short_name in "${!RESULTS[@]}"; do
    echo "  ${RESULTS[$model_short_name]}: $model_short_name"
done
echo "Results saved to: $RESULTS_DIR"
echo "=========================================="

exit 0
