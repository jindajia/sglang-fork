#!/bin/bash
# eval_kv_rotation.sh — Local SGLang evaluation script (non-Docker)
#
# Hadamard rotation settings are configured per-entry in MODEL_CONFIGS.

set -eo pipefail

# =============================================================================
# Model Configuration
# =============================================================================
# Format: "hadamard|rotate_v|hadamard_order|model_name|tp_size|ep_size|dp_size|gpu_devices|tasks"
#   hadamard       : 0 or 1 — enable Hadamard rotation on K/Q
#   rotate_v       : 0 or 1 — also rotate V and de-rotate attention output
#   hadamard_order : block size for block-Hadamard (e.g. 16)
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
#   aime24_think
#   math_500_think
MODEL_CONFIGS=(
    "0|0|16|Qwen/Qwen3-4B-Thinking-2507|4|1|2|0,1,2,3,4,5,6,7|gpqa_think:5,humaneval_think:5,customized_livecodebench_think:5,aime24_think:5,math_500_think:5"
)

# =============================================================================
# Server & Eval Config
# =============================================================================
SERVER_PORT=30001
NUM_WORKERS=64

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TORE_EVAL_DIR="$SCRIPT_DIR/tore-eval"
RESULTS_DIR="$SCRIPT_DIR/eval_results"
LOGS_DIR="$SCRIPT_DIR/eval_logs"

mkdir -p "$RESULTS_DIR" "$LOGS_DIR/inference_logs" "$LOGS_DIR/batch_logs"

# =============================================================================
# Helper Functions
# =============================================================================

extract_model_short_name() {
    basename "$1"
}

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$BATCH_LOG_FILE"
}

# =============================================================================
# eval_single_model
# =============================================================================

eval_single_model() {
    local hadamard="$1"
    local rotate_v="$2"
    local hadamard_order="$3"
    local model_name="$4"
    local tp_size="$5"
    local ep_size="$6"
    local dp_size="$7"
    local gpu_devices="$8"
    local tasks="$9"
    local model_short_name
    model_short_name=$(extract_model_short_name "$model_name")
    BATCH_LOG_FILE="$LOGS_DIR/batch_logs/${model_short_name}.log"

    log_message "=========================================="
    log_message "Model:    $model_name"
    log_message "TP/EP/DP: $tp_size/$ep_size/$dp_size"
    log_message "GPUs:     $gpu_devices"
    log_message "Tasks:    $tasks"
    log_message "HADAMARD=$hadamard  ROTATE_V=$rotate_v  HADAMARD_ORDER=$hadamard_order"
    log_message "=========================================="

    # ------------------------------------------------------------------
    # Step 1: Start SGLang server
    # ------------------------------------------------------------------
    log_message "Starting SGLang server on port $SERVER_PORT..."

    HADAMARD=$hadamard \
    ROTATE_V=$rotate_v \
    HADAMARD_ORDER=$hadamard_order \
    CUDA_VISIBLE_DEVICES=$gpu_devices \
    python3 -m sglang.launch_server \
        --model-path "$model_name" \
        --max-running-requests 32 \
        --max-queued-requests 32 \
        --page-size 128 \
        --chunked-prefill-size 4096 \
        --mem-fraction-static 0.8 \
        --pp-max-micro-batch-size 32 \
        --kv-cache-dtype auto \
        --prefill-attention-backend fa3 \
        --decode-attention-backend triton \
        --sampling-backend flashinfer \
        --tensor-parallel-size "$tp_size" \
        --data-parallel-size "$dp_size" \
        --host 0.0.0.0 \
        --port "$SERVER_PORT" \
        > "$LOGS_DIR/inference_logs/${model_short_name}_server.log" 2>&1 &

    SERVER_PID=$!
    log_message "Server started (PID: $SERVER_PID)"

    # ------------------------------------------------------------------
    # Step 2: Wait for server to be ready
    # ------------------------------------------------------------------
    log_message "Waiting for server to be ready..."
    MAX_WAIT=1800
    ELAPSED=0
    while [ $ELAPSED -lt $MAX_WAIT ]; do
        if curl -s "http://localhost:${SERVER_PORT}/health" > /dev/null 2>&1; then
            log_message "✓ Server ready (${ELAPSED}s)"
            break
        fi
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            log_message "✗ Server process died"
            tail -50 "$LOGS_DIR/inference_logs/${model_short_name}_server.log"
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
            RUN_DIR="$RESULTS_DIR/${model_short_name}_${TASK_NAME}/run${RUN_IDX}"
            mkdir -p "$RUN_DIR"
            log_message "  Run ${RUN_IDX}/${REPEAT} -> $RUN_DIR"

            cd "$SCRIPT_DIR"
            set +e
            PYTHONPATH="$TORE_EVAL_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
            python3 -m tore_eval.eval \
                --framework preset \
                --preset_name "$TASK_NAME" \
                --model_name_or_path "$model_name" \
                --provider custom \
                --base_url "http://localhost:${SERVER_PORT}/v1" \
                --api_key "" \
                --num_workers "$NUM_WORKERS" \
                --log_file "${RUN_DIR}/samples.jsonl" \
                --loggers "{\"local\": {\"output_dir\": \"${RUN_DIR}\"}}"
            TASK_EXIT=$?
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
            TASK_DIR="$RESULTS_DIR/${model_short_name}_${TASK_NAME}"
            log_message "Aggregating ${REPEAT} runs for $TASK_NAME..."
            python3 - <<PYEOF
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
echo "[$(date '+%Y-%m-%d %H:%M:%S')] HADAMARD=$HADAMARD  ROTATE_V=$ROTATE_V  HADAMARD_ORDER=$HADAMARD_ORDER"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] =========================================="
echo ""

OVERALL_EXIT=0
for config in "${MODEL_CONFIGS[@]}"; do
    IFS='|' read -r hadamard rotate_v hadamard_order model_name tp_size ep_size dp_size gpu_devices tasks <<< "$config"
    eval_single_model "$hadamard" "$rotate_v" "$hadamard_order" "$model_name" "$tp_size" "$ep_size" "$dp_size" "$gpu_devices" "$tasks" || OVERALL_EXIT=$?
done

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All done. Exit code: $OVERALL_EXIT"
exit $OVERALL_EXIT
