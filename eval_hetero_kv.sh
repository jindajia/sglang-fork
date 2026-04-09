#!/bin/bash
# eval_hetero_kv.sh — Heterogeneous KV cache evaluation for DeepSeek-V3.1
#
# Tests different KV cache dtype configurations:
#   - fp8_e4m3 (all layers)
#   - fp4_e2m1 (all layers)
#   - fp4_e2m1 with select layers using fp8_e4m3 (heterogeneous)
#
# Uses tore_eval for accuracy benchmarks (GPQA etc.)

set -eo pipefail

cleanup() {
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Interrupted — killing all child processes..."
    kill -- -$$ 2>/dev/null || true
    exit 130
}
trap cleanup INT TERM

# =============================================================================
# Environment (from temp.sh)
# =============================================================================
eval "$(/scratch/jisenli2/miniconda/bin/conda shell.bash hook)" && conda activate sglang_hetero_kv_env
export HF_HOME=/scratch/huggingface
# export HF_HUB_OFFLINE=1
export FLASHINFER_CACHE_DIR=/scratch/jisenli2/.cache/flashinfer
export SGLANG_DG_CACHE_DIR=/scratch/jisenli2/.cache/deep_gemm
export CUDA_HOME=/usr/local/cuda-12.9
export CPATH=/usr/local/cuda-12.9/targets/x86_64-linux/include${CPATH:+:$CPATH}
export LIBRARY_PATH=/usr/local/cuda-12.9/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

# =============================================================================
# Model Configs
# =============================================================================
# Format: "kv_cache_dtype|per_layer_dtype|model_path|eval_gpus|eval_tp|tasks"
#
#   kv_cache_dtype  : fp8_e4m3, fp4_e2m1, etc. (passed to --kv-cache-dtype)
#   per_layer_dtype : "" (none) or e.g. "0-1:fp8_e4m3" (passed to --kv-cache-per-layer-dtype)
#   model_path      : local path or HuggingFace model ID
#   eval_gpus       : comma-separated GPU IDs
#   eval_tp         : tensor parallel size
#   tasks           : comma-separated tore_eval task names with optional :N repeat
#                     Each name maps to a YAML config in eval_configs/ (e.g. kimi_gpqa_think)
#                     or a built-in preset name (e.g. gpqa_think)

TASKS_DEFAULT="kimi_gpqa_think:3"

MODEL_CONFIGS=(
    # --- nvidia/Kimi-K2.5-NVFP4 on 8x B200 (TP=8) ---
    # 1) All layers fp8_e4m3 (model default)
    "fp8_e4m3||nvidia/Kimi-K2.5-NVFP4|0,1,2,3,4,5,6,7|8|${TASKS_DEFAULT}"

    # 2) All layers fp4_e2m1
    "fp4_e2m1||nvidia/Kimi-K2.5-NVFP4|0,1,2,3,4,5,6,7|8|${TASKS_DEFAULT}"

    # 3) fp4_e2m1 with layer 0-1 using fp8_e4m3
    "fp4_e2m1|0-1:fp8_e4m3|nvidia/Kimi-K2.5-NVFP4|0,1,2,3,4,5,6,7|8|${TASKS_DEFAULT}"
)

# =============================================================================
# Server & Eval Config
# =============================================================================
BASE_PORT=30100
NUM_WORKERS=4

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TORE_EVAL_DIR="$SCRIPT_DIR/tore-eval"
RESULTS_DIR="$SCRIPT_DIR/eval_results"
LOGS_DIR="$SCRIPT_DIR/eval_logs"

PYTHON="$(which python3)"

export TRITON_CACHE_DIR="/dev/shm/triton_cache_$USER"

GPU_FREE_MEM_MB="${GPU_FREE_MEM_MB:-500}"
GPU_POLL_INTERVAL="${GPU_POLL_INTERVAL:-60}"

mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

# =============================================================================
# Helpers
# =============================================================================

extract_model_short_name() { basename "$1"; }

unique_log_path() {
    local base="$1"
    if [ ! -e "$base" ]; then echo "$base"; return; fi
    local i=1
    while [ -e "${base}-${i}" ]; do i=$((i + 1)); done
    echo "${base}-${i}"
}

BATCH_LOG_FILE=""
log_message() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$BATCH_LOG_FILE"; }

wait_for_server() {
    local port="$1" pid="$2" label="$3"
    local max_wait=1800 elapsed=0
    log_message "Waiting for $label (port $port)..."
    while [ $elapsed -lt $max_wait ]; do
        if curl -s "http://localhost:${port}/health" > /dev/null 2>&1; then
            log_message "Server ready (${elapsed}s)"
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            log_message "Server process died"
            return 1
        fi
        [ $((elapsed % 60)) -eq 0 ] && [ $elapsed -gt 0 ] && log_message "  Still waiting... ${elapsed}s"
        sleep 5
        elapsed=$((elapsed + 5))
    done
    log_message "Server timeout after ${max_wait}s"
    return 1
}

stop_server() {
    local pid="$1" label="$2"
    log_message "Stopping $label (PID $pid)..."
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    log_message "Stopped $label"
}

gpus_are_free() {
    local gpu_list="$1"
    IFS=',' read -ra GPU_IDS <<< "$gpu_list"
    for gpu_id in "${GPU_IDS[@]}"; do
        local used_mb
        used_mb=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null | tr -d '[:space:]')
        if [ -z "$used_mb" ]; then return 1; fi
        if [ "$used_mb" -ge "$GPU_FREE_MEM_MB" ]; then return 1; fi
    done
    return 0
}

wait_for_gpus_free() {
    local gpu_list="$1" label="$2"
    local waited=0
    while ! gpus_are_free "$gpu_list"; do
        if [ $((waited % 300)) -eq 0 ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for GPU(s) [$gpu_list] to be free... ($((waited / 60))min) [$label]"
        fi
        sleep "$GPU_POLL_INTERVAL"
        waited=$((waited + GPU_POLL_INTERVAL))
    done
}

# =============================================================================
# eval_single_config
# =============================================================================

eval_single_config() {
    local kv_cache_dtype="$1" per_layer_dtype="$2" model_path="$3" \
          gpu_devices="$4" tp_size="$5" tasks="$6" server_port="$7"

    local model_short
    model_short="$(extract_model_short_name "$model_path")"

    # Build suffix for result dirs
    local suffix="${kv_cache_dtype}"
    if [ -n "$per_layer_dtype" ]; then
        # e.g. "0-1:fp8_e4m3" -> "0-1_fp8_e4m3"
        suffix="${kv_cache_dtype}_perlayer_$(echo "$per_layer_dtype" | tr ':' '_')"
    fi

    mkdir -p "$LOGS_DIR/${model_short}"
    BATCH_LOG_FILE=$(unique_log_path "$LOGS_DIR/${model_short}/${suffix}.log")

    log_message "=========================================="
    log_message "Model:          $model_path"
    log_message "KV cache dtype: $kv_cache_dtype"
    log_message "Per-layer dtype: ${per_layer_dtype:-none}"
    log_message "TP: $tp_size  GPUs: $gpu_devices"
    log_message "Tasks:          $tasks"
    log_message "=========================================="

    # Build server command
    local server_log
    server_log=$(unique_log_path "$LOGS_DIR/${model_short}/${suffix}_server.log")

    local per_layer_flag=""
    if [ -n "$per_layer_dtype" ]; then
        per_layer_flag="--kv-cache-per-layer-dtype $per_layer_dtype"
    fi

    log_message "Starting server (port $server_port)..."
    CUDA_VISIBLE_DEVICES=$gpu_devices \
    "$PYTHON" -m sglang.launch_server \
        --model-path "$model_path" \
        --max-running-requests 32 \
        --max-queued-requests 32 \
        --page-size 128 \
        --chunked-prefill-size 4096 \
        --mem-fraction-static 0.8 \
        --pp-max-micro-batch-size 32 \
        --prefill-attention-backend trtllm_mla \
        --decode-attention-backend trtllm_mla \
        --trust-remote-code \
        --quantization modelopt_fp4 \
        --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 119}' \
        --enable-metrics \
        --enable-flashinfer-allreduce-fusion \
        --log-requests \
        --log-requests-level 0 \
        --enable-hierarchical-cache \
        --hicache-ratio 1.0 \
        --hicache-io-backend kernel \
        --hicache-write-policy write_through \
        --kv-cache-dtype "$kv_cache_dtype" \
        $per_layer_flag \
        --tensor-parallel-size "$tp_size" \
        --data-parallel-size 1 \
        --moe-runner-backend flashinfer_trtllm \
        --tool-call-parser kimi_k2 \
        --reasoning-parser kimi_k2 \
        --host 0.0.0.0 \
        --port "$server_port" \
        > "$server_log" 2>&1 &
    local server_pid=$!
    log_message "Server PID: $server_pid"

    if ! wait_for_server "$server_port" "$server_pid" "SGLang server"; then
        tail -50 "$server_log" | tee -a "$BATCH_LOG_FILE"
        stop_server "$server_pid" "SGLang server"
        return 1
    fi

    # Run tore_eval
    local overall_exit=0
    IFS=',' read -ra TASK_LIST <<< "$tasks"
    for TASK_WITH_REPEAT in "${TASK_LIST[@]}"; do
        IFS=':' read -r TASK_NAME REPEAT RUN_RANGE <<< "$TASK_WITH_REPEAT"
        REPEAT="${REPEAT:-1}"
        if [[ -n "$RUN_RANGE" ]]; then
            RUN_START="${RUN_RANGE%-*}"
            RUN_END="${RUN_RANGE#*-}"
        else
            RUN_START=1
            RUN_END=$REPEAT
        fi

        log_message "=========================================="
        log_message "Task: $TASK_NAME (repeat x${REPEAT}, runs ${RUN_START}-${RUN_END})"
        log_message "=========================================="

        for RUN_IDX in $(seq $RUN_START $RUN_END); do
            RUN_DIR="$RESULTS_DIR/${model_short}/${TASK_NAME}/${suffix}/run${RUN_IDX}"
            if [ -f "${RUN_DIR}/results.jsonl" ]; then
                log_message "  Run ${RUN_IDX}/${REPEAT} already done, skipping"
                continue
            fi
            mkdir -p "$RUN_DIR"
            log_message "  Run ${RUN_IDX}/${REPEAT} -> $RUN_DIR"

            cd "$SCRIPT_DIR"
            local yaml_config="$SCRIPT_DIR/eval_configs/${TASK_NAME}.yaml"
            if [ ! -f "$yaml_config" ]; then
                log_message "ERROR: eval config not found: $yaml_config"
                return 1
            fi
            set +e
            OPENAI_LOG=warning \
            HTTPX_LOG_LEVEL=warning \
            "$PYTHON" -m tore_eval.eval \
                "$yaml_config" \
                --model_name_or_path "$model_path" \
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
                log_message "  Run ${RUN_IDX} failed (exit: $TASK_EXIT)"
                overall_exit=$TASK_EXIT
            else
                log_message "  Run ${RUN_IDX} completed"
            fi
        done

        # Aggregate across runs if repeat > 1
        if [ $REPEAT -gt 1 ]; then
            TASK_DIR="$RESULTS_DIR/${model_short}/${TASK_NAME}/${suffix}"
            log_message "Aggregating ${REPEAT} runs for $TASK_NAME..."
            "$PYTHON" - <<PYEOF
import json, math, os
task_dir = "${TASK_DIR}"
n_runs = ${REPEAT}
missing = [i for i in range(1, n_runs + 1)
           if not os.path.exists(os.path.join(task_dir, f"run{i}", "results.jsonl"))]
if missing:
    print(f"Skipping aggregation: run(s) {missing} missing results.jsonl")
    exit(0)
all_metrics = []
for run_idx in range(1, n_runs + 1):
    results_file = os.path.join(task_dir, f"run{run_idx}", "results.jsonl")
    with open(results_file) as f:
        for line in f:
            line = line.strip()
            if not line: continue
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
    all_keys = set(k for m in all_metrics for k in m)
    aggregated = {}
    for key in sorted(all_keys):
        values = [m[key] for m in all_metrics if key in m and m[key] is not None]
        if not values: continue
        n = len(values)
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / n if n > 1 else 0.0
        std = math.sqrt(variance)
        half_range = max(max(values) - mean, mean - min(values))
        aggregated[key] = {"mean": mean, "std": std, "half_range": half_range, "values": values, "n_runs": n}
    out_file = os.path.join(task_dir, "aggregated.json")
    with open(out_file, "w") as f:
        json.dump(aggregated, f, indent=4)
    print(f"Aggregated {len(aggregated)} metrics from {len(all_metrics)} runs -> {out_file}")
    for key, stats in aggregated.items():
        print(f"  {key}: mean={stats['mean']:.4f}  std={stats['std']:.4f}  half_range={stats['half_range']:.4f}")
PYEOF
        fi
    done

    stop_server "$server_pid" "SGLang server"
    return $overall_exit
}

# =============================================================================
# Preflight
# =============================================================================

# 1. tore-eval submodule initialized
if [ ! -f "$TORE_EVAL_DIR/setup.py" ] && [ ! -f "$TORE_EVAL_DIR/pyproject.toml" ]; then
    echo "ERROR: tore-eval submodule not initialized."
    echo "       Run: git submodule update --init --recursive"
    exit 1
fi

# =============================================================================
# Main — sequential execution
# =============================================================================

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] =========================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Heterogeneous KV Cache Evaluation"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Configs: ${#MODEL_CONFIGS[@]} entry(s)"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] =========================================="
echo ""

OVERALL_EXIT=0
N=${#MODEL_CONFIGS[@]}

for i in "${!MODEL_CONFIGS[@]}"; do
    config="${MODEL_CONFIGS[$i]}"
    IFS='|' read -r kv_cache_dtype per_layer_dtype model_path gpu_devices tp_size tasks <<< "$config"
    # Strip whitespace
    kv_cache_dtype="${kv_cache_dtype// /}"; per_layer_dtype="${per_layer_dtype// /}"
    model_path="${model_path// /}"; gpu_devices="${gpu_devices// /}"
    tp_size="${tp_size// /}"; tasks="${tasks// /}"

    server_port=$((BASE_PORT + i))
    model_short="$(extract_model_short_name "$model_path")"
    label="[$((i+1))/${N}] ${model_short} kv=${kv_cache_dtype} perlayer=${per_layer_dtype:-none} (gpu=$gpu_devices)"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $label"
    wait_for_gpus_free "$gpu_devices" "$label"

    eval_single_config "$kv_cache_dtype" "$per_layer_dtype" "$model_path" \
                       "$gpu_devices" "$tp_size" "$tasks" "$server_port"
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED: $label"
        OVERALL_EXIT=1
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] OK: $label"
    fi

    # Cooldown between configs
    if [ $((i + 1)) -lt "$N" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Cooldown 30s..."
        sleep 30
    fi
done

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All done. Exit: $OVERALL_EXIT"
exit $OVERALL_EXIT
