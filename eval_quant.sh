#!/bin/bash
# eval_quant.sh — SGLang evaluation for Hadamard / Hadamard+QR KV cache
#
# Modes:
#   BASE        — no rotation (BF16, FP8, or INT4 KV)
#   Rotation    — Hadamard rotation only (INT4 KV)
#   Rotation_QR — Hadamard rotation + learned Q-rotation matrix (INT4 KV)
#
# TODO: KMEANS mode (Hadamard + K-means centroids, 3-stage pipeline):
#   Stage 1 — Dump BF16 KV cache via lm_eval (idempotent)
#   Stage 2 — Compute per-layer K-means centroids (flash-kmeans, idempotent)
#   Stage 3 — Eval with INT4 server + centroid lookup
#   Reference implementation: /data/jisenli2/kv_rotation/eval_quant.sh
#   Config fields to add: num_layers|n_clusters|dump_gpus|dump_tp|dump_ep|dump_dp|kmeans_gpu
#
# GPU scheduling: configs are processed in order.
#   - Poll until required GPUs are free, then launch in the background.
#   - If the next config shares a GPU, wait for current job to finish before launching.
#   - Otherwise configs on non-overlapping GPUs run in parallel.

set -eo pipefail

cleanup() {
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Interrupted — killing all child processes..."
    kill -- -$$ 2>/dev/null || true
    exit 130
}
trap cleanup INT TERM

# =============================================================================
# Model Configs
# =============================================================================
# Format: "mode|rotate_k|rotate_v|hadamard_order|kv_dtype|model_name|q_rotation_path|eval_gpus|tp|ep|dp|tasks"
#
#   mode            : BASE, Rotation, or Rotation_QR
#   rotate_k        : 1 = rotate K before INT4 quantization (HADAMARD=1); ignored for BASE
#   rotate_v        : 1 = also rotate V; ignored for BASE
#   hadamard_order  : block size for block-Hadamard, e.g. 16, 64, 128; ignored for BASE
#   kv_dtype        : BF16, INT4, or FP8
#                     BASE — kv_dtype describes model weight format (BF16/FP8 → auto KV)
#                     Rotation/Rotation_QR — INT4 only (BF16/FP8 → auto KV if needed)
#   model_name      : full HuggingFace model ID
#   q_rotation_path : path to learned .pt rotation file (Rotation_QR only; use "" for others)
#   eval_gpus       : comma-separated GPU IDs for eval server (also used for overlap detection)
#   tp              : tensor parallel size
#   ep              : expert parallel size
#   dp              : data parallel size
#   tasks           : comma-separated preset names with optional :N repeat (e.g. gpqa_think:5)
#
# Available preset names:
#   gpqa_think, humaneval_think, customized_livecodebench_think, aime25_think, math_500_think
#
TASKS_ALL="math_500_think:5,aime25_think:5,gpqa_think:5,humaneval_think:5,customized_livecodebench_think:5"
TASKS_ONCE="math_500_think:1,aime25_think:1,gpqa_think:1,humaneval_think:1,customized_livecodebench_think:1"

MODEL_CONFIGS=(
    # mode         |rk|rv|ho |dtype|model                          |q_rotation_path                                                                          |gpus|tp|ep|dp|tasks
    # "BASE        |0 |0 |0  |BF16 |Qwen/Qwen3-4B-Thinking-2507   |                                                                                         |0   |1 |1 |1 |${TASKS_ALL}"
    # "BASE        |0 |0 |0  |INT4 |Qwen/Qwen3-4B-Thinking-2507   |                                                                                         |1   |1 |1 |1 |${TASKS_ONCE}"
    # "Rotation    |1 |0 |16 |INT4 |Qwen/Qwen3-4B-Thinking-2507   |                                                                                         |2   |1 |1 |1 |${TASKS_ALL}"
    "Rotation_QR |1 |0 |16 |INT4 |Qwen/Qwen3-4B-Thinking-2507   |/data/jisenli2/zhongzhu_kv/q_rotation_layer_second_moment_damp01.pt                      |3   |1 |1 |1 |${TASKS_ALL}"
)

# =============================================================================
# Server & Eval Config
# =============================================================================
BASE_PORT=30100
NUM_WORKERS=64

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TORE_EVAL_DIR="$SCRIPT_DIR/tore-eval"
RESULTS_DIR="$SCRIPT_DIR/eval_results"
LOGS_DIR="$SCRIPT_DIR/eval_logs"

export HF_HOME=/data/shared/huggingface
if [ -z "$HUGGING_FACE_HUB_TOKEN" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
    export HUGGING_FACE_HUB_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi
export HF_TOKEN="$HUGGING_FACE_HUB_TOKEN"

# Hardcoded conda env path
CONDA_ENV_DIR="/data/$USER/miniconda/envs/zhongzhu_kv"
PYTHON="$CONDA_ENV_DIR/bin/python3"

export TRITON_CACHE_DIR="/data/$USER/.triton/cache"

GPU_FREE_MEM_MB="${GPU_FREE_MEM_MB:-500}"
GPU_POLL_INTERVAL="${GPU_POLL_INTERVAL:-240}"

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

log_message() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$BATCH_LOG_FILE"; }

wait_for_server() {
    local port="$1" pid="$2" label="$3"
    local max_wait=1800 elapsed=0
    log_message "Waiting for $label (port $port)..."
    while [ $elapsed -lt $max_wait ]; do
        if curl -s "http://localhost:${port}/health" > /dev/null 2>&1; then
            log_message "✓ $label ready (${elapsed}s)"
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            log_message "✗ $label process died"
            return 1
        fi
        [ $((elapsed % 60)) -eq 0 ] && [ $elapsed -gt 0 ] && log_message "  Still waiting... ${elapsed}s"
        sleep 5
        elapsed=$((elapsed + 5))
    done
    log_message "✗ $label timeout after ${max_wait}s"
    return 1
}

stop_server() {
    local pid="$1" label="$2"
    log_message "Stopping $label (PID $pid)..."
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    log_message "✓ $label stopped"
}

gpus_are_free() {
    local gpu_list="$1"
    IFS=',' read -ra GPU_IDS <<< "$gpu_list"
    for gpu_id in "${GPU_IDS[@]}"; do
        local used_mb
        used_mb=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null | tr -d '[:space:]')
        if [ -z "$used_mb" ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: could not query GPU $gpu_id, assuming busy"
            return 1
        fi
        if [ "$used_mb" -ge "$GPU_FREE_MEM_MB" ]; then return 1; fi
    done
    return 0
}

wait_for_gpus_free() {
    local gpu_list="$1" label="$2"
    local waited=0
    while ! gpus_are_free "$gpu_list"; do
        if [ $((waited % 1200)) -eq 0 ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for GPU(s) [$gpu_list] to be free... ($((waited / 60))min) [$label]"
        fi
        sleep "$GPU_POLL_INTERVAL"
        waited=$((waited + GPU_POLL_INTERVAL))
    done
    if [ "$waited" -gt 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU(s) [$gpu_list] now free after $((waited / 60))min"
    fi
}

# =============================================================================
# eval_single_model
# =============================================================================

eval_single_model() {
    local mode="$1" rotate_k="$2" rotate_v="$3" hadamard_order="$4" kv_dtype="$5" \
          model_name="$6" q_rotation_path="$7" \
          tp_size="$8" ep_size="$9" dp_size="${10}" gpu_devices="${11}" \
          tasks="${12}" server_port="${13}"

    local model_short
    model_short="$(extract_model_short_name "$model_name")"

    # Validate mode
    if [[ "$mode" != "BASE" && "$mode" != "Rotation" && "$mode" != "Rotation_QR" ]]; then
        echo "ERROR: mode must be BASE, Rotation, or Rotation_QR, got: '$mode'"
        return 1
    fi

    # Validate q_rotation_path for Rotation_QR
    if [[ "$mode" == "Rotation_QR" ]]; then
        if [[ -z "$q_rotation_path" ]]; then
            echo "ERROR: Rotation_QR mode requires q_rotation_path to be set"
            return 1
        fi
        if [[ ! -f "$q_rotation_path" ]]; then
            echo "ERROR: q_rotation_path does not exist: $q_rotation_path"
            return 1
        fi
    fi

    # Determine kv_cache_dtype
    # BASE always uses auto; Rotation/Rotation_QR: INT4 → int4, BF16/FP8 → auto
    local kv_cache_dtype
    if [[ "$mode" == "BASE" ]]; then
        kv_cache_dtype="auto"
    else
        case "$kv_dtype" in
            INT4)     kv_cache_dtype="int4" ;;
            BF16|FP8) kv_cache_dtype="auto" ;;
            *) echo "ERROR: unknown kv_dtype '$kv_dtype'"; return 1 ;;
        esac
    fi

    local kv_dtype_lower="${kv_dtype,,}"
    local rot_suffix
    case "$mode" in
        BASE)
            if [[ "$kv_dtype" == "FP8" ]]; then
                rot_suffix="baseline_fp8_kv_bf16"
            else
                rot_suffix="baseline_${kv_dtype_lower}"
            fi
            ;;
        Rotation)
            rot_suffix="rotation_${kv_dtype_lower}_${rotate_k}_${rotate_v}_${hadamard_order}"
            ;;
        Rotation_QR)
            rot_suffix="rotation_qr_${kv_dtype_lower}_${rotate_k}_${rotate_v}_${hadamard_order}"
            ;;
    esac

    # BF16 KV cache uses more memory; reduce mem_fraction to leave room for CUDA graphs
    local mem_fraction
    if [[ "$kv_cache_dtype" == "auto" ]]; then
        mem_fraction="0.65"
    else
        mem_fraction="0.8"
    fi

    # int4 requires fa3 prefill + triton decode
    local prefill_backend decode_backend
    if [[ "$kv_cache_dtype" == "int4" ]]; then
        prefill_backend="fa3"
        decode_backend="triton"
    else
        prefill_backend="fa3"
        decode_backend="triton"
    fi

    mkdir -p "$LOGS_DIR/batch_logs/${model_short}" "$LOGS_DIR/inference_logs/${model_short}"
    BATCH_LOG_FILE=$(unique_log_path "$LOGS_DIR/batch_logs/${model_short}/${rot_suffix}.log")

    log_message "=========================================="
    log_message "Mode:          $mode"
    log_message "Model:         $model_name"
    log_message "TP/EP/DP:      $tp_size/$ep_size/$dp_size"
    log_message "GPUs:          $gpu_devices"
    log_message "KV dtype:      $kv_dtype (cache: $kv_cache_dtype)"
    log_message "rotate_k=$rotate_k  rotate_v=$rotate_v  hadamard_order=$hadamard_order"
    [[ "$mode" == "Rotation_QR" ]] && log_message "Q rotation:    $q_rotation_path"
    log_message "Tasks:         $tasks"
    log_message "=========================================="

    # ------------------------------------------------------------------
    # Start SGLang server
    # ------------------------------------------------------------------
    unset DUMP_KVCACHE DUMP_KVCACHE_TOKENS DUMP_KVCACHE_DIR
    log_message "Starting SGLang server on port $server_port..."
    SERVER_LOG=$(unique_log_path "$LOGS_DIR/inference_logs/${model_short}/${rot_suffix}_server.log")

    local local_pythonpath="${SCRIPT_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}"

    local -a server_env=(
        PYTHONPATH="${local_pythonpath}"
        CUDA_VISIBLE_DEVICES="${gpu_devices}"
        LIBRARY_PATH="/usr/local/cuda/targets/x86_64-linux/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
        LD_LIBRARY_PATH="/usr/local/cuda/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    )

    case "$mode" in
        BASE)
            server_env+=(HADAMARD=0)
            ;;
        Rotation)
            server_env+=(
                HADAMARD="${rotate_k}"
                HADAMARD_ORDER="${hadamard_order}"
                ROTATE_V="${rotate_v}"
            )
            ;;
        Rotation_QR)
            server_env+=(
                HADAMARD="${rotate_k}"
                HADAMARD_ORDER="${hadamard_order}"
                ROTATE_V="${rotate_v}"
                SGLANG_Q_ROTATION_PATH="${q_rotation_path}"
            )
            ;;
    esac

    env "${server_env[@]}" \
        "$PYTHON" -m sglang.launch_server \
            --model-path "$model_name" \
            --max-running-requests 32 \
            --max-queued-requests 256 \
            --page-size 128 \
            --chunked-prefill-size 4096 \
            --mem-fraction-static "$mem_fraction" \
            --pp-max-micro-batch-size 32 \
            --kv-cache-dtype "$kv_cache_dtype" \
            --prefill-attention-backend "$prefill_backend" \
            --decode-attention-backend "$decode_backend" \
            --sampling-backend flashinfer \
            --tensor-parallel-size "$tp_size" \
            --expert-parallel-size "$ep_size" \
            --data-parallel-size "$dp_size" \
            --host 0.0.0.0 \
            --port "$server_port" \
            --trust-remote-code \
            > "$SERVER_LOG" 2>&1 &
    local server_pid=$!
    log_message "Server started (PID: $server_pid)"

    if ! wait_for_server "$server_port" "$server_pid" "SGLang server"; then
        tail -50 "$SERVER_LOG" | tee -a "$BATCH_LOG_FILE"
        stop_server "$server_pid" "SGLang server"
        return 1
    fi

    # ------------------------------------------------------------------
    # Run evaluations
    # ------------------------------------------------------------------
    local overall_exit=0
    IFS=',' read -ra TASK_LIST <<< "$tasks"
    for TASK_WITH_REPEAT in "${TASK_LIST[@]}"; do
        # Format: task:N  or  task:N:start-end
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
            RUN_DIR="$RESULTS_DIR/${model_short}/${TASK_NAME}/${rot_suffix}/run${RUN_IDX}"
            if [ -f "${RUN_DIR}/results.jsonl" ]; then
                log_message "  Run ${RUN_IDX}/${REPEAT} already done, skipping"
                continue
            fi
            mkdir -p "$RUN_DIR"
            log_message "  Run ${RUN_IDX}/${REPEAT} -> $RUN_DIR"

            cd "$SCRIPT_DIR"
            set +e
            OPENAI_LOG=warning \
            HTTPX_LOG_LEVEL=warning \
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

        # Aggregate across runs if repeat > 1
        if [ $REPEAT -gt 1 ]; then
            TASK_DIR="$RESULTS_DIR/${model_short}/${TASK_NAME}/${rot_suffix}"
            log_message "Aggregating ${REPEAT} runs for $TASK_NAME..."
            "$PYTHON" - <<PYEOF
import json, math, os
task_dir = "${TASK_DIR}"
n_runs = ${REPEAT}
missing = [i for i in range(1, n_runs + 1)
           if not os.path.exists(os.path.join(task_dir, f"run{i}", "results.jsonl"))]
if missing:
    print(f"Skipping aggregation: run(s) {missing} missing results.jsonl — will aggregate when all {n_runs} runs complete")
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
    print(f"✓ Aggregated {len(aggregated)} metrics from {len(all_metrics)} runs -> {out_file}")
    for key, stats in aggregated.items():
        print(f"  {key}: mean={stats['mean']:.4f}  std={stats['std']:.4f}  half_range={stats['half_range']:.4f}")
PYEOF
        fi
    done

    stop_server "$server_pid" "SGLang server"
    return $overall_exit
}

# =============================================================================
# Preflight checks
# =============================================================================

# 1. tore-eval submodule initialized
if [ ! -f "$TORE_EVAL_DIR/setup.py" ] && [ ! -f "$TORE_EVAL_DIR/pyproject.toml" ]; then
    echo "ERROR: tore-eval submodule not initialized."
    echo "       Run: git submodule update --init --recursive"
    exit 1
fi

# 2. conda env exists
if [ ! -f "$PYTHON" ]; then
    echo "ERROR: conda env 'zhongzhu_kv' not found (expected Python at $PYTHON)."
    echo "       Run: bash setup_env.sh"
    exit 1
fi

# 3. HuggingFace auth (needed for togethercomputer/* private datasets)
_hf_ok=0
if [ -n "$HUGGING_FACE_HUB_TOKEN" ] || [ -f "$HOME/.cache/huggingface/token" ]; then
    _hf_ok=1
fi
if [ "$_hf_ok" -eq 0 ]; then
    _whoami=$(huggingface-cli whoami 2>/dev/null || true)
    if echo "$_whoami" | grep -q "togethercomputer"; then
        _hf_ok=1
    fi
fi
if [ "$_hf_ok" -eq 0 ]; then
    echo "ERROR: No HuggingFace token found and 'huggingface-cli whoami' does not show togethercomputer org."
    echo "       Run: huggingface-cli login"
    echo "       Or set HUGGING_FACE_HUB_TOKEN env var."
    exit 1
fi
unset _hf_ok _whoami

# 4. datasets
bash "$SCRIPT_DIR/prepare_datasets.sh" "$PYTHON" "$SCRIPT_DIR"

# =============================================================================
# Main — sequential scheduling with GPU-aware overlap detection
# =============================================================================

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] =========================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Hadamard/QR KV Cache Accuracy Evaluation"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Configs:   ${#MODEL_CONFIGS[@]} entry(s)"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Python:    $PYTHON"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU free threshold: ${GPU_FREE_MEM_MB} MB"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] =========================================="
echo ""

OVERALL_EXIT=0
declare -a PIDS
declare -a EXIT_CODES
declare -A CONFIG_LABELS
N=${#MODEL_CONFIGS[@]}

for i in "${!MODEL_CONFIGS[@]}"; do
    config="${MODEL_CONFIGS[$i]}"
    # Strip whitespace from each field (allows aligned formatting in MODEL_CONFIGS)
    IFS='|' read -r mode rotate_k rotate_v hadamard_order kv_dtype model_name \
                    q_rotation_path \
                    gpu_devices tp_size ep_size dp_size tasks <<< "$config"
    mode="${mode// /}"
    rotate_k="${rotate_k// /}"
    rotate_v="${rotate_v// /}"
    hadamard_order="${hadamard_order// /}"
    kv_dtype="${kv_dtype// /}"
    model_name="${model_name// /}"
    q_rotation_path="${q_rotation_path// /}"
    gpu_devices="${gpu_devices// /}"
    tp_size="${tp_size// /}"
    ep_size="${ep_size// /}"
    dp_size="${dp_size// /}"
    tasks="${tasks// /}"

    model_short="$(extract_model_short_name "$model_name")"
    kv_dtype_lower="${kv_dtype,,}"
    case "$mode" in
        BASE)
            if [[ "$kv_dtype" == "FP8" ]]; then
                rot_suffix="baseline_fp8_kv_bf16"
            else
                rot_suffix="baseline_${kv_dtype_lower}"
            fi
            ;;
        Rotation)
            rot_suffix="rotation_${kv_dtype_lower}_${rotate_k}_${rotate_v}_${hadamard_order}"
            ;;
        Rotation_QR)
            rot_suffix="rotation_qr_${kv_dtype_lower}_${rotate_k}_${rotate_v}_${hadamard_order}"
            ;;
        *)
            echo "ERROR: unknown mode '$mode'"; exit 1 ;;
    esac

    server_port=$((BASE_PORT + i))
    label="${model_short}_${rot_suffix} (port=$server_port, gpu=$gpu_devices)"
    CONFIG_LABELS[$i]="$label"
    EXIT_CODES[$i]=-1

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$((i+1))/${N}] Waiting for GPU(s) [$gpu_devices]: $label"
    wait_for_gpus_free "$gpu_devices" "$label"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching: $label"
    eval_single_model "$mode" "$rotate_k" "$rotate_v" "$hadamard_order" "$kv_dtype" \
                      "$model_name" "$q_rotation_path" \
                      "$tp_size" "$ep_size" "$dp_size" "$gpu_devices" \
                      "$tasks" "$server_port" &
    PIDS[$i]=$!

    # If next config shares any GPU, wait for current job to finish first
    next=$((i + 1))
    if [ "$next" -lt "$N" ]; then
        # eval_gpus is field 8 in the config format
        next_gpu=$(echo "${MODEL_CONFIGS[$next]}" | cut -d'|' -f8 | tr -d ' ')
        overlap=0
        IFS=',' read -ra CUR_GPUS <<< "$gpu_devices"
        IFS=',' read -ra NXT_GPUS <<< "$next_gpu"
        for cg in "${CUR_GPUS[@]}"; do
            for ng in "${NXT_GPUS[@]}"; do
                if [ "$cg" = "$ng" ]; then overlap=1; break 2; fi
            done
        done
        if [ "$overlap" -eq 1 ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Next config overlaps GPU(s) [$next_gpu], waiting for current job to finish..."
            wait "${PIDS[$i]}"
            EXIT_CODES[$i]=$?
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Cooling down 60s for GPU memory to release..."
            sleep 60
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] No GPU overlap with next config, sleeping 60s before launching next..."
            sleep 60
        fi
    fi
done

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All configs launched, waiting for completion..."
echo ""

for i in "${!PIDS[@]}"; do
    if [ "${EXIT_CODES[$i]}" -eq -1 ]; then
        wait "${PIDS[$i]}"
        EXIT_CODES[$i]=$?
    fi
    if [ "${EXIT_CODES[$i]}" -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ ${CONFIG_LABELS[$i]}"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ ${CONFIG_LABELS[$i]}"
        OVERALL_EXIT=1
    fi
done

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All done. Exit: $OVERALL_EXIT"
exit $OVERALL_EXIT
