#!/bin/bash
# eval_quant.sh — SGLang evaluation with optional K-means centroid pipeline
#
# Extends seq_eval_kv_rotation.sh with a new KMEANS mode:
#   BASE  — baseline, no rotation, no kmeans  (Stage 3 only)
#   QUANT — Hadamard rotation, no kmeans       (Stage 3 only)
#   KMEANS — Hadamard rotation + K-means       (Stage 1 → Stage 2 → Stage 3)
#
# Stage 1: Dump KV cache  (BF16 server + lm_eval, single GPU, TP=EP=DP=1, idempotent)
# Stage 2: K-means        (compute per-layer centroids, single GPU, idempotent)
# Stage 3: Eval           (INT4 server + tore_eval tasks)
#
# Execution logic (same as seq_eval_kv_rotation.sh):
#   Configs are processed in order. For each config:
#     1. Poll nvidia-smi until all required GPUs are free.
#     2. Launch the pipeline in the background.
#     3. If the next config shares any GPU, wait for the current job to finish,
#        sleep 60s for GPU memory to release, then continue.
#        Otherwise move on immediately (parallel on non-overlapping GPUs).

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
# Format: "mode|hadamard|rotate_v|hadamard_order|kv_dtype|model_name|num_layers|n_clusters|dump_gpus|dump_tp|dump_ep|dump_dp|kmeans_gpu|eval_gpus|eval_tp|eval_ep|eval_dp|tasks"
#
#   mode           : BASE, QUANT, or KMEANS
#                    BASE  — no rotation, no kmeans; kv_dtype still applies
#                    QUANT — Hadamard rotation, no kmeans (like seq_eval_kv_rotation.sh)
#                    KMEANS — Hadamard rotation + K-means centroids
#   hadamard       : 0 or 1 (ignored for BASE)
#   rotate_v       : 0 or 1 (ignored for BASE)
#   hadamard_order : e.g. 16, 64, 128 (ignored for BASE)
#   kv_dtype       : BF16, INT4, or FP8 (FP8 maps to --kv-cache-dtype fp8_e4m3)
#   model_name     : full HuggingFace model ID
#   num_layers     : number of transformer layers (used in KMEANS for dump verification)
#   n_clusters     : K-means cluster count (KMEANS only; set 0 for BASE/QUANT)
#   dump_gpus      : comma-separated GPU IDs for Stage 1 BF16 dump server (KMEANS only; set 0 for others)
#   dump_tp        : tensor parallel size for dump server
#   dump_ep        : expert parallel size for dump server
#   dump_dp        : data parallel size for dump server
#   kmeans_gpu     : single GPU ID for Stage 2 K-means (KMEANS only; set 0 for others)
#   eval_gpus      : comma-separated GPU IDs for eval server
#                    (also used for GPU-overlap detection between configs)
#   eval_tp        : tensor parallel size for eval server
#   eval_ep        : expert parallel size for eval server
#   eval_dp        : data parallel size for eval server
#   tasks          : comma-separated tore_eval preset names with optional :N repeat
#
# Available preset names:
#   gpqa_think, humaneval_think, customized_livecodebench_think, aime25_think, math_500_think
#
TASKS_ALL="math_500_think:5,aime25_think:5,gpqa_think:5,humaneval_think:5,customized_livecodebench_think:5"
TASKS_ONCE="math_500_think:1,aime25_think:1,gpqa_think:1,humaneval_think:1,customized_livecodebench_think:1"
TASKS_GLM_ALL="GLM_math_500_think:3,GLM_aime25_think:3,GLM_gpqa_think:3,GLM_humaneval_think:3,GLM_customized_livecodebench_think:3"
TASKS_GLM_ONCE="GLM_math_500_think:1,GLM_aime25_think:1,GLM_gpqa_think:1,GLM_humaneval_think:1,GLM_customized_livecodebench_think:1"

# =============================================================================
# Model → num_layers lookup table
# Add new models here. Script will error if a MODEL_CONFIGS entry is not found.
# =============================================================================
declare -A MODEL_NUM_LAYERS=(
    ["Qwen/Qwen3-4B-Thinking-2507"]=36
    ["Qwen/Qwen3-8B"]=36
    ["zai-org/GLM-4.7-FP8"]=92
)

MODEL_CONFIGS=(
    # mode  |h|rv|ho |dtype|model                   |clusters|dump_gpus|dtp|dep|ddp|kmeans_gpu     |eval_gpus|etp|eep|edp|tasks
    # "BASE  |0|0|0  |BF16|Qwen/Qwen3-4B-Thinking-2507|0 |0        |1  |1  |1  |0   |0                |1  |1  |1  |${TASKS_ALL}"
    # "BASE  |0|0|0  |INT4|Qwen/Qwen3-4B-Thinking-2507|0 |0        |1  |1  |1  |0   |1                |1  |1  |1  |${TASKS_ONCE}"
    # "QUANT |1|0|16 |INT4|Qwen/Qwen3-4B-Thinking-2507|0 |0        |1  |1  |1  |0   |2                |1  |1  |1  |${TASKS_ALL}"
    # "QUANT |1|1|16 |INT4|Qwen/Qwen3-4B-Thinking-2507|0 |0        |1  |1  |1  |0   |3                |1  |1  |1  |${TASKS_ALL}"
    # Round 1 — 4 GPU groups in parallel (hadamard_order=16)
    # "KMEANS|1|0|16 |INT4|Qwen/Qwen3-4B-Thinking-2507|1      |0,1,2,3  |4  |1  |1  |0,1,2,3,4,5,6,7|0,1      |2  |1  |1  |${TASKS_ALL}"
    # "KMEANS|1|0|16 |INT4|Qwen/Qwen3-4B-Thinking-2507|16      |0,1,2,3  |4  |1  |1  |0,1,2,3,4,5,6,7|2,3      |2  |1  |1  |${TASKS_ALL}"
    # "KMEANS|1|0|16 |INT4|Qwen/Qwen3-4B-Thinking-2507|256      |0,1,2,3  |4  |1  |1  |0,1,2,3,4,5,6,7|4,5      |2  |1  |1  |${TASKS_ALL}"
    # "KMEANS|1|0|16 |INT4|Qwen/Qwen3-4B-Thinking-2507|2048      |0,1,2,3  |4  |1  |1  |0,1,2,3,4,5,6,7|6,7      |2  |1  |1  |${TASKS_ONCE}"
    # Round 2 — hadamard_order=64
    # "KMEANS|1|0|64 |INT4|Qwen/Qwen3-4B-Thinking-2507|1      |0,1,2,3  |4  |1  |1  |0,1,2,3,4,5,6,7|0,1      |2  |1  |1  |${TASKS_ALL}"
    # "KMEANS|1|0|64 |INT4|Qwen/Qwen3-4B-Thinking-2507|16      |0,1,2,3  |4  |1  |1  |0,1,2,3,4,5,6,7|2,3      |2  |1  |1  |${TASKS_ALL}"
    # "KMEANS|1|0|64 |INT4|Qwen/Qwen3-4B-Thinking-2507|256      |0,1,2,3  |4  |1  |1  |0,1,2,3,4,5,6,7|4,5      |2  |1  |1  |${TASKS_ALL}"
    # "KMEANS|1|0|64 |INT4|Qwen/Qwen3-4B-Thinking-2507|2048      |0,1,2,3  |4  |1  |1  |0,1,2,3,4,5,6,7|6,7      |2  |1  |1  |${TASKS_ONCE}"
    # Round 3 — hadamard_order=128
    # "KMEANS|1|0|128 |INT4|Qwen/Qwen3-4B-Thinking-2507|1      |0,1,2,3  |4  |1  |1  |0,1,2,3,4,5,6,7|0,1      |2  |1  |1  |${TASKS_ALL}"
    # "KMEANS|1|0|128 |INT4|Qwen/Qwen3-4B-Thinking-2507|16      |0,1,2,3  |4  |1  |1  |0,1,2,3,4,5,6,7|2,3      |2  |1  |1  |${TASKS_ALL}"
    # "KMEANS|1|0|128 |INT4|Qwen/Qwen3-4B-Thinking-2507|256      |0,1,2,3  |4  |1  |1  |0,1,2,3,4,5,6,7|4,5      |2  |1  |1  |${TASKS_ALL}"
    # "KMEANS|1|0|128 |INT4|Qwen/Qwen3-4B-Thinking-2507|2048      |0,1,2,3  |4  |1  |1  |0,1,2,3,4,5,6,7|6,7      |2  |1  |1  |${TASKS_ONCE}"
    # kmeans c=2048 gpqa rerun
    # "KMEANS|0|0|16 |INT4|Qwen/Qwen3-4B-Thinking-2507|2048 |0,1,2,3  |4  |1  |1  |0,1,2,3,4,5,6,7|2,3  |2  |1  |1  |gpqa_think:5:2-3"
    # "KMEANS|0|0|16 |INT4|Qwen/Qwen3-4B-Thinking-2507|2048 |0,1,2,3  |4  |1  |1  |0,1,2,3,4,5,6,7|4,5  |2  |1  |1  |gpqa_think:5:4-4"
    # "KMEANS|0|0|16 |INT4|Qwen/Qwen3-4B-Thinking-2507|2048 |0,1,2,3  |4  |1  |1  |0,1,2,3,4,5,6,7|6,7  |2  |1  |1  |gpqa_think:5:5-5"
    # R3 (ho=128) rerun
    # "KMEANS|1|0|128 |INT4|Qwen/Qwen3-4B-Thinking-2507|2048 |0,1,2,3  |4  |1  |1  |0,1,2,3,4,5,6,7|0,1  |2  |1  |1  |math_500_think:5,aime25_think:5"
    # "KMEANS|1|0|128 |INT4|Qwen/Qwen3-4B-Thinking-2507|2048 |0,1,2,3  |4  |1  |1  |0,1,2,3,4,5,6,7|2,3  |2  |1  |1  |gpqa_think:5,humaneval_think:5"
    # "KMEANS|1|0|128 |INT4|Qwen/Qwen3-4B-Thinking-2507|256  |0,1,2,3  |4  |1  |1  |0,1,2,3,4,5,6,7|4,5  |2  |1  |1  |humaneval_think:5,customized_livecodebench_think:5"
    # "KMEANS|1|0|128 |INT4|Qwen/Qwen3-4B-Thinking-2507|16   |0,1,2,3  |4  |1  |1  |0,1,2,3,4,5,6,7|6,7  |2  |1  |1  |customized_livecodebench_think:5"
    # # R3 (ho=128) rerun — queue
    # "KMEANS|1|0|128 |INT4|Qwen/Qwen3-4B-Thinking-2507|1    |0,1,2,3  |4  |1  |1  |0,1,2,3,4,5,6,7|0,1  |2  |1  |1  |customized_livecodebench_think:5"
    # "KMEANS|1|0|128 |INT4|Qwen/Qwen3-4B-Thinking-2507|2048 |0,1,2,3  |4  |1  |1  |0,1,2,3,4,5,6,7|6,7  |2  |1  |1  |customized_livecodebench_think:5"

    # ==========================================================================
    # GLM-4.7-FP8  (TP=8, all 8 GPUs; all configs sequential)
    # ==========================================================================
    # mode  |h|rv|ho |dtype|model              |clusters|dump_gpus        |dtp|dep|ddp|kmeans_gpu|eval_gpus        |etp|eep|edp|tasks
    # Baseline
    "BASE  |0|0|0  |FP8 |zai-org/GLM-4.7-FP8|0    |0                 |1  |1  |1  |0          |0,1,2,3,4,5,6,7  |8  |1  |1  |${TASKS_GLM_ALL}"
    # "BASE  |0|0|0  |INT4|zai-org/GLM-4.7-FP8|0    |0                 |1  |1  |1  |0          |0,1,2,3,4,5,6,7  |8  |1  |1  |${TASKS_GLM_ALL}"
    # Pure rotation — no K-means (QUANT mode)
    # "QUANT |1|0|16 |INT4|zai-org/GLM-4.7-FP8|0    |0                 |1  |1  |1  |0          |0,1,2,3,4,5,6,7  |8  |1  |1  |${TASKS_GLM_ALL}"
    # "QUANT |1|1|16 |INT4|zai-org/GLM-4.7-FP8|0    |0                 |1  |1  |1  |0          |0,1,2,3,4,5,6,7  |8  |1  |1  |${TASKS_GLM_ALL}"
    # "QUANT |1|0|64 |INT4|zai-org/GLM-4.7-FP8|0    |0                 |1  |1  |1  |0          |0,1,2,3,4,5,6,7  |8  |1  |1  |${TASKS_GLM_ALL}"
    # "QUANT |1|1|64 |INT4|zai-org/GLM-4.7-FP8|0    |0                 |1  |1  |1  |0          |0,1,2,3,4,5,6,7  |8  |1  |1  |${TASKS_GLM_ALL}"
    # "QUANT |1|0|128|INT4|zai-org/GLM-4.7-FP8|0    |0                 |1  |1  |1  |0          |0,1,2,3,4,5,6,7  |8  |1  |1  |${TASKS_GLM_ALL}"
    # "QUANT |1|1|128|INT4|zai-org/GLM-4.7-FP8|0    |0                 |1  |1  |1  |0          |0,1,2,3,4,5,6,7  |8  |1  |1  |${TASKS_GLM_ALL}"
)

# =============================================================================
# Server & Eval Config
# =============================================================================
BASE_PORT=30100
NUM_WORKERS=64

# Stage 1 (KMEANS only): lm_eval task(s) to trigger KV cache dump
DUMP_LM_EVAL_TASKS="mmlu_pro"
DUMP_LM_EVAL_LIMIT=16   # per subtask; mmlu_pro has 14 subtasks → 16×14≈224 total
DUMP_TOKENS=20000

# Base directory for KV dump files and centroids
KV_DUMP_BASE="${KV_DUMP_BASE:-/data/$USER/kv-cache}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TORE_EVAL_DIR="$SCRIPT_DIR/tore-eval"
RESULTS_DIR="$SCRIPT_DIR/eval_results"
LOGS_DIR="$SCRIPT_DIR/eval_logs"

export HF_HOME=/data/shared/huggingface
# Load HF token for private dataset access
if [ -z "$HUGGING_FACE_HUB_TOKEN" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
    export HUGGING_FACE_HUB_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi
export HF_TOKEN="$HUGGING_FACE_HUB_TOKEN"

CONDA_BASE="/data/$USER/miniconda"
CONDA_ENV_NAME="sglang_env"
CONDA_ENV_DIR="$CONDA_BASE/envs/$CONDA_ENV_NAME"
PYTHON="$CONDA_ENV_DIR/bin/python3"
LM_EVAL="$CONDA_ENV_DIR/bin/lm_eval"

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
# Stage 1: Dump KV Cache  (KMEANS mode only)
# =============================================================================

run_stage1_dump() {
    local model_name="$1" num_layers="$2" dump_gpus="$3" dump_tp="$4" dump_ep="$5" dump_dp="$6" dump_port="$7"
    local model_short dump_dir
    model_short="$(extract_model_short_name "$model_name")"
    # Path: {KV_DUMP_BASE}/{model_short}/{task}-{dump_tokens}-tokens/
    dump_dir="${KV_DUMP_BASE}/${model_short}/${DUMP_LM_EVAL_TASKS}-${DUMP_TOKENS}-tokens"
    mkdir -p "$dump_dir" "$LOGS_DIR/dump/${model_short}"

    log_message "--- Stage 1: Dump KV Cache ---"
    log_message "  GPUs: $dump_gpus (TP=$dump_tp EP=$dump_ep DP=$dump_dp, port=$dump_port)"
    log_message "  Dump dir: $dump_dir  tokens: $DUMP_TOKENS  tasks: $DUMP_LM_EVAL_TASKS"

    # Skip if all layer files already present
    local all_done=1
    for layer_id in $(seq 0 $((num_layers - 1))); do
        if [ ! -f "$dump_dir/kv_calibration_layer_${layer_id}.pt" ]; then
            all_done=0; break
        fi
    done
    if [ "$all_done" -eq 1 ]; then
        log_message "✓ Stage 1 skipped: all $num_layers layer files already exist"
        return 0
    fi

    local server_log
    server_log=$(unique_log_path "$LOGS_DIR/dump/${model_short}/stage1_server.log")

    DUMP_KVCACHE=true \
    DUMP_KVCACHE_TOKENS=$DUMP_TOKENS \
    DUMP_KVCACHE_DIR="$dump_dir" \
    CUDA_VISIBLE_DEVICES=$dump_gpus \
    PATH="$(dirname "$PYTHON"):$PATH" \
    LIBRARY_PATH="/usr/local/cuda/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$PYTHON" -m sglang.launch_server \
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
        --tensor-parallel-size "$dump_tp" \
        --expert-parallel-size "$dump_ep" \
        --data-parallel-size "$dump_dp" \
        --host 0.0.0.0 \
        --port "$dump_port" \
        --trust-remote-code \
        --disable-cuda-graph \
        > "$server_log" 2>&1 &
    local server_pid=$!
    log_message "Dump server started (PID: $server_pid)"

    if ! wait_for_server "$dump_port" "$server_pid" "dump server"; then
        tail -50 "$server_log" | tee -a "$BATCH_LOG_FILE"
        stop_server "$server_pid" "dump server"
        return 1
    fi

    local lm_eval_log
    lm_eval_log=$(unique_log_path "$LOGS_DIR/dump/${model_short}/stage1_${DUMP_LM_EVAL_TASKS}_${DUMP_TOKENS}_lm_eval.log")
    log_message "Running lm_eval ($DUMP_LM_EVAL_TASKS)..."
    set +e
    CUDA_VISIBLE_DEVICES="" \
    "$LM_EVAL" \
        --model local-completions \
        --tasks "$DUMP_LM_EVAL_TASKS" \
        --limit "$DUMP_LM_EVAL_LIMIT" \
        --model_args "model=${model_name},base_url=http://localhost:${dump_port}/v1/completions,max_model_len=20000,num_concurrent=32,max_retries=1,tokenized_requests=False" \
        2>&1 | tee "$lm_eval_log"
    local lm_exit=${PIPESTATUS[0]}
    set -e
    [ $lm_exit -ne 0 ] && log_message "WARNING: lm_eval exited $lm_exit (may still have enough tokens)"

    stop_server "$server_pid" "dump server"

    local missing=0
    for layer_id in $(seq 0 $((num_layers - 1))); do
        [ ! -f "$dump_dir/kv_calibration_layer_${layer_id}.pt" ] && missing=$((missing + 1))
    done
    if [ $missing -gt 0 ]; then
        log_message "✗ Stage 1 failed: $missing/$num_layers layer files missing"
        return 1
    fi
    log_message "✓ Stage 1 done: $num_layers layers -> $dump_dir"
}

# =============================================================================
# Stage 2: K-means  (KMEANS mode only)
# =============================================================================

run_stage2_kmeans() {
    local model_name="$1" num_layers="$2" n_clusters="$3" kmeans_gpu="$4"
    local model_short dump_dir centroids_dir
    model_short="$(extract_model_short_name "$model_name")"
    dump_dir="${KV_DUMP_BASE}/${model_short}/${DUMP_LM_EVAL_TASKS}-${DUMP_TOKENS}-tokens"
    centroids_dir="${dump_dir}/c_${n_clusters}"
    mkdir -p "$centroids_dir" "$LOGS_DIR/dump/${model_short}"

    log_message "--- Stage 2: K-means (n_clusters=$n_clusters, GPU=$kmeans_gpu) ---"
    log_message "  Dump dir: $dump_dir  ->  $centroids_dir"

    # Skip if all centroid files already present
    local all_done=1
    for layer_id in $(seq 0 $((num_layers - 1))); do
        if [ ! -f "$centroids_dir/k_layer_${layer_id}_clusters_${n_clusters}_centers.pt" ] || \
           [ ! -f "$centroids_dir/v_layer_${layer_id}_clusters_${n_clusters}_centers.pt" ]; then
            all_done=0; break
        fi
    done
    if [ "$all_done" -eq 1 ]; then
        log_message "✓ Stage 2 skipped: all centroid files already exist"
        return 0
    fi

    local kmeans_log
    kmeans_log=$(unique_log_path "$LOGS_DIR/dump/${model_short}/stage2_c${n_clusters}.log")
    log_message "Running K-means... (log: $kmeans_log)"

    CUDA_VISIBLE_DEVICES=$kmeans_gpu \
    "$PYTHON" - <<PYEOF 2>&1 | tee "$kmeans_log"
import torch, os
from flash_kmeans import batch_kmeans_Euclid

num_layers = ${num_layers}
n_clusters = ${n_clusters}
dump_dir   = "${dump_dir}"
save_dir   = "${centroids_dir}"
os.makedirs(save_dir, exist_ok=True)

for layer_id in range(num_layers):
    print(f"Layer {layer_id}/{num_layers} ...", flush=True)
    data = torch.load(os.path.join(dump_dir, f"kv_calibration_layer_{layer_id}.pt"), map_location="cpu")
    T, H, D = data["k"].shape

    k = data["k"].flatten().view(T, -1)[None, ...].to("cuda:0")
    _, k_centers, k_iters = batch_kmeans_Euclid(k, n_clusters=n_clusters, max_iters=200, tol=1e-4, verbose=False, use_heuristic=False)
    k_centers = k_centers.squeeze(0)
    torch.save(k_centers, os.path.join(save_dir, f"k_layer_{layer_id}_clusters_{n_clusters}_centers.pt"))
    print(f"  k: shape={k_centers.shape}, iters={k_iters}", flush=True)

    v = data["v"].flatten().view(T, -1)[None, ...].to("cuda:0")
    _, v_centers, v_iters = batch_kmeans_Euclid(v, n_clusters=n_clusters, max_iters=200, tol=1e-4, verbose=False, use_heuristic=False)
    v_centers = v_centers.squeeze(0)
    torch.save(v_centers, os.path.join(save_dir, f"v_layer_{layer_id}_clusters_{n_clusters}_centers.pt"))
    print(f"  v: shape={v_centers.shape}, iters={v_iters}", flush=True)

print("Done.", flush=True)
PYEOF

    log_message "✓ Stage 2 done: centroids -> $centroids_dir"
}

# =============================================================================
# eval_single_model  (Stage 3 for all modes)
# =============================================================================

eval_single_model() {
    local mode="$1" hadamard="$2" rotate_v="$3" hadamard_order="$4" kv_dtype="$5" \
          model_name="$6" num_layers="$7" n_clusters="$8" \
          dump_gpus="$9" dump_tp="${10}" dump_ep="${11}" dump_dp="${12}" \
          kmeans_gpu="${13}" \
          tp_size="${14}" ep_size="${15}" dp_size="${16}" gpu_devices="${17}" \
          tasks="${18}" server_port="${19}" dump_port="${20}"

    local model_short
    model_short="$(extract_model_short_name "$model_name")"

    # Validate mode
    if [[ "$mode" != "BASE" && "$mode" != "QUANT" && "$mode" != "KMEANS" ]]; then
        echo "ERROR: mode must be BASE, QUANT, or KMEANS, got: '$mode'"
        return 1
    fi

    # BASE: force no rotation, no kmeans
    if [[ "$mode" == "BASE" ]]; then
        hadamard=0
        rotate_v=0
    fi

    # Validate kv_dtype
    if [[ "$kv_dtype" != "BF16" && "$kv_dtype" != "INT4" && "$kv_dtype" != "FP8" ]]; then
        echo "ERROR: kv_dtype must be BF16, INT4, or FP8, got: '$kv_dtype'"
        return 1
    fi
    # BASE mode always uses auto (BF16) KV cache regardless of kv_dtype.
    # kv_dtype for BASE only describes the model weight format (e.g. GLM-4.7-FP8).
    # For QUANT/KMEANS: BF16/FP8 → auto, INT4 → int4.
    local kv_cache_dtype
    if [[ "$mode" == "BASE" ]]; then
        kv_cache_dtype="auto"
    else
        case "$kv_dtype" in
            BF16|FP8) kv_cache_dtype="auto" ;;
            INT4)     kv_cache_dtype="int4" ;;
        esac
    fi

    local kv_dtype_lower="${kv_dtype,,}"
    local rot_suffix
    if [[ "$mode" == "BASE" ]]; then
        if [[ "$kv_dtype" == "FP8" ]]; then
            rot_suffix="baseline_fp8_kv_bf16"
        else
            rot_suffix="baseline_${kv_dtype_lower}"
        fi
    elif [[ "$mode" == "QUANT" ]]; then
        rot_suffix="quant_${kv_dtype_lower}_${hadamard}_${rotate_v}_${hadamard_order}"
    else
        if [[ "$hadamard" == "0" && "$rotate_v" == "0" ]]; then
            rot_suffix="kmeans_${n_clusters}"
        else
            rot_suffix="kmeans_quant_${n_clusters}_${kv_dtype_lower}_${hadamard}_${rotate_v}_${hadamard_order}"
        fi
    fi

    mkdir -p "$LOGS_DIR/batch_logs/${model_short}" "$LOGS_DIR/inference_logs/${model_short}"
    BATCH_LOG_FILE=$(unique_log_path "$LOGS_DIR/batch_logs/${model_short}/${rot_suffix}.log")

    log_message "=========================================="
    log_message "Mode:      $mode"
    log_message "Model:     $model_name"
    log_message "TP/EP/DP:  $tp_size/$ep_size/$dp_size"
    log_message "GPUs:      $gpu_devices"
    log_message "KV dtype:  $kv_dtype"
    log_message "HADAMARD=$hadamard  ROTATE_V=$rotate_v  HADAMARD_ORDER=$hadamard_order"
    [[ "$mode" == "KMEANS" ]] && log_message "N_CLUSTERS=$n_clusters  dump_gpus=$dump_gpus (TP=$dump_tp EP=$dump_ep DP=$dump_dp)  kmeans_gpu=$kmeans_gpu"
    log_message "Tasks:     $tasks"
    log_message "=========================================="

    # ------------------------------------------------------------------
    # Stage 1 & 2: KMEANS mode only
    # Check if centroids already exist (num_layers * 2 .pt files).
    # If not, delegate to dump_centroids.sh (idempotent) instead of
    # running Stage 1/2 inline — avoids duplicate server starts.
    # ------------------------------------------------------------------
    if [[ "$mode" == "KMEANS" ]]; then
        local _cdir="${KV_DUMP_BASE}/$(extract_model_short_name "$model_name")/${DUMP_LM_EVAL_TASKS}-${DUMP_TOKENS}-tokens/c_${n_clusters}"
        local _expected=$((num_layers * 2))
        local _actual
        _actual=$(find "${_cdir}" -name "*.pt" 2>/dev/null | wc -l)

        if [ "$_actual" -ge "$_expected" ]; then
            log_message "✓ Centroids already complete ($_actual/${_expected} files): $_cdir"
        else
            log_message "Centroids incomplete ($_actual/${_expected}), running Stage 1 & 2..."
            run_stage1_dump "$model_name" "$num_layers" "$dump_gpus" "$dump_tp" "$dump_ep" "$dump_dp" "$dump_port"
            run_stage2_kmeans "$model_name" "$num_layers" "$n_clusters" "$kmeans_gpu"
        fi
    fi

    # ------------------------------------------------------------------
    # Stage 3: Start SGLang server
    # ------------------------------------------------------------------
    # Unset dump env vars to avoid accidentally re-dumping KV cache during eval
    unset DUMP_KVCACHE
    unset DUMP_KVCACHE_TOKENS
    unset DUMP_KVCACHE_DIR
    log_message "Starting SGLang server on port $server_port..."
    local server_log
    SERVER_LOG=$(unique_log_path "$LOGS_DIR/inference_logs/${model_short}/${rot_suffix}_server.log")

    local centroids_path=""
    if [[ "$mode" == "KMEANS" ]]; then
        centroids_path="${KV_DUMP_BASE}/$(extract_model_short_name "$model_name")/${DUMP_LM_EVAL_TASKS}-${DUMP_TOKENS}-tokens/c_${n_clusters}"
    fi

    # Determine mem-fraction-static: BF16 KV cache uses 2x GPU memory vs FP8,
    # leaving insufficient room (~15 GB) for CUDA graph capture on large models.
    # Use 0.75 for BF16 KV cache to free ~4 GB more per GPU for CUDA graphs.
    local mem_fraction
    if [[ "$kv_cache_dtype" == "auto" ]]; then
        mem_fraction="0.65"
    else
        mem_fraction="0.8"
    fi

    HADAMARD=$hadamard \
    ROTATE_V=$rotate_v \
    HADAMARD_ORDER=$hadamard_order \
    N_CLUSTERS=$n_clusters \
    SGLANG_KV_CENTROIDS_PATH="$centroids_path" \
    CUDA_VISIBLE_DEVICES=$gpu_devices \
    PATH="$(dirname "$PYTHON"):$PATH" \
    LIBRARY_PATH="/usr/local/cuda/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$PYTHON" -m sglang.launch_server \
        --model-path "$model_name" \
        --max-running-requests 32 \
        --max-queued-requests 256 \
        --page-size 128 \
        --chunked-prefill-size 4096 \
        --mem-fraction-static "$mem_fraction" \
        --pp-max-micro-batch-size 32 \
        --kv-cache-dtype "$kv_cache_dtype" \
        --prefill-attention-backend fa3 \
        --decode-attention-backend triton \
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
            # Skip if this run already has results
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
missing = [i for i in range(1, 6)
           if not os.path.exists(os.path.join(task_dir, f"run{i}", "results.jsonl"))]
if missing:
    print(f"Skipping aggregation: run(s) {missing} missing results.jsonl — will aggregate when all 5 runs complete")
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
# GPU free check
# =============================================================================

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
    echo "ERROR: conda env '$CONDA_ENV_NAME' not found (expected Python at $PYTHON)."
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

# 3. datasets
bash "$SCRIPT_DIR/prepare_datasets.sh" "$PYTHON" "$SCRIPT_DIR"

# =============================================================================
# Main — sequential scheduling with GPU-aware overlap detection
# =============================================================================

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] =========================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] KV-cache K-means + Rotation Evaluation"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Configs: ${#MODEL_CONFIGS[@]} entry(s)"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU free threshold: ${GPU_FREE_MEM_MB} MB"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] KV dump base: ${KV_DUMP_BASE}"
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
    IFS='|' read -r mode hadamard rotate_v hadamard_order kv_dtype model_name \
                    n_clusters \
                    dump_gpus dump_tp dump_ep dump_dp \
                    kmeans_gpu \
                    gpu_devices tp_size ep_size dp_size tasks <<< "$config"
    mode="${mode// /}"
    hadamard="${hadamard// /}"
    rotate_v="${rotate_v// /}"
    hadamard_order="${hadamard_order// /}"
    kv_dtype="${kv_dtype// /}"
    model_name="${model_name// /}"
    n_clusters="${n_clusters// /}"

    # Lookup num_layers from table; error if model not registered
    num_layers="${MODEL_NUM_LAYERS[$model_name]}"
    if [[ -z "$num_layers" ]]; then
        echo "ERROR: model '$model_name' not found in MODEL_NUM_LAYERS table."
        echo "       Add it at the top of this script and retry."
        exit 1
    fi
    dump_gpus="${dump_gpus// /}"
    dump_tp="${dump_tp// /}"
    dump_ep="${dump_ep// /}"
    dump_dp="${dump_dp// /}"
    kmeans_gpu="${kmeans_gpu// /}"
    gpu_devices="${gpu_devices// /}"
    tp_size="${tp_size// /}"
    ep_size="${ep_size// /}"
    dp_size="${dp_size// /}"
    tasks="${tasks// /}"

    model_short="$(extract_model_short_name "$model_name")"
    kv_dtype_lower="${kv_dtype,,}"
    if [[ "$mode" == "BASE" ]]; then
        if [[ "$kv_dtype" == "FP8" ]]; then
            rot_suffix="baseline_fp8_kv_bf16"
        else
            rot_suffix="baseline_${kv_dtype_lower}"
        fi
    elif [[ "$mode" == "QUANT" ]]; then
        rot_suffix="quant_${kv_dtype_lower}_${hadamard}_${rotate_v}_${hadamard_order}"
    else
        if [[ "$hadamard" == "0" && "$rotate_v" == "0" ]]; then
            rot_suffix="kmeans_${n_clusters}"
        else
            rot_suffix="kmeans_quant_${n_clusters}_${kv_dtype_lower}_${hadamard}_${rotate_v}_${hadamard_order}"
        fi
    fi

    server_port=$((BASE_PORT + i))
    dump_port=$((BASE_PORT + 100 + i))
    label="${model_short}_${rot_suffix} (port=$server_port, gpu=$gpu_devices)"
    CONFIG_LABELS[$i]="$label"
    EXIT_CODES[$i]=-1

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$((i+1))/${N}] Waiting for GPU(s) [$gpu_devices]: $label"
    wait_for_gpus_free "$gpu_devices" "$label"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching: $label"
    eval_single_model "$mode" "$hadamard" "$rotate_v" "$hadamard_order" "$kv_dtype" \
                      "$model_name" "$num_layers" "$n_clusters" \
                      "$dump_gpus" "$dump_tp" "$dump_ep" "$dump_dp" \
                      "$kmeans_gpu" \
                      "$tp_size" "$ep_size" "$dp_size" "$gpu_devices" \
                      "$tasks" "$server_port" "$dump_port" &
    PIDS[$i]=$!

    # If next config shares any GPU, wait for current job to finish first
    next=$((i + 1))
    if [ "$next" -lt "$N" ]; then
        next_gpu=$(echo "${MODEL_CONFIGS[$next]}" | cut -d'|' -f13 | tr -d ' ')
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
