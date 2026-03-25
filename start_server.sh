#!/bin/bash
# start_server.sh — Start SGLang server(s) based on MODEL_CONFIGS
#
# Modes:
#   BASE   — baseline, no rotation, no kmeans
#   QUANT  — Hadamard rotation, no kmeans
#   KMEANS — Hadamard rotation + K-means (centroids must already exist)
#
# For each config:
#   1. Poll nvidia-smi until all required GPUs are free.
#   2. Launch the server in the background, logging to kv_rotation/server_logs/.
#   3. If the next config shares any GPU, sleep 60s before continuing.

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
# Format: "mode|hadamard|rotate_v|hadamard_order|kv_dtype|model_name|n_clusters|gpus|tp|ep|dp"
#
#   mode           : BASE, QUANT, or KMEANS
#   hadamard       : 0 or 1 (ignored for BASE)
#   rotate_v       : 0 or 1 (ignored for BASE)
#   hadamard_order : e.g. 16, 64, 128 (ignored for BASE)
#   kv_dtype       : BF16, INT4, or FP8 (FP8 maps to --kv-cache-dtype fp8_e4m3)
#   model_name     : full HuggingFace model ID
#   n_clusters     : K-means cluster count (KMEANS only; set 0 for BASE/QUANT)
#   gpus           : comma-separated GPU IDs for the server
#   tp             : tensor parallel size
#   ep             : expert parallel size
#   dp             : data parallel size

MODEL_CONFIGS=(
    # mode  |h|rv|ho |dtype|model                          |clusters|gpus            |tp|ep|dp
    # "BASE  |0|0|0  |BF16|Qwen/Qwen3-4B-Thinking-2507     |0       |0               |1 |1 |1"
    # "BASE  |0|0|0  |INT4|Qwen/Qwen3-4B-Thinking-2507     |0       |1               |1 |1 |1"
    # "QUANT |1|0|16 |INT4|Qwen/Qwen3-4B-Thinking-2507     |0       |2               |1 |1 |1"
    # "QUANT |1|1|16 |INT4|Qwen/Qwen3-4B-Thinking-2507     |0       |3               |1 |1 |1"
    # "KMEANS|1|0|16 |INT4|Qwen/Qwen3-4B-Thinking-2507     |1       |0,1             |2 |1 |1"
    # "KMEANS|1|0|16 |INT4|Qwen/Qwen3-4B-Thinking-2507     |16      |2,3             |2 |1 |1"
    # "KMEANS|1|0|16 |INT4|Qwen/Qwen3-4B-Thinking-2507     |256     |4,5             |2 |1 |1"
    # "KMEANS|1|0|16 |INT4|Qwen/Qwen3-4B-Thinking-2507     |2048    |6,7             |2 |1 |1"

    # ==========================================================================
    # GLM-4.7-FP8  (TP=8, all 8 GPUs; all configs sequential)
    # ==========================================================================
    # "BASE  |0|0|0  |FP8 |zai-org/GLM-4.7-FP8             |0       |0,1,2,3,4,5,6,7 |8 |1 |1"
    "BASE  |0|0|0  |INT4|zai-org/GLM-4.7-FP8             |0       |0,1,2,3,4,5,6,7 |8 |1 |1"
    # "QUANT |1|0|16 |INT4|zai-org/GLM-4.7-FP8             |0       |0,1,2,3,4,5,6,7 |8 |1 |1"
    # "QUANT |1|1|16 |INT4|zai-org/GLM-4.7-FP8             |0       |0,1,2,3,4,5,6,7 |8 |1 |1"
    # "QUANT |1|0|64 |INT4|zai-org/GLM-4.7-FP8             |0       |0,1,2,3,4,5,6,7 |8 |1 |1"
    # "QUANT |1|0|16 |INT4|zai-org/GLM-4.7-FP8             |0       |0,1,2,3,4,5,6,7 |8 |1 |1"
    # "QUANT |1|0|128|INT4|zai-org/GLM-4.7-FP8             |0       |0,1,2,3,4,5,6,7 |8 |1 |1"
    # "QUANT |1|1|128|INT4|zai-org/GLM-4.7-FP8             |0       |0,1,2,3,4,5,6,7 |8 |1 |1"
)

# =============================================================================
# Server Config
# =============================================================================
BASE_PORT=30100

KV_DUMP_BASE="${KV_DUMP_BASE:-/data/$USER/kv-cache}"
DUMP_LM_EVAL_TASKS="mmlu_pro"
DUMP_TOKENS=20000

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_LOGS_DIR="$SCRIPT_DIR/server_logs"

export HF_HOME=/data/shared/huggingface
if [ -z "$HUGGING_FACE_HUB_TOKEN" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
    export HUGGING_FACE_HUB_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi
export HF_TOKEN="$HUGGING_FACE_HUB_TOKEN"

CONDA_BASE="/data/$USER/miniconda"
CONDA_ENV_NAME="sglang_env"
CONDA_ENV_DIR="$CONDA_BASE/envs/$CONDA_ENV_NAME"
PYTHON="$CONDA_ENV_DIR/bin/python3"

export TRITON_CACHE_DIR="/scratch/jisenli2/.triton/cache"
export TMPDIR="/data/jisenli2/tmp"
mkdir -p "$TMPDIR"

GPU_FREE_MEM_MB="${GPU_FREE_MEM_MB:-500}"
GPU_POLL_INTERVAL="${GPU_POLL_INTERVAL:-240}"

mkdir -p "$SERVER_LOGS_DIR"

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
# start_server — launch SGLang server for a single config
# =============================================================================

start_single_server() {
    local mode="$1" hadamard="$2" rotate_v="$3" hadamard_order="$4" kv_dtype="$5" \
          model_name="$6" n_clusters="$7" \
          gpu_devices="$8" tp_size="$9" ep_size="${10}" dp_size="${11}" \
          server_port="${12}"

    local model_short
    model_short="$(extract_model_short_name "$model_name")"

    if [[ "$mode" != "BASE" && "$mode" != "QUANT" && "$mode" != "KMEANS" ]]; then
        echo "ERROR: mode must be BASE, QUANT, or KMEANS, got: '$mode'"
        return 1
    fi

    if [[ "$mode" == "BASE" ]]; then
        hadamard=0
        rotate_v=0
    fi

    if [[ "$kv_dtype" != "BF16" && "$kv_dtype" != "INT4" && "$kv_dtype" != "FP8" ]]; then
        echo "ERROR: kv_dtype must be BF16, INT4, or FP8, got: '$kv_dtype'"
        return 1
    fi
    local kv_cache_dtype
    case "$kv_dtype" in
        BF16) kv_cache_dtype="auto" ;;
        INT4) kv_cache_dtype="int4" ;;
        *)    kv_cache_dtype="auto" ;;
    esac

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

    mkdir -p "$SERVER_LOGS_DIR/${model_short}"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] =========================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Mode:      $mode"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Model:     $model_name"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] TP/EP/DP:  $tp_size/$ep_size/$dp_size"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPUs:      $gpu_devices"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] KV dtype:  $kv_dtype"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] HADAMARD=$hadamard  ROTATE_V=$rotate_v  HADAMARD_ORDER=$hadamard_order"
    [[ "$mode" == "KMEANS" ]] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] N_CLUSTERS=$n_clusters"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Port:      $server_port"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] =========================================="

    local centroids_path=""
    if [[ "$mode" == "KMEANS" ]]; then
        centroids_path="${KV_DUMP_BASE}/$(extract_model_short_name "$model_name")/${DUMP_LM_EVAL_TASKS}-${DUMP_TOKENS}-tokens/c_${n_clusters}"
    fi

    local server_log
    server_log=$(unique_log_path "$SERVER_LOGS_DIR/${model_short}/${rot_suffix}_server.log")

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting SGLang server on port $server_port..."
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Server log: $server_log"

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
        --mem-fraction-static 0.8 \
        --pp-max-micro-batch-size 32 \
        --kv-cache-dtype "$kv_cache_dtype" \
        --tool-call-parser glm47 \
        --reasoning-parser glm45 \
        --prefill-attention-backend fa3 \
        --decode-attention-backend triton \
        --sampling-backend flashinfer \
        --tensor-parallel-size "$tp_size" \
        --expert-parallel-size "$ep_size" \
        --data-parallel-size "$dp_size" \
        --host 0.0.0.0 \
        --port "$server_port" \
        --trust-remote-code \
        --disable-cuda-graph \
        > "$server_log" 2>&1 &
    local server_pid=$!

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Server started (PID: $server_pid, log: $server_log)"
}

# =============================================================================
# Main
# =============================================================================

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] =========================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] KV-cache Server Launcher"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Configs: ${#MODEL_CONFIGS[@]} entry(s)"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU free threshold: ${GPU_FREE_MEM_MB} MB"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Server logs: ${SERVER_LOGS_DIR}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] =========================================="
echo ""

N=${#MODEL_CONFIGS[@]}

for i in "${!MODEL_CONFIGS[@]}"; do
    config="${MODEL_CONFIGS[$i]}"
    IFS='|' read -r mode hadamard rotate_v hadamard_order kv_dtype model_name \
                    n_clusters \
                    gpu_devices tp_size ep_size dp_size <<< "$config"
    mode="${mode// /}"
    hadamard="${hadamard// /}"
    rotate_v="${rotate_v// /}"
    hadamard_order="${hadamard_order// /}"
    kv_dtype="${kv_dtype// /}"
    model_name="${model_name// /}"
    n_clusters="${n_clusters// /}"
    gpu_devices="${gpu_devices// /}"
    tp_size="${tp_size// /}"
    ep_size="${ep_size// /}"
    dp_size="${dp_size// /}"

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
    label="${model_short}_${rot_suffix} (port=$server_port, gpu=$gpu_devices)"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$((i+1))/${N}] Waiting for GPU(s) [$gpu_devices]: $label"
    wait_for_gpus_free "$gpu_devices" "$label"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching: $label"
    start_single_server "$mode" "$hadamard" "$rotate_v" "$hadamard_order" "$kv_dtype" \
                        "$model_name" "$n_clusters" \
                        "$gpu_devices" "$tp_size" "$ep_size" "$dp_size" \
                        "$server_port"

    # If next config shares any GPU, sleep before proceeding
    next=$((i + 1))
    if [ "$next" -lt "$N" ]; then
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
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Next config overlaps GPU(s) [$next_gpu], sleeping 60s..."
            sleep 60
        fi
    fi
done

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All ${N} server(s) launched."
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Server logs are in: ${SERVER_LOGS_DIR}"
echo ""
