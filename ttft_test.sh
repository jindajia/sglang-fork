#!/bin/bash
# throughput_test.sh — SGLang throughput benchmarking with tore-speed-eval
#
# Measures OTPS / TPS / TTFT across batch sizes and input lengths.
#
# Modes: BASE, QUANT, KMEANS  (same naming as eval_quant.sh)
#   BASE  — no rotation, no kmeans
#   QUANT — Hadamard rotation only
#   KMEANS — Hadamard rotation + K-means centroids
#
# For KMEANS: centroids must already exist locally.
#             Script exits immediately if centroid files are not found.
#             Run dump_centroids.sh first to generate them.
#
# GPU scheduling: configs on non-overlapping GPUs launch in parallel.
# Results: throughput_results/{model_short}/{rot_suffix}/bs{N}_{in_label}.csv

set -eo pipefail

# Redirect TMPDIR away from /tmp (system disk may fill up with other users' torchinductor caches)
export TMPDIR="/data/${USER}/tmp"
mkdir -p "$TMPDIR"

cleanup() {
    trap '' INT TERM   # prevent recursive trap invocation
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Interrupted — killing all child processes..."
    kill -9 -- -$$ 2>/dev/null || true
    exit 130
}
trap cleanup INT TERM

# =============================================================================
# Throughput Test Parameters
# =============================================================================
# BATCH_SIZES=(1 8 16 32)
BATCH_SIZES=(1)
INPUT_LENS=(8192 16384 32768)
OUTPUT_LENS=(10)
NUM_EXAMPLES=32

# =============================================================================
# Model Configs
# =============================================================================
# Format: "mode|hadamard|rotate_v|hadamard_order|kv_dtype|model_name|n_clusters|eval_gpus|eval_tp|eval_ep|eval_dp"
#
#   mode          : BASE, QUANT, or KMEANS
#   hadamard      : 0 or 1 (ignored for BASE)
#   rotate_v      : 0 or 1 (ignored for BASE)
#   hadamard_order: e.g. 16, 64, 128 (ignored for BASE)
#   kv_dtype      : BF16 or INT4
#   model_name    : full HuggingFace model ID
#   n_clusters    : K-means cluster count (KMEANS only; set 0 for BASE/QUANT)
#   eval_gpus     : comma-separated GPU IDs for the server
#   eval_tp       : tensor parallel size
#   eval_ep       : expert parallel size
#   eval_dp       : data parallel size
#
MODEL_CONFIGS=(
    # ---- Qwen/Qwen3-8B  TP=2 ------------------------------------------------
    # "BASE  |0|0|0  |BF16|Qwen/Qwen3-8B|0|0,1|2|1|1"
    # "BASE  |0|0|0  |INT4|Qwen/Qwen3-8B|0|0,1|2|1|1"
    # "QUANT |1|0|128|INT4|Qwen/Qwen3-8B|0|2,3|2|1|1"
    # ---- Qwen/Qwen3-4B-Thinking-2507 ----------------------------------------
    # "BASE  |0|0|0  |BF16|Qwen/Qwen3-4B-Thinking-2507|0|0|1|1|1"
    # "BASE  |0|0|0  |INT4|Qwen/Qwen3-4B-Thinking-2507|0|1|1|1|1"
    # "QUANT |1|0|16 |INT4|Qwen/Qwen3-4B-Thinking-2507|0|2|1|1|1"
    # "QUANT |1|1|16 |INT4|Qwen/Qwen3-4B-Thinking-2507|0|3|1|1|1"
    # "QUANT |1|0|64 |INT4|Qwen/Qwen3-4B-Thinking-2507|0|3|1|1|1"
    # "QUANT |1|1|64 |INT4|Qwen/Qwen3-4B-Thinking-2507|0|5|1|1|1"
    # "QUANT |1|0|128|INT4|Qwen/Qwen3-4B-Thinking-2507|0|4|1|1|1"
    # "QUANT |1|1|128|INT4|Qwen/Qwen3-4B-Thinking-2507|0|7|1|1|1"
    # "QUANT |1|0|512 |INT4|Qwen/Qwen3-4B-Thinking-2507|0|5|1|1|1"
    # "QUANT |1|0|1024 |INT4|Qwen/Qwen3-4B-Thinking-2507|0|6|1|1|1"
    # ---- KMEANS examples (uncomment and fill n_clusters as needed) -----------
    "KMEANS|0|0|0 |INT4|Qwen/Qwen3-4B-Thinking-2507|1|0,1,2,3,4,5,6,7|2|1|4"
    "KMEANS|0|0|0 |INT4|Qwen/Qwen3-4B-Thinking-2507|16|0,1,2,3,4,5,6,7|2|1|4"
    "KMEANS|0|0|0 |INT4|Qwen/Qwen3-4B-Thinking-2507|256|0,1,2,3,4,5,6,7|2|1|4"
    "KMEANS|0|0|0 |INT4|Qwen/Qwen3-4B-Thinking-2507|2048|0,1,2,3,4,5,6,7|2|1|4"

    "KMEANS|0|0|0 |INT4|Qwen/Qwen3-8B|1|0,1,2,3,4,5,6,7|2|1|4"
    "KMEANS|0|0|0 |INT4|Qwen/Qwen3-8B|16|0,1,2,3,4,5,6,7|2|1|4"
    "KMEANS|0|0|0 |INT4|Qwen/Qwen3-8B|256|0,1,2,3,4,5,6,7|2|1|4"
    "KMEANS|0|0|0 |INT4|Qwen/Qwen3-8B|2048|0,1,2,3,4,5,6,7|2|1|4"

    "KMEANS|0|0|0 |INT4|Qwen/Qwen3-32B|1|0,1,2,3,4,5,6,7|2|1|4"
    "KMEANS|0|0|0 |INT4|Qwen/Qwen3-32B|16|0,1,2,3,4,5,6,7|2|1|4"
    "KMEANS|0|0|0 |INT4|Qwen/Qwen3-32B|256|0,1,2,3,4,5,6,7|2|1|4"
    "KMEANS|0|0|0 |INT4|Qwen/Qwen3-32B|2048|0,1,2,3,4,5,6,7|2|1|4"

    "KMEANS|0|0|0 |INT4|zai-org/GLM-4.7-FP8|1|0,1,2,3,4,5,6,7|8|1|1"
    "KMEANS|0|0|0 |INT4|zai-org/GLM-4.7-FP8|16|0,1,2,3,4,5,6,7|8|1|1"
    "KMEANS|0|0|0 |INT4|zai-org/GLM-4.7-FP8|256|0,1,2,3,4,5,6,7|8|1|1"
    "KMEANS|0|0|0 |INT4|zai-org/GLM-4.7-FP8|2048|0,1,2,3,4,5,6,7|8|1|1"
)

# =============================================================================
# Model → num_layers lookup (needed for KMEANS centroid path verification)
# Add new models here before adding them to MODEL_CONFIGS.
# =============================================================================
declare -A MODEL_NUM_LAYERS=(
    ["Qwen/Qwen3-4B-Thinking-2507"]=36
    ["Qwen/Qwen3-8B"]=36
    ["Qwen/Qwen3-32B"]=64
    ["zai-org/GLM-4.7-FP8"]=92
)

# =============================================================================
# Server & Path Config
# =============================================================================
BASE_PORT=30900

# Base directory for KV dump files and centroids (KMEANS mode only)
KV_DUMP_BASE="${KV_DUMP_BASE:-/data/$USER/kv-cache}"
# Task name used when the KV cache was dumped (must match dump_centroids.sh)
DUMP_LM_EVAL_TASKS="mmlu_pro"
DUMP_TOKENS=20000

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TORE_SPEED_EVAL_DIR="$SCRIPT_DIR/tore-speed-eval"
RESULTS_DIR="$SCRIPT_DIR/ttft_results"
LOGS_DIR="$SCRIPT_DIR/ttft_logs"

export HF_HOME=/data/shared/huggingface

CONDA_BASE="/data/$USER/miniconda"
CONDA_ENV_NAME="sglang_env"
CONDA_ENV_DIR="$CONDA_BASE/envs/$CONDA_ENV_NAME"
PYTHON="$CONDA_ENV_DIR/bin/python3"

export TRITON_CACHE_DIR="/dev/shm/triton_cache_$USER"
export FLASHINFER_CACHE_DIR="/data/$USER/.cache/flashinfer"
export SGLANG_DISABLE_FLASHINFER_TRTLLM_AR_FUSION=1

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
        sleep 5 &
        wait $!
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
        # Run sleep in background and wait on it — makes the wait interruptible by signals
        sleep "$GPU_POLL_INTERVAL" &
        wait $!
        waited=$((waited + GPU_POLL_INTERVAL))
    done
    if [ "$waited" -gt 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU(s) [$gpu_list] now free after $((waited / 60))min"
    fi
}

# Convert token count to short label: 8192 → in8k, 16384 → in16k; small values kept as-is: 10 → out10
input_len_label()  { local n=$((${1} / 1024)); [ "$n" -gt 0 ] && echo "in${n}k" || echo "in${1}"; }
output_len_label() { local n=$((${1} / 1024)); [ "$n" -gt 0 ] && echo "out${n}k" || echo "out${1}"; }

# =============================================================================
# extract_per_request_stats
#   Parse SGLang "Finish:" log lines produced during one eval run and append
#   a summary block to per_request_stats.log.
#
#   Args:
#     $1  server_log       — path to the running server log file
#     $2  log_line_before  — line count of server_log before the eval started
#     $3  rot_suffix       — config label (e.g. quant_int4_1_0_16)
#     $4  bs               — batch size used for this run
#     $5  label_in         — input-length label (e.g. in8k)
#     $6  stats_log        — path to the per_request_stats.log file
# =============================================================================
extract_per_request_stats() {
    local server_log="$1" log_line_before="$2" rot_suffix="$3" \
          bs="$4" label_in="$5" stats_log="$6"

    # Grab only the log lines produced during this eval run
    local run_log
    run_log=$(tail -n +"$((log_line_before + 1))" "$server_log" 2>/dev/null || true)

    local finish_count
    finish_count=$(echo "$run_log" | grep -c "Finish:" || true)

    {
        echo ""
        echo "=== [${rot_suffix}] BS=${bs} ${label_in} (${finish_count} Finish lines) ==="
    } | tee -a "$stats_log"

    if [ "$finish_count" -eq 0 ]; then
        echo "  WARNING: no Finish: lines found — --log-requests may not be active" \
            | tee -a "$stats_log"
        return
    fi

    # Parse TPS, OTPS, TTFT from Finish: lines with Python
    # Fields (order not fixed, use regex):
    #   prompt_tokens, completion_tokens, e2e_latency
    #   request_received_ts, prefill_finished_ts (or api_server_dispatch_finish_ts)
    echo "$run_log" \
        | grep "Finish:" \
        | grep -v "HEALTH_CHECK" \
        | "$PYTHON" -c "
import sys, re

tps_vals, otps_vals, ttft_vals = [], [], []

for line in sys.stdin:
    pt   = re.search(r\"'prompt_tokens': (\d+)\", line)
    ct   = re.search(r\"'completion_tokens': (\d+)\", line)
    e2e  = re.search(r\"'e2e_latency': ([\d.]+)\", line)
    recv = re.search(r\"'request_received_ts': ([\d.]+)\", line)
    # prefer prefill_finished_ts; fall back to api_server_dispatch_finish_ts
    pf   = re.search(r\"'prefill_finished_ts': ([\d.]+)\", line) or \
           re.search(r\"'api_server_dispatch_finish_ts': ([\d.]+)\", line)

    if not (pt and ct and e2e):
        continue
    p, c, lat = int(pt.group(1)), int(ct.group(1)), float(e2e.group(1))
    if lat <= 0:
        continue

    tps_vals.append((p + c) / lat)

    if recv and pf:
        ttft = float(pf.group(1)) - float(recv.group(1))
        ttft_vals.append(ttft)
        decode_time = lat - ttft
        if decode_time > 0 and c > 0:
            otps_vals.append(c / decode_time)

def summarize(vals, label, unit):
    if not vals:
        print(f'  {label}: N/A (no data)')
        return
    vals_s = sorted(vals)
    n = len(vals_s)
    mean = sum(vals_s) / n
    p05  = vals_s[max(0, int(n * 0.05))]
    p50  = vals_s[max(0, int(n * 0.50))]
    p95  = vals_s[min(n-1, int(n * 0.95))]
    print(f'  {label} [{unit}]: Mean={mean:.3f}  P50={p50:.3f}  P05={p05:.3f}  P95={p95:.3f}  n={n}')

summarize(tps_vals,  'TPS  (prompt+output / e2e)',    'tok/s')
summarize(otps_vals, 'OTPS (output / decode_time)',   'tok/s')
summarize(ttft_vals, 'TTFT (prefill_finished-recv)',  's')
" 2>&1 | tee -a "$stats_log"
}

# =============================================================================
# KMEANS centroid check — exit immediately if files are missing
# =============================================================================
check_kmeans_centroids() {
    local model_name="$1" num_layers="$2" n_clusters="$3"
    local model_short
    model_short="$(extract_model_short_name "$model_name")"
    local cdir="${KV_DUMP_BASE}/${model_short}/${DUMP_LM_EVAL_TASKS}-${DUMP_TOKENS}-tokens/c_${n_clusters}"
    local expected=$((num_layers * 2))
    local actual
    actual=$(find "${cdir}" -name "*.pt" 2>/dev/null | wc -l)
    if [ "$actual" -lt "$expected" ]; then
        echo "ERROR: KMEANS centroids incomplete ($actual/${expected} .pt files) at:"
        echo "       $cdir"
        echo "       Run dump_centroids.sh first to generate centroids."
        exit 1
    fi
    log_message "✓ KMEANS centroids verified ($actual files): $cdir"
}

# =============================================================================
# benchmark_single_model
# =============================================================================
benchmark_single_model() {
    local mode="$1" hadamard="$2" rotate_v="$3" hadamard_order="$4" kv_dtype="$5" \
          model_name="$6" num_layers="$7" n_clusters="$8" \
          tp_size="$9" ep_size="${10}" dp_size="${11}" gpu_devices="${12}" \
          server_port="${13}"

    local model_short
    model_short="$(extract_model_short_name "$model_name")"

    if [[ "$mode" != "BASE" && "$mode" != "QUANT" && "$mode" != "KMEANS" ]]; then
        echo "ERROR: mode must be BASE, QUANT, or KMEANS, got: '$mode'"
        return 1
    fi

    # BASE: force no rotation
    if [[ "$mode" == "BASE" ]]; then
        hadamard=0
        rotate_v=0
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
        rot_suffix="baseline_${kv_dtype_lower}"
    elif [[ "$mode" == "QUANT" ]]; then
        rot_suffix="quant_${kv_dtype_lower}_${hadamard}_${rotate_v}_${hadamard_order}"
    else
        if [[ "$hadamard" == "0" && "$rotate_v" == "0" ]]; then
            rot_suffix="kmeans_${n_clusters}"
        else
            rot_suffix="kmeans_quant_${n_clusters}_${kv_dtype_lower}_${hadamard}_${rotate_v}_${hadamard_order}"
        fi
    fi

    local result_dir="$RESULTS_DIR/${model_short}/${rot_suffix}"
    local log_dir="$LOGS_DIR/${model_short}"
    mkdir -p "$result_dir" "$log_dir"
    BATCH_LOG_FILE=$(unique_log_path "$log_dir/${rot_suffix}.log")

    log_message "=========================================="
    log_message "Mode:      $mode"
    log_message "Model:     $model_name"
    log_message "TP/EP/DP:  $tp_size/$ep_size/$dp_size"
    log_message "GPUs:      $gpu_devices"
    log_message "KV dtype:  $kv_dtype (cache: $kv_cache_dtype)"
    log_message "HADAMARD=$hadamard  ROTATE_V=$rotate_v  HADAMARD_ORDER=$hadamard_order"
    [[ "$mode" == "KMEANS" ]] && log_message "N_CLUSTERS=$n_clusters"
    log_message "Batch sizes:  ${BATCH_SIZES[*]}"
    log_message "Input lens:   ${INPUT_LENS[*]}"
    log_message "Output lens:  ${OUTPUT_LENS[*]}  num_examples: ${NUM_EXAMPLES}"
    log_message "Results dir:  $result_dir"
    log_message "=========================================="

    # ------------------------------------------------------------------
    # KMEANS: verify centroids exist locally, exit if not
    # ------------------------------------------------------------------
    if [[ "$mode" == "KMEANS" ]]; then
        check_kmeans_centroids "$model_name" "$num_layers" "$n_clusters"
    fi

    # ------------------------------------------------------------------
    # Start SGLang server
    # ------------------------------------------------------------------
    local centroids_path=""
    if [[ "$mode" == "KMEANS" ]]; then
        centroids_path="${KV_DUMP_BASE}/$(extract_model_short_name "$model_name")/${DUMP_LM_EVAL_TASKS}-${DUMP_TOKENS}-tokens/c_${n_clusters}"
    fi

    local mem_fraction="0.8"

    unset DUMP_KVCACHE DUMP_KVCACHE_TOKENS DUMP_KVCACHE_DIR
    log_message "Starting SGLang server on port $server_port..."
    local server_log
    server_log=$(unique_log_path "$log_dir/${rot_suffix}_server.log")

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
        --mem-fraction-static "$mem_fraction" \
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
        --log-requests \
        --log-requests-level 0 \
        --enable-request-time-stats-logging \
        --disable-radix-cache \
        --chunked-prefill-size 32768 \
        --max-prefill-tokens 131072 \
        > "$server_log" 2>&1 &
    local server_pid=$!
    log_message "Server started (PID: $server_pid)"
    # --max-prefill-tokens 262144 \
    # --chunked-prefill-size -1 \
    # --max-running-requests 64 \
    #     --max-queued-requests 256 \
    #     --page-size 128 \
    #     --chunked-prefill-size 8192 \

    if ! wait_for_server "$server_port" "$server_pid" "SGLang server"; then
        tail -50 "$server_log" | tee -a "$BATCH_LOG_FILE"
        stop_server "$server_pid" "SGLang server"
        return 1
    fi

    # ------------------------------------------------------------------
    # Warmup run (results discarded)
    # ------------------------------------------------------------------
    log_message "Running warmup (concurrency=1, in=1024, out=128)..."
    set +e
    cd "$SCRIPT_DIR"
    CUDA_VISIBLE_DEVICES="" \
    "$PYTHON" -m tore_speed_eval.eval \
        --provider=vllm \
        --base_url="http://localhost:${server_port}/v1" \
        --api_key="" \
        --model_name="$model_name" \
        --evaluation_output_path=/dev/null \
        --dataset_type=synthetic \
        --synthetic_input_length=1024 \
        --synthetic_output_length=128 \
        --stream=true \
        --temperature=1.0 \
        --top_p=0.95 \
        --num_gpus=1 \
        --concurrency=1 \
        --num_examples=8 \
        --chat=false \
        2>&1 | tee -a "$BATCH_LOG_FILE"
    set -e
    log_message "✓ Warmup done"

    # ------------------------------------------------------------------
    # Throughput sweeps: BS × input_len
    # ------------------------------------------------------------------
    local stats_log="${result_dir}/per_request_stats.log"
    local overall_exit=0
    for bs in "${BATCH_SIZES[@]}"; do
        for input_len in "${INPUT_LENS[@]}"; do
            for output_len in "${OUTPUT_LENS[@]}"; do
                local label_in label_out
                label_in=$(input_len_label "$input_len")
                label_out=$(output_len_label "$output_len")
                local csv_path="${result_dir}/bs${bs}_${label_in}_${label_out}.csv"

                # Skip if result already exists
                if [ -f "$csv_path" ]; then
                    log_message "  Skip BS=${bs} ${label_in} ${label_out}: $csv_path already exists"
                    continue
                fi

                # Record server log line count before this run for stats extraction
                local log_line_before
                log_line_before=$(wc -l < "$server_log" 2>/dev/null || echo 0)

                local num_examples=$NUM_EXAMPLES
                log_message "  BS=${bs}  input=${label_in}  output=${label_out}  examples=${num_examples}"
                set +e
                cd "$SCRIPT_DIR"
                CUDA_VISIBLE_DEVICES="" \
                "$PYTHON" -m tore_speed_eval.eval \
                    --provider=vllm \
                    --base_url="http://localhost:${server_port}/v1" \
                    --api_key="" \
                    --model_name="$model_name" \
                    --evaluation_output_path="$csv_path" \
                    --dataset_type=synthetic \
                    --synthetic_input_length="$input_len" \
                    --synthetic_output_length="$output_len" \
                    --stream=true \
                    --temperature=1.0 \
                    --top_p=0.95 \
                    --num_gpus="$tp_size" \
                    --concurrency="$bs" \
                    --num_examples="$num_examples" \
                    --chat=false \
                    2>&1 | tee -a "$BATCH_LOG_FILE"
                local eval_exit=${PIPESTATUS[0]}
                set -e

                if [ $eval_exit -ne 0 ]; then
                    log_message "  ✗ BS=${bs} ${label_in} ${label_out} failed (exit: $eval_exit)"
                    overall_exit=$eval_exit
                else
                    log_message "  ✓ BS=${bs} ${label_in} ${label_out} -> $csv_path"
                fi

                # Extract per-request TPS / OTPS / TTFT from server log for this run
                extract_per_request_stats \
                    "$server_log" "$log_line_before" "$rot_suffix" \
                    "$bs" "${label_in}_${label_out}" "$stats_log"
            done
        done
    done

    stop_server "$server_pid" "SGLang server"
    return $overall_exit
}

# =============================================================================
# Preflight checks
# =============================================================================

# 1. tore-speed-eval submodule initialized
if [ ! -f "$TORE_SPEED_EVAL_DIR/setup.py" ] && [ ! -f "$TORE_SPEED_EVAL_DIR/pyproject.toml" ]; then
    echo "ERROR: tore-speed-eval submodule not initialized."
    echo "       Run: git submodule update --init --recursive"
    exit 1
fi

# 2. conda env exists
if [ ! -f "$PYTHON" ]; then
    echo "ERROR: conda env '$CONDA_ENV_NAME' not found (expected Python at $PYTHON)."
    echo "       Run: bash setup_env.sh"
    exit 1
fi

# 3. tore_speed_eval installed; install from submodule if not (use pip show, not python import — avoids slow torch load)
if ! "$CONDA_ENV_DIR/bin/pip" show tore-speed-eval &>/dev/null; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] tore_speed_eval not found — installing from submodule..."
    "$CONDA_ENV_DIR/bin/pip" install -e "$TORE_SPEED_EVAL_DIR" -q
    if ! "$CONDA_ENV_DIR/bin/pip" show tore-speed-eval &>/dev/null; then
        echo "ERROR: tore_speed_eval install failed."
        exit 1
    fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ tore_speed_eval installed"
fi

# =============================================================================
# Main — sequential scheduling with GPU-aware overlap detection
# =============================================================================

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] =========================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] KV-cache Rotation Throughput Benchmark"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Configs:       ${#MODEL_CONFIGS[@]} entry(s)"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Batch sizes:   ${BATCH_SIZES[*]}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Input lens:    ${INPUT_LENS[*]}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Output lens:   ${OUTPUT_LENS[*]}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Num examples:  equals batch size"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU free threshold: ${GPU_FREE_MEM_MB} MB"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] KV dump base:  ${KV_DUMP_BASE}"
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

    # Lookup num_layers; error if model not registered
    num_layers="${MODEL_NUM_LAYERS[$model_name]}"
    if [[ -z "$num_layers" ]]; then
        echo "ERROR: model '$model_name' not found in MODEL_NUM_LAYERS table."
        echo "       Add it at the top of this script and retry."
        exit 1
    fi

    model_short="$(extract_model_short_name "$model_name")"
    kv_dtype_lower="${kv_dtype,,}"
    if [[ "$mode" == "BASE" ]]; then
        rot_suffix="baseline_${kv_dtype_lower}"
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
    CONFIG_LABELS[$i]="$label"
    EXIT_CODES[$i]=-1

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$((i+1))/${N}] Waiting for GPU(s) [$gpu_devices]: $label"
    wait_for_gpus_free "$gpu_devices" "$label"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching: $label"
    benchmark_single_model "$mode" "$hadamard" "$rotate_v" "$hadamard_order" "$kv_dtype" \
                           "$model_name" "$num_layers" "$n_clusters" \
                           "$tp_size" "$ep_size" "$dp_size" "$gpu_devices" \
                           "$server_port" &
    PIDS[$i]=$!

    # If next config shares any GPU, wait for current job to finish first
    next=$((i + 1))
    if [ "$next" -lt "$N" ]; then
        # eval_gpus is field 8 in the new format
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
            sleep 30 &
            wait $!
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] No GPU overlap with next config, sleeping 60s before launching next..."
            sleep 30 &
            wait $!
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
