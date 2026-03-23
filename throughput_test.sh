#!/bin/bash
# throughput_test.sh — SGLang throughput benchmarking for Hadamard / Hadamard+QR KV cache
#
# Modes:
#   BASE         — no rotation (BF16 or INT4 KV)
#   Rotation     — Hadamard rotation only (INT4 KV)
#   Rotation_QR  — Hadamard rotation + learned Q-rotation matrix (INT4 KV)
#
# GPU scheduling: configs on non-overlapping GPUs launch in parallel.
# Results: throughput_results/{model_short}/{rot_suffix}/bs{N}_{in_label}.csv
#
# Usage:
#   PYTHON_BIN=/path/to/python bash throughput_test.sh
#   or set PYTHON_BIN in the environment (same as launch_hadamard_qr_server_tp1.sh)

set -eo pipefail

cleanup() {
    trap '' INT TERM
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Interrupted — killing all child processes..."
    kill -9 -- -$$ 2>/dev/null || true
    exit 130
}
trap cleanup INT TERM

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# =============================================================================
# Throughput Test Parameters
# =============================================================================
BATCH_SIZES=(1 8 16 32)
INPUT_LENS=(8192 16384 32768)
MAX_NEW_TOKENS=1024
NUM_EXAMPLES=96

# =============================================================================
# Model Configs
# =============================================================================
# Format: "mode|rotate_k|rotate_v|hadamard_order|kv_dtype|model_name|q_rotation_path|gpu_devices|tp_size|ep_size|dp_size"
#
#   mode            : BASE, Rotation, or Rotation_QR
#   rotate_k        : 1 = rotate K (HADAMARD=1); ignored for BASE
#   rotate_v        : 1 = also rotate V; ignored for BASE
#   hadamard_order  : e.g. 16, 64 (ignored for BASE)
#   kv_dtype        : BF16 or INT4
#   model_name      : full HuggingFace model ID
#   q_rotation_path : path to .pt file (Rotation_QR only; use "" for BASE/Rotation)
#   gpu_devices     : comma-separated GPU IDs for the server
#   tp_size         : tensor parallel size
#   ep_size         : expert parallel size
#   dp_size         : data parallel size
#
MODEL_CONFIGS=(
    # mode        |rk|rv|ho |dtype|model                          |q_rotation_path                                                                     |gpu|tp|ep|dp
    "BASE        |0 |0 |0  |BF16 |Qwen/Qwen3-4B-Thinking-2507   |                                                                                    |0  |1 |1 |1"
    "BASE        |0 |0 |0  |INT4 |Qwen/Qwen3-4B-Thinking-2507   |                                                                                    |1  |1 |1 |1"
    "Rotation    |1 |0 |16 |INT4 |Qwen/Qwen3-4B-Thinking-2507   |                                                                                    |2  |1 |1 |1"
    "Rotation_QR |1 |0 |16 |INT4 |Qwen/Qwen3-4B-Thinking-2507   |/data/jisenli2/zhongzhu_kv/q_rotation_layer_second_moment_damp01.pt                 |3  |1 |1 |1"
)

# =============================================================================
# Server & Path Config
# =============================================================================
BASE_PORT=30200

# Hardcoded conda env path
CONDA_ENV_DIR="/data/$USER/miniconda/envs/zhongzhu_kv"
PYTHON_BIN="$CONDA_ENV_DIR/bin/python3"

# tore-speed-eval: defaults to the one in the sibling kv_rotation repo; override with env var
TORE_SPEED_EVAL_DIR="${TORE_SPEED_EVAL_DIR:-/data/jisenli2/kv_rotation/tore-speed-eval}"

RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/throughput_results}"
LOGS_DIR="${LOGS_DIR:-${REPO_ROOT}/throughput_logs}"

export HF_HOME="${HF_HOME:-/data/shared/huggingface}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/data/${USER}/.triton/cache}"

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
        sleep "$GPU_POLL_INTERVAL" &
        wait $!
        waited=$((waited + GPU_POLL_INTERVAL))
    done
    if [ "$waited" -gt 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU(s) [$gpu_list] now free after $((waited / 60))min"
    fi
}

# Convert token count to short label: 8192 → in8k
input_len_label() { echo "in$((${1} / 1024))k"; }

# =============================================================================
# extract_per_request_stats
#   Parse SGLang "Finish:" log lines and append a summary block.
# =============================================================================
extract_per_request_stats() {
    local server_log="$1" log_line_before="$2" rot_suffix="$3" \
          bs="$4" label_in="$5" stats_log="$6"

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

    echo "$run_log" \
        | grep "Finish:" \
        | grep -v "HEALTH_CHECK" \
        | "$PYTHON_BIN" -c "
import sys, re

tps_vals, otps_vals, ttft_vals = [], [], []

for line in sys.stdin:
    pt   = re.search(r\"'prompt_tokens': (\d+)\", line)
    ct   = re.search(r\"'completion_tokens': (\d+)\", line)
    e2e  = re.search(r\"'e2e_latency': ([\d.]+)\", line)
    recv = re.search(r\"'request_received_ts': ([\d.]+)\", line)
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
# benchmark_single_model
# =============================================================================
benchmark_single_model() {
    local mode="$1" rotate_k="$2" rotate_v="$3" hadamard_order="$4" kv_dtype="$5" \
          model_name="$6" q_rotation_path="$7" \
          tp_size="$8" ep_size="$9" dp_size="${10}" gpu_devices="${11}" \
          server_port="${12}"

    local model_short
    model_short="$(extract_model_short_name "$model_name")"

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

    local kv_cache_dtype
    case "$kv_dtype" in
        BF16) kv_cache_dtype="auto" ;;
        INT4) kv_cache_dtype="int4" ;;
        *) echo "ERROR: unknown kv_dtype $kv_dtype"; return 1 ;;
    esac

    local kv_dtype_lower="${kv_dtype,,}"
    local rot_suffix
    case "$mode" in
        BASE)         rot_suffix="baseline_${kv_dtype_lower}" ;;
        Rotation)     rot_suffix="rotation_${kv_dtype_lower}_${rotate_k}_${rotate_v}_${hadamard_order}" ;;
        Rotation_QR)  rot_suffix="rotation_qr_${kv_dtype_lower}_${rotate_k}_${rotate_v}_${hadamard_order}" ;;
    esac

    local result_dir="$RESULTS_DIR/${model_short}/${rot_suffix}"
    local log_dir="$LOGS_DIR/${model_short}"
    mkdir -p "$result_dir" "$log_dir"
    BATCH_LOG_FILE=$(unique_log_path "$log_dir/${rot_suffix}.log")

    # BF16 KV uses more memory; leave room for CUDA graphs
    local mem_fraction
    if [[ "$kv_cache_dtype" == "auto" ]]; then
        mem_fraction="0.65"
    else
        mem_fraction="0.8"
    fi

    # Prefill/decode backend: int4 requires fa3 + triton
    local prefill_backend decode_backend
    if [[ "$kv_cache_dtype" == "int4" ]]; then
        prefill_backend="fa3"
        decode_backend="triton"
    else
        prefill_backend="triton"
        decode_backend="triton"
    fi

    local local_pythonpath="${REPO_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"

    log_message "=========================================="
    log_message "Mode:          $mode"
    log_message "Model:         $model_name"
    log_message "TP/EP/DP:      $tp_size/$ep_size/$dp_size"
    log_message "GPUs:          $gpu_devices"
    log_message "KV dtype:      $kv_dtype (cache: $kv_cache_dtype)"
    log_message "rotate_k=$rotate_k  rotate_v=$rotate_v  hadamard_order=$hadamard_order"
    [[ "$mode" == "Rotation_QR" ]] && log_message "Q rotation:    $q_rotation_path"
    log_message "Batch sizes:   ${BATCH_SIZES[*]}"
    log_message "Input lens:    ${INPUT_LENS[*]}"
    log_message "Max new tok:   $MAX_NEW_TOKENS  num_examples: $NUM_EXAMPLES"
    log_message "Results dir:   $result_dir"
    log_message "=========================================="

    # ------------------------------------------------------------------
    # Start SGLang server
    # ------------------------------------------------------------------
    log_message "Starting SGLang server on port $server_port..."
    local server_log
    server_log=$(unique_log_path "$log_dir/${rot_suffix}_server.log")

    local -a server_env=(
        PYTHONPATH="${local_pythonpath}"
        CUDA_VISIBLE_DEVICES="${gpu_devices}"
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
        "$PYTHON_BIN" -m sglang.launch_server \
            --model-path "$model_name" \
            --max-running-requests 64 \
            --max-queued-requests 256 \
            --page-size 128 \
            --chunked-prefill-size 8192 \
            --mem-fraction-static "$mem_fraction" \
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
            --log-requests \
            --log-requests-level 0 \
            > "$server_log" 2>&1 &
    local server_pid=$!
    log_message "Server started (PID: $server_pid)"

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
    cd "$REPO_ROOT"
    CUDA_VISIBLE_DEVICES="" \
    "$PYTHON_BIN" -m tore_speed_eval.eval \
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
        --top_p=0.7 \
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
            local label_in
            label_in=$(input_len_label "$input_len")
            local csv_path="${result_dir}/bs${bs}_${label_in}.csv"

            if [ -f "$csv_path" ]; then
                log_message "  Skip BS=${bs} ${label_in}: $csv_path already exists"
                continue
            fi

            local log_line_before
            log_line_before=$(wc -l < "$server_log" 2>/dev/null || echo 0)

            log_message "  BS=${bs}  input=${label_in}  output=${MAX_NEW_TOKENS}tok  examples=${NUM_EXAMPLES}"
            set +e
            cd "$REPO_ROOT"
            CUDA_VISIBLE_DEVICES="" \
            "$PYTHON_BIN" -m tore_speed_eval.eval \
                --provider=vllm \
                --base_url="http://localhost:${server_port}/v1" \
                --api_key="" \
                --model_name="$model_name" \
                --evaluation_output_path="$csv_path" \
                --dataset_type=synthetic \
                --synthetic_input_length="$input_len" \
                --synthetic_output_length="$MAX_NEW_TOKENS" \
                --stream=true \
                --temperature=1.0 \
                --top_p=0.7 \
                --num_gpus="$tp_size" \
                --concurrency="$bs" \
                --num_examples="$NUM_EXAMPLES" \
                --chat=false \
                2>&1 | tee -a "$BATCH_LOG_FILE"
            local eval_exit=${PIPESTATUS[0]}
            set -e

            if [ $eval_exit -ne 0 ]; then
                log_message "  ✗ BS=${bs} ${label_in} failed (exit: $eval_exit)"
                overall_exit=$eval_exit
            else
                log_message "  ✓ BS=${bs} ${label_in} -> $csv_path"
            fi

            extract_per_request_stats \
                "$server_log" "$log_line_before" "$rot_suffix" \
                "$bs" "$label_in" "$stats_log"
        done
    done

    stop_server "$server_pid" "SGLang server"
    return $overall_exit
}

# =============================================================================
# Preflight checks
# =============================================================================

# 1. tore-speed-eval directory exists
if [ ! -f "${TORE_SPEED_EVAL_DIR}/pyproject.toml" ] && [ ! -f "${TORE_SPEED_EVAL_DIR}/setup.py" ]; then
    echo "ERROR: tore-speed-eval not found at: ${TORE_SPEED_EVAL_DIR}"
    echo "       Set TORE_SPEED_EVAL_DIR env var to the correct path."
    exit 1
fi

# 2. Conda env / Python binary exists
if [ ! -f "$PYTHON_BIN" ]; then
    echo "ERROR: conda env 'zhongzhu_kv' not found (expected Python at $PYTHON_BIN)."
    echo "       Run: bash setup_env.sh"
    exit 1
fi

# 3. tore_speed_eval installed; install from directory if not
if ! "$PYTHON_BIN" -c "import tore_speed_eval" &>/dev/null 2>&1; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] tore_speed_eval not found — installing from ${TORE_SPEED_EVAL_DIR}..."
    "$PYTHON_BIN" -m pip install -e "${TORE_SPEED_EVAL_DIR}" -q
    if ! "$PYTHON_BIN" -c "import tore_speed_eval" &>/dev/null 2>&1; then
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
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Hadamard/QR KV Cache Throughput Benchmark"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Configs:        ${#MODEL_CONFIGS[@]} entry(s)"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Batch sizes:    ${BATCH_SIZES[*]}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Input lens:     ${INPUT_LENS[*]}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Max new tokens: $MAX_NEW_TOKENS"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Num examples:   $NUM_EXAMPLES"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Python:         $PYTHON_BIN"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Repo root:      $REPO_ROOT"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Results dir:    $RESULTS_DIR"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] =========================================="
echo ""

OVERALL_EXIT=0
declare -a PIDS
declare -a EXIT_CODES
declare -A CONFIG_LABELS
N=${#MODEL_CONFIGS[@]}

for i in "${!MODEL_CONFIGS[@]}"; do
    config="${MODEL_CONFIGS[$i]}"
    IFS='|' read -r mode rotate_k rotate_v hadamard_order kv_dtype model_name \
                    q_rotation_path \
                    gpu_devices tp_size ep_size dp_size <<< "$config"
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

    model_short="$(extract_model_short_name "$model_name")"
    kv_dtype_lower="${kv_dtype,,}"
    case "$mode" in
        BASE)         rot_suffix="baseline_${kv_dtype_lower}" ;;
        Rotation)     rot_suffix="rotation_${kv_dtype_lower}_${rotate_k}_${rotate_v}_${hadamard_order}" ;;
        Rotation_QR)  rot_suffix="rotation_qr_${kv_dtype_lower}_${rotate_k}_${rotate_v}_${hadamard_order}" ;;
        *) echo "ERROR: unknown mode '$mode'"; exit 1 ;;
    esac

    server_port=$((BASE_PORT + i))
    label="${model_short}_${rot_suffix} (port=$server_port, gpu=$gpu_devices)"
    CONFIG_LABELS[$i]="$label"
    EXIT_CODES[$i]=-1

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$((i+1))/${N}] Waiting for GPU(s) [$gpu_devices]: $label"
    wait_for_gpus_free "$gpu_devices" "$label"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching: $label"
    benchmark_single_model "$mode" "$rotate_k" "$rotate_v" "$hadamard_order" "$kv_dtype" \
                           "$model_name" "$q_rotation_path" \
                           "$tp_size" "$ep_size" "$dp_size" "$gpu_devices" \
                           "$server_port" &
    PIDS[$i]=$!

    # If next config shares any GPU, wait for current job to finish first
    next=$((i + 1))
    if [ "$next" -lt "$N" ]; then
        # gpu_devices is field 8 in the new config format
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
            sleep 60 &
            wait $!
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] No GPU overlap with next config, sleeping 60s before launching next..."
            sleep 60 &
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
