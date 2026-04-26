#!/bin/bash
# mode2_throughput_test.sh — SGLang throughput benchmark with prefix-cache ON
#
# Mode 2 vs Mode 1 differences:
#   - Prefix cache ENABLED (no --disable-radix-cache)
#   - Real-data dataset (tore-speed-eval narrativeqa), NOT synthetic
#   - Same workload runs N times back-to-back (no server restart) so that
#     run 2+ benefits from the radix tree populated by run 1
#   - Primary metric: later-run throughput (prefix-cache hit); run 1 kept for delta
#
# GPU scheduling: configs on non-overlapping GPUs launch in parallel.
# Results: mode2_throughput_results/{model_short}/{rot_suffix}/bs{N}_{ds}_run{R}.csv

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
# BATCH_SIZES=(1 8 16 32 256)
# NUM_EXAMPLES=(4 32 32 32 256)                                  # paired 1:1 with BATCH_SIZES

BATCH_SIZES=(1 8 16 32)
NUM_EXAMPLES=(4 32 32 32)                                  # paired 1:1 with BATCH_SIZES

MAX_TOKENS="${MAX_TOKENS:-1024}"                           # output cap (input comes from dataset)
NUM_RUNS="${NUM_RUNS:-2}"                                  # repeat-run count for prefix-cache warm-up

# Real-data dataset for prefix-cache-on benchmark
HF_DATASET="${HF_DATASET:-togethercomputer/tore-speed-eval-narrativeqa-100k}"
HF_DATASET_LABEL="${HF_DATASET_LABEL:-nqa100k}"            # short label embedded in CSV filename

# Sampling parameters (applied to main eval)
# Empty → auto-pick per-model-family defaults in benchmark_single_model:
#   Qwen series → 0.7 / 0.95
#   GLM series  → 1.0 / 0.7
TEMPERATURE="${TEMPERATURE:-}"
TOP_P="${TOP_P:-}"

# =============================================================================
# Model Configs
# =============================================================================
# Format: "fuse|mode|hadamard|rotate_v|hadamard_order|kv_dtype|model_name|eval_gpus|eval_tp|eval_ep|eval_dp"
#
#   fuse          : SGLANG_FUSE_HADAMARD_INT4_KV — 1=fused Hadamard, 0=unfused
#   mode          : BASE or QUANT
#   hadamard      : 0 or 1 (ignored for BASE)
#   rotate_v      : 0 or 1 (ignored for BASE)
#   hadamard_order: e.g. 16, 64, 128 (ignored for BASE)
#   kv_dtype      : BF16 or INT4
#   model_name    : full HuggingFace model ID
#   eval_gpus     : comma-separated GPU IDs for the server
#   eval_tp       : tensor parallel size
#   eval_ep       : expert parallel size
#   eval_dp       : data parallel size
#
MODEL_CONFIGS=(
    # ---- INT4 fused kernel, hadamard=1 rotate_v=1 order=16 (donglin-equivalent), TP=1 ----
    "1|QUANT|1|1|16|INT4|Qwen/Qwen3-4B-Thinking-2507|0|1|1|1"
    "1|QUANT|1|1|16|INT4|Qwen/Qwen3-8B|4|1|1|1"
    # "1|QUANT|1|1|16|INT4|zai-org/GLM-4.7-FP8|0,1,2,3,4,5,6,7|8|1|1"
)

# =============================================================================
# Server & Path Config
# =============================================================================
BASE_PORT=32100


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TORE_SPEED_EVAL_DIR="$SCRIPT_DIR/tore-speed-eval"
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/mode2_throughput_results}"
LOGS_DIR="${LOGS_DIR:-$SCRIPT_DIR/mode2_throughput_logs}"

export HF_HOME=/data/shared/huggingface

CONDA_BASE="/data/$USER/miniconda"
CONDA_ENV_NAME="fused_sglang_env"
CONDA_ENV_DIR="$CONDA_BASE/envs/$CONDA_ENV_NAME"
PYTHON="$CONDA_ENV_DIR/bin/python3"

export TRITON_CACHE_DIR="/dev/shm/triton_cache_$USER"
# flashinfer reads FLASHINFER_WORKSPACE_BASE (not FLASHINFER_CACHE_DIR) per flashinfer/jit/env.py
# — derives cache to $FLASHINFER_WORKSPACE_BASE/.cache/flashinfer/
export FLASHINFER_WORKSPACE_BASE="/data/$USER"
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

# Convert token count to short label: 8192 → in8k, 16384 → in16k, 32768 → in32k
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
        | grep -E "Finish:|#running-req" \
        | grep -v "HEALTH_CHECK" \
        | "$PYTHON" -c "
import sys, re

tps_vals, otps_vals, ttft_vals = [], [], []
cache_ratios, cached_tokens_vals = [], []
running_req_vals = []

for line in sys.stdin:
    # Scheduler periodic log line: 'Decode batch. #running-req: N. #token: ...'
    rr = re.search(r'#running-req:\s*(\d+)', line)
    if rr:
        running_req_vals.append(int(rr.group(1)))
        continue

    pt   = re.search(r\"'prompt_tokens': (\d+)\", line)
    ct   = re.search(r\"'completion_tokens': (\d+)\", line)
    e2e  = re.search(r\"'e2e_latency': ([\d.]+)\", line)
    recv = re.search(r\"'request_received_ts': ([\d.]+)\", line)
    cached = re.search(r\"'cached_tokens': (\d+)\", line)
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

    if cached and p > 0:
        cached_n = int(cached.group(1))
        cached_tokens_vals.append(cached_n)
        cache_ratios.append(cached_n / p)

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

summarize(tps_vals,           'TPS  (prompt+output / e2e)',    'tok/s')
# NOTE: new SGLang dropped prefill_finished_ts / request_received_ts fields from the
# Finish: log, so OTPS and TTFT can no longer be derived from server log.
# Refer to tore-speed-eval CSV (ttft_mean / user_tps_mean) for these metrics instead.
summarize(cached_tokens_vals, 'CACHED_TOKENS (cache hit)',     'tokens')
summarize(cache_ratios,       'CACHE_HIT_RATIO',               '(0-1)')
summarize(running_req_vals,   'RUNNING_REQS (server in-flight)', 'reqs')
" 2>&1 | tee -a "$stats_log"
}

# =============================================================================
# benchmark_single_model
# =============================================================================
benchmark_single_model() {
    local mode="$1" hadamard="$2" rotate_v="$3" hadamard_order="$4" kv_dtype="$5" \
          model_name="$6" \
          tp_size="$7" ep_size="$8" dp_size="$9" gpu_devices="${10}" \
          server_port="${11}" fuse_hadamard="${12}"
    if [[ -z "$fuse_hadamard" ]]; then
        echo "ERROR: fuse_hadamard (field 1) missing in config"
        return 1
    fi

    local model_short
    model_short="$(extract_model_short_name "$model_name")"

    if [[ "$mode" != "BASE" && "$mode" != "QUANT" ]]; then
        echo "ERROR: mode must be BASE or QUANT, got: '$mode'"
        return 1
    fi

    # ----- Per-model-family sampling defaults (overridable via TEMPERATURE/TOP_P env) -----
    local eff_temperature eff_top_p
    if [[ "$model_name" == *"GLM"* || "$model_name" == *"glm"* ]]; then
        eff_temperature="${TEMPERATURE:-1.0}"
        eff_top_p="${TOP_P:-0.7}"
    else  # Qwen and fallback
        eff_temperature="${TEMPERATURE:-0.7}"
        eff_top_p="${TOP_P:-0.95}"
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
    local fuse_suffix; fuse_suffix=$([[ "$fuse_hadamard" == "1" ]] && echo "fused_finetuned_v2" || echo "unfused")
    local rot_suffix
    if [[ "$mode" == "BASE" ]]; then
        rot_suffix="baseline_${kv_dtype_lower}"
    else
        rot_suffix="quant_${kv_dtype_lower}_${hadamard}_${rotate_v}_${hadamard_order}_${fuse_suffix}"
    fi

    local result_dir="$RESULTS_DIR/${model_short}/${rot_suffix}"
    local log_dir="$LOGS_DIR/${model_short}"
    mkdir -p "$result_dir" "$log_dir"
    local BATCH_LOG_FILE
    BATCH_LOG_FILE=$(unique_log_path "$log_dir/${rot_suffix}.log")

    log_message "=========================================="
    log_message "Mode:      $mode"
    log_message "Model:     $model_name"
    log_message "TP/EP/DP:  $tp_size/$ep_size/$dp_size"
    log_message "GPUs:      $gpu_devices"
    log_message "KV dtype:  $kv_dtype (cache: $kv_cache_dtype)"
    log_message "HADAMARD=$hadamard  ROTATE_V=$rotate_v  HADAMARD_ORDER=$hadamard_order  FUSE_INT4=$fuse_hadamard"
    log_message "Batch sizes:  ${BATCH_SIZES[*]}"
    log_message "Num examples: ${NUM_EXAMPLES[*]}  (paired 1:1 with batch sizes)"
    log_message "Dataset:      $HF_DATASET (label=$HF_DATASET_LABEL)"
    log_message "Max tokens:   $MAX_TOKENS    Num runs: $NUM_RUNS"
    log_message "Results dir:  $result_dir"
    log_message "=========================================="

    # ------------------------------------------------------------------
    # Build server flags (reused across BS-level server restarts)
    # ------------------------------------------------------------------
    local mem_fraction="0.8"
    local EXTRA_KV_ARGS=()
    # GLM-4.7 chat parsing flags (tool calls / reasoning) per zai-org's recommended launch.
    if [[ "$model_name" == *GLM* || "$model_name" == *glm* ]]; then
        EXTRA_KV_ARGS+=(--tool-call-parser glm47 --reasoning-parser glm45)
    fi

    # YaRN RoPE scaling for Qwen3 models with 32k native context (Qwen3-8B / Qwen3-32B).
    # narrativeqa-100k prompts can exceed 32k → extend to 131072 with YaRN factor=4.0.
    # Qwen3-4B-Thinking-2507 already has 256k native context — skip.
    if [[ "$model_name" == "Qwen/Qwen3-8B" || "$model_name" == "Qwen/Qwen3-32B" ]]; then
        # Note: rope_theta MUST be included; transformers v5 normalizes rope_scaling
        # into rope_parameters by replacement (not merge), so omitting rope_theta
        # makes newer SGLang qwen3.py raise KeyError: 'rope_theta'.
        # --context-length is required: SGLang reads max_position_embeddings (40960)
        # from config.json and does NOT auto-extend it from YaRN factor=4.
        # SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 bypasses SGLang's safety check
        # that refuses --context-length > derived value.
        EXTRA_KV_ARGS+=(--json-model-override-args '{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768,"rope_theta":1000000}}'
                        --context-length 131072)
        export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
    fi

    # Default attention backend: fa3 for prefill + triton for decode
    local ATTN_ARGS=(--prefill-attention-backend fa3 --decode-attention-backend triton)

    unset DUMP_KVCACHE DUMP_KVCACHE_TOKENS DUMP_KVCACHE_DIR

    # ------------------------------------------------------------------
    # Mode 2 eval: server is RESTARTED per BS so each BS starts with a
    # clean radix tree. Within a BS, NUM_RUNS passes run back-to-back
    # against the same server so run 2+ hits the KV populated by run 1.
    # --enable-cache-report populates 'cached_tokens_details' in Finish: log.
    # ------------------------------------------------------------------
    local stats_log="${result_dir}/per_request_stats.log"
    local overall_exit=0
    for idx in "${!BATCH_SIZES[@]}"; do
        bs="${BATCH_SIZES[$idx]}"
        local num_examples="${NUM_EXAMPLES[$idx]}"

        # Skip entire BS if all NUM_RUNS CSVs already exist
        local all_exist=1
        for run_idx in $(seq 1 "$NUM_RUNS"); do
            if [ ! -f "${result_dir}/bs${bs}_${HF_DATASET_LABEL}_run${run_idx}.csv" ]; then
                all_exist=0
                break
            fi
        done
        if [ "$all_exist" -eq 1 ]; then
            log_message "  Skip BS=${bs}: all ${NUM_RUNS} runs already exist"
            continue
        fi

        # ----- Start a fresh SGLang server for this BS -----
        local server_log
        server_log=$(unique_log_path "$log_dir/${rot_suffix}_bs${bs}_server.log")
        log_message "---- BS=${bs}: starting SGLang server on port $server_port (cold radix tree) ----"

        HADAMARD=$hadamard \
        ROTATE_V=$rotate_v \
        HADAMARD_ORDER=$hadamard_order \
        SGLANG_FUSE_HADAMARD_INT4_KV="$fuse_hadamard" \
        CUDA_VISIBLE_DEVICES=$gpu_devices \
        PATH="$(dirname "$PYTHON"):$PATH" \
        LIBRARY_PATH="/usr/local/cuda/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}" \
        LD_LIBRARY_PATH="/usr/local/cuda/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        "$PYTHON" -m sglang.launch_server \
            --model-path "$model_name" \
            --mem-fraction-static "$mem_fraction" \
            --kv-cache-dtype "$kv_cache_dtype" \
            "${ATTN_ARGS[@]}" \
            "${EXTRA_KV_ARGS[@]}" \
            --sampling-backend flashinfer \
            --tensor-parallel-size "$tp_size" \
            --expert-parallel-size "$ep_size" \
            --data-parallel-size "$dp_size" \
            --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 119}' \
            --host 0.0.0.0 \
            --port "$server_port" \
            --trust-remote-code \
            --log-requests \
            --log-requests-level 0 \
            --enable-request-time-stats-logging \
            --enable-cache-report \
            --chunked-prefill-size 32768 \
            --max-running-requests 512 \
            > "$server_log" 2>&1 &
        local server_pid=$!
        log_message "  [BS=${bs}] Server PID=$server_pid, log=$(basename "$server_log")"

        if ! wait_for_server "$server_port" "$server_pid" "SGLang server (BS=$bs)"; then
            tail -50 "$server_log" | tee -a "$BATCH_LOG_FILE"
            stop_server "$server_pid" "SGLang server (BS=$bs)"
            overall_exit=1
            sleep 10 &
            wait $!
            continue
        fi

        # ----- Run NUM_RUNS passes back-to-back against this fresh server -----
        for run_idx in $(seq 1 "$NUM_RUNS"); do
            local csv_path="${result_dir}/bs${bs}_${HF_DATASET_LABEL}_run${run_idx}.csv"

            # Skip if result already exists
            if [ -f "$csv_path" ]; then
                log_message "  Skip BS=${bs} run=${run_idx}: $csv_path already exists"
                continue
            fi

            # Record server log line count before this run for stats extraction
            local log_line_before
            log_line_before=$(wc -l < "$server_log" 2>/dev/null || echo 0)

            log_message "  BS=${bs}  dataset=${HF_DATASET_LABEL}  run=${run_idx}/${NUM_RUNS}  examples=${num_examples}  max_tokens=${MAX_TOKENS}"
            set +e
            cd "$SCRIPT_DIR"
            CUDA_VISIBLE_DEVICES="" \
            "$PYTHON" -m tore_speed_eval.eval \
                --provider=vllm \
                --base_url="http://localhost:${server_port}/v1" \
                --api_key="" \
                --model_name="$model_name" \
                --evaluation_output_path="$csv_path" \
                --dataset_type=hf \
                --hf_dataset="$HF_DATASET" \
                --hf_dataset_column_name=messages \
                --stream=true \
                --chat=true \
                --temperature="$eff_temperature" \
                --top_p="$eff_top_p" \
                --num_gpus="$tp_size" \
                --concurrency="$bs" \
                --num_examples="$num_examples" \
                --max_tokens="$MAX_TOKENS" \
                2>&1 | tee -a "$BATCH_LOG_FILE"
            local eval_exit=${PIPESTATUS[0]}
            set -e

            if [ $eval_exit -ne 0 ]; then
                log_message "  ✗ BS=${bs} run=${run_idx} failed (exit: $eval_exit)"
                overall_exit=$eval_exit
            else
                log_message "  ✓ BS=${bs} run=${run_idx} -> $csv_path"
            fi

            # Extract per-request TPS / OTPS / TTFT / cache-hit from server log for this run
            extract_per_request_stats \
                "$server_log" "$log_line_before" "$rot_suffix" \
                "$bs" "${HF_DATASET_LABEL}_run${run_idx}" "$stats_log"
        done

        # ----- Stop server + cool down before next BS -----
        stop_server "$server_pid" "SGLang server (BS=$bs)"
        sleep 10 &
        wait $!
    done

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
    "$CONDA_ENV_DIR/bin/pip" install -e "$TORE_SPEED_EVAL_DIR"
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
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Num examples:  ${NUM_EXAMPLES[*]}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Dataset:       $HF_DATASET (label=$HF_DATASET_LABEL)"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Max tokens:    $MAX_TOKENS  Num runs: $NUM_RUNS"
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
    IFS='|' read -r fuse_hadamard mode hadamard rotate_v hadamard_order kv_dtype model_name \
                    gpu_devices tp_size ep_size dp_size <<< "$config"
    fuse_hadamard="${fuse_hadamard// /}"
    mode="${mode// /}"
    hadamard="${hadamard// /}"
    rotate_v="${rotate_v// /}"
    hadamard_order="${hadamard_order// /}"
    kv_dtype="${kv_dtype// /}"
    model_name="${model_name// /}"
    gpu_devices="${gpu_devices// /}"
    tp_size="${tp_size// /}"
    ep_size="${ep_size// /}"
    dp_size="${dp_size// /}"

    # Validate mode
    if [[ "$mode" != "BASE" && "$mode" != "QUANT" ]]; then
        echo "ERROR: config [$i] mode must be BASE or QUANT, got '$mode'"
        exit 1
    fi

    model_short="$(extract_model_short_name "$model_name")"
    kv_dtype_lower="${kv_dtype,,}"

    # Build rot_suffix preview for the label (must match benchmark_single_model's logic)
    fuse_suffix=$([[ "$fuse_hadamard" == "1" ]] && echo "fused_finetuned_v2" || echo "unfused")
    if [[ "$mode" == "BASE" ]]; then
        rot_suffix="baseline_${kv_dtype_lower}"
    else
        rot_suffix="quant_${kv_dtype_lower}_${hadamard}_${rotate_v}_${hadamard_order}_${fuse_suffix}"
    fi

    server_port=$((BASE_PORT + i))
    label="${model_short}_${rot_suffix} (port=$server_port, gpu=$gpu_devices)"
    CONFIG_LABELS[$i]="$label"
    EXIT_CODES[$i]=-1

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$((i+1))/${N}] Waiting for GPU(s) [$gpu_devices]: $label"
    wait_for_gpus_free "$gpu_devices" "$label"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching: $label"
    benchmark_single_model "$mode" "$hadamard" "$rotate_v" "$hadamard_order" "$kv_dtype" \
                           "$model_name" \
                           "$tp_size" "$ep_size" "$dp_size" "$gpu_devices" \
                           "$server_port" "$fuse_hadamard" &
    PIDS[$i]=$!

    # If next config shares any GPU, wait for current job to finish first
    next=$((i + 1))
    if [ "$next" -lt "$N" ]; then
        # eval_gpus is field 8 in our schema (fuse|mode|hadamard|rotate_v|hadamard_order|kv_dtype|model_name|eval_gpus|...)
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
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Cooling down 30s for GPU memory to release..."
            sleep 30 &
            wait $!
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] No GPU overlap with next config, sleeping 30s before launching next..."
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
