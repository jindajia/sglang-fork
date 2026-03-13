#!/bin/bash
# dump_centroids.sh — Standalone Stage 1 (KV dump) + Stage 2 (K-means centroids)
#
# Configure MODEL_CONFIGS below and run: bash dump_centroids.sh
#
# Path layout:
#   Dump      : {KV_BASE}/{model_short}/{task}-{dump_tokens}-tokens/kv_calibration_layer_*.pt
#   Centroids : {KV_BASE}/{model_short}/{task}-{dump_tokens}-tokens/c_{clusters}/*.pt
#
# Note on --limit for lm_eval:
#   --limit N applies PER SUBTASK for task groups (e.g. mmlu_pro has ~14 subtasks → N*14 requests).

set -eo pipefail

cleanup() {
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Interrupted — killing all child processes..."
    kill -- -$$ 2>/dev/null || true
    exit 130
}
trap cleanup INT TERM

# =============================================================================
# Model → num_layers lookup table
# Add new models here. Script will error if a MODEL_CONFIGS entry is not found.
# =============================================================================
declare -A MODEL_NUM_LAYERS=(
    ["Qwen/Qwen3-4B-Thinking-2507"]=36
    ["Qwen/Qwen3-8B"]=36
)

# =============================================================================
# Model Configs
# =============================================================================
# Format: "model|clusters|dump_gpus|dtp|dep|ddp|kmeans_gpu"
#
#   model      : full HuggingFace model ID
#   clusters   : K-means cluster count
#   dump_gpus  : comma-separated GPU IDs for Stage 1 BF16 dump server
#   dtp        : tensor parallel size for dump server
#   dep        : expert parallel size for dump server
#   ddp        : data parallel size for dump server
#   kmeans_gpu : GPU(s) for Stage 2 K-means (CUDA_VISIBLE_DEVICES)
#
MODEL_CONFIGS=(
    # model                              |clusters|dump_gpus|dtp|dep|ddp|kmeans_gpu
    "Qwen/Qwen3-4B-Thinking-2507        |256      |0,1  |2  |1  |1  |0,1"
    # "Qwen/Qwen3-4B-Thinking-2507        |16      |2,3  |2  |1  |1  |2,3"
    # "Qwen/Qwen3-4B-Thinking-2507        |256      |4,5  |2  |1  |1  |4,5"
    # "Qwen/Qwen3-4B-Thinking-2507        |2048      |0,1,2,3  |4  |1  |1  |0,1,2,3,4,5,6,7"
    # "Qwen/Qwen3-8B                      |64      |0,1,2,3  |4  |1  |1  |0,1,2,3,4,5,6,7"
)

# =============================================================================
# Global settings
# =============================================================================
TASK="mmlu_pro"
LIMIT=16          # per subtask; mmlu_pro has 14 subtasks → 16×14≈224 total
DUMP_TOKENS=20000
BASE_PORT=30001

KV_BASE="${KV_BASE:-/data/jisenli2/kv-cache}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS_DIR="$SCRIPT_DIR/eval_logs"

export HF_HOME=/data/shared/huggingface

CONDA_BASE="/data/jisenli2/miniconda"
CONDA_ENV_NAME="sglang_env"
CONDA_ENV_DIR="$CONDA_BASE/envs/$CONDA_ENV_NAME"
PYTHON="$CONDA_ENV_DIR/bin/python3"
LM_EVAL="$CONDA_ENV_DIR/bin/lm_eval"

mkdir -p "$LOGS_DIR"

# =============================================================================
# Preflight
# =============================================================================
if [ ! -f "$PYTHON" ]; then
    echo "ERROR: conda env '$CONDA_ENV_NAME' not found (expected Python at $PYTHON)."
    echo "       Run: bash setup_env.sh"
    exit 1
fi

# =============================================================================
# Helpers
# =============================================================================
unique_log_path() {
    local base="$1"
    if [ ! -e "$base" ]; then echo "$base"; return; fi
    local i=1
    while [ -e "${base}-${i}" ]; do i=$((i + 1)); done
    echo "${base}-${i}"
}

LOG_FILE="/dev/null"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"; }

wait_for_server() {
    local port="$1" pid="$2"
    local max_wait=1800 elapsed=0
    while [ $elapsed -lt $max_wait ]; do
        if curl -s "http://localhost:${port}/health" > /dev/null 2>&1; then
            log "✓ Dump server ready (${elapsed}s)"
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            log "✗ Dump server process died"
            return 1
        fi
        if [ $((elapsed % 60)) -eq 0 ] && [ $elapsed -gt 0 ]; then
            log "  Still waiting... ${elapsed}s"
        fi
        sleep 5; elapsed=$((elapsed + 5))
    done
    log "✗ Dump server timeout after ${max_wait}s"
    return 1
}

# =============================================================================
# run_dump_config: Stage 1 + Stage 2 for one config entry
# =============================================================================
run_dump_config() {
    local model_name="$1" n_clusters="$2" dump_gpus="$3" dump_tp="$4" dump_ep="$5" dump_dp="$6" kmeans_gpu="$7" port="$8"
    local model_short num_layers dump_dir centroids_dir

    model_short="$(basename "$model_name")"

    num_layers="${MODEL_NUM_LAYERS[$model_name]}"
    if [[ -z "$num_layers" ]]; then
        echo "ERROR: model '$model_name' not found in MODEL_NUM_LAYERS table."
        echo "       Add it at the top of this script and retry."
        exit 1
    fi

    dump_dir="${KV_BASE}/${model_short}/${TASK}-${DUMP_TOKENS}-tokens"
    centroids_dir="${dump_dir}/c_${n_clusters}"
    mkdir -p "$dump_dir" "$centroids_dir" "$LOGS_DIR/dump/${model_short}"

    LOG_FILE=$(unique_log_path "$LOGS_DIR/dump/${model_short}/dump_c${n_clusters}.log")

    log "=========================================="
    log "Model     : $model_name  (layers=$num_layers)"
    log "Clusters  : $n_clusters"
    log "Dump GPUs : $dump_gpus  (TP=$dump_tp EP=$dump_ep DP=$dump_dp, port=$port)"
    log "KMeans GPU: $kmeans_gpu"
    log "Dump dir  : $dump_dir"
    log "Centroids : $centroids_dir"
    log "=========================================="

    # ------------------------------------------------------------------
    # Stage 1: Dump KV cache
    # ------------------------------------------------------------------
    log "--- Stage 1: KV cache dump ---"

    local all_done=1
    for layer_id in $(seq 0 $((num_layers - 1))); do
        if [ ! -f "$dump_dir/kv_calibration_layer_${layer_id}.pt" ]; then
            all_done=0; break
        fi
    done

    if [ "$all_done" -eq 1 ]; then
        log "✓ Stage 1 skipped: all $num_layers dump files already exist"
    else
        local server_log
        server_log=$(unique_log_path "$LOGS_DIR/dump/${model_short}/stage1_server.log")
        log "Starting dump server... (log: $server_log)"

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
            --port "$port" \
            --trust-remote-code \
            --disable-cuda-graph \
            > "$server_log" 2>&1 &
        local server_pid=$!
        log "Dump server started (PID $server_pid)"

        if ! wait_for_server "$port" "$server_pid"; then
            tail -30 "$server_log" | tee -a "$LOG_FILE"
            kill "$server_pid" 2>/dev/null || true
            return 1
        fi

        local lm_eval_log
        lm_eval_log=$(unique_log_path "$LOGS_DIR/dump/${model_short}/stage1_${TASK}_${DUMP_TOKENS}_lm_eval.log")
        log "Running lm_eval (task=$TASK, --limit=$LIMIT per subtask)..."
        set +e
        CUDA_VISIBLE_DEVICES="" \
        "$LM_EVAL" \
            --model local-completions \
            --tasks "$TASK" \
            --limit "$LIMIT" \
            --model_args "model=${model_name},base_url=http://localhost:${port}/v1/completions,max_model_len=20000,num_concurrent=32,max_retries=1,tokenized_requests=False" \
            2>&1 | tee "$lm_eval_log"
        local lm_exit=${PIPESTATUS[0]}
        set -e
        [ $lm_exit -ne 0 ] && log "WARNING: lm_eval exited $lm_exit (may still have enough tokens)"

        log "Stopping dump server (PID $server_pid)..."
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
        log "✓ Server stopped"

        local missing=0
        for layer_id in $(seq 0 $((num_layers - 1))); do
            [ ! -f "$dump_dir/kv_calibration_layer_${layer_id}.pt" ] && missing=$((missing + 1))
        done
        if [ $missing -gt 0 ]; then
            log "✗ Stage 1 failed: $missing/$num_layers layer files missing"
            return 1
        fi
        log "✓ Stage 1 done: $num_layers layers -> $dump_dir"
    fi

    # ------------------------------------------------------------------
    # Stage 2: K-means
    # ------------------------------------------------------------------
    log "--- Stage 2: K-means (n_clusters=$n_clusters, GPU=$kmeans_gpu) ---"

    local expected=$((num_layers * 2))
    local actual
    actual=$(ls "${centroids_dir}"/*.pt 2>/dev/null | wc -l)

    if [ "$actual" -ge "$expected" ]; then
        log "✓ Stage 2 skipped: $actual/$expected centroid files already exist"
    else
        local kmeans_log
        kmeans_log=$(unique_log_path "$LOGS_DIR/dump/${model_short}/stage2_c${n_clusters}.log")
        log "Running K-means... (log: $kmeans_log)"

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

        actual=$(ls "${centroids_dir}"/*.pt 2>/dev/null | wc -l)
        if [ "$actual" -lt "$expected" ]; then
            log "✗ Stage 2 failed: only $actual/$expected centroid files produced"
            return 1
        fi
        log "✓ Stage 2 done: $expected centroid files -> $centroids_dir"
    fi

    log "✓ All done: $centroids_dir"
}

# =============================================================================
# Main
# =============================================================================
echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] =========================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] dump_centroids.sh"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Configs : ${#MODEL_CONFIGS[@]} entry(s)"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Task    : $TASK  limit: ${LIMIT}/subtask  tokens: $DUMP_TOKENS"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] KV base : $KV_BASE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] =========================================="
echo ""

OVERALL_EXIT=0
for i in "${!MODEL_CONFIGS[@]}"; do
    config="${MODEL_CONFIGS[$i]}"
    IFS='|' read -r model_name n_clusters dump_gpus dump_tp dump_ep dump_dp kmeans_gpu <<< "$config"
    model_name="${model_name// /}"
    n_clusters="${n_clusters// /}"
    dump_gpus="${dump_gpus// /}"
    dump_tp="${dump_tp// /}"
    dump_ep="${dump_ep// /}"
    dump_dp="${dump_dp// /}"
    kmeans_gpu="${kmeans_gpu// /}"

    port=$((BASE_PORT + i))
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$((i+1))/${#MODEL_CONFIGS[@]}] $(basename "$model_name")  clusters=$n_clusters  port=$port"

    if ! run_dump_config "$model_name" "$n_clusters" "$dump_gpus" "$dump_tp" "$dump_ep" "$dump_dp" "$kmeans_gpu" "$port"; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ Failed: $(basename "$model_name")"
        OVERALL_EXIT=1
    fi
done

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All done. Exit: $OVERALL_EXIT"
exit $OVERALL_EXIT
