#!/usr/bin/env bash
set -euo pipefail

# Per-layer sensitivity analysis for int4+Hadamard+QR.
#
# For each layer L, launches a fp16 server where ONLY layer L undergoes
# simulated int4+H+QR quantization (all other layers stay fp16).
# This isolates the quantization impact of each individual layer.
#
# Usage (on 002):
#   bash scripts/run_layer_sensitivity.sh
#
# Optionally override:
#   GPU=0 START_LAYER=10 END_LAYER=20 bash scripts/run_layer_sensitivity.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/charlie/miniconda3/envs/sglangfork/bin/python}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Thinking-2507}"

GPU="${GPU:-0}"
PORT="${PORT:-38001}"

NUM_LAYERS="${NUM_LAYERS:-36}"
START_LAYER="${START_LAYER:-0}"
END_LAYER="${END_LAYER:-35}"

# SIM_MODE:
#   "h_qr"       - fp16 + simulated int4+H+QR on target layer only
#   "h_only"     - fp16 + simulated int4+H on target layer only (no QR)
#   "int4_h_plus_qr" - real int4+H on ALL layers, QR only on target layer
SIM_MODE="${SIM_MODE:-h_qr}"
Q_ROTATION_PATH="${Q_ROTATION_PATH:-/data/shared/charlie/sglangfork/q_rotation_layer_second_moment_damp01.pt}"
PER_LAYER_ROTATION_DIR="${PER_LAYER_ROTATION_DIR:-/data/shared/charlie/sglangfork/per_layer_rotations}"

MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-32}"
MAX_QUEUED_REQUESTS="${MAX_QUEUED_REQUESTS:-64}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.8}"
NUM_WORKERS="${NUM_WORKERS:-32}"

PRESET_NAME="${PRESET_NAME:-gpqa_think}"
REPEAT="${REPEAT:-1}"

MODEL_SHORT_NAME="$(basename "${MODEL_PATH}")"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-layer_sensitivity}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/eval_results/hadamard_qr_tp1/${MODEL_SHORT_NAME}/${PRESET_NAME}/${EXPERIMENT_TAG}}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/eval_logs/hadamard_qr_tp1/${MODEL_SHORT_NAME}/${EXPERIMENT_TAG}}"

LOCAL_PYTHONPATH="${REPO_ROOT}/python:${REPO_ROOT}/tore-eval/src${PYTHONPATH:+:${PYTHONPATH}}"
HEALTH_URL="http://127.0.0.1:${PORT}/health"
BASE_URL="http://127.0.0.1:${PORT}/v1"
MAX_WAIT_SECS="${MAX_WAIT_SECS:-600}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

wait_for_server() {
    local elapsed=0
    while [[ "${elapsed}" -lt "${MAX_WAIT_SECS}" ]]; do
        if curl -s "${HEALTH_URL}" > /dev/null 2>&1; then
            log "Server healthy after ${elapsed}s."
            return 0
        fi
        if (( elapsed > 0 && elapsed % 30 == 0 )); then
            log "Waiting for server... ${elapsed}s"
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    log "ERROR: Server did not become healthy within ${MAX_WAIT_SECS}s"
    return 1
}

kill_server() {
    local pids
    pids=$(pgrep -f "sglang.launch_server.*--port ${PORT}" 2>/dev/null || true)
    if [[ -n "${pids}" ]]; then
        log "Killing server (PIDs: ${pids})..."
        echo "${pids}" | xargs kill -9 2>/dev/null || true
        sleep 3
        pgrep -f "sglang.launch_server.*--port ${PORT}" | xargs kill -9 2>/dev/null || true
        sleep 2
    fi
}

launch_server() {
    local sim_layer="$1"
    local server_log="$2"

    kill_server

    local -a sim_env=(
        PYTHONPATH="${LOCAL_PYTHONPATH}"
        CUDA_VISIBLE_DEVICES="${GPU}"
        HF_HOME="/data/shared/huggingface"
        HADAMARD=1
        HADAMARD_ORDER=16
    )
    local kv_dtype="auto"

    if [[ "${SIM_MODE}" == "int4_h_plus_qr" ]]; then
        local rotation_file="${PER_LAYER_ROTATION_DIR}/q_rotation_layer${sim_layer}.pt"
        if [[ ! -f "${rotation_file}" ]]; then
            log "ERROR: Per-layer rotation file not found: ${rotation_file}"
            log "Run: python scripts/generate_per_layer_rotations.py"
            return 1
        fi
        log "Launching int4+H server with QR on layer ${sim_layer} only"
        sim_env+=(
            SGLANG_Q_ROTATION_PATH="${rotation_file}"
            SGLANG_Q_ROTATION_COMPUTE_DTYPE=float32
            ROTATE_V=0
        )
        kv_dtype="int4"
    elif [[ "${SIM_MODE}" == "h_qr" ]]; then
        log "Launching fp16 server with int4+H+QR simulation on layer ${sim_layer}"
        sim_env+=(SGLANG_QUANT_SIM_LAYERS="${sim_layer}")
        sim_env+=(SGLANG_QUANT_SIM_ROTATION_PATH="${Q_ROTATION_PATH}")
    else
        log "Launching fp16 server with int4+H (no QR) simulation on layer ${sim_layer}"
        sim_env+=(SGLANG_QUANT_SIM_LAYERS="${sim_layer}")
    fi

    env -u SGLANG_Q_ROTATION_PATH -u SGLANG_Q_ROTATION_COMPUTE_DTYPE \
        -u SGLANG_QUANT_SIM_LAYERS -u SGLANG_QUANT_SIM_ROTATION_PATH \
        "${sim_env[@]}" \
        "${PYTHON_BIN}" -m sglang.launch_server \
        --model-path "${MODEL_PATH}" \
        --max-running-requests "${MAX_RUNNING_REQUESTS}" \
        --max-queued-requests "${MAX_QUEUED_REQUESTS}" \
        --page-size 128 \
        --chunked-prefill-size 4096 \
        --mem-fraction-static "${MEM_FRACTION_STATIC}" \
        --pp-max-micro-batch-size 32 \
        --kv-cache-dtype "${kv_dtype}" \
        --prefill-attention-backend fa3 \
        --decode-attention-backend triton \
        --sampling-backend flashinfer \
        --tensor-parallel-size 1 \
        --data-parallel-size 1 \
        --host 0.0.0.0 \
        --port "${PORT}" \
        --trust-remote-code \
        > "${server_log}" 2>&1 &

    log "Server PID: $!"
}

run_eval() {
    local results_dir="$1"
    local eval_log="$2"

    for run_idx in $(seq 1 "${REPEAT}"); do
        local run_dir="${results_dir}/run${run_idx}"
        mkdir -p "${run_dir}"

        log "  Eval run ${run_idx}/${REPEAT} -> ${run_dir}"
        (
            cd "${REPO_ROOT}" && \
            PYTHONPATH="${LOCAL_PYTHONPATH}" \
            HF_HOME="/data/shared/huggingface" \
            "${PYTHON_BIN}" -m tore_eval.eval \
                --framework preset \
                --preset_name "${PRESET_NAME}" \
                --model_name_or_path "${MODEL_PATH}" \
                --provider custom \
                --base_url "${BASE_URL}" \
                --api_key "" \
                --num_workers "${NUM_WORKERS}" \
                --output_dir "${run_dir}" \
                --log_file "${run_dir}/samples.jsonl" \
                --loggers "{\"local\": {\"output_dir\": \"${run_dir}\"}}"
        ) 2>&1 | tee "${eval_log}"
    done
}

write_summary() {
    local results_dir="$1"
    RESULTS_DIR="${results_dir}" "${PYTHON_BIN}" - <<'PY'
import glob, json, os
from pathlib import Path

def load_metrics(f):
    for line in open(f):
        line = line.strip()
        if not line: continue
        m = json.loads(line).get("metrics")
        if isinstance(m, dict): return m
    raise ValueError(f"No metrics in {f}")

d = Path(os.environ["RESULTS_DIR"])
runs = sorted(glob.glob(str(d / "run*" / "results.jsonl")))
s = {"num_runs": len(runs), "metrics_per_run": [load_metrics(r) for r in runs]}
keys = sorted({k for m in s["metrics_per_run"] for k, v in m.items() if isinstance(v, (int, float))})
s["mean_metrics"] = {k: sum(m[k] for m in s["metrics_per_run"] if k in m) / sum(1 for m in s["metrics_per_run"] if k in m) for k in keys}
(d / "summary.json").write_text(json.dumps(s, indent=2))
print(f"Summary: {s['mean_metrics']}")
PY
}

main() {
    local sim_tag
    case "${SIM_MODE}" in
        h_qr)              sim_tag="quant_sim" ;;
        h_only)            sim_tag="quant_sim_h_only" ;;
        int4_h_plus_qr)    sim_tag="int4_h_plus_qr" ;;
        *) echo "Unknown SIM_MODE: ${SIM_MODE}" >&2; exit 1 ;;
    esac

    log "=== Per-layer sensitivity analysis (${SIM_MODE}) ==="
    log "Model: ${MODEL_PATH} (${NUM_LAYERS} layers)"
    log "Layers: ${START_LAYER} to ${END_LAYER}"
    log "GPU: ${GPU}, Port: ${PORT}"
    log "Sim mode: ${SIM_MODE} (results subdir: ${sim_tag})"
    if [[ "${SIM_MODE}" == "h_qr" ]]; then
        log "Q rotation: ${Q_ROTATION_PATH}"
    fi
    log "Results: ${RESULTS_ROOT}"
    log ""

    mkdir -p "${LOG_ROOT}"

    for layer_id in $(seq "${START_LAYER}" "${END_LAYER}"); do
        local tag="layer_${layer_id}"
        local results_dir="${RESULTS_ROOT}/${tag}/${sim_tag}"
        local server_log="${LOG_ROOT}/${tag}_${SIM_MODE}_server.log"
        local eval_log="${LOG_ROOT}/${tag}_${SIM_MODE}_eval.log"

        if [[ -f "${results_dir}/summary.json" ]]; then
            log "Layer ${layer_id}: summary.json exists, skipping."
            continue
        fi

        mkdir -p "${results_dir}" "$(dirname "${server_log}")"

        log "========== Layer ${layer_id}/${END_LAYER} =========="

        launch_server "${layer_id}" "${server_log}"

        if ! wait_for_server; then
            log "ERROR: Server failed for layer ${layer_id}, skipping."
            kill_server
            continue
        fi

        local start_ts
        start_ts=$(date +%s)

        run_eval "${results_dir}" "${eval_log}"
        write_summary "${results_dir}"

        local end_ts elapsed_min
        end_ts=$(date +%s)
        elapsed_min=$(( (end_ts - start_ts) / 60 ))
        log "Layer ${layer_id} done in ${elapsed_min} min."

        kill_server
        log ""
    done

    log "=== All layers done. Results in ${RESULTS_ROOT} ==="

    log ""
    log "=== Summary of all layers (${SIM_MODE}) ==="
    for layer_id in $(seq "${START_LAYER}" "${END_LAYER}"); do
        local summary="${RESULTS_ROOT}/layer_${layer_id}/${sim_tag}/summary.json"
        if [[ -f "${summary}" ]]; then
            local score
            score=$("${PYTHON_BIN}" -c "import json; print(json.load(open('${summary}'))['mean_metrics'].get('gpqa/score', 'N/A'))")
            log "  Layer ${layer_id}: gpqa/score = ${score}"
        fi
    done
}

main "$@"
