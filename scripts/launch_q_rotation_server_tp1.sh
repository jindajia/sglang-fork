#!/usr/bin/env bash
set -euo pipefail

# Step 1 for manual TP=1 Q-rotation evaluation.
# Run this in one terminal and keep it alive.
#
# Modes:
#   MODE=without_qr  -> HADAMARD=0 and no Q rotation
#   MODE=with_qr     -> HADAMARD=0 and enable SGLANG_Q_ROTATION_PATH

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Thinking-2507}"
MODE="${MODE:-without_qr}"

GPU="${GPU:-0}"
PORT="${PORT:-31001}"
TP_SIZE="${TP_SIZE:-1}"
DP_SIZE="${DP_SIZE:-1}"

KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-int4}"
PREFILL_ATTENTION_BACKEND="${PREFILL_ATTENTION_BACKEND:-}"
DECODE_ATTENTION_BACKEND="${DECODE_ATTENTION_BACKEND:-triton}"
SAMPLING_BACKEND="${SAMPLING_BACKEND:-flashinfer}"

MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-32}"
MAX_QUEUED_REQUESTS="${MAX_QUEUED_REQUESTS:-64}"
PAGE_SIZE="${PAGE_SIZE:-128}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-4096}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.8}"
PP_MAX_MICRO_BATCH_SIZE="${PP_MAX_MICRO_BATCH_SIZE:-32}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"

Q_ROTATION_PATH="${Q_ROTATION_PATH:-/data/shared/charlie/sglangfork/q_rotation_layer.pt}"
Q_ROTATION_COMPUTE_DTYPE="${Q_ROTATION_COMPUTE_DTYPE:-float32}"

MODEL_SHORT_NAME="$(basename "${MODEL_PATH}")"
RUN_ID="${RUN_ID:-manual}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/eval_logs/q_rotation_tp1/${MODEL_SHORT_NAME}/${RUN_ID}}"
SERVER_LOG="${LOG_DIR}/${MODE}_server_port${PORT}.log"

LOCAL_PYTHONPATH="${REPO_ROOT}/python:${REPO_ROOT}/tore-eval/src${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${LOG_DIR}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

validate_env() {
    if [[ "${TP_SIZE}" != "1" ]]; then
        echo "This helper is intentionally TP=1 only. Got TP_SIZE=${TP_SIZE}." >&2
        exit 1
    fi

    case "${MODE}" in
        without_qr|with_qr) ;;
        *)
            echo "Unsupported MODE='${MODE}'. Use without_qr or with_qr." >&2
            exit 1
            ;;
    esac

    if [[ "${MODE}" == "with_qr" ]] && [[ ! -f "${Q_ROTATION_PATH}" ]]; then
        echo "Q_ROTATION_PATH does not exist: ${Q_ROTATION_PATH}" >&2
        exit 1
    fi

    if [[ -z "${PREFILL_ATTENTION_BACKEND}" ]]; then
        if [[ "${KV_CACHE_DTYPE}" == "int4" || "${KV_CACHE_DTYPE}" == "int8" ]]; then
            PREFILL_ATTENTION_BACKEND="fa3"
        else
            PREFILL_ATTENTION_BACKEND="triton"
        fi
    fi

    if [[ "${KV_CACHE_DTYPE}" == "int4" || "${KV_CACHE_DTYPE}" == "int8" ]]; then
        if [[ "${PREFILL_ATTENTION_BACKEND}" != "fa3" ]]; then
            echo "KV_CACHE_DTYPE=${KV_CACHE_DTYPE} requires PREFILL_ATTENTION_BACKEND=fa3." >&2
            exit 1
        fi
        if [[ "${DECODE_ATTENTION_BACKEND}" != "triton" ]]; then
            echo "KV_CACHE_DTYPE=${KV_CACHE_DTYPE} requires DECODE_ATTENTION_BACKEND=triton." >&2
            exit 1
        fi
    fi
}

main() {
    validate_env

    local -a cmd=(
        "${PYTHON_BIN}" -m sglang.launch_server
        --model-path "${MODEL_PATH}"
        --max-running-requests "${MAX_RUNNING_REQUESTS}"
        --max-queued-requests "${MAX_QUEUED_REQUESTS}"
        --page-size "${PAGE_SIZE}"
        --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}"
        --mem-fraction-static "${MEM_FRACTION_STATIC}"
        --pp-max-micro-batch-size "${PP_MAX_MICRO_BATCH_SIZE}"
        --kv-cache-dtype "${KV_CACHE_DTYPE}"
        --prefill-attention-backend "${PREFILL_ATTENTION_BACKEND}"
        --decode-attention-backend "${DECODE_ATTENTION_BACKEND}"
        --sampling-backend "${SAMPLING_BACKEND}"
        --tensor-parallel-size "${TP_SIZE}"
        --data-parallel-size "${DP_SIZE}"
        --host 0.0.0.0
        --port "${PORT}"
    )

    if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
        cmd+=(--trust-remote-code)
    fi

    log "Repo root: ${REPO_ROOT}"
    log "Using python: ${PYTHON_BIN}"
    log "Mode: ${MODE}"
    log "Model: ${MODEL_PATH}"
    log "GPU: ${GPU}"
    log "Port: ${PORT}"
    log "KV cache dtype: ${KV_CACHE_DTYPE}"
    log "Prefill backend: ${PREFILL_ATTENTION_BACKEND}"
    log "Decode backend: ${DECODE_ATTENTION_BACKEND}"
    log "Server log: ${SERVER_LOG}"
    if [[ "${MODE}" == "with_qr" ]]; then
        log "Q rotation path: ${Q_ROTATION_PATH}"
    fi
    log "Starting server in foreground. Open another terminal to run eval."
    log "Health check: curl http://127.0.0.1:${PORT}/health"

    if [[ "${MODE}" == "with_qr" ]]; then
        env \
            PYTHONPATH="${LOCAL_PYTHONPATH}" \
            CUDA_VISIBLE_DEVICES="${GPU}" \
            HADAMARD=0 \
            ROTATE_V=0 \
            SGLANG_Q_ROTATION_PATH="${Q_ROTATION_PATH}" \
            SGLANG_Q_ROTATION_COMPUTE_DTYPE="${Q_ROTATION_COMPUTE_DTYPE}" \
            "${cmd[@]}" 2>&1 | tee "${SERVER_LOG}"
    else
        env -u SGLANG_Q_ROTATION_PATH -u SGLANG_Q_ROTATION_COMPUTE_DTYPE \
            PYTHONPATH="${LOCAL_PYTHONPATH}" \
            CUDA_VISIBLE_DEVICES="${GPU}" \
            HADAMARD=0 \
            ROTATE_V=0 \
            "${cmd[@]}" 2>&1 | tee "${SERVER_LOG}"
    fi
}

main "$@"
