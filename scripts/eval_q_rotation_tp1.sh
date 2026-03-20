#!/usr/bin/env bash
set -euo pipefail

# Minimal TP=1 A/B evaluation for offline Q rotation.
# It runs the same preset twice:
#   1. without_qr: HADAMARD=0 and no SGLANG_Q_ROTATION_PATH
#   2. with_qr:    HADAMARD=0 and SGLANG_Q_ROTATION_PATH enabled

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Thinking-2507}"
PRESET_NAME="${PRESET_NAME:-humaneval_think}"
REPEAT="${REPEAT:-1}"

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
NUM_WORKERS="${NUM_WORKERS:-32}"
PAGE_SIZE="${PAGE_SIZE:-128}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-4096}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.8}"
PP_MAX_MICRO_BATCH_SIZE="${PP_MAX_MICRO_BATCH_SIZE:-32}"
COOLDOWN_SECS="${COOLDOWN_SECS:-15}"
MAX_WAIT_SECS="${MAX_WAIT_SECS:-1800}"

RUN_MODES="${RUN_MODES:-without_qr,with_qr}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"

Q_ROTATION_PATH="${Q_ROTATION_PATH:-/data/shared/charlie/sglangfork/q_rotation_layer.pt}"
Q_ROTATION_COMPUTE_DTYPE="${Q_ROTATION_COMPUTE_DTYPE:-float32}"

MODEL_SHORT_NAME="$(basename "${MODEL_PATH}")"
RUN_ID="${RUN_ID:-$(date '+%Y%m%d_%H%M%S')}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/eval_results/q_rotation_tp1/${MODEL_SHORT_NAME}/${PRESET_NAME}/${RUN_ID}}"
LOGS_ROOT="${LOGS_ROOT:-${REPO_ROOT}/eval_logs/q_rotation_tp1/${MODEL_SHORT_NAME}/${PRESET_NAME}/${RUN_ID}}"

LOCAL_PYTHONPATH="${REPO_ROOT}/python:${REPO_ROOT}/tore-eval/src${PYTHONPATH:+:${PYTHONPATH}}"
SERVER_PID=""

mkdir -p "${RESULTS_ROOT}" "${LOGS_ROOT}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

cleanup_server() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        log "Stopping server (PID: ${SERVER_PID})..."
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
    SERVER_PID=""
}

trap cleanup_server EXIT

validate_env() {
    if [[ "${TP_SIZE}" != "1" ]]; then
        echo "This helper is intentionally TP=1 only. Got TP_SIZE=${TP_SIZE}." >&2
        exit 1
    fi

    if [[ "${REPEAT}" -lt 1 ]]; then
        echo "REPEAT must be >= 1, got ${REPEAT}." >&2
        exit 1
    fi

    IFS=',' read -r -a modes <<< "${RUN_MODES}"
    for mode in "${modes[@]}"; do
        case "${mode}" in
            without_qr|with_qr) ;;
            *)
                echo "Unsupported mode '${mode}'. Use RUN_MODES=without_qr,with_qr." >&2
                exit 1
                ;;
        esac
    done

    if [[ "${RUN_MODES}" == *"with_qr"* ]] && [[ ! -f "${Q_ROTATION_PATH}" ]]; then
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

wait_for_server_ready() {
    local server_log="$1"
    local elapsed=0

    while [[ "${elapsed}" -lt "${MAX_WAIT_SECS}" ]]; do
        if curl -s "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
            log "Server ready after ${elapsed}s."
            return 0
        fi

        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            log "Server process exited early. Tail of ${server_log}:"
            tail -n 80 "${server_log}" || true
            return 1
        fi

        if [[ "${elapsed}" -gt 0 ]] && (( elapsed % 30 == 0 )); then
            log "Still waiting for server... ${elapsed}s"
        fi

        sleep 5
        elapsed=$((elapsed + 5))
    done

    log "Timed out waiting for server after ${MAX_WAIT_SECS}s. Tail of ${server_log}:"
    tail -n 80 "${server_log}" || true
    return 1
}

start_server() {
    local mode="$1"
    local server_log="${LOGS_ROOT}/${mode}_server.log"
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

    cleanup_server
    log "Starting ${mode} server on port ${PORT}..."

    if [[ "${mode}" == "with_qr" ]]; then
        PYTHONPATH="${LOCAL_PYTHONPATH}" \
        CUDA_VISIBLE_DEVICES="${GPU}" \
        HADAMARD=0 \
        ROTATE_V=0 \
        SGLANG_Q_ROTATION_PATH="${Q_ROTATION_PATH}" \
        SGLANG_Q_ROTATION_COMPUTE_DTYPE="${Q_ROTATION_COMPUTE_DTYPE}" \
        "${cmd[@]}" > "${server_log}" 2>&1 &
    else
        env -u SGLANG_Q_ROTATION_PATH -u SGLANG_Q_ROTATION_COMPUTE_DTYPE \
            PYTHONPATH="${LOCAL_PYTHONPATH}" \
            CUDA_VISIBLE_DEVICES="${GPU}" \
            HADAMARD=0 \
            ROTATE_V=0 \
            "${cmd[@]}" > "${server_log}" 2>&1 &
    fi

    SERVER_PID=$!
    log "Server PID: ${SERVER_PID}"
    wait_for_server_ready "${server_log}"
}

run_eval_mode() {
    local mode="$1"
    local mode_root="${RESULTS_ROOT}/${mode}"
    mkdir -p "${mode_root}"

    start_server "${mode}"

    for run_idx in $(seq 1 "${REPEAT}"); do
        local run_dir="${mode_root}/run${run_idx}"
        local eval_log="${LOGS_ROOT}/${mode}_run${run_idx}.log"
        mkdir -p "${run_dir}"

        log "Running ${mode} eval ${run_idx}/${REPEAT} -> ${run_dir}"
        (
            cd "${REPO_ROOT}" && \
            PYTHONPATH="${LOCAL_PYTHONPATH}" \
            "${PYTHON_BIN}" -m tore_eval.eval \
                --framework preset \
                --preset_name "${PRESET_NAME}" \
                --model_name_or_path "${MODEL_PATH}" \
                --provider custom \
                --base_url "http://127.0.0.1:${PORT}/v1" \
                --api_key "" \
                --num_workers "${NUM_WORKERS}" \
                --log_file "${run_dir}/samples.jsonl" \
                --loggers "{\"local\": {\"output_dir\": \"${run_dir}\"}}"
        ) 2>&1 | tee "${eval_log}"
    done

    cleanup_server
    if [[ "${COOLDOWN_SECS}" -gt 0 ]]; then
        log "Cooling down ${COOLDOWN_SECS}s before next mode..."
        sleep "${COOLDOWN_SECS}"
    fi
}

write_summary() {
    RESULTS_ROOT="${RESULTS_ROOT}" RUN_MODES="${RUN_MODES}" "${PYTHON_BIN}" - <<'PY'
import glob
import json
import os
from pathlib import Path


def load_metrics(results_file: str) -> dict:
    with open(results_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            metrics = obj.get("metrics")
            if isinstance(metrics, dict):
                return metrics
    raise ValueError(f"No metrics found in {results_file}")


def mean_numeric_metrics(metric_dicts: list[dict]) -> dict:
    keys = sorted(
        {
            key
            for metrics in metric_dicts
            for key, value in metrics.items()
            if isinstance(value, (int, float))
        }
    )
    return {
        key: sum(metrics[key] for metrics in metric_dicts if key in metrics)
        / sum(1 for metrics in metric_dicts if key in metrics)
        for key in keys
    }


results_root = Path(os.environ["RESULTS_ROOT"])
modes = [mode.strip() for mode in os.environ["RUN_MODES"].split(",") if mode.strip()]
summary: dict[str, dict] = {}

for mode in modes:
    run_files = sorted(glob.glob(str(results_root / mode / "run*" / "results.jsonl")))
    if not run_files:
        summary[mode] = {"error": "no results.jsonl files found"}
        continue

    metrics_per_run = [load_metrics(path) for path in run_files]
    summary[mode] = {
        "num_runs": len(metrics_per_run),
        "metrics_per_run": metrics_per_run,
        "mean_metrics": mean_numeric_metrics(metrics_per_run),
    }

comparison = {"modes": summary}

if "without_qr" in summary and "with_qr" in summary:
    baseline = summary["without_qr"].get("mean_metrics", {})
    rotated = summary["with_qr"].get("mean_metrics", {})
    delta = {}
    for key in sorted(set(baseline) & set(rotated)):
        delta[key] = rotated[key] - baseline[key]
    comparison["delta_with_minus_without"] = delta

summary_path = results_root / "comparison.json"
summary_path.write_text(json.dumps(comparison, indent=2))

print(f"Saved comparison summary to {summary_path}")
for mode, mode_summary in summary.items():
    print(f"[{mode}]")
    if "mean_metrics" not in mode_summary:
        print(f"  error: {mode_summary['error']}")
        continue
    for key, value in sorted(mode_summary["mean_metrics"].items()):
        print(f"  {key}: {value}")

delta = comparison.get("delta_with_minus_without")
if delta:
    print("[with_qr - without_qr]")
    for key, value in sorted(delta.items()):
        print(f"  {key}: {value}")
PY
}

main() {
    validate_env

    log "Repo root: ${REPO_ROOT}"
    log "Using python: ${PYTHON_BIN}"
    log "Model: ${MODEL_PATH}"
    log "Preset: ${PRESET_NAME}"
    log "Modes: ${RUN_MODES}"
    log "GPU: ${GPU}"
    log "KV cache dtype: ${KV_CACHE_DTYPE}"
    log "Prefill backend: ${PREFILL_ATTENTION_BACKEND}"
    log "Decode backend: ${DECODE_ATTENTION_BACKEND}"
    log "Results: ${RESULTS_ROOT}"
    log "Logs: ${LOGS_ROOT}"
    if [[ "${RUN_MODES}" == *"with_qr"* ]]; then
        log "Q rotation path: ${Q_ROTATION_PATH}"
    fi

    IFS=',' read -r -a modes <<< "${RUN_MODES}"
    for mode in "${modes[@]}"; do
        run_eval_mode "${mode}"
    done

    write_summary
    log "Finished TP=1 Q-rotation A/B test."
}

main "$@"
