#!/usr/bin/env bash
set -euo pipefail

# Step 2 for manual TP=1 Q-rotation evaluation.
# Run this after launch_q_rotation_server_tp1.sh is already running.
#
# Typical usage:
#   MODE_TAG=without_qr bash scripts/run_q_rotation_eval_tp1.sh
#   MODE_TAG=with_qr    bash scripts/run_q_rotation_eval_tp1.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Thinking-2507}"
PRESET_NAME="${PRESET_NAME:-humaneval_think}"
MODE_TAG="${MODE_TAG:-without_qr}"

PORT="${PORT:-31001}"
BASE_URL="${BASE_URL:-http://127.0.0.1:${PORT}/v1}"
HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:${PORT}/health}"

NUM_WORKERS="${NUM_WORKERS:-32}"
REPEAT="${REPEAT:-1}"
MAX_WAIT_SECS="${MAX_WAIT_SECS:-1800}"

MODEL_SHORT_NAME="$(basename "${MODEL_PATH}")"
RUN_ID="${RUN_ID:-manual}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/eval_results/q_rotation_tp1/${MODEL_SHORT_NAME}/${PRESET_NAME}/${RUN_ID}/${MODE_TAG}}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/eval_logs/q_rotation_tp1/${MODEL_SHORT_NAME}/${PRESET_NAME}/${RUN_ID}}"

LOCAL_PYTHONPATH="${REPO_ROOT}/python:${REPO_ROOT}/tore-eval/src${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

validate_env() {
    if [[ "${REPEAT}" -lt 1 ]]; then
        echo "REPEAT must be >= 1, got ${REPEAT}." >&2
        exit 1
    fi
}

wait_for_server() {
    local elapsed=0
    while [[ "${elapsed}" -lt "${MAX_WAIT_SECS}" ]]; do
        if curl -s "${HEALTH_URL}" > /dev/null 2>&1; then
            log "Server is healthy after ${elapsed}s."
            return 0
        fi

        if [[ "${elapsed}" -gt 0 ]] && (( elapsed % 30 == 0 )); then
            log "Waiting for ${HEALTH_URL} ... ${elapsed}s"
        fi

        sleep 5
        elapsed=$((elapsed + 5))
    done

    echo "Timed out waiting for server health at ${HEALTH_URL}" >&2
    return 1
}

write_summary() {
    RESULTS_DIR="${RESULTS_DIR}" "${PYTHON_BIN}" - <<'PY'
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


results_dir = Path(os.environ["RESULTS_DIR"])
run_files = sorted(glob.glob(str(results_dir / "run*" / "results.jsonl")))

summary = {"num_runs": len(run_files), "metrics_per_run": []}
for run_file in run_files:
    summary["metrics_per_run"].append(load_metrics(run_file))

numeric_keys = sorted(
    {
        key
        for metrics in summary["metrics_per_run"]
        for key, value in metrics.items()
        if isinstance(value, (int, float))
    }
)

summary["mean_metrics"] = {
    key: sum(metrics[key] for metrics in summary["metrics_per_run"] if key in metrics)
    / sum(1 for metrics in summary["metrics_per_run"] if key in metrics)
    for key in numeric_keys
}

summary_path = results_dir / "summary.json"
summary_path.write_text(json.dumps(summary, indent=2))
print(f"Saved summary to {summary_path}")
for key, value in summary["mean_metrics"].items():
    print(f"{key}: {value}")
PY
}

main() {
    validate_env

    log "Repo root: ${REPO_ROOT}"
    log "Using python: ${PYTHON_BIN}"
    log "Mode tag: ${MODE_TAG}"
    log "Model: ${MODEL_PATH}"
    log "Preset: ${PRESET_NAME}"
    log "Base URL: ${BASE_URL}"
    log "Results dir: ${RESULTS_DIR}"

    wait_for_server

    for run_idx in $(seq 1 "${REPEAT}"); do
        run_dir="${RESULTS_DIR}/run${run_idx}"
        eval_log="${LOG_DIR}/${MODE_TAG}_run${run_idx}.log"
        mkdir -p "${run_dir}"

        log "Running eval ${run_idx}/${REPEAT} -> ${run_dir}"
        (
            cd "${REPO_ROOT}" && \
            PYTHONPATH="${LOCAL_PYTHONPATH}" \
            "${PYTHON_BIN}" -m tore_eval.eval \
                --framework preset \
                --preset_name "${PRESET_NAME}" \
                --model_name_or_path "${MODEL_PATH}" \
                --provider custom \
                --base_url "${BASE_URL}" \
                --api_key "" \
                --num_workers "${NUM_WORKERS}" \
                --log_file "${run_dir}/samples.jsonl" \
                --loggers "{\"local\": {\"output_dir\": \"${run_dir}\"}}"
        ) 2>&1 | tee "${eval_log}"
    done

    write_summary
}

main "$@"
