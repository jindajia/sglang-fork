#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

TENSOR_PATH="${TENSOR_PATH:-/data/shared/charlie/sglangfork}"
GROUPING="${GROUPING:-layer}"
ACCUM_DTYPE="${ACCUM_DTYPE:-float64}"
OUTPUT_PATH="${OUTPUT_PATH:-${TENSOR_PATH}/q_statistics_${GROUPING}.pt}"

CMD=(
  python
  "${SCRIPT_DIR}/compute_q_covariance.py"
  --tensor-path "${TENSOR_PATH}"
  --output-path "${OUTPUT_PATH}"
  --grouping "${GROUPING}"
  --accum-dtype "${ACCUM_DTYPE}"
)

if [[ -n "${NUM_LAYERS:-}" ]]; then
  CMD+=(--num-layers "${NUM_LAYERS}")
fi

if [[ -n "${NUM_KV_HEADS:-}" ]]; then
  CMD+=(--num-kv-heads "${NUM_KV_HEADS}")
fi

printf 'Running: '
printf '%q ' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
