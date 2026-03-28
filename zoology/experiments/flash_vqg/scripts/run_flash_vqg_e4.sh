#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/lyj/mnt/project/zoology"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/lyj/miniconda3/envs/flash-vqg/bin/python}"
GPU_ID="${GPU_ID:-0}"
CHECKPOINT_LAUNCH_ID="${CHECKPOINT_LAUNCH_ID:-}"
CHECKPOINT_RUN_ID="${CHECKPOINT_RUN_ID:-}"
PROJECT="${PROJECT:-flash_vqg_mqar}"
ENTITY="${ENTITY:-scu-mclab}"
LAUNCH_ID_PREFIX="${LAUNCH_ID_PREFIX:-flash-vqg-e4}"
METRICS_WHITE_LIST_FILE="${METRICS_WHITE_LIST_FILE:-${ROOT_DIR}/zoology/experiments/flash_vqg/metrics_white_lists/e4.yaml}"
METRICS_WHITE_LIST="${METRICS_WHITE_LIST:-}"

if [[ -z "${CHECKPOINT_LAUNCH_ID}" || -z "${CHECKPOINT_RUN_ID}" ]]; then
  echo "需要提供 CHECKPOINT_LAUNCH_ID 和 CHECKPOINT_RUN_ID." >&2
  echo "示例:" >&2
  echo "  CHECKPOINT_LAUNCH_ID=flash-vqg-e1-xxxx CHECKPOINT_RUN_ID=flash_vqg_h2_accel-... $0" >&2
  exit 1
fi

echo "==> Running E4-A eval-only from ${CHECKPOINT_LAUNCH_ID}/${CHECKPOINT_RUN_ID} on GPU ${GPU_ID}"
CMD=(
  "${PYTHON_BIN}" -m zoology.experiments.flash_vqg.run_flash_vqg_suite
  --eval-only e4a \
  --logger-backend swanlab \
  --analysis remote \
  --checkpoint-launch-id "${CHECKPOINT_LAUNCH_ID}" \
  --checkpoint-run-id "${CHECKPOINT_RUN_ID}" \
  --metrics-white-list-file "${METRICS_WHITE_LIST_FILE}" \
  --project "${PROJECT}" \
  --entity "${ENTITY}" \
  --launch-id-prefix "${LAUNCH_ID_PREFIX}" \
  --gpus "${GPU_ID}"
)

if [[ -n "${METRICS_WHITE_LIST}" ]]; then
  CMD+=(--metrics-white-list "${METRICS_WHITE_LIST}")
fi

"${CMD[@]}"
