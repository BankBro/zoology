#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/lyj/mnt/project/zoology"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
BACKEND="${BACKEND:-accel}"
DMODEL="${DMODEL:-128}"
LR="${LR:-1e-3}"
MAX_EPOCHS="${MAX_EPOCHS:-32}"
PROJECT="${PROJECT:-flash_vqg_mqar}"
ENTITY="${ENTITY:-scu-mclab}"
CACHE_DIR="${CACHE_DIR:-./data/flash_vqg}"
LAUNCH_ID_PREFIX="${LAUNCH_ID_PREFIX:-flash-vqg-e0}"
TRAIN_BATCH_ORDERS="${TRAIN_BATCH_ORDERS:-sequential,global_shuffle,balanced_interleave}"
METRICS_WHITE_LIST_FILE="${METRICS_WHITE_LIST_FILE:-${ROOT_DIR}/zoology/experiments/flash_vqg/metrics_white_lists/e0.yaml}"
METRICS_WHITE_LIST="${METRICS_WHITE_LIST:-}"

echo "==> Running E0 samplers [${TRAIN_BATCH_ORDERS}] on GPU ${GPU_ID}"
CMD=(
  "${PYTHON_BIN}" -m zoology.experiments.flash_vqg.run_flash_vqg_suite
  --flash-only \
  --logger-backend swanlab \
  --analysis remote \
  --backend "${BACKEND}" \
  --block-len 32 \
  --dmodels "${DMODEL}" \
  --learning-rates "${LR}" \
  --max-epochs "${MAX_EPOCHS}" \
  --train-batch-order "${TRAIN_BATCH_ORDERS}" \
  --cache-dir "${CACHE_DIR}" \
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
