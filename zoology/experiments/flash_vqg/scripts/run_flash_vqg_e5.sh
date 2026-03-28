#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/lyj/mnt/project/zoology"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/lyj/miniconda3/envs/flash-vqg/bin/python}"
GPU_ID="${GPU_ID:-0}"
BACKEND="${BACKEND:-accel}"
DMODEL="${DMODEL:-128}"
LR="${LR:-1e-3}"
MAX_EPOCHS="${MAX_EPOCHS:-32}"
PROJECT="${PROJECT:-flash_vqg_mqar}"
ENTITY="${ENTITY:-scu-mclab}"
CACHE_DIR="${CACHE_DIR:-./data/flash_vqg}"
LAUNCH_ID_PREFIX="${LAUNCH_ID_PREFIX:-flash-vqg-e5}"
TRAIN_BATCH_ORDER="${TRAIN_BATCH_ORDER:-global_shuffle}"
NUM_CODEBOOK_VECTORS="${NUM_CODEBOOK_VECTORS:-64,128,256,512}"
METRICS_WHITE_LIST_FILE="${METRICS_WHITE_LIST_FILE:-${ROOT_DIR}/zoology/experiments/flash_vqg/metrics_white_lists/e5.yaml}"
METRICS_WHITE_LIST="${METRICS_WHITE_LIST:-}"

echo "==> Running E5 codebook sweep [${NUM_CODEBOOK_VECTORS}] on GPU ${GPU_ID}"
CMD=(
  "${PYTHON_BIN}" -m zoology.experiments.flash_vqg.run_flash_vqg_suite
  --flash-only \
  --logger-backend swanlab \
  --analysis remote \
  --backend "${BACKEND}" \
  --block-len 32 \
  --dmodels "${DMODEL}" \
  --learning-rates "${LR}" \
  --local-num-blocks 2 \
  --if-remote-enabled true \
  --train-batch-order "${TRAIN_BATCH_ORDER}" \
  --num-codebook-vectors "${NUM_CODEBOOK_VECTORS}" \
  --cache-dir "${CACHE_DIR}" \
  --metrics-white-list-file "${METRICS_WHITE_LIST_FILE}" \
  --project "${PROJECT}" \
  --entity "${ENTITY}" \
  --max-epochs "${MAX_EPOCHS}" \
  --launch-id-prefix "${LAUNCH_ID_PREFIX}" \
  --gpus "${GPU_ID}"
)

if [[ -n "${METRICS_WHITE_LIST}" ]]; then
  CMD+=(--metrics-white-list "${METRICS_WHITE_LIST}")
fi

"${CMD[@]}"
