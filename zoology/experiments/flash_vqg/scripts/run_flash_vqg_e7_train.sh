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
LAUNCH_ID_PREFIX="${LAUNCH_ID_PREFIX:-flash-vqg-e7-train}"
TRAIN_BATCH_ORDER="${TRAIN_BATCH_ORDER:-global_shuffle}"
SEED_VALUES="${SEED_VALUES:-123,456,789}"
DATA_SEED="${DATA_SEED:-123}"
REMOTE_PATH_BACKEND="${REMOTE_PATH_BACKEND:-torch}"
REMOTE_READ_TOPK_VALUES="${REMOTE_READ_TOPK_VALUES:-dense,2,4}"
METRICS_WHITE_LIST_FILE="${METRICS_WHITE_LIST_FILE:-${ROOT_DIR}/zoology/experiments/flash_vqg/metrics_white_lists/e7.yaml}"
METRICS_WHITE_LIST="${METRICS_WHITE_LIST:-}"
ANALYSIS_SOURCE="${ANALYSIS_SOURCE:-remote}"

echo "==> Running E7-train read sweep [${REMOTE_READ_TOPK_VALUES}] with seeds [${SEED_VALUES}] on GPU ${GPU_ID}"
CMD=(
  "${PYTHON_BIN}" -m zoology.experiments.flash_vqg.run_flash_vqg_suite
  --flash-only \
  --logger-backend swanlab \
  --analysis "${ANALYSIS_SOURCE}" \
  --backend "${BACKEND}" \
  --block-len 32 \
  --dmodels "${DMODEL}" \
  --learning-rates "${LR}" \
  --local-num-blocks 2 \
  --if-remote-enabled true \
  --train-batch-order "${TRAIN_BATCH_ORDER}" \
  --seed-values "${SEED_VALUES}" \
  --data-seed "${DATA_SEED}" \
  --fox-remote-path-backend "${REMOTE_PATH_BACKEND}" \
  --fox-remote-read-topk-values "${REMOTE_READ_TOPK_VALUES}" \
  --num-codebook-vectors 128 \
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
