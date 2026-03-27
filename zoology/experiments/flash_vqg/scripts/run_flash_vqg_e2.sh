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
LAUNCH_ID_PREFIX="${LAUNCH_ID_PREFIX:-flash-vqg-e2}"
TRAIN_BATCH_ORDER="${TRAIN_BATCH_ORDER:-global_shuffle}"

echo "==> Running E2 remote_hist_1 vs local_only_1 on GPU ${GPU_ID}"
"${PYTHON_BIN}" -m zoology.experiments.flash_vqg.run_flash_vqg_suite \
  --flash-only \
  --logger-backend swanlab \
  --analysis remote \
  --backend "${BACKEND}" \
  --block-len 32 \
  --dmodels "${DMODEL}" \
  --learning-rates "${LR}" \
  --local-num-blocks 1 \
  --if-remote-enabled true,false \
  --train-batch-order "${TRAIN_BATCH_ORDER}" \
  --cache-dir "${CACHE_DIR}" \
  --project "${PROJECT}" \
  --entity "${ENTITY}" \
  --max-epochs "${MAX_EPOCHS}" \
  --launch-id-prefix "${LAUNCH_ID_PREFIX}" \
  --gpus "${GPU_ID}"
