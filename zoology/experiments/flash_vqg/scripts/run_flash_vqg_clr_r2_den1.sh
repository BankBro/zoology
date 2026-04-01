#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/lyj/mnt/project/zoology"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-1}"
BACKEND="${BACKEND:-torch}"
DMODEL="${DMODEL:-128}"
LR="${LR:-1e-3}"
MAX_EPOCHS="${MAX_EPOCHS:-32}"
PROJECT="${PROJECT:-flash_vqg_mqar}"
ENTITY="${ENTITY:-scu-mclab}"
LAUNCH_ID_PREFIX="${LAUNCH_ID_PREFIX:-flash-vqg-clr-r2-den1}"
BLOCK_LEN="${BLOCK_LEN:-32}"
LOCAL_NUM_BLOCKS="${LOCAL_NUM_BLOCKS:-2}"
NUM_CODEBOOK_VECTORS="${NUM_CODEBOOK_VECTORS:-128}"
TRAIN_BATCH_ORDER="${TRAIN_BATCH_ORDER:-global_shuffle}"
METRICS_WHITE_LIST_FILE="${METRICS_WHITE_LIST_FILE:-${ROOT_DIR}/zoology/experiments/flash_vqg/metrics_white_lists/e0.yaml}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/tmp/flash_vqg_clr_r2_den1}"

mkdir -p "${LOG_DIR}"

echo "==> clr_v1-r2-den1 使用 GPU ${GPU_ID}, launch_id_prefix=${LAUNCH_ID_PREFIX}"
echo "==> 日志目录: ${LOG_DIR}"
echo "==> 当前脚本不启用 remat"

CMD=(
  "${PYTHON_BIN}" -m zoology.experiments.flash_vqg.run_flash_vqg_suite
  --flash-only
  --backend "${BACKEND}"
  --logger-backend swanlab
  --analysis remote
  --block-len "${BLOCK_LEN}"
  --dmodels "${DMODEL}"
  --learning-rates "${LR}"
  --local-num-blocks "${LOCAL_NUM_BLOCKS}"
  --if-remote-enabled true
  --train-batch-order "${TRAIN_BATCH_ORDER}"
  --num-codebook-vectors "${NUM_CODEBOOK_VECTORS}"
  --metrics-white-list-file "${METRICS_WHITE_LIST_FILE}"
  --project "${PROJECT}"
  --entity "${ENTITY}"
  --max-epochs "${MAX_EPOCHS}"
  --gpus "${GPU_ID}"
  --launch-id-prefix "${LAUNCH_ID_PREFIX}"
  --fox-remote-path-backend torch
  --fox-remote-formula clr_v1
  --fox-clr-rank 2
  --fox-clr-use-den-residual true
)

printf '==> 启动命令: '
printf '%q ' "${CMD[@]}"
printf '\n'
"${CMD[@]}" | tee "${LOG_DIR}/clr_v1_r2_den1.log"

echo "==> 实验结束"
echo "==> 日志: ${LOG_DIR}/clr_v1_r2_den1.log"
