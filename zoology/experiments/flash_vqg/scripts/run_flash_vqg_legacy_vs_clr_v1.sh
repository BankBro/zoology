#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/lyj/mnt/project/zoology"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
LEGACY_GPU_ID="${LEGACY_GPU_ID:-0}"
CLR_GPU_ID="${CLR_GPU_ID:-1}"
BACKEND="${BACKEND:-torch}"
DMODEL="${DMODEL:-128}"
LR="${LR:-1e-3}"
MAX_EPOCHS="${MAX_EPOCHS:-32}"
PROJECT="${PROJECT:-flash_vqg_mqar}"
ENTITY="${ENTITY:-scu-mclab}"
CACHE_DIR="${CACHE_DIR:-./data/flash_vqg}"
BASE_LAUNCH_ID_PREFIX="${BASE_LAUNCH_ID_PREFIX:-flash-vqg-legacy-vs-clr}"
LEGACY_LAUNCH_ID_PREFIX="${LEGACY_LAUNCH_ID_PREFIX:-${BASE_LAUNCH_ID_PREFIX}-legacy}"
CLR_LAUNCH_ID_PREFIX="${CLR_LAUNCH_ID_PREFIX:-${BASE_LAUNCH_ID_PREFIX}-clr-v1}"
BLOCK_LEN="${BLOCK_LEN:-32}"
LOCAL_NUM_BLOCKS="${LOCAL_NUM_BLOCKS:-2}"
NUM_CODEBOOK_VECTORS="${NUM_CODEBOOK_VECTORS:-128}"
TRAIN_BATCH_ORDER="${TRAIN_BATCH_ORDER:-global_shuffle}"
METRICS_WHITE_LIST_FILE="${METRICS_WHITE_LIST_FILE:-${ROOT_DIR}/zoology/experiments/flash_vqg/metrics_white_lists/e0.yaml}"
CLR_RANK="${CLR_RANK:-4}"
CLR_USE_DEN_RESIDUAL="${CLR_USE_DEN_RESIDUAL:-true}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/tmp/flash_vqg_legacy_vs_clr_v1}"

mkdir -p "${LOG_DIR}"

echo "==> legacy 使用 GPU ${LEGACY_GPU_ID}, launch_id_prefix=${LEGACY_LAUNCH_ID_PREFIX}"
echo "==> clr_v1 使用 GPU ${CLR_GPU_ID}, launch_id_prefix=${CLR_LAUNCH_ID_PREFIX}"
echo "==> 日志目录: ${LOG_DIR}"
echo "==> 默认假设当前环境中的 flash_vqg editable 安装已经指向新的 worktree"

LEGACY_CMD=(
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
  --gpus "${LEGACY_GPU_ID}"
  --launch-id-prefix "${LEGACY_LAUNCH_ID_PREFIX}"
  --fox-remote-path-backend torch
  --fox-remote-formula legacy
)

CLR_CMD=(
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
  --gpus "${CLR_GPU_ID}"
  --launch-id-prefix "${CLR_LAUNCH_ID_PREFIX}"
  --fox-remote-path-backend torch
  --fox-remote-formula clr_v1
  --fox-clr-rank "${CLR_RANK}"
  --fox-clr-use-den-residual "${CLR_USE_DEN_RESIDUAL}"
)

echo "==> 启动 legacy"
printf '    %q ' "${LEGACY_CMD[@]}"
printf '\n'
"${LEGACY_CMD[@]}" >"${LOG_DIR}/legacy.log" 2>&1 &
LEGACY_PID=$!

echo "==> 启动 clr_v1"
printf '    %q ' "${CLR_CMD[@]}"
printf '\n'
"${CLR_CMD[@]}" >"${LOG_DIR}/clr_v1.log" 2>&1 &
CLR_PID=$!

cleanup() {
  local exit_code=$?
  if [[ $exit_code -ne 0 ]]; then
    echo "==> 脚本异常退出, 正在清理后台任务"
    kill "${LEGACY_PID}" "${CLR_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

wait "${LEGACY_PID}"
wait "${CLR_PID}"

trap - EXIT

echo "==> 两个实验都已结束"
echo "==> legacy 日志: ${LOG_DIR}/legacy.log"
echo "==> clr_v1 日志: ${LOG_DIR}/clr_v1.log"
