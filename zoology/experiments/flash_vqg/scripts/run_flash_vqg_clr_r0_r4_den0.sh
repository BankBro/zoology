#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/lyj/mnt/project/zoology"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
R0_GPU_ID="${R0_GPU_ID:-0}"
R4_GPU_ID="${R4_GPU_ID:-1}"
BACKEND="${BACKEND:-torch}"
DMODEL="${DMODEL:-128}"
LR="${LR:-1e-3}"
MAX_EPOCHS="${MAX_EPOCHS:-32}"
PROJECT="${PROJECT:-flash_vqg_mqar}"
ENTITY="${ENTITY:-scu-mclab}"
BASE_LAUNCH_ID_PREFIX="${BASE_LAUNCH_ID_PREFIX:-flash-vqg-clr-den0-ablation}"
R0_LAUNCH_ID_PREFIX="${R0_LAUNCH_ID_PREFIX:-${BASE_LAUNCH_ID_PREFIX}-r0-den0}"
R4_LAUNCH_ID_PREFIX="${R4_LAUNCH_ID_PREFIX:-${BASE_LAUNCH_ID_PREFIX}-r4-den0}"
BLOCK_LEN="${BLOCK_LEN:-32}"
LOCAL_NUM_BLOCKS="${LOCAL_NUM_BLOCKS:-2}"
NUM_CODEBOOK_VECTORS="${NUM_CODEBOOK_VECTORS:-128}"
TRAIN_BATCH_ORDER="${TRAIN_BATCH_ORDER:-global_shuffle}"
METRICS_WHITE_LIST_FILE="${METRICS_WHITE_LIST_FILE:-${ROOT_DIR}/zoology/experiments/flash_vqg/metrics_white_lists/e0.yaml}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/tmp/flash_vqg_clr_r0_r4_den0}"

mkdir -p "${LOG_DIR}"

echo "==> clr_v1-r0-den0 使用 GPU ${R0_GPU_ID}, launch_id_prefix=${R0_LAUNCH_ID_PREFIX}"
echo "==> clr_v1-r4-den0 使用 GPU ${R4_GPU_ID}, launch_id_prefix=${R4_LAUNCH_ID_PREFIX}"
echo "==> 日志目录: ${LOG_DIR}"
echo "==> 默认假设当前环境中的 flash_vqg editable 安装已经指向新的 worktree"

R0_CMD=(
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
  --gpus "${R0_GPU_ID}"
  --launch-id-prefix "${R0_LAUNCH_ID_PREFIX}"
  --fox-remote-path-backend torch
  --fox-remote-formula clr_v1
  --fox-clr-rank 0
  --fox-clr-use-den-residual false
)

R4_CMD=(
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
  --gpus "${R4_GPU_ID}"
  --launch-id-prefix "${R4_LAUNCH_ID_PREFIX}"
  --fox-remote-path-backend torch
  --fox-remote-formula clr_v1
  --fox-clr-rank 4
  --fox-clr-use-den-residual false
)

echo "==> 启动 clr_v1-r0-den0"
printf '    %q ' "${R0_CMD[@]}"
printf '\n'
"${R0_CMD[@]}" >"${LOG_DIR}/clr_v1_r0_den0.log" 2>&1 &
R0_PID=$!

echo "==> 启动 clr_v1-r4-den0"
printf '    %q ' "${R4_CMD[@]}"
printf '\n'
"${R4_CMD[@]}" >"${LOG_DIR}/clr_v1_r4_den0.log" 2>&1 &
R4_PID=$!

cleanup() {
  local exit_code=$?
  if [[ $exit_code -ne 0 ]]; then
    echo "==> 脚本异常退出, 正在清理后台任务"
    kill "${R0_PID}" "${R4_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

wait "${R0_PID}"
wait "${R4_PID}"

trap - EXIT

echo "==> 两个实验都已结束"
echo "==> clr_v1-r0-den0 日志: ${LOG_DIR}/clr_v1_r0_den0.log"
echo "==> clr_v1-r4-den0 日志: ${LOG_DIR}/clr_v1_r4_den0.log"
