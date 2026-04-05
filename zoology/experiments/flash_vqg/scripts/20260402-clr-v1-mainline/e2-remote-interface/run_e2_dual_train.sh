#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAINLINE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${MAINLINE_DIR}/common_env.sh"

GPU_ID_E2_MAIN="${GPU_ID_E2_MAIN:-0}"
GPU_ID_E2B="${GPU_ID_E2B:-1}"
TRAIN_BATCH_SIZE_E2_MAIN="${TRAIN_BATCH_SIZE_E2_MAIN:-}"
EVAL_BATCH_SIZE_E2_MAIN="${EVAL_BATCH_SIZE_E2_MAIN:-}"
GRADIENT_ACCUMULATION_STEPS_E2_MAIN="${GRADIENT_ACCUMULATION_STEPS_E2_MAIN:-}"
TRAIN_BATCH_SIZE_E2B="${TRAIN_BATCH_SIZE_E2B:-}"
EVAL_BATCH_SIZE_E2B="${EVAL_BATCH_SIZE_E2B:-}"
GRADIENT_ACCUMULATION_STEPS_E2B="${GRADIENT_ACCUMULATION_STEPS_E2B:-}"

if [[ -z "${TRAIN_BATCH_SIZE_E2_MAIN}" || -z "${EVAL_BATCH_SIZE_E2_MAIN}" || -z "${GRADIENT_ACCUMULATION_STEPS_E2_MAIN}" ]]; then
  echo "缺少 E2-main 的 smoke 产出 batch/GA 参数." >&2
  exit 1
fi
if [[ -z "${TRAIN_BATCH_SIZE_E2B}" || -z "${EVAL_BATCH_SIZE_E2B}" || -z "${GRADIENT_ACCUMULATION_STEPS_E2B}" ]]; then
  echo "缺少 E2b 的 smoke 产出 batch/GA 参数." >&2
  exit 1
fi

LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

(
  export GPU_ID="${GPU_ID_E2_MAIN}"
  export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE_E2_MAIN}"
  export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE_E2_MAIN}"
  export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS_E2_MAIN}"
  bash "${SCRIPT_DIR}/run_e2_main_train.sh"
) >"${LOG_DIR}/e2_main.log" 2>&1 &
PID_MAIN=$!

(
  export GPU_ID="${GPU_ID_E2B}"
  export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE_E2B}"
  export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE_E2B}"
  export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS_E2B}"
  bash "${SCRIPT_DIR}/run_e2b_train.sh"
) >"${LOG_DIR}/e2b.log" 2>&1 &
PID_E2B=$!

wait "${PID_MAIN}"
STATUS_MAIN=$?
wait "${PID_E2B}"
STATUS_E2B=$?

if [[ ${STATUS_MAIN} -ne 0 || ${STATUS_E2B} -ne 0 ]]; then
  echo "E2 dual train failed. See ${LOG_DIR}/e2_main.log and ${LOG_DIR}/e2b.log" >&2
  exit 1
fi

echo "E2 dual train finished successfully. Logs: ${LOG_DIR}"
