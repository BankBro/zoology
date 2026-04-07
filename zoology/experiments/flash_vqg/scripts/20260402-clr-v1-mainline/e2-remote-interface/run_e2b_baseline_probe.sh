#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAINLINE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${MAINLINE_DIR}/common_env.sh"

cd "${ROOT_DIR}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-}"
if [[ -z "${TRAIN_BATCH_SIZE}" || -z "${EVAL_BATCH_SIZE}" || -z "${GRADIENT_ACCUMULATION_STEPS}" ]]; then
  ENV_FILE="${SCRIPT_DIR}/e2b_smoke.env"
  if [[ -f "${ENV_FILE}" ]]; then
    # shellcheck source=/dev/null
    source "${ENV_FILE}"
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-}"
    EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-}"
    GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-}"
  fi
fi
if [[ -z "${TRAIN_BATCH_SIZE}" || -z "${EVAL_BATCH_SIZE}" || -z "${GRADIENT_ACCUMULATION_STEPS}" ]]; then
  echo "需要先通过 E2b smoke 确定 TRAIN_BATCH_SIZE / EVAL_BATCH_SIZE / GRADIENT_ACCUMULATION_STEPS." >&2
  exit 1
fi

if [[ "${SEED_VALUES}" == *,* ]]; then
  echo "baseline probe 只接受单个 SEED_VALUES, 当前收到: ${SEED_VALUES}" >&2
  exit 1
fi
if [[ "${SEED_VALUES}" != "${DATA_SEED}" ]]; then
  echo "baseline probe 要求 SEED_VALUES 与 DATA_SEED 相同, 当前收到 SEED_VALUES=${SEED_VALUES}, DATA_SEED=${DATA_SEED}" >&2
  exit 1
fi

LAUNCH_ID_PREFIX="${LAUNCH_ID_PREFIX:-flash-vqg-20260402-clr-v1-e2b-baseline-s${SEED_VALUES}}"
BUILDER_SPEC="${SCRIPT_DIR}/config_builder.py:build_e2b_baseline_probe_configs"

"${PYTHON_BIN}" -m zoology.experiments.flash_vqg.run_flash_vqg_suite \
  --flash-only \
  --logger-backend swanlab \
  --analysis "${ANALYSIS_SOURCE}" \
  --backend "${BACKEND}" \
  --dmodels "${DMODEL}" \
  --learning-rates "${LR}" \
  --max-epochs "${MAX_EPOCHS}" \
  --train-batch-order "${TRAIN_BATCH_ORDER}" \
  --seed-values "${SEED_VALUES}" \
  --data-seed "${DATA_SEED}" \
  --num-codebook-vectors "${NUM_CODEBOOK_VECTORS}" \
  --fox-remote-path-backend "${REMOTE_PATH_BACKEND}" \
  --fox-clr-rank "${FOX_CLR_RANK}" \
  --fox-clr-use-den-residual "${FOX_CLR_USE_DEN_RESIDUAL}" \
  --fox-clr-remat-mode "${FOX_CLR_REMAT_MODE}" \
  --train-batch-size "${TRAIN_BATCH_SIZE}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --cache-dir "${CACHE_DIR}" \
  --metrics-white-list-file "${METRICS_WHITE_LIST_FILE_E2}" \
  --project "${PROJECT}" \
  --entity "${ENTITY}" \
  --launch-id-prefix "${LAUNCH_ID_PREFIX}" \
  --config-builder "${BUILDER_SPEC}" \
  --gpus "${GPU_ID}"
