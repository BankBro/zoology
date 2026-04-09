#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAINLINE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${MAINLINE_DIR}/common_env.sh"

METRICS_WHITE_LIST_FILE_E3="${METRICS_WHITE_LIST_FILE_E3:-${SCRIPT_DIR}/metrics.yaml}"
VQ_TOPK="${VQ_TOPK:-4}"
TAU_VALUE="${TAU_VALUE:?需要提供 TAU_VALUE}"
TAU_TAG="${TAU_TAG:?需要提供 TAU_TAG}"
RUN_ID="${RUN_ID:?需要提供 RUN_ID}"
EXPERIMENT_MODE="${EXPERIMENT_MODE:-dense_t${TAU_TAG}}"
SEED_VALUE="${SEED_VALUE:-123}"
DATA_SEED_VALUE="${DATA_SEED_VALUE:-123}"
REMOTE_READ_MODE="${REMOTE_READ_MODE:-2}"
LAUNCH_ID_PREFIX_E3_TAU="${LAUNCH_ID_PREFIX_E3_TAU:-flash-vqg-20260402-clr-v1-e3-tau-local-${TAU_TAG}}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-}"
if [[ -z "${TRAIN_BATCH_SIZE}" || -z "${EVAL_BATCH_SIZE}" || -z "${GRADIENT_ACCUMULATION_STEPS}" ]]; then
  ENV_FILE="${SCRIPT_DIR}/e3_smoke.env"
  if [[ -f "${ENV_FILE}" ]]; then
    # shellcheck source=/dev/null
    source "${ENV_FILE}"
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-}"
    EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-}"
    GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-}"
  fi
fi
if [[ -z "${TRAIN_BATCH_SIZE}" || -z "${EVAL_BATCH_SIZE}" || -z "${GRADIENT_ACCUMULATION_STEPS}" ]]; then
  echo "需要先通过 E3 smoke 确定 TRAIN_BATCH_SIZE / EVAL_BATCH_SIZE / GRADIENT_ACCUMULATION_STEPS." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export E3_TAU_VALUE="${TAU_VALUE}"
export E3_TAU_TAG="${TAU_TAG}"
export E3_TAU_RUN_ID="${RUN_ID}"
export E3_TAU_EXPERIMENT_MODE="${EXPERIMENT_MODE}"
export E3_TAU_SEED="${SEED_VALUE}"
export E3_TAU_DATA_SEED="${DATA_SEED_VALUE}"
export E3_REMOTE_READ_MODE="${REMOTE_READ_MODE}"

BUILDER_SPEC="${SCRIPT_DIR}/tau_builder.py:build_e3_tau_single_config"

cd "${ROOT_DIR}"

echo "==> Running E3 tau local scan ${RUN_ID} on GPU ${GPU_ID}"
echo "    tau=${TAU_VALUE}, seed=${SEED_VALUE}, data_seed=${DATA_SEED_VALUE}, remote_read=${REMOTE_READ_MODE}"

"${PYTHON_BIN}" -m zoology.experiments.flash_vqg.run_flash_vqg_suite \
  --flash-only \
  --logger-backend swanlab \
  --analysis "${ANALYSIS_SOURCE}" \
  --backend "${BACKEND}" \
  --dmodels "${DMODEL}" \
  --learning-rates "${LR}" \
  --max-epochs "${MAX_EPOCHS}" \
  --train-batch-order "${TRAIN_BATCH_ORDER}" \
  --seed-values "${SEED_VALUE}" \
  --data-seed "${DATA_SEED_VALUE}" \
  --num-codebook-vectors "${NUM_CODEBOOK_VECTORS}" \
  --fox-remote-path-backend "${REMOTE_PATH_BACKEND}" \
  --fox-clr-rank "${FOX_CLR_RANK}" \
  --fox-clr-use-den-residual "${FOX_CLR_USE_DEN_RESIDUAL}" \
  --fox-clr-remat-mode "${FOX_CLR_REMAT_MODE}" \
  --vq-topk "${VQ_TOPK}" \
  --train-batch-size "${TRAIN_BATCH_SIZE}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --cache-dir "${CACHE_DIR}" \
  --metrics-white-list-file "${METRICS_WHITE_LIST_FILE_E3}" \
  --project "${PROJECT}" \
  --entity "${ENTITY}" \
  --launch-id-prefix "${LAUNCH_ID_PREFIX_E3_TAU}" \
  --config-builder "${BUILDER_SPEC}" \
  --gpus "${GPU_ID}"
