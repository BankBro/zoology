#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAINLINE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${MAINLINE_DIR}/common_env.sh"

METRICS_WHITE_LIST_FILE_E3="${METRICS_WHITE_LIST_FILE_E3:-${SCRIPT_DIR}/metrics.yaml}"
E3_TOPK_VALUES="${E3_TOPK_VALUES:-2,4}"
E3_TOPK_TAU="${E3_TOPK_TAU:-0.25}"
E3_TOPK_TAU_TAG="${E3_TOPK_TAU_TAG:-${E3_TOPK_TAU/./}}"
E3_TOPK_SEED="${E3_TOPK_SEED:-123}"
E3_TOPK_DATA_SEED="${E3_TOPK_DATA_SEED:-123}"
LAUNCH_ID_PREFIX_E3_TOPK="${LAUNCH_ID_PREFIX_E3_TOPK:-flash-vqg-20260402-clr-v1-e3-topkwrite-probe-t${E3_TOPK_TAU_TAG}}"

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
export E3_TOPK_VALUES
export E3_TOPK_TAU
export E3_TOPK_TAU_TAG
export E3_TOPK_SEED
export E3_TOPK_DATA_SEED

BUILDER_SPEC="${SCRIPT_DIR}/topk_probe_builder.py:build_e3_topk_probe_configs"

cd "${ROOT_DIR}"

echo "==> Running E3 top-k write probe on GPU ${GPU_ID}"
echo "    topk_values=${E3_TOPK_VALUES}, tau=${E3_TOPK_TAU}, seed=${E3_TOPK_SEED}, data_seed=${E3_TOPK_DATA_SEED}"

"${PYTHON_BIN}" -m zoology.experiments.flash_vqg.run_flash_vqg_suite \
  --flash-only \
  --logger-backend swanlab \
  --analysis "${ANALYSIS_SOURCE}" \
  --backend "${BACKEND}" \
  --dmodels "${DMODEL}" \
  --learning-rates "${LR}" \
  --max-epochs "${MAX_EPOCHS}" \
  --train-batch-order "${TRAIN_BATCH_ORDER}" \
  --seed-values "${E3_TOPK_SEED}" \
  --data-seed "${E3_TOPK_DATA_SEED}" \
  --num-codebook-vectors "${NUM_CODEBOOK_VECTORS}" \
  --fox-remote-path-backend "${REMOTE_PATH_BACKEND}" \
  --fox-clr-rank "${FOX_CLR_RANK}" \
  --fox-clr-use-den-residual "${FOX_CLR_USE_DEN_RESIDUAL}" \
  --fox-clr-remat-mode "${FOX_CLR_REMAT_MODE}" \
  --vq-topk 4 \
  --train-batch-size "${TRAIN_BATCH_SIZE}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --cache-dir "${CACHE_DIR}" \
  --metrics-white-list-file "${METRICS_WHITE_LIST_FILE_E3}" \
  --project "${PROJECT}" \
  --entity "${ENTITY}" \
  --launch-id-prefix "${LAUNCH_ID_PREFIX_E3_TOPK}" \
  --config-builder "${BUILDER_SPEC}" \
  --gpus "${GPU_ID}"
