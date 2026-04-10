#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAINLINE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${MAINLINE_DIR}/common_env.sh"

METRICS_WHITE_LIST_FILE_E3="${METRICS_WHITE_LIST_FILE_E3:-${SCRIPT_DIR}/metrics.yaml}"
VQ_TOPK="${VQ_TOPK:-4}"
E35_CODEBOOK_VALUES="${E35_CODEBOOK_VALUES:-64,256}"
E35_SEED_VALUES="${E35_SEED_VALUES:-123,124}"
E35_DATA_SEED="${E35_DATA_SEED:-123}"
LAUNCH_ID_PREFIX_E35_CB="${LAUNCH_ID_PREFIX_E35_CB:-flash-vqg-20260402-clr-v1-e35-codebook-sweep-t025}"

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
export E35_CODEBOOK_VALUES
export E35_SEED_VALUES
export E35_DATA_SEED

BUILDER_SPEC="${SCRIPT_DIR}/codebook_probe_builder.py:build_e3_codebook_probe_configs"

cd "${ROOT_DIR}"

echo "==> Running E3.5 codebook probe on GPU ${GPU_ID}"
echo "    codebooks=${E35_CODEBOOK_VALUES}, seeds=${E35_SEED_VALUES}, data_seed=${E35_DATA_SEED}"

"${PYTHON_BIN}" -m zoology.experiments.flash_vqg.run_flash_vqg_suite \
  --flash-only \
  --logger-backend swanlab \
  --analysis "${ANALYSIS_SOURCE}" \
  --backend "${BACKEND}" \
  --dmodels "${DMODEL}" \
  --learning-rates "${LR}" \
  --max-epochs "${MAX_EPOCHS}" \
  --train-batch-order "${TRAIN_BATCH_ORDER}" \
  --seed-values "${E35_SEED_VALUES}" \
  --data-seed "${E35_DATA_SEED}" \
  --num-codebook-vectors "${E35_CODEBOOK_VALUES}" \
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
  --launch-id-prefix "${LAUNCH_ID_PREFIX_E35_CB}" \
  --config-builder "${BUILDER_SPEC}" \
  --gpus "${GPU_ID}"
