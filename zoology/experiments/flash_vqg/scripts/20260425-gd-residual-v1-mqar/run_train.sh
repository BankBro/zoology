#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/common_env.sh"

METRICS_WHITE_LIST_FILE="${METRICS_WHITE_LIST_FILE:-${SCRIPT_DIR}/metrics.yaml}"
BUILDER_SPEC="${SCRIPT_DIR}/config_builder.py:build_gd_residual_v1_train_configs"

cd "${ROOT_DIR}"

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
  --fox-remote-formula "${FOX_REMOTE_FORMULA}" \
  --fox-gd-residual-rank "${FOX_GD_RESIDUAL_RANK}" \
  --fox-gd-residual-write-topk "${FOX_GD_RESIDUAL_WRITE_TOPK}" \
  --fox-gd-residual-builder "${FOX_GD_RESIDUAL_BUILDER}" \
  --fox-gd-residual-pack-mode "${FOX_GD_RESIDUAL_PACK_MODE}" \
  --fox-gd-residual-chunk-size "${FOX_GD_RESIDUAL_CHUNK_SIZE}" \
  --fox-gd-residual-mu-min-count "${FOX_GD_RESIDUAL_MU_MIN_COUNT}" \
  --fox-gd-residual-addr-eps "${FOX_GD_RESIDUAL_ADDR_EPS}" \
  --fox-gd-residual-den-eps "${FOX_GD_RESIDUAL_DEN_EPS}" \
  --fox-gd-residual-rho-eps "${FOX_GD_RESIDUAL_RHO_EPS}" \
  --fox-gd-residual-beta-init "${FOX_GD_RESIDUAL_BETA_INIT}" \
  --fox-gd-residual-lambda-init "${FOX_GD_RESIDUAL_LAMBDA_INIT}" \
  --fox-gd-residual-norm-with-gain "${FOX_GD_RESIDUAL_NORM_WITH_GAIN}" \
  --fox-gd-residual-use-separate-addr-codebook "${FOX_GD_RESIDUAL_USE_SEPARATE_ADDR_CODEBOOK}" \
  --vq-score-mode "${VQ_SCORE_MODE}" \
  --vq-weight-mode "${VQ_WEIGHT_MODE}" \
  --vq-update-mode "${VQ_UPDATE_MODE}" \
  --vq-softmax-tau "${VQ_SOFTMAX_TAU}" \
  --vq-topk "${VQ_TOPK}" \
  --train-batch-size "${TRAIN_BATCH_SIZE}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --cache-dir "${CACHE_DIR}" \
  --metrics-white-list-file "${METRICS_WHITE_LIST_FILE}" \
  --project "${PROJECT}" \
  --entity "${ENTITY}" \
  --launch-id-prefix "${LAUNCH_ID_PREFIX_TRAIN}" \
  --config-builder "${BUILDER_SPEC}" \
  --gpus "${GPU_ID}"
