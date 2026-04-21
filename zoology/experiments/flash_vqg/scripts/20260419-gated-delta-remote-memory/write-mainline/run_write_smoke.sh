#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAINLINE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${MAINLINE_DIR}/common_env.sh"

METRICS_WHITE_LIST_FILE_WRITE="${METRICS_WHITE_LIST_FILE_WRITE:-${SCRIPT_DIR}/metrics.yaml}"
SUMMARY_OUT="${SUMMARY_OUT:-${SCRIPT_DIR}/write_smoke_summary.json}"
ENV_OUT="${ENV_OUT:-${SCRIPT_DIR}/write_smoke.env}"

cd "${ROOT_DIR}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/smoke_write_batch_accum.py" \
  --gpu "${GPU_ID}" \
  --backend "${BACKEND}" \
  --dmodels "${DMODEL}" \
  --learning-rates "${LR}" \
  --max-epochs 1 \
  --train-batch-order "${TRAIN_BATCH_ORDER}" \
  --seed-values "${SEED_VALUES}" \
  --data-seed "${DATA_SEED}" \
  --num-codebook-vectors "${NUM_CODEBOOK_VECTORS}" \
  --fox-remote-path-backend "${REMOTE_PATH_BACKEND}" \
  --fox-clr-rank "${FOX_CLR_RANK}" \
  --fox-clr-use-den-residual "${FOX_CLR_USE_DEN_RESIDUAL}" \
  --fox-clr-remat-mode "${FOX_CLR_REMAT_MODE}" \
  --fox-clr-state-write-topk "${FOX_CLR_STATE_WRITE_TOPK}" \
  --fox-clr-delta-target-mode "${FOX_CLR_DELTA_TARGET_MODE}" \
  --vq-topk "${VQ_TOPK}" \
  --cache-dir "${CACHE_DIR}" \
  --metrics-white-list-file "${METRICS_WHITE_LIST_FILE_WRITE}" \
  --project "${PROJECT}" \
  --entity "${ENTITY}" \
  --launch-id-prefix "${LAUNCH_ID_PREFIX_WRITE_SMOKE}" \
  --summary-out "${SUMMARY_OUT}" \
  --env-out "${ENV_OUT}"

echo "已更新 smoke 配置: ${ENV_OUT}"
