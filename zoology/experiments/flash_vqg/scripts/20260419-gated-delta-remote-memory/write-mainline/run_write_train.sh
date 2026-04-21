#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAINLINE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${MAINLINE_DIR}/common_env.sh"

METRICS_WHITE_LIST_FILE_WRITE="${METRICS_WHITE_LIST_FILE_WRITE:-${SCRIPT_DIR}/metrics.yaml}"
VQ_TOPK="${VQ_TOPK:-4}"

cd "${ROOT_DIR}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-}"
if [[ -z "${TRAIN_BATCH_SIZE}" || -z "${EVAL_BATCH_SIZE}" || -z "${GRADIENT_ACCUMULATION_STEPS}" ]]; then
  ENV_FILE="${SCRIPT_DIR}/write_smoke.env"
  if [[ -f "${ENV_FILE}" ]]; then
    # shellcheck source=/dev/null
    source "${ENV_FILE}"
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-}"
    EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-}"
    GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-}"
  fi
fi
if [[ -z "${TRAIN_BATCH_SIZE}" || -z "${EVAL_BATCH_SIZE}" || -z "${GRADIENT_ACCUMULATION_STEPS}" ]]; then
  echo "需要先通过 write smoke 确定 TRAIN_BATCH_SIZE / EVAL_BATCH_SIZE / GRADIENT_ACCUMULATION_STEPS." >&2
  exit 1
fi

BUILDER_SPEC="${SCRIPT_DIR}/config_builder.py:build_write_train_configs"
if "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("ray") is not None else 1)
PY
then
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
    --fox-clr-state-write-topk "${FOX_CLR_STATE_WRITE_TOPK}" \
    --fox-clr-delta-target-mode "${FOX_CLR_DELTA_TARGET_MODE}" \
    --vq-topk "${VQ_TOPK}" \
    --train-batch-size "${TRAIN_BATCH_SIZE}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --cache-dir "${CACHE_DIR}" \
    --metrics-white-list-file "${METRICS_WHITE_LIST_FILE_WRITE}" \
    --project "${PROJECT}" \
    --entity "${ENTITY}" \
    --launch-id-prefix "${LAUNCH_ID_PREFIX_WRITE}" \
    --config-builder "${BUILDER_SPEC}" \
    -p \
    --gpus "${GPU_IDS}"
else
  "${PYTHON_BIN}" "${MAINLINE_DIR}/local_parallel_launch.py" \
    --config-builder "${BUILDER_SPEC}" \
    --logger-backend swanlab \
    --analysis "${ANALYSIS_SOURCE}" \
    --status-interval-sec "${STATUS_INTERVAL_SEC}" \
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
    --fox-clr-state-write-topk "${FOX_CLR_STATE_WRITE_TOPK}" \
    --fox-clr-delta-target-mode "${FOX_CLR_DELTA_TARGET_MODE}" \
    --vq-topk "${VQ_TOPK}" \
    --train-batch-size "${TRAIN_BATCH_SIZE}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --cache-dir "${CACHE_DIR}" \
    --metrics-white-list-file "${METRICS_WHITE_LIST_FILE_WRITE}" \
    --project "${PROJECT}" \
    --entity "${ENTITY}" \
    --launch-id-prefix "${LAUNCH_ID_PREFIX_WRITE}" \
    --gpus "${GPU_IDS}"
fi
