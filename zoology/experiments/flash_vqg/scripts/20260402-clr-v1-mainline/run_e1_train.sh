#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/common_env.sh"

cd "${ROOT_DIR}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-}"
METRICS_WHITE_LIST="${METRICS_WHITE_LIST:-}"

if [[ -z "${TRAIN_BATCH_SIZE}" || -z "${EVAL_BATCH_SIZE}" || -z "${GRADIENT_ACCUMULATION_STEPS}" ]]; then
  echo "需要先确定 TRAIN_BATCH_SIZE / EVAL_BATCH_SIZE / GRADIENT_ACCUMULATION_STEPS." >&2
  echo "建议先运行 ${SCRIPT_DIR}/run_e1_smoke.sh, 然后把推荐值作为环境变量传入." >&2
  exit 1
fi

echo "==> Running clr_v1 E1 soft top-k read [${REMOTE_READ_TOPK_VALUES}] on GPU ${GPU_ID}"
echo "==> train_batch_size=${TRAIN_BATCH_SIZE}, eval_batch_size=${EVAL_BATCH_SIZE}, gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}"

CMD=(
  "${PYTHON_BIN}" -m zoology.experiments.flash_vqg.run_flash_vqg_suite
  --flash-only
  --logger-backend swanlab
  --analysis "${ANALYSIS_SOURCE}"
  --backend "${BACKEND}"
  --block-len "${BLOCK_LEN}"
  --dmodels "${DMODEL}"
  --learning-rates "${LR}"
  --max-epochs "${MAX_EPOCHS}"
  --if-remote-enabled true
  --local-num-blocks "${LOCAL_NUM_BLOCKS}"
  --train-batch-order "${TRAIN_BATCH_ORDER}"
  --seed-values "${SEED_VALUES}"
  --data-seed "${DATA_SEED}"
  --num-codebook-vectors "${NUM_CODEBOOK_VECTORS}"
  --fox-remote-path-backend "${REMOTE_PATH_BACKEND}"
  --fox-remote-formula "${FOX_REMOTE_FORMULA}"
  --fox-clr-rank "${FOX_CLR_RANK}"
  --fox-clr-use-den-residual "${FOX_CLR_USE_DEN_RESIDUAL}"
  --fox-clr-remat-mode "${FOX_CLR_REMAT_MODE}"
  --fox-remote-read-topk-values "${REMOTE_READ_TOPK_VALUES}"
  --train-batch-size "${TRAIN_BATCH_SIZE}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"
  --cache-dir "${CACHE_DIR}"
  --metrics-white-list-file "${METRICS_WHITE_LIST_FILE}"
  --project "${PROJECT}"
  --entity "${ENTITY}"
  --launch-id-prefix "${LAUNCH_ID_PREFIX_E1}"
  --gpus "${GPU_ID}"
)

if [[ -n "${METRICS_WHITE_LIST}" ]]; then
  CMD+=(--metrics-white-list "${METRICS_WHITE_LIST}")
fi

"${CMD[@]}"
