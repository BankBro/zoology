#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAINLINE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${MAINLINE_DIR}/common_env.sh"

LAUNCH_ID_PREFIX_E3_SMOKE="${LAUNCH_ID_PREFIX_E3_SMOKE:-flash-vqg-20260402-clr-v1-e3-smoke}"
METRICS_WHITE_LIST_FILE_E3="${METRICS_WHITE_LIST_FILE_E3:-${SCRIPT_DIR}/metrics.yaml}"
VQ_TOPK="${VQ_TOPK:-4}"

cd "${ROOT_DIR}"

echo "==> Running E3 smoke on GPU ${GPU_ID}"
OUTPUT="$(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/smoke_e3_batch_accum.py" \
    --gpu "${GPU_ID}" \
    --backend "${BACKEND}" \
    --dmodels "${DMODEL}" \
    --learning-rates "${LR}" \
    --train-batch-order "${TRAIN_BATCH_ORDER}" \
    --seed-values "${SEED_VALUES}" \
    --data-seed "${DATA_SEED}" \
    --num-codebook-vectors "${NUM_CODEBOOK_VECTORS}" \
    --fox-remote-path-backend "${REMOTE_PATH_BACKEND}" \
    --fox-clr-rank "${FOX_CLR_RANK}" \
    --fox-clr-use-den-residual "${FOX_CLR_USE_DEN_RESIDUAL}" \
    --fox-clr-remat-mode "${FOX_CLR_REMAT_MODE}" \
    --vq-topk "${VQ_TOPK}" \
    --cache-dir "${CACHE_DIR}" \
    --project "${PROJECT}" \
    --entity "${ENTITY}" \
    --launch-id-prefix "${LAUNCH_ID_PREFIX_E3_SMOKE}" \
    --metrics-white-list-file "${METRICS_WHITE_LIST_FILE_E3}"
)"
printf '%s\n' "${OUTPUT}"

SMOKE_ENV_FILE="$(printf '%s\n' "${OUTPUT}" | awk -F= '/^SMOKE_ENV_FILE=/{print $2}' | tail -n1)"
if [[ -z "${SMOKE_ENV_FILE}" || ! -f "${SMOKE_ENV_FILE}" ]]; then
  echo "E3 smoke 未产出可用的 SMOKE_ENV_FILE." >&2
  exit 1
fi

cp "${SMOKE_ENV_FILE}" "${SCRIPT_DIR}/e3_smoke.env"
echo "E3 smoke 参数已写入 ${SCRIPT_DIR}/e3_smoke.env"
