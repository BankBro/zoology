#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
FLASH_VQG_E5_ROOT="${FLASH_VQG_E5_ROOT:-/home/lyj/mnt/project/Flash-VQG-e5}"
PYTHON_BIN="${PYTHON_BIN:-/home/lyj/miniconda3/envs/flash-vqg/bin/python}"
GPU_ID="${GPU_ID:?需要提供 GPU_ID}"

export PYTHONPATH="${FLASH_VQG_E5_ROOT}/src:${ROOT_DIR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

BACKEND="${BACKEND:-torch}"
DMODEL="${DMODEL:-128}"
LR="${LR:-1e-3}"
MAX_EPOCHS="${MAX_EPOCHS:-32}"
CACHE_DIR="${CACHE_DIR:-./data/flash_vqg}"
PROJECT="${PROJECT:-flash_vqg_clr_v1_mainline}"
ENTITY="${ENTITY:-scu-mclab}"
ANALYSIS_SOURCE="${ANALYSIS_SOURCE:-local}"
LOGGER_BACKEND="${LOGGER_BACKEND:-swanlab}"
TRAIN_BATCH_ORDER="${TRAIN_BATCH_ORDER:-global_shuffle}"
NUM_CODEBOOK_VECTORS="${NUM_CODEBOOK_VECTORS:-128}"
REMOTE_PATH_BACKEND="${REMOTE_PATH_BACKEND:-torch}"
FOX_CLR_RANK="${FOX_CLR_RANK:-4}"
FOX_CLR_USE_DEN_RESIDUAL="${FOX_CLR_USE_DEN_RESIDUAL:-true}"
FOX_CLR_REMAT_MODE="${FOX_CLR_REMAT_MODE:-off}"
VQ_TOPK="${VQ_TOPK:-4}"
E5_SEED="${E5_SEED:-123}"
E5_DATA_SEED="${E5_DATA_SEED:-123}"
LAUNCH_ID_PREFIX_E5="${LAUNCH_ID_PREFIX_E5:-flash-vqg-20260402-clr-v1-e5-retaware}"
METRICS_WHITE_LIST_FILE_E5="${METRICS_WHITE_LIST_FILE_E5:-${SCRIPT_DIR}/metrics.yaml}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-}"
if [[ -z "${TRAIN_BATCH_SIZE}" || -z "${EVAL_BATCH_SIZE}" || -z "${GRADIENT_ACCUMULATION_STEPS}" ]]; then
  ENV_FILE="${SCRIPT_DIR}/../e3-dense-routing/e3_smoke.env"
  if [[ -f "${ENV_FILE}" ]]; then
    # shellcheck source=/dev/null
    source "${ENV_FILE}"
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-}"
    EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-}"
    GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-}"
  fi
fi
if [[ -z "${TRAIN_BATCH_SIZE}" || -z "${EVAL_BATCH_SIZE}" || -z "${GRADIENT_ACCUMULATION_STEPS}" ]]; then
  echo "需要提供 TRAIN_BATCH_SIZE / EVAL_BATCH_SIZE / GRADIENT_ACCUMULATION_STEPS." >&2
  exit 1
fi

"${PYTHON_BIN}" - <<'PY'
import sys
import flash_vqg
import zoology

expected_flash = "/home/lyj/mnt/project/Flash-VQG-e5"
expected_zoo = "/home/lyj/mnt/project/zoology-e5"
print(f"python={sys.executable}")
print(f"sys.path[:5]={sys.path[:5]}")
print(f"flash_vqg={flash_vqg.__file__}")
print(f"zoology={zoology.__file__}")
if expected_flash not in str(flash_vqg.__file__):
    raise SystemExit(f"flash_vqg import guard failed: {flash_vqg.__file__}")
if expected_zoo not in str(zoology.__file__):
    raise SystemExit(f"zoology import guard failed: {zoology.__file__}")
PY

export E5_SEED
export E5_DATA_SEED
E5_CONFIG_BUILDER_FUNC="${E5_CONFIG_BUILDER_FUNC:-build_e5_train_configs}"
BUILDER_SPEC="${SCRIPT_DIR}/config_builder.py:${E5_CONFIG_BUILDER_FUNC}"

cd "${ROOT_DIR}"

echo "==> Running E5 retrieval-aware matrix on GPU ${GPU_ID}"
echo "    seed=${E5_SEED}, data_seed=${E5_DATA_SEED}, launch_id_prefix=${LAUNCH_ID_PREFIX_E5}"
echo "    config_builder=${BUILDER_SPEC}"

"${PYTHON_BIN}" -m zoology.experiments.flash_vqg.run_flash_vqg_suite \
  --flash-only \
  --logger-backend "${LOGGER_BACKEND}" \
  --analysis "${ANALYSIS_SOURCE}" \
  --backend "${BACKEND}" \
  --dmodels "${DMODEL}" \
  --learning-rates "${LR}" \
  --max-epochs "${MAX_EPOCHS}" \
  --train-batch-order "${TRAIN_BATCH_ORDER}" \
  --seed-values "${E5_SEED}" \
  --data-seed "${E5_DATA_SEED}" \
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
  --metrics-white-list-file "${METRICS_WHITE_LIST_FILE_E5}" \
  --project "${PROJECT}" \
  --entity "${ENTITY}" \
  --launch-id-prefix "${LAUNCH_ID_PREFIX_E5}" \
  --config-builder "${BUILDER_SPEC}" \
  --gpus "${GPU_ID}"
