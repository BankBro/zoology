#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
FLASH_VQG_E5_ROOT="${FLASH_VQG_E5_ROOT:-/home/lyj/mnt/project/Flash-VQG-e5}"
PYTHON_BIN="${PYTHON_BIN:-/home/lyj/miniconda3/envs/flash-vqg/bin/python}"
GPU_ID="${GPU_ID:-0}"

export PYTHONPATH="${FLASH_VQG_E5_ROOT}/src:${ROOT_DIR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

LOGGER_BACKEND="${LOGGER_BACKEND:-none}"
ANALYSIS_SOURCE="${ANALYSIS_SOURCE:-off}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
LAUNCH_ID_PREFIX_E5_SMOKE="${LAUNCH_ID_PREFIX_E5_SMOKE:-flash-vqg-20260402-clr-v1-e5-retaware-smoke}"
METRICS_WHITE_LIST_FILE_E5="${METRICS_WHITE_LIST_FILE_E5:-${SCRIPT_DIR}/metrics.yaml}"
E5_SEED="${E5_SEED:-123}"
E5_DATA_SEED="${E5_DATA_SEED:-123}"

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
BUILDER_SPEC="${SCRIPT_DIR}/config_builder.py:build_e5_smoke_configs"

cd "${ROOT_DIR}"

"${PYTHON_BIN}" -m zoology.experiments.flash_vqg.run_flash_vqg_suite \
  --flash-only \
  --logger-backend "${LOGGER_BACKEND}" \
  --analysis "${ANALYSIS_SOURCE}" \
  --backend torch \
  --dmodels 128 \
  --learning-rates 1e-3 \
  --max-epochs "${MAX_EPOCHS}" \
  --train-batch-order global_shuffle \
  --seed-values "${E5_SEED}" \
  --data-seed "${E5_DATA_SEED}" \
  --num-codebook-vectors 128 \
  --fox-remote-path-backend torch \
  --fox-clr-rank 4 \
  --fox-clr-use-den-residual true \
  --fox-clr-remat-mode off \
  --vq-topk 4 \
  --train-batch-size "${TRAIN_BATCH_SIZE}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --cache-dir ./data/flash_vqg \
  --metrics-white-list-file "${METRICS_WHITE_LIST_FILE_E5}" \
  --project flash_vqg_clr_v1_mainline \
  --entity scu-mclab \
  --launch-id-prefix "${LAUNCH_ID_PREFIX_E5_SMOKE}" \
  --config-builder "${BUILDER_SPEC}" \
  --gpus "${GPU_ID}"
