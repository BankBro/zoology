#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/lyj/mnt/project/zoology"
cd "$ROOT_DIR"

SCRIPT_DIR="${ROOT_DIR}/zoology/experiments/flash_vqg/scripts"

PYTHON_BIN="${PYTHON_BIN:-/home/lyj/miniconda3/envs/flash-vqg/bin/python}"
GPU_ID="${GPU_ID:-0}"
BACKEND="${BACKEND:-accel}"
DMODEL="${DMODEL:-128}"
LR="${LR:-1e-3}"
MAX_EPOCHS="${MAX_EPOCHS:-32}"
PROJECT="${PROJECT:-flash_vqg_mqar}"
ENTITY="${ENTITY:-scu-mclab}"
CACHE_DIR="${CACHE_DIR:-./data/flash_vqg}"
TRAIN_BATCH_ORDER="${TRAIN_BATCH_ORDER:-global_shuffle}"
TRAIN_BATCH_ORDERS="${TRAIN_BATCH_ORDERS:-sequential,global_shuffle,balanced_interleave}"

echo "==> Sequentially running E0 -> E1 -> E2 -> E3 on GPU ${GPU_ID}"

PYTHON_BIN="${PYTHON_BIN}" \
GPU_ID="${GPU_ID}" \
BACKEND="${BACKEND}" \
DMODEL="${DMODEL}" \
LR="${LR}" \
MAX_EPOCHS="${MAX_EPOCHS}" \
PROJECT="${PROJECT}" \
ENTITY="${ENTITY}" \
CACHE_DIR="${CACHE_DIR}" \
TRAIN_BATCH_ORDERS="${TRAIN_BATCH_ORDERS}" \
bash "${SCRIPT_DIR}/run_flash_vqg_e0.sh"

PYTHON_BIN="${PYTHON_BIN}" \
GPU_ID="${GPU_ID}" \
BACKEND="${BACKEND}" \
DMODEL="${DMODEL}" \
LR="${LR}" \
MAX_EPOCHS="${MAX_EPOCHS}" \
PROJECT="${PROJECT}" \
ENTITY="${ENTITY}" \
CACHE_DIR="${CACHE_DIR}" \
TRAIN_BATCH_ORDER="${TRAIN_BATCH_ORDER}" \
bash "${SCRIPT_DIR}/run_flash_vqg_e1.sh"

PYTHON_BIN="${PYTHON_BIN}" \
GPU_ID="${GPU_ID}" \
BACKEND="${BACKEND}" \
DMODEL="${DMODEL}" \
LR="${LR}" \
MAX_EPOCHS="${MAX_EPOCHS}" \
PROJECT="${PROJECT}" \
ENTITY="${ENTITY}" \
CACHE_DIR="${CACHE_DIR}" \
TRAIN_BATCH_ORDER="${TRAIN_BATCH_ORDER}" \
bash "${SCRIPT_DIR}/run_flash_vqg_e2.sh"

PYTHON_BIN="${PYTHON_BIN}" \
GPU_ID="${GPU_ID}" \
BACKEND="${BACKEND}" \
DMODEL="${DMODEL}" \
LR="${LR}" \
MAX_EPOCHS="${MAX_EPOCHS}" \
PROJECT="${PROJECT}" \
ENTITY="${ENTITY}" \
CACHE_DIR="${CACHE_DIR}" \
TRAIN_BATCH_ORDER="${TRAIN_BATCH_ORDER}" \
bash "${SCRIPT_DIR}/run_flash_vqg_e3.sh"

echo "==> Finished E0 -> E3"
