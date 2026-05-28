#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ROOT_DIR="${ROOT_DIR:-/home/lyj/mnt/project/worktrees/fla-kblocked/zoology}"
PYTHON_BIN="${PYTHON_BIN:-/home/lyj/miniconda3/envs/flash-vqg-kblocked/bin/python}"
GPU_ID="${GPU_ID:-0}"

BACKEND="${BACKEND:-torch}"
DMODEL="${DMODEL:-128}"
LR="${LR:-1e-3}"
MAX_EPOCHS="${MAX_EPOCHS:-4}"
CACHE_DIR="${CACHE_DIR:-./data/flash_vqg}"
PROJECT="${PROJECT:-flash_vqg_gdn_expanded_k}"
ENTITY="${ENTITY:-scu-mclab}"
SWANLAB_MODE="${SWANLAB_MODE:-cloud}"
ANALYSIS_SOURCE="${ANALYSIS_SOURCE:-off}"

TRAIN_BATCH_ORDER="${TRAIN_BATCH_ORDER:-global_shuffle}"
SEED_VALUES="${SEED_VALUES:-123}"
DATA_SEED="${DATA_SEED:-123}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
VALIDATIONS_PER_EPOCH="${VALIDATIONS_PER_EPOCH:-2}"
DISABLE_EARLY_STOPPING="${DISABLE_EARLY_STOPPING:-true}"

GDN_NUM_HEADS="${GDN_NUM_HEADS:-2}"
GDN_EXPANDED_K_PAIRS="${GDN_EXPANDED_K_PAIRS:?Set GDN_EXPANDED_K_PAIRS, for example 16:1 or 8:2.}"
GDN_USE_GATE="${GDN_USE_GATE:-false}"
GDN_USE_SHORT_CONV="${GDN_USE_SHORT_CONV:-true}"
GDN_CONV_SIZE="${GDN_CONV_SIZE:-4}"
GDN_KERNEL_DTYPE="${GDN_KERNEL_DTYPE:-float32}"

RUN_ID_OVERRIDE="${RUN_ID_OVERRIDE:?Set RUN_ID_OVERRIDE.}"
LAUNCH_ID_PREFIX="${LAUNCH_ID_PREFIX:?Set LAUNCH_ID_PREFIX.}"
EXPERIMENT_MODE_OVERRIDE="${EXPERIMENT_MODE_OVERRIDE:-gdn_expanded_k}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${ROOT_DIR}/artifacts/fla-kblocked-kernel/${LAUNCH_ID_PREFIX}}"
LOG_PATH="${LOG_PATH:-${ARTIFACT_DIR}/train.log}"

EXPANDED_K_SCRIPT_DIR="${EXPANDED_K_SCRIPT_DIR:-${SCRIPT_DIR}/../20260526-gdn-expanded-k}"
METRICS_WHITE_LIST_FILE="${METRICS_WHITE_LIST_FILE:-${EXPANDED_K_SCRIPT_DIR}/metrics.yaml}"
BUILDER_SPEC="${BUILDER_SPEC:-${EXPANDED_K_SCRIPT_DIR}/config_builder.py:build_gdn_expanded_k_configs}"

export SWANLAB_MODE
export GDN_KERNEL_DTYPE
export GDN_NUM_HEADS
export GDN_EXPANDED_K_PAIRS
export GDN_USE_GATE
export GDN_USE_SHORT_CONV
export GDN_CONV_SIZE

mkdir -p "${ARTIFACT_DIR}"
cd "${ROOT_DIR}"

exec env PYTHONNOUSERSITE=1 PYTHONPATH= \
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
    --train-batch-size "${TRAIN_BATCH_SIZE}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --validations-per-epoch "${VALIDATIONS_PER_EPOCH}" \
    --disable-early-stopping "${DISABLE_EARLY_STOPPING}" \
    --cache-dir "${CACHE_DIR}" \
    --metrics-white-list-file "${METRICS_WHITE_LIST_FILE}" \
    --project "${PROJECT}" \
    --entity "${ENTITY}" \
    --launch-id-prefix "${LAUNCH_ID_PREFIX}" \
    --run-id "${RUN_ID_OVERRIDE}" \
    --experiment-mode "${EXPERIMENT_MODE_OVERRIDE}" \
    --config-builder "${BUILDER_SPEC}" \
    --gpus "${GPU_ID}" \
    > "${LOG_PATH}" 2>&1
