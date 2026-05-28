#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  cat >&2 <<'USAGE'
Usage:
  run_stage3_train.sh <target>

Targets:
  local-only   Train cb64-r16 with local_num_blocks=2 and remote disabled.
  local1       Train cb64-r16 with local_num_blocks=1 and remote enabled.
  local4       Train cb64-r16 with local_num_blocks=4 and remote enabled.

Example:
  GPU_ID=0 bash zoology/experiments/flash_vqg/scripts/20260529-flash-local-window-fairness/run_stage3_train.sh local-only
USAGE
  exit 2
fi

TARGET="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="/home/lyj/mnt/project/zoology"
PYTHON_BIN="${PYTHON_BIN:-/home/lyj/miniconda3/envs/flash-vqg/bin/python}"
PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export PYTHONPATH

case "${TARGET}" in
  local-only|local_only|localonly)
    LOCAL_VARIANT="local-only"
    LOCAL_TAG="localonly"
    LOCAL_MODE="local_only"
    ;;
  local1|local-1|local_1)
    LOCAL_VARIANT="local1"
    LOCAL_TAG="local1"
    LOCAL_MODE="local1"
    ;;
  local4|local-4|local_4)
    LOCAL_VARIANT="local4"
    LOCAL_TAG="local4"
    LOCAL_MODE="local4"
    ;;
  *)
    echo "Unknown target: ${TARGET}" >&2
    exit 2
    ;;
esac

export FLASH_LOCAL_WINDOW_VARIANT="${LOCAL_VARIANT}"
export GPU_ID="${GPU_ID:-0}"
export BACKEND="${BACKEND:-torch}"
export REMOTE_PATH_BACKEND="${REMOTE_PATH_BACKEND:-torch}"
export ANALYSIS_SOURCE="${ANALYSIS_SOURCE:-local}"
export SWANLAB_MODE="${SWANLAB_MODE:-cloud}"
export PROJECT="${PROJECT:-flash_vqg_local_window_fairness}"
export ENTITY="${ENTITY:-scu-mclab}"

export DMODEL="${DMODEL:-128}"
export LR="${LR:-1e-3}"
export TRAIN_BATCH_ORDER="${TRAIN_BATCH_ORDER:-global_shuffle}"
export SEED_VALUES="${SEED_VALUES:-123}"
export DATA_SEED="${DATA_SEED:-123}"
export NUM_CODEBOOK_VECTORS="${NUM_CODEBOOK_VECTORS:-64}"

export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
export VALIDATIONS_PER_EPOCH="${VALIDATIONS_PER_EPOCH:-2}"
export MAX_EPOCHS="${MAX_EPOCHS:-4}"
export DISABLE_EARLY_STOPPING="${DISABLE_EARLY_STOPPING:-true}"

export FOX_REMOTE_FORMULA="${FOX_REMOTE_FORMULA:-gd_residual_v1}"
export FOX_REMOTE_READ_TOPK="${FOX_REMOTE_READ_TOPK:-2}"
export FOX_GD_RESIDUAL_RANK="${FOX_GD_RESIDUAL_RANK:-16}"
export FOX_GD_RESIDUAL_WRITE_TOPK="${FOX_GD_RESIDUAL_WRITE_TOPK:-4}"
export FOX_GD_RESIDUAL_BUILDER="${FOX_GD_RESIDUAL_BUILDER:-grouped_chunk_torch_ref}"
export FOX_GD_RESIDUAL_PACK_MODE="${FOX_GD_RESIDUAL_PACK_MODE:-semivec_ref}"
export FOX_GD_RESIDUAL_CHUNK_SIZE="${FOX_GD_RESIDUAL_CHUNK_SIZE:-64}"
export FOX_GD_RESIDUAL_MU_MIN_COUNT="${FOX_GD_RESIDUAL_MU_MIN_COUNT:-0.1}"
export FOX_GD_RESIDUAL_ADDR_EPS="${FOX_GD_RESIDUAL_ADDR_EPS:-1e-6}"
export FOX_GD_RESIDUAL_DEN_EPS="${FOX_GD_RESIDUAL_DEN_EPS:-1e-6}"
export FOX_GD_RESIDUAL_RHO_EPS="${FOX_GD_RESIDUAL_RHO_EPS:-1e-12}"
export FOX_GD_RESIDUAL_BETA_INIT="${FOX_GD_RESIDUAL_BETA_INIT:-0.5}"
export FOX_GD_RESIDUAL_LAMBDA_INIT="${FOX_GD_RESIDUAL_LAMBDA_INIT:-0.05}"
export FOX_GD_RESIDUAL_NORM_WITH_GAIN="${FOX_GD_RESIDUAL_NORM_WITH_GAIN:-false}"
export FOX_GD_RESIDUAL_USE_SEPARATE_ADDR_CODEBOOK="${FOX_GD_RESIDUAL_USE_SEPARATE_ADDR_CODEBOOK:-false}"

export VQ_SCORE_MODE="${VQ_SCORE_MODE:-codebook_dot}"
export VQ_WEIGHT_MODE="${VQ_WEIGHT_MODE:-dense_softmax}"
export VQ_UPDATE_MODE="${VQ_UPDATE_MODE:-grad}"
export VQ_SOFTMAX_TAU="${VQ_SOFTMAX_TAU:-0.25}"
export VQ_TOPK="${VQ_TOPK:-4}"

export CACHE_DIR="${CACHE_DIR:-./data/flash_vqg}"
export METRICS_WHITE_LIST_FILE="${METRICS_WHITE_LIST_FILE:-${SCRIPT_DIR}/../20260425-gd-residual-v1-mqar/metrics.yaml}"
export LAUNCH_ID_PREFIX="${LAUNCH_ID_PREFIX:-flash-vqg-20260529-local-window-fairness-stage3-${LOCAL_TAG}}"
export RUN_ID_OVERRIDE="${RUN_ID_OVERRIDE:-gd-cb64-r16-s123-${LOCAL_TAG}-d123-b64-ga4-fp32-noearly4ep}"
export EXPERIMENT_MODE_OVERRIDE="${EXPERIMENT_MODE_OVERRIDE:-gd_cb64_r16_s123_${LOCAL_MODE}_d123_noearly4ep_b64_ga4_local_window_fairness_stage3}"

BUILDER_SPEC="${SCRIPT_DIR}/stage3_config_builder.py:build_stage3_train_configs"

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
  --fox-remote-read-topk-values "${FOX_REMOTE_READ_TOPK}" \
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
  --gpus "${GPU_ID}"
