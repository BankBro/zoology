#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ROOT_DIR="/home/lyj/mnt/project/zoology"
PYTHON_BIN="${PYTHON_BIN:-/home/lyj/miniconda3/envs/flash-vqg/bin/python}"
PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export PYTHONPATH

if [ $# -lt 1 ]; then
  cat >&2 <<'USAGE'
Usage: run_train.sh <target> [extra_args...]

Diagnostic targets:
  d1-s124          cb256-r4-s124 baseline
  d1-s124-det      cb256-r4-s124 deterministic (TF32 off)
  d1-s125          cb256-r4-s125 baseline (good seed)

Fix-probe targets:
  f1-beta010-s124  cb256-r4-s124 beta_init=0.1
  f1-beta010-s125  cb256-r4-s125 beta_init=0.1
  f2-lambda015-s124 cb256-r4-s124 lambda_init=0.15

Extended:
  d4-s124-8ep      cb256-r4-s124 8 epochs

Zero-residual:
  d3-s124          cb256-r4-s124 lambda_init=0.0

Orthogonal addr_proj:
  a1-s124          cb256-r4-s124 addr_proj orthogonal init

Example:
  GPU_ID=0 bash run_train.sh d1-s124
  GPU_ID=1 bash run_train.sh d1-s124-det
USAGE
  exit 2
fi

TARGET="$1"
shift

# default overrides
export SWANLAB_MODE="${SWANLAB_MODE:-cloud}"

GPU_ID="${GPU_ID:-0}"
ANALYSIS_SOURCE="${ANALYSIS_SOURCE:-local}"
BACKEND="${BACKEND:-torch}"
REMOTE_PATH_BACKEND="${REMOTE_PATH_BACKEND:-torch}"

DMODEL=128
LR=1e-3
TRAIN_BATCH_ORDER=global_shuffle
DATA_SEED=123
TRAIN_BATCH_SIZE=64
EVAL_BATCH_SIZE=16
GRADIENT_ACCUMULATION_STEPS=4
VALIDATIONS_PER_EPOCH=2
DISABLE_EARLY_STOPPING=true

NUM_CODEBOOK_VECTORS=256
FOX_REMOTE_FORMULA=gd_residual_v1
FOX_REMOTE_READ_TOPK=2
FOX_GD_RESIDUAL_RANK=4
FOX_GD_RESIDUAL_WRITE_TOPK=4
FOX_GD_RESIDUAL_BUILDER=grouped_chunk_torch_ref
FOX_GD_RESIDUAL_PACK_MODE=semivec_ref
FOX_GD_RESIDUAL_CHUNK_SIZE=64
FOX_GD_RESIDUAL_MU_MIN_COUNT=0.1
FOX_GD_RESIDUAL_BETA_INIT=0.5
FOX_GD_RESIDUAL_LAMBDA_INIT=0.05
FOX_GD_RESIDUAL_ADDR_PROJ_ORTHOGONAL_INIT=false
IF_REMOTE_ENABLED=true

PROJECT=flash_vqg_gd_seed_diag
ENTITY=scu-mclab
MAX_EPOCHS=4
DETERMINISTIC=false

METRICS_YAML="${SCRIPT_DIR}/../20260425-gd-residual-v1-mqar/metrics.yaml"
BUILDER_SPEC="${SCRIPT_DIR}/../20260425-gd-residual-v1-mqar/config_builder.py:build_gd_residual_v1_train_configs"

case "${TARGET}" in
  d1-s124)
    SEED_VALUES=124
    ;;
  d1-s124-det)
    SEED_VALUES=124
    DETERMINISTIC=true
    ;;
  d1-s125)
    SEED_VALUES=125
    ;;
  d1-s125-det)
    SEED_VALUES=125
    DETERMINISTIC=true
    ;;
  # cb64-r16 targets
  cb64r16-s124)
    SEED_VALUES=124
    NUM_CODEBOOK_VECTORS=64
    FOX_GD_RESIDUAL_RANK=16
    ;;
  cb64r16-s124-det)
    SEED_VALUES=124
    NUM_CODEBOOK_VECTORS=64
    FOX_GD_RESIDUAL_RANK=16
    DETERMINISTIC=true
    ;;
  cb64r16-s125)
    SEED_VALUES=125
    NUM_CODEBOOK_VECTORS=64
    FOX_GD_RESIDUAL_RANK=16
    ;;
  cb64r16-s125-det)
    SEED_VALUES=125
    NUM_CODEBOOK_VECTORS=64
    FOX_GD_RESIDUAL_RANK=16
    DETERMINISTIC=true
    ;;
  a1-s124)
    SEED_VALUES=124
    FOX_GD_RESIDUAL_ADDR_PROJ_ORTHOGONAL_INIT=true
    ;;
  f1-beta010-s124)
    SEED_VALUES=124
    FOX_GD_RESIDUAL_BETA_INIT=0.1
    ;;
  f1-beta010-s125)
    SEED_VALUES=125
    FOX_GD_RESIDUAL_BETA_INIT=0.1
    ;;
  f2-lambda015-s124)
    SEED_VALUES=124
    FOX_GD_RESIDUAL_LAMBDA_INIT=0.15
    ;;
  f3-beta010lambda015-s124)
    SEED_VALUES=124
    FOX_GD_RESIDUAL_BETA_INIT=0.1
    FOX_GD_RESIDUAL_LAMBDA_INIT=0.15
    ;;
  d3-s124)
    SEED_VALUES=124
    IF_REMOTE_ENABLED=false
    BUILDER_SPEC=""  # skip builder, use CLI path which respects --if-remote-enabled
    ;;
  d4-s124-8ep)
    SEED_VALUES=124
    MAX_EPOCHS=8
    ;;
  *)
    echo "Unknown target: ${TARGET}" >&2
    exit 2
    ;;
esac

LAUNCH_ID_PREFIX="flash-vqg-20260530-gd-seed-diag-${TARGET}"
RUN_ID_OVERRIDE="gd-diag-${TARGET}-d${DATA_SEED}-b64-ga4-fp32-noearly4ep"
EXPERIMENT_MODE_OVERRIDE="gd_diag_${TARGET}_d${DATA_SEED}_noearly4ep"

cd "${ROOT_DIR}"

if [ "${DETERMINISTIC}" = "true" ]; then
  export TORCH_DETERMINISTIC=1
fi

BUILDER_ARGS=()
if [ -n "${BUILDER_SPEC}" ]; then
  BUILDER_ARGS=(--config-builder "${BUILDER_SPEC}")
fi

exec "${PYTHON_BIN}" -m zoology.experiments.flash_vqg.run_flash_vqg_suite \
  --flash-only \
  --logger-backend swanlab \
  --analysis "${ANALYSIS_SOURCE}" \
  --backend "${BACKEND}" \
  --dmodels "${DMODEL}" \
  --learning-rates "${LR}" \
  --max-epochs "${MAX_EPOCHS}" \
  --train-batch-order "${TRAIN_BATCH_ORDER}" \
  --if-remote-enabled "${IF_REMOTE_ENABLED}" \
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
  --fox-gd-residual-beta-init "${FOX_GD_RESIDUAL_BETA_INIT}" \
  --fox-gd-residual-lambda-init "${FOX_GD_RESIDUAL_LAMBDA_INIT}" \
  --fox-gd-residual-addr-proj-orthogonal-init "${FOX_GD_RESIDUAL_ADDR_PROJ_ORTHOGONAL_INIT}" \
  --vq-score-mode "${VQ_SCORE_MODE:-codebook_dot}" \
  --vq-weight-mode "${VQ_WEIGHT_MODE:-dense_softmax}" \
  --vq-update-mode "${VQ_UPDATE_MODE:-grad}" \
  --vq-softmax-tau "${VQ_SOFTMAX_TAU:-0.25}" \
  --vq-topk "${VQ_TOPK:-4}" \
  --train-batch-size "${TRAIN_BATCH_SIZE}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --validations-per-epoch "${VALIDATIONS_PER_EPOCH}" \
  --disable-early-stopping "${DISABLE_EARLY_STOPPING}" \
  --cache-dir "${CACHE_DIR:-./data/flash_vqg}" \
  --metrics-white-list-file "${METRICS_YAML}" \
  --project "${PROJECT}" \
  --entity "${ENTITY}" \
  --launch-id-prefix "${LAUNCH_ID_PREFIX}" \
  --run-id "${RUN_ID_OVERRIDE}" \
  --experiment-mode "${EXPERIMENT_MODE_OVERRIDE}" \
  "${BUILDER_ARGS[@]}" \
  --gpus "${GPU_ID}" \
  "$@"
