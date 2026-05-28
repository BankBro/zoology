#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  cat >&2 <<'USAGE'
Usage:
  run_train.sh <target>

Targets:
  cb256-r4-s124
  cb256-r4-s125
  cb64-r16-s124
  cb64-r16-s125

Example:
  GPU_ID=0 bash zoology/experiments/flash_vqg/scripts/20260528-flash-seed-stability/run_train.sh cb64-r16-s124
USAGE
  exit 2
fi

TARGET="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHASE2_RUN="${SCRIPT_DIR}/../20260526-gdn-flash-fairness-phase2/run_train.sh"

case "${TARGET}" in
  cb256-r4-s124)
    export SEED_VALUES=124
    export NUM_CODEBOOK_VECTORS=256
    export FOX_GD_RESIDUAL_RANK=4
    ;;
  cb256-r4-s125)
    export SEED_VALUES=125
    export NUM_CODEBOOK_VECTORS=256
    export FOX_GD_RESIDUAL_RANK=4
    ;;
  cb64-r16-s124)
    export SEED_VALUES=124
    export NUM_CODEBOOK_VECTORS=64
    export FOX_GD_RESIDUAL_RANK=16
    ;;
  cb64-r16-s125)
    export SEED_VALUES=125
    export NUM_CODEBOOK_VECTORS=64
    export FOX_GD_RESIDUAL_RANK=16
    ;;
  *)
    echo "Unknown target: ${TARGET}" >&2
    exit 2
    ;;
esac

export PYTHON_BIN="${PYTHON_BIN:-/home/lyj/miniconda3/envs/flash-vqg/bin/python}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
export VALIDATIONS_PER_EPOCH="${VALIDATIONS_PER_EPOCH:-2}"
export DATA_SEED="${DATA_SEED:-123}"
export MAX_EPOCHS="${MAX_EPOCHS:-4}"
export DISABLE_EARLY_STOPPING="${DISABLE_EARLY_STOPPING:-true}"
export FOX_REMOTE_FORMULA="${FOX_REMOTE_FORMULA:-gd_residual_v1}"
export FOX_REMOTE_READ_TOPK="${FOX_REMOTE_READ_TOPK:-2}"
export FOX_GD_RESIDUAL_WRITE_TOPK="${FOX_GD_RESIDUAL_WRITE_TOPK:-4}"
export FOX_GD_RESIDUAL_BUILDER="${FOX_GD_RESIDUAL_BUILDER:-grouped_chunk_torch_ref}"
export FOX_GD_RESIDUAL_PACK_MODE="${FOX_GD_RESIDUAL_PACK_MODE:-semivec_ref}"
export FOX_GD_RESIDUAL_CHUNK_SIZE="${FOX_GD_RESIDUAL_CHUNK_SIZE:-64}"
export FOX_GD_RESIDUAL_MU_MIN_COUNT="${FOX_GD_RESIDUAL_MU_MIN_COUNT:-0.1}"
export FOX_GD_RESIDUAL_BETA_INIT="${FOX_GD_RESIDUAL_BETA_INIT:-0.5}"
export FOX_GD_RESIDUAL_LAMBDA_INIT="${FOX_GD_RESIDUAL_LAMBDA_INIT:-0.05}"
export SWANLAB_MODE="${SWANLAB_MODE:-cloud}"
export PROJECT="${PROJECT:-flash_vqg_gdn_flash_seed_stability}"
export ENTITY="${ENTITY:-scu-mclab}"
export GPU_ID="${GPU_ID:-0}"

export RUN_ID_OVERRIDE="gd-${TARGET}-d123-b64-ga4-fp32-noearly4ep"
EXPERIMENT_TARGET="${TARGET//-/_}"
export EXPERIMENT_MODE_OVERRIDE="gd_${EXPERIMENT_TARGET}_d123_noearly4ep_b64_ga4_seed_stability_repro"
export LAUNCH_ID_PREFIX="${LAUNCH_ID_PREFIX:-flash-vqg-20260528-flash-seed-stability-${TARGET}}"

exec bash "${PHASE2_RUN}"
