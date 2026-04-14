#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${BEST_LAMBDA:?需要设置 BEST_LAMBDA}"
: "${BEST_LAMBDA_TAG:?需要设置 BEST_LAMBDA_TAG}"

export LAUNCH_ID_PREFIX_E5TRAIN="${LAUNCH_ID_PREFIX_E5TRAIN:-flash-vqg-20260414-e5train-s124-repro}"
export E5TRAIN_SOURCE_LAUNCH_ID="flash-vqg-20260402-clr-v1-e3-t025-s124-d123-2026-04-08-18-28-05"
export E5TRAIN_SOURCE_RUN_ID="dense-t025-s124-d123"
export E5TRAIN_ROW_WEIGHT_MODE="${E5TRAIN_ROW_WEIGHT_MODE:-uniform}"
export E5TRAIN_WARMUP_STEPS="${E5TRAIN_WARMUP_STEPS:-200}"
export E5TRAIN_MAX_EPOCHS="${E5TRAIN_MAX_EPOCHS:-4}"
export SEED_VALUES="124"
export DATA_SEED="123"

export E5TRAIN_CONTROL_RUN_ID="e5train-ctrl-l000-s124-d123"
bash "${SCRIPT_DIR}/run_e5_train_control_single.sh"

export E5TRAIN_LAMBDA="${BEST_LAMBDA}"
export E5TRAIN_LAMBDA_TAG="${BEST_LAMBDA_TAG}"
export E5TRAIN_RUN_ID="e5train-l${BEST_LAMBDA_TAG}-s124-d123"
bash "${SCRIPT_DIR}/run_e5_train_single.sh"
