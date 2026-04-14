#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export LAUNCH_ID_PREFIX_E5TRAIN="${LAUNCH_ID_PREFIX_E5TRAIN:-flash-vqg-20260414-e5train-s123-screening}"
export E5TRAIN_SOURCE_LAUNCH_ID="flash-vqg-20260402-clr-v1-e3-tau-local-t025-2026-04-08-11-45-12"
export E5TRAIN_SOURCE_RUN_ID="dense-t025-s123-d123"
export E5TRAIN_ROW_WEIGHT_MODE="${E5TRAIN_ROW_WEIGHT_MODE:-uniform}"
export E5TRAIN_WARMUP_STEPS="${E5TRAIN_WARMUP_STEPS:-200}"
export E5TRAIN_MAX_EPOCHS="${E5TRAIN_MAX_EPOCHS:-4}"
export SEED_VALUES="123"
export DATA_SEED="123"

export E5TRAIN_CONTROL_RUN_ID="e5train-ctrl-l000-s123-d123"
bash "${SCRIPT_DIR}/run_e5_train_control_single.sh"

for entry in "0.02:002" "0.05:005" "0.10:010"; do
  lambda_value="${entry%%:*}"
  lambda_tag="${entry##*:}"
  export E5TRAIN_LAMBDA="${lambda_value}"
  export E5TRAIN_LAMBDA_TAG="${lambda_tag}"
  export E5TRAIN_RUN_ID="e5train-l${lambda_tag}-s123-d123"
  bash "${SCRIPT_DIR}/run_e5_train_single.sh"
done
