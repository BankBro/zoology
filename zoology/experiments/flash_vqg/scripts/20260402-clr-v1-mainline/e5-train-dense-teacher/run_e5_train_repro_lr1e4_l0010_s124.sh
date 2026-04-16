#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export LAUNCH_ID_PREFIX_E5TRAIN_BASE="${LAUNCH_ID_PREFIX_E5TRAIN_BASE:-flash-vqg-20260416-e5train-s124-repro-lr1e4-l0010}"
export E5TRAIN_SOURCE_LAUNCH_ID="flash-vqg-20260402-clr-v1-e3-t025-s124-d123-2026-04-08-18-28-05"
export E5TRAIN_SOURCE_RUN_ID="dense-t025-s124-d123"
export E5TRAIN_ROW_WEIGHT_MODE="${E5TRAIN_ROW_WEIGHT_MODE:-uniform}"
export E5TRAIN_WARMUP_STEPS="${E5TRAIN_WARMUP_STEPS:-200}"
export E5TRAIN_MAX_EPOCHS="${E5TRAIN_MAX_EPOCHS:-4}"
export LR="${LR:-1e-4}"
export SEED_VALUES="124"
export DATA_SEED="123"

REPRO_GPUS="${REPRO_GPUS:-0,1}"
IFS=',' read -r -a REPRO_GPU_ARRAY <<< "${REPRO_GPUS}"
GPU_COUNT=0
for idx in "${!REPRO_GPU_ARRAY[@]}"; do
  gpu_trimmed="$(echo "${REPRO_GPU_ARRAY[$idx]}" | xargs)"
  if [[ -n "${gpu_trimmed}" ]]; then
    REPRO_GPU_ARRAY[$idx]="${gpu_trimmed}"
    GPU_COUNT=$((GPU_COUNT + 1))
  fi
done

run_control() {
  local gpu_id="$1"
  local launch_prefix="$2"
  (
    export GPU_ID="${gpu_id}"
    export LAUNCH_ID_PREFIX_E5TRAIN="${launch_prefix}"
    export E5TRAIN_LAMBDA_TAG="0000"
    export E5TRAIN_CONTROL_RUN_ID="e5train-ctrl-l0000-s124-d123-lr1e4e4"
    bash "${SCRIPT_DIR}/run_e5_train_control_single.sh"
  )
}

run_teacher() {
  local gpu_id="$1"
  local launch_prefix="$2"
  (
    export GPU_ID="${gpu_id}"
    export LAUNCH_ID_PREFIX_E5TRAIN="${launch_prefix}"
    export E5TRAIN_LAMBDA="0.01"
    export E5TRAIN_LAMBDA_TAG="0010"
    export E5TRAIN_RUN_ID="e5train-l0010-s124-d123-lr1e4e4"
    bash "${SCRIPT_DIR}/run_e5_train_single.sh"
  )
}

if [[ "${GPU_COUNT}" -le 1 ]]; then
  gpu_id="${REPRO_GPU_ARRAY[0]:-0}"
  launch_prefix="${LAUNCH_ID_PREFIX_E5TRAIN_BASE}"
  run_control "${gpu_id}" "${launch_prefix}"
  run_teacher "${gpu_id}" "${launch_prefix}"
  exit 0
fi

gpu_a="${REPRO_GPU_ARRAY[0]}"
gpu_b="${REPRO_GPU_ARRAY[1]}"
launch_prefix_a="${LAUNCH_ID_PREFIX_E5TRAIN_BASE}-gpu${gpu_a}"
launch_prefix_b="${LAUNCH_ID_PREFIX_E5TRAIN_BASE}-gpu${gpu_b}"

echo "==> s124 paired repro 双卡并行: GPU ${gpu_a} 跑 control, GPU ${gpu_b} 跑 lambda=0.010 teacher"

(
  run_control "${gpu_a}" "${launch_prefix_a}"
) &
pid_a=$!

(
  run_teacher "${gpu_b}" "${launch_prefix_b}"
) &
pid_b=$!

cleanup() {
  kill "${pid_a}" "${pid_b}" 2>/dev/null || true
}
trap cleanup INT TERM

wait "${pid_a}"
wait "${pid_b}"
