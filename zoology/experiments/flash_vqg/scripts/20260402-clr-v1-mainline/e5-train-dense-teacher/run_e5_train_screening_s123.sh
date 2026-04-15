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

SCREENING_GPUS="${SCREENING_GPUS:-0,1}"
IFS=',' read -r -a SCREENING_GPU_ARRAY <<< "${SCREENING_GPUS}"
GPU_COUNT=0
for idx in "${!SCREENING_GPU_ARRAY[@]}"; do
  gpu_trimmed="$(echo "${SCREENING_GPU_ARRAY[$idx]}" | xargs)"
  if [[ -n "${gpu_trimmed}" ]]; then
    SCREENING_GPU_ARRAY[$idx]="${gpu_trimmed}"
    GPU_COUNT=$((GPU_COUNT + 1))
  fi
done

run_control() {
  local gpu_id="$1"
  local launch_prefix="$2"
  (
    export GPU_ID="${gpu_id}"
    export LAUNCH_ID_PREFIX_E5TRAIN="${launch_prefix}"
    export E5TRAIN_CONTROL_RUN_ID="e5train-ctrl-l000-s123-d123"
    bash "${SCRIPT_DIR}/run_e5_train_control_single.sh"
  )
}

run_teacher() {
  local gpu_id="$1"
  local launch_prefix="$2"
  local lambda_value="$3"
  local lambda_tag="$4"
  (
    export GPU_ID="${gpu_id}"
    export LAUNCH_ID_PREFIX_E5TRAIN="${launch_prefix}"
    export E5TRAIN_LAMBDA="${lambda_value}"
    export E5TRAIN_LAMBDA_TAG="${lambda_tag}"
    export E5TRAIN_RUN_ID="e5train-l${lambda_tag}-s123-d123"
    bash "${SCRIPT_DIR}/run_e5_train_single.sh"
  )
}

if [[ "${GPU_COUNT}" -le 1 ]]; then
  gpu_id="${SCREENING_GPU_ARRAY[0]:-0}"
  run_control "${gpu_id}" "${LAUNCH_ID_PREFIX_E5TRAIN}"
  for entry in "0.02:002" "0.05:005" "0.10:010"; do
    lambda_value="${entry%%:*}"
    lambda_tag="${entry##*:}"
    run_teacher "${gpu_id}" "${LAUNCH_ID_PREFIX_E5TRAIN}" "${lambda_value}" "${lambda_tag}"
  done
  exit 0
fi

gpu_a="${SCREENING_GPU_ARRAY[0]}"
gpu_b="${SCREENING_GPU_ARRAY[1]}"
launch_prefix_a="${LAUNCH_ID_PREFIX_E5TRAIN}-gpu${gpu_a}"
launch_prefix_b="${LAUNCH_ID_PREFIX_E5TRAIN}-gpu${gpu_b}"

echo "==> P1 screening 双卡并行: GPU ${gpu_a} 跑 control + lambda=0.05, GPU ${gpu_b} 跑 lambda=0.02 + lambda=0.10"

(
  run_control "${gpu_a}" "${launch_prefix_a}"
  run_teacher "${gpu_a}" "${launch_prefix_a}" "0.05" "005"
) &
pid_a=$!

(
  run_teacher "${gpu_b}" "${launch_prefix_b}" "0.02" "002"
  run_teacher "${gpu_b}" "${launch_prefix_b}" "0.10" "010"
) &
pid_b=$!

cleanup() {
  kill "${pid_a}" "${pid_b}" 2>/dev/null || true
}
trap cleanup INT TERM

wait "${pid_a}"
wait "${pid_b}"
