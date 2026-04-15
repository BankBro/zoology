#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export LAUNCH_ID_PREFIX_E5TRAIN_BASE="${LAUNCH_ID_PREFIX_E5TRAIN_BASE:-flash-vqg-20260415-e5train-s123-rescreen-lr1e4}"
export E5TRAIN_SOURCE_LAUNCH_ID="flash-vqg-20260402-clr-v1-e3-tau-local-t025-2026-04-08-11-45-12"
export E5TRAIN_SOURCE_RUN_ID="dense-t025-s123-d123"
export E5TRAIN_ROW_WEIGHT_MODE="${E5TRAIN_ROW_WEIGHT_MODE:-uniform}"
export E5TRAIN_WARMUP_STEPS="${E5TRAIN_WARMUP_STEPS:-200}"
export E5TRAIN_MAX_EPOCHS="${E5TRAIN_MAX_EPOCHS:-4}"
export LR="${LR:-1e-4}"
export SEED_VALUES="123"
export DATA_SEED="123"

RESCREENING_GPUS="${RESCREENING_GPUS:-0,1}"
IFS=',' read -r -a RESCREENING_GPU_ARRAY <<< "${RESCREENING_GPUS}"
GPU_COUNT=0
for idx in "${!RESCREENING_GPU_ARRAY[@]}"; do
  gpu_trimmed="$(echo "${RESCREENING_GPU_ARRAY[$idx]}" | xargs)"
  if [[ -n "${gpu_trimmed}" ]]; then
    RESCREENING_GPU_ARRAY[$idx]="${gpu_trimmed}"
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
    export E5TRAIN_CONTROL_RUN_ID="e5train-ctrl-l0000-s123-d123-lr1e4e4"
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
    export E5TRAIN_RUN_ID="e5train-l${lambda_tag}-s123-d123-lr1e4e4"
    bash "${SCRIPT_DIR}/run_e5_train_single.sh"
  )
}

if [[ "${GPU_COUNT}" -le 1 ]]; then
  gpu_id="${RESCREENING_GPU_ARRAY[0]:-0}"
  run_control "${gpu_id}" "${LAUNCH_ID_PREFIX_E5TRAIN_BASE}"
  for entry in "0.005:0005" "0.01:0010" "0.02:0020"; do
    lambda_value="${entry%%:*}"
    lambda_tag="${entry##*:}"
    run_teacher "${gpu_id}" "${LAUNCH_ID_PREFIX_E5TRAIN_BASE}" "${lambda_value}" "${lambda_tag}"
  done
  exit 0
fi

gpu_a="${RESCREENING_GPU_ARRAY[0]}"
gpu_b="${RESCREENING_GPU_ARRAY[1]}"
launch_prefix_a="${LAUNCH_ID_PREFIX_E5TRAIN_BASE}-gpu${gpu_a}"
launch_prefix_b="${LAUNCH_ID_PREFIX_E5TRAIN_BASE}-gpu${gpu_b}"

echo "==> lr=1e-4 rescreening 双卡并行: GPU ${gpu_a} 跑 control + lambda=0.01, GPU ${gpu_b} 跑 lambda=0.005 + lambda=0.02"

(
  run_control "${gpu_a}" "${launch_prefix_a}"
  run_teacher "${gpu_a}" "${launch_prefix_a}" "0.01" "0010"
) &
pid_a=$!

(
  run_teacher "${gpu_b}" "${launch_prefix_b}" "0.005" "0005"
  run_teacher "${gpu_b}" "${launch_prefix_b}" "0.02" "0020"
) &
pid_b=$!

cleanup() {
  kill "${pid_a}" "${pid_b}" 2>/dev/null || true
}
trap cleanup INT TERM

wait "${pid_a}"
wait "${pid_b}"
