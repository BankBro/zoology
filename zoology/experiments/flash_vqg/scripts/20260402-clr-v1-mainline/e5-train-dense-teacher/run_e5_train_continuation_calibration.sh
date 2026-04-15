#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export LAUNCH_ID_PREFIX_E5TRAIN_BASE="${LAUNCH_ID_PREFIX_E5TRAIN_BASE:-flash-vqg-20260415-e5train-cont-cal}"
export E5TRAIN_SOURCE_LAUNCH_ID="flash-vqg-20260402-clr-v1-e3-tau-local-t025-2026-04-08-11-45-12"
export E5TRAIN_SOURCE_RUN_ID="dense-t025-s123-d123"
export E5TRAIN_ROW_WEIGHT_MODE="${E5TRAIN_ROW_WEIGHT_MODE:-uniform}"
export E5TRAIN_WARMUP_STEPS="${E5TRAIN_WARMUP_STEPS:-0}"
export SEED_VALUES="123"
export DATA_SEED="123"

CALIBRATION_GPUS="${CALIBRATION_GPUS:-0,1}"
IFS=',' read -r -a CALIBRATION_GPU_ARRAY <<< "${CALIBRATION_GPUS}"
GPU_COUNT=0
for idx in "${!CALIBRATION_GPU_ARRAY[@]}"; do
  gpu_trimmed="$(echo "${CALIBRATION_GPU_ARRAY[$idx]}" | xargs)"
  if [[ -n "${gpu_trimmed}" ]]; then
    CALIBRATION_GPU_ARRAY[$idx]="${gpu_trimmed}"
    GPU_COUNT=$((GPU_COUNT + 1))
  fi
done

run_combo() {
  local gpu_id="$1"
  local launch_prefix="$2"
  local lr_value="$3"
  local lr_tag="$4"
  local max_epochs="$5"
  (
    export GPU_ID="${gpu_id}"
    export LAUNCH_ID_PREFIX_E5TRAIN="${launch_prefix}"
    export LR="${lr_value}"
    export E5TRAIN_MAX_EPOCHS="${max_epochs}"
    export E5TRAIN_CONTROL_RUN_ID="e5cal-ctrl-e${max_epochs}-lr${lr_tag}-s123-d123"
    bash "${SCRIPT_DIR}/run_e5_train_control_single.sh"
  )
}

run_schedule() {
  local gpu_id="$1"
  local launch_prefix="$2"
  shift 2
  local combo
  for combo in "$@"; do
    IFS=':' read -r lr_value max_epochs lr_tag <<< "${combo}"
    echo "==> GPU ${gpu_id} 运行 control calibration: lr=${lr_value}, epochs=${max_epochs}"
    run_combo "${gpu_id}" "${launch_prefix}" "${lr_value}" "${lr_tag}" "${max_epochs}"
  done
}

if [[ "${GPU_COUNT}" -le 1 ]]; then
  gpu_id="${CALIBRATION_GPU_ARRAY[0]:-0}"
  launch_prefix="${LAUNCH_ID_PREFIX_E5TRAIN_BASE}"
  run_schedule "${gpu_id}" "${launch_prefix}" \
    "1e-4:1:1e4" \
    "1e-4:2:1e4" \
    "1e-4:4:1e4" \
    "3e-4:1:3e4" \
    "3e-4:2:3e4" \
    "3e-4:4:3e4" \
    "1e-3:1:1e3" \
    "1e-3:2:1e3" \
    "1e-3:4:1e3"
  exit 0
fi

gpu_a="${CALIBRATION_GPU_ARRAY[0]}"
gpu_b="${CALIBRATION_GPU_ARRAY[1]}"
launch_prefix_a="${LAUNCH_ID_PREFIX_E5TRAIN_BASE}-gpu${gpu_a}"
launch_prefix_b="${LAUNCH_ID_PREFIX_E5TRAIN_BASE}-gpu${gpu_b}"

echo "==> continuation calibration 双卡并行:"
echo "   GPU ${gpu_a}: lr=1e-4@4, lr=3e-4@4, lr=1e-3@2, lr=1e-4@1"
echo "   GPU ${gpu_b}: lr=1e-3@4, lr=1e-4@2, lr=3e-4@2, lr=1e-3@1, lr=3e-4@1"

(
  run_schedule "${gpu_a}" "${launch_prefix_a}" \
    "1e-4:4:1e4" \
    "3e-4:4:3e4" \
    "1e-3:2:1e3" \
    "1e-4:1:1e4"
) &
pid_a=$!

(
  run_schedule "${gpu_b}" "${launch_prefix_b}" \
    "1e-3:4:1e3" \
    "1e-4:2:1e4" \
    "3e-4:2:3e4" \
    "1e-3:1:1e3" \
    "3e-4:1:3e4"
) &
pid_b=$!

cleanup() {
  kill "${pid_a}" "${pid_b}" 2>/dev/null || true
}
trap cleanup INT TERM

wait "${pid_a}"
wait "${pid_b}"
