#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:?Set PYTHON_BIN to the selected environment interpreter}"
MACHINE_NAME="${MACHINE_NAME:?Set MACHINE_NAME to 2080ti or 3090}"
GPU="${GPU:?Set GPU to the container-visible physical GPU index}"
FLA_VARIANT="${FLA_VARIANT:?Set FLA_VARIANT to current040, v042 or v050}"
MODEL="${MODEL:-gdn}"
RUN_TYPE="${RUN_TYPE:-formal}"
FLA_SOURCE_ROOT="${FLA_SOURCE_ROOT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/outputs/${MACHINE_NAME}/${FLA_VARIANT}/formal-${MODEL}}"

export FLA_VARIANT FLA_SOURCE_ROOT
export GDN_KERNEL_DTYPE=float32
export TRITON_F32_DEFAULT=ieee
export NVIDIA_TF32_OVERRIDE=0

if ! nvidia-smi -i "${GPU}" >/dev/null 2>&1; then
  echo "容器内 nvidia-smi/NVML 检查失败: GPU=${GPU}." >&2
  exit 1
fi
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -c \
  'import torch; assert torch.cuda.is_available() and torch.cuda.device_count() == 1; print(torch.cuda.get_device_name(0))'

mkdir -p "${OUTPUT_ROOT}"
extra_args=()
if [[ -n "${MAX_TRAIN_STEPS:-}" ]]; then
  extra_args+=(--max-train-steps "${MAX_TRAIN_STEPS}")
fi
if [[ -n "${MAX_VALIDATION_BATCHES:-}" ]]; then
  extra_args+=(--max-validation-batches "${MAX_VALIDATION_BATCHES}")
fi
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" "${SCRIPT_DIR}/formal_quality.py" \
  --model "${MODEL}" --machine "${MACHINE_NAME}" --fla-variant "${FLA_VARIANT}" \
  --run-type "${RUN_TYPE}" \
  "${extra_args[@]}" \
  --output-dir "${OUTPUT_ROOT}"
