#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:?Set PYTHON_BIN to the candidate environment interpreter}"
MACHINE_NAME="${MACHINE_NAME:?Set MACHINE_NAME to 2080ti or 3090}"
GPU="${GPU:?Set GPU to the container-visible physical GPU index}"
FLA_VARIANT="${FLA_VARIANT:?Set FLA_VARIANT to current040, v042 or v050}"
FLA_SOURCE_ROOT="${FLA_SOURCE_ROOT:-}"
REPEATS="${REPEATS:-5}"
MODELS="${MODELS:-gdn flash}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/outputs/${MACHINE_NAME}/${FLA_VARIANT}/benchmark}"

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
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" "${SCRIPT_DIR}/compatibility_benchmark.py" \
  preflight --output "${OUTPUT_ROOT}/preflight.json"

for phase in train eval; do
  compatibility_json="${OUTPUT_ROOT}/gdn-${phase}-compatibility.json"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" "${SCRIPT_DIR}/compatibility_benchmark.py" \
    gdn-compatibility --phase "${phase}" --output "${compatibility_json}"
  "${PYTHON_BIN}" -c \
    'import json,sys; payload=json.load(open(sys.argv[1], encoding="utf-8")); assert payload["success"], payload.get("error")' \
    "${compatibility_json}"
done

for model in ${MODELS}; do
  flash_args=()
  if [[ "${model}" == "flash" ]]; then
    flash_args=(--flash-grouped-chunk-backend triton --flash-selected-read-backend triton_remat)
  fi
  for phase in train eval; do
    for repeat in $(seq 1 "${REPEATS}"); do
      CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" "${SCRIPT_DIR}/compatibility_benchmark.py" \
        run --model "${model}" --phase "${phase}" --metrics-mode core --run-kind timing \
        --warmup 5 --active 10 --repeat-id "${repeat}" \
        --output-dir "${OUTPUT_ROOT}/${model}-${phase}-timing-r${repeat}" \
        "${flash_args[@]}"
    done
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" "${SCRIPT_DIR}/compatibility_benchmark.py" \
      run --model "${model}" --phase "${phase}" --metrics-mode core --run-kind memory \
      --warmup 5 --active 1 --repeat-id 1 \
      --output-dir "${OUTPUT_ROOT}/${model}-${phase}-memory-r1" \
      "${flash_args[@]}"
  done
done
