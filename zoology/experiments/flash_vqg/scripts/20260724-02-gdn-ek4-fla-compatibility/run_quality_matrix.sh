#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MACHINE_NAME="${MACHINE_NAME:?Set MACHINE_NAME to 2080ti or 3090}"
GPU="${GPU:?Set GPU to the container-visible physical GPU index}"
PYTHON_CURRENT="${PYTHON_CURRENT:-/home/lyj/miniconda3/envs/flash-vqg/bin/python}"
PYTHON_V042="${PYTHON_V042:-/home/lyj/miniconda3/envs/flash-vqg-fla042/bin/python}"
PYTHON_V050="${PYTHON_V050:-/home/lyj/miniconda3/envs/flash-vqg-fla050/bin/python}"
SOURCE_V042="${SOURCE_V042:-/home/lyj/mnt/project/fla-worktrees/20260724-02/v0.4.2}"
SOURCE_V050="${SOURCE_V050:-/home/lyj/mnt/project/fla-worktrees/20260724-02/v0.5.0}"
INCLUDE_CURRENT="${INCLUDE_CURRENT:-false}"
SELECTED_FLASH_VARIANT="${SELECTED_FLASH_VARIANT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/outputs/${MACHINE_NAME}/formal-quality}"

export GDN_KERNEL_DTYPE=float32
export TRITON_F32_DEFAULT=ieee
export NVIDIA_TF32_OVERRIDE=0

if ! nvidia-smi -i "${GPU}" >/dev/null 2>&1; then
  echo "容器内 nvidia-smi/NVML 检查失败: GPU=${GPU}." >&2
  exit 1
fi

python_for_variant() {
  case "$1" in
    current040) printf '%s\n' "${PYTHON_CURRENT}" ;;
    v042) printf '%s\n' "${PYTHON_V042}" ;;
    v050) printf '%s\n' "${PYTHON_V050}" ;;
    *) return 2 ;;
  esac
}

source_for_variant() {
  case "$1" in
    current040) printf '%s\n' "" ;;
    v042) printf '%s\n' "${SOURCE_V042}" ;;
    v050) printf '%s\n' "${SOURCE_V050}" ;;
    *) return 2 ;;
  esac
}

run_one() {
  local model="$1"
  local variant="$2"
  local python_bin source_root output_dir log_path
  python_bin="$(python_for_variant "${variant}")"
  source_root="$(source_for_variant "${variant}")"
  output_dir="${OUTPUT_ROOT}/${model}-${variant}"
  log_path="${output_dir}/run.log"
  mkdir -p "${output_dir}"
  echo "start machine=${MACHINE_NAME} model=${model} variant=${variant} output=${output_dir}"
  CUDA_VISIBLE_DEVICES="${GPU}" FLA_VARIANT="${variant}" FLA_SOURCE_ROOT="${source_root}" \
    "${python_bin}" "${SCRIPT_DIR}/formal_quality.py" \
      --model "${model}" --machine "${MACHINE_NAME}" --fla-variant "${variant}" \
      --run-type formal --output-dir "${output_dir}" >"${log_path}" 2>&1
  echo "completed machine=${MACHINE_NAME} model=${model} variant=${variant} result=${output_dir}/result.json"
}

mkdir -p "${OUTPUT_ROOT}"
if [[ "${INCLUDE_CURRENT}" == "true" ]]; then
  run_one gdn current040
fi
run_one gdn v042
run_one gdn v050
if [[ -n "${SELECTED_FLASH_VARIANT}" ]]; then
  run_one flash "${SELECTED_FLASH_VARIANT}"
fi
