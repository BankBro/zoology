#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MACHINE_NAME="${MACHINE_NAME:?Set MACHINE_NAME to 2080ti or 3090}"
GPU="${GPU:?Set GPU to the container-visible physical GPU index}"
PYTHON_V042="${PYTHON_V042:-/home/lyj/miniconda3/envs/flash-vqg-fla042/bin/python}"
PYTHON_V050="${PYTHON_V050:-/home/lyj/miniconda3/envs/flash-vqg-fla050/bin/python}"
SOURCE_V042="${SOURCE_V042:-/home/lyj/mnt/project/fla-worktrees/20260724-02/v0.4.2}"
SOURCE_V050="${SOURCE_V050:-/home/lyj/mnt/project/fla-worktrees/20260724-02/v0.5.0}"
COLD_ID="${COLD_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/outputs/${MACHINE_NAME}/cold-compile-${COLD_ID}}"

export GDN_KERNEL_DTYPE=float32
export TRITON_F32_DEFAULT=ieee
export NVIDIA_TF32_OVERRIDE=0

if ! nvidia-smi -i "${GPU}" >/dev/null 2>&1; then
  echo "容器内 nvidia-smi/NVML 检查失败: GPU=${GPU}." >&2
  exit 1
fi

for variant in v042 v050; do
  if [[ "${variant}" == "v042" ]]; then
    python_bin="${PYTHON_V042}"
    source_root="${SOURCE_V042}"
  else
    python_bin="${PYTHON_V050}"
    source_root="${SOURCE_V050}"
  fi
  for phase in eval train; do
    output_dir="${OUTPUT_ROOT}/${variant}-${phase}"
    cache_dir="${output_dir}/empty-triton-cache"
    output_json="${output_dir}/compatibility.json"
    if [[ -e "${cache_dir}" || -e "${output_json}" ]]; then
      echo "拒绝覆盖已有 cold compile 输出: ${output_dir}" >&2
      exit 1
    fi
    mkdir -p "${cache_dir}"
    echo "start cold machine=${MACHINE_NAME} variant=${variant} phase=${phase}"
    CUDA_VISIBLE_DEVICES="${GPU}" FLA_VARIANT="${variant}" FLA_SOURCE_ROOT="${source_root}" \
      TRITON_CACHE_DIR="${cache_dir}" \
      "${python_bin}" "${SCRIPT_DIR}/compatibility_benchmark.py" \
        gdn-compatibility --phase "${phase}" --output "${output_json}"
    "${python_bin}" -c \
      'import json,sys; payload=json.load(open(sys.argv[1], encoding="utf-8")); assert payload["success"], payload.get("error")' \
      "${output_json}"
    echo "completed cold machine=${MACHINE_NAME} variant=${variant} phase=${phase} output=${output_json}"
  done
done
