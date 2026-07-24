#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
MACHINE_NAME="${MACHINE_NAME:?Set MACHINE_NAME to 2080ti or 3090}"
GPU="${GPU:?Set GPU to the container-visible physical GPU index}"
PYTHON_V042="${PYTHON_V042:-/home/lyj/miniconda3/envs/flash-vqg-fla042/bin/python}"
SOURCE_V042="${SOURCE_V042:-/home/lyj/mnt/project/fla-worktrees/20260724-02/v0.4.2}"
REPEAT_COUNT="${REPEAT_COUNT:-3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/outputs/${MACHINE_NAME}/warmed-epoch-v042}"
EPOCH_RUNNER="${REPO_ROOT}/zoology/experiments/flash_vqg/scripts/20260724-01-flash-vqg-gd-residual-efficiency/efficiency_benchmark.py"

export FLA_VARIANT=v042
export FLA_SOURCE_ROOT="${SOURCE_V042}"
export GDN_KERNEL_DTYPE=float32
export TRITON_F32_DEFAULT=ieee
export NVIDIA_TF32_OVERRIDE=0

if ! nvidia-smi -i "${GPU}" >/dev/null 2>&1; then
  echo "容器内 nvidia-smi/NVML 检查失败: GPU=${GPU}." >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_V042}" -c \
  'import torch; assert torch.cuda.is_available(); assert not torch.backends.cuda.matmul.allow_tf32; print(torch.cuda.get_device_name(0))'

run_one() {
  local model="$1"
  local repeat_id="$2"
  local output_dir output_json log_path
  output_dir="${OUTPUT_ROOT}/${model}-r${repeat_id}"
  output_json="${output_dir}/result.json"
  log_path="${output_dir}/run.log"
  if [[ -e "${output_json}" ]]; then
    "${PYTHON_V042}" -c \
      'import json,sys; p=json.load(open(sys.argv[1], encoding="utf-8")); assert p["optimizer_steps"] == 704 and p["precompiled"] is True' \
      "${output_json}"
    echo "skip completed machine=${MACHINE_NAME} model=${model} repeat=${repeat_id} output=${output_json}"
    return
  fi
  mkdir -p "${output_dir}"
  echo "start machine=${MACHINE_NAME} model=${model} variant=v042 repeat=${repeat_id} output=${output_json}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_V042}" "${EPOCH_RUNNER}" \
    epoch --model "${model}" --repeat-id "${repeat_id}" --precompile \
    --flash-grouped-chunk-backend triton \
    --flash-selected-read-backend triton_remat \
    --output "${output_json}" >"${log_path}" 2>&1
  "${PYTHON_V042}" -c \
    'import json,sys; p=json.load(open(sys.argv[1], encoding="utf-8")); assert p["optimizer_steps"] == 704 and p["precompiled"] is True' \
    "${output_json}"
  echo "completed machine=${MACHINE_NAME} model=${model} variant=v042 repeat=${repeat_id} output=${output_json}"
}

mkdir -p "${OUTPUT_ROOT}"
for repeat_id in $(seq 1 "${REPEAT_COUNT}"); do
  if (( repeat_id % 2 == 1 )); then
    run_one gdn "${repeat_id}"
    run_one flash "${repeat_id}"
  else
    run_one flash "${repeat_id}"
    run_one gdn "${repeat_id}"
  fi
done
