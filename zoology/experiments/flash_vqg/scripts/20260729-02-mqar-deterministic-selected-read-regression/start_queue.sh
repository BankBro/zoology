#!/usr/bin/env bash
set -euo pipefail

machine="${1:-}"
if [[ "${machine}" != "3090" ]]; then
  echo "Usage: $0 3090" >&2
  exit 2
fi

repo_root="${ZOOLOGY_REPO_ROOT:-/home/lyj/mnt/project/zoology}"
script_dir="${repo_root}/zoology/experiments/flash_vqg/scripts/20260729-02-mqar-deterministic-selected-read-regression"
python="/home/lyj/miniconda3/envs/flash-vqg-fla042/bin/python"
run_tag="${MQAR_DETERMINISTIC_SELECTED_RUN_TAG:-$(date +%Y%m%d-%H%M%S)-mqar-deterministic-selected}"
session="mqar-deterministic-selected-${run_tag}"

export MQAR_DETERMINISTIC_SELECTED_RUN_TAG="${run_tag}"
export CUDA_VISIBLE_DEVICES=0
export TRITON_F32_DEFAULT=ieee
export NVIDIA_TF32_OVERRIDE=0
export TORCH_DETERMINISTIC=0
export FLASH_VQG_ROOT="${FLASH_VQG_ROOT:-/home/lyj/mnt/project/Flash-VQG}"
export ZOOLOGY_REPO_ROOT="${repo_root}"

tmux new-session -d -s "${session}" \
  "cd '${repo_root}' && env MQAR_DETERMINISTIC_SELECTED_RUN_TAG='${run_tag}' CUDA_VISIBLE_DEVICES=0 TRITON_F32_DEFAULT=ieee NVIDIA_TF32_OVERRIDE=0 TORCH_DETERMINISTIC=0 FLASH_VQG_ROOT='${FLASH_VQG_ROOT}' ZOOLOGY_REPO_ROOT='${repo_root}' '${python}' '${script_dir}/run_queue.py'"

echo "session=${session}"
echo "run_tag=${run_tag}"
echo "status=${script_dir}/outputs/3090/${run_tag}/status.json"
echo "log=${script_dir}/outputs/3090/${run_tag}/logs"
