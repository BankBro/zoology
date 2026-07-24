#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_ID="20260725-01-current-baselines-longer-mqar"
REPO_ROOT="/home/lyj/mnt/project/zoology"
SCRIPT_DIR="${REPO_ROOT}/zoology/experiments/flash_vqg/scripts/${EXPERIMENT_ID}"
PYTHON="/home/lyj/miniconda3/envs/flash-vqg-fla042/bin/python"
SESSION="${EXPERIMENT_ID}"
LOG="${SCRIPT_DIR}/outputs/queue/tmux.log"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi

mkdir -p "$(dirname "${LOG}")"
tmux new-session -d -s "${SESSION}" \
  "cd '${REPO_ROOT}' && env CUDA_VISIBLE_DEVICES=1 TRITON_F32_DEFAULT=ieee GDN_KERNEL_DTYPE=float32 NVIDIA_TF32_OVERRIDE=0 PYTHONUNBUFFERED=1 '${PYTHON}' '${SCRIPT_DIR}/queue.py' --resume >> '${LOG}' 2>&1"

printf 'started=%s\nlog=%s\n' "${SESSION}" "${LOG}"
