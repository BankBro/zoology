#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_ID="20260725-01-current-baselines-longer-mqar"
MACHINE="${1:-2080ti}"
REPO_ROOT="/home/lyj/mnt/project/zoology"
SCRIPT_DIR="${REPO_ROOT}/zoology/experiments/flash_vqg/scripts/${EXPERIMENT_ID}"
PYTHON="/home/lyj/miniconda3/envs/flash-vqg-fla042/bin/python"

case "${MACHINE}" in
  2080ti)
    CUDA_DEVICE="1"
    SESSION="${EXPERIMENT_ID}"
    OUTPUT_ROOT="${SCRIPT_DIR}/outputs"
    ;;
  3090)
    CUDA_DEVICE="0"
    SESSION="${EXPERIMENT_ID}-3090"
    OUTPUT_ROOT="${SCRIPT_DIR}/outputs/machines/3090"
    ;;
  *)
    echo "machine must be 2080ti or 3090: ${MACHINE}" >&2
    exit 2
    ;;
esac

LOG="${OUTPUT_ROOT}/queue/tmux.log"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi

mkdir -p "$(dirname "${LOG}")"
tmux new-session -d -s "${SESSION}" \
  "cd '${REPO_ROOT}' && env LONGER_MQAR_MACHINE='${MACHINE}' CUDA_VISIBLE_DEVICES='${CUDA_DEVICE}' TRITON_F32_DEFAULT=ieee GDN_KERNEL_DTYPE=float32 NVIDIA_TF32_OVERRIDE=0 PYTHONUNBUFFERED=1 '${PYTHON}' '${SCRIPT_DIR}/run_queue.py' --machine '${MACHINE}' --resume >> '${LOG}' 2>&1"

printf 'started=%s\nmachine=%s\nlog=%s\n' "${SESSION}" "${MACHINE}" "${LOG}"
