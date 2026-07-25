#!/usr/bin/env bash
set -euo pipefail

MACHINE="${MQAR_PRECISION_MACHINE:?Set MQAR_PRECISION_MACHINE to 2080ti or 3090}"
case "${MACHINE}" in
  2080ti) GPU=1 ;;
  3090) GPU=0 ;;
  *) echo "Unsupported machine: ${MACHINE}" >&2; exit 2 ;;
esac

REPO_ROOT="${ZOOLOGY_REPO_ROOT:-/home/lyj/mnt/project/zoology}"
PYTHON="/home/lyj/miniconda3/envs/flash-vqg-fla042/bin/python"
SCRIPT_DIR="${REPO_ROOT}/zoology/experiments/flash_vqg/scripts/20260726-01-mqar-precision-profile"
SESSION="mqar-precision-${MACHINE}"
LOG_DIR="${SCRIPT_DIR}/outputs/machines/${MACHINE}/logs"
mkdir -p "${LOG_DIR}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "Session already exists: ${SESSION}"
  exit 0
fi

COMMAND="cd ${REPO_ROOT} && CUDA_VISIBLE_DEVICES=${GPU} MQAR_PRECISION_MACHINE=${MACHINE} TRITON_F32_DEFAULT=ieee NVIDIA_TF32_OVERRIDE=0 TORCH_DETERMINISTIC=0 ${PYTHON} ${SCRIPT_DIR}/run_queue.py --phase all >> ${LOG_DIR}/queue.log 2>&1"
tmux new-session -d -s "${SESSION}" "bash -lc '${COMMAND}'"
echo "Started ${SESSION}"
