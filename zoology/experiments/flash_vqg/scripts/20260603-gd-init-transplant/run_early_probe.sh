#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/lyj/mnt/project/zoology"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/lyj/miniconda3/envs/flash-vqg/bin/python}"
MATRIX="${MATRIX:-core}"
GPUS="${GPUS:-0,1}"
SWANLAB_MODE="${SWANLAB_MODE:-cloud}"
export SWANLAB_MODE
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

if [[ "${TORCH_DETERMINISTIC:-0}" == "1" ]]; then
  echo "本实验禁止启用 TORCH_DETERMINISTIC=1." >&2
  exit 2
fi

cd "${ROOT_DIR}"
exec "${PYTHON_BIN}" "${SCRIPT_DIR}/build_transplant_configs.py" \
  --mode early \
  --matrix "${MATRIX}" \
  --gpus "${GPUS}" \
  --parallelize \
  --launch
