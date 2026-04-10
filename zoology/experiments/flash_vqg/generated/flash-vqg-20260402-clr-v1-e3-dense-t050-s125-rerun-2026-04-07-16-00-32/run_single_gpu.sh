#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/lyj/mnt/project/zoology"
PYTHON_BIN="/home/lyj/miniconda3/envs/flash-vqg/bin/python"
LAUNCH_ID="flash-vqg-20260402-clr-v1-e3-dense-t050-s125-rerun-2026-04-07-16-00-32"
GEN_DIR="${ROOT_DIR}/zoology/experiments/flash_vqg/generated/${LAUNCH_ID}"
LOG_FILE="${GEN_DIR}/runner.log"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export FLASH_VQG_MANIFEST_PATH="${GEN_DIR}/manifest.json"

cd "${ROOT_DIR}"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] start launch ${LAUNCH_ID}" | tee -a "${LOG_FILE}"
"${PYTHON_BIN}" -m zoology.launch "${GEN_DIR}/launch_configs.py" --launch-id "${LAUNCH_ID}" --gpus 0 2>&1 | tee -a "${LOG_FILE}"
status=${PIPESTATUS[0]}
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] launch exit status=${status}" | tee -a "${LOG_FILE}"
exit ${status}
