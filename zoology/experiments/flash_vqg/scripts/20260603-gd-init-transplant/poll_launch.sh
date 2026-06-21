#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  cat >&2 <<'USAGE'
Usage:
  poll_launch.sh <launch_id>

The default interval is 1800 seconds. Override with INTERVAL_SECONDS=1200.
USAGE
  exit 2
fi

ROOT_DIR="/home/lyj/mnt/project/zoology"
LAUNCH_ID="$1"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-1800}"
GENERATED_DIR="${ROOT_DIR}/zoology/experiments/flash_vqg/generated/${LAUNCH_ID}"
CHECKPOINT_DIR="${ROOT_DIR}/checkpoints/${LAUNCH_ID}"
RESULTS_DIR="${ROOT_DIR}/zoology/analysis/flash_vqg/results/${LAUNCH_ID}"

while true; do
  date -Is
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
  if [[ -d "${GENERATED_DIR}" ]]; then
    find "${GENERATED_DIR}" -maxdepth 1 -type f -printf '%TY-%Tm-%TdT%TH:%TM:%TS %p\n' | sort
  fi
  if [[ -d "${CHECKPOINT_DIR}" ]]; then
    find "${CHECKPOINT_DIR}" -maxdepth 2 -type f \( -name best.pt -o -name last.pt -o -name train_config.json \) -printf '%TY-%Tm-%TdT%TH:%TM:%TS %p\n' | sort | tail -n 30
  fi
  if [[ -d "${RESULTS_DIR}" ]]; then
    find "${RESULTS_DIR}" -maxdepth 3 -type f -printf '%TY-%Tm-%TdT%TH:%TM:%TS %p\n' | sort | tail -n 30
  fi
  sleep "${INTERVAL_SECONDS}"
done
