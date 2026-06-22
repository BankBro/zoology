#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/lyj/mnt/project/zoology"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/lyj/miniconda3/envs/flash-vqg/bin/python}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/config-runtime-smoke-${TIMESTAMP}}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

cd "${ROOT_DIR}"
exec "${PYTHON_BIN}" "${SCRIPT_DIR}/config_runtime_smoke.py" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"
