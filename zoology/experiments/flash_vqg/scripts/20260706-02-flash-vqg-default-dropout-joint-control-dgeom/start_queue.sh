#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  cat >&2 <<'USAGE'
Usage:
  start_queue.sh <queue-name>
USAGE
  exit 2
fi

QUEUE_NAME="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP="${JOINT_DGEOM_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/outputs/${QUEUE_NAME}-${TIMESTAMP}}"
QUEUE_LOG="${OUTPUT_ROOT}/queue.log"
PID_FILE="${OUTPUT_ROOT}/queue.pid"
SESSION_FILE="${OUTPUT_ROOT}/queue.session"

mkdir -p "${OUTPUT_ROOT}"

export OUTPUT_ROOT
export JOINT_DGEOM_TIMESTAMP="${TIMESTAMP}"

printf "%q " bash "${SCRIPT_DIR}/run_queue.sh" "${QUEUE_NAME}" > "${OUTPUT_ROOT}/command.txt"
printf "\n" >> "${OUTPUT_ROOT}/command.txt"

setsid bash "${SCRIPT_DIR}/run_queue.sh" "${QUEUE_NAME}" >"${QUEUE_LOG}" 2>&1 &
queue_pid="$!"
printf "%s\n" "${queue_pid}" > "${PID_FILE}"
printf "setsid\n" > "${SESSION_FILE}"

echo "queue=${QUEUE_NAME}"
echo "pid=${queue_pid}"
echo "output_root=${OUTPUT_ROOT}"
echo "queue_log=${QUEUE_LOG}"
