#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  cat >&2 <<'USAGE'
Usage:
  start_early_trace_queue.sh <queue-name>

Queues:
  2080ti-smoke
  3090-smoke
  2080ti-wave1
  3090-wave1
USAGE
  exit 2
fi

QUEUE_NAME="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP="${EARLY_TRACE_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/outputs/${QUEUE_NAME}-${TIMESTAMP}}"
mkdir -p "${OUTPUT_ROOT}/logs"

QUEUE_LOG="${OUTPUT_ROOT}/queue.log"
PID_FILE="${OUTPUT_ROOT}/queue.pid"

export EARLY_TRACE_TIMESTAMP="${TIMESTAMP}"
export OUTPUT_ROOT

nohup bash "${SCRIPT_DIR}/run_early_trace_queue.sh" "${QUEUE_NAME}" >"${QUEUE_LOG}" 2>&1 &
queue_pid="$!"
printf "%s\n" "${queue_pid}" > "${PID_FILE}"

echo "queue=${QUEUE_NAME}"
echo "pid=${queue_pid}"
echo "output_root=${OUTPUT_ROOT}"
echo "queue_log=${QUEUE_LOG}"
