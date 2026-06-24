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
SESSION_FILE="${OUTPUT_ROOT}/queue.session"

export EARLY_TRACE_TIMESTAMP="${TIMESTAMP}"
export OUTPUT_ROOT

SESSION_NAME="${SESSION_NAME:-etrace-${QUEUE_NAME}-${TIMESTAMP}}"
RUN_ENV_KEYS=(
  EARLY_TRACE_TIMESTAMP
  OUTPUT_ROOT
  MACHINE_NAME
  LOGGER_BACKEND
  MAX_PARALLEL
  PYTHON_BIN
  CACHE_DIR
  SWANLAB_MODE
  PROJECT
  ENTITY
  ANALYSIS_SOURCE
  TRAIN_BATCH_SIZE
  EVAL_BATCH_SIZE
  GRADIENT_ACCUMULATION_STEPS
  READ_TRACE_TRAIN_STEPS
  READ_TRACE_VALID_BATCHES
  READ_TRACE_MAX_SAMPLES
  READ_TRACE_MAX_QUERIES_PER_SAMPLE
  READ_CHURN_PROBE_VALID_BATCHES
  READ_CHURN_PROBE_MAX_SAMPLES
)

RUN_CMD="cd /home/lyj/mnt/project/zoology && env"
for env_key in "${RUN_ENV_KEYS[@]}"; do
  if [[ "${env_key}" == "EARLY_TRACE_TIMESTAMP" || "${env_key}" == "OUTPUT_ROOT" || -n "${!env_key+x}" ]]; then
    printf -v quoted_value "%q" "${!env_key:-}"
    RUN_CMD+=" ${env_key}=${quoted_value}"
  fi
done
printf -v quoted_script "%q" "${SCRIPT_DIR}/run_early_trace_queue.sh"
printf -v quoted_queue "%q" "${QUEUE_NAME}"
printf -v quoted_log "%q" "${QUEUE_LOG}"
RUN_CMD+=" bash ${quoted_script} ${quoted_queue} >${quoted_log} 2>&1"

if command -v tmux >/dev/null 2>&1; then
  tmux new-session -d -s "${SESSION_NAME}" "${RUN_CMD}"
  queue_pid="$(tmux display-message -p -t "${SESSION_NAME}" '#{pane_pid}')"
  printf "%s\n" "${SESSION_NAME}" > "${SESSION_FILE}"
else
  setsid bash -lc "${RUN_CMD}" &
  queue_pid="$!"
  printf "setsid\n" > "${SESSION_FILE}"
fi
printf "%s\n" "${queue_pid}" > "${PID_FILE}"

echo "queue=${QUEUE_NAME}"
echo "pid=${queue_pid}"
echo "session=${SESSION_NAME}"
echo "output_root=${OUTPUT_ROOT}"
echo "queue_log=${QUEUE_LOG}"
