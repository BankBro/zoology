#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  cat >&2 <<'USAGE'
Usage:
  run_early_trace_queue.sh <queue-name>

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

case "${QUEUE_NAME}" in
  2080ti-smoke)
    MACHINE_NAME="${MACHINE_NAME:-2080ti}"
    MAX_PARALLEL="${MAX_PARALLEL:-1}"
    TARGETS=(
      "smoke-default-s123:0"
      "smoke-hard04-s123:0"
    )
    ;;
  3090-smoke)
    MACHINE_NAME="${MACHINE_NAME:-3090}"
    MAX_PARALLEL="${MAX_PARALLEL:-1}"
    TARGETS=(
      "smoke-default-s123:0"
      "smoke-hard04-s123:0"
    )
    ;;
  2080ti-wave1)
    MACHINE_NAME="${MACHINE_NAME:-2080ti}"
    MAX_PARALLEL="${MAX_PARALLEL:-2}"
    TARGETS=(
      "default-s124:0"
      "default-s123:1"
    )
    ;;
  3090-wave1)
    MACHINE_NAME="${MACHINE_NAME:-3090}"
    MAX_PARALLEL="${MAX_PARALLEL:-3}"
    TARGETS=(
      "default-s123:0"
      "hard04-s123:0"
      "default-s124:0"
    )
    ;;
  *)
    echo "Unknown queue: ${QUEUE_NAME}" >&2
    exit 2
    ;;
esac

STATUS_FILE="${OUTPUT_ROOT}/queue-status.tsv"
printf "queue\ttarget\tgpu\tpid\tstatus\tlog\ttrace_output_dir\tstarted_at\tfinished_at\n" > "${STATUS_FILE}"

active_count=0
overall_status=0

_wait_for_slot() {
  while (( active_count >= MAX_PARALLEL )); do
    if ! wait -n; then
      overall_status=1
    fi
    active_count=$((active_count - 1))
  done
}

for item in "${TARGETS[@]}"; do
  _wait_for_slot
  target="${item%%:*}"
  gpu="${item##*:}"
  log_path="${OUTPUT_ROOT}/logs/${target}.log"
  trace_output_dir="${SCRIPT_DIR}/outputs/traces/${MACHINE_NAME}/${target}"
  started_at="$(date -Iseconds)"
  (
    set +e
    child_pid="${BASHPID:-$$}"
    export GPU_ID="${gpu}"
    export MACHINE_NAME
    export TRACE_OUTPUT_DIR="${trace_output_dir}"
    bash "${SCRIPT_DIR}/run_early_trace_train.sh" "${target}" >"${log_path}" 2>&1
    status=$?
    finished_at="$(date -Iseconds)"
    if [[ "${status}" -eq 0 ]]; then
      final_status="completed"
    else
      final_status="failed:${status}"
    fi
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${QUEUE_NAME}" "${target}" "${gpu}" "${child_pid}" "${final_status}" "${log_path}" "${trace_output_dir}" "${started_at}" "${finished_at}" \
      >> "${STATUS_FILE}"
    exit "${status}"
  ) &
  pid="$!"
  active_count=$((active_count + 1))
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${QUEUE_NAME}" "${target}" "${gpu}" "${pid}" "started" "${log_path}" "${trace_output_dir}" "${started_at}" "" \
    >> "${STATUS_FILE}"
  echo "started queue=${QUEUE_NAME} target=${target} gpu=${gpu} pid=${pid} log=${log_path} trace=${trace_output_dir}"
  sleep 5
done

while (( active_count > 0 )); do
  if ! wait -n; then
    overall_status=1
  fi
  active_count=$((active_count - 1))
done

exit "${overall_status}"
