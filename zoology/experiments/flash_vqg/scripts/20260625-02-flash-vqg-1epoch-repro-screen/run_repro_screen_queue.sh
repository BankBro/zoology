#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  cat >&2 <<'USAGE'
Usage:
  run_repro_screen_queue.sh <queue-name>

Queues:
  2080ti-smoke
  3090-smoke
  2080ti-gpu0
  3090-gpu0
USAGE
  exit 2
fi

QUEUE_NAME="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP="${RSCREEN_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/outputs/${QUEUE_NAME}-${TIMESTAMP}}"
PYTHON_BIN="${PYTHON_BIN:-/home/lyj/miniconda3/envs/flash-vqg/bin/python}"
mkdir -p "${OUTPUT_ROOT}/logs"

check_container_gpu_ready() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is not available inside the current container; pause experiment launch." >&2
    exit 1
  fi
  if ! nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi/NVML failed inside the current container; pause experiment launch." >&2
    exit 1
  fi
  if ! "${PYTHON_BIN}" - <<'PY'
import sys
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    print(
        f"torch cuda unavailable: cuda_available={torch.cuda.is_available()} "
        f"device_count={torch.cuda.device_count()}",
        file=sys.stderr,
    )
    raise SystemExit(1)
print(f"container_gpu_ready=true device_count={torch.cuda.device_count()}")
PY
  then
    echo "torch.cuda readiness check failed inside the current container; pause experiment launch." >&2
    exit 1
  fi
}

check_container_gpu_ready

case "${QUEUE_NAME}" in
  2080ti-smoke)
    MACHINE_NAME="${MACHINE_NAME:-2080ti}"
    LOGGER_BACKEND="${LOGGER_BACKEND:-none}"
    TARGETS=(
      "smoke-default-s123:0"
    )
    ;;
  3090-smoke)
    MACHINE_NAME="${MACHINE_NAME:-3090}"
    LOGGER_BACKEND="${LOGGER_BACKEND:-none}"
    TARGETS=(
      "smoke-default-s123:0"
    )
    ;;
  2080ti-gpu0)
    MACHINE_NAME="${MACHINE_NAME:-2080ti}"
    LOGGER_BACKEND="${LOGGER_BACKEND:-none}"
    TARGETS=(
      "default-s123-r1:0"
      "default-s124-r1:0"
      "default-s123-r2:0"
      "default-s124-r2:0"
    )
    ;;
  3090-gpu0)
    MACHINE_NAME="${MACHINE_NAME:-3090}"
    LOGGER_BACKEND="${LOGGER_BACKEND:-none}"
    TARGETS=(
      "default-s123-r1:0"
      "default-s124-r1:0"
      "default-s123-r2:0"
      "default-s124-r2:0"
    )
    ;;
  *)
    echo "Unknown queue: ${QUEUE_NAME}" >&2
    exit 2
    ;;
esac

STATUS_FILE="${OUTPUT_ROOT}/queue-status.tsv"
printf "queue\ttarget\tgpu\tpid\tstatus\tlog\ttrace_output_dir\tstarted_at\tfinished_at\n" > "${STATUS_FILE}"

append_status() {
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" \
    >> "${STATUS_FILE}"
}

for item in "${TARGETS[@]}"; do
  target="${item%%:*}"
  gpu="${item##*:}"
  log_path="${OUTPUT_ROOT}/logs/${target}.log"
  trace_output_dir="${SCRIPT_DIR}/outputs/traces/${MACHINE_NAME}/${target}"
  append_status "${QUEUE_NAME}" "${target}" "${gpu}" "" "pending" "${log_path}" "${trace_output_dir}" "" ""
done

overall_status=0

for item in "${TARGETS[@]}"; do
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
    export LOGGER_BACKEND
    export TRACE_OUTPUT_DIR="${trace_output_dir}"
    bash "${SCRIPT_DIR}/run_repro_screen_train.sh" "${target}" >"${log_path}" 2>&1
    status=$?
    finished_at="$(date -Iseconds)"
    if [[ "${status}" -eq 0 ]]; then
      final_status="completed"
    else
      final_status="failed:${status}"
    fi
    append_status \
      "${QUEUE_NAME}" "${target}" "${gpu}" "${child_pid}" "${final_status}" \
      "${log_path}" "${trace_output_dir}" "${started_at}" "${finished_at}"
    exit "${status}"
  ) &
  pid="$!"
  append_status \
    "${QUEUE_NAME}" "${target}" "${gpu}" "${pid}" "started" \
    "${log_path}" "${trace_output_dir}" "${started_at}" ""
  echo "started queue=${QUEUE_NAME} target=${target} gpu=${gpu} pid=${pid} log=${log_path} trace=${trace_output_dir}"

  wait "${pid}" || {
    overall_status=$?
    echo "fail-fast queue=${QUEUE_NAME} target=${target} status=${overall_status}" >&2
    break
  }
done

exit "${overall_status}"
