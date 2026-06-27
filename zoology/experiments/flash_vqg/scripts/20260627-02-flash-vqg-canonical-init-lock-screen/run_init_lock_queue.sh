#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  cat >&2 <<'USAGE'
Usage:
  run_init_lock_queue.sh <queue-name>

Queues:
  2080ti-gpu0
  3090-gpu0
USAGE
  exit 2
fi

QUEUE_NAME="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/lyj/miniconda3/envs/flash-vqg/bin/python}"
TIMESTAMP="${INITLOCK_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/outputs/${QUEUE_NAME}-${TIMESTAMP}}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-${SCRIPT_DIR}/outputs/canonical-init/cb64r16-s123-init.pt}"

mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/configs"

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

case "${QUEUE_NAME}" in
  2080ti-gpu0)
    MACHINE_NAME="${MACHINE_NAME:-2080ti}"
    TARGETS=("default-s123-r1:0")
    ;;
  3090-gpu0)
    MACHINE_NAME="${MACHINE_NAME:-3090}"
    TARGETS=("default-s123-r1:0" "default-s123-r2:0")
    ;;
  *)
    echo "Unknown queue: ${QUEUE_NAME}" >&2
    exit 2
    ;;
esac

check_container_gpu_ready

"${PYTHON_BIN}" "${SCRIPT_DIR}/init_lock_screen.py" verify-init \
  --machine-name "${MACHINE_NAME}" \
  --checkpoint "${INIT_CHECKPOINT}" \
  --output-json "${OUTPUT_ROOT}/init-verify.json"

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
  config_json="${OUTPUT_ROOT}/configs/${target}.json"
  result_json="${OUTPUT_ROOT}/results/${target}.json"
  mkdir -p "${OUTPUT_ROOT}/results"
  started_at="$(date -Iseconds)"
  (
    set +e
    child_pid="${BASHPID:-$$}"
    export CUDA_VISIBLE_DEVICES="${gpu}"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/init_lock_screen.py" train \
      --machine-name "${MACHINE_NAME}" \
      --target "${target}" \
      --init-checkpoint "${INIT_CHECKPOINT}" \
      --trace-output-dir "${trace_output_dir}" \
      --output-config-json "${config_json}" \
      --output-result-json "${result_json}" \
      --logger-backend "${LOGGER_BACKEND:-none}" >"${log_path}" 2>&1
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
