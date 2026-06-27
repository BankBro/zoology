#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  cat >&2 <<'USAGE'
Usage:
  run_stability_ablation_queue.sh <queue-name>

Queues:
  2080ti-gpu0
  3090-gpu0
USAGE
  exit 2
fi

QUEUE_NAME="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/lyj/miniconda3/envs/flash-vqg/bin/python}"
TIMESTAMP="${STABILITY_ABLATION_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/outputs/${QUEUE_NAME}-${TIMESTAMP}}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-${SCRIPT_DIR}/../20260627-02-flash-vqg-canonical-init-lock-screen/outputs/canonical-init/cb64r16-s123-init.pt}"
POLL_SECONDS="${POLL_SECONDS:-1200}"

mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/configs" "${OUTPUT_ROOT}/results"

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
    TARGETS=("no-embed-dropout-s123-r1:0")
    ;;
  3090-gpu0)
    MACHINE_NAME="${MACHINE_NAME:-3090}"
    TARGETS=("no-embed-dropout-s123-r1:0" "no-embed-dropout-s123-r2:0")
    ;;
  *)
    echo "Unknown queue: ${QUEUE_NAME}" >&2
    exit 2
    ;;
esac

append_status() {
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${10}" "${11}" "${12}" \
    >> "${STATUS_FILE}"
}

monitor_pid() {
  local pid="$1"
  local log_path="$2"
  local target="$3"
  local stable_reported=0
  local checks=0
  while kill -0 "${pid}" >/dev/null 2>&1; do
    checks=$((checks + 1))
    if [[ -f "${log_path}" ]]; then
      if grep -E "Traceback|RuntimeError|CUDA out of memory|loss=nan|loss=inf" "${log_path}" >/dev/null 2>&1; then
        echo "detected-error target=${target} log=${log_path}" >&2
        tail -n 40 "${log_path}" >&2 || true
        return 1
      fi
      if [[ "${stable_reported}" -eq 0 ]] && grep -E "valid/|train/" "${log_path}" >/dev/null 2>&1; then
        stable_reported=1
        echo "stable-training target=${target}; switching to ${POLL_SECONDS}s polling"
      fi
    fi
    if [[ "${stable_reported}" -eq 0 && "${checks}" -lt 4 ]]; then
      sleep 300
    else
      sleep "${POLL_SECONDS}"
    fi
  done
  wait "${pid}"
}

check_container_gpu_ready

"${PYTHON_BIN}" "${SCRIPT_DIR}/stability_ablation.py" verify-init \
  --machine-name "${MACHINE_NAME}" \
  --checkpoint "${INIT_CHECKPOINT}" \
  --output-json "${OUTPUT_ROOT}/init-verify.json"

"${PYTHON_BIN}" "${SCRIPT_DIR}/stability_ablation.py" cache-hash \
  --machine-name "${MACHINE_NAME}" \
  --target "no-embed-dropout-s123-r1" \
  --output-json "${OUTPUT_ROOT}/cache-hash.json"

"${PYTHON_BIN}" "${SCRIPT_DIR}/stability_ablation.py" preflight \
  --machine-name "${MACHINE_NAME}" \
  --target "no-embed-dropout-s123-r1" \
  --output-json "${OUTPUT_ROOT}/preflight.json"

STATUS_FILE="${OUTPUT_ROOT}/queue-status.tsv"
printf "queue\tmachine\ttarget\tvariant\tgpu\tpid\tstatus\tlog\tconfig_json\tresult_json\tstarted_at\tfinished_at\n" > "${STATUS_FILE}"

for item in "${TARGETS[@]}"; do
  target="${item%%:*}"
  gpu="${item##*:}"
  log_path="${OUTPUT_ROOT}/logs/${target}.log"
  config_json="${OUTPUT_ROOT}/configs/${target}.json"
  result_json="${OUTPUT_ROOT}/results/${target}.json"
  append_status "${QUEUE_NAME}" "${MACHINE_NAME}" "${target}" "no-embed-dropout" "${gpu}" "" "pending" "${log_path}" "${config_json}" "${result_json}" "" ""
done

overall_status=0

for item in "${TARGETS[@]}"; do
  target="${item%%:*}"
  gpu="${item##*:}"
  log_path="${OUTPUT_ROOT}/logs/${target}.log"
  trace_output_dir="${SCRIPT_DIR}/outputs/traces/${MACHINE_NAME}/${target}"
  config_json="${OUTPUT_ROOT}/configs/${target}.json"
  result_json="${OUTPUT_ROOT}/results/${target}.json"
  started_at="$(date -Iseconds)"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/stability_ablation.py" train \
      --machine-name "${MACHINE_NAME}" \
      --target "${target}" \
      --variant "no-embed-dropout" \
      --init-checkpoint "${INIT_CHECKPOINT}" \
      --trace-output-dir "${trace_output_dir}" \
      --output-config-json "${config_json}" \
      --output-result-json "${result_json}" \
      --logger-backend "${LOGGER_BACKEND:-none}" >"${log_path}" 2>&1
  ) &
  pid="$!"
  append_status "${QUEUE_NAME}" "${MACHINE_NAME}" "${target}" "no-embed-dropout" "${gpu}" "${pid}" "started" "${log_path}" "${config_json}" "${result_json}" "${started_at}" ""
  echo "started queue=${QUEUE_NAME} machine=${MACHINE_NAME} target=${target} gpu=${gpu} pid=${pid} log=${log_path}"

  if monitor_pid "${pid}" "${log_path}" "${target}"; then
    finished_at="$(date -Iseconds)"
    append_status "${QUEUE_NAME}" "${MACHINE_NAME}" "${target}" "no-embed-dropout" "${gpu}" "${pid}" "completed" "${log_path}" "${config_json}" "${result_json}" "${started_at}" "${finished_at}"
  else
    status=$?
    finished_at="$(date -Iseconds)"
    append_status "${QUEUE_NAME}" "${MACHINE_NAME}" "${target}" "no-embed-dropout" "${gpu}" "${pid}" "failed:${status}" "${log_path}" "${config_json}" "${result_json}" "${started_at}" "${finished_at}"
    overall_status="${status}"
    echo "fail-fast queue=${QUEUE_NAME} target=${target} status=${status}" >&2
    break
  fi
done

exit "${overall_status}"
