#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  cat >&2 <<'USAGE'
Usage:
  run_queue.sh <queue-name>

Queues:
  mstate-2080ti-gpu0
  mstate-3090-gpu0
USAGE
  exit 2
fi

QUEUE_NAME="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${ZOOLOGY_REPO_ROOT:-/home/lyj/mnt/project/zoology}"
PYTHON_BIN="${PYTHON_BIN:-/home/lyj/miniconda3/envs/flash-vqg/bin/python}"
TIMESTAMP="${MSTATE_CONTROL_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/outputs/${QUEUE_NAME}-${TIMESTAMP}}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-${REPO_ROOT}/zoology/experiments/flash_vqg/scripts/20260630-03-flash-vqg-s124-fixed-r4-4ep-confirm/outputs/canonical-init/cb64r16-s124-init.pt}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-704}"
SMOKE_TRAIN_STEPS="${SMOKE_TRAIN_STEPS:-8}"
SMOKE_VALIDATION_BATCHES="${SMOKE_VALIDATION_BATCHES:-16}"
MODE="${MODE:-formal}"
CONTINUE_ON_FAIL="${CONTINUE_ON_FAIL:-1}"
export FLASH_VQG_READ_TRACE_MODE="${FLASH_VQG_READ_TRACE_MODE:-disabled}"

mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/configs" "${OUTPUT_ROOT}/results" "${OUTPUT_ROOT}/preflight"

case "${QUEUE_NAME}" in
  mstate-2080ti-gpu0)
    MACHINE_NAME="${MACHINE_NAME:-2080ti}"
    GPU="${GPU:-0}"
    ;;
  mstate-3090-gpu0)
    MACHINE_NAME="${MACHINE_NAME:-3090}"
    GPU="${GPU:-0}"
    ;;
  *)
    echo "Unknown queue: ${QUEUE_NAME}" >&2
    exit 2
    ;;
esac

TARGETS=(
  "fixed-r16-baseline"
  "r16-update-softcap0p5"
  "r16-mnorm-cap6"
  "r16-update-softcap0p5-injwarm512"
)

if [[ -n "${MSTATE_TARGETS:-}" ]]; then
  read -r -a TARGETS <<< "${MSTATE_TARGETS}"
fi

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

append_status() {
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${10}" "${11}" "${12}" \
    >> "${STATUS_FILE}"
}

monitor_pid() {
  local pid="$1"
  local log_path="$2"
  local label="$3"
  local stable_reported=0
  while kill -0 "${pid}" >/dev/null 2>&1; do
    if [[ -f "${log_path}" ]]; then
      if grep -E "Traceback|RuntimeError|CUDA out of memory|loss=nan|loss=inf|valid/loss=nan|valid/loss=inf" "${log_path}" >/dev/null 2>&1; then
        echo "detected-error label=${label} log=${log_path}" >&2
        tail -n 80 "${log_path}" >&2 || true
        kill "${pid}" >/dev/null 2>&1 || true
        wait "${pid}" >/dev/null 2>&1 || true
        return 1
      fi
      if [[ "${stable_reported}" -eq 0 ]] && grep -E "Train Epoch|Valid Epoch|train/loss|valid/|wrote " "${log_path}" >/dev/null 2>&1; then
        stable_reported=1
        echo "stable-run label=${label}; queue monitor polling every ${POLL_SECONDS}s"
      fi
    fi
    sleep "${POLL_SECONDS}"
  done
  wait "${pid}"
}

run_one() {
  local target="$1"
  local max_train_steps="$2"
  local mode_label="$3"
  local max_validation_batches="${4:-}"
  local variant="${target}"
  local validation_args=()
  if [[ -n "${max_validation_batches}" ]]; then
    validation_args=(--max-validation-batches "${max_validation_batches}")
  fi

  "${PYTHON_BIN}" "${SCRIPT_DIR}/mstate_control.py" cache-hash \
    --machine-name "${MACHINE_NAME}" \
    --target "${target}" \
    --variant "${variant}" \
    --output-json "${OUTPUT_ROOT}/cache-hash-${target}.json"

  "${PYTHON_BIN}" "${SCRIPT_DIR}/mstate_control.py" preflight \
    --machine-name "${MACHINE_NAME}" \
    --target "${target}" \
    --variant "${variant}" \
    --max-epochs "${MAX_EPOCHS}" \
    --max-train-steps "${max_train_steps}" \
    "${validation_args[@]}" \
    --run-suffix "${QUEUE_NAME}-${mode_label}" \
    --output-json "${OUTPUT_ROOT}/preflight-${target}.json"

  "${PYTHON_BIN}" "${SCRIPT_DIR}/batch_preflight.py" \
    --machine-name "${MACHINE_NAME}" \
    --target "${target}" \
    --variant "${variant}" \
    --max-epochs "${MAX_EPOCHS}" \
    --max-train-steps "${max_train_steps}" \
    --run-suffix "${QUEUE_NAME}-${mode_label}" \
    --output-json "${OUTPUT_ROOT}/preflight/batch-order-${target}.json"

  local log_path="${OUTPUT_ROOT}/logs/${target}.log"
  local trace_output_dir="${OUTPUT_ROOT}/traces/${target}"
  local config_json="${OUTPUT_ROOT}/configs/${target}.json"
  local result_json="${OUTPUT_ROOT}/results/${target}.json"
  append_status "${QUEUE_NAME}" "${MACHINE_NAME}" "${target}" "${variant}" "${GPU}" "" "pending" "${log_path}" "${config_json}" "${result_json}" "" ""

  local train_started_at
  train_started_at="$(date -Iseconds)"
  (
    export CUDA_VISIBLE_DEVICES="${GPU}"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/mstate_control.py" train \
      --machine-name "${MACHINE_NAME}" \
      --target "${target}" \
      --variant "${variant}" \
      --init-checkpoint "${INIT_CHECKPOINT}" \
      --trace-output-dir "${trace_output_dir}" \
      --output-config-json "${config_json}" \
      --output-result-json "${result_json}" \
      --logger-backend "${LOGGER_BACKEND:-none}" \
      --max-epochs "${MAX_EPOCHS}" \
      --max-train-steps "${max_train_steps}" \
      "${validation_args[@]}" \
      --run-suffix "${QUEUE_NAME}-${mode_label}" >"${log_path}" 2>&1
  ) &
  local train_pid="$!"
  append_status "${QUEUE_NAME}" "${MACHINE_NAME}" "${target}" "${variant}" "${GPU}" "${train_pid}" "train-started" "${log_path}" "${config_json}" "${result_json}" "${train_started_at}" ""
  echo "started train queue=${QUEUE_NAME} machine=${MACHINE_NAME} target=${target} gpu=${GPU} pid=${train_pid} log=${log_path}"

  local train_status=0
  set +e
  monitor_pid "${train_pid}" "${log_path}" "${target}"
  train_status=$?
  set -e
  local train_finished_at
  train_finished_at="$(date -Iseconds)"
  if [[ "${train_status}" -eq 0 ]]; then
    append_status "${QUEUE_NAME}" "${MACHINE_NAME}" "${target}" "${variant}" "${GPU}" "${train_pid}" "completed" "${log_path}" "${config_json}" "${result_json}" "${train_started_at}" "${train_finished_at}"
    echo "[done] target=${target} train_status=0 log=${log_path}"
  else
    append_status "${QUEUE_NAME}" "${MACHINE_NAME}" "${target}" "${variant}" "${GPU}" "${train_pid}" "failed" "${log_path}" "${config_json}" "${result_json}" "${train_started_at}" "${train_finished_at}"
    echo "[failed] target=${target} train_status=${train_status} log=${log_path}" >&2
    if [[ "${CONTINUE_ON_FAIL}" != "1" ]]; then
      return "${train_status}"
    fi
  fi
  return 0
}

check_container_gpu_ready

"${PYTHON_BIN}" "${SCRIPT_DIR}/mstate_control.py" verify-init \
  --machine-name "${MACHINE_NAME}" \
  --checkpoint "${INIT_CHECKPOINT}" \
  --output-json "${OUTPUT_ROOT}/init-verify.json"

STATUS_FILE="${OUTPUT_ROOT}/queue-status.tsv"
printf "queue\tmachine\ttarget\tvariant\tgpu\tpid\tstatus\tlog\tconfig_json\tresult_json\tstarted_at\tfinished_at\n" > "${STATUS_FILE}"
printf "%s\n" "${TARGETS[@]}" > "${OUTPUT_ROOT}/target-manifest.txt"

case "${MODE}" in
  smoke)
    for target in "${TARGETS[@]}"; do
      run_one "${target}" "${SMOKE_TRAIN_STEPS}" "smoke" "${SMOKE_VALIDATION_BATCHES}"
    done
    ;;
  formal)
    for target in "${TARGETS[@]}"; do
      run_one "${target}" "${MAX_TRAIN_STEPS}" "formal"
    done
    ;;
  *)
    echo "Unknown MODE=${MODE}" >&2
    exit 2
    ;;
esac
