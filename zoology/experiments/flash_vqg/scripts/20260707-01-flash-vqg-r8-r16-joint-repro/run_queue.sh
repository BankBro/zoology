#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 ]]; then
  cat >&2 <<'USAGE'
Usage:
  run_queue.sh <queue-name> <target>

Queues:
  repro-2080ti-gpu0
  repro-2080ti-gpu1
  repro-3090-gpu0

Targets:
  r8-update-softcap0p5-injwarm512-rerun
  r16-update-softcap0p5-injwarm512-rerun
USAGE
  exit 2
fi

QUEUE_NAME="$1"
TARGET="$2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${ZOOLOGY_REPO_ROOT:-/home/lyj/mnt/project/zoology}"
PYTHON_BIN="${PYTHON_BIN:-/home/lyj/miniconda3/envs/flash-vqg/bin/python}"
TIMESTAMP="${R8_R16_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
MODE="${MODE:-formal}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/outputs/${QUEUE_NAME}-${TARGET}-${TIMESTAMP}-${MODE}}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-${REPO_ROOT}/zoology/experiments/flash_vqg/scripts/20260630-03-flash-vqg-s124-fixed-r4-4ep-confirm/outputs/canonical-init/cb64r16-s124-init.pt}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-704}"
SMOKE_TRAIN_STEPS="${SMOKE_TRAIN_STEPS:-8}"
SMOKE_VALIDATION_BATCHES="${SMOKE_VALIDATION_BATCHES:-16}"
CONTINUE_ON_FAIL="${CONTINUE_ON_FAIL:-1}"
export FLASH_VQG_READ_TRACE_MODE="${FLASH_VQG_READ_TRACE_MODE:-disabled}"

case "${QUEUE_NAME}" in
  repro-2080ti-gpu0)
    MACHINE_NAME="${MACHINE_NAME:-2080ti}"
    GPU="${GPU:-0}"
    ;;
  repro-2080ti-gpu1)
    MACHINE_NAME="${MACHINE_NAME:-2080ti}"
    GPU="${GPU:-1}"
    ;;
  repro-3090-gpu0)
    MACHINE_NAME="${MACHINE_NAME:-3090}"
    GPU="${GPU:-0}"
    ;;
  *)
    echo "Unknown queue: ${QUEUE_NAME}" >&2
    exit 2
    ;;
esac

case "${TARGET}" in
  r8-update-softcap0p5-injwarm512-rerun|r16-update-softcap0p5-injwarm512-rerun)
    ;;
  *)
    echo "Unknown target: ${TARGET}" >&2
    exit 2
    ;;
esac

mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/configs" "${OUTPUT_ROOT}/results" "${OUTPUT_ROOT}/preflight"

check_container_gpu_ready() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is not available inside the current container; pause experiment launch." >&2
    exit 1
  fi
  if ! nvidia-smi -i "${GPU}" >/dev/null 2>&1; then
    echo "nvidia-smi/NVML failed for GPU=${GPU} inside the current container; pause experiment launch." >&2
    exit 1
  fi
  if ! CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" - <<'PY'
import sys
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    print(
        f"torch cuda unavailable: cuda_available={torch.cuda.is_available()} "
        f"device_count={torch.cuda.device_count()}",
        file=sys.stderr,
    )
    raise SystemExit(1)
print(f"container_gpu_ready=true visible_device_count={torch.cuda.device_count()}")
PY
  then
    echo "torch.cuda readiness check failed for GPU=${GPU}; pause experiment launch." >&2
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
      if [[ "${stable_reported}" -eq 0 ]] && grep -E "Train Epoch|Valid Epoch|train/loss|valid/" "${log_path}" >/dev/null 2>&1; then
        stable_reported=1
        echo "stable-run label=${label}; queue monitor polling every ${POLL_SECONDS}s"
      fi
    fi
    sleep "${POLL_SECONDS}"
  done
  wait "${pid}"
}

run_one() {
  local max_train_steps="$1"
  local mode_label="$2"
  local max_validation_batches="${3:-}"
  local validation_args=()
  if [[ -n "${max_validation_batches}" ]]; then
    validation_args=(--max-validation-batches "${max_validation_batches}")
  fi

  "${PYTHON_BIN}" "${SCRIPT_DIR}/r8_r16_joint_repro.py" cache-hash \
    --machine-name "${MACHINE_NAME}" \
    --target "${TARGET}" \
    --variant "${TARGET}" \
    --output-json "${OUTPUT_ROOT}/cache-hash-${TARGET}.json"

  "${PYTHON_BIN}" "${SCRIPT_DIR}/r8_r16_joint_repro.py" preflight \
    --machine-name "${MACHINE_NAME}" \
    --target "${TARGET}" \
    --variant "${TARGET}" \
    --max-epochs "${MAX_EPOCHS}" \
    --max-train-steps "${max_train_steps}" \
    "${validation_args[@]}" \
    --run-suffix "${QUEUE_NAME}-${mode_label}" \
    --output-json "${OUTPUT_ROOT}/preflight-${TARGET}.json"

  "${PYTHON_BIN}" "${SCRIPT_DIR}/batch_preflight.py" \
    --machine-name "${MACHINE_NAME}" \
    --target "${TARGET}" \
    --variant "${TARGET}" \
    --max-epochs "${MAX_EPOCHS}" \
    --max-train-steps "${max_train_steps}" \
    --run-suffix "${QUEUE_NAME}-${mode_label}" \
    --output-json "${OUTPUT_ROOT}/preflight/batch-order-${TARGET}.json"

  local log_path="${OUTPUT_ROOT}/logs/${TARGET}.log"
  local trace_output_dir="${OUTPUT_ROOT}/traces/${TARGET}"
  local config_json="${OUTPUT_ROOT}/configs/${TARGET}.json"
  local result_json="${OUTPUT_ROOT}/results/${TARGET}.json"
  append_status "${QUEUE_NAME}" "${MACHINE_NAME}" "${TARGET}" "${TARGET}" "${GPU}" "" "pending" "${log_path}" "${config_json}" "${result_json}" "" ""

  local train_started_at
  train_started_at="$(date -Iseconds)"
  (
    export CUDA_VISIBLE_DEVICES="${GPU}"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/r8_r16_joint_repro.py" train \
      --machine-name "${MACHINE_NAME}" \
      --target "${TARGET}" \
      --variant "${TARGET}" \
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
  append_status "${QUEUE_NAME}" "${MACHINE_NAME}" "${TARGET}" "${TARGET}" "${GPU}" "${train_pid}" "train-started" "${log_path}" "${config_json}" "${result_json}" "${train_started_at}" ""
  echo "started train queue=${QUEUE_NAME} machine=${MACHINE_NAME} target=${TARGET} gpu=${GPU} pid=${train_pid} log=${log_path}"

  local train_status=0
  set +e
  monitor_pid "${train_pid}" "${log_path}" "${TARGET}"
  train_status=$?
  set -e
  local train_finished_at
  train_finished_at="$(date -Iseconds)"
  if [[ "${train_status}" -eq 0 ]]; then
    append_status "${QUEUE_NAME}" "${MACHINE_NAME}" "${TARGET}" "${TARGET}" "${GPU}" "${train_pid}" "completed" "${log_path}" "${config_json}" "${result_json}" "${train_started_at}" "${train_finished_at}"
    echo "[done] target=${TARGET} train_status=0 log=${log_path}"
  else
    append_status "${QUEUE_NAME}" "${MACHINE_NAME}" "${TARGET}" "${TARGET}" "${GPU}" "${train_pid}" "failed" "${log_path}" "${config_json}" "${result_json}" "${train_started_at}" "${train_finished_at}"
    echo "[failed] target=${TARGET} train_status=${train_status} log=${log_path}" >&2
    if [[ "${CONTINUE_ON_FAIL}" != "1" ]]; then
      return "${train_status}"
    fi
  fi
  return 0
}

check_container_gpu_ready

"${PYTHON_BIN}" "${SCRIPT_DIR}/r8_r16_joint_repro.py" verify-init \
  --machine-name "${MACHINE_NAME}" \
  --checkpoint "${INIT_CHECKPOINT}" \
  --output-json "${OUTPUT_ROOT}/init-verify.json"

STATUS_FILE="${OUTPUT_ROOT}/queue-status.tsv"
printf "queue\tmachine\ttarget\tvariant\tgpu\tpid\tstatus\tlog\tconfig_json\tresult_json\tstarted_at\tfinished_at\n" > "${STATUS_FILE}"
printf "%s\n" "${TARGET}" > "${OUTPUT_ROOT}/target-manifest.txt"
printf "%q " bash "${SCRIPT_DIR}/run_queue.sh" "${QUEUE_NAME}" "${TARGET}" > "${OUTPUT_ROOT}/command.txt"
printf "\n" >> "${OUTPUT_ROOT}/command.txt"

case "${MODE}" in
  smoke)
    run_one "${SMOKE_TRAIN_STEPS}" "smoke" "${SMOKE_VALIDATION_BATCHES}"
    ;;
  formal)
    run_one "${MAX_TRAIN_STEPS}" "formal"
    ;;
  *)
    echo "Unknown MODE=${MODE}" >&2
    exit 2
    ;;
esac
