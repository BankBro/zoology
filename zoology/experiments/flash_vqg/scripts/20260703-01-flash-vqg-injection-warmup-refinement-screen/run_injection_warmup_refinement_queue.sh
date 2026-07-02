#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  cat >&2 <<'USAGE'
Usage:
  run_injection_warmup_refinement_queue.sh <queue-name>

Queues:
  injwarmref-2080ti-gpu0-linear704-linear1024
  injwarmref-2080ti-gpu1-silent32linear704
  injwarmref-3090-gpu0-all
  injwarmref-3090-gpu0-linear704
  injwarmref-3090-gpu0-linear1024
  injwarmref-3090-gpu0-silent32linear704
USAGE
  exit 2
fi

QUEUE_NAME="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${ZOOLOGY_REPO_ROOT:-/home/lyj/mnt/project/zoology}"
PYTHON_BIN="${PYTHON_BIN:-/home/lyj/miniconda3/envs/flash-vqg/bin/python}"
TIMESTAMP="${INJECTION_WARMUP_REFINEMENT_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/outputs/${QUEUE_NAME}-${TIMESTAMP}}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-${REPO_ROOT}/zoology/experiments/flash_vqg/scripts/20260630-03-flash-vqg-s124-fixed-r4-4ep-confirm/outputs/canonical-init/cb64r16-s124-init.pt}"
POLL_SECONDS="${POLL_SECONDS:-1200}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-704}"
CAPTURE_STEPS="${CAPTURE_STEPS:-0,1,2,4,8,16,24,32,48,64,96,128,192,256,384,512,704}"

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
  injwarmref-2080ti-gpu0-linear704-linear1024)
    MACHINE_NAME="${MACHINE_NAME:-2080ti}"
    GPU="0"
    TARGETS=("inj-warmup-linear704-r2" "inj-warmup-linear1024-r2")
    ;;
  injwarmref-2080ti-gpu1-silent32linear704)
    MACHINE_NAME="${MACHINE_NAME:-2080ti}"
    GPU="1"
    TARGETS=("inj-warmup-silent32-linear704-r2")
    ;;
  injwarmref-3090-gpu0-all)
    MACHINE_NAME="${MACHINE_NAME:-3090}"
    GPU="0"
    TARGETS=("inj-warmup-linear704-r2" "inj-warmup-linear1024-r2" "inj-warmup-silent32-linear704-r2")
    ;;
  injwarmref-3090-gpu0-linear704)
    MACHINE_NAME="${MACHINE_NAME:-3090}"
    GPU="0"
    TARGETS=("inj-warmup-linear704-r2")
    ;;
  injwarmref-3090-gpu0-linear1024)
    MACHINE_NAME="${MACHINE_NAME:-3090}"
    GPU="0"
    TARGETS=("inj-warmup-linear1024-r2")
    ;;
  injwarmref-3090-gpu0-silent32linear704)
    MACHINE_NAME="${MACHINE_NAME:-3090}"
    GPU="0"
    TARGETS=("inj-warmup-silent32-linear704-r2")
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
  local label="$3"
  local stable_reported=0
  local checks=0
  while kill -0 "${pid}" >/dev/null 2>&1; do
    checks=$((checks + 1))
    if [[ -f "${log_path}" ]]; then
      if grep -E "Traceback|RuntimeError|CUDA out of memory|loss=nan|loss=inf" "${log_path}" >/dev/null 2>&1; then
        echo "detected-error label=${label} log=${log_path}" >&2
        tail -n 80 "${log_path}" >&2 || true
        return 1
      fi
      if [[ "${stable_reported}" -eq 0 ]] && grep -E "Train Epoch|Valid Epoch|train/loss|valid/|wrote " "${log_path}" >/dev/null 2>&1; then
        stable_reported=1
        echo "stable-run label=${label}; switching to explicit ${POLL_SECONDS}s polling"
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

"${PYTHON_BIN}" "${SCRIPT_DIR}/injection_warmup_refinement_screen.py" verify-init \
  --machine-name "${MACHINE_NAME}" \
  --checkpoint "${INIT_CHECKPOINT}" \
  --output-json "${OUTPUT_ROOT}/init-verify.json"

STATUS_FILE="${OUTPUT_ROOT}/queue-status.tsv"
printf "queue\tmachine\ttarget\tvariant\tgpu\tpid\tstatus\tlog\tconfig_json\tresult_json\tstarted_at\tfinished_at\n" > "${STATUS_FILE}"

overall_status=0
for target in "${TARGETS[@]}"; do
  variant="${target}"

  "${PYTHON_BIN}" "${SCRIPT_DIR}/injection_warmup_refinement_screen.py" cache-hash \
    --machine-name "${MACHINE_NAME}" \
    --target "${target}" \
    --variant "${variant}" \
    --output-json "${OUTPUT_ROOT}/cache-hash-${target}.json"

  "${PYTHON_BIN}" "${SCRIPT_DIR}/injection_warmup_refinement_screen.py" preflight \
    --machine-name "${MACHINE_NAME}" \
    --target "${target}" \
    --variant "${variant}" \
    --max-epochs "${MAX_EPOCHS}" \
    --max-train-steps "${MAX_TRAIN_STEPS}" \
    --run-suffix "${QUEUE_NAME}" \
    --output-json "${OUTPUT_ROOT}/preflight-${target}.json"

  log_path="${OUTPUT_ROOT}/logs/${target}.log"
  hash_log_path="${OUTPUT_ROOT}/logs/${target}.hash-probe.log"
  trace_output_dir="${SCRIPT_DIR}/outputs/traces/${MACHINE_NAME}/${target}/${QUEUE_NAME}"
  config_json="${OUTPUT_ROOT}/configs/${target}.json"
  result_json="${OUTPUT_ROOT}/results/${target}.json"
  hash_json="${SCRIPT_DIR}/outputs/hash-probes/${MACHINE_NAME}/${target}/${QUEUE_NAME}/hash_probe.json"
  append_status "${QUEUE_NAME}" "${MACHINE_NAME}" "${target}" "${variant}" "${GPU}" "" "pending" "${log_path}" "${config_json}" "${result_json}" "" ""

  train_started_at="$(date -Iseconds)"
  (
    export CUDA_VISIBLE_DEVICES="${GPU}"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/injection_warmup_refinement_screen.py" train \
      --machine-name "${MACHINE_NAME}" \
      --target "${target}" \
      --variant "${variant}" \
      --init-checkpoint "${INIT_CHECKPOINT}" \
      --trace-output-dir "${trace_output_dir}" \
      --output-config-json "${config_json}" \
      --output-result-json "${result_json}" \
      --logger-backend "${LOGGER_BACKEND:-none}" \
      --max-epochs "${MAX_EPOCHS}" \
      --max-train-steps "${MAX_TRAIN_STEPS}" \
      --run-suffix "${QUEUE_NAME}" >"${log_path}" 2>&1
  ) &
  train_pid="$!"
  append_status "${QUEUE_NAME}" "${MACHINE_NAME}" "${target}" "${variant}" "${GPU}" "${train_pid}" "train-started" "${log_path}" "${config_json}" "${result_json}" "${train_started_at}" ""
  echo "started train queue=${QUEUE_NAME} machine=${MACHINE_NAME} target=${target} gpu=${GPU} pid=${train_pid} log=${log_path}"

  if monitor_pid "${train_pid}" "${log_path}" "train:${target}"; then
    train_status=0
  else
    train_status=$?
  fi
  if [[ "${train_status}" -ne 0 ]]; then
    finished_at="$(date -Iseconds)"
    append_status "${QUEUE_NAME}" "${MACHINE_NAME}" "${target}" "${variant}" "${GPU}" "${train_pid}" "train-failed:${train_status}" "${log_path}" "${config_json}" "${result_json}" "${train_started_at}" "${finished_at}"
    overall_status="${train_status}"
    echo "[done] target=${target} train_status=${train_status} hash_status=not_started" >> "${log_path}"
    break
  fi

  hash_started_at="$(date -Iseconds)"
  (
    export CUDA_VISIBLE_DEVICES="${GPU}"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/injection_warmup_refinement_screen.py" hash-probe \
      --machine-name "${MACHINE_NAME}" \
      --target "${target}" \
      --init-checkpoint "${INIT_CHECKPOINT}" \
      --output-json "${hash_json}" \
      --max-optimizer-steps "${MAX_TRAIN_STEPS}" \
      --capture-optimizer-steps "${CAPTURE_STEPS}" >"${hash_log_path}" 2>&1
  ) &
  hash_pid="$!"
  append_status "${QUEUE_NAME}" "${MACHINE_NAME}" "${target}" "${variant}" "${GPU}" "${hash_pid}" "hash-started" "${hash_log_path}" "${config_json}" "${result_json}" "${hash_started_at}" ""
  echo "started hash queue=${QUEUE_NAME} machine=${MACHINE_NAME} target=${target} gpu=${GPU} pid=${hash_pid} log=${hash_log_path}"

  if monitor_pid "${hash_pid}" "${hash_log_path}" "hash:${target}"; then
    hash_status=0
  else
    hash_status=$?
  fi
  finished_at="$(date -Iseconds)"
  if [[ "${hash_status}" -eq 0 ]]; then
    append_status "${QUEUE_NAME}" "${MACHINE_NAME}" "${target}" "${variant}" "${GPU}" "${hash_pid}" "completed" "${log_path}" "${config_json}" "${result_json}" "${train_started_at}" "${finished_at}"
    echo "[done] target=${target} train_status=0 hash_status=0" >> "${log_path}"
  else
    append_status "${QUEUE_NAME}" "${MACHINE_NAME}" "${target}" "${variant}" "${GPU}" "${hash_pid}" "hash-failed:${hash_status}" "${hash_log_path}" "${config_json}" "${result_json}" "${hash_started_at}" "${finished_at}"
    overall_status="${hash_status}"
    echo "[done] target=${target} train_status=0 hash_status=${hash_status}" >> "${log_path}"
    break
  fi
done

exit "${overall_status}"
