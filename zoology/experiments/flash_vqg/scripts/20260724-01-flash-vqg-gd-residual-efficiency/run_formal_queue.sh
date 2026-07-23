#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${ZOOLOGY_REPO_ROOT:-/home/lyj/mnt/project/zoology}"
PYTHON_BIN="${PYTHON_BIN:-/home/lyj/miniconda3/envs/flash-vqg/bin/python}"
MACHINE_NAME="${MACHINE_NAME:?Set MACHINE_NAME to 2080ti or 3090}"
GPU="${GPU:?Set GPU to the container-visible physical GPU index}"
MODE="${MODE:-formal}"
TIMESTAMP="${EFFICIENCY_FORMAL_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/outputs/${MACHINE_NAME}/formal-${TIMESTAMP}}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-${REPO_ROOT}/zoology/experiments/flash_vqg/scripts/20260630-03-flash-vqg-s124-fixed-r4-4ep-confirm/outputs/canonical-init/cb64r16-s124-init.pt}"
POLL_SECONDS="${POLL_SECONDS:-30}"
if [[ "$#" -gt 0 ]]; then
  TARGETS=("$@")
else
  TARGETS=(s125-baseline-r16-joint s124-baseline-r16-joint)
fi

case "${MODE}" in
  formal)
    MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-704}"
    VALIDATION_ARGS=()
    ;;
  smoke)
    MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-1}"
    VALIDATION_ARGS=(--max-validation-batches "${MAX_VALIDATION_BATCHES:-2}")
    ;;
  *)
    echo "Unknown MODE=${MODE}" >&2
    exit 2
    ;;
esac

valid_target() {
  case "$1" in
    s124-baseline-r16-joint|s125-baseline-r16-joint) return 0 ;;
    *) return 1 ;;
  esac
}

for target in "${TARGETS[@]}"; do
  if ! valid_target "${target}"; then
    echo "Unknown target: ${target}" >&2
    exit 2
  fi
done

mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/configs" "${OUTPUT_ROOT}/results" "${OUTPUT_ROOT}/preflight"
export TRITON_F32_DEFAULT=ieee
export FLASH_VQG_READ_TRACE_MODE=disabled
export NVIDIA_TF32_OVERRIDE=0

if ! nvidia-smi -i "${GPU}" >/dev/null 2>&1; then
  echo "nvidia-smi/NVML failed for GPU=${GPU} inside the current container." >&2
  exit 1
fi
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -c \
  'import torch; assert torch.cuda.is_available() and torch.cuda.device_count() == 1; print(torch.cuda.get_device_name(0))'

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" "${SCRIPT_DIR}/efficiency_benchmark.py" preflight \
  --output "${OUTPUT_ROOT}/preflight/efficiency-preflight.json"
"${PYTHON_BIN}" "${SCRIPT_DIR}/formal_efficiency.py" verify-init \
  --machine-name "${MACHINE_NAME}" \
  --checkpoint "${INIT_CHECKPOINT}" \
  --output-json "${OUTPUT_ROOT}/preflight/init-verify.json"

STATUS_FILE="${OUTPUT_ROOT}/formal-ledger.tsv"
printf "machine\ttarget\tgpu\tpid\tstatus\tstarted_at\tfinished_at\telapsed_seconds\tlog\tconfig_json\tresult_json\n" > "${STATUS_FILE}"
git -C "${REPO_ROOT}" rev-parse HEAD > "${OUTPUT_ROOT}/zoology-commit.txt"
git -C "${REPO_ROOT}/../Flash-VQG" rev-parse HEAD > "${OUTPUT_ROOT}/flash-vqg-commit.txt"

for target in "${TARGETS[@]}"; do
  "${PYTHON_BIN}" "${SCRIPT_DIR}/formal_efficiency.py" cache-hash \
    --machine-name "${MACHINE_NAME}" \
    --target "${target}" \
    --variant "${target}" \
    --output-json "${OUTPUT_ROOT}/preflight/cache-hash-${target}.json"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/formal_efficiency.py" preflight \
    --machine-name "${MACHINE_NAME}" \
    --target "${target}" \
    --variant "${target}" \
    --max-epochs 1 \
    --max-train-steps "${MAX_TRAIN_STEPS}" \
    "${VALIDATION_ARGS[@]}" \
    --run-suffix "${MACHINE_NAME}-${MODE}" \
    --output-json "${OUTPUT_ROOT}/preflight/config-${target}.json"

  log_path="${OUTPUT_ROOT}/logs/${target}.log"
  config_json="${OUTPUT_ROOT}/configs/${target}.json"
  result_json="${OUTPUT_ROOT}/results/${target}.json"
  started_at="$(date -Iseconds)"
  started_epoch="$(date +%s)"
  (
    export CUDA_VISIBLE_DEVICES="${GPU}"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/formal_efficiency.py" train \
      --machine-name "${MACHINE_NAME}" \
      --target "${target}" \
      --variant "${target}" \
      --init-checkpoint "${INIT_CHECKPOINT}" \
      --trace-output-dir "${OUTPUT_ROOT}/traces/${target}" \
      --output-config-json "${config_json}" \
      --output-result-json "${result_json}" \
      --logger-backend none \
      --max-epochs 1 \
      --max-train-steps "${MAX_TRAIN_STEPS}" \
      "${VALIDATION_ARGS[@]}" \
      --run-suffix "${MACHINE_NAME}-${MODE}" > "${log_path}" 2>&1
  ) &
  pid="$!"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t\t\t%s\t%s\t%s\n" \
    "${MACHINE_NAME}" "${target}" "${GPU}" "${pid}" "running" "${started_at}" \
    "${log_path}" "${config_json}" "${result_json}" >> "${STATUS_FILE}"
  echo "started machine=${MACHINE_NAME} target=${target} pid=${pid} log=${log_path}"

  stable=0
  while kill -0 "${pid}" >/dev/null 2>&1; do
    if grep -E "Traceback|CUDA out of memory|loss=nan|loss=inf" "${log_path}" >/dev/null 2>&1; then
      tail -n 80 "${log_path}" >&2 || true
      kill "${pid}" >/dev/null 2>&1 || true
      wait "${pid}" >/dev/null 2>&1 || true
      exit 1
    fi
    if [[ "${stable}" -eq 0 ]] && grep -E "Train Epoch|Valid Epoch|train/loss|valid/" "${log_path}" >/dev/null 2>&1; then
      stable=1
      echo "stable machine=${MACHINE_NAME} target=${target} pid=${pid}"
    fi
    sleep "${POLL_SECONDS}"
  done
  wait "${pid}"
  finished_at="$(date -Iseconds)"
  elapsed_seconds="$(( $(date +%s) - started_epoch ))"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${MACHINE_NAME}" "${target}" "${GPU}" "${pid}" "completed" "${started_at}" \
    "${finished_at}" "${elapsed_seconds}" "${log_path}" "${config_json}" "${result_json}" >> "${STATUS_FILE}"
done
