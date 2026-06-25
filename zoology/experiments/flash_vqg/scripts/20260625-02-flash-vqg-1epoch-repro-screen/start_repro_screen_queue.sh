#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  cat >&2 <<'USAGE'
Usage:
  start_repro_screen_queue.sh <queue-name>

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

QUEUE_LOG="${OUTPUT_ROOT}/queue.log"
PID_FILE="${OUTPUT_ROOT}/queue.pid"
SESSION_FILE="${OUTPUT_ROOT}/queue.session"

export RSCREEN_TIMESTAMP="${TIMESTAMP}"
export OUTPUT_ROOT

SESSION_NAME="${SESSION_NAME:-rscreen-${QUEUE_NAME}-${TIMESTAMP}}"
RUN_ENV_KEYS=(
  RSCREEN_TIMESTAMP
  OUTPUT_ROOT
  ZOOLOGY_REPO_ROOT
  FLASH_VQG_ROOT
  MACHINE_NAME
  LOGGER_BACKEND
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

RUN_CMD="cd ${ZOOLOGY_REPO_ROOT:-/home/lyj/mnt/project/zoology} && env"
for env_key in "${RUN_ENV_KEYS[@]}"; do
  if [[ "${env_key}" == "RSCREEN_TIMESTAMP" || "${env_key}" == "OUTPUT_ROOT" || -n "${!env_key+x}" ]]; then
    printf -v quoted_value "%q" "${!env_key:-}"
    RUN_CMD+=" ${env_key}=${quoted_value}"
  fi
done
printf -v quoted_script "%q" "${SCRIPT_DIR}/run_repro_screen_queue.sh"
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
