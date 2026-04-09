#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="/home/lyj/mnt/project/zoology"
GENERATED_DIR="${ROOT_DIR}/zoology/experiments/flash_vqg/generated"
TIMESTAMP="$(date -u +%Y-%m-%d-%H-%M-%S)"

GPU1_SESSION_NAME="${GPU1_SESSION_NAME:-e3_tau_gpu1_${TIMESTAMP##*-}}"
GPU0_SESSION_NAME="${GPU0_SESSION_NAME:-e3_tau_gpu0_${TIMESTAMP##*-}}"
GPU1_QUEUE_LOG="${GPU1_QUEUE_LOG:-${GENERATED_DIR}/${GPU1_SESSION_NAME}.log}"
GPU0_QUEUE_LOG="${GPU0_QUEUE_LOG:-${GENERATED_DIR}/${GPU0_SESSION_NAME}.log}"
GPU0_WAIT_MANIFEST="${GPU0_WAIT_MANIFEST:-}"

mkdir -p "${GENERATED_DIR}"

create_queue_script() {
  local script_path="$1"
  shift
  cat >"${script_path}" <<'SH'
#!/usr/bin/env bash
set -euo pipefail

wait_for_manifest_terminal() {
  local manifest_path="$1"
  while true; do
    if python - "${manifest_path}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
terminal = {"completed", "failed", "interrupted"}
if not path.exists():
    sys.exit(2)
data = json.loads(path.read_text(encoding="utf-8"))
statuses = [run.get("status") for run in data.get("runs", [])]
if statuses and all(status in terminal for status in statuses):
    sys.exit(0)
sys.exit(1)
PY
    then
      break
    fi
    sleep 60
  done
}
SH
  for line in "$@"; do
    printf '%s\n' "${line}" >>"${script_path}"
  done
  chmod +x "${script_path}"
}

GPU1_SCRIPT="${GENERATED_DIR}/${GPU1_SESSION_NAME}.sh"
GPU0_SCRIPT="${GENERATED_DIR}/${GPU0_SESSION_NAME}.sh"

create_queue_script "${GPU1_SCRIPT}" \
  "cd ${ROOT_DIR}" \
  "GPU_ID=1 TAU_VALUE=0.25 TAU_TAG=025 RUN_ID=dense-t025-s123-d123 EXPERIMENT_MODE=dense_t025 SEED_VALUE=123 DATA_SEED_VALUE=123 LAUNCH_ID_PREFIX_E3_TAU=flash-vqg-20260402-clr-v1-e3-tau-local-t025 bash ${SCRIPT_DIR}/run_e3_tau_single.sh" \
  "GPU_ID=1 TAU_VALUE=0.625 TAU_TAG=0625 RUN_ID=dense-t0625-s123-d123 EXPERIMENT_MODE=dense_t0625 SEED_VALUE=123 DATA_SEED_VALUE=123 LAUNCH_ID_PREFIX_E3_TAU=flash-vqg-20260402-clr-v1-e3-tau-local-t0625 bash ${SCRIPT_DIR}/run_e3_tau_single.sh"

if [[ -n "${GPU0_WAIT_MANIFEST}" ]]; then
  wait_line="wait_for_manifest_terminal \"${GPU0_WAIT_MANIFEST}\""
else
  wait_line=":"
fi

create_queue_script "${GPU0_SCRIPT}" \
  "cd ${ROOT_DIR}" \
  "${wait_line}" \
  "GPU_ID=0 TAU_VALUE=0.375 TAU_TAG=0375 RUN_ID=dense-t0375-s123-d123 EXPERIMENT_MODE=dense_t0375 SEED_VALUE=123 DATA_SEED_VALUE=123 LAUNCH_ID_PREFIX_E3_TAU=flash-vqg-20260402-clr-v1-e3-tau-local-t0375 bash ${SCRIPT_DIR}/run_e3_tau_single.sh" \
  "GPU_ID=0 TAU_VALUE=0.75 TAU_TAG=075 RUN_ID=dense-t075-s123-d123 EXPERIMENT_MODE=dense_t075 SEED_VALUE=123 DATA_SEED_VALUE=123 LAUNCH_ID_PREFIX_E3_TAU=flash-vqg-20260402-clr-v1-e3-tau-local-t075 bash ${SCRIPT_DIR}/run_e3_tau_single.sh"

tmux new-session -d -s "${GPU1_SESSION_NAME}" "cd ${ROOT_DIR} && bash ${GPU1_SCRIPT} > ${GPU1_QUEUE_LOG} 2>&1"
tmux new-session -d -s "${GPU0_SESSION_NAME}" "cd ${ROOT_DIR} && bash ${GPU0_SCRIPT} > ${GPU0_QUEUE_LOG} 2>&1"

echo "gpu1_session=${GPU1_SESSION_NAME}"
echo "gpu1_log=${GPU1_QUEUE_LOG}"
echo "gpu1_script=${GPU1_SCRIPT}"
echo "gpu0_session=${GPU0_SESSION_NAME}"
echo "gpu0_log=${GPU0_QUEUE_LOG}"
echo "gpu0_script=${GPU0_SCRIPT}"
if [[ -n "${GPU0_WAIT_MANIFEST}" ]]; then
  echo "gpu0_wait_manifest=${GPU0_WAIT_MANIFEST}"
fi
