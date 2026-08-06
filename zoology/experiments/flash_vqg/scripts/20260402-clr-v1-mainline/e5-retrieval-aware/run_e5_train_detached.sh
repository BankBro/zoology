#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
GENERATED_DIR="${ROOT_DIR}/zoology/experiments/flash_vqg/generated"
mkdir -p "${GENERATED_DIR}"

choose_gpu() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
    while IFS=, read -r idx mem util; do
      idx="${idx//[[:space:]]/}"
      mem="${mem//[[:space:]]/}"
      util="${util//[[:space:]]/}"
      if [[ "${mem}" =~ ^[0-9]+$ && "${util}" =~ ^[0-9]+$ ]]; then
        if (( mem <= 1024 && util <= 10 )); then
          printf '%s\n' "${idx}"
          return 0
        fi
      fi
    done
}

if [[ -z "${GPU_ID:-}" ]]; then
  GPU_ID="$(choose_gpu || true)"
fi
if [[ -z "${GPU_ID:-}" ]]; then
  echo "未找到空闲 GPU: 需要 memory.used <= 1024 MiB 且 utilization.gpu <= 10%." >&2
  exit 1
fi

TS="$(date +%Y%m%d%H%M%S)"
SESSION_NAME="${SESSION_NAME:-e5_retaware_${TS}}"
RUNNER_SCRIPT="${GENERATED_DIR}/${SESSION_NAME}.sh"
RUNNER_LOG="${GENERATED_DIR}/${SESSION_NAME}.log"

cat > "${RUNNER_SCRIPT}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${ROOT_DIR}"
export GPU_ID="${GPU_ID}"
export FLASH_VQG_E5_ROOT="${FLASH_VQG_E5_ROOT:-/home/lyj/mnt/project/Flash-VQG-e5}"
export PYTHON_BIN="${PYTHON_BIN:-/home/lyj/miniconda3/envs/flash-vqg/bin/python}"
export E5_SEED="${E5_SEED:-123}"
export E5_DATA_SEED="${E5_DATA_SEED:-123}"
export LAUNCH_ID_PREFIX_E5="${LAUNCH_ID_PREFIX_E5:-flash-vqg-20260402-clr-v1-e5-retaware}"
export E5_CONFIG_BUILDER_FUNC="${E5_CONFIG_BUILDER_FUNC:-build_e5_train_configs}"
export LOGGER_BACKEND="${LOGGER_BACKEND:-swanlab}"
export ANALYSIS_SOURCE="${ANALYSIS_SOURCE:-local}"
export MAX_EPOCHS="${MAX_EPOCHS:-32}"
bash "${SCRIPT_DIR}/run_e5_train.sh"
EOF
chmod +x "${RUNNER_SCRIPT}"

tmux new-session -d -s "${SESSION_NAME}" "bash '${RUNNER_SCRIPT}' >> '${RUNNER_LOG}' 2>&1"

echo "session=${SESSION_NAME}"
echo "gpu=${GPU_ID}"
echo "runner_script=${RUNNER_SCRIPT}"
echo "runner_log=${RUNNER_LOG}"
