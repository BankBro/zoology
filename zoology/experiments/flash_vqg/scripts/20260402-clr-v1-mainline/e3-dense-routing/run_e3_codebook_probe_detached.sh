#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="/home/lyj/mnt/project/zoology"
GENERATED_DIR="${ROOT_DIR}/zoology/experiments/flash_vqg/generated"
TIMESTAMP="$(date -u +%Y-%m-%d-%H-%M-%S)"
SESSION_STAMP="${TIMESTAMP//-/}"

GPU_ID="${GPU_ID:-0}"
SESSION_NAME="${SESSION_NAME:-e3_codebook_probe_${SESSION_STAMP}}"
RUNNER_LOG="${RUNNER_LOG:-${GENERATED_DIR}/${SESSION_NAME}.log}"
RUNNER_SCRIPT="${RUNNER_SCRIPT:-${GENERATED_DIR}/${SESSION_NAME}.sh}"

mkdir -p "${GENERATED_DIR}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "未找到 tmux, 无法启动 detached E3.5 训练." >&2
  exit 1
fi

cat >"${RUNNER_SCRIPT}" <<SH
#!/usr/bin/env bash
set -euo pipefail

cd "${ROOT_DIR}"
GPU_ID="${GPU_ID}" \
E35_CODEBOOK_VALUES="${E35_CODEBOOK_VALUES:-64,256}" \
E35_SEED_VALUES="${E35_SEED_VALUES:-123,124}" \
E35_DATA_SEED="${E35_DATA_SEED:-123}" \
LAUNCH_ID_PREFIX_E35_CB="${LAUNCH_ID_PREFIX_E35_CB:-flash-vqg-20260402-clr-v1-e35-codebook-sweep-t025}" \
bash "${SCRIPT_DIR}/run_e3_codebook_probe.sh"
SH
chmod +x "${RUNNER_SCRIPT}"

tmux new-session -d -s "${SESSION_NAME}" "cd ${ROOT_DIR} && bash ${RUNNER_SCRIPT} > ${RUNNER_LOG} 2>&1"

echo "session=${SESSION_NAME}"
echo "runner_log=${RUNNER_LOG}"
echo "runner_script=${RUNNER_SCRIPT}"
