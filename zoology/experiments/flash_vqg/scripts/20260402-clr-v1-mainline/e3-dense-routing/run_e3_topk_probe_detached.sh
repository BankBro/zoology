#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="/home/lyj/mnt/project/zoology"
GENERATED_DIR="${ROOT_DIR}/zoology/experiments/flash_vqg/generated"
TIMESTAMP="$(date -u +%Y-%m-%d-%H-%M-%S)"
SESSION_STAMP="${TIMESTAMP//-/}"

GPU_ID="${GPU_ID:-1}"
SESSION_NAME="${SESSION_NAME:-e3_topkwrite_probe_${SESSION_STAMP}}"
RUNNER_LOG="${RUNNER_LOG:-${GENERATED_DIR}/${SESSION_NAME}.log}"
RUNNER_SCRIPT="${RUNNER_SCRIPT:-${GENERATED_DIR}/${SESSION_NAME}.sh}"

mkdir -p "${GENERATED_DIR}"

cat >"${RUNNER_SCRIPT}" <<SH
#!/usr/bin/env bash
set -euo pipefail

cd "${ROOT_DIR}"
GPU_ID="${GPU_ID}" \
E3_TOPK_VALUES="${E3_TOPK_VALUES:-2,4}" \
E3_TOPK_TAU="${E3_TOPK_TAU:-0.25}" \
E3_TOPK_TAU_TAG="${E3_TOPK_TAU_TAG:-025}" \
E3_TOPK_SEED="${E3_TOPK_SEED:-123}" \
E3_TOPK_DATA_SEED="${E3_TOPK_DATA_SEED:-123}" \
LAUNCH_ID_PREFIX_E3_TOPK="${LAUNCH_ID_PREFIX_E3_TOPK:-flash-vqg-20260402-clr-v1-e3-topkwrite-probe-t025}" \
bash "${SCRIPT_DIR}/run_e3_topk_probe.sh"
SH
chmod +x "${RUNNER_SCRIPT}"

tmux new-session -d -s "${SESSION_NAME}" "cd ${ROOT_DIR} && bash ${RUNNER_SCRIPT} > ${RUNNER_LOG} 2>&1"

echo "session=${SESSION_NAME}"
echo "runner_log=${RUNNER_LOG}"
echo "runner_script=${RUNNER_SCRIPT}"
