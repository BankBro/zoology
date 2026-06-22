#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="/home/lyj/mnt/project/zoology"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/outputs/logs/${TIMESTAMP}}"

if [ $# -ne 1 ]; then
  cat >&2 <<'USAGE'
Usage: launch_wave_tmux.sh <wave>

Waves:
  wave1-3090  cb256-r8 s123, cb256-r4 s123, cb128-r8 s125 repeat, all fixed readk4
USAGE
  exit 2
fi

WAVE="$1"
mkdir -p "${LOG_DIR}"

launch_job() {
  local session_name="$1"
  local gpu_id="$2"
  local target="$3"
  local log_file="${LOG_DIR}/${target}.log"

  if tmux has-session -t "${session_name}" 2>/dev/null; then
    echo "tmux session already exists: ${session_name}" >&2
    return 1
  fi

  tmux new-session -d -s "${session_name}" \
    "cd '${ROOT_DIR}' && GPU_ID='${gpu_id}' bash '${SCRIPT_DIR}/run_fixed_readk4_train.sh' '${target}' > '${log_file}' 2>&1"
  echo "${session_name} ${target} gpu=${gpu_id} log=${log_file}"
}

case "${WAVE}" in
  wave1-3090)
    launch_job readktel-cb256r8-s123 0 cb256r8-readk4-s123
    launch_job readktel-cb256r4-s123 0 cb256r4-readk4-s123
    launch_job readktel-cb128r8-s125 0 cb128r8-readk4-s125-repeat
    ;;
  *)
    echo "Unknown wave: ${WAVE}" >&2
    exit 2
    ;;
esac

echo "log_dir=${LOG_DIR}"
