#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  cat >&2 <<'USAGE'
Usage:
  start_master.sh <smoke|formal>
USAGE
  exit 2
fi

MODE="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP="${R8_R16_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
MASTER_ROOT="${MASTER_ROOT:-${SCRIPT_DIR}/outputs/master-${TIMESTAMP}-${MODE}}"
MASTER_LOG="${MASTER_ROOT}/master.log"
PID_FILE="${MASTER_ROOT}/master.pid"
SESSION_FILE="${MASTER_ROOT}/master.session"

mkdir -p "${MASTER_ROOT}"

export R8_R16_TIMESTAMP="${TIMESTAMP}"
export MASTER_ROOT

printf "%q " bash "${SCRIPT_DIR}/run_master.sh" "${MODE}" > "${MASTER_ROOT}/command.txt"
printf "\n" >> "${MASTER_ROOT}/command.txt"

setsid bash "${SCRIPT_DIR}/run_master.sh" "${MODE}" >"${MASTER_LOG}" 2>&1 &
master_pid="$!"
printf "%s\n" "${master_pid}" > "${PID_FILE}"
printf "setsid\n" > "${SESSION_FILE}"

echo "mode=${MODE}"
echo "pid=${master_pid}"
echo "master_root=${MASTER_ROOT}"
echo "master_log=${MASTER_LOG}"
