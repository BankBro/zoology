#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  cat >&2 <<'USAGE'
Usage:
  run_master.sh <smoke|formal>
USAGE
  exit 2
fi

MODE="$1"
case "${MODE}" in
  smoke|formal) ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${ZOOLOGY_REPO_ROOT:-/home/lyj/mnt/project/zoology}"
TIMESTAMP="${FIXED_INIT_R8_R16_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
MASTER_ROOT="${MASTER_ROOT:-${SCRIPT_DIR}/outputs/master-${TIMESTAMP}-${MODE}}"
HOST_3090="${HOST_3090:-192.168.2.114}"
REMOTE_CONTAINER="${REMOTE_CONTAINER:-Flash-VQG-tun}"
POLL_SECONDS="${POLL_SECONDS:-60}"
SMOKE_TRAIN_STEPS="${SMOKE_TRAIN_STEPS:-8}"
SMOKE_VALIDATION_BATCHES="${SMOKE_VALIDATION_BATCHES:-16}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-704}"
CONTINUE_ON_FAIL_DEFAULT="1"
if [[ "${MODE}" == "smoke" ]]; then
  CONTINUE_ON_FAIL_DEFAULT="0"
fi
CONTINUE_ON_FAIL="${CONTINUE_ON_FAIL:-${CONTINUE_ON_FAIL_DEFAULT}}"

mkdir -p "${MASTER_ROOT}/logs"
MASTER_STATUS="${MASTER_ROOT}/master-status.tsv"
printf "timestamp\tmode\tqueue\tmachine\tgpu\tpid\tstatus\toutput_root\tlog_path\n" > "${MASTER_STATUS}"
printf "%q " bash "${SCRIPT_DIR}/run_master.sh" "${MODE}" > "${MASTER_ROOT}/command.txt"
printf "\n" >> "${MASTER_ROOT}/command.txt"
{
  echo "timestamp=${TIMESTAMP}"
  echo "mode=${MODE}"
  echo "repo_root=${REPO_ROOT}"
  echo "zoology_commit=$(git -C "${REPO_ROOT}" rev-parse --short HEAD)"
  echo "flash_vqg_commit=$(git -C /home/lyj/mnt/project/Flash-VQG rev-parse --short HEAD)"
  echo "host_3090=${HOST_3090}"
} > "${MASTER_ROOT}/env.txt"

append_master_status() {
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$(date -Iseconds)" "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" \
    >> "${MASTER_STATUS}"
}

JOB_PIDS=()
JOB_QUEUES=()
JOB_MACHINES=()
JOB_GPUS=()
JOB_OUTPUT_ROOTS=()
JOB_LOG_PATHS=()

record_job() {
  JOB_PIDS+=("$1")
  JOB_QUEUES+=("$2")
  JOB_MACHINES+=("$3")
  JOB_GPUS+=("$4")
  JOB_OUTPUT_ROOTS+=("$5")
  JOB_LOG_PATHS+=("$6")
}

launch_local_2080ti_gpu1() {
  local queue="fixed-2080ti-gpu1"
  local gpu="1"
  local output_root="${SCRIPT_DIR}/outputs/${queue}-${TIMESTAMP}-${MODE}"
  local log_path="${MASTER_ROOT}/logs/${queue}.supervisor.log"
  (
    cd "${SCRIPT_DIR}"
    FIXED_INIT_R8_R16_TIMESTAMP="${TIMESTAMP}" \
    MODE="${MODE}" \
    CONTINUE_ON_FAIL="${CONTINUE_ON_FAIL}" \
    POLL_SECONDS="${POLL_SECONDS}" \
    SMOKE_TRAIN_STEPS="${SMOKE_TRAIN_STEPS}" \
    SMOKE_VALIDATION_BATCHES="${SMOKE_VALIDATION_BATCHES}" \
    MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS}" \
    OUTPUT_ROOT="${output_root}" \
    GPU="${gpu}" \
    bash "${SCRIPT_DIR}/run_queue.sh" "${queue}"
  ) >"${log_path}" 2>&1 &
  local pid="$!"
  append_master_status "${MODE}" "${queue}" "2080ti" "${gpu}" "${pid}" "started" "${output_root}" "${log_path}"
  record_job "${pid}" "${queue}" "2080ti" "${gpu}" "${output_root}" "${log_path}"
}

launch_remote_3090_gpu0() {
  local queue="fixed-3090-gpu0"
  local gpu="0"
  local output_root="${SCRIPT_DIR}/outputs/${queue}-${TIMESTAMP}-${MODE}"
  local log_path="${MASTER_ROOT}/logs/${queue}.supervisor.log"
  local remote_cmd
  remote_cmd=$(
    printf "cd %q && FIXED_INIT_R8_R16_TIMESTAMP=%q MODE=%q CONTINUE_ON_FAIL=%q POLL_SECONDS=%q SMOKE_TRAIN_STEPS=%q SMOKE_VALIDATION_BATCHES=%q MAX_TRAIN_STEPS=%q OUTPUT_ROOT=%q GPU=%q bash %q %q" \
      "${SCRIPT_DIR}" "${TIMESTAMP}" "${MODE}" "${CONTINUE_ON_FAIL}" "${POLL_SECONDS}" \
      "${SMOKE_TRAIN_STEPS}" "${SMOKE_VALIDATION_BATCHES}" "${MAX_TRAIN_STEPS}" \
      "${output_root}" "${gpu}" "${SCRIPT_DIR}/run_queue.sh" "${queue}"
  )
  (
    ssh "lyj@${HOST_3090}" \
      "docker exec -u lyj ${REMOTE_CONTAINER} bash -lc $(printf '%q' "${remote_cmd}")"
  ) >"${log_path}" 2>&1 &
  local pid="$!"
  append_master_status "${MODE}" "${queue}" "3090" "${gpu}" "${pid}" "started" "${output_root}" "${log_path}"
  record_job "${pid}" "${queue}" "3090" "${gpu}" "${output_root}" "${log_path}"
}

launch_local_2080ti_gpu1
launch_remote_3090_gpu0

failures=0
for idx in "${!JOB_PIDS[@]}"; do
  pid="${JOB_PIDS[$idx]}"
  queue="${JOB_QUEUES[$idx]}"
  machine="${JOB_MACHINES[$idx]}"
  gpu="${JOB_GPUS[$idx]}"
  output_root="${JOB_OUTPUT_ROOTS[$idx]}"
  log_path="${JOB_LOG_PATHS[$idx]}"
  set +e
  wait "${pid}"
  status=$?
  set -e
  if [[ "${status}" -ne 0 ]]; then
    failures=$((failures + 1))
    append_master_status "${MODE}" "${queue}" "${machine}" "${gpu}" "${pid}" "failed:${status}" "${output_root}" "${log_path}"
  else
    append_master_status "${MODE}" "${queue}" "${machine}" "${gpu}" "${pid}" "completed" "${output_root}" "${log_path}"
  fi
done

if [[ "${failures}" -gt 0 ]]; then
  append_master_status "${MODE}" "master" "all" "" "$$" "failed:${failures}" "${MASTER_ROOT}" "${MASTER_ROOT}/logs"
  exit 1
fi

append_master_status "${MODE}" "master" "all" "" "$$" "completed" "${MASTER_ROOT}" "${MASTER_ROOT}/logs"
echo "[master] completed mode=${MODE} master_root=${MASTER_ROOT}"
