#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_ID="20260625-02-flash-vqg-1epoch-repro-screen"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
HOST_MNT_ROOT="${HOST_MNT_ROOT:-$(cd "${REPO_ROOT}/../.." && pwd)}"
CONTAINER_MNT_ROOT="${CONTAINER_MNT_ROOT:-/home/lyj/mnt}"
CONTAINER_REPO_ROOT="${CONTAINER_MNT_ROOT}/project/zoology"
CONTAINER_FLASH_VQG_ROOT="${CONTAINER_MNT_ROOT}/project/Flash-VQG"
CONTAINER_SCRIPT_DIR="${CONTAINER_REPO_ROOT}/zoology/experiments/flash_vqg/scripts/${EXPERIMENT_ID}"
RUNNER_IMAGE="${RUNNER_IMAGE:-flash-vqg-tun-snapshot:0.1}"
HOST_HF_CACHE="${HOST_HF_CACHE:-/home/lyj/.cache/huggingface}"
HOST_DATA_ROOT="${HOST_DATA_ROOT:-/home/lyj/docker/Flash-VQG-tun/data}"
HOST_UID="${HOST_UID:-$(id -u)}"
HOST_GID="${HOST_GID:-$(id -g)}"
PYTHON_BIN="${PYTHON_BIN:-/home/lyj/miniconda3/envs/flash-vqg/bin/python}"

usage() {
  cat >&2 <<'USAGE'
用法:
  run_repro_screen_host.sh preflight <machine-name> <train|smoke> [gpu-device]
  run_repro_screen_host.sh queue <queue-name> [gpu-device]

示例:
  bash run_repro_screen_host.sh preflight 2080ti train 0
  bash run_repro_screen_host.sh queue 2080ti-gpu0 0
USAGE
  exit 2
}

if [[ "$#" -lt 1 ]]; then
  usage
fi

MODE="$1"
shift

ensure_host_paths() {
  for path in "${HOST_MNT_ROOT}" "${HOST_HF_CACHE}" "${HOST_DATA_ROOT}"; do
    if [[ ! -e "${path}" ]]; then
      echo "Host path does not exist: ${path}" >&2
      exit 1
    fi
  done
}

gpu_spec_from_arg() {
  local raw="${1:-0}"
  if [[ "${raw}" == "all" ]]; then
    printf "all"
  else
    printf "device=%s" "${raw}"
  fi
}

machine_name_from_queue() {
  local queue_name="$1"
  case "${queue_name}" in
    2080ti-*)
      printf "2080ti"
      ;;
    3090-*)
      printf "3090"
      ;;
    *)
      echo "Unsupported queue name: ${queue_name}" >&2
      exit 2
      ;;
  esac
}

docker_prefix() {
  local gpu_spec="$1"
  printf \
    'docker run --rm --gpus %q -v %q:%q -v %q:/home/lyj/.cache/huggingface -v %q:/data -w %q %q bash -lc' \
    "${gpu_spec}" \
    "${HOST_MNT_ROOT}" "${CONTAINER_MNT_ROOT}" \
    "${HOST_HF_CACHE}" \
    "${HOST_DATA_ROOT}" \
    "${CONTAINER_REPO_ROOT}" \
    "${RUNNER_IMAGE}"
}

write_docker_command_script() {
  local script_path="$1"
  local gpu_spec="$2"
  local inner_cmd="$3"
  cat > "${script_path}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
$(docker_prefix "${gpu_spec}") $(printf '%q' "${inner_cmd}")
EOF
  chmod +x "${script_path}"
}

run_preflight() {
  if [[ "$#" -lt 2 || "$#" -gt 3 ]]; then
    usage
  fi
  local machine_name="$1"
  local preflight_mode="$2"
  local gpu_device="${3:-0}"
  local timestamp="${RSCREEN_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
  local host_output_dir="${SCRIPT_DIR}/outputs/${machine_name}-preflight-${timestamp}"
  local container_output_json="${CONTAINER_SCRIPT_DIR}/outputs/${machine_name}-preflight-${timestamp}/preflight-${preflight_mode}.json"
  local gpu_spec
  gpu_spec="$(gpu_spec_from_arg "${gpu_device}")"

  ensure_host_paths
  mkdir -p "${host_output_dir}"

  local inner_cmd
  printf -v inner_cmd \
    'chmod o+rx /home/lyj && cd %q && export ZOOLOGY_REPO_ROOT=%q FLASH_VQG_ROOT=%q && exec setpriv --reuid=%q --regid=%q --clear-groups env HOME=%q %q %q --machine-name %q --mode %q --output-json %q' \
    "${CONTAINER_REPO_ROOT}" \
    "${CONTAINER_REPO_ROOT}" \
    "${CONTAINER_FLASH_VQG_ROOT}" \
    "${HOST_UID}" \
    "${HOST_GID}" \
    "${CONTAINER_MNT_ROOT}" \
    "${PYTHON_BIN}" \
    "${CONTAINER_SCRIPT_DIR}/preflight_repro_screen.py" \
    "${machine_name}" \
    "${preflight_mode}" \
    "${container_output_json}"

  local command_script="${host_output_dir}/run-preflight.sh"
  write_docker_command_script "${command_script}" "${gpu_spec}" "${inner_cmd}"
  echo "machine=${machine_name}"
  echo "mode=${preflight_mode}"
  echo "gpu_spec=${gpu_spec}"
  echo "output_json=${host_output_dir}/preflight-${preflight_mode}.json"
  bash "${command_script}"
}

run_queue() {
  if [[ "$#" -lt 1 || "$#" -gt 2 ]]; then
    usage
  fi
  local queue_name="$1"
  local gpu_device="${2:-0}"
  local machine_name
  machine_name="$(machine_name_from_queue "${queue_name}")"
  local timestamp="${RSCREEN_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
  local host_output_root="${SCRIPT_DIR}/outputs/${queue_name}-${timestamp}"
  local container_output_root="${CONTAINER_SCRIPT_DIR}/outputs/${queue_name}-${timestamp}"
  local queue_log="${host_output_root}/queue.log"
  local host_runner_log="${host_output_root}/host-runner.log"
  local pid_file="${host_output_root}/host-runner.pid"
  local session_file="${host_output_root}/host-runner.session"
  local command_file="${host_output_root}/host-runner.command.sh"
  local session_name="${SESSION_NAME:-hrscreen-${queue_name}-${timestamp}}"
  local gpu_spec
  gpu_spec="$(gpu_spec_from_arg "${gpu_device}")"

  ensure_host_paths
  mkdir -p "${host_output_root}/logs"

  local env_keys=(
    RSCREEN_TIMESTAMP
    OUTPUT_ROOT
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

  local inner_cmd="chmod o+rx /home/lyj && cd ${CONTAINER_REPO_ROOT} && mkdir -p ${container_output_root@Q}/logs && env"
  local env_key
  for env_key in "${env_keys[@]}"; do
    local env_value=""
    case "${env_key}" in
      RSCREEN_TIMESTAMP)
        env_value="${timestamp}"
        ;;
      OUTPUT_ROOT)
        env_value="${container_output_root}"
        ;;
      MACHINE_NAME)
        env_value="${machine_name}"
        ;;
      PYTHON_BIN)
        env_value="${PYTHON_BIN}"
        ;;
      *)
        if [[ -n "${!env_key+x}" ]]; then
          env_value="${!env_key}"
        else
          continue
        fi
        ;;
    esac
    printf -v inner_cmd '%s %s=%q' "${inner_cmd}" "${env_key}" "${env_value}"
  done
  printf -v inner_cmd '%s %s=%q %s=%q' \
    "${inner_cmd}" \
    "ZOOLOGY_REPO_ROOT" "${CONTAINER_REPO_ROOT}" \
    "FLASH_VQG_ROOT" "${CONTAINER_FLASH_VQG_ROOT}"
  printf -v inner_cmd '%s setpriv --reuid=%q --regid=%q --clear-groups env HOME=%q bash %q %q >%q 2>&1' \
    "${inner_cmd}" \
    "${HOST_UID}" \
    "${HOST_GID}" \
    "${CONTAINER_MNT_ROOT}" \
    "${CONTAINER_SCRIPT_DIR}/run_repro_screen_queue.sh" \
    "${queue_name}" \
    "${container_output_root}/queue.log"

  write_docker_command_script "${command_file}" "${gpu_spec}" "${inner_cmd}"
  if command -v tmux >/dev/null 2>&1; then
    tmux new-session -d -s "${session_name}" "bash ${command_file@Q} >${host_runner_log@Q} 2>&1"
    local queue_pid
    queue_pid="$(tmux display-message -p -t "${session_name}" '#{pane_pid}')"
    printf "%s\n" "${session_name}" > "${session_file}"
    printf "%s\n" "${queue_pid}" > "${pid_file}"
  else
    setsid bash "${command_file}" >"${host_runner_log}" 2>&1 &
    local queue_pid="$!"
    printf "setsid\n" > "${session_file}"
    printf "%s\n" "${queue_pid}" > "${pid_file}"
  fi

  echo "queue=${queue_name}"
  echo "machine=${machine_name}"
  echo "gpu_spec=${gpu_spec}"
  echo "output_root=${host_output_root}"
  echo "queue_log=${queue_log}"
  echo "host_runner_log=${host_runner_log}"
}

case "${MODE}" in
  preflight)
    run_preflight "$@"
    ;;
  queue)
    run_queue "$@"
    ;;
  *)
    usage
    ;;
esac
