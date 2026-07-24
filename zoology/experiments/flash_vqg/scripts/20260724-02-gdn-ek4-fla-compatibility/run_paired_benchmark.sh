#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_V042="${PYTHON_V042:-/home/lyj/miniconda3/envs/flash-vqg-fla042/bin/python}"
PYTHON_V050="${PYTHON_V050:-/home/lyj/miniconda3/envs/flash-vqg-fla050/bin/python}"
SOURCE_V042="${SOURCE_V042:-/home/lyj/mnt/project/fla-worktrees/20260724-02/v0.4.2}"
SOURCE_V050="${SOURCE_V050:-/home/lyj/mnt/project/fla-worktrees/20260724-02/v0.5.0}"
MACHINE_NAME="${MACHINE_NAME:?Set MACHINE_NAME to 2080ti or 3090}"
GPU="${GPU:?Set GPU to the container-visible physical GPU index}"
REPEATS="${REPEATS:-5}"
MODELS="${MODELS:-gdn flash}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/outputs/${MACHINE_NAME}/paired-benchmark}"

export GDN_KERNEL_DTYPE=float32
export TRITON_F32_DEFAULT=ieee
export NVIDIA_TF32_OVERRIDE=0

if ! nvidia-smi -i "${GPU}" >/dev/null 2>&1; then
  echo "容器内 nvidia-smi/NVML 检查失败: GPU=${GPU}." >&2
  exit 1
fi

python_for_variant() {
  case "$1" in
    v042) printf '%s\n' "${PYTHON_V042}" ;;
    v050) printf '%s\n' "${PYTHON_V050}" ;;
    *) return 2 ;;
  esac
}

source_for_variant() {
  case "$1" in
    v042) printf '%s\n' "${SOURCE_V042}" ;;
    v050) printf '%s\n' "${SOURCE_V050}" ;;
    *) return 2 ;;
  esac
}

run_candidate() {
  local variant="$1"
  shift
  local python_bin source_root
  python_bin="$(python_for_variant "${variant}")"
  source_root="$(source_for_variant "${variant}")"
  CUDA_VISIBLE_DEVICES="${GPU}" FLA_VARIANT="${variant}" FLA_SOURCE_ROOT="${source_root}" \
    "${python_bin}" "$@"
}

mkdir -p "${OUTPUT_ROOT}"
for variant in v042 v050; do
  run_candidate "${variant}" "${SCRIPT_DIR}/compatibility_benchmark.py" \
    preflight --output "${OUTPUT_ROOT}/${variant}/preflight.json"
  for phase in train eval; do
    compatibility_json="${OUTPUT_ROOT}/${variant}/gdn-${phase}-compatibility.json"
    run_candidate "${variant}" "${SCRIPT_DIR}/compatibility_benchmark.py" \
      gdn-compatibility --phase "${phase}" --output "${compatibility_json}"
    "$(python_for_variant "${variant}")" -c \
      'import json,sys; payload=json.load(open(sys.argv[1], encoding="utf-8")); assert payload["success"], payload.get("error")' \
      "${compatibility_json}"
  done
done

for model in ${MODELS}; do
  flash_args=()
  if [[ "${model}" == "flash" ]]; then
    flash_args=(--flash-grouped-chunk-backend triton --flash-selected-read-backend triton_remat)
  fi
  for phase in train eval; do
    for repeat in $(seq 1 "${REPEATS}"); do
      if (( repeat % 2 == 1 )); then
        order=(v042 v050)
      else
        order=(v050 v042)
      fi
      for variant in "${order[@]}"; do
        run_candidate "${variant}" "${SCRIPT_DIR}/compatibility_benchmark.py" \
          run --model "${model}" --phase "${phase}" --metrics-mode core --run-kind timing \
          --warmup 5 --active 10 --repeat-id "${repeat}" \
          --output-dir "${OUTPUT_ROOT}/${variant}/${model}-${phase}-timing-r${repeat}" \
          "${flash_args[@]}"
      done
    done
    for variant in v042 v050; do
      run_candidate "${variant}" "${SCRIPT_DIR}/compatibility_benchmark.py" \
        run --model "${model}" --phase "${phase}" --metrics-mode core --run-kind memory \
        --warmup 5 --active 1 --repeat-id 1 \
        --output-dir "${OUTPUT_ROOT}/${variant}/${model}-${phase}-memory-r1" \
        "${flash_args[@]}"
    done
  done
done
