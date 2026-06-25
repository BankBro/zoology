#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ZOOLOGY_REPO_ROOT:-/home/lyj/mnt/project/zoology}"
FLASH_VQG_ROOT="${FLASH_VQG_ROOT:-/home/lyj/mnt/project/Flash-VQG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/lyj/miniconda3/envs/flash-vqg/bin/python}"
export PYTHONPATH="${FLASH_VQG_ROOT}/src:${ROOT_DIR}:${PYTHONPATH:-}"
export SWANLAB_MODE="${SWANLAB_MODE:-cloud}"

if [[ "$#" -lt 1 ]]; then
  cat >&2 <<'USAGE'
Usage:
  run_repro_screen_train.sh <target> [extra_args...]

Targets:
  smoke-default-s123
  default-s123-r1
  default-s124-r1
  default-s123-r2
  default-s124-r2

Examples:
  GPU_ID=0 bash run_repro_screen_train.sh smoke-default-s123
  GPU_ID=0 bash run_repro_screen_train.sh default-s123-r1
USAGE
  exit 2
fi

TARGET="$1"
shift

GPU_ID="${GPU_ID:-0}"
ANALYSIS_SOURCE="${ANALYSIS_SOURCE:-off}"
LOGGER_BACKEND="${LOGGER_BACKEND:-none}"
PROJECT="${PROJECT:-flash_vqg_1epoch_repro_screen}"
ENTITY="${ENTITY:-scu-mclab}"
MACHINE_NAME="${MACHINE_NAME:-unknown}"

check_container_gpu_ready() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is not available inside the current container; pause experiment launch." >&2
    exit 1
  fi
  if ! nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi/NVML failed inside the current container; pause experiment launch." >&2
    exit 1
  fi
  if ! "${PYTHON_BIN}" - "${GPU_ID}" <<'PY'
import sys
import torch

gpu_id = int(sys.argv[1])
device_count = torch.cuda.device_count()
if not torch.cuda.is_available() or device_count < 1:
    print(
        f"torch cuda unavailable: cuda_available={torch.cuda.is_available()} "
        f"device_count={device_count}",
        file=sys.stderr,
    )
    raise SystemExit(1)
if gpu_id < 0 or gpu_id >= device_count:
    print(f"GPU_ID={gpu_id} outside available device range 0..{device_count - 1}", file=sys.stderr)
    raise SystemExit(1)
print(f"container_gpu_ready=true device_count={device_count} gpu_id={gpu_id}")
PY
  then
    echo "torch.cuda readiness check failed inside the current container; pause experiment launch." >&2
    exit 1
  fi
}

SMOKE_MODE="false"
SETTING=""
SEED_VALUES=""
REPEAT_TAG=""

if [[ "${TARGET}" =~ ^smoke-default-s(123|124)$ ]]; then
  SMOKE_MODE="true"
  SETTING="default"
  SEED_VALUES="${BASH_REMATCH[1]}"
  REPEAT_TAG="smoke"
elif [[ "${TARGET}" =~ ^default-s(123|124)-r([12])$ ]]; then
  SETTING="default"
  SEED_VALUES="${BASH_REMATCH[1]}"
  REPEAT_TAG="r${BASH_REMATCH[2]}"
else
  echo "Unsupported target: ${TARGET}" >&2
  exit 2
fi

METRICS_YAML="${METRICS_YAML:-${SCRIPT_DIR}/metrics.yaml}"
BUILDER_SPEC="${BUILDER_SPEC:-${SCRIPT_DIR}/../20260425-gd-residual-v1-mqar/config_builder.py:build_gd_residual_v1_train_configs}"
TRACE_OUTPUT_DIR="${TRACE_OUTPUT_DIR:-${SCRIPT_DIR}/outputs/traces/${MACHINE_NAME}/${TARGET}}"
LAUNCH_ID_PREFIX="${LAUNCH_ID_PREFIX:-fvqg-20260625-02-rscreen-${MACHINE_NAME}-${TARGET}}"
RUN_ID_OVERRIDE="${RUN_ID_OVERRIDE:-rscreen-cb64r16-${TARGET}-d123-b64ga4-${MACHINE_NAME}}"
EXPERIMENT_MODE_OVERRIDE="${EXPERIMENT_MODE_OVERRIDE:-rscreen_cb64r16_${SETTING}_s${SEED_VALUES}_${REPEAT_TAG}_d123_b64ga4_${MACHINE_NAME}}"

if [[ "${SMOKE_MODE}" == "true" ]]; then
  MAX_EPOCHS="${MAX_EPOCHS:-1}"
  MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-2}"
  MAX_VALIDATION_BATCHES="${MAX_VALIDATION_BATCHES:-1}"
  VALIDATIONS_PER_EPOCH="${VALIDATIONS_PER_EPOCH:-1}"
  READ_TRACE_VALID_BATCHES="${READ_TRACE_VALID_BATCHES:-441}"
  READ_TRACE_MAX_SAMPLES="${READ_TRACE_MAX_SAMPLES:-1}"
  READ_TRACE_MAX_QUERIES_PER_SAMPLE="${READ_TRACE_MAX_QUERIES_PER_SAMPLE:-1}"
  READ_TRACE_TRAIN_STEPS="${READ_TRACE_TRAIN_STEPS:-0,1,2}"
else
  MAX_EPOCHS="${MAX_EPOCHS:-1}"
  MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-}"
  MAX_VALIDATION_BATCHES="${MAX_VALIDATION_BATCHES:-}"
  VALIDATIONS_PER_EPOCH="${VALIDATIONS_PER_EPOCH:-4}"
  READ_TRACE_VALID_BATCHES="${READ_TRACE_VALID_BATCHES:-441}"
  READ_TRACE_MAX_SAMPLES="${READ_TRACE_MAX_SAMPLES:-4}"
  READ_TRACE_MAX_QUERIES_PER_SAMPLE="${READ_TRACE_MAX_QUERIES_PER_SAMPLE:-8}"
  READ_TRACE_TRAIN_STEPS="${READ_TRACE_TRAIN_STEPS:-0,64,130,203,352,448,704}"
fi

mkdir -p "${TRACE_OUTPUT_DIR}"
cd "${ROOT_DIR}"

check_container_gpu_ready

LIMIT_ARGS=()
if [[ -n "${MAX_TRAIN_STEPS}" ]]; then
  LIMIT_ARGS+=(--max-train-steps "${MAX_TRAIN_STEPS}")
fi
if [[ -n "${MAX_VALIDATION_BATCHES}" ]]; then
  LIMIT_ARGS+=(--max-validation-batches "${MAX_VALIDATION_BATCHES}")
fi

exec "${PYTHON_BIN}" -m zoology.experiments.flash_vqg.run_flash_vqg_suite \
  --flash-only \
  --logger-backend "${LOGGER_BACKEND}" \
  --analysis "${ANALYSIS_SOURCE}" \
  --backend torch \
  --block-len 32 \
  --local-num-blocks 2 \
  --dmodels 128 \
  --learning-rates 1e-3 \
  --max-epochs "${MAX_EPOCHS}" \
  "${LIMIT_ARGS[@]}" \
  --train-batch-order global_shuffle \
  --if-remote-enabled true \
  --seed-values "${SEED_VALUES}" \
  --data-seed 123 \
  --num-codebook-vectors 64 \
  --fox-remote-path-backend torch \
  --fox-remote-read-topk-values 2 \
  --fox-remote-formula gd_residual_v1 \
  --fox-gd-residual-rank 16 \
  --fox-gd-residual-write-topk 4 \
  --fox-gd-residual-builder grouped_chunk_torch_ref \
  --fox-gd-residual-pack-mode semivec_ref \
  --fox-gd-residual-chunk-size 64 \
  --fox-gd-residual-mu-min-count 0.1 \
  --fox-gd-residual-beta-init 0.5 \
  --fox-gd-residual-lambda-init 0.05 \
  --vq-score-mode codebook_dot \
  --vq-weight-mode dense_softmax \
  --vq-update-mode grad \
  --vq-softmax-tau 0.25 \
  --vq-topk 4 \
  --train-batch-size "${TRAIN_BATCH_SIZE:-64}" \
  --eval-batch-size "${EVAL_BATCH_SIZE:-16}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-4}" \
  --validations-per-epoch "${VALIDATIONS_PER_EPOCH}" \
  --disable-early-stopping true \
  --read-churn-probe-enabled true \
  --read-churn-probe-valid-batches "${READ_CHURN_PROBE_VALID_BATCHES:-441}" \
  --read-churn-probe-max-samples "${READ_CHURN_PROBE_MAX_SAMPLES:-16}" \
  --read-churn-probe-query-only "${READ_CHURN_PROBE_QUERY_ONLY:-true}" \
  --read-trace-enabled true \
  --read-trace-valid-batches "${READ_TRACE_VALID_BATCHES}" \
  --read-trace-max-samples "${READ_TRACE_MAX_SAMPLES}" \
  --read-trace-query-only "${READ_TRACE_QUERY_ONLY:-true}" \
  --read-trace-max-queries-per-sample "${READ_TRACE_MAX_QUERIES_PER_SAMPLE}" \
  --read-trace-output-dir "${TRACE_OUTPUT_DIR}" \
  --read-trace-train-steps "${READ_TRACE_TRAIN_STEPS}" \
  --cache-dir "${CACHE_DIR:-./data/flash_vqg}" \
  --metrics-white-list-file "${METRICS_YAML}" \
  --project "${PROJECT}" \
  --entity "${ENTITY}" \
  --launch-id-prefix "${LAUNCH_ID_PREFIX}" \
  --run-id "${RUN_ID_OVERRIDE}" \
  --experiment-mode "${EXPERIMENT_MODE_OVERRIDE}" \
  --config-builder "${BUILDER_SPEC}" \
  --gpus "${GPU_ID}" \
  "$@"
