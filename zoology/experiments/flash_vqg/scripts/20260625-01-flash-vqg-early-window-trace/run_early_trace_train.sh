#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/lyj/mnt/project/zoology"
FLASH_VQG_ROOT="/home/lyj/mnt/project/Flash-VQG"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/lyj/miniconda3/envs/flash-vqg/bin/python}"
export PYTHONPATH="${FLASH_VQG_ROOT}/src:${ROOT_DIR}:${PYTHONPATH:-}"
export SWANLAB_MODE="${SWANLAB_MODE:-cloud}"

if [[ "$#" -lt 1 ]]; then
  cat >&2 <<'USAGE'
Usage:
  run_early_trace_train.sh <target> [extra_args...]

Targets:
  smoke-default-s123
  smoke-hard04-s123
  default-s123
  hard04-s123
  default-s124

Examples:
  GPU_ID=0 bash run_early_trace_train.sh smoke-default-s123
  GPU_ID=0 bash run_early_trace_train.sh default-s123
USAGE
  exit 2
fi

TARGET="$1"
shift

GPU_ID="${GPU_ID:-0}"
ANALYSIS_SOURCE="${ANALYSIS_SOURCE:-off}"
PROJECT="${PROJECT:-flash_vqg_early_window_trace}"
ENTITY="${ENTITY:-scu-mclab}"
MACHINE_NAME="${MACHINE_NAME:-unknown}"

SETTING="${TARGET%-s*}"
SEED_VALUES="${TARGET##*-s}"
SMOKE_MODE="false"
if [[ "${SETTING}" == smoke-* ]]; then
  SMOKE_MODE="true"
  SETTING="${SETTING#smoke-}"
fi

if [[ "${SEED_VALUES}" != "123" && "${SEED_VALUES}" != "124" ]]; then
  echo "Unsupported target seed in ${TARGET}. Expected s123 or s124." >&2
  exit 2
fi

WRITE_ARGS=()
case "${SETTING}" in
  default)
    ;;
  hard04)
    WRITE_ARGS+=(
      --fox-gd-residual-write-strength-cap 0.04
      --fox-gd-residual-write-strength-cap-mode hard
    )
    ;;
  *)
    echo "Unsupported target setting in ${TARGET}." >&2
    exit 2
    ;;
esac

METRICS_YAML="${METRICS_YAML:-${SCRIPT_DIR}/metrics.yaml}"
BUILDER_SPEC="${BUILDER_SPEC:-${SCRIPT_DIR}/../20260425-gd-residual-v1-mqar/config_builder.py:build_gd_residual_v1_train_configs}"
TRACE_OUTPUT_DIR="${TRACE_OUTPUT_DIR:-${SCRIPT_DIR}/outputs/traces/${MACHINE_NAME}/${TARGET}}"
LAUNCH_ID_PREFIX="${LAUNCH_ID_PREFIX:-fvqg-20260625-01-etrace-${MACHINE_NAME}-${TARGET}}"
RUN_ID_OVERRIDE="${RUN_ID_OVERRIDE:-etrace-cb64r16-${TARGET}-d123-b64ga4-${MACHINE_NAME}}"
EXPERIMENT_MODE_OVERRIDE="${EXPERIMENT_MODE_OVERRIDE:-etrace_cb64r16_${SETTING}_s${SEED_VALUES}_d123_b64ga4_${MACHINE_NAME}}"

if [[ "${SMOKE_MODE}" == "true" ]]; then
  MAX_EPOCHS="${MAX_EPOCHS:-1}"
  MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-2}"
  MAX_VALIDATION_BATCHES="${MAX_VALIDATION_BATCHES:-1}"
  VALIDATIONS_PER_EPOCH="${VALIDATIONS_PER_EPOCH:-1}"
  READ_TRACE_VALID_BATCHES="${READ_TRACE_VALID_BATCHES:-441}"
  READ_TRACE_MAX_SAMPLES="${READ_TRACE_MAX_SAMPLES:-1}"
  READ_TRACE_MAX_QUERIES_PER_SAMPLE="${READ_TRACE_MAX_QUERIES_PER_SAMPLE:-1}"
  READ_TRACE_TRAIN_STEPS="${READ_TRACE_TRAIN_STEPS:-0,1,2}"
  TRACE_OUTPUT_DIR="${TRACE_OUTPUT_DIR:-${SCRIPT_DIR}/outputs/traces/${MACHINE_NAME}/${TARGET}}"
else
  MAX_EPOCHS="${MAX_EPOCHS:-4}"
  MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-}"
  MAX_VALIDATION_BATCHES="${MAX_VALIDATION_BATCHES:-}"
  VALIDATIONS_PER_EPOCH="${VALIDATIONS_PER_EPOCH:-4}"
  READ_TRACE_VALID_BATCHES="${READ_TRACE_VALID_BATCHES:-441}"
  READ_TRACE_MAX_SAMPLES="${READ_TRACE_MAX_SAMPLES:-4}"
  READ_TRACE_MAX_QUERIES_PER_SAMPLE="${READ_TRACE_MAX_QUERIES_PER_SAMPLE:-8}"
  READ_TRACE_TRAIN_STEPS="${READ_TRACE_TRAIN_STEPS:-0,64,130,203,352,448,705}"
fi

mkdir -p "${TRACE_OUTPUT_DIR}"
cd "${ROOT_DIR}"

LIMIT_ARGS=()
if [[ -n "${MAX_TRAIN_STEPS}" ]]; then
  LIMIT_ARGS+=(--max-train-steps "${MAX_TRAIN_STEPS}")
fi
if [[ -n "${MAX_VALIDATION_BATCHES}" ]]; then
  LIMIT_ARGS+=(--max-validation-batches "${MAX_VALIDATION_BATCHES}")
fi

exec "${PYTHON_BIN}" -m zoology.experiments.flash_vqg.run_flash_vqg_suite \
  --flash-only \
  --logger-backend swanlab \
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
  "${WRITE_ARGS[@]}" \
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
