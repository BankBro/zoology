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
  run_stage2_probe_train.sh <target> [extra_args...]

Targets:
  default-s123
  hard04-s123
  cap0405-s123
  caprel0406late-s123
  default-s124
  hard04-s124
  cap0405-s124
  caprel0406late-s124

Example:
  GPU_ID=0 bash run_stage2_probe_train.sh hard04-s123
USAGE
  exit 2
fi

TARGET="$1"
shift

GPU_ID="${GPU_ID:-0}"
ANALYSIS_SOURCE="${ANALYSIS_SOURCE:-local}"
PROJECT="${PROJECT:-flash_vqg_pressure_telemetry_guard}"
ENTITY="${ENTITY:-scu-mclab}"

SETTING="${TARGET%-s*}"
SEED_VALUES="${TARGET##*-s}"
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
  cap0405)
    WRITE_ARGS+=(
      --fox-gd-residual-write-strength-cap 0.04
      --fox-gd-residual-write-strength-cap-mode hard
      --fox-gd-residual-write-strength-cap-final 0.05
      --fox-gd-residual-write-strength-cap-release-start-train-steps 2820
      --fox-gd-residual-write-strength-cap-release-end-train-steps 8468
      --fox-gd-residual-write-strength-cap-eval-policy scheduled
    )
    ;;
  caprel0406late)
    WRITE_ARGS+=(
      --fox-gd-residual-write-strength-cap 0.04
      --fox-gd-residual-write-strength-cap-mode hard
      --fox-gd-residual-write-strength-cap-final 0.06
      --fox-gd-residual-write-strength-cap-release-start-train-steps 2820
      --fox-gd-residual-write-strength-cap-release-end-train-steps 8468
      --fox-gd-residual-write-strength-cap-eval-policy scheduled
    )
    ;;
  *)
    echo "Unsupported target setting in ${TARGET}." >&2
    exit 2
    ;;
esac

METRICS_YAML="${METRICS_YAML:-${SCRIPT_DIR}/metrics.yaml}"
BUILDER_SPEC="${BUILDER_SPEC:-${SCRIPT_DIR}/../20260425-gd-residual-v1-mqar/config_builder.py:build_gd_residual_v1_train_configs}"
LAUNCH_ID_PREFIX="${LAUNCH_ID_PREFIX:-fvqg-20260624-02-ptel-${TARGET}}"
RUN_ID_OVERRIDE="ptel-cb64r16-${TARGET}-d123-b64ga4"
EXPERIMENT_MODE_OVERRIDE="ptel_cb64r16_${SETTING}_s${SEED_VALUES}_d123_b64ga4"

cd "${ROOT_DIR}"

exec "${PYTHON_BIN}" -m zoology.experiments.flash_vqg.run_flash_vqg_suite \
  --flash-only \
  --logger-backend swanlab \
  --analysis "${ANALYSIS_SOURCE}" \
  --backend torch \
  --block-len 32 \
  --local-num-blocks 2 \
  --dmodels 128 \
  --learning-rates 1e-3 \
  --max-epochs "${MAX_EPOCHS:-4}" \
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
  --validations-per-epoch "${VALIDATIONS_PER_EPOCH:-4}" \
  --disable-early-stopping true \
  --read-churn-probe-enabled true \
  --read-churn-probe-valid-batches "${READ_CHURN_PROBE_VALID_BATCHES:-441}" \
  --read-churn-probe-max-samples "${READ_CHURN_PROBE_MAX_SAMPLES:-16}" \
  --read-churn-probe-query-only "${READ_CHURN_PROBE_QUERY_ONLY:-true}" \
  --read-trace-enabled false \
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
