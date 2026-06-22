#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/lyj/mnt/project/zoology"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/lyj/miniconda3/envs/flash-vqg/bin/python}"
PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export PYTHONPATH
export SWANLAB_MODE="${SWANLAB_MODE:-cloud}"

if [ $# -lt 1 ]; then
  cat >&2 <<'USAGE'
Usage: run_fixed_readk4_train.sh <target> [extra_args...]

Targets:
  cb256r8-readk4-s123
  cb256r4-readk4-s123
  cb128r8-readk4-s125-repeat

Example:
  GPU_ID=0 bash run_fixed_readk4_train.sh cb256r8-readk4-s123
USAGE
  exit 2
fi

TARGET="$1"
shift

GPU_ID="${GPU_ID:-0}"
ANALYSIS_SOURCE="${ANALYSIS_SOURCE:-off}"

DMODEL=128
LR=1e-3
TRAIN_BATCH_ORDER=global_shuffle
DATA_SEED=123
TRAIN_BATCH_SIZE=64
EVAL_BATCH_SIZE=16
GRADIENT_ACCUMULATION_STEPS=4
VALIDATIONS_PER_EPOCH=2
MAX_EPOCHS="${MAX_EPOCHS:-4}"

NUM_CODEBOOK_VECTORS=256
FOX_GD_RESIDUAL_RANK=8
SEED_VALUES=123
FOX_REMOTE_READ_TOPK=4

PROJECT="${PROJECT:-flash_vqg_readk_telemetry_formalization}"
ENTITY="${ENTITY:-scu-mclab}"
METRICS_YAML="${SCRIPT_DIR}/metrics.yaml"

case "${TARGET}" in
  cb256r8-readk4-s123)
    NUM_CODEBOOK_VECTORS=256
    FOX_GD_RESIDUAL_RANK=8
    SEED_VALUES=123
    ;;
  cb256r4-readk4-s123)
    NUM_CODEBOOK_VECTORS=256
    FOX_GD_RESIDUAL_RANK=4
    SEED_VALUES=123
    ;;
  cb128r8-readk4-s125-repeat)
    NUM_CODEBOOK_VECTORS=128
    FOX_GD_RESIDUAL_RANK=8
    SEED_VALUES=125
    ;;
  *)
    echo "Unknown target: ${TARGET}" >&2
    exit 2
    ;;
esac

LAUNCH_ID_PREFIX="flash-vqg-20260622-02-readk-tel-${TARGET}"

cd "${ROOT_DIR}"

exec "${PYTHON_BIN}" -m zoology.experiments.flash_vqg.run_flash_vqg_suite \
  --flash-only \
  --logger-backend swanlab \
  --analysis "${ANALYSIS_SOURCE}" \
  --backend torch \
  --block-len 32 \
  --local-num-blocks 2 \
  --dmodels "${DMODEL}" \
  --learning-rates "${LR}" \
  --max-epochs "${MAX_EPOCHS}" \
  --train-batch-order "${TRAIN_BATCH_ORDER}" \
  --if-remote-enabled true \
  --seed-values "${SEED_VALUES}" \
  --data-seed "${DATA_SEED}" \
  --num-codebook-vectors "${NUM_CODEBOOK_VECTORS}" \
  --fox-remote-path-backend torch \
  --fox-remote-read-topk-values "${FOX_REMOTE_READ_TOPK}" \
  --fox-remote-formula gd_residual_v1 \
  --fox-gd-residual-rank "${FOX_GD_RESIDUAL_RANK}" \
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
  --train-batch-size "${TRAIN_BATCH_SIZE}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --validations-per-epoch "${VALIDATIONS_PER_EPOCH}" \
  --disable-early-stopping true \
  --cache-dir "${CACHE_DIR:-./data/flash_vqg}" \
  --metrics-white-list-file "${METRICS_YAML}" \
  --project "${PROJECT}" \
  --entity "${ENTITY}" \
  --launch-id-prefix "${LAUNCH_ID_PREFIX}" \
  --gpus "${GPU_ID}" \
  "$@"
