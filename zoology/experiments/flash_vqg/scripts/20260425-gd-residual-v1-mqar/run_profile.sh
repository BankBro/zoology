#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/common_env.sh"

PROFILE_SEQ_LEN="${PROFILE_SEQ_LEN:-128}"
PROFILE_MICROBATCHES="${PROFILE_MICROBATCHES:-3}"
PROFILE_ENABLE_TORCH_PROFILER="${PROFILE_ENABLE_TORCH_PROFILER:-0}"
PROFILE_ENABLE_GD_DIAGNOSTICS="${PROFILE_ENABLE_GD_DIAGNOSTICS:-0}"
PROFILE_OUTPUT_DIR="${PROFILE_OUTPUT_DIR:-${ROOT_DIR}/tmp/20260425-gd-residual-v1-profile}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PROFILE_SEQ_LEN
export PROFILE_MICROBATCHES
export PROFILE_ENABLE_TORCH_PROFILER
export PROFILE_ENABLE_GD_DIAGNOSTICS
export PROFILE_OUTPUT_DIR

cd "${ROOT_DIR}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/profile_gd_residual_v1.py" \
  --remote-read-topk "${FOX_REMOTE_READ_TOPK}" \
  --rank "${FOX_GD_RESIDUAL_RANK}" \
  --write-topk "${FOX_GD_RESIDUAL_WRITE_TOPK}" \
  --batch-size "${TRAIN_BATCH_SIZE}" \
  --seq-len "${PROFILE_SEQ_LEN}" \
  --microbatches "${PROFILE_MICROBATCHES}" \
  --output-dir "${PROFILE_OUTPUT_DIR}" \
  --d-model "${DMODEL}" \
  --num-codebook-vectors "${NUM_CODEBOOK_VECTORS}" \
  --vq-softmax-tau "${VQ_SOFTMAX_TAU}" \
  --block-len 32 \
  --builder "${FOX_GD_RESIDUAL_BUILDER}" \
  --pack-mode "${FOX_GD_RESIDUAL_PACK_MODE}" \
  --chunk-size "${FOX_GD_RESIDUAL_CHUNK_SIZE}" \
  --learning-rate "${LR}"
