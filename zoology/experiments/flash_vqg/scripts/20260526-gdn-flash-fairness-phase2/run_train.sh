#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQUIRE_EXPLICIT_PHASE2_ENV="${FLASH_VQG_REQUIRE_EXPLICIT_PHASE2_ENV:-false}"

if [[ "${REQUIRE_EXPLICIT_PHASE2_ENV}" =~ ^(1|true|yes)$ ]]; then
  missing_env=()
  for name in \
    SEED_VALUES \
    DATA_SEED \
    NUM_CODEBOOK_VECTORS \
    FOX_GD_RESIDUAL_RANK \
    RUN_ID_OVERRIDE \
    EXPERIMENT_MODE_OVERRIDE \
    LAUNCH_ID_PREFIX \
    PROJECT
  do
    if [[ -z "${!name+x}" || -z "${!name}" ]]; then
      missing_env+=("${name}")
    fi
  done
  if [[ "${#missing_env[@]}" -gt 0 ]]; then
    echo "FLASH_VQG_REQUIRE_EXPLICIT_PHASE2_ENV=true 时必须显式设置: ${missing_env[*]}" >&2
    exit 2
  fi
fi

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/common_env.sh"

if [[ "$#" -gt 0 ]]; then
  echo "Phase 2 run_train.sh 不接受额外参数, 当前收到: $*" >&2
  exit 2
fi

if [[ "${REQUIRE_EXPLICIT_PHASE2_ENV}" =~ ^(1|true|yes)$ ]]; then
  if [[ "${SEED_VALUES}" == "126" \
    && "${NUM_CODEBOOK_VECTORS}" == "256" \
    && "${FOX_GD_RESIDUAL_RANK}" == "8" \
    && ! "${FLASH_VQG_ALLOW_PHASE2_DEFAULT_TRIPLE:-false}" =~ ^(1|true|yes)$ ]]; then
    echo "explicit phase2 run 不允许隐式使用默认 seed126/cb256-r8; 如确需该默认组合, 设置 FLASH_VQG_ALLOW_PHASE2_DEFAULT_TRIPLE=true." >&2
    exit 2
  fi
  TAG_TEXT="${RUN_ID_OVERRIDE} ${EXPERIMENT_MODE_OVERRIDE} ${LAUNCH_ID_PREFIX}"
  if [[ "${TAG_TEXT}" == *wcap* && -z "${FOX_GD_RESIDUAL_WRITE_STRENGTH_CAP}" ]]; then
    echo "explicit phase2 run 名称包含 wcap, 但 FOX_GD_RESIDUAL_WRITE_STRENGTH_CAP 为空." >&2
    exit 2
  fi
fi

METRICS_WHITE_LIST_FILE="${METRICS_WHITE_LIST_FILE:-${SCRIPT_DIR}/../20260425-gd-residual-v1-mqar/metrics.yaml}"
BUILDER_SPEC="${SCRIPT_DIR}/../20260425-gd-residual-v1-mqar/config_builder.py:build_gd_residual_v1_train_configs"
OPTIONAL_ARGS=()
if [[ -n "${FOX_GD_RESIDUAL_WRITE_STRENGTH_CAP}" ]]; then
  OPTIONAL_ARGS+=(--fox-gd-residual-write-strength-cap "${FOX_GD_RESIDUAL_WRITE_STRENGTH_CAP}")
  OPTIONAL_ARGS+=(--fox-gd-residual-write-strength-cap-mode "${FOX_GD_RESIDUAL_WRITE_STRENGTH_CAP_MODE}")
  OPTIONAL_ARGS+=(
    --fox-gd-residual-write-strength-cap-until-train-steps
    "${FOX_GD_RESIDUAL_WRITE_STRENGTH_CAP_UNTIL_TRAIN_STEPS}"
  )
  if [[ -n "${FOX_GD_RESIDUAL_WRITE_STRENGTH_CAP_FINAL}" ]]; then
    OPTIONAL_ARGS+=(
      --fox-gd-residual-write-strength-cap-final
      "${FOX_GD_RESIDUAL_WRITE_STRENGTH_CAP_FINAL}"
    )
    OPTIONAL_ARGS+=(
      --fox-gd-residual-write-strength-cap-release-start-train-steps
      "${FOX_GD_RESIDUAL_WRITE_STRENGTH_CAP_RELEASE_START_TRAIN_STEPS}"
    )
    OPTIONAL_ARGS+=(
      --fox-gd-residual-write-strength-cap-release-end-train-steps
      "${FOX_GD_RESIDUAL_WRITE_STRENGTH_CAP_RELEASE_END_TRAIN_STEPS}"
    )
    OPTIONAL_ARGS+=(
      --fox-gd-residual-write-strength-cap-eval-policy
      "${FOX_GD_RESIDUAL_WRITE_STRENGTH_CAP_EVAL_POLICY}"
    )
  fi
fi
if [[ -n "${FOX_GD_RESIDUAL_WRITE_BUDGET}" ]]; then
  OPTIONAL_ARGS+=(--fox-gd-residual-write-budget "${FOX_GD_RESIDUAL_WRITE_BUDGET}")
  if [[ -n "${FOX_GD_RESIDUAL_WRITE_BUDGET_FINAL}" ]]; then
    OPTIONAL_ARGS+=(
      --fox-gd-residual-write-budget-final
      "${FOX_GD_RESIDUAL_WRITE_BUDGET_FINAL}"
    )
    OPTIONAL_ARGS+=(
      --fox-gd-residual-write-budget-release-start-train-steps
      "${FOX_GD_RESIDUAL_WRITE_BUDGET_RELEASE_START_TRAIN_STEPS}"
    )
    OPTIONAL_ARGS+=(
      --fox-gd-residual-write-budget-release-end-train-steps
      "${FOX_GD_RESIDUAL_WRITE_BUDGET_RELEASE_END_TRAIN_STEPS}"
    )
    OPTIONAL_ARGS+=(
      --fox-gd-residual-write-budget-eval-policy
      "${FOX_GD_RESIDUAL_WRITE_BUDGET_EVAL_POLICY}"
    )
    OPTIONAL_ARGS+=(
      --fox-gd-residual-write-budget-schedule
      "${FOX_GD_RESIDUAL_WRITE_BUDGET_SCHEDULE}"
    )
  fi
fi
if [[ -n "${FOX_GD_RESIDUAL_WRITE_TOTAL_CAP}" ]]; then
  OPTIONAL_ARGS+=(--fox-gd-residual-write-total-cap "${FOX_GD_RESIDUAL_WRITE_TOTAL_CAP}")
  if [[ -n "${FOX_GD_RESIDUAL_WRITE_TOTAL_CAP_FINAL}" ]]; then
    OPTIONAL_ARGS+=(
      --fox-gd-residual-write-total-cap-final
      "${FOX_GD_RESIDUAL_WRITE_TOTAL_CAP_FINAL}"
    )
    OPTIONAL_ARGS+=(
      --fox-gd-residual-write-total-cap-release-start-train-steps
      "${FOX_GD_RESIDUAL_WRITE_TOTAL_CAP_RELEASE_START_TRAIN_STEPS}"
    )
    OPTIONAL_ARGS+=(
      --fox-gd-residual-write-total-cap-release-end-train-steps
      "${FOX_GD_RESIDUAL_WRITE_TOTAL_CAP_RELEASE_END_TRAIN_STEPS}"
    )
    OPTIONAL_ARGS+=(
      --fox-gd-residual-write-total-cap-eval-policy
      "${FOX_GD_RESIDUAL_WRITE_TOTAL_CAP_EVAL_POLICY}"
    )
    OPTIONAL_ARGS+=(
      --fox-gd-residual-write-total-cap-schedule
      "${FOX_GD_RESIDUAL_WRITE_TOTAL_CAP_SCHEDULE}"
    )
  fi
fi
OPTIONAL_ARGS+=(--fox-gd-residual-write-q-alpha "${FOX_GD_RESIDUAL_WRITE_Q_ALPHA}")
if [[ -n "${FOX_GD_RESIDUAL_M_NORM_CAP}" ]]; then
  OPTIONAL_ARGS+=(--fox-gd-residual-m-norm-cap "${FOX_GD_RESIDUAL_M_NORM_CAP}")
fi
if [[ -n "${FOX_GD_RESIDUAL_UPDATE_NORM_CAP}" ]]; then
  OPTIONAL_ARGS+=(--fox-gd-residual-update-norm-cap "${FOX_GD_RESIDUAL_UPDATE_NORM_CAP}")
fi
if [[ -n "${FOX_GD_RESIDUAL_BETA_CAP}" ]]; then
  OPTIONAL_ARGS+=(--fox-gd-residual-beta-cap "${FOX_GD_RESIDUAL_BETA_CAP}")
  if [[ -n "${FOX_GD_RESIDUAL_BETA_CAP_FINAL}" ]]; then
    OPTIONAL_ARGS+=(--fox-gd-residual-beta-cap-final "${FOX_GD_RESIDUAL_BETA_CAP_FINAL}")
    OPTIONAL_ARGS+=(
      --fox-gd-residual-beta-cap-release-start-train-steps
      "${FOX_GD_RESIDUAL_BETA_CAP_RELEASE_START_TRAIN_STEPS}"
    )
    OPTIONAL_ARGS+=(
      --fox-gd-residual-beta-cap-release-end-train-steps
      "${FOX_GD_RESIDUAL_BETA_CAP_RELEASE_END_TRAIN_STEPS}"
    )
    OPTIONAL_ARGS+=(
      --fox-gd-residual-beta-cap-eval-policy
      "${FOX_GD_RESIDUAL_BETA_CAP_EVAL_POLICY}"
    )
  fi
fi
if [[ -n "${FOX_GD_RESIDUAL_BETA_LOW}" ]]; then
  OPTIONAL_ARGS+=(--fox-gd-residual-beta-low "${FOX_GD_RESIDUAL_BETA_LOW}")
fi
if [[ -n "${FOX_GD_RESIDUAL_BETA_HIGH}" ]]; then
  OPTIONAL_ARGS+=(--fox-gd-residual-beta-high "${FOX_GD_RESIDUAL_BETA_HIGH}")
fi
if [[ -n "${FOX_GD_RESIDUAL_BETA_LOW_FINAL}" ]]; then
  OPTIONAL_ARGS+=(--fox-gd-residual-beta-low-final "${FOX_GD_RESIDUAL_BETA_LOW_FINAL}")
fi
if [[ -n "${FOX_GD_RESIDUAL_BETA_HIGH_FINAL}" ]]; then
  OPTIONAL_ARGS+=(--fox-gd-residual-beta-high-final "${FOX_GD_RESIDUAL_BETA_HIGH_FINAL}")
fi
if [[ -n "${FOX_GD_RESIDUAL_ADDR_INIT_SEED}" ]]; then
  OPTIONAL_ARGS+=(--fox-gd-residual-addr-init-seed "${FOX_GD_RESIDUAL_ADDR_INIT_SEED}")
fi
if [[ -n "${CODEBOOK_INIT_SEED}" ]]; then
  OPTIONAL_ARGS+=(--codebook-init-seed "${CODEBOOK_INIT_SEED}")
fi
cd "${ROOT_DIR}"

"${PYTHON_BIN}" -m zoology.experiments.flash_vqg.run_flash_vqg_suite \
  --flash-only \
  --logger-backend swanlab \
  --analysis "${ANALYSIS_SOURCE}" \
  --backend "${BACKEND}" \
  --dmodels "${DMODEL}" \
  --learning-rates "${LR}" \
  --max-epochs "${MAX_EPOCHS}" \
  --train-batch-order "${TRAIN_BATCH_ORDER}" \
  --seed-values "${SEED_VALUES}" \
  --data-seed "${DATA_SEED}" \
  --num-codebook-vectors "${NUM_CODEBOOK_VECTORS}" \
  --fox-remote-path-backend "${REMOTE_PATH_BACKEND}" \
  --fox-remote-read-topk-values "${FOX_REMOTE_READ_TOPK}" \
  --fox-remote-formula "${FOX_REMOTE_FORMULA}" \
  --fox-gd-residual-rank "${FOX_GD_RESIDUAL_RANK}" \
  --fox-gd-residual-write-topk "${FOX_GD_RESIDUAL_WRITE_TOPK}" \
  --fox-gd-residual-builder "${FOX_GD_RESIDUAL_BUILDER}" \
  --fox-gd-residual-pack-mode "${FOX_GD_RESIDUAL_PACK_MODE}" \
  --fox-gd-residual-chunk-size "${FOX_GD_RESIDUAL_CHUNK_SIZE}" \
  --fox-gd-residual-mu-min-count "${FOX_GD_RESIDUAL_MU_MIN_COUNT}" \
  --fox-gd-residual-addr-eps "${FOX_GD_RESIDUAL_ADDR_EPS}" \
  --fox-gd-residual-den-eps "${FOX_GD_RESIDUAL_DEN_EPS}" \
  --fox-gd-residual-rho-eps "${FOX_GD_RESIDUAL_RHO_EPS}" \
  --fox-gd-residual-addr-init-rng-mode "${FOX_GD_RESIDUAL_ADDR_INIT_RNG_MODE}" \
  --fox-gd-residual-beta-init "${FOX_GD_RESIDUAL_BETA_INIT}" \
  --fox-gd-residual-beta-control-mode "${FOX_GD_RESIDUAL_BETA_CONTROL_MODE}" \
  --fox-gd-residual-beta-sigmoid-temp "${FOX_GD_RESIDUAL_BETA_SIGMOID_TEMP}" \
  --fox-gd-residual-beta-band-release-start-train-steps "${FOX_GD_RESIDUAL_BETA_BAND_RELEASE_START_TRAIN_STEPS}" \
  --fox-gd-residual-beta-band-release-end-train-steps "${FOX_GD_RESIDUAL_BETA_BAND_RELEASE_END_TRAIN_STEPS}" \
  --fox-gd-residual-beta-band-eval-policy "${FOX_GD_RESIDUAL_BETA_BAND_EVAL_POLICY}" \
  --fox-gd-residual-beta-band-schedule "${FOX_GD_RESIDUAL_BETA_BAND_SCHEDULE}" \
  --fox-gd-residual-lambda-init "${FOX_GD_RESIDUAL_LAMBDA_INIT}" \
  --fox-gd-residual-lambda-floor "${FOX_GD_RESIDUAL_LAMBDA_FLOOR}" \
  --fox-gd-residual-write-strength-mode "${FOX_GD_RESIDUAL_WRITE_STRENGTH_MODE}" \
  "${OPTIONAL_ARGS[@]}" \
  --fox-gd-residual-norm-with-gain "${FOX_GD_RESIDUAL_NORM_WITH_GAIN}" \
  --fox-gd-residual-use-separate-addr-codebook "${FOX_GD_RESIDUAL_USE_SEPARATE_ADDR_CODEBOOK}" \
  --vq-score-mode "${VQ_SCORE_MODE}" \
  --vq-weight-mode "${VQ_WEIGHT_MODE}" \
  --vq-update-mode "${VQ_UPDATE_MODE}" \
  --vq-softmax-tau "${VQ_SOFTMAX_TAU}" \
  --codebook-init-rng-mode "${CODEBOOK_INIT_RNG_MODE}" \
  --vq-topk "${VQ_TOPK}" \
  --train-batch-size "${TRAIN_BATCH_SIZE}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --validations-per-epoch "${VALIDATIONS_PER_EPOCH}" \
  --disable-early-stopping "${DISABLE_EARLY_STOPPING}" \
  --cache-dir "${CACHE_DIR}" \
  --metrics-white-list-file "${METRICS_WHITE_LIST_FILE}" \
  --project "${PROJECT}" \
  --entity "${ENTITY}" \
  --launch-id-prefix "${LAUNCH_ID_PREFIX}" \
  --run-id "${RUN_ID_OVERRIDE}" \
  --experiment-mode "${EXPERIMENT_MODE_OVERRIDE}" \
  --config-builder "${BUILDER_SPEC}" \
  --gpus "${GPU_ID}"
