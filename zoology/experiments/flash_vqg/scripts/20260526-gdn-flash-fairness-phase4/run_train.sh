#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/common_env.sh"

if [[ "$#" -gt 0 ]]; then
  echo "Phase 4 run_train.sh 不接受额外参数, 当前收到: $*" >&2
  exit 2
fi

METRICS_WHITE_LIST_FILE="${METRICS_WHITE_LIST_FILE:-${SCRIPT_DIR}/../20260526-gdn-flash-fairness-phase3/metrics.yaml}"
BUILDER_SPEC="${SCRIPT_DIR}/../20260526-gdn-flash-fairness-phase3/config_builder.py:build_gdn_flash_fairness_phase3_configs"

cd "${ROOT_DIR}"

CMD=(
  "${PYTHON_BIN}" -m zoology.experiments.flash_vqg.run_flash_vqg_suite
  --flash-only
  --logger-backend swanlab
  --analysis "${ANALYSIS_SOURCE}"
  --backend "${BACKEND}"
  --dmodels "${DMODEL}"
  --learning-rates "${LR}"
  --max-epochs "${MAX_EPOCHS}"
  --train-batch-order "${TRAIN_BATCH_ORDER}"
  --seed-values "${SEED_VALUES}"
  --data-seed "${DATA_SEED}"
  --train-batch-size "${TRAIN_BATCH_SIZE}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"
  --validations-per-epoch "${VALIDATIONS_PER_EPOCH}"
  --disable-early-stopping "${DISABLE_EARLY_STOPPING}"
  --cache-dir "${CACHE_DIR}"
  --metrics-white-list-file "${METRICS_WHITE_LIST_FILE}"
  --project "${PROJECT}"
  --entity "${ENTITY}"
  --launch-id-prefix "${LAUNCH_ID_PREFIX}"
  --run-id "${RUN_ID_OVERRIDE}"
  --experiment-mode "${EXPERIMENT_MODE_OVERRIDE}"
  --config-builder "${BUILDER_SPEC}"
  --gpus "${GPU_ID}"
)

if [[ "${PARALLELIZE}" == "true" ]]; then
  CMD+=(-p)
fi

"${CMD[@]}"
