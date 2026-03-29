#!/usr/bin/env bash
set -euo pipefail

# ===========================================================================
# CLR Delta V1 vs CLR V1 comparison experiment
# Sweeps over (formula, rank) pairs for fair comparison.
# ===========================================================================

ROOT_DIR="/home/lyj/mnt/project/zoology"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-python}"
BACKEND="${BACKEND:-torch}"
LOGGER_BACKEND="${LOGGER_BACKEND:-swanlab}"
ANALYSIS_SOURCE="${ANALYSIS_SOURCE:-remote}"
DMODEL="${DMODEL:-128}"
LR="${LR:-1e-3}"
MAX_EPOCHS="${MAX_EPOCHS:-32}"
PROJECT="${PROJECT:-flash_vqg_mqar}"
ENTITY="${ENTITY:-scu-mclab}"
CACHE_DIR="${CACHE_DIR:-./data/flash_vqg}"
BASE_LAUNCH_ID_PREFIX="${BASE_LAUNCH_ID_PREFIX:-flash-vqg-clr-delta-v1-vs-clr-v1}"
BLOCK_LEN="${BLOCK_LEN:-32}"
LOCAL_NUM_BLOCKS="${LOCAL_NUM_BLOCKS:-2}"
NUM_CODEBOOK_VECTORS="${NUM_CODEBOOK_VECTORS:-128}"
TRAIN_BATCH_ORDER="${TRAIN_BATCH_ORDER:-global_shuffle}"
DATA_SEED="${DATA_SEED:-123}"
GPUS="${GPUS:-0}"
METRICS_WHITE_LIST_FILE="${METRICS_WHITE_LIST_FILE:-${ROOT_DIR}/zoology/experiments/flash_vqg/metrics_white_lists/e0.yaml}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/tmp/flash_vqg_clr_delta_v1_vs_clr_v1}"
RANKS_CSV="${RANKS_CSV:-2,4,8}"
FORMULAS_CSV="${FORMULAS_CSV:-clr_v1,clr_delta_v1}"

timestamp_utc="$(date -u +%Y-%m-%d-%H-%M-%S)"
LAUNCH_ID="${LAUNCH_ID:-${BASE_LAUNCH_ID_PREFIX}-${timestamp_utc}}"
GENERATED_DIR="${ROOT_DIR}/zoology/experiments/flash_vqg/generated/${LAUNCH_ID}"
MANIFEST_PATH="${GENERATED_DIR}/manifest.json"

mkdir -p "${GENERATED_DIR}" "${LOG_DIR}"

IFS=',' read -r -a GPU_IDS <<< "${GPUS}"
IFS=',' read -r -a RANKS <<< "${RANKS_CSV}"
IFS=',' read -r -a FORMULAS <<< "${FORMULAS_CSV}"

echo "==> launch_id: ${LAUNCH_ID}"
echo "==> formulas: ${FORMULAS_CSV}"
echo "==> ranks: ${RANKS_CSV}"
echo "==> gpus: ${GPUS}"
echo "==> generated dir: ${GENERATED_DIR}"
echo "==> log dir: ${LOG_DIR}"

# Build comma-separated rank list for Python
RANKS_PY=""
for rank in "${RANKS[@]}"; do
  rank_trimmed="$(echo "${rank}" | xargs)"
  [[ -n "${RANKS_PY}" ]] && RANKS_PY+=", "
  RANKS_PY+="${rank_trimmed}"
done

# Build comma-separated formula list for Python
FORMULAS_PY=""
for formula in "${FORMULAS[@]}"; do
  formula_trimmed="$(echo "${formula}" | xargs)"
  [[ -n "${FORMULAS_PY}" ]] && FORMULAS_PY+=", "
  FORMULAS_PY+="\"${formula_trimmed}\""
done

cat >"${GENERATED_DIR}/prepare_sweep.py" <<PY
# -*- coding: utf-8 -*-
from pathlib import Path

from zoology.experiments.flash_vqg.flash_vqg_suite import build_configs
from zoology.experiments.flash_vqg.manifest import initialize_manifest
from zoology.experiments.flash_vqg.run_flash_vqg_suite import _resolve_metrics_white_list

root = Path(r"${GENERATED_DIR}")
metrics_white_list = _resolve_metrics_white_list(
    metrics_white_list_raw=None,
    metrics_white_list_file=r"${METRICS_WHITE_LIST_FILE}",
)

configs = []
for fox_remote_formula in [${FORMULAS_PY}]:
    for fox_clr_rank in [${RANKS_PY}]:
        configs.extend(
            build_configs(
                sweep_id="${BASE_LAUNCH_ID_PREFIX}",
                flash_backend="${BACKEND}",
                logger_backend="${LOGGER_BACKEND}",
                include_gdn=False,
                block_len=${BLOCK_LEN},
                dmodels=[${DMODEL}],
                learning_rates=[${LR}],
                if_remote_enabled=True,
                local_num_blocks=${LOCAL_NUM_BLOCKS},
                train_batch_order="${TRAIN_BATCH_ORDER}",
                data_seed=${DATA_SEED},
                num_codebook_vectors_values=[${NUM_CODEBOOK_VECTORS}],
                fox_remote_path_backend="torch",
                fox_remote_formula=fox_remote_formula,
                fox_clr_rank=fox_clr_rank,
                fox_clr_use_den_residual=True,
                fox_clr_remat_mode="post_phase1",
                cache_dir="${CACHE_DIR}",
                metrics_white_list=metrics_white_list,
                wandb_project="${PROJECT}",
                wandb_entity="${ENTITY}",
                max_epochs=${MAX_EPOCHS},
            )
        )

# Write individual config files
for idx, config in enumerate(configs):
    fvqg_kwargs = config.model.sequence_mixer.kwargs["configs"][-1]["kwargs"]
    formula_tag = fvqg_kwargs["fox_remote_formula"].replace("_", "")
    rank_val = int(fvqg_kwargs["fox_clr_rank"])
    config_path = root / f"launch_config_{idx:02d}_{formula_tag}_r{rank_val}.py"
    config_path.write_text(
        f'''# -*- coding: utf-8 -*-
from zoology.experiments.flash_vqg.flash_vqg_suite import build_configs
from zoology.experiments.flash_vqg.run_flash_vqg_suite import _resolve_metrics_white_list

metrics_white_list = _resolve_metrics_white_list(
    metrics_white_list_raw=None,
    metrics_white_list_file=r"${METRICS_WHITE_LIST_FILE}",
)

configs = build_configs(
    sweep_id="${BASE_LAUNCH_ID_PREFIX}",
    flash_backend="${BACKEND}",
    logger_backend="${LOGGER_BACKEND}",
    include_gdn=False,
    block_len=${BLOCK_LEN},
    dmodels=[${DMODEL}],
    learning_rates=[${LR}],
    if_remote_enabled=True,
    local_num_blocks=${LOCAL_NUM_BLOCKS},
    train_batch_order="${TRAIN_BATCH_ORDER}",
    data_seed=${DATA_SEED},
    num_codebook_vectors_values=[${NUM_CODEBOOK_VECTORS}],
    fox_remote_path_backend="torch",
    fox_remote_formula="{fvqg_kwargs["fox_remote_formula"]}",
    fox_clr_rank={rank_val},
    fox_clr_use_den_residual=True,
    fox_clr_remat_mode="post_phase1",
    cache_dir="${CACHE_DIR}",
    metrics_white_list=metrics_white_list,
    wandb_project="${PROJECT}",
    wandb_entity="${ENTITY}",
    max_epochs=${MAX_EPOCHS},
)
''',
        encoding="utf-8",
    )

run_ids = [config.run_id for config in configs]
initialize_manifest(
    manifest_path=Path(r"${MANIFEST_PATH}"),
    launch_id="${LAUNCH_ID}",
    sweep_id="${BASE_LAUNCH_ID_PREFIX}",
    logger_backend="${LOGGER_BACKEND}",
    project="${PROJECT}",
    entity="${ENTITY}",
    run_ids=run_ids,
    launch_config_file=root / "prepare_sweep.py",
)

print("Prepared run_ids:")
for run_id in run_ids:
    print(run_id)
PY

"${PYTHON_BIN}" "${GENERATED_DIR}/prepare_sweep.py" | tee "${LOG_DIR}/prepare.log"

mapfile -t CONFIG_FILES < <(find "${GENERATED_DIR}" -maxdepth 1 -type f -name 'launch_config_*.py' | sort)

if [[ ${#CONFIG_FILES[@]} -eq 0 ]]; then
  echo "没有生成任何 launch config 文件" >&2
  exit 1
fi

launch_one() {
  local config_file="$1"
  local gpu_id="$2"
  local base_name
  base_name="$(basename "${config_file}" .py)"
  local log_file="${LOG_DIR}/${base_name}.log"

  echo "==> 启动 ${base_name} 使用 GPU ${gpu_id}"
  env FLASH_VQG_MANIFEST_PATH="${MANIFEST_PATH}" \
    "${PYTHON_BIN}" -m zoology.launch "${config_file}" --launch-id "${LAUNCH_ID}" --gpus "${gpu_id}" \
    2>&1 | tee "${log_file}"
}

for config_index in "${!CONFIG_FILES[@]}"; do
  gpu_slot=$((config_index % ${#GPU_IDS[@]}))
  gpu_id="$(echo "${GPU_IDS[$gpu_slot]}" | xargs)"
  launch_one "${CONFIG_FILES[$config_index]}" "${gpu_id}"
done

ANALYSIS_CMD=(
  "${PYTHON_BIN}" -m zoology.analysis.flash_vqg.run_flash_vqg_analysis
  --launch-id "${LAUNCH_ID}"
  --source "${ANALYSIS_SOURCE}"
)

echo "==> analysis command:"
printf '    %q ' "${ANALYSIS_CMD[@]}"
printf '\n'

"${ANALYSIS_CMD[@]}" 2>&1 | tee "${LOG_DIR}/analysis.log"

echo "==> completed launch_id=${LAUNCH_ID}"
echo "==> prepare log: ${LOG_DIR}/prepare.log"
echo "==> analysis log: ${LOG_DIR}/analysis.log"
