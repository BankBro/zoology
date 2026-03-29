#!/usr/bin/env bash
set -euo pipefail

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
BASE_LAUNCH_ID_PREFIX="${BASE_LAUNCH_ID_PREFIX:-flash-vqg-clr-rank-sweep-den1-remat}"
BLOCK_LEN="${BLOCK_LEN:-32}"
LOCAL_NUM_BLOCKS="${LOCAL_NUM_BLOCKS:-2}"
NUM_CODEBOOK_VECTORS="${NUM_CODEBOOK_VECTORS:-128}"
TRAIN_BATCH_ORDER="${TRAIN_BATCH_ORDER:-global_shuffle}"
DATA_SEED="${DATA_SEED:-123}"
GPUS="${GPUS:-1}"
METRICS_WHITE_LIST_FILE="${METRICS_WHITE_LIST_FILE:-${ROOT_DIR}/zoology/experiments/flash_vqg/metrics_white_lists/e0.yaml}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/tmp/flash_vqg_clr_r2_r4_r8_den1_single_launch}"
RANKS_CSV="${RANKS_CSV:-2,4,8}"

timestamp_utc="$(date -u +%Y-%m-%d-%H-%M-%S)"
LAUNCH_ID="${LAUNCH_ID:-${BASE_LAUNCH_ID_PREFIX}-${timestamp_utc}}"
GENERATED_DIR="${ROOT_DIR}/zoology/experiments/flash_vqg/generated/${LAUNCH_ID}"
MANIFEST_PATH="${GENERATED_DIR}/manifest.json"

mkdir -p "${GENERATED_DIR}" "${LOG_DIR}"

IFS=',' read -r -a GPU_IDS <<< "${GPUS}"
IFS=',' read -r -a RANKS <<< "${RANKS_CSV}"

if [[ ${#GPU_IDS[@]} -eq 0 ]]; then
  echo "GPUS 不能为空" >&2
  exit 1
fi

if [[ ${#RANKS[@]} -eq 0 ]]; then
  echo "RANKS_CSV 不能为空" >&2
  exit 1
fi

echo "==> launch_id: ${LAUNCH_ID}"
echo "==> launch_id_prefix: ${BASE_LAUNCH_ID_PREFIX}"
echo "==> ranks: ${RANKS_CSV}"
echo "==> gpus: ${GPUS}"
echo "==> generated dir: ${GENERATED_DIR}"
echo "==> log dir: ${LOG_DIR}"

RANKS_SERIALIZED=""
for rank in "${RANKS[@]}"; do
  rank_trimmed="$(echo "${rank}" | xargs)"
  if [[ -z "${rank_trimmed}" ]]; then
    echo "RANKS_CSV 包含空 rank" >&2
    exit 1
  fi
  if [[ -n "${RANKS_SERIALIZED}" ]]; then
    RANKS_SERIALIZED+=", "
  fi
  RANKS_SERIALIZED+="${rank_trimmed}"
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
for fox_clr_rank in [${RANKS_SERIALIZED}]:
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
            fox_remote_formula="clr_v1",
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

for idx, config in enumerate(configs):
    current_rank = int(
        config.model.sequence_mixer.kwargs["configs"][-1]["kwargs"]["fox_clr_rank"]
    )
    config_path = root / f"launch_config_{idx:02d}_r{current_rank}.py"
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
    fox_remote_formula="clr_v1",
    fox_clr_rank={current_rank},
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

mapfile -t CONFIG_FILES < <(find "${GENERATED_DIR}" -maxdepth 1 -type f -name 'launch_config_*_r*.py' | sort)

if [[ ${#CONFIG_FILES[@]} -eq 0 ]]; then
  echo "没有生成任何 launch config 文件" >&2
  exit 1
fi

launch_one() {
  local config_file="$1"
  local gpu_id="$2"
  local base_name
  local rank_tag
  base_name="$(basename "${config_file}" .py)"
  rank_tag="${base_name##*_}"
  local log_file="${LOG_DIR}/${rank_tag}.log"

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
