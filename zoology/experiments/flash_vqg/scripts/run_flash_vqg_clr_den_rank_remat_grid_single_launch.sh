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
BASE_LAUNCH_ID_PREFIX="${BASE_LAUNCH_ID_PREFIX:-flash-vqg-clr-grid-den-rank-remat}"
BLOCK_LEN="${BLOCK_LEN:-32}"
LOCAL_NUM_BLOCKS="${LOCAL_NUM_BLOCKS:-2}"
NUM_CODEBOOK_VECTORS="${NUM_CODEBOOK_VECTORS:-128}"
TRAIN_BATCH_ORDER="${TRAIN_BATCH_ORDER:-global_shuffle}"
DATA_SEED="${DATA_SEED:-123}"
GPU_ID="${GPU_ID:-1}"
METRICS_WHITE_LIST_FILE="${METRICS_WHITE_LIST_FILE:-${ROOT_DIR}/zoology/experiments/flash_vqg/metrics_white_lists/e0.yaml}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/tmp/flash_vqg_clr_den_rank_remat_grid_single_launch}"
RANKS_CSV="${RANKS_CSV:-2,4,8}"
DEN_VALUES_CSV="${DEN_VALUES_CSV:-0,1}"
REMAT_VALUES_CSV="${REMAT_VALUES_CSV:-off,on}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-}"
SKIP_KNOWN_OOM_COMBOS="${SKIP_KNOWN_OOM_COMBOS:-0}"
PARALLELIZE="${PARALLELIZE:-1}"

timestamp_utc="$(date -u +%Y-%m-%d-%H-%M-%S)"
LAUNCH_ID="${LAUNCH_ID:-${BASE_LAUNCH_ID_PREFIX}-${timestamp_utc}}"
GENERATED_DIR="${ROOT_DIR}/zoology/experiments/flash_vqg/generated/${LAUNCH_ID}"
MANIFEST_PATH="${GENERATED_DIR}/manifest.json"

mkdir -p "${GENERATED_DIR}" "${LOG_DIR}"

IFS=',' read -r -a RANKS <<< "${RANKS_CSV}"
IFS=',' read -r -a DEN_VALUES <<< "${DEN_VALUES_CSV}"
IFS=',' read -r -a REMAT_VALUES <<< "${REMAT_VALUES_CSV}"

if [[ ${#RANKS[@]} -eq 0 ]]; then
  echo "RANKS_CSV 不能为空" >&2
  exit 1
fi

if [[ ${#DEN_VALUES[@]} -eq 0 ]]; then
  echo "DEN_VALUES_CSV 不能为空" >&2
  exit 1
fi

if [[ ${#REMAT_VALUES[@]} -eq 0 ]]; then
  echo "REMAT_VALUES_CSV 不能为空" >&2
  exit 1
fi

echo "==> launch_id: ${LAUNCH_ID}"
echo "==> launch_id_prefix: ${BASE_LAUNCH_ID_PREFIX}"
echo "==> ranks: ${RANKS_CSV}"
echo "==> den values: ${DEN_VALUES_CSV}"
echo "==> remat values: ${REMAT_VALUES_CSV}"
echo "==> gradient accumulation steps: ${GRADIENT_ACCUMULATION_STEPS}"
echo "==> train batch size: ${TRAIN_BATCH_SIZE:-<default>}"
echo "==> eval batch size: ${EVAL_BATCH_SIZE:-<default>}"
echo "==> gpu: ${GPU_ID}"
echo "==> parallelize: ${PARALLELIZE}"
echo "==> generated dir: ${GENERATED_DIR}"
echo "==> log dir: ${LOG_DIR}"
echo "==> skip known oom combos: ${SKIP_KNOWN_OOM_COMBOS}"

IFS=',' read -r -a GPU_IDS_RAW <<< "${GPU_ID}"
GPU_IDS=()
for gpu in "${GPU_IDS_RAW[@]}"; do
  gpu_trimmed="$(echo "${gpu}" | xargs)"
  if [[ -z "${gpu_trimmed}" ]]; then
    echo "GPU_ID 包含空项" >&2
    exit 1
  fi
  GPU_IDS+=("${gpu_trimmed}")
done
GPU_COUNT="${#GPU_IDS[@]}"

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

DEN_VALUES_SERIALIZED=""
for den in "${DEN_VALUES[@]}"; do
  den_trimmed="$(echo "${den}" | xargs)"
  if [[ "${den_trimmed}" != "0" && "${den_trimmed}" != "1" ]]; then
    echo "DEN_VALUES_CSV 只支持 0 或 1, 当前收到: ${den_trimmed}" >&2
    exit 1
  fi
  if [[ -n "${DEN_VALUES_SERIALIZED}" ]]; then
    DEN_VALUES_SERIALIZED+=", "
  fi
  DEN_VALUES_SERIALIZED+="${den_trimmed}"
done

REMAT_VALUES_SERIALIZED=""
for remat in "${REMAT_VALUES[@]}"; do
  remat_trimmed="$(echo "${remat}" | xargs)"
  if [[ "${remat_trimmed}" != "off" && "${remat_trimmed}" != "on" ]]; then
    echo "REMAT_VALUES_CSV 只支持 off 或 on, 当前收到: ${remat_trimmed}" >&2
    exit 1
  fi
  if [[ -n "${REMAT_VALUES_SERIALIZED}" ]]; then
    REMAT_VALUES_SERIALIZED+=", "
  fi
  REMAT_VALUES_SERIALIZED+="'${remat_trimmed}'"
done

EXPECTED_CONFIG_COUNT=0
for rank in "${RANKS[@]}"; do
  rank_trimmed="$(echo "${rank}" | xargs)"
  for den in "${DEN_VALUES[@]}"; do
    den_trimmed="$(echo "${den}" | xargs)"
    for remat in "${REMAT_VALUES[@]}"; do
      remat_trimmed="$(echo "${remat}" | xargs)"
      if [[ "${SKIP_KNOWN_OOM_COMBOS}" == "1" && "${rank_trimmed}" == "8" && "${den_trimmed}" == "1" && "${remat_trimmed}" == "off" ]]; then
        continue
      fi
      EXPECTED_CONFIG_COUNT=$((EXPECTED_CONFIG_COUNT + 1))
    done
  done
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
train_batch_size = ${TRAIN_BATCH_SIZE:-None}
eval_batch_size = ${EVAL_BATCH_SIZE:-None}

configs = []
for fox_clr_rank in [${RANKS_SERIALIZED}]:
    for den_flag in [${DEN_VALUES_SERIALIZED}]:
        for remat_flag in [${REMAT_VALUES_SERIALIZED}]:
            if ${SKIP_KNOWN_OOM_COMBOS} and fox_clr_rank == 8 and den_flag == 1 and remat_flag == "off":
                continue

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
                    fox_clr_use_den_residual=bool(den_flag),
                    fox_clr_remat_mode="post_phase1" if remat_flag == "on" else "off",
                    gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS},
                    train_batch_size=train_batch_size,
                    eval_batch_size=eval_batch_size,
                    cache_dir="${CACHE_DIR}",
                    metrics_white_list=metrics_white_list,
                    wandb_project="${PROJECT}",
                    wandb_entity="${ENTITY}",
                    max_epochs=${MAX_EPOCHS},
                )
            )

for idx, config in enumerate(configs):
    kwargs = config.model.sequence_mixer.kwargs["configs"][-1]["kwargs"]
    current_rank = int(kwargs["fox_clr_rank"])
    den_tag = int(bool(kwargs["fox_clr_use_den_residual"]))
    remat_tag = "on" if kwargs["fox_clr_remat_mode"] == "post_phase1" else "off"
    config_path = root / f"launch_config_{idx:02d}_r{current_rank}_den{den_tag}_remat-{remat_tag}.py"
    config_path.write_text(
        f'''# -*- coding: utf-8 -*-
from zoology.experiments.flash_vqg.flash_vqg_suite import build_configs
from zoology.experiments.flash_vqg.run_flash_vqg_suite import _resolve_metrics_white_list

metrics_white_list = _resolve_metrics_white_list(
    metrics_white_list_raw=None,
    metrics_white_list_file=r"${METRICS_WHITE_LIST_FILE}",
)
train_batch_size = ${TRAIN_BATCH_SIZE:-None}
eval_batch_size = ${EVAL_BATCH_SIZE:-None}

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
    fox_clr_use_den_residual={bool(kwargs["fox_clr_use_den_residual"])!r},
    fox_clr_remat_mode={kwargs["fox_clr_remat_mode"]!r},
    gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS},
    train_batch_size=train_batch_size,
    eval_batch_size=eval_batch_size,
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

mapfile -t CONFIG_FILES < <(find "${GENERATED_DIR}" -maxdepth 1 -type f -name 'launch_config_*_r*_den*_remat-*.py' | sort)

if [[ ${#CONFIG_FILES[@]} -ne ${EXPECTED_CONFIG_COUNT} ]]; then
  echo "预期生成 ${EXPECTED_CONFIG_COUNT} 个 launch config, 实际得到 ${#CONFIG_FILES[@]}" >&2
  exit 1
fi

RAY_AVAILABLE=0
if [[ "${PARALLELIZE}" == "1" ]]; then
  if "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import importlib.util
import sys

sys.exit(0 if importlib.util.find_spec("ray") is not None else 1)
PY
  then
    RAY_AVAILABLE=1
  fi
fi

launch_with_ray() {
  local log_file="${LOG_DIR}/launch.log"
  local launch_cmd=(
    "${PYTHON_BIN}" -m zoology.launch
    "${GENERATED_DIR}/prepare_sweep.py"
    --launch-id "${LAUNCH_ID}"
    --gpus "${GPU_ID}"
    -p
  )

  echo "==> launch mode: aggregate-ray"
  echo "==> launch command:"
  printf '    %q ' "${launch_cmd[@]}"
  printf '\n'

  env FLASH_VQG_MANIFEST_PATH="${MANIFEST_PATH}" \
    "${launch_cmd[@]}" 2>&1 | tee "${log_file}"
}

write_shard_config() {
  local shard_index="$1"
  shift
  local shard_file="${GENERATED_DIR}/launch_shard_${shard_index}.py"
  {
    echo "# -*- coding: utf-8 -*-"
    echo "from importlib import util"
    echo
    echo "CONFIG_FILES = ["
    local config_file
    for config_file in "$@"; do
      printf '    r"%s",\n' "${config_file}"
    done
    echo "]"
    echo
    echo "configs = []"
    echo "for idx, config_file in enumerate(CONFIG_FILES):"
    echo "    spec = util.spec_from_file_location(f\"shard_config_{idx}\", config_file)"
    echo "    module = util.module_from_spec(spec)"
    echo "    spec.loader.exec_module(module)"
    echo "    configs.extend(module.configs)"
  } > "${shard_file}"
  printf '%s\n' "${shard_file}"
}

launch_shard_worker() {
  local shard_file="$1"
  local gpu="$2"
  local shard_index="$3"
  local log_file="${LOG_DIR}/worker_gpu${gpu}_shard${shard_index}.log"
  local launch_cmd=(
    "${PYTHON_BIN}" -m zoology.launch
    "${shard_file}"
    --launch-id "${LAUNCH_ID}"
    --gpus "${gpu}"
  )

  {
    echo "==> worker ${shard_index} command:"
    printf '    %q ' "${launch_cmd[@]}"
    printf '\n'
  } >&2

  (
    set -o pipefail
    env FLASH_VQG_MANIFEST_PATH="${MANIFEST_PATH}" \
      "${launch_cmd[@]}" 2>&1 | tee "${log_file}"
  ) &
  SHARD_PIDS+=("$!")
}

record_log_file_for_configs() {
  local log_file="$1"
  shift
  env MANIFEST_PATH="${MANIFEST_PATH}" LOG_FILE="${log_file}" "${PYTHON_BIN}" - "$@" <<'PY'
import importlib.util
import json
import os
import sys
from pathlib import Path

manifest_path = Path(os.environ["MANIFEST_PATH"])
log_file = os.environ["LOG_FILE"]
config_modules = sys.argv[1:]

run_ids = []
for idx, config_module in enumerate(config_modules):
    spec = importlib.util.spec_from_file_location(f"log_manifest_config_{idx}", config_module)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run_ids.extend(config.run_id for config in module.configs)

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
for run in manifest.get("runs", []):
    if run.get("run_id") not in run_ids:
        continue
    run.setdefault("local", {})["log_file"] = log_file

manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

launch_with_manual_shards() {
  echo "==> launch mode: manual-shards"
  echo "==> ray 不可用, 回退到多 worker 分片并发"

  local shard_files=()
  local shard_index
  local config_index
  local gpu
  local shard_configs=()
  SHARD_PIDS=()

  for ((shard_index = 0; shard_index < GPU_COUNT; shard_index++)); do
    shard_configs=()
    for ((config_index = shard_index; config_index < ${#CONFIG_FILES[@]}; config_index += GPU_COUNT)); do
      shard_configs+=("${CONFIG_FILES[config_index]}")
    done
    if [[ ${#shard_configs[@]} -eq 0 ]]; then
      continue
    fi
    shard_files+=("$(write_shard_config "${shard_index}" "${shard_configs[@]}")")
  done

  for ((shard_index = 0; shard_index < ${#shard_files[@]}; shard_index++)); do
    gpu="${GPU_IDS[shard_index]}"
    log_file="${LOG_DIR}/worker_gpu${gpu}_shard${shard_index}.log"
    record_log_file_for_configs "${log_file}" "${shard_files[shard_index]}"
    launch_shard_worker "${shard_files[shard_index]}" "${gpu}" "${shard_index}"
  done

  local failed=0
  local pid
  for pid in "${SHARD_PIDS[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done

  if [[ "${failed}" == "1" ]]; then
    echo "至少一个 worker 失败, 请检查 ${LOG_DIR}/worker_gpu*_shard*.log" >&2
    exit 1
  fi
}

if [[ "${PARALLELIZE}" == "1" && "${RAY_AVAILABLE}" == "1" ]]; then
  record_log_file_for_configs "${LOG_DIR}/launch.log" "${GENERATED_DIR}/prepare_sweep.py"
  launch_with_ray
elif [[ "${GPU_COUNT}" -gt 1 ]]; then
  launch_with_manual_shards
else
  echo "==> launch mode: single-gpu"
  LAUNCH_CMD=(
    "${PYTHON_BIN}" -m zoology.launch
    "${GENERATED_DIR}/prepare_sweep.py"
    --launch-id "${LAUNCH_ID}"
    --gpus "${GPU_ID}"
  )
  echo "==> launch command:"
  printf '    %q ' "${LAUNCH_CMD[@]}"
  printf '\n'

  record_log_file_for_configs "${LOG_DIR}/launch.log" "${GENERATED_DIR}/prepare_sweep.py"

  env FLASH_VQG_MANIFEST_PATH="${MANIFEST_PATH}" \
    "${LAUNCH_CMD[@]}" 2>&1 | tee "${LOG_DIR}/launch.log"
fi

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
echo "==> launch log: ${LOG_DIR}/launch.log"
echo "==> analysis log: ${LOG_DIR}/analysis.log"
