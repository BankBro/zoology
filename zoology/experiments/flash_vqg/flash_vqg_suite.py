import math
from collections.abc import Mapping
from typing import Iterable

from zoology.config import DataConfig, LoggerConfig, TrainConfig
from zoology.data.multiquery_ar import MQARConfig
from zoology.experiments.models_repo import add_flash_vqg, add_gated_delta_net
from zoology.experiments.flash_vqg.metrics_white_list import (
    derive_flash_metric_controls,
    normalize_metrics_white_list,
)


DEFAULT_VOCAB_SIZE = 8_192
DEFAULT_DMODELS = [128]
DEFAULT_LEARNING_RATES = [1e-4, 3e-4, 1e-3, 3e-3]
DEFAULT_WANDB_PROJECT = "flash_vqg_vs_gdn"
DEFAULT_WANDB_ENTITY = "scu-mclab"
DEFAULT_MAX_EPOCHS = 32
DEFAULT_TRAIN_BATCH_ORDER = "sequential"
DEFAULT_CACHE_DIR = "./data/flash_vqg"
DEFAULT_BLOCK_LENS = [8]
DEFAULT_IF_REMOTE_ENABLED = [True]
DEFAULT_LOCAL_NUM_BLOCKS = [2]
DEFAULT_NUM_CODEBOOK_VECTORS_MAP = {64: 64, 128: 128, 256: 256}
DEFAULT_TRAIN_SEED = 123
DEFAULT_DATA_SEED = 123
DEFAULT_TRAIN_BATCH_SIZE = 256
DEFAULT_EVAL_BATCH_SIZE = 32
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 1
DEFAULT_VALIDATIONS_PER_EPOCH = 1
DEFAULT_VQ_SCORE_MODE = "l2"
DEFAULT_VQ_WEIGHT_MODE = "one-hot"
DEFAULT_VQ_UPDATE_MODE = "ema"
DEFAULT_VQ_SOFTMAX_TAU = 1.0
DEFAULT_CODEBOOK_INIT_RNG_MODE = "global"
DEFAULT_CODEBOOK_INIT_SEED = None
DEFAULT_VQ_TOPK = 4
DEFAULT_FOX_REMOTE_READ_TOPK_INITIAL = None
DEFAULT_FOX_REMOTE_READ_TOPK_FINAL = None
DEFAULT_FOX_REMOTE_READ_TOPK_RELEASE_START_TRAIN_STEPS = 0
DEFAULT_FOX_REMOTE_READ_TOPK_RELEASE_END_TRAIN_STEPS = 0
DEFAULT_FOX_REMOTE_READ_TOPK_SCHEDULE = "linear_int"
DEFAULT_FOX_REMOTE_READ_TOPK_EVAL_POLICY = "scheduled"
DEFAULT_FOX_GD_RESIDUAL_RANK = 16
DEFAULT_FOX_GD_RESIDUAL_WRITE_TOPK = 4
DEFAULT_FOX_GD_RESIDUAL_BUILDER = "grouped_chunk_torch_ref"
DEFAULT_FOX_GD_RESIDUAL_PACK_MODE = "semivec_ref"
DEFAULT_FOX_GD_RESIDUAL_CHUNK_SIZE = 64
DEFAULT_FOX_GD_RESIDUAL_MU_MIN_COUNT = 1.0
DEFAULT_FOX_GD_RESIDUAL_ADDR_EPS = 1e-6
DEFAULT_FOX_GD_RESIDUAL_DEN_EPS = 1e-6
DEFAULT_FOX_GD_RESIDUAL_RHO_EPS = 1e-12
DEFAULT_FOX_GD_RESIDUAL_ADDR_INIT_RNG_MODE = "global"
DEFAULT_FOX_GD_RESIDUAL_ADDR_INIT_SEED = None
DEFAULT_FOX_GD_RESIDUAL_BETA_INIT = 0.5
DEFAULT_FOX_GD_RESIDUAL_BETA_CAP = None
DEFAULT_FOX_GD_RESIDUAL_BETA_CAP_FINAL = None
DEFAULT_FOX_GD_RESIDUAL_BETA_CAP_RELEASE_START_TRAIN_STEPS = 0
DEFAULT_FOX_GD_RESIDUAL_BETA_CAP_RELEASE_END_TRAIN_STEPS = 0
DEFAULT_FOX_GD_RESIDUAL_BETA_CAP_EVAL_POLICY = "final"
DEFAULT_FOX_GD_RESIDUAL_BETA_CONTROL_MODE = "hard_cap"
DEFAULT_FOX_GD_RESIDUAL_BETA_SIGMOID_TEMP = 1.0
DEFAULT_FOX_GD_RESIDUAL_BETA_LOW = None
DEFAULT_FOX_GD_RESIDUAL_BETA_HIGH = None
DEFAULT_FOX_GD_RESIDUAL_BETA_LOW_FINAL = None
DEFAULT_FOX_GD_RESIDUAL_BETA_HIGH_FINAL = None
DEFAULT_FOX_GD_RESIDUAL_BETA_BAND_RELEASE_START_TRAIN_STEPS = 0
DEFAULT_FOX_GD_RESIDUAL_BETA_BAND_RELEASE_END_TRAIN_STEPS = 0
DEFAULT_FOX_GD_RESIDUAL_BETA_BAND_EVAL_POLICY = "final"
DEFAULT_FOX_GD_RESIDUAL_BETA_BAND_SCHEDULE = "smoothstep"
DEFAULT_FOX_GD_RESIDUAL_LAMBDA_INIT = 0.05
DEFAULT_FOX_GD_RESIDUAL_LAMBDA_FLOOR = 0.0
DEFAULT_FOX_GD_RESIDUAL_WRITE_STRENGTH_MODE = "renorm_topk"
DEFAULT_FOX_GD_RESIDUAL_WRITE_STRENGTH_CAP = None
DEFAULT_FOX_GD_RESIDUAL_WRITE_STRENGTH_CAP_MODE = "hard"
DEFAULT_FOX_GD_RESIDUAL_WRITE_STRENGTH_CAP_UNTIL_TRAIN_STEPS = 0
DEFAULT_FOX_GD_RESIDUAL_WRITE_STRENGTH_CAP_FINAL = None
DEFAULT_FOX_GD_RESIDUAL_WRITE_STRENGTH_CAP_RELEASE_START_TRAIN_STEPS = 0
DEFAULT_FOX_GD_RESIDUAL_WRITE_STRENGTH_CAP_RELEASE_END_TRAIN_STEPS = 0
DEFAULT_FOX_GD_RESIDUAL_WRITE_STRENGTH_CAP_EVAL_POLICY = "final"
DEFAULT_FOX_GD_RESIDUAL_WRITE_BUDGET = None
DEFAULT_FOX_GD_RESIDUAL_WRITE_BUDGET_FINAL = None
DEFAULT_FOX_GD_RESIDUAL_WRITE_BUDGET_RELEASE_START_TRAIN_STEPS = 0
DEFAULT_FOX_GD_RESIDUAL_WRITE_BUDGET_RELEASE_END_TRAIN_STEPS = 0
DEFAULT_FOX_GD_RESIDUAL_WRITE_BUDGET_EVAL_POLICY = "final"
DEFAULT_FOX_GD_RESIDUAL_WRITE_BUDGET_SCHEDULE = "smoothstep"
DEFAULT_FOX_GD_RESIDUAL_WRITE_TOTAL_CAP = None
DEFAULT_FOX_GD_RESIDUAL_WRITE_TOTAL_CAP_FINAL = None
DEFAULT_FOX_GD_RESIDUAL_WRITE_TOTAL_CAP_RELEASE_START_TRAIN_STEPS = 0
DEFAULT_FOX_GD_RESIDUAL_WRITE_TOTAL_CAP_RELEASE_END_TRAIN_STEPS = 0
DEFAULT_FOX_GD_RESIDUAL_WRITE_TOTAL_CAP_EVAL_POLICY = "final"
DEFAULT_FOX_GD_RESIDUAL_WRITE_TOTAL_CAP_SCHEDULE = "smoothstep"
DEFAULT_FOX_GD_RESIDUAL_WRITE_Q_ALPHA = 1.0
DEFAULT_FOX_GD_RESIDUAL_M_NORM_CAP = None
DEFAULT_FOX_GD_RESIDUAL_UPDATE_NORM_CAP = None
DEFAULT_FOX_GD_RESIDUAL_NORM_WITH_GAIN = False
DEFAULT_FOX_GD_RESIDUAL_USE_SEPARATE_ADDR_CODEBOOK = False
DEFAULT_FOX_GD_RESIDUAL_ADDR_PROJ_ORTHOGONAL_INIT = False


def _normalize_dmodels(dmodels: Iterable[int] | None) -> list[int]:
    values = DEFAULT_DMODELS if dmodels is None else list(dmodels)
    normalized = sorted({int(v) for v in values})
    unsupported = [v for v in normalized if v not in {64, 128, 256}]
    if unsupported:
        raise ValueError(f"暂不支持这些 d_model: {unsupported}. 当前仅支持 64, 128, 256.")
    return normalized


def _normalize_learning_rates(learning_rates: Iterable[float] | None) -> list[float]:
    values = DEFAULT_LEARNING_RATES if learning_rates is None else list(learning_rates)
    return [float(v) for v in values]


def _normalize_seed_values(
    seed_values: Iterable[int] | None = None,
    seed: int | None = None,
) -> list[int]:
    if seed_values is not None and seed is not None:
        raise ValueError("seed_values 和 seed 不能同时传入.")

    raw_values: Iterable[int]
    if seed_values is not None:
        raw_values = seed_values
    elif seed is not None:
        raw_values = [seed]
    else:
        raw_values = [DEFAULT_TRAIN_SEED]

    normalized: list[int] = []
    seen: set[int] = set()
    for value in raw_values:
        parsed = int(value)
        if parsed < 0:
            raise ValueError(f"seed 必须是非负整数, 当前收到: {value}")
        if parsed not in seen:
            normalized.append(parsed)
            seen.add(parsed)
    if not normalized:
        raise ValueError("seed_values 不能为空.")
    return normalized


def _normalize_data_seed(data_seed: int | None = None) -> int:
    if data_seed is None:
        return DEFAULT_DATA_SEED
    parsed = int(data_seed)
    if parsed < 0:
        raise ValueError(f"data_seed 必须是非负整数, 当前收到: {data_seed}")
    return parsed


def _normalize_positive_int(value: int | None, *, field_name: str) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{field_name} 必须是正整数, 当前收到: {value}")
    return parsed


def _normalize_num_codebook_vectors_values(
    num_codebook_vectors_values: Iterable[int] | None,
) -> list[int] | None:
    if num_codebook_vectors_values is None:
        return None

    normalized: list[int] = []
    seen: set[int] = set()
    for value in num_codebook_vectors_values:
        parsed = int(value)
        if parsed <= 0:
            raise ValueError(
                f"num_codebook_vectors 必须是正整数, 当前收到: {value}"
            )
        if parsed not in seen:
            normalized.append(parsed)
            seen.add(parsed)
    if not normalized:
        raise ValueError("num_codebook_vectors_values 不能为空.")
    return normalized


def _normalize_num_codebook_vectors_map(
    num_codebook_vectors_map: Mapping[int, int] | None,
) -> dict[int, int] | None:
    if num_codebook_vectors_map is None:
        return None

    normalized: dict[int, int] = {}
    for d_model, num_codes in num_codebook_vectors_map.items():
        parsed_d_model = int(d_model)
        parsed_num_codes = int(num_codes)
        if parsed_d_model <= 0:
            raise ValueError(f"d_model 必须是正整数, 当前收到: {d_model}")
        if parsed_num_codes <= 0:
            raise ValueError(
                f"num_codebook_vectors 必须是正整数, 当前收到: {num_codes}"
            )
        normalized[parsed_d_model] = parsed_num_codes
    if not normalized:
        raise ValueError("num_codebook_vectors_map 不能为空.")
    return normalized


def _normalize_block_len_values(
    block_len_values: Iterable[int] | None = None,
    block_len: int | None = None,
) -> list[int]:
    if block_len_values is not None and block_len is not None:
        raise ValueError("block_len_values 和 block_len 不能同时传入.")

    raw_values: Iterable[int]
    if block_len_values is not None:
        raw_values = block_len_values
    elif block_len is not None:
        raw_values = [block_len]
    else:
        raw_values = DEFAULT_BLOCK_LENS

    normalized: list[int] = []
    seen: set[int] = set()
    for value in raw_values:
        parsed = int(value)
        if parsed <= 0:
            raise ValueError(f"block_len 必须是正整数, 当前收到: {value}")
        if parsed not in seen:
            normalized.append(parsed)
            seen.add(parsed)
    return normalized


def _normalize_paired_block_local_values(
    paired_block_local_values: Iterable[tuple[int, int]] | None = None,
    *,
    block_len_values: Iterable[int] | None = None,
    block_len: int | None = None,
    local_num_blocks_values: Iterable[int] | None = None,
    local_num_blocks: int | None = None,
) -> list[tuple[int, int]] | None:
    if paired_block_local_values is None:
        return None

    if any(value is not None for value in (block_len_values, block_len, local_num_blocks_values, local_num_blocks)):
        raise ValueError(
            "paired_block_local_values 不能与 block_len/block_len_values/local_num_blocks/local_num_blocks_values 同时传入."
        )

    normalized: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for block_len_value, local_num_blocks_value in paired_block_local_values:
        normalized_block_len = int(block_len_value)
        normalized_local_num_blocks = int(local_num_blocks_value)
        if normalized_block_len <= 0:
            raise ValueError(f"block_len 必须是正整数, 当前收到: {block_len_value}")
        if normalized_local_num_blocks <= 0:
            raise ValueError(
                f"local_num_blocks 必须是正整数, 当前收到: {local_num_blocks_value}"
            )
        pair = (normalized_block_len, normalized_local_num_blocks)
        if pair not in seen:
            normalized.append(pair)
            seen.add(pair)
    if not normalized:
        raise ValueError("paired_block_local_values 不能为空.")
    return normalized


def _normalize_train_batch_order(train_batch_order: str) -> str:
    normalized = str(train_batch_order).lower()
    valid_orders = {"sequential", "global_shuffle", "balanced_interleave"}
    if normalized not in valid_orders:
        raise ValueError(
            f"train_batch_order 只能是 {sorted(valid_orders)}, 当前收到: {train_batch_order}"
        )
    return normalized


def _normalize_vq_softmax_tau(vq_softmax_tau: float) -> float:
    tau = float(vq_softmax_tau)
    if tau <= 0.0:
        raise ValueError(f"vq_softmax_tau 必须是正数, 当前收到: {vq_softmax_tau}")
    return tau


def _normalize_vq_topk(vq_topk: int) -> int:
    parsed = int(vq_topk)
    if parsed <= 0:
        raise ValueError(f"vq_topk 必须是正整数, 当前收到: {vq_topk}")
    return parsed


def _normalize_validations_per_epoch(validations_per_epoch: int | None) -> int:
    parsed = (
        DEFAULT_VALIDATIONS_PER_EPOCH
        if validations_per_epoch is None
        else int(validations_per_epoch)
    )
    if parsed <= 0:
        raise ValueError(
            f"validations_per_epoch 必须是正整数, 当前收到: {validations_per_epoch}"
        )
    return parsed


def _normalize_train_batch_orders(
    train_batch_orders: Iterable[str] | None = None,
    train_batch_order: str | None = None,
) -> list[str]:
    if train_batch_orders is not None and train_batch_order is not None:
        raise ValueError("train_batch_orders 和 train_batch_order 不能同时传入.")

    raw_values: Iterable[str]
    if train_batch_orders is not None:
        raw_values = train_batch_orders
    elif train_batch_order is not None:
        raw_values = [train_batch_order]
    else:
        raw_values = [DEFAULT_TRAIN_BATCH_ORDER]

    normalized: list[str] = []
    seen: set[str] = set()
    for value in raw_values:
        order = _normalize_train_batch_order(value)
        if order not in seen:
            normalized.append(order)
            seen.add(order)
    return normalized


def _normalize_if_remote_enabled_values(
    if_remote_enabled_values: Iterable[bool] | None = None,
    if_remote_enabled: bool | None = None,
) -> list[bool]:
    if if_remote_enabled_values is not None and if_remote_enabled is not None:
        raise ValueError("if_remote_enabled_values 和 if_remote_enabled 不能同时传入.")

    raw_values: Iterable[bool]
    if if_remote_enabled_values is not None:
        raw_values = if_remote_enabled_values
    elif if_remote_enabled is not None:
        raw_values = [if_remote_enabled]
    else:
        raw_values = DEFAULT_IF_REMOTE_ENABLED

    normalized: list[bool] = []
    seen: set[bool] = set()
    for value in raw_values:
        parsed = bool(value)
        if parsed not in seen:
            normalized.append(parsed)
            seen.add(parsed)
    return normalized


def _normalize_local_num_blocks_values(
    local_num_blocks_values: Iterable[int] | None = None,
    local_num_blocks: int | None = None,
) -> list[int]:
    if local_num_blocks_values is not None and local_num_blocks is not None:
        raise ValueError("local_num_blocks_values 和 local_num_blocks 不能同时传入.")

    raw_values: Iterable[int]
    if local_num_blocks_values is not None:
        raw_values = local_num_blocks_values
    elif local_num_blocks is not None:
        raw_values = [local_num_blocks]
    else:
        raw_values = DEFAULT_LOCAL_NUM_BLOCKS

    normalized: list[int] = []
    seen: set[int] = set()
    for value in raw_values:
        parsed = int(value)
        if parsed <= 0:
            raise ValueError(f"local_num_blocks 必须是正整数, 当前收到: {value}")
        if parsed not in seen:
            normalized.append(parsed)
            seen.add(parsed)
    return normalized


def _normalize_fox_remote_path_backend(
    fox_remote_path_backend: str | None,
    *,
    flash_backend: str,
) -> str:
    if fox_remote_path_backend is None:
        return "triton" if flash_backend == "accel" else "torch"

    normalized = str(fox_remote_path_backend).lower()
    if normalized not in {"torch", "triton"}:
        raise ValueError(
            f"fox_remote_path_backend 只能是 ['torch', 'triton'], 当前收到: {fox_remote_path_backend}"
        )
    return normalized


def _normalize_fox_remote_read_topk_values(
    fox_remote_read_topk_values: Iterable[int | None] | None = None,
    *,
    fox_remote_read_topk: int | None = None,
) -> list[int | None]:
    if fox_remote_read_topk_values is not None and fox_remote_read_topk is not None:
        raise ValueError("fox_remote_read_topk_values 和 fox_remote_read_topk 不能同时传入.")

    raw_values: Iterable[int | None]
    if fox_remote_read_topk_values is not None:
        raw_values = fox_remote_read_topk_values
    elif fox_remote_read_topk is not None:
        raw_values = [fox_remote_read_topk]
    else:
        raw_values = [None]

    normalized: list[int | None] = []
    seen: set[int | None] = set()
    for value in raw_values:
        parsed = None if value is None else int(value)
        if parsed is not None and parsed <= 0:
            raise ValueError(
                f"fox_remote_read_topk 必须是正整数或 None, 当前收到: {value}"
            )
        if parsed not in seen:
            normalized.append(parsed)
            seen.add(parsed)
    if not normalized:
        raise ValueError("fox_remote_read_topk_values 不能为空.")
    return normalized


def _normalize_fox_remote_read_topk_schedule(
    *,
    fox_remote_read_topk_initial: int | None,
    fox_remote_read_topk_final: int | None,
    fox_remote_read_topk_release_start_train_steps: int,
    fox_remote_read_topk_release_end_train_steps: int,
    fox_remote_read_topk_schedule: str,
    fox_remote_read_topk_eval_policy: str,
) -> tuple[int | None, int | None, int, int, str, str]:
    initial = _normalize_positive_int(
        fox_remote_read_topk_initial,
        field_name="fox_remote_read_topk_initial",
    )
    final = _normalize_positive_int(
        fox_remote_read_topk_final,
        field_name="fox_remote_read_topk_final",
    )
    if (initial is None) != (final is None):
        raise ValueError(
            "fox_remote_read_topk_initial 和 fox_remote_read_topk_final 必须成对设置."
        )

    start = int(fox_remote_read_topk_release_start_train_steps)
    end = int(fox_remote_read_topk_release_end_train_steps)
    if start < 0:
        raise ValueError(
            "fox_remote_read_topk_release_start_train_steps 必须是非负整数."
        )
    if end < 0:
        raise ValueError(
            "fox_remote_read_topk_release_end_train_steps 必须是非负整数."
        )
    if initial is not None and end <= start:
        raise ValueError(
            "fox_remote_read_topk_release_end_train_steps 必须大于 release_start."
        )

    schedule = str(fox_remote_read_topk_schedule).lower()
    if schedule not in {"linear_int", "step"}:
        raise ValueError(
            "fox_remote_read_topk_schedule 只能是 ['linear_int', 'step'], "
            f"当前收到: {fox_remote_read_topk_schedule}"
        )
    eval_policy = str(fox_remote_read_topk_eval_policy).lower()
    if eval_policy not in {"scheduled", "final"}:
        raise ValueError(
            "fox_remote_read_topk_eval_policy 只能是 ['scheduled', 'final'], "
            f"当前收到: {fox_remote_read_topk_eval_policy}"
        )
    return initial, final, start, end, schedule, eval_policy


def _normalize_fox_remote_formula(fox_remote_formula: str | None) -> str:
    normalized = "legacy" if fox_remote_formula is None else str(fox_remote_formula).lower()
    if normalized not in {"legacy", "clr_v1", "clr_delta_v1", "gd_residual_v1"}:
        raise ValueError(
            "fox_remote_formula 只能是 "
            "['legacy', 'clr_v1', 'clr_delta_v1', 'gd_residual_v1'], "
            f"当前收到: {fox_remote_formula}"
        )
    return normalized


def _normalize_fox_clr_rank(fox_clr_rank: int | None) -> int:
    rank = 4 if fox_clr_rank is None else int(fox_clr_rank)
    if rank < 0:
        raise ValueError(f"fox_clr_rank 必须是非负整数, 当前收到: {fox_clr_rank}")
    return rank


def _normalize_fox_clr_remat_mode(fox_clr_remat_mode: str | None) -> str:
    mode = "off" if fox_clr_remat_mode is None else str(fox_clr_remat_mode).lower()
    if mode not in {"off", "post_phase1"}:
        raise ValueError(
            "fox_clr_remat_mode 只能是 ['off', 'post_phase1'], "
            f"当前收到: {fox_clr_remat_mode}"
        )
    return mode


def _normalize_fox_clr_selector_mode(fox_clr_selector_mode: str | None) -> str:
    mode = "den_aware" if fox_clr_selector_mode is None else str(fox_clr_selector_mode).lower()
    if mode not in {"den_aware", "score_only"}:
        raise ValueError(
            "fox_clr_selector_mode 只能是 ['den_aware', 'score_only'], "
            f"当前收到: {fox_clr_selector_mode}"
        )
    return mode


def _normalize_fox_clr_merge_mode(fox_clr_merge_mode: str | None) -> str:
    mode = "shared_den" if fox_clr_merge_mode is None else str(fox_clr_merge_mode).lower()
    if mode not in {"shared_den", "shared_local_den", "residual_add"}:
        raise ValueError(
            "fox_clr_merge_mode 只能是 ['shared_den', 'shared_local_den', 'residual_add'], "
            f"当前收到: {fox_clr_merge_mode}"
        )
    return mode


def _normalize_fox_clr_gate_mode(fox_clr_gate_mode: str | None) -> str:
    mode = "off" if fox_clr_gate_mode is None else str(fox_clr_gate_mode).lower()
    if mode not in {"off", "shared_query_linear"}:
        raise ValueError(
            "fox_clr_gate_mode 只能是 ['off', 'shared_query_linear'], "
            f"当前收到: {fox_clr_gate_mode}"
        )
    return mode


def _normalize_fox_clr_residual_update_mode(fox_clr_residual_update_mode: str | None) -> str:
    mode = "additive" if fox_clr_residual_update_mode is None else str(fox_clr_residual_update_mode).lower()
    if mode not in {"additive", "delta"}:
        raise ValueError(
            "fox_clr_residual_update_mode 只能是 ['additive', 'delta'], "
            f"当前收到: {fox_clr_residual_update_mode}"
        )
    return mode


def _normalize_fox_clr_residual_forget_mode(fox_clr_residual_forget_mode: str | None) -> str:
    mode = "global" if fox_clr_residual_forget_mode is None else str(fox_clr_residual_forget_mode).lower()
    if mode not in {"global", "code_aware"}:
        raise ValueError(
            "fox_clr_residual_forget_mode 只能是 ['global', 'code_aware'], "
            f"当前收到: {fox_clr_residual_forget_mode}"
        )
    return mode


def _normalize_fox_clr_state_write_topk(fox_clr_state_write_topk: int | None) -> int:
    topk = 4 if fox_clr_state_write_topk is None else int(fox_clr_state_write_topk)
    if topk <= 0:
        raise ValueError(f"fox_clr_state_write_topk 必须是正整数, 当前收到: {fox_clr_state_write_topk}")
    return topk


def _normalize_fox_clr_delta_target_mode(fox_clr_delta_target_mode: str | None) -> str:
    mode = "residual_to_coarse" if fox_clr_delta_target_mode is None else str(fox_clr_delta_target_mode).lower()
    if mode not in {"residual_to_coarse"}:
        raise ValueError(
            "fox_clr_delta_target_mode 只能是 ['residual_to_coarse'], "
            f"当前收到: {fox_clr_delta_target_mode}"
        )
    return mode


def _normalize_fox_gd_residual_rank(fox_gd_residual_rank: int | None) -> int:
    rank = (
        DEFAULT_FOX_GD_RESIDUAL_RANK
        if fox_gd_residual_rank is None
        else int(fox_gd_residual_rank)
    )
    if rank <= 0:
        raise ValueError(f"fox_gd_residual_rank 必须是正整数, 当前收到: {fox_gd_residual_rank}")
    return rank


def _normalize_fox_gd_residual_write_topk(fox_gd_residual_write_topk: int | None) -> int:
    topk = (
        DEFAULT_FOX_GD_RESIDUAL_WRITE_TOPK
        if fox_gd_residual_write_topk is None
        else int(fox_gd_residual_write_topk)
    )
    if topk <= 0:
        raise ValueError(
            f"fox_gd_residual_write_topk 必须是正整数, 当前收到: {fox_gd_residual_write_topk}"
        )
    return topk


def _normalize_fox_gd_residual_builder(fox_gd_residual_builder: str | None) -> str:
    builder = (
        DEFAULT_FOX_GD_RESIDUAL_BUILDER
        if fox_gd_residual_builder is None
        else str(fox_gd_residual_builder).lower()
    )
    if builder not in {"token_step_ref", "grouped_chunk_torch_ref"}:
        raise ValueError(
            "fox_gd_residual_builder 只能是 ['token_step_ref', 'grouped_chunk_torch_ref'], "
            f"当前收到: {fox_gd_residual_builder}"
        )
    return builder


def _normalize_fox_gd_residual_pack_mode(fox_gd_residual_pack_mode: str | None) -> str:
    pack_mode = (
        DEFAULT_FOX_GD_RESIDUAL_PACK_MODE
        if fox_gd_residual_pack_mode is None
        else str(fox_gd_residual_pack_mode).lower()
    )
    if pack_mode not in {"loop_ref", "semivec_ref"}:
        raise ValueError(
            "fox_gd_residual_pack_mode 只能是 ['loop_ref', 'semivec_ref'], "
            f"当前收到: {fox_gd_residual_pack_mode}"
        )
    return pack_mode


def _normalize_fox_gd_residual_write_strength_mode(
    fox_gd_residual_write_strength_mode: str | None,
) -> str:
    mode = (
        DEFAULT_FOX_GD_RESIDUAL_WRITE_STRENGTH_MODE
        if fox_gd_residual_write_strength_mode is None
        else str(fox_gd_residual_write_strength_mode).lower()
    )
    if mode not in {
        "renorm_topk",
        "topk_mass_scaled",
        "renorm_topk_top1_scaled",
        "renorm_topk_top1_sq_scaled",
        "budgeted_topk_beta",
        "budgeted_topk_beta_scaled_cap",
        "budgeted_topk_beta_scaled_peak_total_cap",
    }:
        raise ValueError(
            "fox_gd_residual_write_strength_mode 只能是 "
            "['renorm_topk', 'topk_mass_scaled', 'renorm_topk_top1_scaled', "
            "'renorm_topk_top1_sq_scaled', 'budgeted_topk_beta', "
            "'budgeted_topk_beta_scaled_cap', "
            "'budgeted_topk_beta_scaled_peak_total_cap'], "
            f"当前收到: {fox_gd_residual_write_strength_mode}"
        )
    return mode


def _normalize_fox_gd_residual_write_strength_cap_mode(value: str | None) -> str:
    mode = (
        DEFAULT_FOX_GD_RESIDUAL_WRITE_STRENGTH_CAP_MODE
        if value is None
        else str(value).lower()
    )
    if mode not in {"hard", "smooth_exp", "smooth_l4", "softplus"}:
        raise ValueError(
            "fox_gd_residual_write_strength_cap_mode 只能是 "
            "['hard', 'smooth_exp', 'smooth_l4', 'softplus'], "
            f"当前收到: {value}"
        )
    return mode


def _normalize_local_rng_mode(value: str | None, *, field_name: str) -> str:
    mode = "global" if value is None else str(value).lower()
    if mode not in {"global", "local_burn", "local_noburn"}:
        raise ValueError(
            f"{field_name} 只能是 ['global', 'local_burn', 'local_noburn'], "
            f"当前收到: {value}"
        )
    return mode


def _normalize_optional_int_seed(value: int | None, *, field_name: str) -> int | None:
    if value is None:
        return None
    seed = int(value)
    if seed < 0:
        raise ValueError(f"{field_name} 必须是非负整数或 None, 当前收到: {value}")
    return seed


def _normalize_fox_gd_residual_chunk_size(fox_gd_residual_chunk_size: int | None) -> int:
    chunk_size = (
        DEFAULT_FOX_GD_RESIDUAL_CHUNK_SIZE
        if fox_gd_residual_chunk_size is None
        else int(fox_gd_residual_chunk_size)
    )
    if chunk_size <= 0:
        raise ValueError(
            f"fox_gd_residual_chunk_size 必须是正整数, 当前收到: {fox_gd_residual_chunk_size}"
        )
    return chunk_size


def _normalize_fox_gd_residual_mu_min_count(
    fox_gd_residual_mu_min_count: float | None,
) -> float:
    value = (
        DEFAULT_FOX_GD_RESIDUAL_MU_MIN_COUNT
        if fox_gd_residual_mu_min_count is None
        else float(fox_gd_residual_mu_min_count)
    )
    if value < 0.0:
        raise ValueError(
            "fox_gd_residual_mu_min_count 必须是非负数, "
            f"当前收到: {fox_gd_residual_mu_min_count}"
        )
    return value


def _normalize_fox_gd_residual_positive_float(
    value: float | None,
    *,
    field_name: str,
    default: float,
) -> float:
    resolved = default if value is None else float(value)
    if resolved <= 0.0 or not math.isfinite(resolved):
        raise ValueError(f"{field_name} 必须是有限正数, 当前收到: {value}")
    return resolved


def _normalize_fox_gd_residual_optional_positive_float(
    value: float | None,
    *,
    field_name: str,
) -> float | None:
    if value is None:
        return None
    resolved = float(value)
    if resolved <= 0.0:
        raise ValueError(f"{field_name} 必须是正数或 None, 当前收到: {value}")
    return resolved


def _normalize_fox_gd_residual_prob(
    value: float | None,
    *,
    field_name: str,
    default: float,
) -> float:
    resolved = default if value is None else float(value)
    if not (0.0 < resolved < 1.0):
        raise ValueError(f"{field_name} 必须在 (0, 1) 内, 当前收到: {value}")
    return resolved


def _normalize_fox_gd_residual_optional_prob_cap(
    value: float | None,
    *,
    field_name: str,
) -> float | None:
    if value is None:
        return None
    resolved = float(value)
    if not (0.0 < resolved <= 1.0):
        raise ValueError(f"{field_name} 必须在 (0, 1] 内或为 None, 当前收到: {value}")
    return resolved


def _normalize_fox_gd_residual_optional_prob_bound(
    value: float | None,
    *,
    field_name: str,
) -> float | None:
    if value is None:
        return None
    resolved = float(value)
    if not (0.0 <= resolved <= 1.0):
        raise ValueError(f"{field_name} 必须在 [0, 1] 内或为 None, 当前收到: {value}")
    return resolved


def _normalize_fox_gd_residual_beta_control_mode(value: str | None) -> str:
    mode = (
        DEFAULT_FOX_GD_RESIDUAL_BETA_CONTROL_MODE
        if value is None
        else str(value).lower()
    )
    if mode not in {"hard_cap", "bounded_sigmoid"}:
        raise ValueError(
            "fox_gd_residual_beta_control_mode 只能是 "
            "['hard_cap', 'bounded_sigmoid'], "
            f"当前收到: {value}"
        )
    return mode


def _normalize_fox_gd_residual_beta_band_eval_policy(value: str | None) -> str:
    policy = (
        DEFAULT_FOX_GD_RESIDUAL_BETA_BAND_EVAL_POLICY
        if value is None
        else str(value).lower()
    )
    if policy not in {"final", "scheduled"}:
        raise ValueError(
            "fox_gd_residual_beta_band_eval_policy 只能是 "
            "['final', 'scheduled'], "
            f"当前收到: {value}"
        )
    return policy


def _normalize_fox_gd_residual_beta_band_schedule(value: str | None) -> str:
    schedule = (
        DEFAULT_FOX_GD_RESIDUAL_BETA_BAND_SCHEDULE
        if value is None
        else str(value).lower()
    )
    if schedule not in {"smoothstep", "cosine"}:
        raise ValueError(
            "fox_gd_residual_beta_band_schedule 只能是 "
            "['smoothstep', 'cosine'], "
            f"当前收到: {value}"
        )
    return schedule


def _normalize_fox_gd_residual_prob_floor(
    value: float | None,
    *,
    field_name: str,
    default: float,
) -> float:
    resolved = default if value is None else float(value)
    if not (0.0 <= resolved < 1.0):
        raise ValueError(f"{field_name} 必须在 [0, 1) 内, 当前收到: {value}")
    return resolved


def _sampler_run_tag(train_batch_order: str) -> str:
    return {
        "sequential": "seq",
        "global_shuffle": "gshuffle",
        "balanced_interleave": "binterleave",
    }[train_batch_order]


def _structure_run_tag(*, local_num_blocks: int, if_remote_enabled: bool) -> str:
    return f"local{int(local_num_blocks)}-remote{int(bool(if_remote_enabled))}"


def _remote_read_run_tag(read_topk: int | None) -> str:
    return "dense" if read_topk is None else f"top{int(read_topk)}"


def _optional_float_run_tag(value: float | None, *, prefix: str) -> str:
    if value is None:
        return ""
    value_tag = str(float(value)).replace("-", "m").replace(".", "p")
    return f"-{prefix}{value_tag}"


def _remote_formula_run_tag(
    *,
    fox_remote_formula: str,
    fox_clr_rank: int,
    fox_clr_use_den_residual: bool,
    fox_remote_read_topk_initial: int | None = DEFAULT_FOX_REMOTE_READ_TOPK_INITIAL,
    fox_remote_read_topk_final: int | None = DEFAULT_FOX_REMOTE_READ_TOPK_FINAL,
    fox_remote_read_topk_release_start_train_steps: int = DEFAULT_FOX_REMOTE_READ_TOPK_RELEASE_START_TRAIN_STEPS,
    fox_remote_read_topk_release_end_train_steps: int = DEFAULT_FOX_REMOTE_READ_TOPK_RELEASE_END_TRAIN_STEPS,
    fox_remote_read_topk_schedule: str = DEFAULT_FOX_REMOTE_READ_TOPK_SCHEDULE,
    fox_remote_read_topk_eval_policy: str = DEFAULT_FOX_REMOTE_READ_TOPK_EVAL_POLICY,
    fox_gd_residual_rank: int = DEFAULT_FOX_GD_RESIDUAL_RANK,
    fox_gd_residual_write_topk: int = DEFAULT_FOX_GD_RESIDUAL_WRITE_TOPK,
    fox_gd_residual_builder: str = DEFAULT_FOX_GD_RESIDUAL_BUILDER,
    fox_gd_residual_pack_mode: str = DEFAULT_FOX_GD_RESIDUAL_PACK_MODE,
    fox_gd_residual_write_strength_mode: str = DEFAULT_FOX_GD_RESIDUAL_WRITE_STRENGTH_MODE,
    fox_gd_residual_write_strength_cap: float | None = DEFAULT_FOX_GD_RESIDUAL_WRITE_STRENGTH_CAP,
    fox_gd_residual_write_strength_cap_mode: str = DEFAULT_FOX_GD_RESIDUAL_WRITE_STRENGTH_CAP_MODE,
    fox_gd_residual_write_strength_cap_until_train_steps: int = DEFAULT_FOX_GD_RESIDUAL_WRITE_STRENGTH_CAP_UNTIL_TRAIN_STEPS,
    fox_gd_residual_write_strength_cap_final: float | None = DEFAULT_FOX_GD_RESIDUAL_WRITE_STRENGTH_CAP_FINAL,
    fox_gd_residual_write_strength_cap_release_start_train_steps: int = DEFAULT_FOX_GD_RESIDUAL_WRITE_STRENGTH_CAP_RELEASE_START_TRAIN_STEPS,
    fox_gd_residual_write_strength_cap_release_end_train_steps: int = DEFAULT_FOX_GD_RESIDUAL_WRITE_STRENGTH_CAP_RELEASE_END_TRAIN_STEPS,
    fox_gd_residual_write_strength_cap_eval_policy: str = DEFAULT_FOX_GD_RESIDUAL_WRITE_STRENGTH_CAP_EVAL_POLICY,
    fox_gd_residual_write_budget: float | None = DEFAULT_FOX_GD_RESIDUAL_WRITE_BUDGET,
    fox_gd_residual_write_budget_final: float | None = DEFAULT_FOX_GD_RESIDUAL_WRITE_BUDGET_FINAL,
    fox_gd_residual_write_budget_release_start_train_steps: int = DEFAULT_FOX_GD_RESIDUAL_WRITE_BUDGET_RELEASE_START_TRAIN_STEPS,
    fox_gd_residual_write_budget_release_end_train_steps: int = DEFAULT_FOX_GD_RESIDUAL_WRITE_BUDGET_RELEASE_END_TRAIN_STEPS,
    fox_gd_residual_write_budget_eval_policy: str = DEFAULT_FOX_GD_RESIDUAL_WRITE_BUDGET_EVAL_POLICY,
    fox_gd_residual_write_budget_schedule: str = DEFAULT_FOX_GD_RESIDUAL_WRITE_BUDGET_SCHEDULE,
    fox_gd_residual_write_total_cap: float | None = DEFAULT_FOX_GD_RESIDUAL_WRITE_TOTAL_CAP,
    fox_gd_residual_write_total_cap_final: float | None = DEFAULT_FOX_GD_RESIDUAL_WRITE_TOTAL_CAP_FINAL,
    fox_gd_residual_write_total_cap_release_start_train_steps: int = DEFAULT_FOX_GD_RESIDUAL_WRITE_TOTAL_CAP_RELEASE_START_TRAIN_STEPS,
    fox_gd_residual_write_total_cap_release_end_train_steps: int = DEFAULT_FOX_GD_RESIDUAL_WRITE_TOTAL_CAP_RELEASE_END_TRAIN_STEPS,
    fox_gd_residual_write_total_cap_eval_policy: str = DEFAULT_FOX_GD_RESIDUAL_WRITE_TOTAL_CAP_EVAL_POLICY,
    fox_gd_residual_write_total_cap_schedule: str = DEFAULT_FOX_GD_RESIDUAL_WRITE_TOTAL_CAP_SCHEDULE,
    fox_gd_residual_write_q_alpha: float = DEFAULT_FOX_GD_RESIDUAL_WRITE_Q_ALPHA,
    fox_gd_residual_m_norm_cap: float | None = DEFAULT_FOX_GD_RESIDUAL_M_NORM_CAP,
    fox_gd_residual_update_norm_cap: float | None = DEFAULT_FOX_GD_RESIDUAL_UPDATE_NORM_CAP,
    fox_gd_residual_beta_cap: float | None = DEFAULT_FOX_GD_RESIDUAL_BETA_CAP,
    fox_gd_residual_beta_cap_final: float | None = DEFAULT_FOX_GD_RESIDUAL_BETA_CAP_FINAL,
    fox_gd_residual_beta_cap_release_start_train_steps: int = DEFAULT_FOX_GD_RESIDUAL_BETA_CAP_RELEASE_START_TRAIN_STEPS,
    fox_gd_residual_beta_cap_release_end_train_steps: int = DEFAULT_FOX_GD_RESIDUAL_BETA_CAP_RELEASE_END_TRAIN_STEPS,
    fox_gd_residual_beta_cap_eval_policy: str = DEFAULT_FOX_GD_RESIDUAL_BETA_CAP_EVAL_POLICY,
    fox_gd_residual_beta_control_mode: str = DEFAULT_FOX_GD_RESIDUAL_BETA_CONTROL_MODE,
    fox_gd_residual_beta_sigmoid_temp: float = DEFAULT_FOX_GD_RESIDUAL_BETA_SIGMOID_TEMP,
    fox_gd_residual_beta_low: float | None = DEFAULT_FOX_GD_RESIDUAL_BETA_LOW,
    fox_gd_residual_beta_high: float | None = DEFAULT_FOX_GD_RESIDUAL_BETA_HIGH,
    fox_gd_residual_beta_low_final: float | None = DEFAULT_FOX_GD_RESIDUAL_BETA_LOW_FINAL,
    fox_gd_residual_beta_high_final: float | None = DEFAULT_FOX_GD_RESIDUAL_BETA_HIGH_FINAL,
    fox_gd_residual_beta_band_release_start_train_steps: int = DEFAULT_FOX_GD_RESIDUAL_BETA_BAND_RELEASE_START_TRAIN_STEPS,
    fox_gd_residual_beta_band_release_end_train_steps: int = DEFAULT_FOX_GD_RESIDUAL_BETA_BAND_RELEASE_END_TRAIN_STEPS,
    fox_gd_residual_beta_band_eval_policy: str = DEFAULT_FOX_GD_RESIDUAL_BETA_BAND_EVAL_POLICY,
    fox_gd_residual_beta_band_schedule: str = DEFAULT_FOX_GD_RESIDUAL_BETA_BAND_SCHEDULE,
    fox_gd_residual_lambda_floor: float = DEFAULT_FOX_GD_RESIDUAL_LAMBDA_FLOOR,
) -> str:
    if fox_remote_formula == "legacy":
        return "legacy"
    if fox_remote_formula == "clr_delta_v1":
        return f"clrdelta1-r{int(fox_clr_rank)}-den{int(bool(fox_clr_use_den_residual))}"
    if fox_remote_formula == "gd_residual_v1":
        builder_tag = "gctref" if fox_gd_residual_builder == "grouped_chunk_torch_ref" else "tsref"
        pack_tag = "semivec" if fox_gd_residual_pack_mode == "semivec_ref" else "loop"
        read_schedule_tag = ""
        if fox_remote_read_topk_initial is not None:
            read_schedule_tag = (
                f"-readsched{int(fox_remote_read_topk_initial)}to"
                f"{int(fox_remote_read_topk_final)}"
                f"-rel{int(fox_remote_read_topk_release_start_train_steps)}to"
                f"{int(fox_remote_read_topk_release_end_train_steps)}"
                f"-{fox_remote_read_topk_schedule}"
                f"-eval{fox_remote_read_topk_eval_policy}"
            )
        return (
            f"gdr1-r{int(fox_gd_residual_rank)}-"
            f"wk{int(fox_gd_residual_write_topk)}-"
            f"{builder_tag}-{pack_tag}"
            f"{read_schedule_tag}"
            f"{'' if fox_gd_residual_write_strength_mode == DEFAULT_FOX_GD_RESIDUAL_WRITE_STRENGTH_MODE else f'-wmode{fox_gd_residual_write_strength_mode}'}"
            f"{_optional_float_run_tag(fox_gd_residual_write_strength_cap, prefix='wcap')}"
            f"{'' if fox_gd_residual_write_strength_cap_mode == DEFAULT_FOX_GD_RESIDUAL_WRITE_STRENGTH_CAP_MODE else f'-wcapmode{fox_gd_residual_write_strength_cap_mode}'}"
            f"{'' if int(fox_gd_residual_write_strength_cap_until_train_steps) == DEFAULT_FOX_GD_RESIDUAL_WRITE_STRENGTH_CAP_UNTIL_TRAIN_STEPS else f'-wcapuntil{int(fox_gd_residual_write_strength_cap_until_train_steps)}'}"
            f"{_optional_float_run_tag(fox_gd_residual_write_strength_cap_final, prefix='wcapfinal')}"
            f"{'' if int(fox_gd_residual_write_strength_cap_release_start_train_steps) == DEFAULT_FOX_GD_RESIDUAL_WRITE_STRENGTH_CAP_RELEASE_START_TRAIN_STEPS else f'-wcaprelstart{int(fox_gd_residual_write_strength_cap_release_start_train_steps)}'}"
            f"{'' if int(fox_gd_residual_write_strength_cap_release_end_train_steps) == DEFAULT_FOX_GD_RESIDUAL_WRITE_STRENGTH_CAP_RELEASE_END_TRAIN_STEPS else f'-wcaprelend{int(fox_gd_residual_write_strength_cap_release_end_train_steps)}'}"
            f"{'' if fox_gd_residual_write_strength_cap_eval_policy == DEFAULT_FOX_GD_RESIDUAL_WRITE_STRENGTH_CAP_EVAL_POLICY else f'-wcapeval{fox_gd_residual_write_strength_cap_eval_policy}'}"
            f"{_optional_float_run_tag(fox_gd_residual_write_budget, prefix='wbudget')}"
            f"{_optional_float_run_tag(fox_gd_residual_write_budget_final, prefix='wbudgetfinal')}"
            f"{'' if int(fox_gd_residual_write_budget_release_start_train_steps) == DEFAULT_FOX_GD_RESIDUAL_WRITE_BUDGET_RELEASE_START_TRAIN_STEPS else f'-wbudgetrelstart{int(fox_gd_residual_write_budget_release_start_train_steps)}'}"
            f"{'' if int(fox_gd_residual_write_budget_release_end_train_steps) == DEFAULT_FOX_GD_RESIDUAL_WRITE_BUDGET_RELEASE_END_TRAIN_STEPS else f'-wbudgetrelend{int(fox_gd_residual_write_budget_release_end_train_steps)}'}"
            f"{'' if fox_gd_residual_write_budget_eval_policy == DEFAULT_FOX_GD_RESIDUAL_WRITE_BUDGET_EVAL_POLICY else f'-wbudgeteval{fox_gd_residual_write_budget_eval_policy}'}"
            f"{'' if fox_gd_residual_write_budget_schedule == DEFAULT_FOX_GD_RESIDUAL_WRITE_BUDGET_SCHEDULE else f'-wbudgetsched{fox_gd_residual_write_budget_schedule}'}"
            f"{_optional_float_run_tag(fox_gd_residual_write_total_cap, prefix='wtotalcap')}"
            f"{_optional_float_run_tag(fox_gd_residual_write_total_cap_final, prefix='wtotalcapfinal')}"
            f"{'' if int(fox_gd_residual_write_total_cap_release_start_train_steps) == DEFAULT_FOX_GD_RESIDUAL_WRITE_TOTAL_CAP_RELEASE_START_TRAIN_STEPS else f'-wtotalcaprelstart{int(fox_gd_residual_write_total_cap_release_start_train_steps)}'}"
            f"{'' if int(fox_gd_residual_write_total_cap_release_end_train_steps) == DEFAULT_FOX_GD_RESIDUAL_WRITE_TOTAL_CAP_RELEASE_END_TRAIN_STEPS else f'-wtotalcaprelend{int(fox_gd_residual_write_total_cap_release_end_train_steps)}'}"
            f"{'' if fox_gd_residual_write_total_cap_eval_policy == DEFAULT_FOX_GD_RESIDUAL_WRITE_TOTAL_CAP_EVAL_POLICY else f'-wtotalcapeval{fox_gd_residual_write_total_cap_eval_policy}'}"
            f"{'' if fox_gd_residual_write_total_cap_schedule == DEFAULT_FOX_GD_RESIDUAL_WRITE_TOTAL_CAP_SCHEDULE else f'-wtotalcapsched{fox_gd_residual_write_total_cap_schedule}'}"
            f"{'' if float(fox_gd_residual_write_q_alpha) == DEFAULT_FOX_GD_RESIDUAL_WRITE_Q_ALPHA else _optional_float_run_tag(fox_gd_residual_write_q_alpha, prefix='wqalpha')}"
            f"{_optional_float_run_tag(fox_gd_residual_m_norm_cap, prefix='mcap')}"
            f"{_optional_float_run_tag(fox_gd_residual_update_norm_cap, prefix='ucap')}"
            f"{_optional_float_run_tag(fox_gd_residual_beta_cap, prefix='betacap')}"
            f"{_optional_float_run_tag(fox_gd_residual_beta_cap_final, prefix='betacapfinal')}"
            f"{'' if int(fox_gd_residual_beta_cap_release_start_train_steps) == DEFAULT_FOX_GD_RESIDUAL_BETA_CAP_RELEASE_START_TRAIN_STEPS else f'-betacaprelstart{int(fox_gd_residual_beta_cap_release_start_train_steps)}'}"
            f"{'' if int(fox_gd_residual_beta_cap_release_end_train_steps) == DEFAULT_FOX_GD_RESIDUAL_BETA_CAP_RELEASE_END_TRAIN_STEPS else f'-betacaprelend{int(fox_gd_residual_beta_cap_release_end_train_steps)}'}"
            f"{'' if fox_gd_residual_beta_cap_eval_policy == DEFAULT_FOX_GD_RESIDUAL_BETA_CAP_EVAL_POLICY else f'-betacapeval{fox_gd_residual_beta_cap_eval_policy}'}"
            f"{'' if fox_gd_residual_beta_control_mode == DEFAULT_FOX_GD_RESIDUAL_BETA_CONTROL_MODE else f'-betactrl{fox_gd_residual_beta_control_mode}'}"
            f"{'' if float(fox_gd_residual_beta_sigmoid_temp) == DEFAULT_FOX_GD_RESIDUAL_BETA_SIGMOID_TEMP else _optional_float_run_tag(fox_gd_residual_beta_sigmoid_temp, prefix='betatemp')}"
            f"{_optional_float_run_tag(fox_gd_residual_beta_low, prefix='betalow')}"
            f"{_optional_float_run_tag(fox_gd_residual_beta_high, prefix='betahigh')}"
            f"{_optional_float_run_tag(fox_gd_residual_beta_low_final, prefix='betalowfinal')}"
            f"{_optional_float_run_tag(fox_gd_residual_beta_high_final, prefix='betahighfinal')}"
            f"{'' if int(fox_gd_residual_beta_band_release_start_train_steps) == DEFAULT_FOX_GD_RESIDUAL_BETA_BAND_RELEASE_START_TRAIN_STEPS else f'-betabandrelstart{int(fox_gd_residual_beta_band_release_start_train_steps)}'}"
            f"{'' if int(fox_gd_residual_beta_band_release_end_train_steps) == DEFAULT_FOX_GD_RESIDUAL_BETA_BAND_RELEASE_END_TRAIN_STEPS else f'-betabandrelend{int(fox_gd_residual_beta_band_release_end_train_steps)}'}"
            f"{'' if fox_gd_residual_beta_band_eval_policy == DEFAULT_FOX_GD_RESIDUAL_BETA_BAND_EVAL_POLICY else f'-betabandeval{fox_gd_residual_beta_band_eval_policy}'}"
            f"{'' if fox_gd_residual_beta_band_schedule == DEFAULT_FOX_GD_RESIDUAL_BETA_BAND_SCHEDULE else f'-betabandsched{fox_gd_residual_beta_band_schedule}'}"
            f"{'' if fox_gd_residual_lambda_floor == DEFAULT_FOX_GD_RESIDUAL_LAMBDA_FLOOR else _optional_float_run_tag(fox_gd_residual_lambda_floor, prefix='lambdafloor')}"
        )
    return f"clr1-r{int(fox_clr_rank)}-den{int(bool(fox_clr_use_den_residual))}"


def _clr_remat_run_tag(fox_clr_remat_mode: str) -> str:
    return {
        "off": "off",
        "post_phase1": "postp1",
    }[fox_clr_remat_mode]


def _clr_residual_write_run_tag(
    *,
    fox_clr_residual_update_mode: str,
    fox_clr_residual_forget_mode: str,
    fox_clr_state_write_topk: int,
) -> str:
    return (
        f"rwrite-{fox_clr_residual_update_mode}-"
        f"{fox_clr_residual_forget_mode}-wk{int(fox_clr_state_write_topk)}"
    )


def _build_data_config(
    vocab_size: int,
    train_batch_order: str,
    data_seed: int = DEFAULT_DATA_SEED,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> tuple[DataConfig, int]:
    train_configs = [
        MQARConfig(vocab_size=vocab_size, input_seq_len=64, num_examples=100_000, num_kv_pairs=4),
        MQARConfig(vocab_size=vocab_size, input_seq_len=128, num_examples=20_000, num_kv_pairs=8),
        MQARConfig(vocab_size=vocab_size, input_seq_len=256, num_examples=20_000, num_kv_pairs=16),
        MQARConfig(vocab_size=vocab_size, input_seq_len=256, num_examples=20_000, num_kv_pairs=32),
        MQARConfig(vocab_size=vocab_size, input_seq_len=256, num_examples=20_000, num_kv_pairs=64),
    ]
    test_configs = [
        MQARConfig(vocab_size=vocab_size, input_seq_len=64, num_examples=1_000, num_kv_pairs=4),
        MQARConfig(vocab_size=vocab_size, input_seq_len=64, num_examples=1_000, num_kv_pairs=8),
        MQARConfig(vocab_size=vocab_size, input_seq_len=64, num_examples=1_000, num_kv_pairs=16),
        MQARConfig(vocab_size=vocab_size, input_seq_len=128, num_examples=1_000, num_kv_pairs=32),
        MQARConfig(vocab_size=vocab_size, input_seq_len=256, num_examples=1_000, num_kv_pairs=64),
        MQARConfig(vocab_size=vocab_size, input_seq_len=512, num_examples=1_000, num_kv_pairs=64),
        MQARConfig(vocab_size=vocab_size, input_seq_len=512, num_examples=1_000, num_kv_pairs=128),
        MQARConfig(vocab_size=vocab_size, input_seq_len=1024, num_examples=1_000, num_kv_pairs=256),
    ]
    input_seq_len = max(c.input_seq_len for c in train_configs + test_configs)
    data = DataConfig(
        train_configs=train_configs,
        test_configs=test_configs,
        batch_size=(DEFAULT_TRAIN_BATCH_SIZE, DEFAULT_EVAL_BATCH_SIZE),
        train_batch_order=train_batch_order,
        seed=data_seed,
        cache_dir=cache_dir,
    )
    return data, input_seq_len


def _build_conv_mixer(input_seq_len: int) -> dict:
    return dict(
        name="zoology.mixers.base_conv.BaseConv",
        kwargs={
            "l_max": input_seq_len,
            "kernel_size": 3,
            "implicit_long_conv": True,
        },
    )


def _flash_run_tag(*, flash_backend: str, block_len: int) -> str:
    backend_tag = "accel" if flash_backend == "accel" else "torch"
    normalized_block_len = int(block_len)
    if normalized_block_len == 8:
        block_tag = ""
    elif normalized_block_len == 32:
        block_tag = "-block32"
    else:
        block_tag = f"-block{normalized_block_len}"
    return f"flash_vqg_h2_{backend_tag}{block_tag}"


def _extract_flash_num_codebook_vectors(model) -> int:
    configs = model.sequence_mixer.kwargs["configs"]
    flash_vqg_mixer = configs[-1]
    return int(flash_vqg_mixer["kwargs"]["num_codebook_vectors"])


def build_configs(
    *,
    sweep_id: str = "flash-vqg-suite",
    flash_backend: str = "accel",
    logger_backend: str = "wandb",
    include_gdn: bool = True,
    block_len: int | None = None,
    block_len_values: Iterable[int] | None = None,
    paired_block_local_values: Iterable[tuple[int, int]] | None = None,
    dmodels: Iterable[int] | None = None,
    learning_rates: Iterable[float] | None = None,
    if_remote_enabled_values: Iterable[bool] | None = None,
    if_remote_enabled: bool | None = None,
    local_num_blocks_values: Iterable[int] | None = None,
    local_num_blocks: int | None = None,
    wandb_project: str = DEFAULT_WANDB_PROJECT,
    wandb_entity: str = DEFAULT_WANDB_ENTITY,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    train_batch_orders: Iterable[str] | None = None,
    train_batch_order: str | None = None,
    seed_values: Iterable[int] | None = None,
    seed: int | None = None,
    data_seed: int = DEFAULT_DATA_SEED,
    num_codebook_vectors_values: Iterable[int] | None = None,
    num_codebook_vectors_map: Mapping[int, int] | None = None,
    fox_remote_path_backend: str | None = None,
    fox_remote_read_topk_values: Iterable[int | None] | None = None,
    fox_remote_read_topk: int | None = None,
    fox_remote_read_topk_initial: int | None = DEFAULT_FOX_REMOTE_READ_TOPK_INITIAL,
    fox_remote_read_topk_final: int | None = DEFAULT_FOX_REMOTE_READ_TOPK_FINAL,
    fox_remote_read_topk_release_start_train_steps: int = DEFAULT_FOX_REMOTE_READ_TOPK_RELEASE_START_TRAIN_STEPS,
    fox_remote_read_topk_release_end_train_steps: int = DEFAULT_FOX_REMOTE_READ_TOPK_RELEASE_END_TRAIN_STEPS,
    fox_remote_read_topk_schedule: str = DEFAULT_FOX_REMOTE_READ_TOPK_SCHEDULE,
    fox_remote_read_topk_eval_policy: str = DEFAULT_FOX_REMOTE_READ_TOPK_EVAL_POLICY,
    fox_remote_formula: str = "legacy",
    fox_clr_rank: int = 4,
    fox_clr_use_den_residual: bool = True,
    fox_clr_remat_mode: str = "off",
    fox_clr_selector_mode: str = "den_aware",
    fox_clr_merge_mode: str = "shared_den",
    fox_clr_gate_mode: str = "off",
    fox_clr_lambda_remote: float = 1.0,
    fox_clr_gate_init_bias: float = -2.0,
    fox_clr_residual_update_mode: str = "additive",
    fox_clr_residual_forget_mode: str = "global",
    fox_clr_state_write_topk: int = 4,
    fox_clr_delta_target_mode: str = "residual_to_coarse",
    fox_gd_residual_rank: int = DEFAULT_FOX_GD_RESIDUAL_RANK,
    fox_gd_residual_write_topk: int = DEFAULT_FOX_GD_RESIDUAL_WRITE_TOPK,
    fox_gd_residual_builder: str = DEFAULT_FOX_GD_RESIDUAL_BUILDER,
    fox_gd_residual_pack_mode: str = DEFAULT_FOX_GD_RESIDUAL_PACK_MODE,
    fox_gd_residual_chunk_size: int = DEFAULT_FOX_GD_RESIDUAL_CHUNK_SIZE,
    fox_gd_residual_mu_min_count: float = DEFAULT_FOX_GD_RESIDUAL_MU_MIN_COUNT,
    fox_gd_residual_addr_eps: float = DEFAULT_FOX_GD_RESIDUAL_ADDR_EPS,
    fox_gd_residual_den_eps: float = DEFAULT_FOX_GD_RESIDUAL_DEN_EPS,
    fox_gd_residual_rho_eps: float = DEFAULT_FOX_GD_RESIDUAL_RHO_EPS,
    fox_gd_residual_addr_init_rng_mode: str = DEFAULT_FOX_GD_RESIDUAL_ADDR_INIT_RNG_MODE,
    fox_gd_residual_addr_init_seed: int | None = DEFAULT_FOX_GD_RESIDUAL_ADDR_INIT_SEED,
    fox_gd_residual_beta_init: float = DEFAULT_FOX_GD_RESIDUAL_BETA_INIT,
    fox_gd_residual_beta_cap: float | None = DEFAULT_FOX_GD_RESIDUAL_BETA_CAP,
    fox_gd_residual_beta_cap_final: float | None = DEFAULT_FOX_GD_RESIDUAL_BETA_CAP_FINAL,
    fox_gd_residual_beta_cap_release_start_train_steps: int = DEFAULT_FOX_GD_RESIDUAL_BETA_CAP_RELEASE_START_TRAIN_STEPS,
    fox_gd_residual_beta_cap_release_end_train_steps: int = DEFAULT_FOX_GD_RESIDUAL_BETA_CAP_RELEASE_END_TRAIN_STEPS,
    fox_gd_residual_beta_cap_eval_policy: str = DEFAULT_FOX_GD_RESIDUAL_BETA_CAP_EVAL_POLICY,
    fox_gd_residual_beta_control_mode: str = DEFAULT_FOX_GD_RESIDUAL_BETA_CONTROL_MODE,
    fox_gd_residual_beta_sigmoid_temp: float = DEFAULT_FOX_GD_RESIDUAL_BETA_SIGMOID_TEMP,
    fox_gd_residual_beta_low: float | None = DEFAULT_FOX_GD_RESIDUAL_BETA_LOW,
    fox_gd_residual_beta_high: float | None = DEFAULT_FOX_GD_RESIDUAL_BETA_HIGH,
    fox_gd_residual_beta_low_final: float | None = DEFAULT_FOX_GD_RESIDUAL_BETA_LOW_FINAL,
    fox_gd_residual_beta_high_final: float | None = DEFAULT_FOX_GD_RESIDUAL_BETA_HIGH_FINAL,
    fox_gd_residual_beta_band_release_start_train_steps: int = DEFAULT_FOX_GD_RESIDUAL_BETA_BAND_RELEASE_START_TRAIN_STEPS,
    fox_gd_residual_beta_band_release_end_train_steps: int = DEFAULT_FOX_GD_RESIDUAL_BETA_BAND_RELEASE_END_TRAIN_STEPS,
    fox_gd_residual_beta_band_eval_policy: str = DEFAULT_FOX_GD_RESIDUAL_BETA_BAND_EVAL_POLICY,
    fox_gd_residual_beta_band_schedule: str = DEFAULT_FOX_GD_RESIDUAL_BETA_BAND_SCHEDULE,
    fox_gd_residual_lambda_init: float = DEFAULT_FOX_GD_RESIDUAL_LAMBDA_INIT,
    fox_gd_residual_lambda_floor: float = DEFAULT_FOX_GD_RESIDUAL_LAMBDA_FLOOR,
    fox_gd_residual_write_strength_mode: str = DEFAULT_FOX_GD_RESIDUAL_WRITE_STRENGTH_MODE,
    fox_gd_residual_write_strength_cap: float | None = DEFAULT_FOX_GD_RESIDUAL_WRITE_STRENGTH_CAP,
    fox_gd_residual_write_strength_cap_mode: str = DEFAULT_FOX_GD_RESIDUAL_WRITE_STRENGTH_CAP_MODE,
    fox_gd_residual_write_strength_cap_until_train_steps: int = DEFAULT_FOX_GD_RESIDUAL_WRITE_STRENGTH_CAP_UNTIL_TRAIN_STEPS,
    fox_gd_residual_write_strength_cap_final: float | None = DEFAULT_FOX_GD_RESIDUAL_WRITE_STRENGTH_CAP_FINAL,
    fox_gd_residual_write_strength_cap_release_start_train_steps: int = DEFAULT_FOX_GD_RESIDUAL_WRITE_STRENGTH_CAP_RELEASE_START_TRAIN_STEPS,
    fox_gd_residual_write_strength_cap_release_end_train_steps: int = DEFAULT_FOX_GD_RESIDUAL_WRITE_STRENGTH_CAP_RELEASE_END_TRAIN_STEPS,
    fox_gd_residual_write_strength_cap_eval_policy: str = DEFAULT_FOX_GD_RESIDUAL_WRITE_STRENGTH_CAP_EVAL_POLICY,
    fox_gd_residual_write_budget: float | None = DEFAULT_FOX_GD_RESIDUAL_WRITE_BUDGET,
    fox_gd_residual_write_budget_final: float | None = DEFAULT_FOX_GD_RESIDUAL_WRITE_BUDGET_FINAL,
    fox_gd_residual_write_budget_release_start_train_steps: int = DEFAULT_FOX_GD_RESIDUAL_WRITE_BUDGET_RELEASE_START_TRAIN_STEPS,
    fox_gd_residual_write_budget_release_end_train_steps: int = DEFAULT_FOX_GD_RESIDUAL_WRITE_BUDGET_RELEASE_END_TRAIN_STEPS,
    fox_gd_residual_write_budget_eval_policy: str = DEFAULT_FOX_GD_RESIDUAL_WRITE_BUDGET_EVAL_POLICY,
    fox_gd_residual_write_budget_schedule: str = DEFAULT_FOX_GD_RESIDUAL_WRITE_BUDGET_SCHEDULE,
    fox_gd_residual_write_total_cap: float | None = DEFAULT_FOX_GD_RESIDUAL_WRITE_TOTAL_CAP,
    fox_gd_residual_write_total_cap_final: float | None = DEFAULT_FOX_GD_RESIDUAL_WRITE_TOTAL_CAP_FINAL,
    fox_gd_residual_write_total_cap_release_start_train_steps: int = DEFAULT_FOX_GD_RESIDUAL_WRITE_TOTAL_CAP_RELEASE_START_TRAIN_STEPS,
    fox_gd_residual_write_total_cap_release_end_train_steps: int = DEFAULT_FOX_GD_RESIDUAL_WRITE_TOTAL_CAP_RELEASE_END_TRAIN_STEPS,
    fox_gd_residual_write_total_cap_eval_policy: str = DEFAULT_FOX_GD_RESIDUAL_WRITE_TOTAL_CAP_EVAL_POLICY,
    fox_gd_residual_write_total_cap_schedule: str = DEFAULT_FOX_GD_RESIDUAL_WRITE_TOTAL_CAP_SCHEDULE,
    fox_gd_residual_write_q_alpha: float = DEFAULT_FOX_GD_RESIDUAL_WRITE_Q_ALPHA,
    fox_gd_residual_m_norm_cap: float | None = DEFAULT_FOX_GD_RESIDUAL_M_NORM_CAP,
    fox_gd_residual_update_norm_cap: float | None = DEFAULT_FOX_GD_RESIDUAL_UPDATE_NORM_CAP,
    fox_gd_residual_norm_with_gain: bool = DEFAULT_FOX_GD_RESIDUAL_NORM_WITH_GAIN,
    fox_gd_residual_use_separate_addr_codebook: bool = DEFAULT_FOX_GD_RESIDUAL_USE_SEPARATE_ADDR_CODEBOOK,
    fox_gd_residual_addr_proj_orthogonal_init: bool = DEFAULT_FOX_GD_RESIDUAL_ADDR_PROJ_ORTHOGONAL_INIT,
    experiment_part: str | None = None,
    experiment_mode: str | None = None,
    vq_score_mode: str = DEFAULT_VQ_SCORE_MODE,
    vq_weight_mode: str = DEFAULT_VQ_WEIGHT_MODE,
    vq_update_mode: str = DEFAULT_VQ_UPDATE_MODE,
    vq_softmax_tau: float = DEFAULT_VQ_SOFTMAX_TAU,
    codebook_init_rng_mode: str = DEFAULT_CODEBOOK_INIT_RNG_MODE,
    codebook_init_seed: int | None = DEFAULT_CODEBOOK_INIT_SEED,
    vq_topk: int = DEFAULT_VQ_TOPK,
    gradient_accumulation_steps: int = DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    validations_per_epoch: int = DEFAULT_VALIDATIONS_PER_EPOCH,
    early_stopping_metric: str | None = "valid/accuracy",
    early_stopping_threshold: float | None = 0.99,
    train_batch_size: int | None = None,
    eval_batch_size: int | None = None,
    cache_dir: str = DEFAULT_CACHE_DIR,
    metrics_white_list: Iterable[str] | None = None,
    read_churn_probe_enabled: bool = False,
    read_churn_probe_valid_batches: Iterable[int] | None = None,
    read_churn_probe_max_samples: int = 16,
    read_churn_probe_query_only: bool = True,
) -> list[TrainConfig]:
    flash_backend = str(flash_backend).lower()
    if flash_backend not in {"accel", "torch"}:
        raise ValueError(f"flash_backend 只能是 'accel' 或 'torch', 当前收到: {flash_backend}")
    logger_backend = str(logger_backend).lower()
    if logger_backend not in {"wandb", "swanlab", "none"}:
        raise ValueError(
            f"logger_backend 只能是 ['none', 'swanlab', 'wandb'], 当前收到: {logger_backend}"
        )

    paired_block_local_list = _normalize_paired_block_local_values(
        paired_block_local_values,
        block_len_values=block_len_values,
        block_len=block_len,
        local_num_blocks_values=local_num_blocks_values,
        local_num_blocks=local_num_blocks,
    )
    if paired_block_local_list is None:
        block_len_list = _normalize_block_len_values(
            block_len_values=block_len_values,
            block_len=block_len,
        )
        local_num_blocks_list = _normalize_local_num_blocks_values(
            local_num_blocks_values=local_num_blocks_values,
            local_num_blocks=local_num_blocks,
        )
        structure_pairs = [
            (current_block_len, current_local_num_blocks)
            for current_block_len in block_len_list
            for current_local_num_blocks in local_num_blocks_list
        ]
    else:
        structure_pairs = paired_block_local_list

    dmodels_list = _normalize_dmodels(dmodels)
    learning_rates_list = _normalize_learning_rates(learning_rates)
    seed_values_list = _normalize_seed_values(seed_values=seed_values, seed=seed)
    normalized_data_seed = _normalize_data_seed(data_seed)
    train_batch_orders_list = _normalize_train_batch_orders(train_batch_orders, train_batch_order)
    normalized_metrics_white_list = normalize_metrics_white_list(metrics_white_list)
    metric_controls = derive_flash_metric_controls(normalized_metrics_white_list)
    read_churn_probe_valid_batches = (
        [0]
        if read_churn_probe_valid_batches is None
        else [int(idx) for idx in read_churn_probe_valid_batches]
    )
    normalized_num_codebook_vectors_values = _normalize_num_codebook_vectors_values(
        num_codebook_vectors_values
    )
    normalized_num_codebook_vectors_map = _normalize_num_codebook_vectors_map(
        num_codebook_vectors_map
    )
    if (
        normalized_num_codebook_vectors_values is not None
        and normalized_num_codebook_vectors_map is not None
    ):
        raise ValueError(
            "num_codebook_vectors_values 和 num_codebook_vectors_map 不能同时传入."
        )
    if_remote_enabled_list = _normalize_if_remote_enabled_values(
        if_remote_enabled_values=if_remote_enabled_values,
        if_remote_enabled=if_remote_enabled,
    )
    resolved_remote_path_backend = _normalize_fox_remote_path_backend(
        fox_remote_path_backend,
        flash_backend=flash_backend,
    )
    resolved_remote_formula = _normalize_fox_remote_formula(fox_remote_formula)
    resolved_clr_rank = _normalize_fox_clr_rank(fox_clr_rank)
    resolved_clr_remat_mode = _normalize_fox_clr_remat_mode(fox_clr_remat_mode)
    resolved_clr_selector_mode = _normalize_fox_clr_selector_mode(fox_clr_selector_mode)
    resolved_clr_merge_mode = _normalize_fox_clr_merge_mode(fox_clr_merge_mode)
    resolved_clr_gate_mode = _normalize_fox_clr_gate_mode(fox_clr_gate_mode)
    resolved_clr_residual_update_mode = _normalize_fox_clr_residual_update_mode(fox_clr_residual_update_mode)
    resolved_clr_residual_forget_mode = _normalize_fox_clr_residual_forget_mode(fox_clr_residual_forget_mode)
    resolved_clr_state_write_topk = _normalize_fox_clr_state_write_topk(fox_clr_state_write_topk)
    resolved_clr_delta_target_mode = _normalize_fox_clr_delta_target_mode(fox_clr_delta_target_mode)
    resolved_gd_residual_rank = _normalize_fox_gd_residual_rank(fox_gd_residual_rank)
    resolved_gd_residual_write_topk = _normalize_fox_gd_residual_write_topk(fox_gd_residual_write_topk)
    resolved_gd_residual_builder = _normalize_fox_gd_residual_builder(fox_gd_residual_builder)
    resolved_gd_residual_pack_mode = _normalize_fox_gd_residual_pack_mode(fox_gd_residual_pack_mode)
    resolved_gd_residual_chunk_size = _normalize_fox_gd_residual_chunk_size(fox_gd_residual_chunk_size)
    resolved_gd_residual_mu_min_count = _normalize_fox_gd_residual_mu_min_count(fox_gd_residual_mu_min_count)
    resolved_gd_residual_addr_eps = _normalize_fox_gd_residual_positive_float(
        fox_gd_residual_addr_eps,
        field_name="fox_gd_residual_addr_eps",
        default=DEFAULT_FOX_GD_RESIDUAL_ADDR_EPS,
    )
    resolved_gd_residual_den_eps = _normalize_fox_gd_residual_positive_float(
        fox_gd_residual_den_eps,
        field_name="fox_gd_residual_den_eps",
        default=DEFAULT_FOX_GD_RESIDUAL_DEN_EPS,
    )
    resolved_gd_residual_rho_eps = _normalize_fox_gd_residual_positive_float(
        fox_gd_residual_rho_eps,
        field_name="fox_gd_residual_rho_eps",
        default=DEFAULT_FOX_GD_RESIDUAL_RHO_EPS,
    )
    resolved_gd_residual_addr_init_rng_mode = _normalize_local_rng_mode(
        fox_gd_residual_addr_init_rng_mode,
        field_name="fox_gd_residual_addr_init_rng_mode",
    )
    resolved_gd_residual_addr_init_seed = _normalize_optional_int_seed(
        fox_gd_residual_addr_init_seed,
        field_name="fox_gd_residual_addr_init_seed",
    )
    if (
        resolved_gd_residual_addr_init_rng_mode != "global"
        and resolved_gd_residual_addr_init_seed is None
    ):
        raise ValueError(
            "fox_gd_residual_addr_init_seed 必须在 "
            "fox_gd_residual_addr_init_rng_mode 非 global 时设置."
        )
    resolved_gd_residual_beta_init = _normalize_fox_gd_residual_prob(
        fox_gd_residual_beta_init,
        field_name="fox_gd_residual_beta_init",
        default=DEFAULT_FOX_GD_RESIDUAL_BETA_INIT,
    )
    resolved_gd_residual_beta_cap = _normalize_fox_gd_residual_optional_prob_cap(
        fox_gd_residual_beta_cap,
        field_name="fox_gd_residual_beta_cap",
    )
    resolved_gd_residual_beta_cap_final = _normalize_fox_gd_residual_optional_prob_cap(
        fox_gd_residual_beta_cap_final,
        field_name="fox_gd_residual_beta_cap_final",
    )
    resolved_gd_residual_beta_cap_release_start_train_steps = int(
        fox_gd_residual_beta_cap_release_start_train_steps
    )
    resolved_gd_residual_beta_cap_release_end_train_steps = int(
        fox_gd_residual_beta_cap_release_end_train_steps
    )
    if resolved_gd_residual_beta_cap_release_start_train_steps < 0:
        raise ValueError(
            "fox_gd_residual_beta_cap_release_start_train_steps 必须是非负整数."
        )
    if resolved_gd_residual_beta_cap_release_end_train_steps < 0:
        raise ValueError(
            "fox_gd_residual_beta_cap_release_end_train_steps 必须是非负整数."
        )
    if (
        resolved_gd_residual_beta_cap_final is not None
        and resolved_gd_residual_beta_cap is None
    ):
        raise ValueError(
            "fox_gd_residual_beta_cap_final 需要同时设置 "
            "fox_gd_residual_beta_cap."
        )
    if (
        resolved_gd_residual_beta_cap_final is not None
        and resolved_gd_residual_beta_cap_release_end_train_steps
        <= resolved_gd_residual_beta_cap_release_start_train_steps
    ):
        raise ValueError(
            "fox_gd_residual_beta_cap_release_end_train_steps 必须大于 "
            "release_start."
        )
    resolved_gd_residual_beta_cap_eval_policy = str(
        fox_gd_residual_beta_cap_eval_policy
    ).lower()
    if resolved_gd_residual_beta_cap_eval_policy not in {"final", "scheduled"}:
        raise ValueError(
            "fox_gd_residual_beta_cap_eval_policy 只能是 "
            "['final', 'scheduled'], "
            f"当前收到: {fox_gd_residual_beta_cap_eval_policy}"
        )
    resolved_gd_residual_beta_control_mode = (
        _normalize_fox_gd_residual_beta_control_mode(
            fox_gd_residual_beta_control_mode
        )
    )
    resolved_gd_residual_beta_sigmoid_temp = _normalize_fox_gd_residual_positive_float(
        fox_gd_residual_beta_sigmoid_temp,
        field_name="fox_gd_residual_beta_sigmoid_temp",
        default=DEFAULT_FOX_GD_RESIDUAL_BETA_SIGMOID_TEMP,
    )
    resolved_gd_residual_beta_low = _normalize_fox_gd_residual_optional_prob_bound(
        fox_gd_residual_beta_low,
        field_name="fox_gd_residual_beta_low",
    )
    resolved_gd_residual_beta_high = _normalize_fox_gd_residual_optional_prob_bound(
        fox_gd_residual_beta_high,
        field_name="fox_gd_residual_beta_high",
    )
    resolved_gd_residual_beta_low_final = _normalize_fox_gd_residual_optional_prob_bound(
        fox_gd_residual_beta_low_final,
        field_name="fox_gd_residual_beta_low_final",
    )
    resolved_gd_residual_beta_high_final = _normalize_fox_gd_residual_optional_prob_bound(
        fox_gd_residual_beta_high_final,
        field_name="fox_gd_residual_beta_high_final",
    )
    resolved_gd_residual_beta_band_release_start_train_steps = int(
        fox_gd_residual_beta_band_release_start_train_steps
    )
    resolved_gd_residual_beta_band_release_end_train_steps = int(
        fox_gd_residual_beta_band_release_end_train_steps
    )
    if resolved_gd_residual_beta_band_release_start_train_steps < 0:
        raise ValueError(
            "fox_gd_residual_beta_band_release_start_train_steps 必须是非负整数."
        )
    if resolved_gd_residual_beta_band_release_end_train_steps < 0:
        raise ValueError(
            "fox_gd_residual_beta_band_release_end_train_steps 必须是非负整数."
        )
    resolved_gd_residual_beta_band_eval_policy = (
        _normalize_fox_gd_residual_beta_band_eval_policy(
            fox_gd_residual_beta_band_eval_policy
        )
    )
    resolved_gd_residual_beta_band_schedule = (
        _normalize_fox_gd_residual_beta_band_schedule(
            fox_gd_residual_beta_band_schedule
        )
    )
    beta_band_final_set = (
        resolved_gd_residual_beta_low_final is not None
        or resolved_gd_residual_beta_high_final is not None
    )
    if resolved_gd_residual_beta_control_mode == "bounded_sigmoid":
        if (
            resolved_gd_residual_beta_low is None
            or resolved_gd_residual_beta_high is None
        ):
            raise ValueError(
                "fox_gd_residual_beta_low 和 fox_gd_residual_beta_high "
                "必须在 bounded_sigmoid 模式下设置."
            )
        if not (resolved_gd_residual_beta_low < resolved_gd_residual_beta_high):
            raise ValueError(
                "fox_gd_residual_beta_low 必须小于 fox_gd_residual_beta_high."
            )
        if not (
            resolved_gd_residual_beta_low
            < resolved_gd_residual_beta_init
            < resolved_gd_residual_beta_high
        ):
            raise ValueError(
                "fox_gd_residual_beta_init 必须位于 bounded beta band 内部."
            )
    if beta_band_final_set:
        if (
            resolved_gd_residual_beta_low_final is None
            or resolved_gd_residual_beta_high_final is None
        ):
            raise ValueError(
                "fox_gd_residual_beta_low_final 和 "
                "fox_gd_residual_beta_high_final 必须成对设置."
            )
        if not (
            resolved_gd_residual_beta_low_final
            < resolved_gd_residual_beta_high_final
        ):
            raise ValueError(
                "fox_gd_residual_beta_low_final 必须小于 "
                "fox_gd_residual_beta_high_final."
            )
        if (
            resolved_gd_residual_beta_band_release_end_train_steps
            <= resolved_gd_residual_beta_band_release_start_train_steps
        ):
            raise ValueError(
                "fox_gd_residual_beta_band_release_end_train_steps 必须大于 "
                "release_start."
            )
    resolved_gd_residual_lambda_init = _normalize_fox_gd_residual_prob(
        fox_gd_residual_lambda_init,
        field_name="fox_gd_residual_lambda_init",
        default=DEFAULT_FOX_GD_RESIDUAL_LAMBDA_INIT,
    )
    resolved_gd_residual_lambda_floor = _normalize_fox_gd_residual_prob_floor(
        fox_gd_residual_lambda_floor,
        field_name="fox_gd_residual_lambda_floor",
        default=DEFAULT_FOX_GD_RESIDUAL_LAMBDA_FLOOR,
    )
    resolved_gd_residual_write_strength_mode = (
        _normalize_fox_gd_residual_write_strength_mode(
            fox_gd_residual_write_strength_mode
        )
    )
    resolved_gd_residual_write_strength_cap = _normalize_fox_gd_residual_optional_positive_float(
        fox_gd_residual_write_strength_cap,
        field_name="fox_gd_residual_write_strength_cap",
    )
    resolved_gd_residual_write_strength_cap_mode = (
        _normalize_fox_gd_residual_write_strength_cap_mode(
            fox_gd_residual_write_strength_cap_mode
        )
    )
    resolved_gd_residual_write_strength_cap_until_train_steps = int(
        fox_gd_residual_write_strength_cap_until_train_steps
    )
    if resolved_gd_residual_write_strength_cap_until_train_steps < 0:
        raise ValueError(
            "fox_gd_residual_write_strength_cap_until_train_steps 必须是非负整数."
        )
    resolved_gd_residual_write_strength_cap_final = _normalize_fox_gd_residual_optional_positive_float(
        fox_gd_residual_write_strength_cap_final,
        field_name="fox_gd_residual_write_strength_cap_final",
    )
    resolved_gd_residual_write_strength_cap_release_start_train_steps = int(
        fox_gd_residual_write_strength_cap_release_start_train_steps
    )
    resolved_gd_residual_write_strength_cap_release_end_train_steps = int(
        fox_gd_residual_write_strength_cap_release_end_train_steps
    )
    if resolved_gd_residual_write_strength_cap_release_start_train_steps < 0:
        raise ValueError(
            "fox_gd_residual_write_strength_cap_release_start_train_steps 必须是非负整数."
        )
    if resolved_gd_residual_write_strength_cap_release_end_train_steps < 0:
        raise ValueError(
            "fox_gd_residual_write_strength_cap_release_end_train_steps 必须是非负整数."
        )
    if (
        resolved_gd_residual_write_strength_cap_final is not None
        and resolved_gd_residual_write_strength_cap is None
    ):
        raise ValueError(
            "fox_gd_residual_write_strength_cap_final 需要同时设置 "
            "fox_gd_residual_write_strength_cap."
        )
    if (
        resolved_gd_residual_write_strength_cap_final is not None
        and resolved_gd_residual_write_strength_cap_release_end_train_steps
        <= resolved_gd_residual_write_strength_cap_release_start_train_steps
    ):
        raise ValueError(
            "fox_gd_residual_write_strength_cap_release_end_train_steps 必须大于 "
            "release_start."
        )
    resolved_gd_residual_write_strength_cap_eval_policy = str(
        fox_gd_residual_write_strength_cap_eval_policy
    ).lower()
    if resolved_gd_residual_write_strength_cap_eval_policy not in {"final", "scheduled"}:
        raise ValueError(
            "fox_gd_residual_write_strength_cap_eval_policy 只能是 "
            "['final', 'scheduled'], "
            f"当前收到: {fox_gd_residual_write_strength_cap_eval_policy}"
        )
    resolved_gd_residual_write_budget = _normalize_fox_gd_residual_optional_positive_float(
        fox_gd_residual_write_budget,
        field_name="fox_gd_residual_write_budget",
    )
    resolved_gd_residual_write_budget_final = _normalize_fox_gd_residual_optional_positive_float(
        fox_gd_residual_write_budget_final,
        field_name="fox_gd_residual_write_budget_final",
    )
    resolved_gd_residual_write_budget_release_start_train_steps = int(
        fox_gd_residual_write_budget_release_start_train_steps
    )
    resolved_gd_residual_write_budget_release_end_train_steps = int(
        fox_gd_residual_write_budget_release_end_train_steps
    )
    if resolved_gd_residual_write_budget_release_start_train_steps < 0:
        raise ValueError(
            "fox_gd_residual_write_budget_release_start_train_steps 必须是非负整数."
        )
    if resolved_gd_residual_write_budget_release_end_train_steps < 0:
        raise ValueError(
            "fox_gd_residual_write_budget_release_end_train_steps 必须是非负整数."
        )
    resolved_gd_residual_write_budget_eval_policy = str(
        fox_gd_residual_write_budget_eval_policy
    ).lower()
    if resolved_gd_residual_write_budget_eval_policy not in {"final", "scheduled"}:
        raise ValueError(
            "fox_gd_residual_write_budget_eval_policy 只能是 "
            "['final', 'scheduled'], "
            f"当前收到: {fox_gd_residual_write_budget_eval_policy}"
        )
    resolved_gd_residual_write_budget_schedule = str(
        fox_gd_residual_write_budget_schedule
    ).lower()
    if resolved_gd_residual_write_budget_schedule not in {"smoothstep", "cosine"}:
        raise ValueError(
            "fox_gd_residual_write_budget_schedule 只能是 "
            "['smoothstep', 'cosine'], "
            f"当前收到: {fox_gd_residual_write_budget_schedule}"
        )
    if resolved_gd_residual_write_budget_final is not None:
        if resolved_gd_residual_write_budget is None:
            raise ValueError(
                "fox_gd_residual_write_budget_final 需要同时设置 "
                "fox_gd_residual_write_budget."
            )
        if (
            resolved_gd_residual_write_budget_release_end_train_steps
            <= resolved_gd_residual_write_budget_release_start_train_steps
        ):
            raise ValueError(
                "fox_gd_residual_write_budget_release_end_train_steps 必须大于 "
                "release_start."
            )
    resolved_gd_residual_write_total_cap = _normalize_fox_gd_residual_optional_positive_float(
        fox_gd_residual_write_total_cap,
        field_name="fox_gd_residual_write_total_cap",
    )
    resolved_gd_residual_write_total_cap_final = _normalize_fox_gd_residual_optional_positive_float(
        fox_gd_residual_write_total_cap_final,
        field_name="fox_gd_residual_write_total_cap_final",
    )
    resolved_gd_residual_write_total_cap_release_start_train_steps = int(
        fox_gd_residual_write_total_cap_release_start_train_steps
    )
    resolved_gd_residual_write_total_cap_release_end_train_steps = int(
        fox_gd_residual_write_total_cap_release_end_train_steps
    )
    if resolved_gd_residual_write_total_cap_release_start_train_steps < 0:
        raise ValueError(
            "fox_gd_residual_write_total_cap_release_start_train_steps 必须是非负整数."
        )
    if resolved_gd_residual_write_total_cap_release_end_train_steps < 0:
        raise ValueError(
            "fox_gd_residual_write_total_cap_release_end_train_steps 必须是非负整数."
        )
    resolved_gd_residual_write_total_cap_eval_policy = str(
        fox_gd_residual_write_total_cap_eval_policy
    ).lower()
    if resolved_gd_residual_write_total_cap_eval_policy not in {"final", "scheduled"}:
        raise ValueError(
            "fox_gd_residual_write_total_cap_eval_policy 只能是 "
            "['final', 'scheduled'], "
            f"当前收到: {fox_gd_residual_write_total_cap_eval_policy}"
        )
    resolved_gd_residual_write_total_cap_schedule = str(
        fox_gd_residual_write_total_cap_schedule
    ).lower()
    if resolved_gd_residual_write_total_cap_schedule not in {"smoothstep", "cosine"}:
        raise ValueError(
            "fox_gd_residual_write_total_cap_schedule 只能是 "
            "['smoothstep', 'cosine'], "
            f"当前收到: {fox_gd_residual_write_total_cap_schedule}"
        )
    if resolved_gd_residual_write_total_cap_final is not None:
        if resolved_gd_residual_write_total_cap is None:
            raise ValueError(
                "fox_gd_residual_write_total_cap_final 需要同时设置 "
                "fox_gd_residual_write_total_cap."
            )
        if (
            resolved_gd_residual_write_total_cap_release_end_train_steps
            <= resolved_gd_residual_write_total_cap_release_start_train_steps
        ):
            raise ValueError(
                "fox_gd_residual_write_total_cap_release_end_train_steps 必须大于 "
                "release_start."
            )
    resolved_gd_residual_write_q_alpha = _normalize_fox_gd_residual_positive_float(
        fox_gd_residual_write_q_alpha,
        field_name="fox_gd_residual_write_q_alpha",
        default=DEFAULT_FOX_GD_RESIDUAL_WRITE_Q_ALPHA,
    )
    budgeted_write_modes = {
        "budgeted_topk_beta",
        "budgeted_topk_beta_scaled_cap",
        "budgeted_topk_beta_scaled_peak_total_cap",
    }
    if resolved_gd_residual_write_strength_mode in budgeted_write_modes:
        if resolved_gd_residual_write_budget is None:
            raise ValueError(
                "budgeted write strength mode 需要设置 "
                "fox_gd_residual_write_budget."
            )
    if resolved_gd_residual_write_strength_mode == "budgeted_topk_beta":
        if resolved_gd_residual_write_strength_cap is not None:
            raise ValueError(
                "budgeted_topk_beta 使用 fox_gd_residual_write_budget, "
                "不能同时设置 fox_gd_residual_write_strength_cap."
            )
        if resolved_gd_residual_write_total_cap is not None:
            raise ValueError(
                "budgeted_topk_beta 使用 fox_gd_residual_write_budget, "
                "不能同时设置 fox_gd_residual_write_total_cap."
            )
    elif resolved_gd_residual_write_strength_mode == "budgeted_topk_beta_scaled_cap":
        if resolved_gd_residual_write_strength_cap is None:
            raise ValueError(
                "budgeted_topk_beta_scaled_cap 需要同时设置 "
                "fox_gd_residual_write_strength_cap."
            )
        if resolved_gd_residual_write_total_cap is not None:
            raise ValueError(
                "budgeted_topk_beta_scaled_cap 不能同时设置 "
                "fox_gd_residual_write_total_cap."
            )
    elif (
        resolved_gd_residual_write_strength_mode
        == "budgeted_topk_beta_scaled_peak_total_cap"
    ):
        if resolved_gd_residual_write_strength_cap is None:
            raise ValueError(
                "budgeted_topk_beta_scaled_peak_total_cap 需要同时设置 "
                "fox_gd_residual_write_strength_cap."
            )
        if resolved_gd_residual_write_total_cap is None:
            raise ValueError(
                "budgeted_topk_beta_scaled_peak_total_cap 需要同时设置 "
                "fox_gd_residual_write_total_cap."
            )
    elif resolved_gd_residual_write_budget is not None:
        raise ValueError(
            "fox_gd_residual_write_budget 只能与 "
            "budgeted write strength mode 一起使用."
        )
    elif resolved_gd_residual_write_total_cap is not None:
        raise ValueError(
            "fox_gd_residual_write_total_cap 只能与 "
            "budgeted_topk_beta_scaled_peak_total_cap 一起使用."
        )
    resolved_gd_residual_m_norm_cap = _normalize_fox_gd_residual_optional_positive_float(
        fox_gd_residual_m_norm_cap,
        field_name="fox_gd_residual_m_norm_cap",
    )
    resolved_gd_residual_update_norm_cap = _normalize_fox_gd_residual_optional_positive_float(
        fox_gd_residual_update_norm_cap,
        field_name="fox_gd_residual_update_norm_cap",
    )
    resolved_gd_residual_norm_with_gain = bool(fox_gd_residual_norm_with_gain)
    resolved_gd_residual_use_separate_addr_codebook = bool(
        fox_gd_residual_use_separate_addr_codebook
    )
    resolved_gd_residual_addr_proj_orthogonal_init = bool(
        fox_gd_residual_addr_proj_orthogonal_init
    )
    resolved_vq_score_mode = str(vq_score_mode).lower()
    resolved_vq_weight_mode = str(vq_weight_mode).lower()
    resolved_vq_update_mode = str(vq_update_mode).lower()
    resolved_vq_softmax_tau = _normalize_vq_softmax_tau(vq_softmax_tau)
    resolved_codebook_init_rng_mode = _normalize_local_rng_mode(
        codebook_init_rng_mode,
        field_name="codebook_init_rng_mode",
    )
    resolved_codebook_init_seed = _normalize_optional_int_seed(
        codebook_init_seed,
        field_name="codebook_init_seed",
    )
    if resolved_codebook_init_rng_mode != "global" and resolved_codebook_init_seed is None:
        raise ValueError(
            "codebook_init_seed 必须在 codebook_init_rng_mode 非 global 时设置."
        )
    resolved_vq_topk = _normalize_vq_topk(vq_topk)
    resolved_gradient_accumulation_steps = _normalize_positive_int(
        gradient_accumulation_steps,
        field_name="gradient_accumulation_steps",
    ) or DEFAULT_GRADIENT_ACCUMULATION_STEPS
    resolved_validations_per_epoch = _normalize_validations_per_epoch(
        validations_per_epoch
    )
    resolved_train_batch_size = _normalize_positive_int(
        train_batch_size,
        field_name="train_batch_size",
    )
    resolved_eval_batch_size = _normalize_positive_int(
        eval_batch_size,
        field_name="eval_batch_size",
    )
    remote_read_topk_list = _normalize_fox_remote_read_topk_values(
        fox_remote_read_topk_values,
        fox_remote_read_topk=fox_remote_read_topk,
    )
    (
        resolved_remote_read_topk_initial,
        resolved_remote_read_topk_final,
        resolved_remote_read_topk_release_start_train_steps,
        resolved_remote_read_topk_release_end_train_steps,
        resolved_remote_read_topk_schedule,
        resolved_remote_read_topk_eval_policy,
    ) = _normalize_fox_remote_read_topk_schedule(
        fox_remote_read_topk_initial=fox_remote_read_topk_initial,
        fox_remote_read_topk_final=fox_remote_read_topk_final,
        fox_remote_read_topk_release_start_train_steps=(
            fox_remote_read_topk_release_start_train_steps
        ),
        fox_remote_read_topk_release_end_train_steps=(
            fox_remote_read_topk_release_end_train_steps
        ),
        fox_remote_read_topk_schedule=fox_remote_read_topk_schedule,
        fox_remote_read_topk_eval_policy=fox_remote_read_topk_eval_policy,
    )
    weighted_routing_write = (
        resolved_vq_score_mode == "codebook_dot"
        and resolved_vq_weight_mode in {"dense_softmax", "topk_softmax"}
        and resolved_vq_update_mode == "grad"
    )
    if resolved_remote_formula in ("clr_v1", "clr_delta_v1"):
        if flash_backend != "torch":
            raise ValueError(f"fox_remote_formula='{resolved_remote_formula}' 目前只支持 flash_backend='torch'.")
        if resolved_remote_path_backend != "torch":
            raise ValueError(f"fox_remote_formula='{resolved_remote_formula}' 目前只支持 fox_remote_path_backend='torch'.")
        if resolved_remote_formula == "clr_delta_v1" and any(value is not None for value in remote_read_topk_list):
            raise ValueError(f"fox_remote_formula='{resolved_remote_formula}' 暂不支持 fox_remote_read_topk.")
        if resolved_remote_read_topk_initial is not None:
            raise ValueError(
                f"fox_remote_formula='{resolved_remote_formula}' 暂不支持 fox_remote_read_topk schedule."
            )
        if resolved_clr_rank == 0 and bool(fox_clr_use_den_residual):
            raise ValueError("fox_clr_rank=0 只能与 fox_clr_use_den_residual=False 搭配使用.")
        if resolved_clr_merge_mode == "shared_den" and resolved_clr_selector_mode != "den_aware":
            raise ValueError("fox_clr_merge_mode='shared_den' 要求 fox_clr_selector_mode='den_aware'.")
        if resolved_clr_merge_mode != "residual_add" and resolved_clr_gate_mode != "off":
            raise ValueError("fox_clr_gate_mode='shared_query_linear' 只支持 fox_clr_merge_mode='residual_add'.")
        if resolved_clr_merge_mode == "shared_den" and abs(float(fox_clr_lambda_remote) - 1.0) > 1e-9:
            raise ValueError("fox_clr_lambda_remote=1.0 是 fox_clr_merge_mode='shared_den' 的固定要求.")
        if (
            resolved_clr_remat_mode == "post_phase1"
            and metric_controls["enable_layer_metrics"]
        ):
            raise ValueError(
                "fox_clr_remat_mode='post_phase1' 目前不支持 enable_layer_metrics=True."
            )
        if resolved_remote_formula == "clr_delta_v1":
            remote_read_topk_list = [None]
            if (
                resolved_clr_residual_update_mode != "additive"
                or resolved_clr_residual_forget_mode != "global"
                or resolved_clr_state_write_topk != 4
            ):
                raise ValueError(
                    "fox_clr_residual_* 开关当前只支持 fox_remote_formula='clr_v1' 的 weighted write 路径."
                )
    elif resolved_remote_formula == "gd_residual_v1":
        if flash_backend != "torch":
            raise ValueError("fox_remote_formula='gd_residual_v1' 目前只支持 flash_backend='torch'.")
        if resolved_remote_path_backend != "torch":
            raise ValueError(
                "fox_remote_formula='gd_residual_v1' 目前只支持 fox_remote_path_backend='torch'."
            )
        if resolved_gd_residual_write_topk > resolved_vq_topk and resolved_vq_weight_mode == "topk_softmax":
            raise ValueError(
                "当 gd_residual_v1 使用 vq_weight_mode='topk_softmax' 时, "
                "fox_gd_residual_write_topk 必须 <= vq_topk."
            )
        if resolved_gd_residual_use_separate_addr_codebook:
            raise ValueError(
                "fox_gd_residual_use_separate_addr_codebook 当前不支持, "
                "gd_residual_v1 reference 版本只能保持 False."
            )
        if resolved_vq_score_mode not in {"codebook_dot", "attn_dot", "mlp"}:
            raise ValueError(
                "gd_residual_v1 requires routing VQ. "
                "vq_score_mode 必须是 ['codebook_dot', 'attn_dot', 'mlp'] 之一."
            )
        if resolved_vq_weight_mode not in {"dense_softmax", "topk_softmax"}:
            raise ValueError(
                "gd_residual_v1 requires soft routing weights. "
                "vq_weight_mode 必须是 ['dense_softmax', 'topk_softmax'] 之一."
            )
        if resolved_vq_update_mode != "grad":
            raise ValueError(
                "gd_residual_v1 requires trainable routing codebook. "
                "vq_update_mode 必须是 'grad'."
            )
    elif resolved_remote_read_topk_initial is not None:
        raise ValueError("fox_remote_read_topk schedule 目前只支持 fox_remote_formula='gd_residual_v1'.")
    elif resolved_clr_remat_mode != "off":
        raise ValueError("fox_clr_remat_mode 目前只支持 fox_remote_formula='clr_v1' 或 'clr_delta_v1'.")
    if weighted_routing_write and resolved_remote_formula not in {"clr_v1", "gd_residual_v1"}:
        raise ValueError("weighted routing write 当前只支持 fox_remote_formula='clr_v1' 或 'gd_residual_v1'.")
    if weighted_routing_write and resolved_clr_remat_mode != "off":
        raise ValueError("weighted routing write 当前只支持 fox_clr_remat_mode='off'.")
    if not weighted_routing_write and (
        resolved_clr_residual_update_mode != "additive"
        or resolved_clr_residual_forget_mode != "global"
        or resolved_clr_state_write_topk != 4
    ):
        raise ValueError(
            "fox_clr_residual_* 开关当前只支持 weighted routing write 路径."
        )
    include_seed_suffix = seed_values is not None or seed is not None or len(seed_values_list) > 1
    include_read_suffix = (
        fox_remote_read_topk_values is not None
        or fox_remote_read_topk is not None
        or len(remote_read_topk_list) > 1
        or resolved_remote_read_topk_initial is not None
    )
    if normalized_num_codebook_vectors_values is not None:
        codebook_variants = [
            {
                "variant_id": f"cb{num_codes}",
                "num_codebook_vectors": num_codes,
            }
            for num_codes in normalized_num_codebook_vectors_values
        ]
    else:
        resolved_num_codebook_vectors_map = (
            dict(normalized_num_codebook_vectors_map)
            if normalized_num_codebook_vectors_map is not None
            else dict(DEFAULT_NUM_CODEBOOK_VECTORS_MAP)
        )
        missing_dmodels = [
            d_model
            for d_model in dmodels_list
            if d_model not in resolved_num_codebook_vectors_map
        ]
        if missing_dmodels:
            raise ValueError(
                "num_codebook_vectors_map 缺少这些 d_model 的配置: "
                f"{missing_dmodels}"
            )
        codebook_variants = [
            {
                "variant_id": "map",
                "num_codebook_vectors": resolved_num_codebook_vectors_map,
            }
        ]
    if resolved_remote_formula == "gd_residual_v1":
        candidate_codebook_sizes: list[int] = []
        for variant in codebook_variants:
            num_codes = variant["num_codebook_vectors"]
            if isinstance(num_codes, dict):
                candidate_codebook_sizes.extend(int(value) for value in num_codes.values())
            else:
                candidate_codebook_sizes.append(int(num_codes))
        if any(resolved_gd_residual_write_topk > num_codes for num_codes in candidate_codebook_sizes):
            raise ValueError("fox_gd_residual_write_topk 不能超过 num_codebook_vectors.")

    data_configs: dict[str, DataConfig] = {}
    input_seq_len = None
    for order in train_batch_orders_list:
        data, current_input_seq_len = _build_data_config(
            vocab_size,
            order,
            data_seed=normalized_data_seed,
            cache_dir=cache_dir,
        )
        if resolved_train_batch_size is not None or resolved_eval_batch_size is not None:
            train_bs, eval_bs = data.batch_size
            data.batch_size = (
                resolved_train_batch_size or int(train_bs),
                resolved_eval_batch_size or int(eval_bs),
            )
        data_configs[order] = data
        if input_seq_len is None:
            input_seq_len = current_input_seq_len
    assert input_seq_len is not None

    conv_mixer = _build_conv_mixer(input_seq_len)
    model_factory_kwargs = {
        "state_mixer": dict(name="torch.nn.Identity", kwargs={}),
        "vocab_size": vocab_size,
    }

    flash_models_by_structure: dict[tuple[int, int, bool, str, int | None], list] = {}
    for current_block_len, current_local_num_blocks in structure_pairs:
        for current_if_remote_enabled in if_remote_enabled_list:
            for codebook_variant in codebook_variants:
                for current_remote_read_topk in remote_read_topk_list:
                    flash_models = add_flash_vqg(
                        [],
                        conv_mixer,
                        input_seq_len,
                        model_factory_kwargs,
                        num_heads=2,
                        if_remote_enabled=current_if_remote_enabled,
                        num_codebook_vectors=codebook_variant["num_codebook_vectors"],
                        block_len=current_block_len,
                        vq_use_triton_shortcodes=(flash_backend == "accel"),
                        fox_state_build_backend="triton" if flash_backend == "accel" else "torch",
                        fox_remote_path_backend=resolved_remote_path_backend,
                        fox_remote_read_topk=current_remote_read_topk,
                        fox_remote_read_topk_initial=resolved_remote_read_topk_initial,
                        fox_remote_read_topk_final=resolved_remote_read_topk_final,
                        fox_remote_read_topk_release_start_train_steps=(
                            resolved_remote_read_topk_release_start_train_steps
                        ),
                        fox_remote_read_topk_release_end_train_steps=(
                            resolved_remote_read_topk_release_end_train_steps
                        ),
                        fox_remote_read_topk_schedule=resolved_remote_read_topk_schedule,
                        fox_remote_read_topk_eval_policy=(
                            resolved_remote_read_topk_eval_policy
                        ),
                        fox_remote_formula=resolved_remote_formula,
                        fox_clr_rank=resolved_clr_rank,
                        fox_clr_use_den_residual=bool(fox_clr_use_den_residual),
                        fox_clr_remat_mode=resolved_clr_remat_mode,
                        fox_clr_selector_mode=resolved_clr_selector_mode,
                        fox_clr_merge_mode=resolved_clr_merge_mode,
                        fox_clr_gate_mode=resolved_clr_gate_mode,
                        fox_clr_lambda_remote=float(fox_clr_lambda_remote),
                        fox_clr_gate_init_bias=float(fox_clr_gate_init_bias),
                        fox_clr_residual_update_mode=resolved_clr_residual_update_mode,
                        fox_clr_residual_forget_mode=resolved_clr_residual_forget_mode,
                        fox_clr_state_write_topk=resolved_clr_state_write_topk,
                        fox_clr_delta_target_mode=resolved_clr_delta_target_mode,
                        fox_gd_residual_rank=resolved_gd_residual_rank,
                        fox_gd_residual_write_topk=resolved_gd_residual_write_topk,
                        fox_gd_residual_builder=resolved_gd_residual_builder,
                        fox_gd_residual_pack_mode=resolved_gd_residual_pack_mode,
                        fox_gd_residual_chunk_size=resolved_gd_residual_chunk_size,
                        fox_gd_residual_mu_min_count=resolved_gd_residual_mu_min_count,
                        fox_gd_residual_addr_eps=resolved_gd_residual_addr_eps,
                        fox_gd_residual_den_eps=resolved_gd_residual_den_eps,
                        fox_gd_residual_rho_eps=resolved_gd_residual_rho_eps,
                        fox_gd_residual_addr_init_rng_mode=(
                            resolved_gd_residual_addr_init_rng_mode
                        ),
                        fox_gd_residual_addr_init_seed=(
                            resolved_gd_residual_addr_init_seed
                        ),
                        fox_gd_residual_beta_init=resolved_gd_residual_beta_init,
                        fox_gd_residual_beta_cap=resolved_gd_residual_beta_cap,
                        fox_gd_residual_beta_cap_final=(
                            resolved_gd_residual_beta_cap_final
                        ),
                        fox_gd_residual_beta_cap_release_start_train_steps=(
                            resolved_gd_residual_beta_cap_release_start_train_steps
                        ),
                        fox_gd_residual_beta_cap_release_end_train_steps=(
                            resolved_gd_residual_beta_cap_release_end_train_steps
                        ),
                        fox_gd_residual_beta_cap_eval_policy=(
                            resolved_gd_residual_beta_cap_eval_policy
                        ),
                        fox_gd_residual_beta_control_mode=(
                            resolved_gd_residual_beta_control_mode
                        ),
                        fox_gd_residual_beta_sigmoid_temp=(
                            resolved_gd_residual_beta_sigmoid_temp
                        ),
                        fox_gd_residual_beta_low=resolved_gd_residual_beta_low,
                        fox_gd_residual_beta_high=resolved_gd_residual_beta_high,
                        fox_gd_residual_beta_low_final=(
                            resolved_gd_residual_beta_low_final
                        ),
                        fox_gd_residual_beta_high_final=(
                            resolved_gd_residual_beta_high_final
                        ),
                        fox_gd_residual_beta_band_release_start_train_steps=(
                            resolved_gd_residual_beta_band_release_start_train_steps
                        ),
                        fox_gd_residual_beta_band_release_end_train_steps=(
                            resolved_gd_residual_beta_band_release_end_train_steps
                        ),
                        fox_gd_residual_beta_band_eval_policy=(
                            resolved_gd_residual_beta_band_eval_policy
                        ),
                        fox_gd_residual_beta_band_schedule=(
                            resolved_gd_residual_beta_band_schedule
                        ),
                        fox_gd_residual_lambda_init=resolved_gd_residual_lambda_init,
                        fox_gd_residual_lambda_floor=resolved_gd_residual_lambda_floor,
                        fox_gd_residual_write_strength_mode=(
                            resolved_gd_residual_write_strength_mode
                        ),
                        fox_gd_residual_write_strength_cap=(
                            resolved_gd_residual_write_strength_cap
                        ),
                        fox_gd_residual_write_strength_cap_mode=(
                            resolved_gd_residual_write_strength_cap_mode
                        ),
                        fox_gd_residual_write_strength_cap_until_train_steps=(
                            resolved_gd_residual_write_strength_cap_until_train_steps
                        ),
                        fox_gd_residual_write_strength_cap_final=(
                            resolved_gd_residual_write_strength_cap_final
                        ),
                        fox_gd_residual_write_strength_cap_release_start_train_steps=(
                            resolved_gd_residual_write_strength_cap_release_start_train_steps
                        ),
                        fox_gd_residual_write_strength_cap_release_end_train_steps=(
                            resolved_gd_residual_write_strength_cap_release_end_train_steps
                        ),
                        fox_gd_residual_write_strength_cap_eval_policy=(
                            resolved_gd_residual_write_strength_cap_eval_policy
                        ),
                        fox_gd_residual_write_budget=resolved_gd_residual_write_budget,
                        fox_gd_residual_write_budget_final=(
                            resolved_gd_residual_write_budget_final
                        ),
                        fox_gd_residual_write_budget_release_start_train_steps=(
                            resolved_gd_residual_write_budget_release_start_train_steps
                        ),
                        fox_gd_residual_write_budget_release_end_train_steps=(
                            resolved_gd_residual_write_budget_release_end_train_steps
                        ),
                        fox_gd_residual_write_budget_eval_policy=(
                            resolved_gd_residual_write_budget_eval_policy
                        ),
                        fox_gd_residual_write_budget_schedule=(
                            resolved_gd_residual_write_budget_schedule
                        ),
                        fox_gd_residual_write_total_cap=(
                            resolved_gd_residual_write_total_cap
                        ),
                        fox_gd_residual_write_total_cap_final=(
                            resolved_gd_residual_write_total_cap_final
                        ),
                        fox_gd_residual_write_total_cap_release_start_train_steps=(
                            resolved_gd_residual_write_total_cap_release_start_train_steps
                        ),
                        fox_gd_residual_write_total_cap_release_end_train_steps=(
                            resolved_gd_residual_write_total_cap_release_end_train_steps
                        ),
                        fox_gd_residual_write_total_cap_eval_policy=(
                            resolved_gd_residual_write_total_cap_eval_policy
                        ),
                        fox_gd_residual_write_total_cap_schedule=(
                            resolved_gd_residual_write_total_cap_schedule
                        ),
                        fox_gd_residual_write_q_alpha=(
                            resolved_gd_residual_write_q_alpha
                        ),
                        fox_gd_residual_m_norm_cap=resolved_gd_residual_m_norm_cap,
                        fox_gd_residual_update_norm_cap=(
                            resolved_gd_residual_update_norm_cap
                        ),
                        fox_gd_residual_norm_with_gain=resolved_gd_residual_norm_with_gain,
                        fox_gd_residual_use_separate_addr_codebook=(
                            resolved_gd_residual_use_separate_addr_codebook
                        ),
                        fox_gd_residual_addr_proj_orthogonal_init=(
                            resolved_gd_residual_addr_proj_orthogonal_init
                        ),
                        experiment_part=experiment_part,
                        experiment_mode=experiment_mode,
                        local_num_blocks=current_local_num_blocks,
                        use_time_mixing="kv_shift",
                        vq_score_mode=resolved_vq_score_mode,
                        vq_weight_mode=resolved_vq_weight_mode,
                        vq_update_mode=resolved_vq_update_mode,
                        vq_softmax_tau=resolved_vq_softmax_tau,
                        codebook_init_rng_mode=resolved_codebook_init_rng_mode,
                        codebook_init_seed=resolved_codebook_init_seed,
                        vq_topk=resolved_vq_topk,
                        if_value_silu=True,
                        if_output_gate_use_rmsnorm=True,
                        output_gate_activation="swish",
                        fox_if_local_use_vq_k=False,
                        enable_layer_metrics=metric_controls["enable_layer_metrics"],
                        fox_phase2_metrics_mode=metric_controls["fox_phase2_metrics_mode"],
                    )
                    flash_models = [m for m in flash_models if m.d_model in dmodels_list]
                    flash_models_by_structure[
                        (
                            current_block_len,
                            current_local_num_blocks,
                            current_if_remote_enabled,
                            codebook_variant["variant_id"],
                            current_remote_read_topk,
                        )
                    ] = sorted(
                        flash_models,
                        key=lambda m: m.d_model,
                    )

    gdn_models = []
    if include_gdn:
        gdn_models = add_gated_delta_net([], conv_mixer, input_seq_len, model_factory_kwargs)
        gdn_models = [m for m in gdn_models if m.d_model in dmodels_list]
        gdn_models = sorted(gdn_models, key=lambda m: m.d_model)

    configs: list[TrainConfig] = []
    logger = LoggerConfig(
        backend=logger_backend,
        project_name=wandb_project,
        entity=wandb_entity,
    )
    effective_train_batch_size = resolved_train_batch_size or DEFAULT_TRAIN_BATCH_SIZE
    effective_eval_batch_size = resolved_eval_batch_size or DEFAULT_EVAL_BATCH_SIZE
    include_batch_accum_suffix = any(
        (
            effective_train_batch_size != DEFAULT_TRAIN_BATCH_SIZE,
            effective_eval_batch_size != DEFAULT_EVAL_BATCH_SIZE,
            resolved_gradient_accumulation_steps != DEFAULT_GRADIENT_ACCUMULATION_STEPS,
            resolved_validations_per_epoch != DEFAULT_VALIDATIONS_PER_EPOCH,
        )
    )
    for order in train_batch_orders_list:
        sampler_tag = _sampler_run_tag(order)
        data = data_configs[order]
        for lr in learning_rates_list:
            for current_block_len, current_local_num_blocks in structure_pairs:
                flash_tag = _flash_run_tag(
                    flash_backend=flash_backend,
                    block_len=current_block_len,
                )
                for current_if_remote_enabled in if_remote_enabled_list:
                    structure_tag = _structure_run_tag(
                        local_num_blocks=current_local_num_blocks,
                        if_remote_enabled=current_if_remote_enabled,
                    )
                    for codebook_variant in codebook_variants:
                        for current_remote_read_topk in remote_read_topk_list:
                            read_tag = _remote_read_run_tag(current_remote_read_topk)
                            for current_seed in seed_values_list:
                                for model in flash_models_by_structure[
                                    (
                                        current_block_len,
                                        current_local_num_blocks,
                                        current_if_remote_enabled,
                                        codebook_variant["variant_id"],
                                        current_remote_read_topk,
                                    )
                                ]:
                                    num_codebook_vectors = _extract_flash_num_codebook_vectors(model)
                                    run_id = (
                                        f"{flash_tag}-dmodel{model.d_model}-cb{num_codebook_vectors}-"
                                        f"lr{lr:.1e}-{structure_tag}-sampler-{sampler_tag}"
                                    )
                                    run_id = (
                                        f"{run_id}-rformula-"
                                        f"{_remote_formula_run_tag(
                                            fox_remote_formula=resolved_remote_formula,
                                            fox_clr_rank=resolved_clr_rank,
                                            fox_clr_use_den_residual=bool(fox_clr_use_den_residual),
                                            fox_remote_read_topk_initial=(
                                                resolved_remote_read_topk_initial
                                            ),
                                            fox_remote_read_topk_final=(
                                                resolved_remote_read_topk_final
                                            ),
                                            fox_remote_read_topk_release_start_train_steps=(
                                                resolved_remote_read_topk_release_start_train_steps
                                            ),
                                            fox_remote_read_topk_release_end_train_steps=(
                                                resolved_remote_read_topk_release_end_train_steps
                                            ),
                                            fox_remote_read_topk_schedule=(
                                                resolved_remote_read_topk_schedule
                                            ),
                                            fox_remote_read_topk_eval_policy=(
                                                resolved_remote_read_topk_eval_policy
                                            ),
                                            fox_gd_residual_rank=resolved_gd_residual_rank,
                                            fox_gd_residual_write_topk=resolved_gd_residual_write_topk,
                                            fox_gd_residual_builder=resolved_gd_residual_builder,
                                            fox_gd_residual_pack_mode=resolved_gd_residual_pack_mode,
                                            fox_gd_residual_write_strength_mode=(
                                                resolved_gd_residual_write_strength_mode
                                            ),
                                            fox_gd_residual_write_strength_cap=(
                                                resolved_gd_residual_write_strength_cap
                                            ),
                                            fox_gd_residual_write_strength_cap_mode=(
                                                resolved_gd_residual_write_strength_cap_mode
                                            ),
                                            fox_gd_residual_write_strength_cap_until_train_steps=(
                                                resolved_gd_residual_write_strength_cap_until_train_steps
                                            ),
                                            fox_gd_residual_write_strength_cap_final=(
                                                resolved_gd_residual_write_strength_cap_final
                                            ),
                                            fox_gd_residual_write_strength_cap_release_start_train_steps=(
                                                resolved_gd_residual_write_strength_cap_release_start_train_steps
                                            ),
                                            fox_gd_residual_write_strength_cap_release_end_train_steps=(
                                                resolved_gd_residual_write_strength_cap_release_end_train_steps
                                            ),
                                            fox_gd_residual_write_strength_cap_eval_policy=(
                                                resolved_gd_residual_write_strength_cap_eval_policy
                                            ),
                                            fox_gd_residual_write_budget=(
                                                resolved_gd_residual_write_budget
                                            ),
                                            fox_gd_residual_write_budget_final=(
                                                resolved_gd_residual_write_budget_final
                                            ),
                                            fox_gd_residual_write_budget_release_start_train_steps=(
                                                resolved_gd_residual_write_budget_release_start_train_steps
                                            ),
                                            fox_gd_residual_write_budget_release_end_train_steps=(
                                                resolved_gd_residual_write_budget_release_end_train_steps
                                            ),
                                            fox_gd_residual_write_budget_eval_policy=(
                                                resolved_gd_residual_write_budget_eval_policy
                                            ),
                                            fox_gd_residual_write_budget_schedule=(
                                                resolved_gd_residual_write_budget_schedule
                                            ),
                                            fox_gd_residual_write_total_cap=(
                                                resolved_gd_residual_write_total_cap
                                            ),
                                            fox_gd_residual_write_total_cap_final=(
                                                resolved_gd_residual_write_total_cap_final
                                            ),
                                            fox_gd_residual_write_total_cap_release_start_train_steps=(
                                                resolved_gd_residual_write_total_cap_release_start_train_steps
                                            ),
                                            fox_gd_residual_write_total_cap_release_end_train_steps=(
                                                resolved_gd_residual_write_total_cap_release_end_train_steps
                                            ),
                                            fox_gd_residual_write_total_cap_eval_policy=(
                                                resolved_gd_residual_write_total_cap_eval_policy
                                            ),
                                            fox_gd_residual_write_total_cap_schedule=(
                                                resolved_gd_residual_write_total_cap_schedule
                                            ),
                                            fox_gd_residual_write_q_alpha=(
                                                resolved_gd_residual_write_q_alpha
                                            ),
                                            fox_gd_residual_m_norm_cap=(
                                                resolved_gd_residual_m_norm_cap
                                            ),
                                            fox_gd_residual_update_norm_cap=(
                                                resolved_gd_residual_update_norm_cap
                                            ),
                                            fox_gd_residual_beta_cap=(
                                                resolved_gd_residual_beta_cap
                                            ),
                                            fox_gd_residual_beta_cap_final=(
                                                resolved_gd_residual_beta_cap_final
                                            ),
                                            fox_gd_residual_beta_cap_release_start_train_steps=(
                                                resolved_gd_residual_beta_cap_release_start_train_steps
                                            ),
                                            fox_gd_residual_beta_cap_release_end_train_steps=(
                                                resolved_gd_residual_beta_cap_release_end_train_steps
                                            ),
                                            fox_gd_residual_beta_cap_eval_policy=(
                                                resolved_gd_residual_beta_cap_eval_policy
                                            ),
                                            fox_gd_residual_beta_control_mode=(
                                                resolved_gd_residual_beta_control_mode
                                            ),
                                            fox_gd_residual_beta_sigmoid_temp=(
                                                resolved_gd_residual_beta_sigmoid_temp
                                            ),
                                            fox_gd_residual_beta_low=(
                                                resolved_gd_residual_beta_low
                                            ),
                                            fox_gd_residual_beta_high=(
                                                resolved_gd_residual_beta_high
                                            ),
                                            fox_gd_residual_beta_low_final=(
                                                resolved_gd_residual_beta_low_final
                                            ),
                                            fox_gd_residual_beta_high_final=(
                                                resolved_gd_residual_beta_high_final
                                            ),
                                            fox_gd_residual_beta_band_release_start_train_steps=(
                                                resolved_gd_residual_beta_band_release_start_train_steps
                                            ),
                                            fox_gd_residual_beta_band_release_end_train_steps=(
                                                resolved_gd_residual_beta_band_release_end_train_steps
                                            ),
                                            fox_gd_residual_beta_band_eval_policy=(
                                                resolved_gd_residual_beta_band_eval_policy
                                            ),
                                            fox_gd_residual_beta_band_schedule=(
                                                resolved_gd_residual_beta_band_schedule
                                            ),
                                            fox_gd_residual_lambda_floor=(
                                                resolved_gd_residual_lambda_floor
                                            ),
                                        )}"
                                    )
                                    if resolved_codebook_init_rng_mode != "global":
                                        run_id = (
                                            f"{run_id}-cbinit-"
                                            f"{resolved_codebook_init_rng_mode}-"
                                            f"s{resolved_codebook_init_seed}"
                                        )
                                    if resolved_gd_residual_addr_init_rng_mode != "global":
                                        run_id = (
                                            f"{run_id}-addrinit-"
                                            f"{resolved_gd_residual_addr_init_rng_mode}-"
                                            f"s{resolved_gd_residual_addr_init_seed}"
                                        )
                                    if resolved_remote_formula in ("clr_v1", "clr_delta_v1"):
                                        run_id = (
                                            f"{run_id}-rremat-"
                                            f"{_clr_remat_run_tag(resolved_clr_remat_mode)}"
                                        )
                                    if weighted_routing_write:
                                        run_id = (
                                            f"{run_id}-"
                                            f"{_clr_residual_write_run_tag(
                                                fox_clr_residual_update_mode=resolved_clr_residual_update_mode,
                                                fox_clr_residual_forget_mode=resolved_clr_residual_forget_mode,
                                                fox_clr_state_write_topk=resolved_clr_state_write_topk,
                                            )}"
                                        )
                                    if include_read_suffix:
                                        run_id = f"{run_id}-rread-{read_tag}"
                                    if include_seed_suffix:
                                        run_id = f"{run_id}-seed{current_seed}"
                                    if include_batch_accum_suffix:
                                        train_bs, eval_bs = data.batch_size
                                        run_id = (
                                            f"{run_id}-tbs{int(train_bs)}-ebs{int(eval_bs)}-"
                                            f"ga{resolved_gradient_accumulation_steps}"
                                        )
                                        if resolved_validations_per_epoch != DEFAULT_VALIDATIONS_PER_EPOCH:
                                            run_id = f"{run_id}-vpe{resolved_validations_per_epoch}"
                                    # Deep-copy model per seed and inject codebook_init_seed
                                    model_cur = model.model_copy(deep=True)
                                    for sub_cfg in model_cur.sequence_mixer.kwargs.get("configs", []):
                                        if hasattr(sub_cfg, "kwargs"):
                                            sub_cfg.kwargs["codebook_init_seed"] = current_seed
                                    configs.append(
                                        TrainConfig(
                                            model=model_cur,
                                            data=data,
                                            learning_rate=lr,
                                            max_epochs=max_epochs,
                                            gradient_accumulation_steps=resolved_gradient_accumulation_steps,
                                            validations_per_epoch=resolved_validations_per_epoch,
                                            early_stopping_metric=early_stopping_metric,
                                            early_stopping_threshold=early_stopping_threshold,
                                            logger=logger,
                                            metrics_white_list=normalized_metrics_white_list,
                                            read_churn_probe_enabled=read_churn_probe_enabled,
                                            read_churn_probe_valid_batches=read_churn_probe_valid_batches,
                                            read_churn_probe_max_samples=read_churn_probe_max_samples,
                                            read_churn_probe_query_only=read_churn_probe_query_only,
                                            slice_keys=["num_kv_pairs", "input_seq_len", "mqar_case"],
                                            sweep_id=sweep_id,
                                            seed=current_seed,
                                            run_id=run_id,
                                        )
                                    )
            for current_seed in seed_values_list:
                for model in gdn_models:
                    run_id = (
                        f"gated_delta_net-dmodel{model.d_model}-lr{lr:.1e}-sampler-{sampler_tag}"
                    )
                    if include_seed_suffix:
                        run_id = f"{run_id}-seed{current_seed}"
                    if include_batch_accum_suffix:
                        train_bs, eval_bs = data.batch_size
                        run_id = (
                            f"{run_id}-tbs{int(train_bs)}-ebs{int(eval_bs)}-"
                            f"ga{resolved_gradient_accumulation_steps}"
                        )
                        if resolved_validations_per_epoch != DEFAULT_VALIDATIONS_PER_EPOCH:
                            run_id = f"{run_id}-vpe{resolved_validations_per_epoch}"
                    configs.append(
                        TrainConfig(
                            model=model,
                            data=data,
                            learning_rate=lr,
                            max_epochs=max_epochs,
                            gradient_accumulation_steps=resolved_gradient_accumulation_steps,
                            validations_per_epoch=resolved_validations_per_epoch,
                            early_stopping_metric=early_stopping_metric,
                            early_stopping_threshold=early_stopping_threshold,
                            logger=logger,
                            metrics_white_list=normalized_metrics_white_list,
                            read_churn_probe_enabled=read_churn_probe_enabled,
                            read_churn_probe_valid_batches=read_churn_probe_valid_batches,
                            read_churn_probe_max_samples=read_churn_probe_max_samples,
                            read_churn_probe_query_only=read_churn_probe_query_only,
                            slice_keys=["num_kv_pairs", "input_seq_len", "mqar_case"],
                            sweep_id=sweep_id,
                            seed=current_seed,
                            run_id=run_id,
                        )
                    )
    return configs
