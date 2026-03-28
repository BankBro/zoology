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


def _sampler_run_tag(train_batch_order: str) -> str:
    return {
        "sequential": "seq",
        "global_shuffle": "gshuffle",
        "balanced_interleave": "binterleave",
    }[train_batch_order]


def _structure_run_tag(*, local_num_blocks: int, if_remote_enabled: bool) -> str:
    return f"local{int(local_num_blocks)}-remote{int(bool(if_remote_enabled))}"


def _build_data_config(
    vocab_size: int,
    train_batch_order: str,
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
    batch_size = 256
    data = DataConfig(
        train_configs=train_configs,
        test_configs=test_configs,
        batch_size=(batch_size, batch_size // 8),
        train_batch_order=train_batch_order,
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
    num_codebook_vectors_values: Iterable[int] | None = None,
    num_codebook_vectors_map: Mapping[int, int] | None = None,
    cache_dir: str = DEFAULT_CACHE_DIR,
    metrics_white_list: Iterable[str] | None = None,
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
    train_batch_orders_list = _normalize_train_batch_orders(train_batch_orders, train_batch_order)
    normalized_metrics_white_list = normalize_metrics_white_list(metrics_white_list)
    metric_controls = derive_flash_metric_controls(normalized_metrics_white_list)
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

    data_configs: dict[str, DataConfig] = {}
    input_seq_len = None
    for order in train_batch_orders_list:
        data, current_input_seq_len = _build_data_config(vocab_size, order, cache_dir)
        data_configs[order] = data
        if input_seq_len is None:
            input_seq_len = current_input_seq_len
    assert input_seq_len is not None

    conv_mixer = _build_conv_mixer(input_seq_len)
    model_factory_kwargs = {
        "state_mixer": dict(name="torch.nn.Identity", kwargs={}),
        "vocab_size": vocab_size,
    }

    flash_models_by_structure: dict[tuple[int, int, bool, str], list] = {}
    for current_block_len, current_local_num_blocks in structure_pairs:
        for current_if_remote_enabled in if_remote_enabled_list:
            for codebook_variant in codebook_variants:
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
                    fox_remote_path_backend="triton" if flash_backend == "accel" else "torch",
                    local_num_blocks=current_local_num_blocks,
                    use_time_mixing="kv_shift",
                    vq_score_mode="l2",
                    vq_weight_mode="one-hot",
                    vq_update_mode="ema",
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
                        for model in flash_models_by_structure[
                            (
                                current_block_len,
                                current_local_num_blocks,
                                current_if_remote_enabled,
                                codebook_variant["variant_id"],
                            )
                        ]:
                            num_codebook_vectors = _extract_flash_num_codebook_vectors(model)
                            configs.append(
                                TrainConfig(
                                    model=model,
                                    data=data,
                                    learning_rate=lr,
                                    max_epochs=max_epochs,
                                    logger=logger,
                                    metrics_white_list=normalized_metrics_white_list,
                                    slice_keys=["num_kv_pairs", "input_seq_len", "mqar_case"],
                                    sweep_id=sweep_id,
                                    run_id=(
                                        f"{flash_tag}-dmodel{model.d_model}-cb{num_codebook_vectors}-"
                                        f"lr{lr:.1e}-{structure_tag}-sampler-{sampler_tag}"
                                    ),
                                )
                            )
            for model in gdn_models:
                configs.append(
                    TrainConfig(
                        model=model,
                        data=data,
                        learning_rate=lr,
                        max_epochs=max_epochs,
                        logger=logger,
                        metrics_white_list=normalized_metrics_white_list,
                        slice_keys=["num_kv_pairs", "input_seq_len", "mqar_case"],
                        sweep_id=sweep_id,
                        run_id=f"gated_delta_net-dmodel{model.d_model}-lr{lr:.1e}-sampler-{sampler_tag}",
                    )
                )
    return configs
