from typing import Iterable

from zoology.config import DataConfig, LoggerConfig, TrainConfig
from zoology.data.multiquery_ar import MQARConfig
from zoology.experiments.models_repo import add_flash_vqg, add_gated_delta_net


DEFAULT_VOCAB_SIZE = 8_192
DEFAULT_DMODELS = [128]
DEFAULT_LEARNING_RATES = [1e-4, 3e-4, 1e-3, 3e-3]
DEFAULT_WANDB_PROJECT = "flash_vqg_vs_gdn"
DEFAULT_WANDB_ENTITY = "scu-mclab"
DEFAULT_MAX_EPOCHS = 32
DEFAULT_TRAIN_BATCH_ORDER = "sequential"
DEFAULT_CACHE_DIR = "./data/flash_vqg"
DEFAULT_IF_REMOTE_ENABLED = [True]
DEFAULT_LOCAL_NUM_BLOCKS = [2]


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
    block_tag = "-block32" if int(block_len) == 32 else ""
    return f"flash_vqg_h2_{backend_tag}{block_tag}"


def build_configs(
    *,
    sweep_id: str = "flash-vqg-suite",
    flash_backend: str = "accel",
    logger_backend: str = "wandb",
    include_gdn: bool = True,
    block_len: int = 8,
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
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> list[TrainConfig]:
    flash_backend = str(flash_backend).lower()
    if flash_backend not in {"accel", "torch"}:
        raise ValueError(f"flash_backend 只能是 'accel' 或 'torch', 当前收到: {flash_backend}")
    logger_backend = str(logger_backend).lower()
    if logger_backend not in {"wandb", "swanlab", "none"}:
        raise ValueError(
            f"logger_backend 只能是 ['none', 'swanlab', 'wandb'], 当前收到: {logger_backend}"
        )

    dmodels_list = _normalize_dmodels(dmodels)
    learning_rates_list = _normalize_learning_rates(learning_rates)
    train_batch_orders_list = _normalize_train_batch_orders(train_batch_orders, train_batch_order)
    if_remote_enabled_list = _normalize_if_remote_enabled_values(
        if_remote_enabled_values=if_remote_enabled_values,
        if_remote_enabled=if_remote_enabled,
    )
    local_num_blocks_list = _normalize_local_num_blocks_values(
        local_num_blocks_values=local_num_blocks_values,
        local_num_blocks=local_num_blocks,
    )

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

    flash_models_by_structure: dict[tuple[int, bool], list] = {}
    for current_local_num_blocks in local_num_blocks_list:
        for current_if_remote_enabled in if_remote_enabled_list:
            flash_models = add_flash_vqg(
                [],
                conv_mixer,
                input_seq_len,
                model_factory_kwargs,
                num_heads=2,
                if_remote_enabled=current_if_remote_enabled,
                num_codebook_vectors={64: 64, 128: 128, 256: 256},
                block_len=int(block_len),
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
                enable_layer_metrics=True,
            )
            flash_models = [m for m in flash_models if m.d_model in dmodels_list]
            flash_models_by_structure[(current_local_num_blocks, current_if_remote_enabled)] = sorted(
                flash_models,
                key=lambda m: m.d_model,
            )

    gdn_models = []
    if include_gdn:
        gdn_models = add_gated_delta_net([], conv_mixer, input_seq_len, model_factory_kwargs)
        gdn_models = [m for m in gdn_models if m.d_model in dmodels_list]
        gdn_models = sorted(gdn_models, key=lambda m: m.d_model)

    flash_tag = _flash_run_tag(flash_backend=flash_backend, block_len=int(block_len))
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
            for current_local_num_blocks in local_num_blocks_list:
                for current_if_remote_enabled in if_remote_enabled_list:
                    structure_tag = _structure_run_tag(
                        local_num_blocks=current_local_num_blocks,
                        if_remote_enabled=current_if_remote_enabled,
                    )
                    for model in flash_models_by_structure[(current_local_num_blocks, current_if_remote_enabled)]:
                        configs.append(
                            TrainConfig(
                                model=model,
                                data=data,
                                learning_rate=lr,
                                max_epochs=max_epochs,
                                logger=logger,
                                slice_keys=["num_kv_pairs", "input_seq_len", "mqar_case"],
                                sweep_id=sweep_id,
                                run_id=(
                                    f"{flash_tag}-dmodel{model.d_model}-lr{lr:.1e}-"
                                    f"{structure_tag}-sampler-{sampler_tag}"
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
                        slice_keys=["num_kv_pairs", "input_seq_len", "mqar_case"],
                        sweep_id=sweep_id,
                        run_id=f"gated_delta_net-dmodel{model.d_model}-lr{lr:.1e}-sampler-{sampler_tag}",
                    )
                )
    return configs
