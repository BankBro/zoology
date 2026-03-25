import uuid
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


def _build_data_config(vocab_size: int) -> tuple[DataConfig, int]:
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
        cache_dir="/data/sim/zoology",
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


def _build_sweep_name(
    *,
    flash_backend: str,
    include_gdn: bool,
    block_len: int,
    dmodels: list[int],
) -> str:
    scope = "flash-vs-gdn" if include_gdn else "flash-only"
    dmodel_tag = "d" + "-".join(str(v) for v in dmodels)
    block_tag = f"block{int(block_len)}"
    suffix = uuid.uuid4().hex[:6]
    return f"flash-vqg-suite-{scope}-{flash_backend}-{block_tag}-{dmodel_tag}-{suffix}"


def _flash_run_tag(*, flash_backend: str, block_len: int) -> str:
    backend_tag = "accel" if flash_backend == "accel" else "torch"
    block_tag = "-block32" if int(block_len) == 32 else ""
    return f"flash_vqg_h2_{backend_tag}{block_tag}"


def build_configs(
    *,
    flash_backend: str = "accel",
    include_gdn: bool = True,
    block_len: int = 8,
    dmodels: Iterable[int] | None = None,
    learning_rates: Iterable[float] | None = None,
    wandb_project: str = DEFAULT_WANDB_PROJECT,
    wandb_entity: str = DEFAULT_WANDB_ENTITY,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
) -> list[TrainConfig]:
    flash_backend = str(flash_backend).lower()
    if flash_backend not in {"accel", "torch"}:
        raise ValueError(f"flash_backend 只能是 'accel' 或 'torch', 当前收到: {flash_backend}")

    dmodels_list = _normalize_dmodels(dmodels)
    learning_rates_list = _normalize_learning_rates(learning_rates)

    data, input_seq_len = _build_data_config(vocab_size)
    conv_mixer = _build_conv_mixer(input_seq_len)
    model_factory_kwargs = {
        "state_mixer": dict(name="torch.nn.Identity", kwargs={}),
        "vocab_size": vocab_size,
    }

    flash_models = add_flash_vqg(
        [],
        conv_mixer,
        input_seq_len,
        model_factory_kwargs,
        num_heads=2,
        if_remote_enabled=True,
        num_codebook_vectors={64: 64, 128: 128, 256: 256},
        block_len=int(block_len),
        vq_use_triton_shortcodes=(flash_backend == "accel"),
        fox_state_build_backend="triton" if flash_backend == "accel" else "torch",
        fox_remote_path_backend="triton" if flash_backend == "accel" else "torch",
        local_num_blocks=2,
        use_time_mixing="kv_shift",
        vq_score_mode="l2",
        vq_weight_mode="one-hot",
        vq_update_mode="ema",
        if_value_silu=True,
        if_output_gate_use_rmsnorm=True,
        output_gate_activation="swish",
        fox_if_local_use_vq_k=False,
    )
    flash_models = [m for m in flash_models if m.d_model in dmodels_list]
    flash_models = sorted(flash_models, key=lambda m: m.d_model)

    gdn_models = []
    if include_gdn:
        gdn_models = add_gated_delta_net([], conv_mixer, input_seq_len, model_factory_kwargs)
        gdn_models = [m for m in gdn_models if m.d_model in dmodels_list]
        gdn_models = sorted(gdn_models, key=lambda m: m.d_model)

    sweep_name = _build_sweep_name(
        flash_backend=flash_backend,
        include_gdn=bool(include_gdn),
        block_len=int(block_len),
        dmodels=dmodels_list,
    )
    flash_tag = _flash_run_tag(flash_backend=flash_backend, block_len=int(block_len))

    configs: list[TrainConfig] = []
    logger = LoggerConfig(project_name=wandb_project, entity=wandb_entity)
    for lr in learning_rates_list:
        for model in flash_models:
            configs.append(
                TrainConfig(
                    model=model,
                    data=data,
                    learning_rate=lr,
                    max_epochs=max_epochs,
                    logger=logger,
                    slice_keys=["num_kv_pairs", "input_seq_len", "mqar_case"],
                    sweep_id=sweep_name,
                    run_id=f"{flash_tag}-dmodel{model.d_model}-lr{lr:.1e}",
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
                    sweep_id=sweep_name,
                    run_id=f"gated_delta_net-dmodel{model.d_model}-lr{lr:.1e}",
                )
            )
    return configs
