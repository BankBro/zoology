from __future__ import annotations

import os

from zoology.experiments.flash_vqg.flash_vqg_suite import build_configs
from zoology.experiments.flash_vqg.run_flash_vqg_suite import (
    _parse_csv_floats,
    _parse_csv_ints,
    _parse_seed_values,
    _resolve_metrics_white_list,
)


def _require_single(values: list, *, field_name: str):
    if len(values) != 1:
        raise ValueError(f"{field_name} 当前 builder 只支持单值, 当前收到: {values}")
    return values[0]


def _copy_config(config):
    return config.model_copy(deep=True) if hasattr(config, "model_copy") else config.copy(deep=True)


def _rewrite_run_id(config, *, run_id: str):
    config = _copy_config(config)
    config.run_id = run_id
    return config


def _parse_bool_arg(value, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    raise ValueError(f"{field_name} 必须是 bool 或 true/false, 当前收到: {value!r}")


def _parse_positive_int(value, *, field_name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{field_name} 必须是正整数, 当前收到: {value!r}")
    return parsed


def _parse_positive_float(value, *, field_name: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise ValueError(f"{field_name} 必须是正数, 当前收到: {value!r}")
    return parsed


def _normalize_expand_v(value: float) -> int | float:
    return int(value) if float(value).is_integer() else value


def _format_expand_v(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value).replace(".", "p")


def _apply_gdn_hparams(
    config,
    *,
    num_heads: int,
    expand_v: float,
    use_gate: bool,
    use_short_conv: bool,
    conv_size: int,
):
    config = _copy_config(config)
    sequence_mixer = config.model.sequence_mixer
    mixer_configs = sequence_mixer.kwargs.get("configs") if sequence_mixer is not None else None
    if not isinstance(mixer_configs, list):
        raise ValueError("GDN config 缺少 sequence_mixer.kwargs.configs.")

    found = False
    for mixer in mixer_configs:
        mixer_name = mixer.get("name") if isinstance(mixer, dict) else getattr(mixer, "name", None)
        if mixer_name != "zoology.mixers.gated_delta_net.GatedDeltaNet":
            continue
        mixer_kwargs = mixer.setdefault("kwargs", {}) if isinstance(mixer, dict) else mixer.kwargs
        mixer_kwargs.update(
            {
                "num_heads": num_heads,
                "expand_v": expand_v,
                "use_gate": use_gate,
                "use_short_conv": use_short_conv,
                "conv_size": conv_size,
            }
        )
        found = True
    if not found:
        raise ValueError("未找到 zoology.mixers.gated_delta_net.GatedDeltaNet mixer.")
    return config


def build_gdn_capacity_up_config(args):
    dmodels = _parse_csv_ints(args.dmodels)
    learning_rates = _parse_csv_floats(args.learning_rates)
    seed_values = _parse_seed_values(args.seed_values) if args.seed_values is not None else [123]

    d_model = _require_single(dmodels, field_name="dmodels")
    learning_rate = _require_single(learning_rates, field_name="learning_rates")
    seed_value = _require_single(seed_values, field_name="seed_values")
    data_seed = int(args.data_seed)
    metrics_white_list = _resolve_metrics_white_list(
        metrics_white_list_raw=args.metrics_white_list,
        metrics_white_list_file=args.metrics_white_list_file,
    )
    disable_early_stopping = _parse_bool_arg(
        getattr(args, "disable_early_stopping", "false"),
        field_name="disable_early_stopping",
    )
    num_heads = _parse_positive_int(os.environ.get("GDN_NUM_HEADS", 2), field_name="GDN_NUM_HEADS")
    expand_v = _normalize_expand_v(
        _parse_positive_float(os.environ.get("GDN_EXPAND_V", 2), field_name="GDN_EXPAND_V")
    )
    use_gate = _parse_bool_arg(os.environ.get("GDN_USE_GATE", "false"), field_name="GDN_USE_GATE")
    use_short_conv = _parse_bool_arg(
        os.environ.get("GDN_USE_SHORT_CONV", "true"),
        field_name="GDN_USE_SHORT_CONV",
    )
    conv_size = _parse_positive_int(os.environ.get("GDN_CONV_SIZE", 4), field_name="GDN_CONV_SIZE")

    configs = build_configs(
        sweep_id=args.launch_id_prefix,
        flash_backend=args.backend,
        logger_backend=args.logger_backend,
        include_gdn=True,
        dmodels=[d_model],
        learning_rates=[learning_rate],
        if_remote_enabled=True,
        local_num_blocks=2,
        train_batch_order=args.train_batch_order,
        seed_values=[seed_value],
        data_seed=data_seed,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        cache_dir=args.cache_dir,
        wandb_project=args.project,
        wandb_entity=args.entity,
        max_epochs=args.max_epochs,
        validations_per_epoch=int(getattr(args, "validations_per_epoch", 1)),
        early_stopping_metric=None if disable_early_stopping else "valid/accuracy",
        early_stopping_threshold=None if disable_early_stopping else 0.99,
        metrics_white_list=metrics_white_list,
    )

    gdn_configs = [config for config in configs if getattr(config.model, "name", None) == "gated_delta_net"]
    if len(gdn_configs) != 1:
        raise RuntimeError(f"Expected exactly 1 gated_delta_net config, got {len(gdn_configs)}")

    config = _apply_gdn_hparams(
        gdn_configs[0],
        num_heads=num_heads,
        expand_v=expand_v,
        use_gate=use_gate,
        use_short_conv=use_short_conv,
        conv_size=conv_size,
    )
    default_run_id = (
        f"gdn-usegate{int(use_gate)}-h{num_heads}-ev{_format_expand_v(expand_v)}"
        f"-s{seed_value}-d{data_seed}"
    )
    run_id = str(getattr(args, "run_id", None) or default_run_id)
    return [_rewrite_run_id(config, run_id=run_id)]
