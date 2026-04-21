from __future__ import annotations

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


def _rewrite_run_id(config, *, run_id: str):
    config = config.model_copy(deep=True) if hasattr(config, "model_copy") else config.copy(deep=True)
    config.run_id = run_id
    return config


def build_gdn_default_config(args):
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
        metrics_white_list=metrics_white_list,
    )

    gdn_configs = [config for config in configs if getattr(config.model, "name", None) == "gated_delta_net"]
    if len(gdn_configs) != 1:
        raise RuntimeError(f"Expected exactly 1 gated_delta_net config, got {len(gdn_configs)}")

    run_id = f"gated_delta_net-default-s{seed_value}-d{data_seed}"
    return [_rewrite_run_id(gdn_configs[0], run_id=run_id)]
