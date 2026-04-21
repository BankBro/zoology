from __future__ import annotations

from zoology.experiments.flash_vqg.flash_vqg_suite import build_configs
from zoology.experiments.flash_vqg.run_flash_vqg_suite import (
    _parse_csv_floats,
    _parse_csv_ints,
    _parse_seed_values,
    _resolve_metrics_white_list,
)


MODE_SPECS = (
    ("additive", "additive", "global"),
    ("gated-only", "additive", "code_aware"),
    ("delta-only", "delta", "global"),
    ("gated-delta", "delta", "code_aware"),
)


def _require_single(values: list, *, field_name: str):
    if len(values) != 1:
        raise ValueError(f"{field_name} 当前 builder 只支持单值, 当前收到: {values}")
    return values[0]


def _rewrite_run_id(config, *, run_id: str):
    config = config.model_copy(deep=True) if hasattr(config, "model_copy") else config.copy(deep=True)
    config.run_id = run_id
    return config


def _common_builder_kwargs(args, *, update_mode: str, forget_mode: str):
    dmodels = _parse_csv_ints(args.dmodels)
    learning_rates = _parse_csv_floats(args.learning_rates)
    seed_values = _parse_seed_values(args.seed_values) if args.seed_values is not None else [123]
    num_codebook_vectors_values = (
        _parse_csv_ints(args.num_codebook_vectors)
        if args.num_codebook_vectors is not None
        else [128]
    )
    d_model = _require_single(dmodels, field_name="dmodels")
    learning_rate = _require_single(learning_rates, field_name="learning_rates")
    seed_value = _require_single(seed_values, field_name="seed_values")
    num_codebook_vectors = _require_single(
        num_codebook_vectors_values,
        field_name="num_codebook_vectors",
    )
    metrics_white_list = _resolve_metrics_white_list(
        metrics_white_list_raw=args.metrics_white_list,
        metrics_white_list_file=args.metrics_white_list_file,
    )
    return (
        dict(
            sweep_id=args.launch_id_prefix,
            flash_backend=args.backend,
            logger_backend=args.logger_backend,
            include_gdn=False,
            block_len=32,
            local_num_blocks=2,
            dmodels=[d_model],
            learning_rates=[learning_rate],
            if_remote_enabled=True,
            train_batch_order=args.train_batch_order,
            seed_values=[seed_value],
            data_seed=args.data_seed,
            num_codebook_vectors_values=[num_codebook_vectors],
            fox_remote_path_backend=args.fox_remote_path_backend or "torch",
            fox_remote_read_topk=2,
            fox_remote_formula="clr_v1",
            fox_clr_rank=args.fox_clr_rank,
            fox_clr_use_den_residual=(args.fox_clr_use_den_residual == "true"),
            fox_clr_remat_mode=args.fox_clr_remat_mode,
            fox_clr_selector_mode="den_aware",
            fox_clr_merge_mode="shared_den",
            fox_clr_gate_mode="off",
            fox_clr_lambda_remote=1.0,
            fox_clr_residual_update_mode=update_mode,
            fox_clr_residual_forget_mode=forget_mode,
            fox_clr_state_write_topk=int(args.fox_clr_state_write_topk),
            fox_clr_delta_target_mode=str(args.fox_clr_delta_target_mode),
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            cache_dir=args.cache_dir,
            wandb_project=args.project,
            wandb_entity=args.entity,
            max_epochs=args.max_epochs,
            metrics_white_list=metrics_white_list,
            experiment_part="gated_delta_write",
            experiment_mode=update_mode if forget_mode == "global" else f"{update_mode}_{forget_mode}",
            vq_score_mode="codebook_dot",
            vq_weight_mode="dense_softmax",
            vq_update_mode="grad",
            vq_softmax_tau=0.25,
            vq_topk=int(args.vq_topk),
        ),
        seed_value,
        int(args.data_seed),
    )


def _build_single(args, *, mode_name: str, update_mode: str, forget_mode: str):
    kwargs, seed_value, data_seed = _common_builder_kwargs(
        args,
        update_mode=update_mode,
        forget_mode=forget_mode,
    )
    configs = build_configs(**kwargs)
    if len(configs) != 1:
        raise RuntimeError(f"Expected exactly 1 config for {mode_name}, got {len(configs)}")
    run_id = f"dense-t025-{mode_name}-s{seed_value}-d{data_seed}"
    return _rewrite_run_id(configs[0], run_id=run_id)


def build_write_train_configs(args):
    return [
        _build_single(
            args,
            mode_name=mode_name,
            update_mode=update_mode,
            forget_mode=forget_mode,
        )
        for mode_name, update_mode, forget_mode in MODE_SPECS
    ]


def build_write_tail_configs(args):
    tail_modes = ("delta-only", "gated-delta")
    selected = [spec for spec in MODE_SPECS if spec[0] in tail_modes]
    return [
        _build_single(
            args,
            mode_name=mode_name,
            update_mode=update_mode,
            forget_mode=forget_mode,
        )
        for mode_name, update_mode, forget_mode in selected
    ]


def build_write_smoke_configs(args):
    smoke_modes = ("additive", "gated-delta")
    selected = [spec for spec in MODE_SPECS if spec[0] in smoke_modes]
    return [
        _build_single(
            args,
            mode_name=mode_name,
            update_mode=update_mode,
            forget_mode=forget_mode,
        )
        for mode_name, update_mode, forget_mode in selected
    ]
