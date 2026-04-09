from __future__ import annotations

from zoology.experiments.flash_vqg.flash_vqg_suite import build_configs
from zoology.experiments.flash_vqg.run_flash_vqg_suite import (
    _parse_csv_floats,
    _parse_csv_ints,
    _parse_seed_values,
    _resolve_metrics_white_list,
)


def _common_builder_kwargs(
    args,
    *,
    experiment_mode: str,
    vq_score_mode: str,
    vq_weight_mode: str,
    vq_update_mode: str,
    vq_softmax_tau: float,
):
    dmodels = _parse_csv_ints(args.dmodels)
    learning_rates = _parse_csv_floats(args.learning_rates)
    seed_values = _parse_seed_values(args.seed_values) if args.seed_values is not None else [123]
    num_codebook_vectors_values = (
        _parse_csv_ints(args.num_codebook_vectors)
        if args.num_codebook_vectors is not None
        else [128]
    )
    metrics_white_list = _resolve_metrics_white_list(
        metrics_white_list_raw=args.metrics_white_list,
        metrics_white_list_file=args.metrics_white_list_file,
    )
    return dict(
        sweep_id=args.launch_id_prefix,
        flash_backend=args.backend,
        logger_backend=args.logger_backend,
        include_gdn=False,
        block_len=32,
        local_num_blocks=2,
        dmodels=dmodels,
        learning_rates=learning_rates,
        if_remote_enabled=True,
        train_batch_order=args.train_batch_order,
        seed_values=seed_values,
        data_seed=args.data_seed,
        num_codebook_vectors_values=num_codebook_vectors_values,
        fox_remote_path_backend=args.fox_remote_path_backend or "torch",
        fox_remote_read_topk=2,
        fox_remote_formula="clr_v1",
        fox_clr_rank=args.fox_clr_rank,
        fox_clr_use_den_residual=(args.fox_clr_use_den_residual == "true"),
        fox_clr_remat_mode=args.fox_clr_remat_mode,
        fox_clr_selector_mode="den_aware",
        fox_clr_merge_mode="shared_den",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        cache_dir=args.cache_dir,
        wandb_project=args.project,
        wandb_entity=args.entity,
        max_epochs=args.max_epochs,
        metrics_white_list=metrics_white_list,
        experiment_part="e3_dense",
        experiment_mode=experiment_mode,
        vq_score_mode=vq_score_mode,
        vq_weight_mode=vq_weight_mode,
        vq_update_mode=vq_update_mode,
        vq_softmax_tau=float(vq_softmax_tau),
        vq_topk=int(args.vq_topk),
    )


def _rewrite_run_id(config, *, run_id: str):
    config = config.model_copy(deep=True) if hasattr(config, "model_copy") else config.copy(deep=True)
    config.run_id = run_id
    return config


def _build_single(
    args,
    *,
    experiment_mode: str,
    vq_score_mode: str,
    vq_weight_mode: str,
    vq_update_mode: str,
    vq_softmax_tau: float,
    run_id: str,
):
    kwargs = _common_builder_kwargs(
        args,
        experiment_mode=experiment_mode,
        vq_score_mode=vq_score_mode,
        vq_weight_mode=vq_weight_mode,
        vq_update_mode=vq_update_mode,
        vq_softmax_tau=vq_softmax_tau,
    )
    configs = build_configs(**kwargs)
    if len(configs) != 1:
        raise RuntimeError(f"Expected exactly 1 config for {experiment_mode}, got {len(configs)}")
    return _rewrite_run_id(configs[0], run_id=run_id)


def build_e3_train_configs(args):
    configs = [
        _build_single(
            args,
            experiment_mode="baseline",
            vq_score_mode="l2",
            vq_weight_mode="one-hot",
            vq_update_mode="ema",
            vq_softmax_tau=1.0,
            run_id="baseline",
        )
    ]
    for value, tag in ((0.5, "050"), (1.0, "100"), (2.0, "200")):
        configs.append(
            _build_single(
                args,
                experiment_mode=f"dense_t{tag}",
                vq_score_mode="codebook_dot",
                vq_weight_mode="dense_softmax",
                vq_update_mode="grad",
                vq_softmax_tau=value,
                run_id=f"dense-t{tag}",
            )
        )
    return configs


def build_e3_smoke_configs(args):
    return [
        _build_single(
            args,
            experiment_mode="baseline",
            vq_score_mode="l2",
            vq_weight_mode="one-hot",
            vq_update_mode="ema",
            vq_softmax_tau=1.0,
            run_id="baseline",
        ),
        _build_single(
            args,
            experiment_mode="dense_t100",
            vq_score_mode="codebook_dot",
            vq_weight_mode="dense_softmax",
            vq_update_mode="grad",
            vq_softmax_tau=1.0,
            run_id="dense-t100",
        ),
    ]
