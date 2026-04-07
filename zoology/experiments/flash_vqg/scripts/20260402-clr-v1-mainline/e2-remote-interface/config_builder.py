from __future__ import annotations

from zoology.experiments.flash_vqg.flash_vqg_suite import build_configs
from zoology.experiments.flash_vqg.run_flash_vqg_suite import (
    _parse_csv_floats,
    _parse_csv_ints,
    _parse_seed_values,
    _resolve_metrics_white_list,
)


def _common_builder_kwargs(args, *, experiment_part: str, experiment_mode: str, selector_mode: str, merge_mode: str, gate_mode: str, lambda_remote: float, gate_init_bias: float):
    dmodels = _parse_csv_ints(args.dmodels)
    learning_rates = _parse_csv_floats(args.learning_rates)
    seed_values = _parse_seed_values(args.seed_values) if args.seed_values is not None else [123]
    num_codebook_vectors_values = _parse_csv_ints(args.num_codebook_vectors) if args.num_codebook_vectors is not None else [128]
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
        fox_remote_path_backend=args.fox_remote_path_backend or 'torch',
        fox_remote_read_topk=2,
        fox_remote_formula='clr_v1',
        fox_clr_rank=args.fox_clr_rank,
        fox_clr_use_den_residual=(args.fox_clr_use_den_residual == 'true'),
        fox_clr_remat_mode=args.fox_clr_remat_mode,
        fox_clr_selector_mode=selector_mode,
        fox_clr_merge_mode=merge_mode,
        fox_clr_gate_mode=gate_mode,
        fox_clr_lambda_remote=lambda_remote,
        fox_clr_gate_init_bias=gate_init_bias,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        cache_dir=args.cache_dir,
        wandb_project=args.project,
        wandb_entity=args.entity,
        max_epochs=args.max_epochs,
        metrics_white_list=metrics_white_list,
        experiment_part=experiment_part,
        experiment_mode=experiment_mode,
    )


def _rewrite_run_id(config, *, run_id: str):
    config = config.model_copy(deep=True) if hasattr(config, "model_copy") else config.copy(deep=True)
    config.run_id = run_id
    return config


def _build_single(args, *, experiment_part: str, experiment_mode: str, selector_mode: str, merge_mode: str, gate_mode: str = 'off', lambda_remote: float = 1.0, gate_init_bias: float = -2.0, run_id: str):
    kwargs = _common_builder_kwargs(
        args,
        experiment_part=experiment_part,
        experiment_mode=experiment_mode,
        selector_mode=selector_mode,
        merge_mode=merge_mode,
        gate_mode=gate_mode,
        lambda_remote=lambda_remote,
        gate_init_bias=gate_init_bias,
    )
    configs = build_configs(**kwargs)
    if len(configs) != 1:
        raise RuntimeError(f'Expected exactly 1 config for {experiment_part}/{experiment_mode}, got {len(configs)}')
    return _rewrite_run_id(configs[0], run_id=run_id)


def _resolve_probe_seed_value(args, *, probe_name: str) -> int:
    seed_values = _parse_seed_values(args.seed_values) if args.seed_values is not None else [123]
    if len(seed_values) != 1:
        raise RuntimeError(f'{probe_name} 需要且只接受 1 个 seed, 当前收到: {seed_values}')
    seed_value = int(seed_values[0])
    data_seed = int(args.data_seed)
    if seed_value != data_seed:
        raise RuntimeError(f'{probe_name} 要求 seed == data_seed, 当前收到 seed={seed_value}, data_seed={data_seed}')
    return seed_value


def build_e2_main_train_configs(args):
    configs = [
        _build_single(args, experiment_part='e2_main', experiment_mode='baseline', selector_mode='den_aware', merge_mode='shared_den', run_id='baseline'),
    ]
    for value, tag in ((0.25, '025'), (0.5, '050'), (1.0, '100')):
        configs.append(_build_single(args, experiment_part='e2_main', experiment_mode='2a', selector_mode='score_only', merge_mode='residual_add', lambda_remote=value, run_id=f'2a-l{tag}'))
        configs.append(_build_single(args, experiment_part='e2_main', experiment_mode='2b', selector_mode='score_only', merge_mode='shared_local_den', lambda_remote=value, run_id=f'2b-l{tag}'))
        configs.append(_build_single(args, experiment_part='e2_main', experiment_mode='2c', selector_mode='score_only', merge_mode='residual_add', gate_mode='shared_query_linear', lambda_remote=value, gate_init_bias=-2.0, run_id=f'2c-lmax{tag}'))
    return configs


def build_e2_main_smoke_configs(args):
    return [
        _build_single(args, experiment_part='e2_main', experiment_mode='baseline', selector_mode='den_aware', merge_mode='shared_den', run_id='baseline'),
        _build_single(args, experiment_part='e2_main', experiment_mode='2a', selector_mode='score_only', merge_mode='residual_add', lambda_remote=0.5, run_id='2a-l050'),
        _build_single(args, experiment_part='e2_main', experiment_mode='2b', selector_mode='score_only', merge_mode='shared_local_den', lambda_remote=0.5, run_id='2b-l050'),
        _build_single(args, experiment_part='e2_main', experiment_mode='2c', selector_mode='score_only', merge_mode='residual_add', gate_mode='shared_query_linear', lambda_remote=0.5, gate_init_bias=-2.0, run_id='2c-lmax050'),
    ]


def build_e2_main_tail_configs(args):
    return [
        _build_single(
            args,
            experiment_part='e2_main',
            experiment_mode='2c',
            selector_mode='score_only',
            merge_mode='residual_add',
            gate_mode='shared_query_linear',
            lambda_remote=1.0,
            gate_init_bias=-2.0,
            run_id='2c-lmax100',
        )
    ]


def build_e2b_train_configs(args):
    configs = [
        _build_single(args, experiment_part='e2b', experiment_mode='baseline', selector_mode='den_aware', merge_mode='shared_den', run_id='baseline'),
    ]
    for value, tag in ((0.25, '025'), (0.5, '050'), (1.0, '100')):
        configs.append(_build_single(args, experiment_part='e2b', experiment_mode='2a', selector_mode='den_aware', merge_mode='residual_add', lambda_remote=value, run_id=f'e2b-2a-l{tag}'))
        configs.append(_build_single(args, experiment_part='e2b', experiment_mode='2b', selector_mode='den_aware', merge_mode='shared_local_den', lambda_remote=value, run_id=f'e2b-2b-l{tag}'))
        configs.append(_build_single(args, experiment_part='e2b', experiment_mode='2c', selector_mode='den_aware', merge_mode='residual_add', gate_mode='shared_query_linear', lambda_remote=value, gate_init_bias=-2.0, run_id=f'e2b-2c-lmax{tag}'))
    return configs


def build_e2b_smoke_configs(args):
    return [
        _build_single(args, experiment_part='e2b', experiment_mode='baseline', selector_mode='den_aware', merge_mode='shared_den', run_id='baseline'),
        _build_single(args, experiment_part='e2b', experiment_mode='2a', selector_mode='den_aware', merge_mode='residual_add', lambda_remote=0.5, run_id='e2b-2a-l050'),
        _build_single(args, experiment_part='e2b', experiment_mode='2b', selector_mode='den_aware', merge_mode='shared_local_den', lambda_remote=0.5, run_id='e2b-2b-l050'),
        _build_single(args, experiment_part='e2b', experiment_mode='2c', selector_mode='den_aware', merge_mode='residual_add', gate_mode='shared_query_linear', lambda_remote=0.5, gate_init_bias=-2.0, run_id='e2b-2c-lmax050'),
    ]


def build_e2b_tail_configs(args):
    return [
        _build_single(
            args,
            experiment_part='e2b',
            experiment_mode='2c',
            selector_mode='den_aware',
            merge_mode='residual_add',
            gate_mode='shared_query_linear',
            lambda_remote=1.0,
            gate_init_bias=-2.0,
            run_id='e2b-2c-lmax100',
        )
    ]


def build_e2b_2a_l100_probe_configs(args):
    seed_value = _resolve_probe_seed_value(args, probe_name='e2b-2a-l100 probe')
    return [
        _build_single(
            args,
            experiment_part='e2b_probe',
            experiment_mode='2a',
            selector_mode='den_aware',
            merge_mode='residual_add',
            gate_mode='off',
            lambda_remote=1.0,
            gate_init_bias=-2.0,
            run_id=f'e2b-2a-l100-s{seed_value}',
        )
    ]


def build_e2b_baseline_probe_configs(args):
    seed_value = _resolve_probe_seed_value(args, probe_name='e2b baseline probe')
    return [
        _build_single(
            args,
            experiment_part='e2b_probe',
            experiment_mode='baseline',
            selector_mode='den_aware',
            merge_mode='shared_den',
            gate_mode='off',
            lambda_remote=1.0,
            gate_init_bias=-2.0,
            run_id=f'baseline-s{seed_value}',
        )
    ]
