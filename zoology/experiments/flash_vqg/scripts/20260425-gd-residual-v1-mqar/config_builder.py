from __future__ import annotations

from zoology.experiments.flash_vqg.flash_vqg_suite import build_configs
from zoology.experiments.flash_vqg.run_flash_vqg_suite import (
    _parse_csv_floats,
    _parse_csv_ints,
    _parse_remote_read_topk_values,
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


def _copy_config(config):
    return config.model_copy(deep=True) if hasattr(config, "model_copy") else config.copy(deep=True)


def _apply_smoke_data_budget(config):
    smoke_train_examples = {
        (64, 4): 128,
        (128, 8): 64,
        (256, 16): 64,
        (256, 32): 64,
        (256, 64): 64,
    }
    smoke_test_examples = {
        (64, 4): 4,
        (64, 8): 4,
        (64, 16): 4,
        (128, 32): 4,
        (256, 64): 4,
        (512, 64): 4,
        (512, 128): 4,
        (1024, 256): 4,
    }

    config = _copy_config(config)
    config.data.train_configs = [
        segment.model_copy(
            update={"num_examples": smoke_train_examples[(segment.input_seq_len, segment.num_kv_pairs)]}
        )
        if hasattr(segment, "model_copy")
        else segment.copy(
            update={"num_examples": smoke_train_examples[(segment.input_seq_len, segment.num_kv_pairs)]}
        )
        for segment in config.data.train_configs
    ]
    config.data.test_configs = [
        segment.model_copy(
            update={"num_examples": smoke_test_examples[(segment.input_seq_len, segment.num_kv_pairs)]}
        )
        if hasattr(segment, "model_copy")
        else segment.copy(
            update={"num_examples": smoke_test_examples[(segment.input_seq_len, segment.num_kv_pairs)]}
        )
        for segment in config.data.test_configs
    ]
    train_batch_size, _ = config.data.batch_size
    config.data.batch_size = (int(train_batch_size), 1)
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


def _read_topk_tag(value: int | None) -> str:
    return "dense" if value is None else f"top{int(value)}"


def _resolve_remote_read_topk_values(args) -> list[int | None]:
    raw = getattr(args, "fox_remote_read_topk_values", None)
    if raw is None:
        return [2]
    return _parse_remote_read_topk_values(str(raw))


def _common_builder_kwargs(args, *, experiment_mode: str):
    resolved_experiment_mode = getattr(args, "experiment_mode", None) or experiment_mode
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
    remote_read_topk_values = _resolve_remote_read_topk_values(args)

    remote_formula = str(getattr(args, "fox_remote_formula", "gd_residual_v1"))
    if remote_formula != "gd_residual_v1":
        raise ValueError(
            "20260425-gd-residual-v1-mqar builder 只支持 "
            "fox_remote_formula='gd_residual_v1'."
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
            data_seed=int(args.data_seed),
            num_codebook_vectors_values=[num_codebook_vectors],
            fox_remote_path_backend=str(getattr(args, "fox_remote_path_backend", "torch") or "torch"),
            fox_remote_read_topk_values=remote_read_topk_values,
            fox_remote_formula=remote_formula,
            fox_gd_residual_rank=int(getattr(args, "fox_gd_residual_rank", 16)),
            fox_gd_residual_write_topk=int(getattr(args, "fox_gd_residual_write_topk", 4)),
            fox_gd_residual_builder=str(
                getattr(args, "fox_gd_residual_builder", "grouped_chunk_torch_ref")
            ),
            fox_gd_residual_pack_mode=str(
                getattr(args, "fox_gd_residual_pack_mode", "semivec_ref")
            ),
            fox_gd_residual_chunk_size=int(getattr(args, "fox_gd_residual_chunk_size", 64)),
            fox_gd_residual_mu_min_count=float(
                getattr(args, "fox_gd_residual_mu_min_count", 1.0)
            ),
            fox_gd_residual_addr_eps=float(getattr(args, "fox_gd_residual_addr_eps", 1e-6)),
            fox_gd_residual_den_eps=float(getattr(args, "fox_gd_residual_den_eps", 1e-6)),
            fox_gd_residual_rho_eps=float(getattr(args, "fox_gd_residual_rho_eps", 1e-12)),
            fox_gd_residual_beta_init=float(getattr(args, "fox_gd_residual_beta_init", 0.5)),
            fox_gd_residual_lambda_init=float(getattr(args, "fox_gd_residual_lambda_init", 0.05)),
            fox_gd_residual_norm_with_gain=_parse_bool_arg(
                getattr(args, "fox_gd_residual_norm_with_gain", False),
                field_name="fox_gd_residual_norm_with_gain",
            ),
            fox_gd_residual_use_separate_addr_codebook=_parse_bool_arg(
                getattr(args, "fox_gd_residual_use_separate_addr_codebook", False),
                field_name="fox_gd_residual_use_separate_addr_codebook",
            ),
            vq_score_mode=str(getattr(args, "vq_score_mode", "codebook_dot")),
            vq_weight_mode=str(getattr(args, "vq_weight_mode", "dense_softmax")),
            vq_update_mode=str(getattr(args, "vq_update_mode", "grad")),
            vq_softmax_tau=float(getattr(args, "vq_softmax_tau", 1.0)),
            vq_topk=int(getattr(args, "vq_topk", 4)),
            gradient_accumulation_steps=int(args.gradient_accumulation_steps),
            train_batch_size=int(args.train_batch_size) if args.train_batch_size is not None else None,
            eval_batch_size=int(args.eval_batch_size) if args.eval_batch_size is not None else None,
            cache_dir=args.cache_dir,
            wandb_project=args.project,
            wandb_entity=args.entity,
            max_epochs=int(args.max_epochs),
            metrics_white_list=metrics_white_list,
            experiment_part="gd_residual_v1_mqar",
            experiment_mode=resolved_experiment_mode,
            validations_per_epoch=int(getattr(args, "validations_per_epoch", 1)),
        ),
        seed_value,
        int(args.data_seed),
        remote_read_topk_values,
    )


def _build_single(args, *, experiment_mode: str):
    kwargs, seed_value, data_seed, remote_read_topk_values = _common_builder_kwargs(
        args,
        experiment_mode=experiment_mode,
    )
    configs = build_configs(**kwargs)
    if len(configs) != 1:
        raise RuntimeError(f"Expected exactly 1 config for {experiment_mode}, got {len(configs)}")
    read_topk = _require_single(remote_read_topk_values, field_name="fox_remote_read_topk_values")
    train_batch_size = int(args.train_batch_size) if args.train_batch_size is not None else 256
    run_id = (
        f"gd-residual-v1-{experiment_mode}-s{seed_value}-d{data_seed}"
        f"-rread-{_read_topk_tag(read_topk)}"
        f"-r{int(getattr(args, 'fox_gd_residual_rank', 16))}"
        f"-wk{int(getattr(args, 'fox_gd_residual_write_topk', 4))}"
        f"-b{train_batch_size}"
    )
    run_id = str(getattr(args, "run_id", None) or run_id)
    return _rewrite_run_id(configs[0], run_id=run_id)


def build_gd_residual_v1_smoke_configs(args):
    return [_apply_smoke_data_budget(_build_single(args, experiment_mode="smoke"))]


def build_gd_residual_v1_train_configs(args):
    return [_build_single(args, experiment_mode="train")]
