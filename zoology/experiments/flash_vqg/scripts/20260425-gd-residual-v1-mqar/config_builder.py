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


def _optional_float_arg(args, name: str):
    value = getattr(args, name, None)
    if value is None:
        return None
    return float(value)


def _float_tag(value) -> str:
    return str(float(value)).replace("-", "m").replace(".", "p")


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


def _parse_csv_ints_arg(args, name: str, *, default: str) -> list[int]:
    value = getattr(args, name, default)
    if value is None:
        value = default
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    if not str(value).strip():
        return []
    return _parse_csv_ints(str(value))


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
    disable_early_stopping = _parse_bool_arg(
        getattr(args, "disable_early_stopping", "false"),
        field_name="disable_early_stopping",
    )
    read_churn_probe_enabled = _parse_bool_arg(
        getattr(args, "read_churn_probe_enabled", "false"),
        field_name="read_churn_probe_enabled",
    )
    read_churn_probe_valid_batches = _parse_csv_ints_arg(
        args,
        "read_churn_probe_valid_batches",
        default="0",
    )
    read_churn_probe_max_samples = int(getattr(args, "read_churn_probe_max_samples", 16))
    if read_churn_probe_max_samples <= 0:
        raise ValueError("read_churn_probe_max_samples 必须是正整数.")
    read_churn_probe_query_only = _parse_bool_arg(
        getattr(args, "read_churn_probe_query_only", "true"),
        field_name="read_churn_probe_query_only",
    )
    read_trace_enabled = _parse_bool_arg(
        getattr(args, "read_trace_enabled", "false"),
        field_name="read_trace_enabled",
    )
    read_trace_valid_batches = _parse_csv_ints_arg(
        args,
        "read_trace_valid_batches",
        default="0",
    )
    read_trace_max_samples = int(getattr(args, "read_trace_max_samples", 4))
    if read_trace_max_samples <= 0:
        raise ValueError("read_trace_max_samples 必须是正整数.")
    read_trace_query_only = _parse_bool_arg(
        getattr(args, "read_trace_query_only", "true"),
        field_name="read_trace_query_only",
    )
    read_trace_max_queries_per_sample = int(
        getattr(args, "read_trace_max_queries_per_sample", 8)
    )
    if read_trace_max_queries_per_sample <= 0:
        raise ValueError("read_trace_max_queries_per_sample 必须是正整数.")
    read_trace_train_steps = _parse_csv_ints_arg(
        args,
        "read_trace_train_steps",
        default="",
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
            fox_gd_residual_addr_init_rng_mode=str(
                getattr(args, "fox_gd_residual_addr_init_rng_mode", "global")
            ),
            fox_gd_residual_addr_init_seed=getattr(
                args, "fox_gd_residual_addr_init_seed", None
            ),
            fox_gd_residual_beta_init=float(getattr(args, "fox_gd_residual_beta_init", 0.5)),
            fox_gd_residual_beta_cap=(
                None
                if getattr(args, "fox_gd_residual_beta_cap", None) is None
                else float(getattr(args, "fox_gd_residual_beta_cap"))
            ),
            fox_gd_residual_beta_cap_final=(
                None
                if getattr(args, "fox_gd_residual_beta_cap_final", None) is None
                else float(getattr(args, "fox_gd_residual_beta_cap_final"))
            ),
            fox_gd_residual_beta_cap_release_start_train_steps=int(
                getattr(
                    args,
                    "fox_gd_residual_beta_cap_release_start_train_steps",
                    0,
                )
            ),
            fox_gd_residual_beta_cap_release_end_train_steps=int(
                getattr(
                    args,
                    "fox_gd_residual_beta_cap_release_end_train_steps",
                    0,
                )
            ),
            fox_gd_residual_beta_cap_eval_policy=str(
                getattr(args, "fox_gd_residual_beta_cap_eval_policy", "final")
            ),
            fox_gd_residual_beta_control_mode=str(
                getattr(args, "fox_gd_residual_beta_control_mode", "hard_cap")
            ),
            fox_gd_residual_beta_sigmoid_temp=float(
                getattr(args, "fox_gd_residual_beta_sigmoid_temp", 1.0)
            ),
            fox_gd_residual_beta_low=_optional_float_arg(
                args, "fox_gd_residual_beta_low"
            ),
            fox_gd_residual_beta_high=_optional_float_arg(
                args, "fox_gd_residual_beta_high"
            ),
            fox_gd_residual_beta_low_final=_optional_float_arg(
                args, "fox_gd_residual_beta_low_final"
            ),
            fox_gd_residual_beta_high_final=_optional_float_arg(
                args, "fox_gd_residual_beta_high_final"
            ),
            fox_gd_residual_beta_band_release_start_train_steps=int(
                getattr(
                    args,
                    "fox_gd_residual_beta_band_release_start_train_steps",
                    0,
                )
            ),
            fox_gd_residual_beta_band_release_end_train_steps=int(
                getattr(
                    args,
                    "fox_gd_residual_beta_band_release_end_train_steps",
                    0,
                )
            ),
            fox_gd_residual_beta_band_eval_policy=str(
                getattr(args, "fox_gd_residual_beta_band_eval_policy", "final")
            ),
            fox_gd_residual_beta_band_schedule=str(
                getattr(args, "fox_gd_residual_beta_band_schedule", "smoothstep")
            ),
            fox_gd_residual_lambda_init=float(getattr(args, "fox_gd_residual_lambda_init", 0.05)),
            fox_gd_residual_lambda_floor=float(
                getattr(args, "fox_gd_residual_lambda_floor", 0.0)
            ),
            fox_gd_residual_write_strength_mode=str(
                getattr(args, "fox_gd_residual_write_strength_mode", "renorm_topk")
            ),
            fox_gd_residual_write_strength_cap=(
                None
                if getattr(args, "fox_gd_residual_write_strength_cap", None) is None
                else float(getattr(args, "fox_gd_residual_write_strength_cap"))
            ),
            fox_gd_residual_write_strength_cap_mode=str(
                getattr(args, "fox_gd_residual_write_strength_cap_mode", "hard")
            ),
            fox_gd_residual_write_strength_cap_until_train_steps=int(
                getattr(
                    args,
                    "fox_gd_residual_write_strength_cap_until_train_steps",
                    0,
                )
            ),
            fox_gd_residual_write_strength_cap_final=(
                None
                if getattr(args, "fox_gd_residual_write_strength_cap_final", None) is None
                else float(getattr(args, "fox_gd_residual_write_strength_cap_final"))
            ),
            fox_gd_residual_write_strength_cap_release_start_train_steps=int(
                getattr(
                    args,
                    "fox_gd_residual_write_strength_cap_release_start_train_steps",
                    0,
                )
            ),
            fox_gd_residual_write_strength_cap_release_end_train_steps=int(
                getattr(
                    args,
                    "fox_gd_residual_write_strength_cap_release_end_train_steps",
                    0,
                )
            ),
            fox_gd_residual_write_strength_cap_eval_policy=str(
                getattr(args, "fox_gd_residual_write_strength_cap_eval_policy", "final")
            ),
            fox_gd_residual_write_budget=(
                None
                if getattr(args, "fox_gd_residual_write_budget", None) is None
                else float(getattr(args, "fox_gd_residual_write_budget"))
            ),
            fox_gd_residual_write_budget_final=(
                None
                if getattr(args, "fox_gd_residual_write_budget_final", None) is None
                else float(getattr(args, "fox_gd_residual_write_budget_final"))
            ),
            fox_gd_residual_write_budget_release_start_train_steps=int(
                getattr(
                    args,
                    "fox_gd_residual_write_budget_release_start_train_steps",
                    0,
                )
            ),
            fox_gd_residual_write_budget_release_end_train_steps=int(
                getattr(
                    args,
                    "fox_gd_residual_write_budget_release_end_train_steps",
                    0,
                )
            ),
            fox_gd_residual_write_budget_eval_policy=str(
                getattr(args, "fox_gd_residual_write_budget_eval_policy", "final")
            ),
            fox_gd_residual_write_budget_schedule=str(
                getattr(args, "fox_gd_residual_write_budget_schedule", "smoothstep")
            ),
            fox_gd_residual_write_total_cap=(
                None
                if getattr(args, "fox_gd_residual_write_total_cap", None) is None
                else float(getattr(args, "fox_gd_residual_write_total_cap"))
            ),
            fox_gd_residual_write_total_cap_final=(
                None
                if getattr(args, "fox_gd_residual_write_total_cap_final", None) is None
                else float(getattr(args, "fox_gd_residual_write_total_cap_final"))
            ),
            fox_gd_residual_write_total_cap_release_start_train_steps=int(
                getattr(
                    args,
                    "fox_gd_residual_write_total_cap_release_start_train_steps",
                    0,
                )
            ),
            fox_gd_residual_write_total_cap_release_end_train_steps=int(
                getattr(
                    args,
                    "fox_gd_residual_write_total_cap_release_end_train_steps",
                    0,
                )
            ),
            fox_gd_residual_write_total_cap_eval_policy=str(
                getattr(args, "fox_gd_residual_write_total_cap_eval_policy", "final")
            ),
            fox_gd_residual_write_total_cap_schedule=str(
                getattr(args, "fox_gd_residual_write_total_cap_schedule", "smoothstep")
            ),
            fox_gd_residual_write_q_alpha=float(
                getattr(args, "fox_gd_residual_write_q_alpha", 1.0)
            ),
            fox_gd_residual_m_norm_cap=(
                None
                if getattr(args, "fox_gd_residual_m_norm_cap", None) is None
                else float(getattr(args, "fox_gd_residual_m_norm_cap"))
            ),
            fox_gd_residual_update_norm_cap=(
                None
                if getattr(args, "fox_gd_residual_update_norm_cap", None) is None
                else float(getattr(args, "fox_gd_residual_update_norm_cap"))
            ),
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
            codebook_init_rng_mode=str(getattr(args, "codebook_init_rng_mode", "global")),
            codebook_init_seed=getattr(args, "codebook_init_seed", None),
            vq_topk=int(getattr(args, "vq_topk", 4)),
            gradient_accumulation_steps=int(args.gradient_accumulation_steps),
            train_batch_size=int(args.train_batch_size) if args.train_batch_size is not None else None,
            eval_batch_size=int(args.eval_batch_size) if args.eval_batch_size is not None else None,
            cache_dir=args.cache_dir,
            wandb_project=args.project,
            wandb_entity=args.entity,
            max_epochs=int(args.max_epochs),
            max_train_steps=getattr(args, "max_train_steps", None),
            max_validation_batches=getattr(args, "max_validation_batches", None),
            metrics_white_list=metrics_white_list,
            read_churn_probe_enabled=read_churn_probe_enabled,
            read_churn_probe_valid_batches=read_churn_probe_valid_batches,
            read_churn_probe_max_samples=read_churn_probe_max_samples,
            read_churn_probe_query_only=read_churn_probe_query_only,
            read_trace_enabled=read_trace_enabled,
            read_trace_valid_batches=read_trace_valid_batches,
            read_trace_max_samples=read_trace_max_samples,
            read_trace_query_only=read_trace_query_only,
            read_trace_max_queries_per_sample=read_trace_max_queries_per_sample,
            read_trace_output_dir=getattr(args, "read_trace_output_dir", None),
            read_trace_train_steps=read_trace_train_steps,
            experiment_part="gd_residual_v1_mqar",
            experiment_mode=resolved_experiment_mode,
            validations_per_epoch=int(getattr(args, "validations_per_epoch", 1)),
            early_stopping_metric=None if disable_early_stopping else "valid/accuracy",
            early_stopping_threshold=None if disable_early_stopping else 0.99,
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
    write_strength_mode = str(
        getattr(args, "fox_gd_residual_write_strength_mode", "renorm_topk")
    )
    if write_strength_mode != "renorm_topk":
        run_id = f"{run_id}-wmode{write_strength_mode}"
    codebook_init_rng_mode = str(getattr(args, "codebook_init_rng_mode", "global"))
    if codebook_init_rng_mode != "global":
        run_id = (
            f"{run_id}-cbinit{codebook_init_rng_mode}"
            f"s{getattr(args, 'codebook_init_seed', None)}"
        )
    addr_init_rng_mode = str(
        getattr(args, "fox_gd_residual_addr_init_rng_mode", "global")
    )
    if addr_init_rng_mode != "global":
        run_id = (
            f"{run_id}-addrinit{addr_init_rng_mode}"
            f"s{getattr(args, 'fox_gd_residual_addr_init_seed', None)}"
        )
    write_strength_cap = getattr(args, "fox_gd_residual_write_strength_cap", None)
    if write_strength_cap is not None:
        cap_tag = str(write_strength_cap).replace(".", "p")
        run_id = f"{run_id}-wcap{cap_tag}"
        cap_mode = str(getattr(args, "fox_gd_residual_write_strength_cap_mode", "hard"))
        if cap_mode != "hard":
            run_id = f"{run_id}-wcapmode{cap_mode}"
        cap_until = int(
            getattr(args, "fox_gd_residual_write_strength_cap_until_train_steps", 0)
        )
        if cap_until != 0:
            run_id = f"{run_id}-wcapuntil{cap_until}"
        cap_final = getattr(args, "fox_gd_residual_write_strength_cap_final", None)
        if cap_final is not None:
            cap_final_tag = str(cap_final).replace(".", "p")
            run_id = f"{run_id}-wcapfinal{cap_final_tag}"
            rel_start = int(
                getattr(
                    args,
                    "fox_gd_residual_write_strength_cap_release_start_train_steps",
                    0,
                )
            )
            rel_end = int(
                getattr(
                    args,
                    "fox_gd_residual_write_strength_cap_release_end_train_steps",
                    0,
                )
            )
            run_id = f"{run_id}-wcaprel{rel_start}to{rel_end}"
            cap_eval_policy = str(
                getattr(args, "fox_gd_residual_write_strength_cap_eval_policy", "final")
            )
            if cap_eval_policy != "final":
                run_id = f"{run_id}-wcapeval{cap_eval_policy}"
    write_budget = getattr(args, "fox_gd_residual_write_budget", None)
    if write_budget is not None:
        budget_tag = str(write_budget).replace(".", "p")
        run_id = f"{run_id}-wbudget{budget_tag}"
        budget_final = getattr(args, "fox_gd_residual_write_budget_final", None)
        if budget_final is not None:
            budget_final_tag = str(budget_final).replace(".", "p")
            rel_start = int(
                getattr(
                    args,
                    "fox_gd_residual_write_budget_release_start_train_steps",
                    0,
                )
            )
            rel_end = int(
                getattr(
                    args,
                    "fox_gd_residual_write_budget_release_end_train_steps",
                    0,
                )
            )
            budget_eval_policy = str(
                getattr(args, "fox_gd_residual_write_budget_eval_policy", "final")
            )
            budget_schedule = str(
                getattr(args, "fox_gd_residual_write_budget_schedule", "smoothstep")
            )
            run_id = (
                f"{run_id}-wbudgetfinal{budget_final_tag}"
                f"-wbudgetrel{rel_start}to{rel_end}"
            )
            if budget_eval_policy != "final":
                run_id = f"{run_id}-wbudgeteval{budget_eval_policy}"
            if budget_schedule != "smoothstep":
                run_id = f"{run_id}-wbudgetsched{budget_schedule}"
    write_total_cap = getattr(args, "fox_gd_residual_write_total_cap", None)
    if write_total_cap is not None:
        total_cap_tag = str(write_total_cap).replace(".", "p")
        run_id = f"{run_id}-wtotalcap{total_cap_tag}"
        total_cap_final = getattr(args, "fox_gd_residual_write_total_cap_final", None)
        if total_cap_final is not None:
            total_cap_final_tag = str(total_cap_final).replace(".", "p")
            rel_start = int(
                getattr(
                    args,
                    "fox_gd_residual_write_total_cap_release_start_train_steps",
                    0,
                )
            )
            rel_end = int(
                getattr(
                    args,
                    "fox_gd_residual_write_total_cap_release_end_train_steps",
                    0,
                )
            )
            total_cap_eval_policy = str(
                getattr(args, "fox_gd_residual_write_total_cap_eval_policy", "final")
            )
            total_cap_schedule = str(
                getattr(args, "fox_gd_residual_write_total_cap_schedule", "smoothstep")
            )
            run_id = (
                f"{run_id}-wtotalcapfinal{total_cap_final_tag}"
                f"-wtotalcaprel{rel_start}to{rel_end}"
            )
            if total_cap_eval_policy != "final":
                run_id = f"{run_id}-wtotalcapeval{total_cap_eval_policy}"
            if total_cap_schedule != "smoothstep":
                run_id = f"{run_id}-wtotalcapsched{total_cap_schedule}"
    write_q_alpha = float(getattr(args, "fox_gd_residual_write_q_alpha", 1.0))
    if write_q_alpha != 1.0:
        run_id = f"{run_id}-wqalpha{_float_tag(write_q_alpha)}"
    m_norm_cap = getattr(args, "fox_gd_residual_m_norm_cap", None)
    if m_norm_cap is not None:
        m_norm_cap_tag = str(m_norm_cap).replace(".", "p")
        run_id = f"{run_id}-mcap{m_norm_cap_tag}"
    update_norm_cap = getattr(args, "fox_gd_residual_update_norm_cap", None)
    if update_norm_cap is not None:
        update_norm_cap_tag = str(update_norm_cap).replace(".", "p")
        run_id = f"{run_id}-ucap{update_norm_cap_tag}"
    beta_cap = getattr(args, "fox_gd_residual_beta_cap", None)
    if beta_cap is not None:
        beta_cap_tag = str(beta_cap).replace(".", "p")
        run_id = f"{run_id}-betacap{beta_cap_tag}"
        beta_cap_final = getattr(args, "fox_gd_residual_beta_cap_final", None)
        if beta_cap_final is not None:
            beta_cap_final_tag = str(beta_cap_final).replace(".", "p")
            run_id = f"{run_id}-betacapfinal{beta_cap_final_tag}"
            beta_rel_start = int(
                getattr(
                    args,
                    "fox_gd_residual_beta_cap_release_start_train_steps",
                    0,
                )
            )
            beta_rel_end = int(
                getattr(
                    args,
                    "fox_gd_residual_beta_cap_release_end_train_steps",
                    0,
                )
            )
            run_id = f"{run_id}-betacaprel{beta_rel_start}to{beta_rel_end}"
            beta_cap_eval_policy = str(
                getattr(args, "fox_gd_residual_beta_cap_eval_policy", "final")
            )
            if beta_cap_eval_policy != "final":
                run_id = f"{run_id}-betacapeval{beta_cap_eval_policy}"
    beta_control_mode = str(getattr(args, "fox_gd_residual_beta_control_mode", "hard_cap"))
    if beta_control_mode != "hard_cap":
        run_id = f"{run_id}-betactrl{beta_control_mode}"
    beta_temp = float(getattr(args, "fox_gd_residual_beta_sigmoid_temp", 1.0))
    if beta_temp != 1.0:
        run_id = f"{run_id}-betatemp{_float_tag(beta_temp)}"
    beta_low = getattr(args, "fox_gd_residual_beta_low", None)
    if beta_low is not None:
        run_id = f"{run_id}-betalow{_float_tag(beta_low)}"
    beta_high = getattr(args, "fox_gd_residual_beta_high", None)
    if beta_high is not None:
        run_id = f"{run_id}-betahigh{_float_tag(beta_high)}"
    beta_low_final = getattr(args, "fox_gd_residual_beta_low_final", None)
    if beta_low_final is not None:
        run_id = f"{run_id}-betalowfinal{_float_tag(beta_low_final)}"
    beta_high_final = getattr(args, "fox_gd_residual_beta_high_final", None)
    if beta_high_final is not None:
        run_id = f"{run_id}-betahighfinal{_float_tag(beta_high_final)}"
    if beta_low_final is not None or beta_high_final is not None:
        beta_band_rel_start = int(
            getattr(
                args,
                "fox_gd_residual_beta_band_release_start_train_steps",
                0,
            )
        )
        beta_band_rel_end = int(
            getattr(
                args,
                "fox_gd_residual_beta_band_release_end_train_steps",
                0,
            )
        )
        run_id = f"{run_id}-betabandrel{beta_band_rel_start}to{beta_band_rel_end}"
        beta_band_eval_policy = str(
            getattr(args, "fox_gd_residual_beta_band_eval_policy", "final")
        )
        if beta_band_eval_policy != "final":
            run_id = f"{run_id}-betabandeval{beta_band_eval_policy}"
        beta_band_schedule = str(
            getattr(args, "fox_gd_residual_beta_band_schedule", "smoothstep")
        )
        if beta_band_schedule != "smoothstep":
            run_id = f"{run_id}-betabandsched{beta_band_schedule}"
    lambda_floor = float(getattr(args, "fox_gd_residual_lambda_floor", 0.0))
    if lambda_floor != 0.0:
        lambda_floor_tag = str(lambda_floor).replace(".", "p")
        run_id = f"{run_id}-lambdafloor{lambda_floor_tag}"
    run_id = str(getattr(args, "run_id", None) or run_id)
    return _rewrite_run_id(configs[0], run_id=run_id)


def build_gd_residual_v1_smoke_configs(args):
    return [_apply_smoke_data_budget(_build_single(args, experiment_mode="smoke"))]


def build_gd_residual_v1_train_configs(args):
    return [_build_single(args, experiment_mode="train")]
