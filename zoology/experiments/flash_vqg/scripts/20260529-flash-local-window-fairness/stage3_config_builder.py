from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from zoology.experiments.flash_vqg.flash_vqg_suite import build_configs

_BASE_BUILDER_PATH = (
    Path(__file__).resolve().parent
    / "../20260425-gd-residual-v1-mqar/config_builder.py"
).resolve()

_VARIANTS = {
    "local-only": {
        "aliases": {"local-only", "local_only", "localonly"},
        "local_num_blocks": 2,
        "if_remote_enabled": False,
        "run_tag": "localonly",
        "mode_tag": "local_only",
    },
    "local1": {
        "aliases": {"local1", "local-1", "local_1"},
        "local_num_blocks": 1,
        "if_remote_enabled": True,
        "run_tag": "local1",
        "mode_tag": "local1",
    },
    "local4": {
        "aliases": {"local4", "local-4", "local_4"},
        "local_num_blocks": 4,
        "if_remote_enabled": True,
        "run_tag": "local4",
        "mode_tag": "local4",
    },
}

_ALIAS_TO_VARIANT = {
    alias: name
    for name, spec in _VARIANTS.items()
    for alias in spec["aliases"]
}


def _load_base_builder():
    spec = importlib.util.spec_from_file_location(
        "gd_residual_v1_config_builder_for_local_window_fairness",
        _BASE_BUILDER_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 base builder: {_BASE_BUILDER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _normalise_variant(raw: str | None) -> str:
    if raw is None:
        raise ValueError(
            "必须通过 FLASH_LOCAL_WINDOW_VARIANT, run_id 或 experiment_mode 指定 "
            "local-only/local1/local4."
        )
    token = str(raw).strip().lower().replace("_", "-")
    if token in _ALIAS_TO_VARIANT:
        return _ALIAS_TO_VARIANT[token]
    raise ValueError(f"不支持的 local window variant: {raw!r}")


def _infer_variant(args) -> str:
    env_value = os.environ.get("FLASH_LOCAL_WINDOW_VARIANT")
    if env_value:
        return _normalise_variant(env_value)

    candidates = [
        str(getattr(args, "run_id", "") or "").lower(),
        str(getattr(args, "experiment_mode", "") or "").lower(),
        str(getattr(args, "launch_id_prefix", "") or "").lower(),
    ]
    for text in candidates:
        if not text:
            continue
        if "localonly" in text or "local-only" in text or "local_only" in text:
            return "local-only"
        if "local1" in text or "local-1" in text or "local_1" in text:
            return "local1"
        if "local4" in text or "local-4" in text or "local_4" in text:
            return "local4"
    raise ValueError(
        "无法从环境变量, run_id, experiment_mode 或 launch_id_prefix 推断 "
        "local window variant."
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


def _read_topk_tag(value: int | None) -> str:
    return "dense" if value is None else f"top{int(value)}"


def _default_run_id(args, *, variant_name: str, seed_value: int, data_seed: int, remote_read_topk: int | None) -> str:
    spec = _VARIANTS[variant_name]
    train_batch_size = int(args.train_batch_size) if args.train_batch_size is not None else 256
    grad_accum = int(getattr(args, "gradient_accumulation_steps", 1))
    rank = int(getattr(args, "fox_gd_residual_rank", 16))
    num_codes = _require_single(
        _load_base_builder()._parse_csv_ints(args.num_codebook_vectors),
        field_name="num_codebook_vectors",
    )
    max_epochs = int(getattr(args, "max_epochs", 4))
    return (
        f"gd-cb{num_codes}-r{rank}-s{seed_value}-{spec['run_tag']}-d{data_seed}"
        f"-rread-{_read_topk_tag(remote_read_topk)}"
        f"-b{train_batch_size}-ga{grad_accum}-fp32-noearly{max_epochs}ep"
    )


def build_stage3_train_configs(args):
    """Build one formal stage-3 local-window training ablation config.

    The only intended structural differences from the gd_residual_v1 cb64-r16
    anchor are local_num_blocks and if_remote_enabled.
    """

    variant_name = _infer_variant(args)
    variant = _VARIANTS[variant_name]
    base = _load_base_builder()
    kwargs, seed_value, data_seed, remote_read_topk_values = base._common_builder_kwargs(
        args,
        experiment_mode=getattr(args, "experiment_mode", None) or "train",
    )

    kwargs["local_num_blocks"] = int(variant["local_num_blocks"])
    kwargs["if_remote_enabled"] = bool(variant["if_remote_enabled"])
    kwargs["experiment_part"] = "20260529_flash_local_window_fairness"

    configs = build_configs(**kwargs)
    if len(configs) != 1:
        raise RuntimeError(f"Expected exactly 1 config for {variant_name}, got {len(configs)}")

    read_topk = _require_single(remote_read_topk_values, field_name="fox_remote_read_topk_values")
    explicit_run_id = str(getattr(args, "run_id", "") or "").strip()
    run_id = explicit_run_id or _default_run_id(
        args,
        variant_name=variant_name,
        seed_value=seed_value,
        data_seed=data_seed,
        remote_read_topk=read_topk,
    )
    return [_rewrite_run_id(configs[0], run_id=run_id)]
