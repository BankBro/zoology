# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from zoology.experiments.flash_vqg.flash_vqg_suite import build_configs
from zoology.experiments.flash_vqg.manifest import (
    load_manifest,
    resolve_best_checkpoint_from_manifest,
)


_BASE_BUILDER_PATH = Path(__file__).resolve().parents[1] / "e3-dense-routing" / "config_builder.py"
_SPEC = importlib.util.spec_from_file_location(
    "flash_vqg_e5_train_base_builder",
    _BASE_BUILDER_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"无法加载 E3 base builder: {_BASE_BUILDER_PATH}")
_BASE_BUILDER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE_BUILDER)


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        raise RuntimeError(f"缺少必须环境变量: {name}")
    return str(value).strip()


def _get_flash_vqg_kwargs(config) -> dict:
    configs = config.model.sequence_mixer.kwargs.get("configs", [])
    for mixer_cfg in configs:
        if mixer_cfg.get("name") == "zoology.mixers.flash_vqg.FlashVQGMixer":
            kwargs = mixer_cfg.get("kwargs")
            if isinstance(kwargs, dict):
                return kwargs
    raise RuntimeError("未在 sequence_mixer 中找到 FlashVQGMixer 配置.")


def build_e5_train_single_config(args):
    source_launch_id = _require_env("E5TRAIN_SOURCE_LAUNCH_ID")
    source_run_id = _require_env("E5TRAIN_SOURCE_RUN_ID")
    lambda_value = float(_require_env("E5TRAIN_LAMBDA"))
    lambda_tag = _require_env("E5TRAIN_LAMBDA_TAG")
    run_id = _require_env("E5TRAIN_RUN_ID")
    row_weight_mode = os.environ.get("E5TRAIN_ROW_WEIGHT_MODE", "uniform").strip().lower()
    warmup_steps = int(os.environ.get("E5TRAIN_WARMUP_STEPS", "200"))

    if row_weight_mode not in {"uniform", "adv_relu"}:
        raise RuntimeError(
            f"E5TRAIN_ROW_WEIGHT_MODE 只能是 uniform 或 adv_relu, 当前收到: {row_weight_mode}"
        )
    if warmup_steps < 0:
        raise RuntimeError(
            f"E5TRAIN_WARMUP_STEPS 必须是非负整数, 当前收到: {warmup_steps}"
        )
    if lambda_value < 0.0:
        raise RuntimeError(f"E5TRAIN_LAMBDA 必须是非负数, 当前收到: {lambda_value}")

    source_manifest = load_manifest(source_launch_id)
    init_checkpoint_path = resolve_best_checkpoint_from_manifest(source_manifest, source_run_id)

    kwargs = _BASE_BUILDER._common_builder_kwargs(
        args,
        experiment_mode=f"e5_train_dense_teacher_l{lambda_tag}",
        vq_score_mode="codebook_dot",
        vq_weight_mode="dense_softmax",
        vq_update_mode="grad",
        vq_softmax_tau=0.25,
    )

    configs = build_configs(**kwargs)
    if len(configs) != 1:
        raise RuntimeError(f"Expected exactly 1 config for e5-train, got {len(configs)}")

    config = _BASE_BUILDER._rewrite_run_id(configs[0], run_id=run_id)
    config.max_epochs = int(os.environ.get("E5TRAIN_MAX_EPOCHS", str(args.max_epochs)))
    config.init_checkpoint_path = str(init_checkpoint_path)
    config.init_checkpoint_source_launch_id = source_launch_id
    config.init_checkpoint_source_run_id = source_run_id
    config.init_checkpoint_strict = True

    flash_kwargs = _get_flash_vqg_kwargs(config)
    is_control = abs(float(lambda_value)) < 1e-12
    experiment_mode = (
        f"control_l{lambda_tag}"
        if is_control
        else f"dense_teacher_l{lambda_tag}_{row_weight_mode}"
    )
    flash_kwargs.update(
        {
            "experiment_part": "e5_train_dense_teacher",
            "experiment_mode": experiment_mode,
            "fox_dense_teacher_mode": "dense_value",
            "fox_dense_teacher_loss_mode": "sparse_top2_ce",
            "fox_dense_teacher_layer_idx": 1,
            "fox_dense_teacher_lambda": float(lambda_value),
            "fox_dense_teacher_tau_teacher": 1.0,
            "fox_dense_teacher_row_weight_mode": row_weight_mode,
            "fox_dense_teacher_warmup_steps": int(warmup_steps),
        }
    )
    return [config]
