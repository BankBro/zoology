# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from zoology.experiments.flash_vqg.flash_vqg_suite import build_configs

_BASE_BUILDER_PATH = Path(__file__).with_name("config_builder.py")
_SPEC = importlib.util.spec_from_file_location(
    "flash_vqg_e3_dense_base_builder",
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


def _parse_remote_read_mode(raw: str) -> int | None:
    normalized = str(raw).strip().lower()
    if normalized in {"", "dense", "none", "null"}:
        return None
    try:
        parsed = int(normalized)
    except ValueError as exc:
        raise RuntimeError(
            f"E3_REMOTE_READ_MODE 只能是 dense 或正整数, 当前收到: {raw}"
        ) from exc
    if parsed <= 0:
        raise RuntimeError(
            f"E3_REMOTE_READ_MODE 只能是 dense 或正整数, 当前收到: {raw}"
        )
    return parsed


def build_e3_tau_single_config(args):
    tau_value = float(_require_env("E3_TAU_VALUE"))
    tau_tag = os.environ.get("E3_TAU_TAG", "").strip() or str(tau_value).replace(".", "p")
    run_id = os.environ.get("E3_TAU_RUN_ID", "").strip() or f"dense-t{tau_tag}"
    experiment_mode = (
        os.environ.get("E3_TAU_EXPERIMENT_MODE", "").strip()
        or f"dense_t{tau_tag}"
    )
    seed_value = int(os.environ.get("E3_TAU_SEED", "123"))
    data_seed = int(os.environ.get("E3_TAU_DATA_SEED", "123"))
    remote_read_topk = _parse_remote_read_mode(os.environ.get("E3_REMOTE_READ_MODE", "2"))

    kwargs = _BASE_BUILDER._common_builder_kwargs(
        args,
        experiment_mode=experiment_mode,
        vq_score_mode="codebook_dot",
        vq_weight_mode="dense_softmax",
        vq_update_mode="grad",
        vq_softmax_tau=tau_value,
    )
    kwargs["seed_values"] = [seed_value]
    kwargs["data_seed"] = data_seed
    kwargs["fox_remote_read_topk"] = remote_read_topk

    configs = build_configs(**kwargs)
    if len(configs) != 1:
        raise RuntimeError(
            f"Expected exactly 1 config for {experiment_mode}, got {len(configs)}"
        )
    return [_BASE_BUILDER._rewrite_run_id(configs[0], run_id=run_id)]
