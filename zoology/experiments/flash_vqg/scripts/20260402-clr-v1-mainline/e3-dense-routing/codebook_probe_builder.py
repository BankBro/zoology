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


def _require_env(name: str, *, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if value is None or str(value).strip() == "":
        raise RuntimeError(f"缺少必须环境变量: {name}")
    return str(value).strip()


def _parse_positive_ints(raw: str, *, env_name: str) -> list[int]:
    values: list[int] = []
    seen: set[int] = set()
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        parsed = int(token)
        if parsed <= 0:
            raise ValueError(f"{env_name} 只能包含正整数, 当前收到: {token}")
        if parsed not in seen:
            values.append(parsed)
            seen.add(parsed)
    if not values:
        raise ValueError(f"{env_name} 不能为空.")
    return values


def _parse_nonnegative_ints(raw: str, *, env_name: str) -> list[int]:
    values: list[int] = []
    seen: set[int] = set()
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        parsed = int(token)
        if parsed < 0:
            raise ValueError(f"{env_name} 只能包含非负整数, 当前收到: {token}")
        if parsed not in seen:
            values.append(parsed)
            seen.add(parsed)
    if not values:
        raise ValueError(f"{env_name} 不能为空.")
    return values


def build_e3_codebook_probe_configs(args):
    codebook_values = _parse_positive_ints(
        _require_env("E35_CODEBOOK_VALUES", default="64,256"),
        env_name="E35_CODEBOOK_VALUES",
    )
    seed_values = _parse_nonnegative_ints(
        _require_env("E35_SEED_VALUES", default="123,124"),
        env_name="E35_SEED_VALUES",
    )
    data_seed = int(_require_env("E35_DATA_SEED", default="123"))
    if data_seed < 0:
        raise ValueError(f"E35_DATA_SEED 必须是非负整数, 当前收到: {data_seed}")

    configs = []
    for num_codebook_vectors in codebook_values:
        for seed_value in seed_values:
            kwargs = _BASE_BUILDER._common_builder_kwargs(
                args,
                experiment_mode=f"dense_t025_cb{num_codebook_vectors}",
                vq_score_mode="codebook_dot",
                vq_weight_mode="dense_softmax",
                vq_update_mode="grad",
                vq_softmax_tau=0.25,
            )
            kwargs["seed_values"] = [seed_value]
            kwargs["data_seed"] = data_seed
            kwargs["num_codebook_vectors_values"] = [num_codebook_vectors]
            kwargs["fox_remote_read_topk"] = 2

            built = build_configs(**kwargs)
            if len(built) != 1:
                raise RuntimeError(
                    "Expected exactly 1 config for "
                    f"dense_t025_cb{num_codebook_vectors}, got {len(built)}"
                )

            run_id = (
                f"dense-t025-cb{num_codebook_vectors}-s{seed_value}-d{data_seed}"
            )
            configs.append(_BASE_BUILDER._rewrite_run_id(built[0], run_id=run_id))
    return configs
