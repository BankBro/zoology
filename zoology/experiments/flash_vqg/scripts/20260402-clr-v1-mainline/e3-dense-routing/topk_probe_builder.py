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


def _parse_topk_values(raw: str) -> list[int]:
    values: list[int] = []
    seen: set[int] = set()
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        parsed = int(token)
        if parsed < 2:
            raise ValueError(f"topk_softmax 要求 vq_topk >= 2, 当前收到: {parsed}")
        if parsed not in seen:
            values.append(parsed)
            seen.add(parsed)
    if not values:
        raise ValueError("E3_TOPK_VALUES 不能为空.")
    return values


def _tau_tag_from_value(tau_value: float) -> str:
    text = str(tau_value).strip()
    if "." in text:
        whole, frac = text.split(".", 1)
        frac = frac.rstrip("0")
        if frac:
            return f"{whole}{frac}"
        return whole
    return text


def build_e3_topk_probe_configs(args):
    topk_values = _parse_topk_values(_require_env("E3_TOPK_VALUES", default="2,4"))
    tau_value = float(_require_env("E3_TOPK_TAU", default="0.25"))
    tau_tag = _require_env("E3_TOPK_TAU_TAG", default=_tau_tag_from_value(tau_value))
    seed_value = int(_require_env("E3_TOPK_SEED", default="123"))
    data_seed = int(_require_env("E3_TOPK_DATA_SEED", default="123"))

    configs = []
    for topk_value in topk_values:
        kwargs = _BASE_BUILDER._common_builder_kwargs(
            args,
            experiment_mode=f"topk_write_t{tau_tag}_k{topk_value}",
            vq_score_mode="codebook_dot",
            vq_weight_mode="topk_softmax",
            vq_update_mode="grad",
            vq_softmax_tau=tau_value,
        )
        kwargs["seed_values"] = [seed_value]
        kwargs["data_seed"] = data_seed
        kwargs["vq_topk"] = int(topk_value)

        built = build_configs(**kwargs)
        if len(built) != 1:
            raise RuntimeError(
                "Expected exactly 1 config for "
                f"topk_write_t{tau_tag}_k{topk_value}, got {len(built)}"
            )
        run_id = f"topk{topk_value}-t{tau_tag}-s{seed_value}-d{data_seed}"
        configs.append(_BASE_BUILDER._rewrite_run_id(built[0], run_id=run_id))
    return configs
