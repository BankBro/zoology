# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from zoology.experiments.flash_vqg.flash_vqg_suite import build_configs

_E3_BUILDER_PATH = Path(__file__).parents[1] / "e3-dense-routing" / "config_builder.py"
_SPEC = importlib.util.spec_from_file_location(
    "flash_vqg_e5_e3_base_builder",
    _E3_BUILDER_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"无法加载 E3 base builder: {_E3_BUILDER_PATH}")
_E3_BUILDER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_E3_BUILDER)


def _int_env(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _rewrite_run_id(config, *, run_id: str):
    config = config.model_copy(deep=True) if hasattr(config, "model_copy") else config.copy(deep=True)
    config.run_id = run_id
    return config


def _build_single(
    args,
    *,
    experiment_mode: str,
    run_id: str,
    retrieval_loss_enabled: bool,
    retrieval_loss_lambda: float,
    retrieval_loss_tau: float,
):
    kwargs = _E3_BUILDER._common_builder_kwargs(
        args,
        experiment_mode=experiment_mode,
        vq_score_mode="codebook_dot",
        vq_weight_mode="dense_softmax",
        vq_update_mode="grad",
        vq_softmax_tau=0.25,
    )
    kwargs.update(
        experiment_part="e5_retaware",
        experiment_mode=experiment_mode,
        seed_values=[_int_env("E5_SEED", 123)],
        data_seed=_int_env("E5_DATA_SEED", 123),
        retrieval_loss_enabled=bool(retrieval_loss_enabled),
        retrieval_loss_lambda=float(retrieval_loss_lambda),
        retrieval_loss_tau=float(retrieval_loss_tau),
    )
    configs = build_configs(**kwargs)
    if len(configs) != 1:
        raise RuntimeError(f"Expected exactly 1 config for {experiment_mode}, got {len(configs)}")
    return _rewrite_run_id(configs[0], run_id=run_id)


def build_e5_train_configs(args):
    seed = _int_env("E5_SEED", 123)
    data_seed = _int_env("E5_DATA_SEED", 123)
    suffix = f"s{seed}-d{data_seed}"
    return [
        _build_single(
            args,
            experiment_mode="e5_retoff",
            run_id=f"dense-t025-retoff-{suffix}",
            retrieval_loss_enabled=False,
            retrieval_loss_lambda=0.0,
            retrieval_loss_tau=1.0,
        ),
        _build_single(
            args,
            experiment_mode="e5_retl002_t050",
            run_id=f"dense-t025-retl002-t050-{suffix}",
            retrieval_loss_enabled=True,
            retrieval_loss_lambda=0.02,
            retrieval_loss_tau=0.5,
        ),
        _build_single(
            args,
            experiment_mode="e5_retl002_t100",
            run_id=f"dense-t025-retl002-t100-{suffix}",
            retrieval_loss_enabled=True,
            retrieval_loss_lambda=0.02,
            retrieval_loss_tau=1.0,
        ),
        _build_single(
            args,
            experiment_mode="e5_retl005_t100",
            run_id=f"dense-t025-retl005-t100-{suffix}",
            retrieval_loss_enabled=True,
            retrieval_loss_lambda=0.05,
            retrieval_loss_tau=1.0,
        ),
    ]


def build_e5_reton_train_configs(args):
    return build_e5_train_configs(args)[1:]


def build_e5_smoke_configs(args):
    seed = _int_env("E5_SEED", 123)
    data_seed = _int_env("E5_DATA_SEED", 123)
    suffix = f"s{seed}-d{data_seed}"
    configs = [
        _build_single(
            args,
            experiment_mode="e5_smoke_retoff",
            run_id=f"dense-t025-retoff-smoke-{suffix}",
            retrieval_loss_enabled=False,
            retrieval_loss_lambda=0.0,
            retrieval_loss_tau=1.0,
        ),
        _build_single(
            args,
            experiment_mode="e5_smoke_retl002_t100",
            run_id=f"dense-t025-retl002-t100-smoke-{suffix}",
            retrieval_loss_enabled=True,
            retrieval_loss_lambda=0.02,
            retrieval_loss_tau=1.0,
        ),
    ]
    for config in configs:
        config.data.train_configs = config.data.train_configs[:1]
        config.data.test_configs = config.data.test_configs[:1]
        config.data.train_configs[0].num_examples = 8
        config.data.test_configs[0].num_examples = 4
        config.data.cache_dir = None
        config.data.force_cache = False
    return configs
