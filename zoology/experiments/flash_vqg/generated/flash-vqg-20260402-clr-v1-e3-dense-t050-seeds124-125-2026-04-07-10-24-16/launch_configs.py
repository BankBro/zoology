# -*- coding: utf-8 -*-
# 此文件由临时 seed scan 脚本自动生成.
# 目标: 仅复用 e3 dense-t050, 追加 seed 124 和 125, 单 GPU 顺序执行.

import argparse
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "prev_e3_launch",
    r"/home/lyj/mnt/project/zoology/zoology/experiments/flash_vqg/generated/flash-vqg-20260402-clr-v1-e3-2026-04-06-20-01-22/launch_configs.py",
)
_prev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prev)


def _build_dense_t050_for_seed(seed: int, run_id: str):
    base = vars(_prev._builder_args).copy()
    base.update({
        "seed_values": str(seed),
        "launch_id_prefix": "flash-vqg-20260402-clr-v1-e3-dense-t050-seeds124-125",
    })
    args = argparse.Namespace(**base)
    configs = _prev._builder(args)
    target = next(config for config in configs if config.run_id == "dense-t050")
    target = target.model_copy(deep=True) if hasattr(target, "model_copy") else target.copy(deep=True)
    target.run_id = run_id
    return target


configs = [
    _build_dense_t050_for_seed(124, "dense-t050-s124"),
    _build_dense_t050_for_seed(125, "dense-t050-s125"),
]
