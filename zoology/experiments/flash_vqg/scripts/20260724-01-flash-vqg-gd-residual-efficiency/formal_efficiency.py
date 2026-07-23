#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import torch


EXPERIMENT_ID = "20260724-01-flash-vqg-gd-residual-efficiency"
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get("ZOOLOGY_REPO_ROOT", "/home/lyj/mnt/project/zoology")).resolve()
SOURCE = (
    REPO_ROOT
    / "zoology/experiments/flash_vqg/scripts/"
    / "20260709-01-flash-vqg-default-dropout-r16-support-confidence-screen/"
    / "support_confidence_screen.py"
)
ARTIFACT_DIR = REPO_ROOT / "docs/artifacts" / EXPERIMENT_ID
TARGETS = ("s125-baseline-r16-joint", "s124-baseline-r16-joint")
SEEDS = (125, 124)
BACKEND_SETTINGS = {
    "fox_gd_residual_grouped_chunk_backend": "triton",
    "fox_gd_residual_selected_read_backend": "triton_remat",
    "fox_gd_residual_selected_read_chunk_size": 2048,
}


def _load_source():
    spec = importlib.util.spec_from_file_location("efficiency_formal_base", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load source script: {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_source()


def _configure_base() -> None:
    variants: dict[str, dict[str, Any]] = {}
    for target in TARGETS:
        source_spec = BASE.VARIANTS[target]
        variants[target] = {**source_spec, **BACKEND_SETTINGS}

    BASE.EXPERIMENT_ID = EXPERIMENT_ID
    BASE.SCRIPT_DIR = SCRIPT_DIR
    BASE.ARTIFACT_DIR = ARTIFACT_DIR
    BASE.METRICS_YAML = SOURCE.parent / "metrics.yaml"
    BASE.SEEDS = SEEDS
    BASE.VARIANT_NAMES = ("baseline-r16-joint",)
    BASE.TARGETS = TARGETS
    BASE.VARIANTS = variants
    BASE._FLASH_EXTRA_KEYS = tuple(
        dict.fromkeys((*BASE._FLASH_EXTRA_KEYS, *BACKEND_SETTINGS.keys()))
    )


def main() -> int:
    os.environ.setdefault("TRITON_F32_DEFAULT", "ieee")
    os.environ["FLASH_VQG_READ_TRACE_MODE"] = "disabled"
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    _configure_base()
    return int(BASE.main())


if __name__ == "__main__":
    raise SystemExit(main())
