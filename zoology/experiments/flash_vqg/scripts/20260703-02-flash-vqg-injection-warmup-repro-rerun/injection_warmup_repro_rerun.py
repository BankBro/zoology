#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get("ZOOLOGY_REPO_ROOT", "/home/lyj/mnt/project/zoology")).resolve()
SOURCE_SCRIPT = (
    REPO_ROOT
    / "zoology/experiments/flash_vqg/scripts/"
    / "20260702-03-flash-vqg-injection-warmup-screen/injection_warmup_screen.py"
)
EXPERIMENT_ID = "20260703-02-flash-vqg-injection-warmup-repro-rerun"


def _load_source():
    spec = importlib.util.spec_from_file_location("injection_warmup_screen_source", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load source script: {SOURCE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SRC = _load_source()
TARGETS = SRC.TARGETS
VARIANTS = SRC.VARIANTS
BASEMOD = SRC.BASEMOD
ARTIFACT_DIR = REPO_ROOT / "docs/artifacts" / EXPERIMENT_ID
_ORIGINAL_BUILD_CONFIG = SRC.BASEMOD.BASEMOD.BASE.build_config


def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _pop_global_read_trace_flag(argv: list[str]) -> tuple[list[str], bool]:
    cleaned: list[str] = []
    enabled = _truthy(os.environ.get("FLASH_VQG_ENABLE_READ_TRACE"))
    for arg in argv:
        if arg == "--enable-read-trace":
            enabled = True
            continue
        if arg == "--disable-read-trace":
            enabled = False
            continue
        cleaned.append(arg)
    return cleaned, enabled


def _patch_experiment_identity() -> None:
    SRC.EXPERIMENT_ID = EXPERIMENT_ID
    SRC.ARTIFACT_DIR = REPO_ROOT / "docs/artifacts" / EXPERIMENT_ID
    SRC.BASEMOD.EXPERIMENT_ID = EXPERIMENT_ID
    SRC.BASEMOD.ARTIFACT_DIR = SRC.ARTIFACT_DIR
    SRC.BASEMOD.BASEMOD.EXPERIMENT_ID = EXPERIMENT_ID
    SRC.BASEMOD.BASEMOD.ARTIFACT_DIR = SRC.ARTIFACT_DIR
    SRC.BASEMOD.BASEMOD.BASE.EXPERIMENT_ID = EXPERIMENT_ID


def _disable_read_trace(config: Any) -> None:
    config.read_trace_enabled = False
    config.read_trace_train_steps = []
    config.read_trace_valid_batches = []
    config.read_trace_output_dir = None
    config.read_churn_probe_enabled = False
    config.read_churn_probe_valid_batches = []
    config.train_inline_event_trace_enabled = False
    config.train_inline_event_trace_steps = []
    config.train_inline_event_trace_output_dir = None


def _patch_read_trace_mode(*, enable_read_trace: bool) -> None:
    def build_config(*args: Any, **kwargs: Any):
        config = _ORIGINAL_BUILD_CONFIG(*args, **kwargs)
        if not enable_read_trace:
            _disable_read_trace(config)
        else:
            config.train_inline_event_trace_enabled = False
            config.train_inline_event_trace_steps = []
            config.train_inline_event_trace_output_dir = None
        return config

    SRC.BASEMOD.BASEMOD.BASE.build_config = build_config


def main() -> int:
    argv, enable_read_trace = _pop_global_read_trace_flag(sys.argv[1:])
    _patch_experiment_identity()
    _patch_read_trace_mode(enable_read_trace=enable_read_trace)
    os.environ["FLASH_VQG_READ_TRACE_MODE"] = "enabled" if enable_read_trace else "disabled"
    sys.argv = [sys.argv[0], *argv]
    return SRC.BASEMOD.BASEMOD.main()


if __name__ == "__main__":
    raise SystemExit(main())
