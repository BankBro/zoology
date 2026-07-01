#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get("ZOOLOGY_REPO_ROOT", "/home/lyj/mnt/project/zoology")).resolve()
SOURCE_SCRIPT = (
    REPO_ROOT
    / "zoology/experiments/flash_vqg/scripts/"
    / "20260701-02-flash-vqg-default-dropout-amplifier-trace/default_dropout_amplifier_trace.py"
)
EXPERIMENT_ID = "20260701-03-flash-vqg-default-dropout-1ep-bridge-trace"
TRACE_STEPS = [0, 16, 64, 128, 256, 384, 512, 704]


def _load_source():
    spec = importlib.util.spec_from_file_location("default_dropout_amplifier_trace_base", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load source script: {SOURCE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_source()

BASE.SCRIPT_DIR = SCRIPT_DIR
BASE.EXPERIMENT_ID = EXPERIMENT_ID
BASE.ARTIFACT_DIR = REPO_ROOT / "docs/artifacts" / EXPERIMENT_ID
BASE.TRACE_TRAIN_STEPS = list(TRACE_STEPS)
BASE.DEFAULT_CAPTURE_STEPS = ",".join(str(step) for step in TRACE_STEPS)
BASE.DEFAULT_MAX_EPOCHS = 1
BASE.DEFAULT_MAX_TRAIN_STEPS = 704
BASE.TARGETS = ("default-r2", "default-r4", "dropout005-r4")
BASE.BASE.EXPECTED_TOTAL_OPTIMIZER_STEPS = BASE.BASE.EXPECTED_STEPS_PER_EPOCH * BASE.DEFAULT_MAX_EPOCHS
BASE._patch_base()


if __name__ == "__main__":
    raise SystemExit(BASE.main())

