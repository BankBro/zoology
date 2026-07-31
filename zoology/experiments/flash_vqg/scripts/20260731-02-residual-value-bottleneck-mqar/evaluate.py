#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

LOCAL_DIR = Path(__file__).resolve().parent
if str(LOCAL_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_DIR))

from common import REPO_ROOT, VARIANTS, descriptor, load_json
import experiment as experiment_module


BASE_PATH = REPO_ROOT / "zoology/experiments/flash_vqg/scripts/20260731-01-selected-read-warp-mqar-screen/evaluate.py"


def _load_base():
    previous = sys.modules.get("experiment")
    sys.modules["experiment"] = experiment_module
    try:
        spec = importlib.util.spec_from_file_location("residual_value_evaluate_base", BASE_PATH)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load evaluator: {BASE_PATH}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            sys.modules.pop("experiment", None)
        else:
            sys.modules["experiment"] = previous


BASE = _load_base()


def _source_from_result(result: dict[str, Any]) -> dict[str, Any]:
    checkpoint = result["last_checkpoint"]
    return {
        "source_id": f"3090-{result['variant']}-s{result['seed']}-bf16-last",
        "machine": "3090",
        "model": "flash",
        "variant": result["variant"],
        "residual_value_dim": int(result["residual_value_dim"]),
        "remat_mode": "post_phase1",
        "seed": int(result["seed"]),
        "train_precision": "bf16",
        "checkpoint_role": "last",
        "checkpoint_path": checkpoint["path"],
        "checkpoint_file_sha256": checkpoint["file_sha256"],
        "checkpoint_model_state_sha256": checkpoint["model_state_sha256"],
    }


def sources(phase: str) -> list[dict[str, Any]]:
    if phase != "screen":
        raise ValueError(f"Unsupported phase: {phase}.")
    selected = []
    for variant in VARIANTS:
        row = descriptor(variant, 123)
        path = experiment_module.result_path(variant, row["seed"], phase)
        result = load_json(path)
        if result.get("status") != "completed":
            raise RuntimeError(f"Training result is incomplete: {path}")
        selected.append(_source_from_result(result))
    return selected


_BASE_EVENT_PAYLOAD = BASE.event_payload


def event_payload(source: dict[str, Any], case, phase: str) -> dict[str, Any]:
    event = _BASE_EVENT_PAYLOAD(source, case, phase)
    event["train_precision"] = "bf16"
    event["eval_precision"] = "bf16"
    event["residual_value_dim"] = source["residual_value_dim"]
    return event


BASE.BASE.sources = sources
BASE.BASE.event_payload = event_payload


def main() -> int:
    BASE.BASE.evaluate("screen")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
