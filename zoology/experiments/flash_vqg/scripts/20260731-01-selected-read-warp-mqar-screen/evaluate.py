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


BASE_PATH = REPO_ROOT / "zoology/experiments/flash_vqg/scripts/20260729-02-mqar-deterministic-selected-read-regression/evaluate.py"


def _load_base():
    previous_experiment = sys.modules.get("experiment")
    sys.modules["experiment"] = experiment_module
    try:
        spec = importlib.util.spec_from_file_location("selected_warp_mqar_evaluate_base", BASE_PATH)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load evaluator: {BASE_PATH}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if previous_experiment is None:
            sys.modules.pop("experiment", None)
        else:
            sys.modules["experiment"] = previous_experiment


BASE = _load_base()


def _source_from_result(result: dict[str, Any]) -> dict[str, Any]:
    checkpoint = result["last_checkpoint"]
    return {
        "source_id": f"3090-{result['variant']}-s{result['seed']}-bf16-last",
        "machine": "3090",
        "model": "flash",
        "variant": result["variant"],
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
        path = experiment_module.result_path(row["variant"], row["seed"], phase)
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
    return event


def runtime_audit(result: dict[str, Any]) -> None:
    states = result.get("model_runtime_state") or {}
    audits = [
        value.get("fox_gd_residual_triton_runtime_audit")
        for value in states.values()
        if value.get("fox_gd_residual_triton_runtime_audit") is not None
    ]
    if not audits:
        raise RuntimeError("Evaluation did not record Flash Triton runtime audit.")
    fallback_keys = ("grouped_fallbacks", "selected_fallbacks", "persistent_fallbacks")
    for audit in audits:
        if int(audit.get("selected_calls", 0)) <= 0:
            raise RuntimeError(f"Evaluation missed selected Triton calls: {audit}")
        if any(int(audit.get(key, 0)) for key in fallback_keys):
            raise RuntimeError(f"Evaluation recorded a fallback: {audit}")
        if "actual_core_dtype" in audit and audit["actual_core_dtype"] != "float32":
            raise RuntimeError(f"Evaluation core dtype mismatch: {audit}")


BASE.sources = sources
BASE._runtime_audit = runtime_audit
BASE.event_payload = event_payload


def main() -> int:
    BASE.evaluate("screen")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
