#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

LOCAL_DIR = Path(__file__).resolve().parent
if str(LOCAL_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_DIR))

from common import BUILDERS, REPO_ROOT, VARIANTS, descriptor, load_json, training_descriptors
import experiment as experiment_module


BASE_PATH = REPO_ROOT / "zoology/experiments/flash_vqg/scripts/20260729-02-mqar-deterministic-selected-read-regression/evaluate.py"


def _load_base():
    sys.modules["experiment"] = experiment_module
    spec = importlib.util.spec_from_file_location("k2_mqar_evaluate_base", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load base evaluator: {BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_base()


def _source_from_result(result: dict[str, Any], role: str) -> dict[str, Any]:
    checkpoint = result[f"{role}_checkpoint"]
    precision = result["train_precision"]
    return {
        "source_id": f"3090-{result['variant']}-s{result['seed']}-{precision}-{role}",
        "machine": "3090",
        "model": "flash",
        "variant": result["variant"],
        "remat_mode": result["remat_mode"],
        "seed": int(result["seed"]),
        "train_precision": precision,
        "checkpoint_role": role,
        "checkpoint_path": checkpoint["path"],
        "checkpoint_file_sha256": checkpoint["file_sha256"],
        "checkpoint_model_state_sha256": checkpoint["model_state_sha256"],
    }


def sources(phase: str) -> list[dict[str, Any]]:
    descriptors = training_descriptors() if phase == "formal" else [
        descriptor(variant, 123) for variant in VARIANTS
    ]
    roles = ("last", "best") if phase == "formal" else ("last",)
    selected = []
    for row in descriptors:
        path = experiment_module.result_path(row["variant"], row["seed"], phase)
        result = load_json(path)
        if result.get("status") != "completed":
            raise RuntimeError(f"Training result is not complete: {path}")
        selected.extend(_source_from_result(result, role) for role in roles)
    return selected


_BASE_EVENT_PAYLOAD = BASE.event_payload


def event_payload(source: dict[str, Any], case, phase: str) -> dict[str, Any]:
    event = _BASE_EVENT_PAYLOAD(source, case, phase)
    precision = source["train_precision"]
    event["train_precision"] = precision
    event["eval_precision"] = precision
    if precision == "fp32":
        event["event_id"] = event["event_id"].replace("-bf16-", "-fp32-")
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


def evaluate(phase: str) -> dict[str, Any]:
    if phase not in {"screen", "formal", "diagnostic_fp32"}:
        raise ValueError(f"Unsupported evaluation phase: {phase}")
    return BASE.evaluate(phase)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=("screen", "formal", "diagnostic_fp32"), required=True
    )
    args = parser.parse_args()
    evaluate(args.phase)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
