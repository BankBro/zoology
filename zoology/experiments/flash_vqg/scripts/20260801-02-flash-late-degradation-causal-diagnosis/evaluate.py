#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

LOCAL_DIR = Path(__file__).resolve().parent
if str(LOCAL_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_DIR))

from causal_common import (
    EXPERIMENT_ID,
    LONGER_CASES,
    STANDARD_CASES,
    atomic_write_json,
    case_id,
    load_json,
    run_root,
    run_tag,
    utc_now,
)
import experiment


UPSTREAM_PATH = (
    experiment.UPSTREAM_DIR / "evaluate.py"
)


def _load_upstream():
    spec = importlib.util.spec_from_file_location("late_degradation_eval_upstream", UPSTREAM_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load upstream evaluator: {UPSTREAM_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


UPSTREAM = _load_upstream()
UPSTREAM.EXPERIMENT_ID = EXPERIMENT_ID
UPSTREAM.run_root = run_root
UPSTREAM.run_tag = run_tag


def _audit_flash_result(result: dict[str, Any], arm: str) -> None:
    states = result.get("model_runtime_state") or {}
    audits = [
        state["fox_gd_residual_triton_runtime_audit"]
        for state in states.values()
        if state.get("fox_gd_residual_triton_runtime_audit") is not None
    ]
    if not audits:
        raise RuntimeError("Flash evaluation runtime audit is missing.")
    keys = ("grouped_fallbacks", "selected_fallbacks", "persistent_fallbacks")
    selected = sum(int(audit.get("selected_calls", 0)) for audit in audits)
    persistent = sum(int(audit.get("persistent_calls", 0)) for audit in audits)
    fallbacks = sum(int(audit.get(key, 0)) for audit in audits for key in keys)
    expected_persistent = arm.startswith("fastest-")
    if selected <= 0 or fallbacks or ((persistent > 0) != expected_persistent):
        raise RuntimeError(f"Flash evaluation runtime audit failed: {audits}")


UPSTREAM._audit_flash_result = _audit_flash_result


def source_from_result(result: dict[str, Any], role: str) -> dict[str, Any]:
    checkpoint = result[f"{role}_checkpoint"]
    return {
        "source_id": f"3090-{result['arm']}-s{result['seed']}-bf16-{role}",
        "machine": "3090",
        "arm": result["arm"],
        "model": "flash",
        "seed": int(result["seed"]),
        "train_precision": "bf16",
        "checkpoint_role": role,
        "checkpoint_path": checkpoint["path"],
        "checkpoint_file_sha256": checkpoint["file_sha256"],
        "checkpoint_model_state_sha256": checkpoint["model_state_sha256"],
    }


def sources(selection: dict[str, Any]) -> list[dict[str, Any]]:
    selected = []
    for arm in selection["formal_arms"]:
        for seed in selection["seeds"]:
            path = experiment.result_path(arm, seed, "formal", "fixed")
            result = load_json(path)
            if result.get("status") != "completed":
                raise RuntimeError(f"Formal result is incomplete: {path}")
            selected.extend(source_from_result(result, role) for role in ("best", "last"))
    return selected


def initial_batch_size(sequence_length: int) -> int:
    if sequence_length <= 1024:
        return 64
    if sequence_length <= 2048:
        return 32
    if sequence_length <= 4096:
        return 16
    return 8


def run_case(source: dict[str, Any], case) -> tuple[dict[str, Any], int]:
    batch_size = initial_batch_size(case[0])
    while batch_size >= 1:
        event = UPSTREAM.event_payload(source, case, batch_size, "formal-causal")
        result = UPSTREAM.run_event(event)
        if result.get("status") == "completed":
            return result, batch_size
        if result.get("status") != "oom":
            raise RuntimeError(f"Evaluation failed: {result}")
        batch_size //= 2
    raise RuntimeError(f"No feasible evaluation batch for {source['source_id']} {case_id(case)}")


def result_record(source: dict[str, Any], case, result: dict[str, Any], batch: int) -> dict[str, Any]:
    return {
        "source_id": source["source_id"],
        "arm": source["arm"],
        "seed": source["seed"],
        "checkpoint_role": source["checkpoint_role"],
        "checkpoint_model_state_sha256": source["checkpoint_model_state_sha256"],
        "case": case_id(case),
        "input_seq_len": case[0],
        "num_kv_pairs": case[1],
        "num_examples": case[2],
        "batch_size": batch,
        "accuracy": result["accuracy"],
        "loss": result["loss_sample_weighted"],
        "dataset_hash": result["dataset_hash"],
        "wall_clock_sec": result["wall_clock_sec"],
    }


def invariance_check(source: dict[str, Any], case, batch: int, result: dict[str, Any]) -> dict[str, Any]:
    smaller = max(1, batch // 2)
    if smaller == batch:
        return {"passed": True, "skipped": True}
    event = UPSTREAM.event_payload(source, case, smaller, "invariance-causal")
    other = UPSTREAM.run_event(event)
    if other.get("status") != "completed":
        raise RuntimeError(f"Invariance evaluation failed: {other}")
    return UPSTREAM._compare_invariance(result, other)


def evaluate(selection_path: Path) -> dict[str, Any]:
    selection = load_json(selection_path)
    all_sources = sources(selection)
    cases = STANDARD_CASES + LONGER_CASES
    records = []
    invariance = []
    checked_arms: set[str] = set()
    for source in all_sources:
        for case in cases:
            result, batch = run_case(source, case)
            records.append(result_record(source, case, result, batch))
            should_check = (
                source["arm"] not in checked_arms
                and source["checkpoint_role"] == "last"
                and source["seed"] == selection["seeds"][0]
                and case in (STANDARD_CASES[-1], LONGER_CASES[-1])
            )
            if should_check:
                invariance.append(
                    {
                        "arm": source["arm"],
                        "case": case_id(case),
                        "result": invariance_check(source, case, batch, result),
                    }
                )
                if case == LONGER_CASES[-1]:
                    checked_arms.add(source["arm"])
    output = run_root() / "evaluation" / "formal-summary.json"
    payload = {
        "status": "completed",
        "experiment_id": EXPERIMENT_ID,
        "completed_at_utc": utc_now(),
        "selection": selection,
        "records": records,
        "invariance": invariance,
    }
    atomic_write_json(output, payload)
    csv_path = run_root() / "evaluation" / "formal-metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    args = parser.parse_args()
    payload = evaluate(args.selection)
    print(json.dumps({"status": payload["status"], "records": len(payload["records"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
