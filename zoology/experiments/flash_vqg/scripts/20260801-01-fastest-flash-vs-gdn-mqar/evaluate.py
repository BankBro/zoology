#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

LOCAL_DIR = Path(__file__).resolve().parent
if str(LOCAL_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_DIR))

from common import (
    ARMS,
    BATCH_CANDIDATES,
    EVAL_CASES,
    EXPERIMENT_ID,
    FASTEST,
    FLASH_ROOT,
    GDN,
    LONGER_CASES,
    PYTHON,
    REPO_ROOT,
    STANDARD_CASES,
    atomic_write_json,
    case_id,
    descriptor,
    load_json,
    run_root,
    run_tag,
    safe_name,
    stable_json_sha256,
    training_descriptors,
    utc_now,
)
import experiment


EVAL_EVENT_SCRIPT = (
    REPO_ROOT
    / "zoology/experiments/flash_vqg/scripts"
    / "20260726-01-mqar-precision-profile"
    / "eval_event.py"
)
SCREEN_CASES = (STANDARD_CASES[-1],) + LONGER_CASES[1:]
SMOKE_CASES = (STANDARD_CASES[-1], LONGER_CASES[-1])


def git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def source_from_result(result: dict[str, Any], role: str) -> dict[str, Any]:
    checkpoint = result[f"{role}_checkpoint"]
    return {
        "source_id": f"3090-{result['arm']}-s{result['seed']}-bf16-{role}",
        "machine": "3090",
        "arm": result["arm"],
        "model": result["model"],
        "seed": int(result["seed"]),
        "train_precision": "bf16",
        "checkpoint_role": role,
        "checkpoint_path": checkpoint["path"],
        "checkpoint_file_sha256": checkpoint["file_sha256"],
        "checkpoint_model_state_sha256": checkpoint["model_state_sha256"],
    }


def sources(phase: str) -> list[dict[str, Any]]:
    descriptors = training_descriptors("formal" if phase == "formal" else phase)
    roles = ("last", "best") if phase == "formal" else ("last",)
    selected = []
    for row in descriptors:
        path = experiment.result_path(row["arm"], row["seed"], phase)
        result = load_json(path)
        if result.get("status") != "completed":
            raise RuntimeError(f"Training result is incomplete: {path}")
        selected.extend(source_from_result(result, role) for role in roles)
    return selected


def event_payload(
    source: dict[str, Any],
    case: tuple[int, int, int, str | None],
    batch_size: int,
    mode: str,
    *,
    max_batches: int = 0,
) -> dict[str, Any]:
    sequence_length, num_kv_pairs, num_examples, dataset_hash = case
    zoology_commit = git_commit(REPO_ROOT)
    flash_commit = git_commit(FLASH_ROOT)
    state_short = source["checkpoint_model_state_sha256"][:16]
    event_id = (
        f"{mode}-{source['source_id']}-{case_id(case)}-b{batch_size}-"
        f"{state_short}-z{zoology_commit[:8]}-f{flash_commit[:8]}"
    )
    return {
        "experiment_id": EXPERIMENT_ID,
        "run_tag": run_tag(),
        "zoology_commit": zoology_commit,
        "flash_commit": flash_commit,
        "event_id": event_id,
        "mode": mode,
        "machine": "3090",
        "arm": source["arm"],
        "model": source["model"],
        "seed": source["seed"],
        "checkpoint_role": source["checkpoint_role"],
        "train_precision": "bf16",
        "eval_precision": "bf16",
        "checkpoint_path": source["checkpoint_path"],
        "checkpoint_file_sha256": source["checkpoint_file_sha256"],
        "checkpoint_model_state_sha256": source["checkpoint_model_state_sha256"],
        "input_seq_len": sequence_length,
        "num_kv_pairs": num_kv_pairs,
        "num_examples": num_examples,
        "eval_batch_size": int(batch_size),
        "eval_seed": 123,
        "expected_dataset_hash": dataset_hash,
        "dataset_policy": "generated_seeded",
        "max_batches": int(max_batches),
        "controlled_interrupt_after_batches": 0,
    }


def _event_paths(mode: str, event_id: str) -> dict[str, Path]:
    root = run_root() / "evaluation" / mode / safe_name(event_id)
    return {
        "event": root / "event.json",
        "progress": root / "progress.json",
        "result": root / "result.json",
        "log": root / "event.log",
    }


def _audit_flash_result(result: dict[str, Any], arm: str) -> None:
    states = result.get("model_runtime_state") or {}
    audits = [
        state["fox_gd_residual_triton_runtime_audit"]
        for state in states.values()
        if state.get("fox_gd_residual_triton_runtime_audit") is not None
    ]
    if not audits:
        raise RuntimeError("Flash evaluation runtime audit is missing.")
    fallback_keys = ("grouped_fallbacks", "selected_fallbacks", "persistent_fallbacks")
    selected = sum(int(audit.get("selected_calls", 0)) for audit in audits)
    persistent = sum(int(audit.get("persistent_calls", 0)) for audit in audits)
    fallbacks = sum(int(audit.get(key, 0)) for audit in audits for key in fallback_keys)
    if selected <= 0 or fallbacks:
        raise RuntimeError(f"Flash evaluation runtime audit failed: {audits}")
    if (persistent > 0) != (arm == FASTEST):
        raise RuntimeError(f"Unexpected persistent evaluation path: {audits}")


def _audit_result(result: dict[str, Any], event: dict[str, Any]) -> None:
    if result.get("status") != "completed":
        raise RuntimeError(f"Evaluation is incomplete: {result}")
    if event["model"] != GDN:
        _audit_flash_result(result, event["arm"])
    expected_hash = event.get("expected_dataset_hash")
    if expected_hash and result.get("dataset_hash") != expected_hash:
        raise RuntimeError("Evaluation dataset hash mismatch.")


def _execute_event(command: list[str], paths: dict[str, Path], event: dict[str, Any]) -> int:
    paths["log"].parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["MQAR_PRECISION_MACHINE"] = "3090"
    env["GDN_KERNEL_DTYPE"] = "bfloat16"
    with paths["log"].open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        while process.poll() is None:
            atomic_write_json(
                run_root() / "heartbeat.json",
                {"phase": event["mode"], "event_id": event["event_id"], "updated_at_utc": utc_now()},
            )
            time.sleep(5)
    return int(process.returncode)


def run_event(event: dict[str, Any]) -> dict[str, Any]:
    paths = _event_paths(event["mode"], event["event_id"])
    atomic_write_json(paths["event"], event)
    if paths["result"].exists():
        result = load_json(paths["result"])
        if result.get("status") == "oom":
            return result
        progress = load_json(paths["progress"]) if paths["progress"].exists() else {}
        if result.get("status") == "completed" and progress.get("event_identity_sha256") == stable_json_sha256(event):
            _audit_result(result, event)
            return result
    command = [
        str(PYTHON),
        str(EVAL_EVENT_SCRIPT),
        "--event",
        str(paths["event"]),
        "--progress",
        str(paths["progress"]),
        "--result",
        str(paths["result"]),
    ]
    return_code = _execute_event(command, paths, event)
    result = load_json(paths["result"])
    result.update(
        {
            "return_code": return_code,
            "event_path": str(paths["event"].resolve()),
            "result_path": str(paths["result"].resolve()),
            "log_path": str(paths["log"].resolve()),
        }
    )
    atomic_write_json(paths["result"], result)
    if return_code == 0:
        _audit_result(result, event)
    return result


def _compare_invariance(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    loss_deltas = [
        abs(float(a) - float(b))
        for a, b in zip(left["sample_loss_values"], right["sample_loss_values"])
    ]
    query_predictions = left.get("query_prediction_sample_sha256", []) == right.get(
        "query_prediction_sample_sha256",
        [],
    )
    checks = {
        "query_predictions": query_predictions,
        "accuracy": left["sample_accuracy_values"] == right["sample_accuracy_values"],
        "loss": max(loss_deltas, default=0.0) <= 5e-3,
        "dataset": left["dataset_hash"] == right["dataset_hash"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "max_sample_loss_delta": max(loss_deltas, default=0.0),
    }


def _profile_one(source: dict[str, Any], case) -> dict[str, Any]:
    attempts = []
    selected_result = None
    selected_batch = None
    for batch in BATCH_CANDIDATES:
        event = event_payload(source, case, batch, "capacity")
        result = run_event(event)
        attempts.append({"batch_size": batch, "status": result.get("status")})
        if result.get("status") == "completed":
            selected_batch, selected_result = batch, result
            break
        if result.get("status") != "oom":
            raise RuntimeError(f"Non-OOM capacity failure: {result}")
    if selected_batch is None or selected_result is None:
        raise RuntimeError(f"No viable batch for {source['arm']} {case_id(case)}.")
    smaller = next((value for value in BATCH_CANDIDATES if value < selected_batch), None)
    if smaller is None:
        raise RuntimeError(f"Batch invariance is unavailable below B1: {case_id(case)}.")
    smaller_result = run_event(event_payload(source, case, smaller, "invariance"))
    _audit_result(smaller_result, event_payload(source, case, smaller, "invariance"))
    invariance = _compare_invariance(selected_result, smaller_result)
    if not invariance["passed"]:
        raise RuntimeError(f"Batch invariance failed: {source['arm']} {case_id(case)}.")
    return {
        "arm": source["arm"],
        "case": case_id(case),
        "selected_batch_size": selected_batch,
        "smaller_batch_size": smaller,
        "attempts": attempts,
        "invariance": invariance,
    }


def profile_batches() -> dict[str, Any]:
    representatives = {source["arm"]: source for source in sources("screen")}
    profiles = [
        _profile_one(representatives[arm], case)
        for arm in ARMS
        for case in EVAL_CASES
    ]
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "run_tag": run_tag(),
        "status": "passed",
        "profiles": profiles,
        "recorded_at_utc": utc_now(),
    }
    atomic_write_json(run_root() / "evaluation" / "batch-profile.json", payload)
    return payload


def batch_profile() -> dict[tuple[str, str], int]:
    payload = load_json(run_root() / "evaluation" / "batch-profile.json")
    if payload.get("status") != "passed":
        raise RuntimeError("Evaluation batch profile did not pass.")
    return {
        (row["arm"], row["case"]): int(row["selected_batch_size"])
        for row in payload["profiles"]
    }


def _record(source: dict[str, Any], case, result: dict[str, Any], status: str) -> dict[str, Any]:
    return {
        "logical_source_id": source["source_id"],
        "arm": source["arm"],
        "model": source["model"],
        "seed": source["seed"],
        "checkpoint_role": source["checkpoint_role"],
        "case": case_id(case),
        "shape": f"{case[0]}x{case[1]}",
        "num_examples": case[2],
        "status": status,
        "accuracy": result["accuracy"],
        "loss": result["loss_sample_weighted"],
        "query_accuracy": result["query_accuracy"],
        "dataset_hash": result["dataset_hash"],
        "wall_clock_sec": result.get("wall_clock_sec"),
        "peak_allocated_mib": result.get("peak_allocated_mib"),
        "raw_result_path": result.get("result_path") or result.get("progress_path"),
    }


def evaluate_phase(phase: str) -> dict[str, Any]:
    selected_sources = sources(phase)
    cases = SMOKE_CASES if phase == "smoke" else SCREEN_CASES if phase == "screen" else EVAL_CASES
    profiles = batch_profile() if phase != "smoke" else {}
    physical: dict[tuple[str, str, str], dict[str, Any]] = {}
    records = []
    for source in selected_sources:
        for case in cases:
            batch = 1 if phase == "smoke" else profiles[(source["arm"], case_id(case))]
            key = (
                source["arm"],
                source["checkpoint_model_state_sha256"],
                case_id(case),
            )
            if key in physical:
                result, status = physical[key], "deduplicated"
            else:
                event = event_payload(source, case, batch, phase, max_batches=1 if phase == "smoke" else 0)
                result = run_event(event)
                _audit_result(result, event)
                physical[key], status = result, "completed"
            records.append(_record(source, case, result, status))
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "run_tag": run_tag(),
        "phase": phase,
        "status": "passed",
        "logical_events": len(records),
        "physical_events": len(physical),
        "records": records,
        "completed_at_utc": utc_now(),
    }
    path = run_root() / "evaluation" / f"{phase}-summary.json"
    atomic_write_json(path, payload)
    return payload


def repro() -> dict[str, Any]:
    profiles = batch_profile()
    case = STANDARD_CASES[-1]
    rows = []
    unique = {}
    for source in sources("formal"):
        key = (source["arm"], source["checkpoint_model_state_sha256"])
        unique.setdefault(key, source)
    formal = load_json(run_root() / "evaluation" / "formal-summary.json")
    reference = {
        (row["logical_source_id"], row["case"]): row for row in formal["records"]
    }
    for source in unique.values():
        batch = profiles[(source["arm"], case_id(case))]
        event = event_payload(source, case, batch, "repro")
        result = run_event(event)
        target = reference[(source["source_id"], case_id(case))]
        row = {
            "source_id": source["source_id"],
            "dataset_hash_match": result["dataset_hash"] == target["dataset_hash"],
            "accuracy_delta_abs": abs(float(result["accuracy"]) - float(target["accuracy"])),
        }
        row["passed"] = row["dataset_hash_match"] and row["accuracy_delta_abs"] <= 1e-12
        if not row["passed"]:
            raise RuntimeError(f"Endpoint repro failed: {row}")
        rows.append(row)
    payload = {"status": "passed", "rows": rows, "recorded_at_utc": utc_now()}
    atomic_write_json(run_root() / "evaluation" / "repro.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("profile", "smoke", "screen", "formal", "repro"),
    )
    args = parser.parse_args()
    if args.command == "profile":
        profile_batches()
    elif args.command == "repro":
        repro()
    else:
        evaluate_phase(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
