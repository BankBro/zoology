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
    EXPERIMENT_ID,
    FLASH_ROOT,
    LONGER_CASES,
    PYTHON,
    REPO_ROOT,
    VARIANTS,
    atomic_write_json,
    load_json,
    run_root,
    run_tag,
    safe_name,
    stable_json_sha256,
    training_descriptors,
    utc_now,
)
from experiment import result_path


EVAL_EVENT_SCRIPT = (
    REPO_ROOT
    / "zoology/experiments/flash_vqg/scripts"
    / "20260726-01-mqar-precision-profile"
    / "eval_event.py"
)


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
        "source_id": f"3090-{result['variant']}-s{result['seed']}-bf16-{role}",
        "machine": "3090",
        "model": "flash",
        "variant": result["variant"],
        "remat_mode": result["remat_mode"],
        "seed": int(result["seed"]),
        "train_precision": "bf16",
        "checkpoint_role": role,
        "checkpoint_path": checkpoint["path"],
        "checkpoint_file_sha256": checkpoint["file_sha256"],
        "checkpoint_model_state_sha256": checkpoint["model_state_sha256"],
    }


def sources(phase: str) -> list[dict[str, Any]]:
    selected = []
    descriptors = training_descriptors() if phase == "formal" else [
        {"variant": variant, "seed": 124} for variant in VARIANTS
    ]
    roles = ("last", "best") if phase == "formal" else ("last",)
    for row in descriptors:
        path = result_path(row["variant"], row["seed"], phase)
        result = load_json(path)
        if result.get("status") != "completed":
            raise RuntimeError(f"Training result is not complete: {path}")
        selected.extend(source_from_result(result, role) for role in roles)
    return selected


def event_payload(
    source: dict[str, Any],
    case: tuple[int, int, int, str],
    phase: str,
) -> dict[str, Any]:
    sequence_length, num_kv_pairs, batch_size, dataset_hash = case
    zoology_commit = git_commit(REPO_ROOT)
    flash_commit = git_commit(FLASH_ROOT)
    checkpoint_short = source["checkpoint_model_state_sha256"][:16]
    event_id = (
        f"{phase}-{source['source_id']}-bf16-{sequence_length}x{num_kv_pairs}-"
        f"n500-b{batch_size}-{checkpoint_short}-"
        f"z{zoology_commit[:8]}-f{flash_commit[:8]}"
    )
    return {
        "experiment_id": EXPERIMENT_ID,
        "run_tag": run_tag(),
        "zoology_commit": zoology_commit,
        "flash_commit": flash_commit,
        "event_id": event_id,
        "mode": phase,
        "machine": "3090",
        "model": "flash",
        "variant": source["variant"],
        "remat_mode": source["remat_mode"],
        "seed": source["seed"],
        "checkpoint_role": source["checkpoint_role"],
        "train_precision": "bf16",
        "eval_precision": "bf16",
        "checkpoint_path": source["checkpoint_path"],
        "checkpoint_file_sha256": source["checkpoint_file_sha256"],
        "checkpoint_model_state_sha256": source["checkpoint_model_state_sha256"],
        "input_seq_len": sequence_length,
        "num_kv_pairs": num_kv_pairs,
        "num_examples": 500,
        "eval_batch_size": batch_size,
        "eval_seed": 123,
        "expected_dataset_hash": dataset_hash,
        "dataset_policy": "generated_seeded",
        "max_batches": 1 if phase == "smoke" else 0,
        "controlled_interrupt_after_batches": 0,
    }


def _paths(phase: str, event_id: str) -> dict[str, Path]:
    root = run_root() / "evaluation" / phase / safe_name(event_id)
    return {
        "event": root / "event.json",
        "progress": root / "progress.json",
        "result": root / "result.json",
        "log": root / "event.log",
    }


def _runtime_audit(result: dict[str, Any]) -> None:
    states = result.get("model_runtime_state") or {}
    audits = [
        value.get("fox_gd_residual_triton_runtime_audit")
        for value in states.values()
        if value.get("fox_gd_residual_triton_runtime_audit") is not None
    ]
    if not audits:
        raise RuntimeError("Evaluation did not record Flash Triton runtime audit.")
    for audit in audits:
        if int(audit.get("grouped_calls", 0)) <= 0 or int(audit.get("selected_calls", 0)) <= 0:
            raise RuntimeError(f"Evaluation missed Triton calls: {audit}")
        if int(audit.get("grouped_fallbacks", 0)) or int(audit.get("selected_fallbacks", 0)):
            raise RuntimeError(f"Evaluation recorded a fallback: {audit}")
        if audit.get("actual_core_dtype") != "float32":
            raise RuntimeError(f"Evaluation core dtype mismatch: {audit}")


def run_event(event: dict[str, Any]) -> dict[str, Any]:
    paths = _paths(event["mode"], event["event_id"])
    atomic_write_json(paths["event"], event)
    if paths["result"].exists() and paths["progress"].exists():
        result = load_json(paths["result"])
        progress = load_json(paths["progress"])
        if (
            result.get("status") == "completed"
            and progress.get("event_identity_sha256") == stable_json_sha256(event)
        ):
            _runtime_audit(result)
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
    result = load_json(paths["result"])
    result["return_code"] = int(process.returncode)
    result["event_path"] = str(paths["event"].resolve())
    result["result_path"] = str(paths["result"].resolve())
    result["log_path"] = str(paths["log"].resolve())
    atomic_write_json(paths["result"], result)
    if process.returncode != 0 or result.get("status") != "completed":
        raise RuntimeError(f"Evaluation failed: {paths['result']}")
    _runtime_audit(result)
    return result


def evaluate(phase: str) -> dict[str, Any]:
    selected_sources = sources(phase)
    cases = (
        (LONGER_CASES[0], LONGER_CASES[-1])
        if phase == "smoke"
        else LONGER_CASES
    )
    records = []
    physical: dict[tuple[str, int, int], dict[str, Any]] = {}
    for source in selected_sources:
        for case in cases:
            sequence_length, num_kv_pairs, _batch_size, _hash = case
            key = (source["checkpoint_model_state_sha256"], sequence_length, num_kv_pairs)
            if key in physical:
                result = physical[key]
                status = "deduplicated"
            else:
                event = event_payload(source, case, phase)
                result = run_event(event)
                physical[key] = result
                status = "completed"
            records.append(
                {
                    "logical_source_id": source["source_id"],
                    "variant": source["variant"],
                    "remat_mode": source["remat_mode"],
                    "seed": source["seed"],
                    "checkpoint_role": source["checkpoint_role"],
                    "shape": f"{sequence_length}x{num_kv_pairs}",
                    "status": status,
                    "accuracy": result["accuracy"],
                    "loss": result["loss_sample_weighted"],
                    "dataset_hash": result["dataset_hash"],
                    "wall_clock_sec": result.get("wall_clock_sec"),
                    "peak_allocated_mib": result.get("peak_allocated_mib"),
                    "raw_result_path": result.get("result_path") or result.get("progress_path"),
                }
            )
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
    print(json.dumps({"status": "passed", "summary": str(path)}, sort_keys=True))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("smoke", "formal"), required=True)
    args = parser.parse_args()
    evaluate(args.phase)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
