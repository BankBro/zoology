#!/usr/bin/env python3
from __future__ import annotations

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
    LONGER_CASES,
    PYTHON,
    REPO_ROOT,
    VARIANTS,
    atomic_write_json,
    load_json,
    run_root,
    run_tag,
    stable_json_sha256,
    sha256_file,
    utc_now,
)
from experiment import git_value, result_path


EVAL_SCRIPT = REPO_ROOT / "zoology/experiments/flash_vqg/scripts/20260726-01-mqar-precision-profile/eval_event.py"


def evaluation_checkpoint(variant: str, training: dict[str, Any]) -> Path:
    source = Path(training["last_checkpoint"]["path"])
    shadow = run_root() / "evaluation" / "checkpoint-shadow" / variant
    shadow.mkdir(parents=True, exist_ok=True)
    shadow_checkpoint = shadow / "last.pt"
    if not shadow_checkpoint.exists():
        os.link(source, shadow_checkpoint)
    source_config = source.parent / "train_config.json"
    payload = json.loads(source_config.read_text(encoding="utf-8"))
    identity = payload.get("resume_identity") or {}
    identity["seed"] = str(identity.get("seed", 123))
    payload["resume_identity"] = identity
    atomic_write_json(shadow / "train_config.json", payload)
    atomic_write_json(
        shadow / "shadow-metadata.json",
        {
            "source_checkpoint": str(source.resolve()),
            "source_checkpoint_sha256": sha256_file(source),
            "source_train_config": str(source_config.resolve()),
            "source_train_config_sha256": sha256_file(source_config),
            "shadow_checkpoint": str(shadow_checkpoint.resolve()),
            "shadow_train_config_sha256": sha256_file(shadow / "train_config.json"),
            "repair": "resume_identity.seed int-to-string compatibility fix",
        },
    )
    return shadow_checkpoint


def event_payload(variant: str, training: dict[str, Any], case) -> dict[str, Any]:
    sequence_length, num_kv_pairs, batch_size, dataset_hash = case
    checkpoint = training["last_checkpoint"]
    checkpoint_path = evaluation_checkpoint(variant, training)
    zoology_commit = git_value(REPO_ROOT, "rev-parse", "HEAD")
    flash_commit = git_value(Path("/home/lyj/mnt/project/Flash-VQG"), "rev-parse", "HEAD")
    return {
        "experiment_id": EXPERIMENT_ID,
        "run_tag": run_tag(),
        "zoology_commit": zoology_commit,
        "flash_commit": flash_commit,
        "event_id": (
            f"probe-{variant}-{sequence_length}x{num_kv_pairs}-b{batch_size}-"
            f"{checkpoint['model_state_sha256'][:16]}-z{zoology_commit[:8]}-f{flash_commit[:8]}"
        ),
        "mode": "probe",
        "machine": "2080ti",
        "model": "flash",
        "variant": variant,
        "remat_mode": "post_phase1",
        "seed": 123,
        "checkpoint_role": "last",
        "train_precision": "fp32",
        "eval_precision": "fp32",
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_file_sha256": checkpoint["file_sha256"],
        "checkpoint_model_state_sha256": checkpoint["model_state_sha256"],
        "input_seq_len": sequence_length,
        "num_kv_pairs": num_kv_pairs,
        "num_examples": 500,
        "eval_batch_size": batch_size,
        "eval_seed": 123,
        "expected_dataset_hash": dataset_hash,
        "dataset_policy": "generated_seeded",
        "max_batches": 0,
        "controlled_interrupt_after_batches": 0,
    }


def run_event(event: dict[str, Any]) -> dict[str, Any]:
    root = run_root() / "evaluation" / event["event_id"]
    paths = {name: root / f"{name}.json" for name in ("event", "progress", "result")}
    paths["log"] = root / "event.log"
    atomic_write_json(paths["event"], event)
    if paths["result"].exists() and paths["progress"].exists():
        result = load_json(paths["result"])
        progress = load_json(paths["progress"])
        if result.get("status") == "completed" and progress.get("event_identity_sha256") == stable_json_sha256(event):
            return result
    command = [
        str(PYTHON),
        str(EVAL_SCRIPT),
        "--event", str(paths["event"]),
        "--progress", str(paths["progress"]),
        "--result", str(paths["result"]),
    ]
    paths["log"].parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["MQAR_PRECISION_MACHINE"] = "2080ti"
    env["GDN_KERNEL_DTYPE"] = "float32"
    with paths["log"].open("a", encoding="utf-8") as log:
        process = subprocess.run(command, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    result = load_json(paths["result"])
    if process.returncode != 0 or result.get("status") != "completed":
        raise RuntimeError(f"Evaluation failed: {paths['result']}")
    return result


def standard_accuracy(training: dict[str, Any]) -> float:
    metrics = training["last_checkpoint"]["metrics"]
    return float(metrics["valid/mqar_case/accuracy-1024x256"])


def evaluate() -> dict[str, Any]:
    records = []
    standards = {}
    for variant in VARIANTS:
        training = load_json(result_path(variant, "screen"))
        if training.get("status") != "completed":
            raise RuntimeError(f"Training is incomplete: {variant}")
        standards[variant] = standard_accuracy(training)
        for case in LONGER_CASES:
            result = run_event(event_payload(variant, training, case))
            records.append(
                {
                    "variant": variant,
                    "shape": f"{case[0]}x{case[1]}",
                    "accuracy": float(result["accuracy"]),
                    "loss": float(result["loss_sample_weighted"]),
                    "dataset_hash": result["dataset_hash"],
                    "wall_clock_sec": result.get("wall_clock_sec"),
                    "peak_allocated_mib": result.get("peak_allocated_mib"),
                }
            )
    by_key = {(row["variant"], row["shape"]): row for row in records}
    extrapolation = [f"{case[0]}x{case[1]}" for case in LONGER_CASES[1:]]
    standard_delta = standards["a1-block256-k2r8"] - standards["a1-reference"]
    extrapolation_deltas = [
        by_key[("a1-block256-k2r8", shape)]["accuracy"]
        - by_key[("a1-reference", shape)]["accuracy"]
        for shape in extrapolation
    ]
    macro_delta = sum(extrapolation_deltas) / len(extrapolation_deltas)
    passed = standard_delta >= -0.02 and macro_delta >= -0.05
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "run_tag": run_tag(),
        "status": "passed" if passed else "quality_rejected",
        "completed_at_utc": utc_now(),
        "standard_accuracy": standards,
        "standard_delta": standard_delta,
        "extrapolation_deltas": dict(zip(extrapolation, extrapolation_deltas, strict=True)),
        "extrapolation_macro_delta": macro_delta,
        "thresholds": {"standard_delta_min": -0.02, "extrapolation_macro_delta_min": -0.05},
        "records": records,
    }
    atomic_write_json(run_root() / "evaluation" / "summary.json", payload)
    print(json.dumps({"status": payload["status"], "standard_delta": standard_delta, "macro_delta": macro_delta}))
    return payload


if __name__ == "__main__":
    raise SystemExit(0 if evaluate()["status"] == "passed" else 2)
