#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
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
    SEED,
    VARIANTS,
    atomic_write_json,
    load_json,
    run_root,
    run_tag,
    sha256_file,
    stable_json_sha256,
    utc_now,
)
from experiment import git_value, result_path


EVAL_SCRIPT = (
    REPO_ROOT
    / "zoology/experiments/flash_vqg/scripts"
    / "20260726-01-mqar-precision-profile/eval_event.py"
)


def evaluation_checkpoint(variant: str, training: dict[str, Any]) -> Path:
    source = Path(training["last_checkpoint"]["path"])
    shadow = run_root() / "evaluation/checkpoint-shadow" / variant
    shadow.mkdir(parents=True, exist_ok=True)
    target = shadow / "last.pt"
    if not target.exists():
        os.link(source, target)
    source_config = source.parent / "train_config.json"
    payload = json.loads(source_config.read_text(encoding="utf-8"))
    identity = payload.get("resume_identity") or {}
    identity["seed"] = str(identity.get("seed", SEED))
    payload["resume_identity"] = identity
    atomic_write_json(shadow / "train_config.json", payload)
    atomic_write_json(
        shadow / "shadow-metadata.json",
        {
            "source_checkpoint": str(source.resolve()),
            "source_checkpoint_sha256": sha256_file(source),
            "source_train_config_sha256": sha256_file(source_config),
            "shadow_checkpoint": str(target.resolve()),
            "shadow_train_config_sha256": sha256_file(shadow / "train_config.json"),
        },
    )
    return target


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
            f"block64-{variant}-{sequence_length}x{num_kv_pairs}-b{batch_size}-"
            f"{checkpoint['model_state_sha256'][:16]}-z{zoology_commit[:8]}-f{flash_commit[:8]}"
        ),
        "mode": "probe",
        "machine": "2080ti",
        "model": "flash",
        "variant": variant,
        "remat_mode": VARIANTS[variant],
        "seed": SEED,
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
    log_path = root / "event.log"
    atomic_write_json(paths["event"], event)
    if paths["result"].exists() and paths["progress"].exists():
        result, progress = load_json(paths["result"]), load_json(paths["progress"])
        if (
            result.get("status") == "completed"
            and progress.get("event_identity_sha256") == stable_json_sha256(event)
        ):
            return result
    command = [
        str(PYTHON),
        str(EVAL_SCRIPT),
        "--event",
        str(paths["event"]),
        "--progress",
        str(paths["progress"]),
        "--result",
        str(paths["result"]),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["MQAR_PRECISION_MACHINE"] = "2080ti"
    environment["GDN_KERNEL_DTYPE"] = "float32"
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    result = load_json(paths["result"])
    if process.returncode != 0 or result.get("status") != "completed":
        raise RuntimeError(f"Evaluation failed: {paths['result']}")
    return result


def standard_accuracy(training: dict[str, Any]) -> float:
    return float(training["last_checkpoint"]["metrics"]["valid/mqar_case/accuracy-1024x256"])


def trajectory_gate(trainings: dict[str, dict[str, Any]]) -> dict[str, Any]:
    a0, a1 = trainings["a0-block64"], trainings["a1-block64"]
    left = {int(row["step"]): float(row["loss"]) for row in a0["telemetry"]["records"]}
    right = {int(row["step"]): float(row["loss"]) for row in a1["telemetry"]["records"]}
    common_steps = sorted(set(left) & set(right))
    max_abs = max((abs(right[step] - left[step]) for step in common_steps), default=float("inf"))
    model_equal = a0["resume"]["model_state_sha256"] == a1["resume"]["model_state_sha256"]
    optimizer_equal = (
        a0["resume"]["optimizer_state_sha256"]
        == a1["resume"]["optimizer_state_sha256"]
    )
    a0_config = a0["gate_autotune"]["layer_norm_gated_bwd_kernel"]["best_config"]
    a1_config = a1["gate_autotune"]["layer_norm_gated_bwd_kernel"]["best_config"]
    configs_equal = a0_config == a1_config and a0_config is not None
    exact = model_equal and optimizer_equal and max_abs <= 1.0e-4
    return {
        "common_steps": len(common_steps),
        "max_abs_loss_delta": max_abs,
        "model_state_equal": model_equal,
        "optimizer_state_equal": optimizer_equal,
        "a0_gate_bwd_config": a0_config,
        "a1_gate_bwd_config": a1_config,
        "gate_bwd_configs_equal": configs_equal,
        "requires_fla_replay": not exact,
        "passed": exact,
    }


def evaluate() -> dict[str, Any]:
    trainings = {
        variant: load_json(result_path(variant, "screen")) for variant in VARIANTS
    }
    if any(result.get("status") != "completed" for result in trainings.values()):
        raise RuntimeError("Training is incomplete.")
    standards = {variant: standard_accuracy(result) for variant, result in trainings.items()}
    records = []
    for variant, training in trainings.items():
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
    indexed = {(row["variant"], row["shape"]): row for row in records}
    extrapolation = [f"{case[0]}x{case[1]}" for case in LONGER_CASES[1:]]
    standard_delta = standards["a1-block64"] - standards["a0-block64"]
    extrapolation_deltas = {
        shape: indexed[("a1-block64", shape)]["accuracy"]
        - indexed[("a0-block64", shape)]["accuracy"]
        for shape in extrapolation
    }
    macro_delta = sum(extrapolation_deltas.values()) / len(extrapolation_deltas)
    quality_passed = standard_delta >= -0.01 and macro_delta >= -0.02
    trajectory = trajectory_gate(trainings)
    if not quality_passed:
        status = "quality_rejected"
    elif trajectory["requires_fla_replay"]:
        status = "requires_fla_replay"
    else:
        status = "passed"
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "run_tag": run_tag(),
        "status": status,
        "completed_at_utc": utc_now(),
        "standard_accuracy": standards,
        "standard_delta": standard_delta,
        "extrapolation_deltas": extrapolation_deltas,
        "extrapolation_macro_delta": macro_delta,
        "thresholds": {
            "standard_delta_min": -0.01,
            "extrapolation_macro_delta_min": -0.02,
            "matched_config_loss_max_abs": 1.0e-4,
        },
        "trajectory_gate": trajectory,
        "records": records,
    }
    atomic_write_json(run_root() / "evaluation/summary.json", payload)
    print(json.dumps({"status": status, "standard_delta": standard_delta, "macro_delta": macro_delta}))
    return payload


if __name__ == "__main__":
    result = evaluate()
    raise SystemExit({"passed": 0, "quality_rejected": 2, "requires_fla_replay": 3}[result["status"]])
