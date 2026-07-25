from __future__ import annotations

import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from common import (
    ALL_EVAL_CASES,
    BATCH_CANDIDATES,
    EXPERIMENT_ID,
    FLASH_ROOT,
    GDN_KERNEL_DTYPE,
    LONGER_DATASET_HASHES,
    MACHINES,
    PYTHON,
    REPO_ROOT,
    SCRIPT_DIR,
    atomic_write_json,
    examples_for_shape,
    load_json,
    machine_name,
    output_root,
    safe_name,
    shape_name,
    stable_json_sha256,
    utc_now,
)


EVAL_EVENT_SCRIPT = SCRIPT_DIR / "eval_event.py"


def _git_commit(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


class EvalEventError(RuntimeError):
    def __init__(self, message: str, result: dict[str, Any]):
        super().__init__(message)
        self.result = result


class EventRunner:
    def __init__(self, heartbeat: Callable[[str], None]):
        self.machine = machine_name()
        self.root = output_root() / "evaluation"
        self.heartbeat = heartbeat

    def _paths(self, mode: str, event_id: str):
        base = self.root / mode / safe_name(event_id)
        return {
            "event": base / "event.json",
            "progress": base / "progress.json",
            "result": base / "result.json",
            "log": base / "event.log",
        }

    def _run_process(self, command: list[str], log_path: Path, event_id: str) -> int:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=os.environ.copy(),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            while process.poll() is None:
                self.heartbeat(event_id)
                time.sleep(5)
            return int(process.returncode)

    def run(self, event: dict[str, Any], mode: str) -> dict[str, Any]:
        paths = self._paths(mode, event["event_id"])
        atomic_write_json(paths["event"], event)
        if paths["result"].exists() and paths["progress"].exists():
            result = load_json(paths["result"])
            progress = load_json(paths["progress"])
            if (
                result.get("status") == "completed"
                and progress.get("event_identity_sha256")
                == stable_json_sha256(event)
            ):
                self._audit_completed_result(event, result)
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
        return_code = self._run_process(command, paths["log"], event["event_id"])
        if return_code == 75:
            return_code = self._run_process(command, paths["log"], event["event_id"])
        result = load_json(paths["result"])
        result["return_code"] = return_code
        result["event_path"] = str(paths["event"].resolve())
        result["log_path"] = str(paths["log"].resolve())
        atomic_write_json(paths["result"], result)
        if result.get("status") == "completed":
            self._audit_completed_result(event, result)
        return result

    @staticmethod
    def _audit_completed_result(
        event: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        if event["model"] != "flash":
            return
        runtime = result.get("model_runtime_state") or {}
        audits = [
            value.get("fox_gd_residual_triton_runtime_audit")
            for value in runtime.values()
            if value.get("fox_gd_residual_triton_runtime_audit") is not None
        ]
        if not audits:
            raise RuntimeError("Flash eval did not record Triton runtime audit.")
        for audit in audits:
            if int(audit["grouped_calls"]) <= 0 or int(audit["selected_calls"]) <= 0:
                raise RuntimeError(f"Flash eval missed a Triton core call: {audit}")
            if int(audit["grouped_fallbacks"]) or int(audit["selected_fallbacks"]):
                raise RuntimeError(f"Flash eval recorded a fallback: {audit}")
            if audit["actual_core_dtype"] != "float32":
                raise RuntimeError(f"Flash core dtype audit failed: {audit}")
            if (
                event["eval_precision"] in {"fp16", "bf16"}
                and int(audit["selected_precast_low_precision_calls"]) <= 0
            ):
                raise RuntimeError(
                    f"Flash eval did not exercise a low-precision boundary: {audit}"
                )


def event_payload(
    *,
    source: dict[str, Any],
    eval_precision: str,
    shape: tuple[int, int],
    batch_size: int,
    mode: str,
    max_batches: int = 0,
    controlled_interrupt: bool = False,
    num_examples_override: int | None = None,
) -> dict[str, Any]:
    sequence_length, num_kv_pairs = shape
    num_examples = (
        int(num_examples_override)
        if num_examples_override is not None
        else examples_for_shape(shape)
    )
    shape_id = shape_name(*shape)
    source_id = source["source_id"]
    checkpoint_short = source["checkpoint_model_state_sha256"][:16]
    zoology_commit = _git_commit(REPO_ROOT)
    flash_commit = _git_commit(FLASH_ROOT)
    event_id = (
        f"{mode}-{source_id}-eval{eval_precision}-{shape_id}-"
        f"n{num_examples}-b{batch_size}-{checkpoint_short}-"
        f"z{zoology_commit[:8]}-f{flash_commit[:8]}"
    )
    expected_hash = (
        LONGER_DATASET_HASHES.get(shape_id)
        if num_examples == 500
        else None
    )
    return {
        "experiment_id": EXPERIMENT_ID,
        "zoology_commit": zoology_commit,
        "flash_commit": flash_commit,
        "event_id": event_id,
        "mode": mode,
        "machine": machine_name(),
        "model": source["model"],
        "seed": source["seed"],
        "checkpoint_role": source["checkpoint_role"],
        "train_precision": source["train_precision"],
        "eval_precision": eval_precision,
        "checkpoint_path": source["checkpoint_path"],
        "checkpoint_file_sha256": source["checkpoint_file_sha256"],
        "checkpoint_model_state_sha256": source[
            "checkpoint_model_state_sha256"
        ],
        "input_seq_len": sequence_length,
        "num_kv_pairs": num_kv_pairs,
        "num_examples": num_examples,
        "eval_batch_size": int(batch_size),
        "eval_seed": 123,
        "expected_dataset_hash": expected_hash,
        "max_batches": int(max_batches),
        "controlled_interrupt_after_batches": 1 if controlled_interrupt else 0,
    }


def load_batch_profile() -> dict[str, int]:
    path = output_root() / "gates" / "batch-profile.json"
    payload = load_json(path)
    if payload.get("status") != "passed":
        raise RuntimeError("Batch profile gate has not passed.")
    return {
        row["key"]: int(row["selected_batch_size"])
        for row in payload["profiles"]
    }


def batch_key(
    model: str,
    eval_precision: str,
    shape: tuple[int, int],
    num_examples: int,
) -> str:
    return (
        f"{machine_name()}:{model}:{eval_precision}:"
        f"{shape_name(*shape)}:n{num_examples}"
    )


def _compare_invariance(
    selected: dict[str, Any],
    smaller: dict[str, Any],
    precision: str,
) -> dict[str, Any]:
    prediction_match = (
        selected["prediction_sample_sha256"]
        == smaller["prediction_sample_sha256"]
    )
    accuracy_match = selected["sample_accuracy_values"] == smaller[
        "sample_accuracy_values"
    ]
    loss_deltas = [
        abs(float(left) - float(right))
        for left, right in zip(
            selected["sample_loss_values"],
            smaller["sample_loss_values"],
        )
    ]
    max_loss_delta = max(loss_deltas, default=0.0)
    tolerance = 1e-5 if precision == "fp32" else 5e-3
    passed = prediction_match and accuracy_match and max_loss_delta <= tolerance
    return {
        "passed": passed,
        "prediction_match": prediction_match,
        "accuracy_match": accuracy_match,
        "max_sample_loss_delta": max_loss_delta,
        "loss_tolerance": tolerance,
    }


def search_batch_profiles(
    runner: EventRunner,
    smoke_sources: list[dict[str, Any]],
) -> dict[str, Any]:
    profiles = []
    invariance_rows = []
    machine = machine_name()
    for model in ("gdn", "flash"):
        for eval_precision in MACHINES[machine]["eval_precisions"]:
            source = next(
                row
                for row in smoke_sources
                if row["model"] == model
                and row["train_precision"] == eval_precision
                and row["seed"] == 123
            )
            for sequence_length, num_kv_pairs, num_examples in ALL_EVAL_CASES:
                shape = (sequence_length, num_kv_pairs)
                selected = None
                selected_result = None
                attempts = []
                for candidate in BATCH_CANDIDATES:
                    if candidate > num_examples:
                        continue
                    event = event_payload(
                        source=source,
                        eval_precision=eval_precision,
                        shape=shape,
                        batch_size=candidate,
                        mode="capacity",
                        num_examples_override=num_examples,
                    )
                    result = runner.run(event, "capacity")
                    attempts.append(
                        {
                            "batch_size": candidate,
                            "status": result.get("status"),
                            "failure_type": result.get("failure_type"),
                        }
                    )
                    if result.get("status") == "completed":
                        selected = candidate
                        selected_result = result
                        break
                    if result.get("status") != "oom":
                        raise EvalEventError(
                            f"Non-OOM capacity failure: {event['event_id']}",
                            result,
                        )
                if selected is None or selected_result is None:
                    raise RuntimeError(
                        f"No viable eval batch for {model}, {eval_precision}, {shape}."
                    )
                smaller_candidates = [
                    value for value in BATCH_CANDIDATES if value < selected
                ]
                if not smaller_candidates:
                    raise RuntimeError(
                        f"Cannot run batch invariance below batch size 1: {shape}."
                    )
                smaller_batch = max(smaller_candidates)
                smaller_event = event_payload(
                    source=source,
                    eval_precision=eval_precision,
                    shape=shape,
                    batch_size=smaller_batch,
                    mode="invariance",
                    num_examples_override=num_examples,
                )
                smaller_result = runner.run(smaller_event, "invariance")
                if smaller_result.get("status") != "completed":
                    raise EvalEventError(
                        f"Batch invariance run failed: {smaller_event['event_id']}",
                        smaller_result,
                    )
                invariance = _compare_invariance(
                    selected_result,
                    smaller_result,
                    eval_precision,
                )
                invariance_rows.append(
                    {
                        "key": batch_key(
                            model,
                            eval_precision,
                            shape,
                            num_examples,
                        ),
                        "selected_batch_size": selected,
                        "smaller_batch_size": smaller_batch,
                        **invariance,
                    }
                )
                if not invariance["passed"]:
                    raise RuntimeError(
                        f"Batch invariance failed: {invariance_rows[-1]}"
                    )
                profiles.append(
                    {
                        "key": batch_key(
                            model,
                            eval_precision,
                            shape,
                            num_examples,
                        ),
                        "model": model,
                        "eval_precision": eval_precision,
                        "shape": shape_name(*shape),
                        "num_examples": num_examples,
                        "selected_batch_size": selected,
                        "attempts": attempts,
                    }
                )
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "machine": machine,
        "status": "passed",
        "recorded_at_utc": utc_now(),
        "profiles": profiles,
        "batch_invariance": invariance_rows,
    }
    atomic_write_json(output_root() / "gates" / "batch-profile.json", payload)
    return payload


def run_smoke_evaluations(
    runner: EventRunner,
    sources: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    profile = load_batch_profile()
    records = []
    for source in sources:
        for eval_precision in MACHINES[machine_name()]["eval_precisions"]:
            for sequence_length, num_kv_pairs, num_examples in ALL_EVAL_CASES:
                shape = (sequence_length, num_kv_pairs)
                batch_size = profile[
                    batch_key(
                        source["model"],
                        eval_precision,
                        shape,
                        num_examples,
                    )
                ]
                event = event_payload(
                    source=source,
                    eval_precision=eval_precision,
                    shape=shape,
                    batch_size=batch_size,
                    mode="smoke",
                    max_batches=3,
                    controlled_interrupt=shape == (8190, 2047),
                    num_examples_override=num_examples,
                )
                result = runner.run(event, "smoke")
                if result.get("status") != "completed":
                    raise EvalEventError(
                        f"Eval smoke failed: {event['event_id']}",
                        result,
                    )
                expected_examples = min(
                    num_examples,
                    3 * batch_size,
                )
                if int(result["num_examples"]) != expected_examples:
                    raise RuntimeError("Eval smoke processed an unexpected sample count.")
                records.append(
                    {
                        "event_id": event["event_id"],
                        "status": result["status"],
                        "num_examples": result["num_examples"],
                        "dataset_hash": result["dataset_hash"],
                    }
                )
    return records


def run_full_evaluations(
    runner: EventRunner,
    sources: list[dict[str, Any]],
    *,
    priority_longest: bool = False,
    mode: str = "formal",
    eval_precisions: tuple[str, ...] | None = None,
    shapes_override: tuple[tuple[int, int, int], ...] | None = None,
) -> list[dict[str, Any]]:
    profile = load_batch_profile()
    cases = list(shapes_override or ALL_EVAL_CASES)
    if priority_longest:
        priority = [(8190, 512, 500), (8190, 2047, 500)]
        cases = priority + [case for case in cases if case not in priority]
    records = []
    physical_results: dict[tuple[str, str, tuple[int, int]], dict[str, Any]] = {}
    selected_precisions = (
        eval_precisions
        if eval_precisions is not None
        else MACHINES[machine_name()]["eval_precisions"]
    )
    for source in sources:
        for eval_precision in selected_precisions:
            for sequence_length, num_kv_pairs, num_examples in cases:
                shape = (sequence_length, num_kv_pairs)
                physical_key = (
                    source["checkpoint_model_state_sha256"],
                    eval_precision,
                    (sequence_length, num_kv_pairs, num_examples),
                )
                if physical_key in physical_results:
                    result = physical_results[physical_key]
                    records.append(
                        {
                            "event_id": result["event_id"],
                            "logical_source_id": source["source_id"],
                            "status": "deduplicated",
                            "machine": source["machine"],
                            "model": source["model"],
                            "seed": source["seed"],
                            "checkpoint_role": source["checkpoint_role"],
                            "train_precision": source["train_precision"],
                            "eval_precision": eval_precision,
                            "shape": shape_name(*shape),
                            "num_examples": num_examples,
                            "accuracy": result["accuracy"],
                            "loss": result["loss_sample_weighted"],
                            "dataset_hash": result["dataset_hash"],
                        }
                    )
                    continue
                batch_size = profile[
                    batch_key(
                        source["model"],
                        eval_precision,
                        shape,
                        num_examples,
                    )
                ]
                event = event_payload(
                    source=source,
                    eval_precision=eval_precision,
                    shape=shape,
                    batch_size=batch_size,
                    mode=mode,
                    num_examples_override=num_examples,
                )
                result = runner.run(event, mode)
                if result.get("status") != "completed":
                    raise EvalEventError(
                        f"Formal eval failed: {event['event_id']}",
                        result,
                    )
                physical_results[physical_key] = result
                records.append(
                    {
                        "event_id": event["event_id"],
                        "logical_source_id": source["source_id"],
                        "status": "completed",
                        "machine": source["machine"],
                        "model": source["model"],
                        "seed": source["seed"],
                        "checkpoint_role": source["checkpoint_role"],
                        "train_precision": source["train_precision"],
                        "eval_precision": eval_precision,
                        "shape": shape_name(*shape),
                        "num_examples": num_examples,
                        "accuracy": result["accuracy"],
                        "loss": result["loss_sample_weighted"],
                        "dataset_hash": result["dataset_hash"],
                        "wall_clock_sec": result["wall_clock_sec"],
                        "peak_allocated_mib": result["peak_allocated_mib"],
                    }
                )
    return records
