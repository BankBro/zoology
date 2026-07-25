#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

LOCAL_SCRIPT_DIR = Path(__file__).resolve().parent
if str(LOCAL_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_SCRIPT_DIR))

from common import (  # noqa: E402
    EXPERIMENT_ID,
    GDN_KERNEL_DTYPE,
    MACHINES,
    PYTHON,
    REPO_ROOT,
    SCRIPT_DIR,
    atomic_write_json,
    load_json,
    machine_name,
    output_root,
    sha256_file,
    stable_json_sha256,
    training_descriptors,
    utc_now,
)
from eval_queue import (  # noqa: E402
    EventRunner,
    run_full_evaluations,
    run_smoke_evaluations,
    search_batch_profiles,
)


EXPERIMENT_SCRIPT = SCRIPT_DIR / "experiment.py"


class MachineQueue:
    def __init__(self):
        self.machine = machine_name()
        self.root = output_root()
        self.status_path = self.root / "status.json"
        self.heartbeat_path = self.root / "heartbeat.json"
        self.log_dir = self.root / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.event_runner = EventRunner(self.heartbeat)
        self.current_phase = "initializing"
        self.completed_nodes = 0

    def status(self, status: str, current: str = "", **extra: Any) -> None:
        atomic_write_json(
            self.status_path,
            {
                "experiment_id": EXPERIMENT_ID,
                "machine": self.machine,
                "status": status,
                "phase": self.current_phase,
                "current": current,
                "completed_nodes": self.completed_nodes,
                "updated_at_utc": utc_now(),
                **extra,
            },
        )

    def heartbeat(self, current: str) -> None:
        atomic_write_json(
            self.heartbeat_path,
            {
                "experiment_id": EXPERIMENT_ID,
                "machine": self.machine,
                "phase": self.current_phase,
                "current": current,
                "pid": os.getpid(),
                "updated_at_utc": utc_now(),
            },
        )
        self.status("running", current)

    def run_process(
        self,
        command: list[str],
        log_path: Path,
        current: str,
    ) -> int:
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
                self.heartbeat(current)
                time.sleep(5)
            return int(process.returncode)

    def run_preflight(self) -> None:
        self.current_phase = "preflight"
        os.environ["GDN_KERNEL_DTYPE"] = "float32"
        command = [str(PYTHON), str(EXPERIMENT_SCRIPT), "preflight"]
        return_code = self.run_process(
            command,
            self.log_dir / "preflight.log",
            "preflight",
        )
        if return_code != 0:
            raise RuntimeError("Preflight failed.")
        payload = load_json(self.root / "preflight.json")
        if payload.get("status") != "passed":
            raise RuntimeError("Preflight gate is not passed.")

    def _training_result_path(self, descriptor: dict[str, Any], phase: str) -> Path:
        run = (
            f"{descriptor['model']}-s{descriptor['seed']}-"
            f"{descriptor['train_precision']}-b64ga4-{phase}"
        )
        return self.root / "training" / phase / run / "result.json"

    def _training_command(self, descriptor: dict[str, Any], phase: str) -> list[str]:
        return [
            str(PYTHON),
            str(EXPERIMENT_SCRIPT),
            "train",
            "--model",
            descriptor["model"],
            "--seed",
            str(descriptor["seed"]),
            "--precision",
            descriptor["train_precision"],
            "--phase",
            phase,
        ]

    def _audit_training(self, result: dict[str, Any], phase: str) -> None:
        if result.get("status") != "completed":
            raise RuntimeError(f"Training did not complete: {result}")
        resume = result["resume_audit"]
        expected_updates = 3 if phase in {"smoke", "stress"} else None
        if expected_updates is not None and int(resume["optimizer_step"]) != expected_updates:
            raise RuntimeError("Smoke did not complete three successful optimizer updates.")
        if int(resume["grad_scaler_skips"]) > 2:
            raise RuntimeError("GradScaler skip audit failed.")
        if result["model"] == "flash":
            runtime_audits = [
                value.get("fox_gd_residual_triton_runtime_audit")
                for value in resume["runtime_state"].values()
                if value.get("fox_gd_residual_triton_runtime_audit") is not None
            ]
            if not runtime_audits:
                raise RuntimeError("Flash training Triton runtime audit is missing.")
            for audit in runtime_audits:
                if int(audit["grouped_calls"]) <= 0 or int(audit["selected_calls"]) <= 0:
                    raise RuntimeError(f"Flash training missed Triton calls: {audit}")
                if int(audit["grouped_fallbacks"]) or int(audit["selected_fallbacks"]):
                    raise RuntimeError(f"Flash training recorded a fallback: {audit}")
                if audit["actual_core_dtype"] != "float32":
                    raise RuntimeError(f"Flash training core dtype failed: {audit}")
        for role in ("last", "best"):
            checkpoint = result[f"{role}_checkpoint"]
            if not checkpoint["finite_metrics"]:
                raise RuntimeError("Checkpoint contains non-finite metrics.")
            if sha256_file(Path(checkpoint["path"])) != checkpoint["file_sha256"]:
                raise RuntimeError("Checkpoint hash changed after training.")

    def run_training(
        self,
        descriptor: dict[str, Any],
        phase: str,
        *,
        require_controlled_resume: bool,
    ) -> dict[str, Any]:
        result_path = self._training_result_path(descriptor, phase)
        if result_path.exists():
            existing = load_json(result_path)
            if existing.get("status") == "completed":
                if require_controlled_resume:
                    evidence = result_path.parent / "controlled-stop-evidence.json"
                    if not evidence.exists():
                        raise RuntimeError("Controlled resume evidence is missing.")
                self._audit_training(existing, phase)
                return existing
        os.environ["GDN_KERNEL_DTYPE"] = GDN_KERNEL_DTYPE[
            descriptor["train_precision"]
        ]
        log_path = self.log_dir / f"train-{phase}-{descriptor['descriptor_id']}.log"
        command = self._training_command(descriptor, phase)
        controlled_observed = False
        max_invocations = 4 if require_controlled_resume else 3
        for invocation in range(max_invocations):
            current = f"train:{phase}:{descriptor['descriptor_id']}:try{invocation + 1}"
            return_code = self.run_process(command, log_path, current)
            result = load_json(result_path)
            if return_code == 75 and result.get("status") == "controlled_stop":
                controlled_observed = True
                atomic_write_json(
                    result_path.parent / "controlled-stop-evidence.json",
                    result,
                )
                continue
            if return_code == 0 and result.get("status") == "completed":
                if require_controlled_resume and not controlled_observed:
                    evidence = result_path.parent / "controlled-stop-evidence.json"
                    if not evidence.exists():
                        raise RuntimeError("Controlled resume evidence is missing.")
                self._audit_training(result, phase)
                return result
            error = str(result.get("error", ""))
            hard = any(
                token in error.lower()
                for token in (
                    "out of memory",
                    "non-finite",
                    "nan",
                    "identity mismatch",
                    "triton kernel",
                    "fallback",
                )
            )
            if hard or invocation + 1 >= max_invocations:
                raise RuntimeError(f"Training failed: {result}")
        raise RuntimeError("Training invocation loop exhausted.")

    @staticmethod
    def source_from_result(
        result: dict[str, Any],
        role: str = "last",
    ) -> dict[str, Any]:
        checkpoint = result[f"{role}_checkpoint"]
        return {
            "source_id": (
                f"{result['machine']}-{result['model']}-s{result['seed']}-"
                f"{result['train_precision']}-{role}"
            ),
            "machine": result["machine"],
            "model": result["model"],
            "seed": int(result["seed"]),
            "train_precision": result["train_precision"],
            "checkpoint_role": role,
            "checkpoint_path": checkpoint["path"],
            "checkpoint_file_sha256": checkpoint["file_sha256"],
            "checkpoint_model_state_sha256": checkpoint[
                "model_state_sha256"
            ],
        }

    def run_training_smokes(self) -> list[dict[str, Any]]:
        self.current_phase = "training_smoke"
        sources = []
        stress_records = []
        for descriptor in training_descriptors(self.machine):
            result = self.run_training(
                descriptor,
                "smoke",
                require_controlled_resume=True,
            )
            sources.append(self.source_from_result(result))
            self.completed_nodes += 1
            if descriptor["model"] == "flash":
                stress = self.run_training(
                    descriptor,
                    "stress",
                    require_controlled_resume=False,
                )
                runtime = stress["resume_audit"]["runtime_state"]
                counters = [
                    int(value["fox_gd_residual_train_forward_count"])
                    for value in runtime.values()
                ]
                if not counters or any(value < 2060 for value in counters):
                    raise RuntimeError("Flash full-injection stress counter audit failed.")
                stress_records.append(
                    {
                        "descriptor_id": descriptor["descriptor_id"],
                        "status": stress["status"],
                        "runtime_counters": counters,
                    }
                )
                self.completed_nodes += 1
        atomic_write_json(
            self.root / "gates" / "training-smoke.json",
            {
                "status": "passed",
                "machine": self.machine,
                "sources": sources,
                "stress": stress_records,
                "recorded_at_utc": utc_now(),
            },
        )
        return sources

    def _old_canary_sources(self) -> list[dict[str, Any]]:
        old = (
            REPO_ROOT
            / "zoology/experiments/flash_vqg/scripts/"
            / "20260725-01-current-baselines-longer-mqar/outputs"
        )
        if self.machine == "3090":
            old = old / "machines" / "3090"
        sources = []
        suffix = "-3090" if self.machine == "3090" else ""
        for model in ("gdn", "flash"):
            result_path = (
                old
                / "formal"
                / f"{model}-s123-fixedinit-s124-d123-b64ga4{suffix}-formal"
                / "result.json"
            )
            result = load_json(result_path)
            checkpoint = result["last_checkpoint"]
            sources.append(
                {
                    "source_id": f"{self.machine}-{model}-legacy-fp32-canary",
                    "machine": self.machine,
                    "model": model,
                    "seed": 123,
                    "train_precision": "legacy_fp32",
                    "checkpoint_role": "last",
                    "checkpoint_path": checkpoint["path"],
                    "checkpoint_file_sha256": checkpoint["file_sha256"],
                    "checkpoint_model_state_sha256": checkpoint[
                        "model_state_sha256"
                    ],
                }
            )
        return sources

    def _old_result_path(self, model: str) -> Path:
        old = (
            REPO_ROOT
            / "zoology/experiments/flash_vqg/scripts/"
            / "20260725-01-current-baselines-longer-mqar/outputs"
        )
        if self.machine == "3090":
            old = old / "machines" / "3090"
        suffix = "-3090" if self.machine == "3090" else ""
        return (
            old
            / "formal"
            / f"{model}-s123-fixedinit-s124-d123-b64ga4{suffix}-formal"
            / "result.json"
        )

    def audit_canaries(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        audits = []
        for model in ("gdn", "flash"):
            old_result = load_json(self._old_result_path(model))
            metrics = old_result["last_checkpoint"]["metrics"]
            for record in records:
                if record.get("model") != model or int(record["num_examples"]) != 1000:
                    continue
                expected = float(
                    metrics[f"valid/mqar_case/accuracy-{record['shape']}"]
                )
                delta = abs(float(record["accuracy"]) - expected)
                audits.append(
                    {
                        "model": model,
                        "shape": record["shape"],
                        "num_examples": 1000,
                        "accuracy": record["accuracy"],
                        "reference_accuracy": expected,
                        "accuracy_delta": delta,
                        "passed": delta <= 1e-12,
                    }
                )
        if len(audits) != 16 or not all(row["passed"] for row in audits):
            raise RuntimeError(f"Legacy standard canary audit failed: {audits}")
        return audits

    def run_eval_smokes(self, sources: list[dict[str, Any]]) -> None:
        self.current_phase = "capacity_search"
        batch_gate = self.root / "gates" / "batch-profile.json"
        if not batch_gate.exists() or load_json(batch_gate).get("status") != "passed":
            search_batch_profiles(self.event_runner, sources)
        self.current_phase = "eval_smoke"
        records_path = self.root / "gates" / "eval-smoke.json"
        if not records_path.exists() or load_json(records_path).get("status") != "passed":
            records = run_smoke_evaluations(self.event_runner, sources)
            atomic_write_json(
                records_path,
                {
                    "status": "passed",
                    "machine": self.machine,
                    "events": records,
                    "recorded_at_utc": utc_now(),
                },
            )
        self.current_phase = "legacy_canary"
        canary_path = self.root / "gates" / "legacy-canary.json"
        if not canary_path.exists() or load_json(canary_path).get("status") != "passed":
            records = run_full_evaluations(
                self.event_runner,
                self._old_canary_sources(),
                mode="canary",
                eval_precisions=("fp32",),
            )
            audits = self.audit_canaries(records)
            atomic_write_json(
                canary_path,
                {
                    "status": "passed",
                    "machine": self.machine,
                    "events": records,
                    "standard_accuracy_audit": audits,
                    "recorded_at_utc": utc_now(),
                },
            )

    def write_local_smoke_gate(self) -> None:
        paths = [
            self.root / "preflight.json",
            self.root / "gates" / "training-smoke.json",
            self.root / "gates" / "batch-profile.json",
            self.root / "gates" / "eval-smoke.json",
            self.root / "gates" / "legacy-canary.json",
        ]
        payloads = [load_json(path) for path in paths]
        if any(payload.get("status") != "passed" for payload in payloads):
            raise RuntimeError("A local smoke gate has not passed.")
        payload = {
            "experiment_id": EXPERIMENT_ID,
            "machine": self.machine,
            "status": "passed",
            "gate_files": [
                {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for path in paths
            ],
            "binding_sha256": stable_json_sha256(payloads),
            "recorded_at_utc": utc_now(),
        }
        atomic_write_json(
            self.root / "gates" / "LOCAL_SMOKE_PASSED.json",
            payload,
        )

    def smoke(self) -> None:
        self.run_preflight()
        training_gate = self.root / "gates" / "training-smoke.json"
        if training_gate.exists() and load_json(training_gate).get("status") == "passed":
            sources = load_json(training_gate)["sources"]
        else:
            sources = self.run_training_smokes()
        self.run_eval_smokes(sources)
        self.write_local_smoke_gate()
        self.status("waiting_global_gate", "LOCAL_SMOKE_PASSED")

    def wait_for_global_gate(self) -> None:
        self.current_phase = "wait_global_gate"
        gate = self.root / "gates" / "GLOBAL_FORMAL_GATE.json"
        while True:
            if gate.exists():
                payload = load_json(gate)
                if payload.get("status") == "passed":
                    return
            self.heartbeat("waiting:GLOBAL_FORMAL_GATE")
            time.sleep(30)

    def _formal_sources(self, result: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            self.source_from_result(result, role)
            for role in ("last", "best")
        ]

    def formal(self) -> None:
        gate = self.root / "gates" / "GLOBAL_FORMAL_GATE.json"
        if not gate.exists() or load_json(gate).get("status") != "passed":
            raise RuntimeError("Formal queue requires the global smoke gate.")
        self.current_phase = "formal"
        detail_path = self.root / "formal-detail.json"
        detail = load_json(detail_path) if detail_path.exists() else []
        completed_descriptors = {
            row["descriptor_id"]
            for row in detail
            if row.get("status") == "completed"
        }
        deferred_first_sources = None
        first_descriptor_id = "2080ti-gdn-s123-fp32"
        first_row = next(
            (
                row
                for row in detail
                if row.get("descriptor_id") == first_descriptor_id
            ),
            None,
        )
        if (
            self.machine == "2080ti"
            and first_row is not None
            and len(first_row.get("evaluation", [])) < 52
        ):
            deferred_first_sources = self._formal_sources(
                first_row["training_result"]
            )
        for index, descriptor in enumerate(training_descriptors(self.machine)):
            if descriptor["descriptor_id"] in completed_descriptors:
                continue
            result = self.run_training(
                descriptor,
                "formal",
                require_controlled_resume=False,
            )
            sources = self._formal_sources(result)
            priority_first = (
                self.machine == "2080ti"
                and descriptor["model"] == "gdn"
                and descriptor["seed"] == 123
                and descriptor["train_precision"] == "fp32"
            )
            if priority_first:
                priority_records = run_full_evaluations(
                    self.event_runner,
                    sources,
                    mode="formal-priority",
                    eval_precisions=("fp32", "fp16"),
                    shapes_override=((8190, 512, 500), (8190, 2047, 500)),
                )
                atomic_write_json(
                    self.root / "gates" / "FIRST_2080_LONGEST_PASSED.json",
                    {
                        "status": "passed",
                        "machine": self.machine,
                        "events": priority_records,
                        "recorded_at_utc": utc_now(),
                    },
                )
                deferred_first_sources = sources
                eval_records = priority_records
            else:
                eval_records = run_full_evaluations(
                    self.event_runner,
                    sources,
                    mode="formal",
                )
            detail.append(
                {
                    "descriptor_id": descriptor["descriptor_id"],
                    "status": "completed",
                    "training_result": result,
                    "evaluation": eval_records,
                    "completed_at_utc": utc_now(),
                }
            )
            atomic_write_json(detail_path, detail)
            self.completed_nodes += 1
            is_second_2080_run = (
                self.machine == "2080ti"
                and descriptor["model"] == "flash"
                and descriptor["seed"] == 123
                and descriptor["train_precision"] == "fp32"
            )
            if is_second_2080_run and deferred_first_sources is not None:
                remaining_shapes = tuple(
                    case
                    for case in __import__("common").ALL_EVAL_CASES
                    if case not in {(8190, 512, 500), (8190, 2047, 500)}
                )
                deferred_records = run_full_evaluations(
                    self.event_runner,
                    deferred_first_sources,
                    mode="formal-deferred",
                    shapes_override=remaining_shapes,
                )
                target_row = next(
                    row
                    for row in detail
                    if row["descriptor_id"] == first_descriptor_id
                )
                target_row["evaluation"].extend(deferred_records)
                atomic_write_json(detail_path, detail)
                deferred_first_sources = None
        if deferred_first_sources is not None:
            remaining_shapes = tuple(
                case
                for case in __import__("common").ALL_EVAL_CASES
                if case not in {(8190, 512, 500), (8190, 2047, 500)}
            )
            deferred_records = run_full_evaluations(
                self.event_runner,
                deferred_first_sources,
                mode="formal-deferred",
                shapes_override=remaining_shapes,
            )
            target_row = next(
                row
                for row in detail
                if row["descriptor_id"] == first_descriptor_id
            )
            target_row["evaluation"].extend(deferred_records)
            atomic_write_json(detail_path, detail)
        self.status("completed", "formal-complete", formal_runs=len(detail))


def acquire_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"Queue lock is already held: {path}") from exc
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("smoke", "formal", "all"), default="all")
    args = parser.parse_args()
    lock = acquire_lock(output_root() / "queue.lock")
    queue = MachineQueue()
    try:
        queue.status("running", "queue-start")
        if args.phase in {"smoke", "all"}:
            queue.smoke()
        if args.phase == "all":
            queue.wait_for_global_gate()
        if args.phase in {"formal", "all"}:
            queue.formal()
        return 0
    except BaseException as exc:
        queue.status(
            "failed",
            "queue-failed",
            error=f"{type(exc).__name__}: {exc}",
            traceback_tail="\n".join(traceback.format_exc().splitlines()[-40:]),
        )
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
