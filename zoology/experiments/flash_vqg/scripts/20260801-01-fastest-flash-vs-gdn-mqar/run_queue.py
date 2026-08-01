#!/usr/bin/env python3
from __future__ import annotations

import fcntl
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

LOCAL_DIR = Path(__file__).resolve().parent
if str(LOCAL_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_DIR))

from common import (
    ARMS,
    FASTEST,
    FLASH_ROOT,
    FORMAL_ORDER,
    GDN,
    PYTHON,
    REPO_ROOT,
    atomic_write_json,
    load_json,
    run_root,
    run_tag,
    sha256_file,
    stable_json_sha256,
    utc_now,
)
import experiment


EXPERIMENT_SCRIPT = LOCAL_DIR / "experiment.py"
EVALUATE_SCRIPT = LOCAL_DIR / "evaluate.py"
ANALYZE_SCRIPT = LOCAL_DIR / "analyze.py"


def git_value(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        text=True,
        capture_output=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


class Queue:
    def __init__(self):
        self.root = run_root()
        self.phase = "initializing"
        self.completed_jobs = 0

    def status(self, value: str, **extra: Any) -> None:
        atomic_write_json(
            self.root / "status.json",
            {
                "status": value,
                "phase": self.phase,
                "run_tag": run_tag(),
                "completed_jobs": self.completed_jobs,
                "updated_at_utc": utc_now(),
                **extra,
            },
        )

    def source_guard(self) -> None:
        expected = load_json(self.root / "preflight.json")["environment"]
        actual = {
            "zoology_commit": git_value(REPO_ROOT, "rev-parse", "HEAD"),
            "flash_commit": git_value(FLASH_ROOT, "rev-parse", "HEAD"),
            "zoology_status": git_value(REPO_ROOT, "status", "--short"),
            "flash_status": git_value(FLASH_ROOT, "status", "--short"),
        }
        reference = {key: expected[key] for key in actual}
        if actual != reference:
            raise RuntimeError(f"Source changed after preflight: {actual}")

    def process(
        self,
        name: str,
        command: list[str],
        *,
        accepted: tuple[int, ...] = (0,),
    ) -> int:
        log = self.root / "logs" / f"{name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        self.status("running", job=name, command=command, log=str(log))
        env = os.environ.copy()
        env["GDN_KERNEL_DTYPE"] = "bfloat16"
        with log.open("a", encoding="utf-8") as handle:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            while process.poll() is None:
                atomic_write_json(
                    self.root / "heartbeat.json",
                    {"phase": self.phase, "job": name, "updated_at_utc": utc_now()},
                )
                time.sleep(5)
        code = int(process.returncode)
        if code not in accepted:
            raise RuntimeError(f"Process failed: {name}, code={code}, log={log}")
        return code

    def _current_result_hash(self, arm: str, seed: int, phase: str) -> str:
        config = experiment.build_config(arm, seed, phase)
        return stable_json_sha256(experiment.normalized_config(config))

    def _audit_existing(self, result: dict[str, Any], arm: str, seed: int, phase: str) -> None:
        if result.get("status") != "completed":
            raise RuntimeError("Existing result is incomplete.")
        if result.get("normalized_config_sha256") != self._current_result_hash(arm, seed, phase):
            raise RuntimeError("Existing result config identity mismatch.")
        for role in ("last", "best"):
            checkpoint = result[f"{role}_checkpoint"]
            path = Path(checkpoint["path"])
            if not path.is_file() or sha256_file(path) != checkpoint["file_sha256"]:
                raise RuntimeError(f"Existing checkpoint identity mismatch: {path}")

    def train(self, arm: str, seed: int, phase: str, controlled_resume: bool) -> None:
        path = experiment.result_path(arm, seed, phase)
        if path.exists() and load_json(path).get("status") == "completed":
            self._audit_existing(load_json(path), arm, seed, phase)
            if controlled_resume and not (path.parent / "controlled-stop-evidence.json").exists():
                raise RuntimeError("Controlled resume evidence is missing.")
            return
        command = [
            str(PYTHON),
            str(EXPERIMENT_SCRIPT),
            "train",
            "--arm",
            arm,
            "--seed",
            str(seed),
            "--phase",
            phase,
        ]
        observed_stop = False
        for attempt in range(4 if controlled_resume else 2):
            self.source_guard()
            name = f"train-{phase}-{arm}-s{seed}-try{attempt + 1}"
            code = self.process(name, command, accepted=(0, 75))
            result = load_json(path)
            if code == 75 and result.get("status") == "controlled_stop":
                observed_stop = True
                atomic_write_json(path.parent / "controlled-stop-evidence.json", result)
                continue
            if code == 0 and result.get("status") == "completed":
                self._audit_existing(result, arm, seed, phase)
                if controlled_resume and not observed_stop:
                    raise RuntimeError("Controlled stop was not observed.")
                self.completed_jobs += 1
                return
            raise RuntimeError(f"Training failed: {result}")
        raise RuntimeError(f"Training attempts exhausted: {arm}, {seed}, {phase}.")

    def run_command(self, stage: str, command: list[str]) -> None:
        self.source_guard()
        self.process(stage, command)
        self.completed_jobs += 1

    def run(self) -> int:
        self.phase = "preflight"
        self.process("preflight", [str(PYTHON), str(EXPERIMENT_SCRIPT), "preflight"])
        if load_json(self.root / "preflight.json").get("status") != "passed":
            raise RuntimeError("Preflight did not pass.")

        self.phase = "smoke"
        for arm in ARMS:
            self.train(arm, 123, "smoke", controlled_resume=True)
        self.run_command("evaluate-smoke", [str(PYTHON), str(EVALUATE_SCRIPT), "smoke"])

        self.phase = "screen"
        for arm in ARMS:
            self.train(arm, 123, "screen", controlled_resume=False)
        self.run_command("profile-eval-batches", [str(PYTHON), str(EVALUATE_SCRIPT), "profile"])
        self.run_command("evaluate-screen", [str(PYTHON), str(EVALUATE_SCRIPT), "screen"])
        self.run_command(
            "analyze-screen",
            [str(PYTHON), str(ANALYZE_SCRIPT), "--phase", "screen"],
        )

        self.phase = "formal"
        for arm, seed in FORMAL_ORDER:
            self.train(arm, seed, "formal", controlled_resume=False)
        self.run_command("evaluate-formal", [str(PYTHON), str(EVALUATE_SCRIPT), "formal"])
        self.run_command("evaluate-repro", [str(PYTHON), str(EVALUATE_SCRIPT), "repro"])
        self.run_command(
            "analyze-formal",
            [str(PYTHON), str(ANALYZE_SCRIPT), "--phase", "formal"],
        )
        summary = load_json(self.root / "analysis" / "formal-summary.json")
        self.phase = "complete"
        self.status("completed", summary=summary["quality_decision"])
        atomic_write_json(
            self.root / "DONE.json",
            {
                "status": "completed",
                "run_tag": run_tag(),
                "completed_jobs": self.completed_jobs,
                "completed_at_utc": utc_now(),
            },
        )
        return 0


def acquire_lock():
    path = run_root() / "queue.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", encoding="utf-8")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def main() -> int:
    lock = acquire_lock()
    queue = Queue()
    try:
        return queue.run()
    except BaseException as exc:
        queue.status(
            "failed",
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
        print(traceback.format_exc(), file=sys.stderr)
        return 1
    finally:
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
