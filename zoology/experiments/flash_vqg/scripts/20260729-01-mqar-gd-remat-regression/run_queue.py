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

LOCAL_DIR = Path(__file__).resolve().parent
if str(LOCAL_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_DIR))

from common import (
    FLASH_ROOT,
    FORMAL_ORDER,
    PYTHON,
    REPO_ROOT,
    VARIANTS,
    atomic_write_json,
    load_json,
    run_root,
    run_tag,
    utc_now,
)
from experiment import result_path


EXPERIMENT_SCRIPT = LOCAL_DIR / "experiment.py"
TRAJECTORY_SCRIPT = LOCAL_DIR / "trajectory.py"
EVALUATE_SCRIPT = LOCAL_DIR / "evaluate.py"
COLLECT_SCRIPT = LOCAL_DIR / "collect_artifacts.py"


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


class Queue:
    def __init__(self):
        self.root = run_root()
        self.logs = self.root / "logs"
        self.status_path = self.root / "status.json"
        self.current_phase = "initializing"

    def status(self, state: str, **extra: Any) -> None:
        atomic_write_json(
            self.status_path,
            {
                "run_tag": run_tag(),
                "status": state,
                "phase": self.current_phase,
                "updated_at_utc": utc_now(),
                **extra,
            },
        )

    def run_process(
        self,
        name: str,
        command: list[str],
        *,
        accepted: tuple[int, ...] = (0,),
    ) -> int:
        if name != "preflight" and (self.root / "preflight.json").exists():
            self.source_guard()
        log_path = self.logs / f"{name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.status("running", command=command, log_path=str(log_path))
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
                atomic_write_json(
                    self.root / "heartbeat.json",
                    {"phase": self.current_phase, "name": name, "updated_at_utc": utc_now()},
                )
                time.sleep(5)
        code = int(process.returncode)
        if code not in accepted:
            raise RuntimeError(f"Process failed with code {code}: {name}, {log_path}")
        return code

    @staticmethod
    def _git_value(root: Path, *args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            text=True,
            capture_output=True,
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    def source_guard(self) -> None:
        payload = load_json(self.root / "preflight.json")
        expected = payload["environment"]
        actual = {
            "zoology_commit": self._git_value(REPO_ROOT, "rev-parse", "HEAD"),
            "flash_commit": self._git_value(FLASH_ROOT, "rev-parse", "HEAD"),
            "zoology_status": self._git_value(REPO_ROOT, "status", "--short"),
            "flash_status": self._git_value(FLASH_ROOT, "status", "--short"),
        }
        if actual != {key: expected[key] for key in actual}:
            raise RuntimeError(f"Source identity changed after preflight: {actual}")

    def require_preflight(self) -> None:
        payload = load_json(self.root / "preflight.json")
        if payload.get("status") != "passed":
            raise RuntimeError("Preflight gate did not pass.")

    def preflight(self) -> None:
        self.current_phase = "preflight"
        path = self.root / "preflight.json"
        if not path.exists() or load_json(path).get("status") != "passed":
            self.run_process("preflight", [str(PYTHON), str(EXPERIMENT_SCRIPT), "preflight"])
        self.require_preflight()

    def trajectory(self) -> None:
        self.current_phase = "trajectory"
        path = self.root / "trajectory" / "summary.json"
        if not path.exists() or load_json(path).get("status") != "passed":
            self.run_process("trajectory", [str(PYTHON), str(TRAJECTORY_SCRIPT)])
        if load_json(path).get("status") != "passed":
            raise RuntimeError("Trajectory gate did not pass.")

    def train(self, variant: str, seed: int, phase: str) -> dict[str, Any]:
        path = result_path(variant, seed, phase)
        if path.exists() and load_json(path).get("status") == "completed":
            return load_json(path)
        command = [
            str(PYTHON),
            str(EXPERIMENT_SCRIPT),
            "train",
            "--variant",
            variant,
            "--seed",
            str(seed),
            "--phase",
            phase,
        ]
        name = f"train-{phase}-{variant}-s{seed}"
        code = self.run_process(name, command, accepted=(0, 75))
        if code == 75:
            self.run_process(name, command)
        result = load_json(path)
        if result.get("status") != "completed":
            raise RuntimeError(f"Training did not complete: {path}")
        return result

    def smoke(self) -> None:
        self.current_phase = "smoke"
        for variant in VARIANTS:
            self.train(variant, 124, "smoke")
        path = self.root / "evaluation" / "smoke-summary.json"
        if not path.exists() or load_json(path).get("status") != "passed":
            self.run_process(
                "evaluate-smoke",
                [str(PYTHON), str(EVALUATE_SCRIPT), "--phase", "smoke"],
            )

    def formal(self) -> None:
        self.current_phase = "formal"
        for variant, seed in FORMAL_ORDER:
            self.train(variant, seed, "formal")

    def evaluate(self) -> None:
        self.current_phase = "evaluate"
        path = self.root / "evaluation" / "formal-summary.json"
        if not path.exists() or load_json(path).get("status") != "passed":
            self.run_process(
                "evaluate-formal",
                [str(PYTHON), str(EVALUATE_SCRIPT), "--phase", "formal"],
            )

    def collect(self) -> None:
        self.current_phase = "collect"
        code = self.run_process(
            "collect",
            [str(PYTHON), str(COLLECT_SCRIPT)],
            accepted=(0, 2),
        )
        summary = load_json(self.root / "final-summary" / "summary.json")
        if code == 2 or summary.get("status") != "passed":
            self.status("quality_failed", summary=summary)
            raise RuntimeError("A1 failed the registered MQAR non-inferiority gate.")

    def run(self) -> None:
        self.preflight()
        self.trajectory()
        self.smoke()
        self.formal()
        self.evaluate()
        self.collect()
        self.current_phase = "complete"
        self.status("completed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    lock = acquire_lock(run_root() / "queue.lock")
    queue = Queue()
    try:
        queue.run()
        return 0
    except BaseException as exc:
        state = "quality_failed" if queue.current_phase == "collect" else "failed"
        queue.status(
            state,
            error=f"{type(exc).__name__}: {exc}",
            traceback_tail="\n".join(traceback.format_exc().splitlines()[-40:]),
        )
        print(traceback.format_exc(), file=sys.stderr)
        return 1
    finally:
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
