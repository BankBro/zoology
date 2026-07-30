#!/usr/bin/env python3
from __future__ import annotations

import fcntl
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

LOCAL_DIR = Path(__file__).resolve().parent
if str(LOCAL_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_DIR))

from common import (
    FLASH_ROOT,
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
EVALUATE_SCRIPT = LOCAL_DIR / "evaluate.py"


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

    def status(self, value: str, **extra) -> None:
        atomic_write_json(
            self.root / "status.json",
            {
                "status": value,
                "phase": self.phase,
                "run_tag": run_tag(),
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
        if actual != {key: expected[key] for key in actual}:
            raise RuntimeError(f"Source changed after preflight: {actual}")

    def process(self, name: str, command: list[str], accepted=(0,)) -> int:
        log = self.root / "logs" / f"{name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        self.status("running", job=name, command=command, log=str(log))
        with log.open("a", encoding="utf-8") as handle:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=os.environ.copy(),
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

    def train(self, variant: str, phase: str) -> None:
        path = result_path(variant, phase)
        if path.exists() and load_json(path).get("status") == "completed":
            return
        self.source_guard()
        self.process(
            f"train-{phase}-{variant}",
            [
                str(PYTHON),
                str(EXPERIMENT_SCRIPT),
                "train",
                "--variant",
                variant,
                "--phase",
                phase,
            ],
        )

    def run(self) -> int:
        self.phase = "preflight"
        self.process("preflight", [str(PYTHON), str(EXPERIMENT_SCRIPT), "preflight"])
        if load_json(self.root / "preflight.json").get("status") != "passed":
            raise RuntimeError("Preflight did not pass.")
        self.phase = "smoke"
        for variant in VARIANTS:
            self.train(variant, "smoke")
        self.phase = "screen"
        for variant in VARIANTS:
            self.train(variant, "screen")
        self.phase = "evaluate"
        code = self.process(
            "evaluate",
            [str(PYTHON), str(EVALUATE_SCRIPT)],
            accepted=(0, 2, 3),
        )
        summary = load_json(self.root / "evaluation/summary.json")
        self.phase = "complete"
        self.status(summary["status"], summary=summary)
        return code


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
