#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _bootstrap_machine() -> str:
    machine = os.environ.get("LONGER_MQAR_MACHINE", "").strip().lower()
    if "--machine" in sys.argv:
        index = sys.argv.index("--machine")
        if index + 1 >= len(sys.argv):
            raise RuntimeError("--machine缺少值.")
        cli_machine = sys.argv[index + 1].strip().lower()
        if machine and machine != cli_machine:
            raise RuntimeError(f"CLI machine与环境不一致: {cli_machine} vs {machine}")
        machine = cli_machine
    machine = machine or "2080ti"
    if machine not in {"2080ti", "3090"}:
        raise RuntimeError(f"machine必须为2080ti或3090, 实际为{machine!r}.")
    os.environ["LONGER_MQAR_MACHINE"] = machine
    return machine


EXPERIMENT_ID = "20260725-01-current-baselines-longer-mqar"
MACHINE = _bootstrap_machine()
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get("ZOOLOGY_REPO_ROOT", "/home/lyj/mnt/project/zoology")).resolve()
OUTPUT_ROOT = SCRIPT_DIR / "outputs" if MACHINE == "2080ti" else SCRIPT_DIR / "outputs/machines" / MACHINE
QUEUE_DIR = OUTPUT_ROOT / "queue"
GATE_DIR = OUTPUT_ROOT / "gates"
EXPERIMENT = SCRIPT_DIR / "experiment.py"
EVAL_RUNNER = SCRIPT_DIR / "longer_mqar_runner.py"
JOB_ORDER = (("flash", 123), ("gdn", 123), ("flash", 124), ("gdn", 124), ("flash", 125), ("gdn", 125))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


class FormalQueue:
    def __init__(self, resume: bool):
        self.resume = resume
        self.status_path = QUEUE_DIR / "status.json"
        self.started_at = utc_now()
        QUEUE_DIR.mkdir(parents=True, exist_ok=True)
        (QUEUE_DIR / "FAILED.json").unlink(missing_ok=True)

    def status(self, phase: str, current: str = "", **extra: Any) -> None:
        write_json(self.status_path, {
            "experiment_id": EXPERIMENT_ID,
            "machine": MACHINE,
            "status": "running",
            "phase": phase,
            "current": current,
            "started_at_utc": self.started_at,
            "updated_at_utc": utc_now(),
            **extra,
        })

    def run_command(
        self,
        label: str,
        command: list[str],
        phase: str,
        **status_extra: Any,
    ) -> None:
        log_path = QUEUE_DIR / "logs" / f"{label}.log"
        command_status_path = QUEUE_DIR / "commands" / f"{label}.json"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command_started_at = utc_now()
        self.status(
            phase,
            label,
            command=command,
            log_path=str(log_path),
            command_status_path=str(command_status_path),
            **status_extra,
        )
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["TRITON_F32_DEFAULT"] = "ieee"
        env["GDN_KERNEL_DTYPE"] = "float32"
        env["NVIDIA_TF32_OVERRIDE"] = "0"
        env["LONGER_MQAR_MACHINE"] = MACHINE
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n[{utc_now()}] START {' '.join(command)}\n")
            log.flush()
            proc = subprocess.run(command, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
            log.write(f"[{utc_now()}] END returncode={proc.returncode}\n")
        write_json(command_status_path, {
            "machine": MACHINE,
            "label": label,
            "phase": phase,
            "command": command,
            "log_path": str(log_path),
            "started_at_utc": command_started_at,
            "ended_at_utc": utc_now(),
            "returncode": proc.returncode,
            "status": "completed" if proc.returncode == 0 else "failed",
            **status_extra,
        })
        if proc.returncode != 0:
            raise RuntimeError(f"队列命令失败: {label}, returncode={proc.returncode}, log={log_path}")

    def preflight(self) -> None:
        self.run_command(
            "preflight",
            [sys.executable, str(EXPERIMENT), "preflight", "--output", str(OUTPUT_ROOT / "preflight.json")],
            "preflight",
        )

    def train_matrix(self, run_type: str) -> Path:
        for index, (model, seed) in enumerate(JOB_ORDER, start=1):
            command = [
                sys.executable, str(EXPERIMENT), "train",
                "--model", model, "--seed", str(seed), "--run-type", run_type,
            ]
            if self.resume:
                command.append("--resume")
            self.run_command(
                f"{run_type}-train-{model}-s{seed}",
                command,
                f"{run_type}_training",
                completed=index - 1,
                expected=len(JOB_ORDER),
            )
        manifest = OUTPUT_ROOT / run_type / "source-manifest.json"
        self.run_command(
            f"{run_type}-build-manifest",
            [sys.executable, str(EXPERIMENT), "build-manifest", "--run-type", run_type, "--output", str(manifest)],
            f"{run_type}_checkpoint_audit",
        )
        return manifest

    def shape_smoke(self) -> None:
        for model in ("flash", "gdn"):
            self.run_command(
                f"shape-smoke-{model}",
                [
                    sys.executable, str(EXPERIMENT), "shape-smoke", "--model", model,
                    "--output", str(OUTPUT_ROOT / "shape-smoke" / f"{model}.json"),
                ],
                "shape_smoke",
            )

    def eval_queue(self, manifest: Path, mode: str) -> Path:
        output = OUTPUT_ROOT / ("smoke-dag-eval" if mode == "smoke-dag" else "formal-eval")
        command = [
            sys.executable, str(EVAL_RUNNER),
            "--manifest", str(manifest), "--output-dir", str(output), "--mode", mode,
        ]
        if self.resume:
            command.append("--resume")
        self.run_command(f"eval-{mode}", command, f"eval_{mode}")
        return output

    def run(self) -> dict[str, Any]:
        try:
            self.preflight()
            self.shape_smoke()
            smoke_manifest = self.train_matrix("smoke")
            smoke_eval = self.eval_queue(smoke_manifest, "smoke-dag")
            smoke_verification = json.loads((smoke_eval / "verification.json").read_text(encoding="utf-8"))
            if smoke_verification.get("status") != "completed":
                raise RuntimeError(f"端到端smoke未完成: {smoke_verification}")
            write_json(GATE_DIR / "SMOKE_DONE.json", {
                "status": "passed",
                "experiment_id": EXPERIMENT_ID,
                "machine": MACHINE,
                "training_smoke_runs": 6,
                "eval_verification": str(smoke_eval / "verification.json"),
                "recorded_at_utc": utc_now(),
            })

            formal_manifest = self.train_matrix("formal")
            formal_eval = self.eval_queue(formal_manifest, "formal")
            verification = json.loads((formal_eval / "verification.json").read_text(encoding="utf-8"))
            if verification.get("status") != "completed":
                raise RuntimeError(f"正式eval未完成: {verification}")
            done = {
                "experiment_id": EXPERIMENT_ID,
                "machine": MACHINE,
                "status": "completed",
                "started_at_utc": self.started_at,
                "ended_at_utc": utc_now(),
                "training_runs": 6,
                "formal_manifest": str(formal_manifest),
                "formal_eval_verification": str(formal_eval / "verification.json"),
                "report_phase_ready": True,
            }
            write_json(QUEUE_DIR / "DONE.json", done)
            write_json(self.status_path, {**done, "phase": "done", "updated_at_utc": utc_now()})
            return done
        except BaseException as exc:
            failed = {
                "experiment_id": EXPERIMENT_ID,
                "machine": MACHINE,
                "status": "failed",
                "started_at_utc": self.started_at,
                "ended_at_utc": utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            write_json(QUEUE_DIR / "FAILED.json", failed)
            write_json(self.status_path, {**failed, "phase": "failed", "updated_at_utc": utc_now()})
            raise


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="当前基线Longer-MQAR可恢复fail-fast自动队列.")
    root.add_argument("--machine", choices=("2080ti", "3090"), default=MACHINE)
    root.add_argument("--resume", action="store_true")
    return root


def main() -> int:
    args = parser().parse_args()
    if args.machine != MACHINE:
        raise RuntimeError(f"bootstrap machine异常: {MACHINE} vs {args.machine}")
    done = FormalQueue(resume=args.resume).run()
    print(json.dumps(done, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
