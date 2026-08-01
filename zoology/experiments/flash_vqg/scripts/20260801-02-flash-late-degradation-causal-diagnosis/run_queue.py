#!/usr/bin/env python3
from __future__ import annotations

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

from causal_common import (
    INTERACTION_ARMS,
    PYTHON,
    REPO_ROOT,
    SEEDS,
    SINGLE_FACTOR_ARMS,
    atomic_write_json,
    load_json,
    run_root,
    run_tag,
    sha256_file,
    stable_json_sha256,
    utc_now,
)
import analyze
import experiment


EXPERIMENT_SCRIPT = LOCAL_DIR / "experiment.py"
EVALUATE_SCRIPT = LOCAL_DIR / "evaluate.py"
ANALYZE_SCRIPT = LOCAL_DIR / "analyze.py"


class Queue:
    def __init__(self) -> None:
        self.root = run_root()
        self.phase = "initializing"
        self.completed_jobs = 0
        self.started_at_utc = utc_now()
        self.executed: list[dict[str, Any]] = []

    def status(self, value: str, **extra: Any) -> None:
        atomic_write_json(
            self.root / "status.json",
            {
                "status": value,
                "phase": self.phase,
                "run_tag": run_tag(),
                "completed_jobs": self.completed_jobs,
                "started_at_utc": self.started_at_utc,
                "updated_at_utc": utc_now(),
                "executed": self.executed,
                **extra,
            },
        )

    def source_guard(self) -> None:
        expected = load_json(self.root / "preflight.json")["environment"]
        keys = ("zoology_commit", "flash_commit", "zoology_status", "flash_status")
        actual = {
            "zoology_commit": experiment.git_value(experiment.REPO_ROOT, "rev-parse", "HEAD"),
            "flash_commit": experiment.git_value(experiment.FLASH_ROOT, "rev-parse", "HEAD"),
            "zoology_status": experiment.git_value(experiment.REPO_ROOT, "status", "--short"),
            "flash_status": experiment.git_value(experiment.FLASH_ROOT, "status", "--short"),
        }
        if actual != {key: expected[key] for key in keys}:
            raise RuntimeError(f"Source changed after preflight: {actual}")

    def process(self, name: str, command: list[str]) -> None:
        log = self.root / "logs" / f"{name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        self.status("running", job=name, command=command, log=str(log))
        env = os.environ.copy()
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": "0",
                "TRITON_F32_DEFAULT": "ieee",
                "NVIDIA_TF32_OVERRIDE": "0",
                "GDN_KERNEL_DTYPE": "bfloat16",
            }
        )
        started = time.perf_counter()
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
        record = {
            "name": name,
            "return_code": int(process.returncode),
            "wall_clock_sec": time.perf_counter() - started,
            "log": str(log.resolve()),
        }
        self.executed.append(record)
        if process.returncode != 0:
            raise RuntimeError(f"Process failed: {name}, code={process.returncode}, log={log}")
        self.completed_jobs += 1

    def _config_hash(self, arm: str, seed: int, phase: str, gate_mode: str) -> str:
        config = experiment.build_config(arm, seed, phase, gate_mode)
        return stable_json_sha256(experiment.normalized_config(config))

    def audit_existing(self, path: Path, arm: str, seed: int, phase: str, gate_mode: str) -> bool:
        if not path.exists():
            return False
        result = load_json(path)
        if result.get("status") != "completed":
            return False
        if result.get("normalized_config_sha256") != self._config_hash(arm, seed, phase, gate_mode):
            raise RuntimeError(f"Existing result config mismatch: {path}")
        for role in ("last", "best"):
            checkpoint = result[f"{role}_checkpoint"]
            target = Path(checkpoint["path"])
            if not target.is_file() or sha256_file(target) != checkpoint["file_sha256"]:
                raise RuntimeError(f"Existing checkpoint mismatch: {target}")
        return True

    def train(self, arm: str, seed: int, phase: str, gate_mode: str = "fixed") -> None:
        path = experiment.result_path(arm, seed, phase, gate_mode)
        if self.audit_existing(path, arm, seed, phase, gate_mode):
            return
        self.source_guard()
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
            "--gate-mode",
            gate_mode,
        ]
        self.process(f"train-{phase}-{arm}-s{seed}-{gate_mode}", command)
        if not self.audit_existing(path, arm, seed, phase, gate_mode):
            raise RuntimeError(f"Training result audit failed: {path}")

    def ensure_screen(self, arm: str, seed: int, gate_mode: str = "fixed") -> dict[str, Any]:
        self.train(arm, 123, "smoke", gate_mode)
        self.train(arm, seed, "screen", gate_mode)
        return analyze.screen_summary(arm, seed, gate_mode)

    def confirm_candidate(self, candidate: str) -> dict[str, Any]:
        self.ensure_screen("ctrl-bridge", 125)
        self.ensure_screen(candidate, 125)
        decision = analyze.aggregate_effect("ctrl-bridge", candidate, [123, 125])
        if decision["decision"] != "inconclusive":
            return decision
        self.ensure_screen("ctrl-bridge", 124)
        self.ensure_screen(candidate, 124)
        return analyze.aggregate_effect("ctrl-bridge", candidate, [123, 124, 125])

    def reproduce_controls(self) -> dict[str, Any]:
        current = self.ensure_screen("ctrl-current", 123)
        bridge = self.ensure_screen("ctrl-bridge", 123)
        payload = {"current": current, "bridge": bridge}
        if current["retention"] > -0.05:
            payload["decision"] = "fixed_fla_did_not_reproduce"
        elif bridge["retention"] < -0.02:
            payload["decision"] = "current_source_bridge_failed"
        else:
            payload["decision"] = "reproduced"
        atomic_write_json(self.root / "analysis" / "control-reproduction.json", payload)
        return payload

    def fixed_fla_fallback(self) -> dict[str, Any]:
        for seed in (123, 125):
            self.ensure_screen("ctrl-current", seed, "default")
            self.ensure_screen("ctrl-bridge", seed, "default")
        result = analyze.aggregate_effect(
            "ctrl-bridge", "ctrl-current", [123, 125], "default"
        )
        return {"root": "fla-autotune", "default_fla": result}

    def locate_root(self) -> dict[str, Any]:
        for candidate in SINGLE_FACTOR_ARMS:
            row = self.ensure_screen(candidate, 123)
            effect = analyze.paired_effect("ctrl-bridge", candidate, 123)
            if effect["classification"] == "stable":
                continue
            confirmed = self.confirm_candidate(candidate)
            if confirmed["decision"] == "confirmed_cause":
                return {"root": candidate, "confirmation": confirmed, "screen": row}
        for candidate in INTERACTION_ARMS:
            self.ensure_screen(candidate, 123)
            effect = analyze.paired_effect("ctrl-bridge", candidate, 123)
            if effect["classification"] == "stable":
                continue
            confirmed = self.confirm_candidate(candidate)
            if confirmed["decision"] == "confirmed_cause":
                return {"root": candidate, "confirmation": confirmed}
        confirmed = self.confirm_candidate("ctrl-current")
        if confirmed["decision"] == "confirmed_cause":
            return {"root": "three-way-interaction", "confirmation": confirmed}
        return {"root": "unresolved", "confirmation": confirmed}

    def mechanism_matrix(self) -> dict[str, Any]:
        arms = (
            "ctrl-bridge",
            "mechanism-window128",
            "mechanism-boundary64",
            "factor-block",
        )
        for arm in arms:
            for seed in (123, 125):
                self.ensure_screen(arm, seed)
        window = analyze.aggregate_effect("ctrl-bridge", "mechanism-window128", [123, 125])
        boundary = analyze.aggregate_effect("ctrl-bridge", "mechanism-boundary64", [123, 125])
        block = analyze.aggregate_effect("ctrl-bridge", "factor-block", [123, 125])
        if "inconclusive" in {window["decision"], boundary["decision"]}:
            for arm in arms:
                self.ensure_screen(arm, 124)
            window = analyze.aggregate_effect("ctrl-bridge", "mechanism-window128", list(SEEDS))
            boundary = analyze.aggregate_effect("ctrl-bridge", "mechanism-boundary64", list(SEEDS))
            block = analyze.aggregate_effect("ctrl-bridge", "factor-block", list(SEEDS))
        w_cause = window["decision"] == "confirmed_cause"
        b_cause = boundary["decision"] == "confirmed_cause"
        unresolved = "inconclusive" in {window["decision"], boundary["decision"]}
        mechanism = "unresolved" if unresolved else (
            "both" if w_cause and b_cause else (
                "window_span" if w_cause else ("boundary_granularity" if b_cause else "interaction")
            )
        )
        payload = {
            "mechanism": mechanism,
            "window": window,
            "boundary": boundary,
            "block": block,
        }
        atomic_write_json(self.root / "analysis" / "block-mechanism.json", payload)
        return payload

    @staticmethod
    def fastest_correction(root: str, mechanism: dict[str, Any] | None) -> str:
        if root == "factor-block":
            if mechanism and mechanism["mechanism"] == "window_span":
                return "fastest-block64-local1"
            return "fastest-block32-local2"
        if root == "factor-backend":
            return "fastest-selected-torch"
        if root == "factor-chunk":
            return "fastest-chunk2048"
        return "fastest-bridge"

    def formal_pair(self, arms: list[str]) -> None:
        for arm in arms:
            self.train(arm, 123, "formal", "fixed")
            self.train(arm, 125, "formal", "fixed")

    def default_fla_recheck(self, stable: str, culprit: str) -> dict[str, Any]:
        for arm in (stable, culprit):
            for seed in (123, 125):
                self.ensure_screen(arm, seed, "default")
        return analyze.aggregate_effect(stable, culprit, [123, 125], "default")

    def fastest_transfer(self, correction: str) -> dict[str, Any]:
        rows = []
        for seed in (123, 125):
            current = self.ensure_screen("fastest-current", seed)
            fixed = self.ensure_screen(correction, seed)
            rows.append(
                {
                    "seed": seed,
                    "current_retention": current["retention"],
                    "corrected_retention": fixed["retention"],
                    "improvement": fixed["retention"] - current["retention"],
                }
            )
        passed = all(row["improvement"] >= 0.05 for row in rows)
        payload = {"current": "fastest-current", "corrected": correction, "rows": rows, "passed": passed}
        atomic_write_json(self.root / "analysis" / "fastest-transfer.json", payload)
        if passed:
            self.formal_pair(["fastest-current", correction])
        return payload

    def run_evaluation(self, formal_arms: list[str]) -> None:
        selection = {
            "formal_arms": formal_arms,
            "seeds": [123, 125],
            "created_at_utc": utc_now(),
        }
        path = self.root / "formal-selection.json"
        atomic_write_json(path, selection)
        self.source_guard()
        self.process(
            "evaluate-formal",
            [str(PYTHON), str(EVALUATE_SCRIPT), "--selection", str(path)],
        )
        analyze.formal_summary(formal_arms, [123, 125])

    def run(self) -> int:
        self.phase = "preflight"
        self.process("preflight", [str(PYTHON), str(EXPERIMENT_SCRIPT), "preflight"])
        preflight = load_json(self.root / "preflight.json")
        if preflight.get("status") != "passed":
            raise RuntimeError("Preflight did not pass.")

        self.phase = "control-reproduction"
        controls = self.reproduce_controls()
        if controls["decision"] == "fixed_fla_did_not_reproduce":
            summary = self.fixed_fla_fallback()
            self.status("completed", summary=summary)
            return 0
        if controls["decision"] != "reproduced":
            raise RuntimeError("Current-source historical bridge failed; source bisection is required.")

        self.phase = "causal-screen"
        root = self.locate_root()
        if root["root"] == "unresolved":
            raise RuntimeError("No registered factor or interaction explained the degradation.")

        mechanism = None
        if root["root"] == "factor-block":
            self.phase = "block-mechanism"
            mechanism = self.mechanism_matrix()

        stable = "ctrl-bridge"
        culprit = "ctrl-current" if root["root"] == "three-way-interaction" else root["root"]
        self.phase = "canonical-confirm"
        self.formal_pair([stable, culprit])
        default_fla = self.default_fla_recheck(stable, culprit)

        self.phase = "fastest-transfer"
        correction = self.fastest_correction(root["root"], mechanism)
        transfer = self.fastest_transfer(correction)
        formal_arms = [stable, culprit]
        if transfer["passed"]:
            formal_arms.extend(["fastest-current", correction])

        self.phase = "formal-evaluation"
        self.run_evaluation(formal_arms)
        summaries = [
            analyze.screen_summary(arm, seed, gate)
            for gate in ("fixed", "default")
            for arm in (stable, culprit)
            for seed in (123, 125)
        ]
        analyze.write_curve_csv(summaries)
        summary = {
            "status": "completed",
            "root": root,
            "mechanism": mechanism,
            "default_fla": default_fla,
            "fastest_transfer": transfer,
            "formal_arms": formal_arms,
            "fresh_data_required": True,
        }
        atomic_write_json(self.root / "queue-summary.json", summary)
        atomic_write_json(
            self.root / "DONE.json",
            {"status": "completed", "completed_at_utc": utc_now(), "summary": summary},
        )
        self.phase = "complete"
        self.status("completed", summary=summary)
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
