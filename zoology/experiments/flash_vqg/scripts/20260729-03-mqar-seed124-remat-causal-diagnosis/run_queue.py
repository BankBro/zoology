#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from common import (
    PYTHON,
    REPO_ROOT,
    SCRIPT_DIR,
    VARIANTS,
    atomic_write_json,
    load_json,
    run_root,
    run_tag,
    utc_now,
)


STEP_LIMITS = (16, 32, 128, 704, 2816)


def trace_path(label: str, variant: str) -> Path:
    return run_root() / "probes" / label / variant / "trace.jsonl"


class Queue:
    def __init__(self) -> None:
        self.root = run_root()
        self.logs = self.root / "logs"
        self.started_at = utc_now()
        self.commands: list[dict[str, Any]] = []

    def run_process(self, name: str, command: list[str]) -> None:
        stdout_path = self.logs / f"{name}.stdout.log"
        stderr_path = self.logs / f"{name}.stderr.log"
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env.setdefault("TRITON_F32_DEFAULT", "ieee")
        env.setdefault("NVIDIA_TF32_OVERRIDE", "0")
        env.setdefault("TORCH_DETERMINISTIC", "0")
        started = time.perf_counter()
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr:
            result = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=env,
                text=True,
                stdout=stdout,
                stderr=stderr,
            )
        record = {
            "name": name,
            "command": command,
            "return_code": int(result.returncode),
            "wall_clock_sec": time.perf_counter() - started,
            "stdout": str(stdout_path.resolve()),
            "stderr": str(stderr_path.resolve()),
        }
        self.commands.append(record)
        atomic_write_json(self.root / "queue-progress.json", self.payload("running"))
        if result.returncode != 0:
            raise RuntimeError(f"Queue command failed: {name}, see {stderr_path}")

    def probe(
        self,
        *,
        label: str,
        variant: str,
        steps: int,
        detail_window: int | None = None,
    ) -> None:
        command = [
            str(PYTHON),
            str(SCRIPT_DIR / "probe.py"),
            "--variant",
            variant,
            "--label",
            label,
            "--max-train-steps",
            str(steps),
        ]
        if detail_window is not None:
            command.extend(["--detail-window", str(detail_window)])
        self.run_process(f"probe-{label}-{variant}", command)

    def compare(
        self,
        *,
        name: str,
        left: Path,
        right: Path,
    ) -> dict[str, Any]:
        output = self.root / "comparisons" / f"{name}.json"
        self.run_process(
            f"compare-{name}",
            [
                str(PYTHON),
                str(SCRIPT_DIR / "compare_traces.py"),
                "--left",
                str(left),
                "--right",
                str(right),
                "--output",
                str(output),
            ],
        )
        return load_json(output)

    def locate_initial_difference(self) -> tuple[str, int, dict[str, Any]]:
        for steps in STEP_LIMITS:
            label = f"initial-{steps}"
            for variant in VARIANTS:
                self.probe(label=label, variant=variant, steps=steps)
            comparison = self.compare(
                name=f"a0-vs-a1-{steps}",
                left=trace_path(label, "a0-fixed-off"),
                right=trace_path(label, "a1-fixed-post-phase1"),
            )
            window = comparison["first_mismatch_window"]
            if window is not None:
                return label, int(window), comparison
        raise RuntimeError("A0/A1 difference did not reproduce through 2816 steps.")

    def repeatability(
        self,
        *,
        initial_label: str,
        window: int,
    ) -> dict[str, Any]:
        label = f"repeat-through-{window + 1}"
        result = {}
        for variant in VARIANTS:
            self.probe(label=label, variant=variant, steps=window + 1)
            result[variant] = self.compare(
                name=f"repeat-{variant}",
                left=trace_path(initial_label, variant),
                right=trace_path(label, variant),
            )
        return result

    @staticmethod
    def classify_repeatability(result: dict[str, Any]) -> str:
        a0_exact = result["a0-fixed-off"]["exact_on_common_events"]
        a1_exact = result["a1-fixed-post-phase1"]["exact_on_common_events"]
        if a0_exact and a1_exact:
            return "cross_variant_remat_difference"
        if not a0_exact and a1_exact:
            return "a0_intrinsic_nondeterminism"
        if a0_exact and not a1_exact:
            return "a1_intrinsic_nondeterminism"
        return "both_variants_intrinsically_nondeterministic"

    def detail(self, window: int) -> dict[str, Any]:
        label = f"detail-window-{window}"
        for variant in VARIANTS:
            self.probe(
                label=label,
                variant=variant,
                steps=window + 1,
                detail_window=window,
            )
        return self.compare(
            name=f"detail-a0-vs-a1-window-{window}",
            left=trace_path(label, "a0-fixed-off"),
            right=trace_path(label, "a1-fixed-post-phase1"),
        )

    def payload(self, status: str, **extra: Any) -> dict[str, Any]:
        return {
            "experiment_id": "20260729-03-mqar-seed124-remat-causal-diagnosis",
            "run_tag": run_tag(),
            "status": status,
            "started_at_utc": self.started_at,
            "updated_at_utc": utc_now(),
            "commands": self.commands,
            **extra,
        }

    def run(self) -> dict[str, Any]:
        self.run_process("preflight", [str(PYTHON), str(SCRIPT_DIR / "preflight.py")])
        initial_label, window, initial = self.locate_initial_difference()
        repeats = self.repeatability(initial_label=initial_label, window=window)
        classification = self.classify_repeatability(repeats)
        detail = self.detail(window)
        payload = self.payload(
            "needs_causal_intervention",
            first_mismatch_window=window,
            initial_label=initial_label,
            initial_comparison=initial,
            repeatability=repeats,
            repeatability_classification=classification,
            detail_comparison=detail,
        )
        atomic_write_json(self.root / "queue-summary.json", payload)
        return payload


def main() -> int:
    queue = Queue()
    try:
        payload = queue.run()
    except BaseException as exc:
        traceback.print_exc()
        payload = queue.payload(
            "failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        atomic_write_json(queue.root / "queue-summary.json", payload)
        return 1
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
