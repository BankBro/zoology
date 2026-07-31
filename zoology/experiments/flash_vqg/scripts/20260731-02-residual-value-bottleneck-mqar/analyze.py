#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

LOCAL_DIR = Path(__file__).resolve().parent
if str(LOCAL_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_DIR))

from common import (
    BASELINE,
    EXTRAPOLATION_SHAPES,
    EXPERIMENT_ID,
    Q0_RELATIVE_DELTA_MIN,
    VARIANTS,
    atomic_write_json,
    load_json,
    run_root,
    run_tag,
    utc_now,
)
from experiment import result_path


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def _training(variant: str) -> dict[str, Any]:
    path = result_path(variant, 123, "screen")
    result = load_json(path)
    if result.get("status") != "completed":
        raise RuntimeError(f"Training result is incomplete: {path}")
    return result


def _evaluation_index() -> dict[tuple[str, str], dict[str, Any]]:
    summary = load_json(run_root() / "evaluation/screen-summary.json")
    return {(row["variant"], row["shape"]): row for row in summary["records"]}


def _standard(result: dict[str, Any]) -> float:
    metrics = result["last_checkpoint"]["metrics"]
    return float(metrics["valid/mqar_case/accuracy-1024x256"])


def _relative_delta(candidate: float, baseline: float) -> float:
    return (candidate - baseline) / max(abs(baseline), 1e-12)


def _fla_config(result: dict[str, Any]) -> str:
    config = result.get("gate_autotune", {}).get("layer_norm_gated_bwd_kernel", {}).get(
        "best_config"
    )
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def quality_rows() -> list[dict[str, Any]]:
    index = _evaluation_index()
    baseline = _training(BASELINE)
    baseline_standard = _standard(baseline)
    baseline_extrap = {
        shape: float(index[(BASELINE, shape)]["accuracy"])
        for shape in EXTRAPOLATION_SHAPES
    }
    baseline_macro = sum(baseline_extrap.values()) / len(baseline_extrap)
    rows = []
    for variant in VARIANTS[1:]:
        candidate = _training(variant)
        candidate_extrap = {
            shape: float(index[(variant, shape)]["accuracy"])
            for shape in EXTRAPOLATION_SHAPES
        }
        candidate_macro = sum(candidate_extrap.values()) / len(candidate_extrap)
        standard_relative = _relative_delta(_standard(candidate), baseline_standard)
        macro_relative = _relative_delta(candidate_macro, baseline_macro)
        rows.append(
            {
                "candidate": variant,
                "seed": 123,
                "baseline_standard": baseline_standard,
                "candidate_standard": _standard(candidate),
                "standard_relative_delta": standard_relative,
                "baseline_extrapolation_macro": baseline_macro,
                "candidate_extrapolation_macro": candidate_macro,
                "extrapolation_macro_relative_delta": macro_relative,
                **{
                    f"relative_delta_{shape}": _relative_delta(
                        candidate_extrap[shape], baseline_extrap[shape]
                    )
                    for shape in EXTRAPOLATION_SHAPES
                },
                "standard_pass": standard_relative >= Q0_RELATIVE_DELTA_MIN,
                "extrapolation_pass": macro_relative >= Q0_RELATIVE_DELTA_MIN,
                "fla_config_match": _fla_config(baseline) == _fla_config(candidate),
            }
        )
    return rows


def system_rows() -> list[dict[str, Any]]:
    rows = []
    for variant in VARIANTS:
        result = _training(variant)
        telemetry = result["telemetry"]
        rows.append(
            {
                "variant": variant,
                "residual_value_dim": result["residual_value_dim"],
                "wall_clock_sec": result["wall_clock_sec"],
                "optimizer_step_wall_sec_p50": telemetry["optimizer_step_wall_sec_p50"],
                "peak_allocated_mib": telemetry["peak_allocated_mib"],
                "peak_reserved_mib": telemetry["peak_reserved_mib"],
                "fallbacks": result["runtime_audit"]["fallbacks"],
            }
        )
    return rows


def analyze() -> dict[str, Any]:
    rows = quality_rows()
    systems = system_rows()
    runtime_passed = all(row["fallbacks"] == 0 for row in systems)
    candidates = {
        row["candidate"]: {
            "standard": bool(row["standard_pass"]),
            "extrapolation": bool(row["extrapolation_pass"]),
            "fla_config": bool(row["fla_config_match"]),
            "passed": bool(
                runtime_passed
                and row["standard_pass"]
                and row["extrapolation_pass"]
                and row["fla_config_match"]
            ),
        }
        for row in rows
    }
    passed = sum(int(row["passed"]) for row in candidates.values())
    status = "passed" if passed == len(candidates) else "quality_mixed" if passed else "quality_rejected"
    output = run_root() / "analysis"
    _write_csv(output / "screen-quality.csv", rows)
    _write_csv(output / "screen-system.csv", systems)
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "run_tag": run_tag(),
        "phase": "screen",
        "status": status,
        "candidate_status": candidates,
        "relative_delta_min": Q0_RELATIVE_DELTA_MIN,
        "quality_rows": rows,
        "system_rows": systems,
        "completed_at_utc": utc_now(),
    }
    atomic_write_json(output / "screen-summary.json", payload)
    print(json.dumps({"status": status}, sort_keys=True))
    return payload


if __name__ == "__main__":
    result = analyze()
    raise SystemExit(0 if result["status"] == "passed" else 2)
