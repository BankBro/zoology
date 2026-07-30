#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

LOCAL_DIR = Path(__file__).resolve().parent
if str(LOCAL_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_DIR))

from common import (
    EXTRAPOLATION_DELTA_MIN,
    EXTRAPOLATION_SHAPES,
    EXPERIMENT_ID,
    SEEDS,
    STANDARD_DELTA_MIN,
    VARIANTS,
    atomic_write_json,
    load_json,
    run_root,
    run_tag,
    utc_now,
)
from experiment import result_path


P0 = "p0-a1-block64"
K2 = "k2-persistent-p8"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _training(phase: str, seed: int) -> dict[str, dict[str, Any]]:
    results = {}
    for variant in VARIANTS:
        path = result_path(variant, seed, phase)
        result = load_json(path)
        if result.get("status") != "completed":
            raise RuntimeError(f"Training result is incomplete: {path}")
        results[variant] = result
    return results


def _evaluation_index(phase: str) -> dict[tuple[str, int, str, str], dict[str, Any]]:
    summary = load_json(run_root() / "evaluation" / f"{phase}-summary.json")
    return {
        (row["variant"], int(row["seed"]), row["checkpoint_role"], row["shape"]): row
        for row in summary["records"]
    }


def _standard(result: dict[str, Any], role: str) -> float:
    metrics = result[f"{role}_checkpoint"]["metrics"]
    return float(metrics["valid/mqar_case/accuracy-1024x256"])


def _config_signature(result: dict[str, Any]) -> str:
    config = result.get("gate_autotune", {}).get("layer_norm_gated_bwd_kernel", {}).get("best_config")
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def quality_rows(phase: str, role: str) -> list[dict[str, Any]]:
    index = _evaluation_index(phase)
    seeds = SEEDS if phase == "formal" else (123,)
    rows = []
    for seed in seeds:
        training = _training(phase, seed)
        standard_p0 = _standard(training[P0], role)
        standard_k2 = _standard(training[K2], role)
        deltas = {
            shape: float(index[(K2, seed, role, shape)]["accuracy"])
            - float(index[(P0, seed, role, shape)]["accuracy"])
            for shape in EXTRAPOLATION_SHAPES
        }
        macro = sum(deltas.values()) / len(deltas)
        p0_config, k2_config = _config_signature(training[P0]), _config_signature(training[K2])
        rows.append(
            {
                "seed": seed,
                "checkpoint_role": role,
                "p0_standard": standard_p0,
                "k2_standard": standard_k2,
                "standard_delta": standard_k2 - standard_p0,
                **{f"delta_{shape}": value for shape, value in deltas.items()},
                "extrapolation_macro_delta": macro,
                "standard_pass": standard_k2 - standard_p0 >= STANDARD_DELTA_MIN,
                "extrapolation_pass": macro >= EXTRAPOLATION_DELTA_MIN,
                "fla_config_match": p0_config == k2_config and p0_config != "null",
                "p0_model_hash": training[P0]["resume_audit"]["model_state_sha256"],
                "k2_model_hash": training[K2]["resume_audit"]["model_state_sha256"],
                "p0_optimizer_hash": training[P0]["resume_audit"]["optimizer_state_sha256"],
                "k2_optimizer_hash": training[K2]["resume_audit"]["optimizer_state_sha256"],
            }
        )
    return rows


def quality_decision(rows: list[dict[str, Any]], runtime_passed: bool) -> dict[str, Any]:
    standard = [float(row["standard_delta"]) for row in rows]
    extrapolation = [float(row["extrapolation_macro_delta"]) for row in rows]
    checks = {
        "runtime": runtime_passed,
        "per_seed_standard": all(value >= STANDARD_DELTA_MIN for value in standard),
        "per_seed_extrapolation": all(value >= EXTRAPOLATION_DELTA_MIN for value in extrapolation),
        "mean_standard": statistics.fmean(standard) >= STANDARD_DELTA_MIN,
        "mean_extrapolation": statistics.fmean(extrapolation) >= EXTRAPOLATION_DELTA_MIN,
        "fla_configs_match": all(bool(row["fla_config_match"]) for row in rows),
    }
    if not checks["runtime"]:
        status = "correctness_failed"
    elif not checks["fla_configs_match"]:
        status = "requires_fla_replay"
    elif not all(value for key, value in checks.items() if key != "fla_configs_match"):
        status = "quality_rejected"
    else:
        status = "passed"
    return {
        "status": status,
        "checks": checks,
        "standard_mean_delta": statistics.fmean(standard),
        "standard_population_sd": statistics.pstdev(standard),
        "extrapolation_mean_delta": statistics.fmean(extrapolation),
        "extrapolation_population_sd": statistics.pstdev(extrapolation),
    }


def system_rows(phase: str) -> list[dict[str, Any]]:
    seeds = SEEDS if phase == "formal" else (123,)
    rows = []
    for seed in seeds:
        for variant in VARIANTS:
            result = _training(phase, seed)[variant]
            telemetry = result["telemetry"]
            rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "wall_clock_sec": result["wall_clock_sec"],
                    "optimizer_step_wall_sec_p50": telemetry["optimizer_step_wall_sec_p50"],
                    "peak_allocated_mib": telemetry["peak_allocated_mib"],
                    "peak_reserved_mib": telemetry["peak_reserved_mib"],
                    "persistent_calls": result["runtime_audit"]["persistent_calls"],
                    "fallbacks": result["runtime_audit"]["fallbacks"],
                }
            )
    return rows


def analyze(phase: str) -> dict[str, Any]:
    last_rows = quality_rows(phase, "last")
    best_rows = quality_rows(phase, "best") if phase == "formal" else []
    systems = system_rows(phase)
    runtime_passed = all(row["fallbacks"] == 0 for row in systems)
    decision = quality_decision(last_rows, runtime_passed)
    output = run_root() / "analysis"
    _write_csv(output / f"{phase}-quality-last.csv", last_rows)
    if best_rows:
        _write_csv(output / f"{phase}-quality-best.csv", best_rows)
    _write_csv(output / f"{phase}-system.csv", systems)
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "run_tag": run_tag(),
        "phase": phase,
        **decision,
        "thresholds": {
            "standard_delta_min": STANDARD_DELTA_MIN,
            "extrapolation_delta_min": EXTRAPOLATION_DELTA_MIN,
        },
        "last_rows": last_rows,
        "best_rows": best_rows,
        "system_rows": systems,
        "completed_at_utc": utc_now(),
    }
    atomic_write_json(output / f"{phase}-summary.json", payload)
    print(json.dumps({"status": payload["status"], "phase": phase}, sort_keys=True))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=("screen", "formal", "diagnostic_fp32"), required=True
    )
    args = parser.parse_args()
    result = analyze(args.phase)
    return {"passed": 0, "quality_rejected": 2, "requires_fla_replay": 3, "correctness_failed": 4}[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
