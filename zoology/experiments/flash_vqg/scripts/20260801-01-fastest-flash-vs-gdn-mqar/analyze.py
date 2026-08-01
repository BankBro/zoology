#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

LOCAL_DIR = Path(__file__).resolve().parent
if str(LOCAL_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_DIR))

from common import (
    ARMS,
    CANONICAL,
    EVAL_CASES,
    EXPERIMENT_ID,
    EXTRAPOLATION_SHAPES,
    FASTEST,
    GDN,
    SEEDS,
    STANDARD_CASES,
    atomic_write_json,
    case_id,
    load_json,
    run_root,
    run_tag,
    utc_now,
)
import experiment


PAIRS = (
    (FASTEST, GDN),
    (FASTEST, CANONICAL),
    (CANONICAL, GDN),
)
STANDARD_IDS = tuple(case_id(case) for case in STANDARD_CASES)
ENDPOINT_ID = case_id(STANDARD_CASES[-1])
LONGER_ENDPOINT_ID = "1024x256-n500"
EXTRAPOLATION_IDS = tuple(f"{shape}-n500" for shape in EXTRAPOLATION_SHAPES)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)
    temporary.replace(path)


def evaluation_index(phase: str) -> dict[tuple[str, int, str, str], dict[str, Any]]:
    summary = load_json(run_root() / "evaluation" / f"{phase}-summary.json")
    if summary.get("status") != "passed":
        raise RuntimeError(f"Evaluation did not pass: {phase}.")
    index = {}
    for row in summary["records"]:
        key = (row["arm"], int(row["seed"]), row["checkpoint_role"], row["case"])
        if key in index:
            raise RuntimeError(f"Duplicate evaluation row: {key}.")
        index[key] = row
    return index


def _mean(values: list[float]) -> float:
    return statistics.fmean(values)


def metric_row(
    index: dict[tuple[str, int, str, str], dict[str, Any]],
    arm: str,
    seed: int,
    role: str,
    phase: str,
) -> dict[str, Any]:
    endpoint = float(index[(arm, seed, role, ENDPOINT_ID)]["accuracy"])
    extrapolation = [
        float(index[(arm, seed, role, case)]["accuracy"])
        for case in EXTRAPOLATION_IDS
    ]
    row = {
        "arm": arm,
        "seed": seed,
        "checkpoint_role": role,
        "endpoint_1024x256": endpoint,
        "extrapolation_macro": _mean(extrapolation),
    }
    if phase == "formal":
        standard = [float(index[(arm, seed, role, case)]["accuracy"]) for case in STANDARD_IDS]
        longer_endpoint = float(index[(arm, seed, role, LONGER_ENDPOINT_ID)]["accuracy"])
        row["standard_macro"] = _mean(standard)
        row["longer_endpoint_1024x256"] = longer_endpoint
        row["extrapolation_retention"] = (
            row["extrapolation_macro"] / longer_endpoint if longer_endpoint else 0.0
        )
    for case in EXTRAPOLATION_IDS:
        row[f"accuracy_{case}"] = float(index[(arm, seed, role, case)]["accuracy"])
    return row


def metric_rows(phase: str) -> list[dict[str, Any]]:
    index = evaluation_index(phase)
    seeds = SEEDS if phase == "formal" else (123,)
    roles = ("last", "best") if phase == "formal" else ("last",)
    return [
        metric_row(index, arm, seed, role, phase)
        for role in roles
        for seed in seeds
        for arm in ARMS
    ]


def _numeric_metrics(row: dict[str, Any]) -> tuple[str, ...]:
    excluded = {"arm", "seed", "checkpoint_role"}
    return tuple(key for key, value in row.items() if key not in excluded and isinstance(value, float))


def paired_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index = {(row["arm"], row["seed"], row["checkpoint_role"]): row for row in rows}
    output = []
    roles = sorted({row["checkpoint_role"] for row in rows})
    seeds = sorted({int(row["seed"]) for row in rows})
    for left, right in PAIRS:
        for role in roles:
            for seed in seeds:
                left_row, right_row = index[(left, seed, role)], index[(right, seed, role)]
                output.append(
                    {
                        "left": left,
                        "right": right,
                        "seed": seed,
                        "checkpoint_role": role,
                        **{
                            f"delta_{metric}": float(left_row[metric]) - float(right_row[metric])
                            for metric in _numeric_metrics(left_row)
                        },
                    }
                )
    return output


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(row["arm"], row["checkpoint_role"])].append(row)
    output = []
    for (arm, role), values in sorted(buckets.items()):
        record: dict[str, Any] = {"arm": arm, "checkpoint_role": role, "n_seeds": len(values)}
        for metric in _numeric_metrics(values[0]):
            samples = [float(row[metric]) for row in values]
            record[f"{metric}_mean"] = _mean(samples)
            record[f"{metric}_population_sd"] = statistics.pstdev(samples)
            record[f"{metric}_min"] = min(samples)
            record[f"{metric}_max"] = max(samples)
        output.append(record)
    return output


def paired_aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(row["left"], row["right"], row["checkpoint_role"])].append(row)
    output = []
    for key, values in sorted(buckets.items()):
        left, right, role = key
        record: dict[str, Any] = {"left": left, "right": right, "checkpoint_role": role, "n_seeds": len(values)}
        metrics = tuple(name for name in values[0] if name.startswith("delta_"))
        for metric in metrics:
            samples = [float(row[metric]) for row in values]
            record[f"{metric}_mean"] = _mean(samples)
            record[f"{metric}_population_sd"] = statistics.pstdev(samples)
            record[f"{metric}_positive_seeds"] = sum(value > 0 for value in samples)
            record[f"{metric}_worst"] = min(samples)
        output.append(record)
    return output


def system_rows(phase: str) -> list[dict[str, Any]]:
    seeds = SEEDS if phase == "formal" else (123,)
    rows = []
    for seed in seeds:
        for arm in ARMS:
            result = load_json(experiment.result_path(arm, seed, phase))
            if result.get("status") != "completed":
                raise RuntimeError(f"Training result is incomplete: {arm}, {seed}, {phase}.")
            telemetry = result["telemetry"]
            bwd = result.get("gate_autotune", {}).get("layer_norm_gated_bwd_kernel", {})
            rows.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "phase": phase,
                    "wall_clock_sec": result["wall_clock_sec"],
                    "optimizer_step_wall_sec_p50": telemetry.get("optimizer_step_wall_sec_p50"),
                    "peak_allocated_mib": telemetry.get("peak_allocated_mib"),
                    "peak_reserved_mib": telemetry.get("peak_reserved_mib"),
                    "optimizer_step": result["resume_audit"]["optimizer_step"],
                    "fallbacks": result["resume_audit"]["runtime_audit"]["fallbacks"],
                    "fla_bwd_config": json.dumps(bwd.get("best_config"), sort_keys=True),
                    "model_state_sha256": result["resume_audit"]["model_state_sha256"],
                    "optimizer_state_sha256": result["resume_audit"]["optimizer_state_sha256"],
                }
            )
    return rows


def dataset_audit(phase: str) -> dict[str, Any]:
    summary = load_json(run_root() / "evaluation" / f"{phase}-summary.json")
    hashes: dict[str, set[str]] = defaultdict(set)
    for row in summary["records"]:
        hashes[row["case"]].add(row["dataset_hash"])
    mismatches = {case: sorted(values) for case, values in hashes.items() if len(values) != 1}
    return {"passed": not mismatches, "case_count": len(hashes), "mismatches": mismatches}


def quality_decision(paired: list[dict[str, Any]], phase: str) -> dict[str, Any]:
    target = [
        row
        for row in paired
        if row["left"] == FASTEST
        and row["right"] == CANONICAL
        and row["checkpoint_role"] == "last"
    ]
    endpoint = [float(row["delta_endpoint_1024x256"]) for row in target]
    extrapolation = [float(row["delta_extrapolation_macro"]) for row in target]
    checks = {
        "endpoint_mean_ge_minus_0_05": _mean(endpoint) >= -0.05,
        "extrapolation_mean_ge_minus_0_05": _mean(extrapolation) >= -0.05,
        "no_seed_below_minus_0_10": min(endpoint + extrapolation) >= -0.10,
    }
    if phase == "formal":
        standard = [float(row["delta_standard_macro"]) for row in target]
        checks["standard_mean_ge_minus_0_05"] = _mean(standard) >= -0.05
    return {
        "quality_retained": all(checks.values()),
        "checks": checks,
        "endpoint_mean_delta": _mean(endpoint),
        "extrapolation_mean_delta": _mean(extrapolation),
    }


def analyze(phase: str) -> dict[str, Any]:
    rows = metric_rows(phase)
    paired = paired_rows(rows)
    aggregates = aggregate_rows(rows)
    paired_summary = paired_aggregate(paired)
    systems = system_rows(phase)
    datasets = dataset_audit(phase)
    if not datasets["passed"]:
        raise RuntimeError(f"Dataset identity failed: {datasets}")
    decision = quality_decision(paired, phase)
    output = run_root() / "analysis"
    write_csv(output / f"{phase}-metrics.csv", rows)
    write_csv(output / f"{phase}-paired-deltas.csv", paired)
    write_csv(output / f"{phase}-aggregate.csv", aggregates)
    write_csv(output / f"{phase}-paired-summary.csv", paired_summary)
    write_csv(output / f"{phase}-system.csv", systems)
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "run_tag": run_tag(),
        "phase": phase,
        "status": "completed",
        "quality_decision": decision,
        "dataset_audit": datasets,
        "metrics": rows,
        "paired_deltas": paired,
        "aggregates": aggregates,
        "paired_summary": paired_summary,
        "systems": systems,
        "completed_at_utc": utc_now(),
    }
    atomic_write_json(output / f"{phase}-summary.json", payload)
    print(json.dumps({"status": payload["status"], "quality": decision}, ensure_ascii=False))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("screen", "formal"), required=True)
    args = parser.parse_args()
    analyze(args.phase)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
