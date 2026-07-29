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
    EXTRAPOLATION_SHAPES,
    EXPERIMENT_ID,
    REPO_ROOT,
    SEEDS,
    atomic_write_json,
    load_json,
    run_root,
    run_tag,
    training_descriptors,
    utc_now,
)
from experiment import result_path


STANDARD_MARGIN = -0.01
EXTRAPOLATION_MACRO_MARGIN = -0.02
HISTORICAL_SUMMARY = (
    REPO_ROOT
    / "docs/artifacts/20260729-01-mqar-gd-remat-regression/summary.json"
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_training() -> list[dict[str, Any]]:
    rows = []
    for row in training_descriptors():
        result = load_json(result_path(row["variant"], row["seed"], "formal"))
        if result.get("status") != "completed":
            raise RuntimeError(f"Formal training is incomplete: {row['descriptor_id']}")
        rows.append(result)
    return rows


def training_rows(training: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in training:
        checkpoint = result["last_checkpoint"]
        metrics = checkpoint["metrics"]
        rows.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "descriptor_id": result["descriptor_id"],
                "variant": result["variant"],
                "remat_mode": result["remat_mode"],
                "seed": result["seed"],
                "data_seed": result["data_seed"],
                "precision": result["train_precision"],
                "final_epoch": checkpoint["epoch"],
                "valid_loss": metrics["valid/loss"],
                "valid_accuracy": metrics["valid/accuracy"],
                "mqar_1024x256_accuracy": metrics[
                    "valid/mqar_case/accuracy-1024x256"
                ],
                "wall_clock_sec": result["wall_clock_sec"],
                "optimizer_step_wall_sec_p50": result["telemetry"].get(
                    "optimizer_step_wall_sec_p50"
                ),
                "peak_allocated_mib": result["telemetry"].get(
                    "peak_allocated_mib"
                ),
                "peak_reserved_mib": result["telemetry"].get(
                    "peak_reserved_mib"
                ),
                "last_checkpoint_path": checkpoint["path"],
                "last_checkpoint_sha256": checkpoint["file_sha256"],
                "last_model_state_sha256": checkpoint["model_state_sha256"],
                "best_checkpoint_path": result["best_checkpoint"]["path"],
                "best_checkpoint_sha256": result["best_checkpoint"][
                    "file_sha256"
                ],
                "best_model_state_sha256": result["best_checkpoint"][
                    "model_state_sha256"
                ],
                "resolved_config_path": result["resolved_config_path"],
                "resolved_config_sha256": result["resolved_config_sha256"],
            }
        )
    return rows


def quality_summary(
    training: list[dict[str, Any]],
    evaluation: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    train_rows = training_rows(training)
    standard = {
        (row["variant"], int(row["seed"])): float(row["mqar_1024x256_accuracy"])
        for row in train_rows
    }
    last_eval = {
        (row["variant"], int(row["seed"]), row["shape"]): float(row["accuracy"])
        for row in evaluation
        if row["checkpoint_role"] == "last"
    }
    paired = []
    standard_deltas = []
    extrapolation_deltas = []
    for seed in SEEDS:
        delta = standard[("a1-fixed-post-phase1", seed)] - standard[("a0-fixed-off", seed)]
        standard_deltas.append(delta)
        paired.append(
            {
                "checkpoint_role": "last",
                "seed": seed,
                "metric": "standard_mqar_1024x256",
                "a0": standard[("a0-fixed-off", seed)],
                "a1": standard[("a1-fixed-post-phase1", seed)],
                "a1_minus_a0": delta,
            }
        )
        for shape in EXTRAPOLATION_SHAPES:
            a0 = last_eval[("a0-fixed-off", seed, shape)]
            a1 = last_eval[("a1-fixed-post-phase1", seed, shape)]
            delta = a1 - a0
            extrapolation_deltas.append(delta)
            paired.append(
                {
                    "checkpoint_role": "last",
                    "seed": seed,
                    "metric": shape,
                    "a0": a0,
                    "a1": a1,
                    "a1_minus_a0": delta,
                }
            )
    standard_mean = statistics.mean(standard_deltas)
    extrapolation_macro = statistics.mean(extrapolation_deltas)
    hash_rows = final_hash_rows(training)
    checks = {
        "six_training_runs": len(training) == 6,
        "paired_final_hashes_equal": all(row["hashes_equal"] for row in hash_rows),
        "standard_noninferiority": standard_mean >= STANDARD_MARGIN,
        "extrapolation_macro_noninferiority": (
            extrapolation_macro >= EXTRAPOLATION_MACRO_MARGIN
        ),
    }
    summary = {
        "passed": all(checks.values()),
        "checks": checks,
        "standard_margin": STANDARD_MARGIN,
        "standard_paired_deltas": standard_deltas,
        "standard_mean_delta": standard_mean,
        "standard_population_sd": statistics.pstdev(standard_deltas),
        "extrapolation_macro_margin": EXTRAPOLATION_MACRO_MARGIN,
        "extrapolation_paired_deltas": extrapolation_deltas,
        "extrapolation_macro_mean_delta": extrapolation_macro,
        "extrapolation_population_sd": statistics.pstdev(extrapolation_deltas),
    }
    return summary, paired


def final_hash_rows(training: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
        (result["variant"], int(result["seed"])): result for result in training
    }
    rows = []
    for seed in SEEDS:
        a0 = indexed[("a0-fixed-off", seed)]
        a1 = indexed[("a1-fixed-post-phase1", seed)]
        a0_hash = a0["last_checkpoint"]["model_state_sha256"]
        a1_hash = a1["last_checkpoint"]["model_state_sha256"]
        rows.append(
            {
                "seed": seed,
                "a0_model_state_sha256": a0_hash,
                "a1_model_state_sha256": a1_hash,
                "hashes_equal": a0_hash == a1_hash,
            }
        )
    return rows


def historical_rows(quality: dict[str, Any]) -> list[dict[str, Any]]:
    historical = load_json(HISTORICAL_SUMMARY)["quality"]
    return [
        {
            "metric": "standard_mqar_1024x256",
            "historical_mean_delta": historical["standard_mean_delta"],
            "fixed_mean_delta": quality["standard_mean_delta"],
            "improvement": (
                quality["standard_mean_delta"]
                - historical["standard_mean_delta"]
            ),
        },
        {
            "metric": "four_slice_extrapolation_macro",
            "historical_mean_delta": historical["extrapolation_macro_mean_delta"],
            "fixed_mean_delta": quality["extrapolation_macro_mean_delta"],
            "improvement": (
                quality["extrapolation_macro_mean_delta"]
                - historical["extrapolation_macro_mean_delta"]
            ),
        },
    ]


def result_status(quality: dict[str, Any]) -> str:
    checks = quality["checks"]
    quality_passed = (
        checks["standard_noninferiority"]
        and checks["extrapolation_macro_noninferiority"]
    )
    if quality["passed"]:
        return "passed"
    if quality_passed and not checks["paired_final_hashes_equal"]:
        return "quality_recovered_but_not_deterministic"
    return "not_alleviated"


def system_rows(training: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = training_rows(training)
    grouped = {}
    for variant in ("a0-fixed-off", "a1-fixed-post-phase1"):
        selected = [row for row in rows if row["variant"] == variant]
        grouped[variant] = {
            "variant": variant,
            "runs": len(selected),
            "wall_clock_sec_mean": statistics.mean(
                float(row["wall_clock_sec"]) for row in selected
            ),
            "optimizer_step_wall_sec_p50_mean": statistics.mean(
                float(row["optimizer_step_wall_sec_p50"]) for row in selected
            ),
            "peak_allocated_mib_max": max(
                float(row["peak_allocated_mib"]) for row in selected
            ),
            "peak_reserved_mib_max": max(
                float(row["peak_reserved_mib"]) for row in selected
            ),
            "step_time_ratio_vs_a0": None,
            "peak_allocated_ratio_vs_a0": None,
        }
    a0, a1 = grouped["a0-fixed-off"], grouped["a1-fixed-post-phase1"]
    a1["step_time_ratio_vs_a0"] = (
        a1["optimizer_step_wall_sec_p50_mean"]
        / a0["optimizer_step_wall_sec_p50_mean"]
    )
    a1["peak_allocated_ratio_vs_a0"] = (
        a1["peak_allocated_mib_max"] / a0["peak_allocated_mib_max"]
    )
    return [a0, a1]


def collect(output_dir: Path) -> dict[str, Any]:
    training = load_training()
    evaluation_payload = load_json(
        run_root() / "evaluation" / "formal-summary.json"
    )
    determinism = load_json(run_root() / "determinism" / "summary.json")
    evaluation = evaluation_payload["records"]
    quality, paired = quality_summary(training, evaluation)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "training-final.csv", training_rows(training))
    _write_csv(output_dir / "longer-mqar-detail.csv", evaluation)
    _write_csv(output_dir / "paired-quality.csv", paired)
    _write_csv(output_dir / "final-hash-pairs.csv", final_hash_rows(training))
    _write_csv(output_dir / "historical-comparison.csv", historical_rows(quality))
    _write_csv(output_dir / "system-summary.csv", system_rows(training))
    atomic_write_json(output_dir / "determinism-summary.json", determinism)
    status = result_status(quality)
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "run_tag": run_tag(),
        "status": status,
        "quality": quality,
        "training_runs": len(training),
        "evaluation_logical_events": evaluation_payload["logical_events"],
        "evaluation_physical_events": evaluation_payload["physical_events"],
        "collected_at_utc": utc_now(),
        "raw_run_root": str(run_root().resolve()),
    }
    atomic_write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or (run_root() / "final-summary")
    summary = collect(output)
    return 0 if summary["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
