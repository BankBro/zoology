#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from common import EXPERIMENT_ID, REPO_ROOT, sha256_file, write_json


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs"
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "docs/artifacts" / EXPERIMENT_ID


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def machine_from_path(path: Path) -> str:
    for part in path.parts:
        if part in {"2080ti", "3090"}:
            return part
    return "unknown"


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summary_metric(payload: dict[str, Any], metric: str, statistic: str = "p50"):
    for row in payload.get("summaries", []):
        if row.get("metric") == metric:
            return row.get(statistic)
    return None


def collect_benchmarks(output_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(output_root.rglob("summary.json")):
        if "paired-benchmark" not in path.parts:
            continue
        payload = load_json(path)
        if payload.get("run_kind") not in {"timing", "memory"}:
            continue
        env = payload.get("environment") or {}
        memory = payload.get("memory") or {}
        rows.append(
            {
                "machine": machine_from_path(path),
                "fla_variant": env.get("fla_variant"),
                "fla_version": env.get("fla_version"),
                "torch": env.get("torch"),
                "triton": env.get("triton"),
                "model": (payload.get("model") or {}).get("name"),
                "phase": payload.get("phase"),
                "run_kind": payload.get("run_kind"),
                "repeat_id": payload.get("repeat_id"),
                "warmup": payload.get("warmup"),
                "active": payload.get("active"),
                "wall_ms_p50": summary_metric(payload, "wall_ms"),
                "wall_ms_p90": summary_metric(payload, "wall_ms", "p90"),
                "cuda_total_ms_p50": summary_metric(payload, "cuda_total_ms"),
                "backbone_ms_p50": summary_metric(payload, "backbone"),
                "backward_ms_p50": summary_metric(payload, "backward"),
                "peak_allocated_bytes": memory.get("peak_allocated_bytes"),
                "peak_reserved_bytes": memory.get("peak_reserved_bytes"),
                "source": relative(path),
            }
        )
    return rows


def collect_compatibility(output_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(output_root.rglob("gdn-*-compatibility*.json")):
        payload = load_json(path)
        env = payload.get("environment") or {}
        rows.append(
            {
                "machine": machine_from_path(path),
                "fla_variant": env.get("fla_variant"),
                "fla_version": env.get("fla_version"),
                "phase": payload.get("phase"),
                "success": payload.get("success"),
                "error": payload.get("error"),
                "elapsed_seconds": payload.get("elapsed_seconds"),
                "peak_allocated_bytes": payload.get("peak_allocated_bytes"),
                "peak_reserved_bytes": payload.get("peak_reserved_bytes"),
                "canonical_paired_matrix": "paired-benchmark" in path.parts,
                "scope": (
                    "paired-production"
                    if "paired-benchmark" in path.parts
                    else "cold-empty-cache"
                    if any(part.startswith("cold-compile-") for part in path.parts)
                    else "standalone"
                ),
                "source": relative(path),
            }
        )
    return rows


def collect_equivalence(output_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(output_root.rglob("*.json")):
        if "equivalence" not in path.parts:
            continue
        payload = load_json(path)
        if "comparisons" not in payload or "reference_environment" not in payload:
            continue
        for comparison in payload["comparisons"]:
            rows.append(
                {
                    "comparison": path.stem,
                    "kind": payload.get("kind"),
                    "model": payload.get("model"),
                    "reference_fla": (payload.get("reference_environment") or {}).get(
                        "fla_version"
                    ),
                    "candidate_fla": (payload.get("candidate_environment") or {}).get(
                        "fla_version"
                    ),
                    "loss_abs": payload.get("loss_abs"),
                    "tensor": comparison.get("name"),
                    "exact": comparison.get("exact"),
                    "max_abs": comparison.get("max_abs"),
                    "relative_l2": comparison.get("relative_l2"),
                    "tensor_passed": comparison.get("passed"),
                    "comparison_passed": payload.get("passed"),
                    "source": relative(path),
                }
            )
    return rows


def collect_quality(output_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(output_root.rglob("result.json")):
        payload = load_json(path)
        if payload.get("experiment_id") != EXPERIMENT_ID:
            continue
        metrics = payload.get("final_metrics") or {}
        rows.append(
            {
                "machine": payload.get("machine"),
                "fla_variant": payload.get("fla_variant"),
                "model": payload.get("model"),
                "run_type": payload.get("run_type"),
                "status": payload.get("status"),
                "configured_max_epochs": payload.get("configured_max_epochs"),
                "final_epoch": payload.get("final_epoch"),
                "expected_optimizer_steps": (payload.get("preflight") or {}).get(
                    "expected_optimizer_steps"
                ),
                "wall_clock_sec": payload.get("wall_clock_sec"),
                "valid_loss": metrics.get("valid/loss"),
                "valid_accuracy": metrics.get("valid/accuracy"),
                "valid_1024x256": metrics.get("valid/mqar_case/accuracy-1024x256"),
                "valid_512x128": metrics.get("valid/mqar_case/accuracy-512x128"),
                "started_at_utc": payload.get("started_at_utc"),
                "ended_at_utc": payload.get("ended_at_utc"),
                "run_id": payload.get("run_id"),
                "launch_id": payload.get("launch_id"),
                "last_checkpoint_sha256": payload.get("last_checkpoint_sha256"),
                "model_state_sha256": payload.get("model_state_sha256"),
                "fla_version": (payload.get("environment") or {}).get("fla_version"),
                "torch": (payload.get("environment") or {}).get("torch"),
                "triton": (payload.get("environment") or {}).get("triton"),
                "source": relative(path),
            }
        )
    return rows


def bootstrap_median_ci(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    rng = np.random.default_rng(2026072402)
    samples = np.asarray(values, dtype=np.float64)
    indexes = rng.integers(0, len(samples), size=(20000, len(samples)))
    medians = np.median(samples[indexes], axis=1)
    low, high = np.quantile(medians, [0.025, 0.975])
    return float(low), float(high)


def compare_versions(benchmark_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    timing = defaultdict(dict)
    memory = defaultdict(dict)
    for row in benchmark_rows:
        key = (row["machine"], row["model"], row["phase"])
        if row["run_kind"] == "timing":
            timing[(key, row["repeat_id"])][row["fla_variant"]] = row
        elif row["run_kind"] == "memory":
            memory[key][row["fla_variant"]] = row
    ratios = defaultdict(list)
    for (key, repeat_id), variants in timing.items():
        if {"v042", "v050"}.issubset(variants):
            ratios[key].append(
                (
                    int(repeat_id),
                    float(variants["v050"]["wall_ms_p50"])
                    / float(variants["v042"]["wall_ms_p50"]),
                )
            )
    rows = []
    for key in sorted(ratios):
        paired = sorted(ratios[key])
        values = [value for _, value in paired]
        low, high = bootstrap_median_ci(values)
        mem = memory.get(key, {})
        memory_ratio = None
        if {"v042", "v050"}.issubset(mem):
            memory_ratio = float(mem["v050"]["peak_allocated_bytes"]) / float(
                mem["v042"]["peak_allocated_bytes"]
            )
        rows.append(
            {
                "machine": key[0],
                "model": key[1],
                "phase": key[2],
                "paired_repeats": len(values),
                "repeat_ratios_v050_over_v042": ";".join(
                    f"{repeat}:{value:.9f}" for repeat, value in paired
                ),
                "median_time_ratio_v050_over_v042": statistics.median(values),
                "median_time_change_pct": (statistics.median(values) - 1.0) * 100.0,
                "bootstrap_ratio_ci95_low": low,
                "bootstrap_ratio_ci95_high": high,
                "v050_pair_wins": sum(value < 1.0 for value in values),
                "no_time_regression_over_2pct": statistics.median(values) <= 1.02,
                "stable_positive_gain": high is not None and high < 1.0,
                "peak_allocated_ratio_v050_over_v042": memory_ratio,
                "memory_within_5pct": memory_ratio is None or memory_ratio <= 1.05,
            }
        )
    return rows


def compare_models(benchmark_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    timing: dict[tuple[str, str, str, str], dict[int, float]] = defaultdict(dict)
    memory: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in benchmark_rows:
        key = (row["machine"], row["fla_variant"], row["phase"], row["model"])
        if row["run_kind"] == "timing" and row["wall_ms_p50"] is not None:
            timing[key][int(row["repeat_id"])] = float(row["wall_ms_p50"])
        elif row["run_kind"] == "memory":
            memory[key] = row
    rows = []
    scopes = sorted({key[:3] for key in timing})
    for machine, variant, phase in scopes:
        flash_key = (machine, variant, phase, "flash")
        gdn_key = (machine, variant, phase, "gdn")
        if flash_key not in timing or gdn_key not in timing:
            continue
        flash = timing[flash_key]
        gdn = timing[gdn_key]
        paired_ids = sorted(set(flash).intersection(gdn))
        paired_ratios = [flash[index] / gdn[index] for index in paired_ids]
        ratio_low, ratio_high = bootstrap_median_ci(paired_ratios)
        flash_median = statistics.median(flash.values())
        gdn_median = statistics.median(gdn.values())
        flash_memory = memory.get(flash_key)
        gdn_memory = memory.get(gdn_key)
        memory_ratio = None
        if flash_memory is not None and gdn_memory is not None:
            memory_ratio = float(flash_memory["peak_allocated_bytes"]) / float(
                gdn_memory["peak_allocated_bytes"]
            )
        rows.append(
            {
                "machine": machine,
                "fla_variant": variant,
                "phase": phase,
                "flash_repeats": len(flash),
                "gdn_repeats": len(gdn),
                "flash_wall_ms_p50_median": flash_median,
                "gdn_wall_ms_p50_median": gdn_median,
                "flash_over_gdn_time_ratio": flash_median / gdn_median,
                "paired_repeat_ratios": ";".join(
                    f"{index}:{flash[index] / gdn[index]:.9f}" for index in paired_ids
                ),
                "paired_ratio_bootstrap_ci95_low": ratio_low,
                "paired_ratio_bootstrap_ci95_high": ratio_high,
                "time_within_2x": flash_median / gdn_median <= 2.0,
                "flash_peak_allocated_bytes": (
                    flash_memory["peak_allocated_bytes"] if flash_memory else None
                ),
                "gdn_peak_allocated_bytes": (
                    gdn_memory["peak_allocated_bytes"] if gdn_memory else None
                ),
                "flash_over_gdn_allocated_ratio": memory_ratio,
                "memory_within_2x": memory_ratio is not None and memory_ratio <= 2.0,
            }
        )
    return rows


def collect_warmed_epochs(output_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(output_root.rglob("result.json")):
        if "warmed-epoch-v042" not in path.parts:
            continue
        payload = load_json(path)
        if "total_wall_seconds" not in payload or "optimizer_steps" not in payload:
            continue
        model = payload.get("model") or {}
        environment = payload.get("environment") or {}
        validations = payload.get("validations") or []
        final_validation = validations[-1] if validations else {}
        rows.append(
            {
                "machine": machine_from_path(path),
                "fla_variant": "v042",
                "fla_version": "0.4.2",
                "model": model.get("name"),
                "repeat_id": payload.get("repeat_id"),
                "seed": payload.get("seed"),
                "optimizer_steps": payload.get("optimizer_steps"),
                "precompiled": payload.get("precompiled"),
                "batch_order_match": (payload.get("batch_order") or {}).get("match"),
                "total_wall_seconds": payload.get("total_wall_seconds"),
                "train_wall_seconds_excluding_validation": payload.get(
                    "train_wall_seconds_excluding_validation"
                ),
                "validation_wall_seconds": payload.get("validation_wall_seconds"),
                "final_validation_accuracy": final_validation.get("accuracy"),
                "peak_allocated_bytes": payload.get("peak_allocated_bytes"),
                "peak_reserved_bytes": payload.get("peak_reserved_bytes"),
                "torch": environment.get("torch"),
                "zoology_commit": environment.get("zoology_commit"),
                "source": relative(path),
            }
        )
    return rows


def compare_warmed_epochs(epoch_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in epoch_rows:
        grouped[(row["machine"], row["model"])].append(row)
    rows = []
    for machine in sorted({key[0] for key in grouped}):
        flash = grouped.get((machine, "flash"), [])
        gdn = grouped.get((machine, "gdn"), [])
        if not flash or not gdn:
            continue

        def median(field: str, source: list[dict[str, Any]]) -> float:
            return statistics.median(float(row[field]) for row in source)

        flash_total = median("total_wall_seconds", flash)
        gdn_total = median("total_wall_seconds", gdn)
        flash_train = median("train_wall_seconds_excluding_validation", flash)
        gdn_train = median("train_wall_seconds_excluding_validation", gdn)
        flash_valid = median("validation_wall_seconds", flash)
        gdn_valid = median("validation_wall_seconds", gdn)
        flash_memory = median("peak_allocated_bytes", flash)
        gdn_memory = median("peak_allocated_bytes", gdn)
        rows.append(
            {
                "machine": machine,
                "fla_variant": "v042",
                "flash_repeats": len(flash),
                "gdn_repeats": len(gdn),
                "flash_total_wall_seconds_median": flash_total,
                "gdn_total_wall_seconds_median": gdn_total,
                "flash_over_gdn_total_ratio": flash_total / gdn_total,
                "total_within_2x": flash_total / gdn_total <= 2.0,
                "flash_train_wall_seconds_median": flash_train,
                "gdn_train_wall_seconds_median": gdn_train,
                "flash_over_gdn_train_ratio": flash_train / gdn_train,
                "flash_validation_wall_seconds_median": flash_valid,
                "gdn_validation_wall_seconds_median": gdn_valid,
                "flash_over_gdn_validation_ratio": flash_valid / gdn_valid,
                "flash_peak_allocated_bytes_median": flash_memory,
                "gdn_peak_allocated_bytes_median": gdn_memory,
                "flash_over_gdn_allocated_ratio": flash_memory / gdn_memory,
                "memory_within_2x": flash_memory / gdn_memory <= 2.0,
                "all_runs_valid": all(
                    row["optimizer_steps"] == 704
                    and bool(row["precompiled"])
                    and bool(row["batch_order_match"])
                    for row in flash + gdn
                ),
            }
        )
    return rows


def _flash_reference() -> dict[str, dict[str, float]]:
    path = (
        REPO_ROOT
        / "docs/artifacts/20260724-01-flash-vqg-gd-residual-efficiency/formal-quality.csv"
    )
    references = {}
    if not path.exists():
        return references
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["training_seed"]) != 124:
                continue
            references[row["machine"]] = {
                "valid_accuracy": float(row["valid_accuracy"]),
                "valid_1024x256": float(row["accuracy_1024x256"]),
            }
    return references


def assess_quality(
    quality: list[dict[str, Any]], selected_variant: str
) -> list[dict[str, Any]]:
    index = {
        (row["model"], row["machine"], row["fla_variant"]): row
        for row in quality
        if row["run_type"] == "formal"
    }
    requirements = [
        ("gdn", "2080ti", "current040", None),
        ("gdn", "2080ti", "v042", ("gdn", "2080ti", "current040")),
        ("gdn", "3090", "v042", ("gdn", "2080ti", "v042")),
        ("gdn", "2080ti", "v050", ("gdn", "2080ti", "current040")),
        ("gdn", "3090", "v050", ("gdn", "2080ti", "v050")),
        ("flash", "2080ti", selected_variant, "historical_flash"),
        ("flash", "3090", selected_variant, "historical_flash"),
    ]
    flash_reference = _flash_reference()
    rows = []
    for model, machine, variant, reference_key in requirements:
        current = index.get((model, machine, variant))
        if reference_key == "historical_flash":
            reference = flash_reference.get(machine)
            reference_label = "20260724-01 seed124 same-machine Flash"
        elif reference_key is None:
            reference = None
            reference_label = None
        else:
            reference_row = index.get(reference_key)
            reference = (
                {
                    "valid_accuracy": reference_row["valid_accuracy"],
                    "valid_1024x256": reference_row["valid_1024x256"],
                }
                if reference_row is not None
                else None
            )
            reference_label = "/".join(reference_key)
        completed = current is not None and current["status"] == "completed"
        overall_delta = None
        hard_delta = None
        overall_pass = False
        hard_pass = False
        if completed and current["valid_accuracy"] is not None:
            if reference is None and reference_key is None:
                overall_pass = True
            elif reference is not None:
                overall_delta = float(current["valid_accuracy"]) - float(
                    reference["valid_accuracy"]
                )
                overall_pass = overall_delta >= -0.01
        if completed and current["valid_1024x256"] is not None:
            hard_pass = float(current["valid_1024x256"]) >= 0.85
            if reference is not None:
                hard_delta = float(current["valid_1024x256"]) - float(
                    reference["valid_1024x256"]
                )
                hard_pass = hard_pass and hard_delta >= -0.04
        rows.append(
            {
                "model": model,
                "machine": machine,
                "fla_variant": variant,
                "required_for_selection": True,
                "result_present": current is not None,
                "status": current.get("status") if current else "missing",
                "valid_accuracy": current.get("valid_accuracy") if current else None,
                "reference_valid_accuracy": (
                    reference.get("valid_accuracy") if reference else None
                ),
                "valid_accuracy_delta": overall_delta,
                "overall_non_regression_pass": overall_pass,
                "valid_1024x256": current.get("valid_1024x256") if current else None,
                "reference_valid_1024x256": (
                    reference.get("valid_1024x256") if reference else None
                ),
                "valid_1024x256_delta": hard_delta,
                "hard_accuracy_pass": hard_pass,
                "reference": reference_label,
                "quality_pass": bool(completed and overall_pass and hard_pass),
            }
        )
    return rows


def selection_summary(
    compatibility: list[dict[str, Any]],
    equivalence: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    quality_gates: list[dict[str, Any]],
    model_comparisons: list[dict[str, Any]],
    epoch_comparisons: list[dict[str, Any]],
) -> dict[str, Any]:
    canonical_compat = [row for row in compatibility if row["canonical_paired_matrix"]]
    compatibility_complete = len(canonical_compat) >= 8
    compatibility_passed = compatibility_complete and all(
        bool(row["success"]) for row in canonical_compat
    )
    equivalence_groups = {
        (row["comparison"], row["comparison_passed"]) for row in equivalence
    }
    equivalence_complete = len({name for name, _ in equivalence_groups}) >= 5
    equivalence_passed = equivalence_complete and all(passed for _, passed in equivalence_groups)
    timing_complete = len(comparisons) == 8 and all(row["paired_repeats"] == 5 for row in comparisons)
    no_time_regression = timing_complete and all(
        row["no_time_regression_over_2pct"] for row in comparisons
    )
    no_memory_regression = timing_complete and all(row["memory_within_5pct"] for row in comparisons)
    stable_gain = any(row["stable_positive_gain"] for row in comparisons)
    v050_performance_gate = no_time_regression and no_memory_regression and stable_gain
    provisional = "v050" if v050_performance_gate else "v042"
    quality_complete = len(quality_gates) == 7 and all(
        row["result_present"] for row in quality_gates
    )
    quality_finished = quality_complete and all(
        row["status"] == "completed" for row in quality_gates
    )
    quality_passed = quality_finished and all(row["quality_pass"] for row in quality_gates)
    selected_core = [row for row in model_comparisons if row["fla_variant"] == "v042"]
    selected_core_complete = len(selected_core) == 4
    selected_core_passed = selected_core_complete and all(
        row["time_within_2x"] and row["memory_within_2x"] for row in selected_core
    )
    warmed_epoch_complete = len(epoch_comparisons) == 2 and all(
        row["flash_repeats"] >= 3 and row["gdn_repeats"] >= 3
        for row in epoch_comparisons
    )
    warmed_epoch_passed = warmed_epoch_complete and all(
        row["total_within_2x"] and row["memory_within_2x"] and row["all_runs_valid"]
        for row in epoch_comparisons
    )
    return {
        "compatibility_complete": compatibility_complete,
        "compatibility_passed": compatibility_passed,
        "equivalence_complete": equivalence_complete,
        "equivalence_passed": equivalence_passed,
        "timing_complete": timing_complete,
        "no_time_regression_over_2pct": no_time_regression,
        "no_memory_regression_over_5pct": no_memory_regression,
        "stable_positive_gain_present": stable_gain,
        "v050_performance_gate": v050_performance_gate,
        "provisional_selected_variant": provisional,
        "formal_quality_complete": quality_complete,
        "formal_quality_finished": quality_finished,
        "formal_quality_passed": quality_passed,
        "selected_v042_core_efficiency_complete": selected_core_complete,
        "selected_v042_core_efficiency_passed": selected_core_passed,
        "warmed_epoch_efficiency_complete": warmed_epoch_complete,
        "warmed_epoch_efficiency_passed": warmed_epoch_passed,
        "original_efficiency_blocker_closed": bool(
            selected_core_passed and warmed_epoch_passed
        ),
        "final_selection_ready": bool(
            compatibility_passed
            and equivalence_passed
            and timing_complete
            and quality_passed
        ),
        "final_selected_variant": provisional if quality_passed else None,
        "selection_policy": (
            "选择 v050 仅当全部兼容/等价/质量门槛通过, 八个共同 benchmark 单元均不回退超过 2%, "
            "peak allocated 均不恶化超过 5%, 且至少一个单元存在 95% bootstrap 稳定收益; 否则选择 v042."
        ),
    }


def source_manifest(output_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(output_root.rglob("*")):
        if not path.is_file() or path.suffix not in {".json", ".csv"}:
            continue
        rows.append(
            {
                "source_machine": machine_from_path(path),
                "source_path": str(path.resolve()),
                "mirror_path": relative(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "mirrored": True,
                "git_tracked": False,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="汇总 GDN FLA 兼容性实验 artifact.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    args = parser.parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    benchmarks = collect_benchmarks(args.output_root)
    compatibility = collect_compatibility(args.output_root)
    equivalence = collect_equivalence(args.output_root)
    quality = collect_quality(args.output_root)
    comparisons = compare_versions(benchmarks)
    model_comparisons = compare_models(benchmarks)
    warmed_epochs = collect_warmed_epochs(args.output_root)
    epoch_comparisons = compare_warmed_epochs(warmed_epochs)
    performance_gate = bool(
        len(comparisons) == 8
        and all(row["no_time_regression_over_2pct"] for row in comparisons)
        and all(row["memory_within_5pct"] for row in comparisons)
        and any(row["stable_positive_gain"] for row in comparisons)
    )
    provisional_variant = "v050" if performance_gate else "v042"
    quality_gates = assess_quality(quality, provisional_variant)
    selection = selection_summary(
        compatibility,
        equivalence,
        comparisons,
        quality_gates,
        model_comparisons,
        epoch_comparisons,
    )
    write_csv(args.artifact_dir / "benchmark-runs.csv", benchmarks)
    write_csv(args.artifact_dir / "compatibility.csv", compatibility)
    write_csv(args.artifact_dir / "equivalence.csv", equivalence)
    write_csv(args.artifact_dir / "version-comparison.csv", comparisons)
    write_csv(args.artifact_dir / "model-comparison.csv", model_comparisons)
    write_csv(args.artifact_dir / "warmed-epoch.csv", warmed_epochs)
    write_csv(args.artifact_dir / "warmed-epoch-ratios.csv", epoch_comparisons)
    write_csv(args.artifact_dir / "quality-1ep.csv", quality)
    write_csv(args.artifact_dir / "quality-gates.csv", quality_gates)
    write_csv(args.artifact_dir / "source-manifest.csv", source_manifest(args.output_root))
    write_json(
        args.artifact_dir / "metadata.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "output_root": relative(args.output_root),
            "counts": {
                "benchmark_rows": len(benchmarks),
                "compatibility_rows": len(compatibility),
                "equivalence_rows": len(equivalence),
                "version_comparison_rows": len(comparisons),
                "model_comparison_rows": len(model_comparisons),
                "warmed_epoch_rows": len(warmed_epochs),
                "warmed_epoch_ratio_rows": len(epoch_comparisons),
                "quality_rows": len(quality),
            },
            "selection": selection,
        },
    )
    print(json.dumps(selection, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
