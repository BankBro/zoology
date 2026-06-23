#!/usr/bin/env python3
"""Collect existing cb64-r16 write-control evidence into auditable tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path("/home/lyj/mnt/project/zoology")
ARTIFACT_DIR = ROOT / "docs/artifacts/20260624-01-flash-vqg-write-control-failure-audit"
ANALYSIS_ROOT = ROOT / "zoology/analysis/flash_vqg/results"
GENERATED_ROOT = ROOT / "zoology/experiments/flash_vqg/generated"

HARD = "valid/mqar_case/accuracy-1024x256"

METRICS = [
    "valid/loss",
    "valid/accuracy",
    HARD,
    "valid/attn/gd_residual_m_norm_mean",
    "valid/attn/gd_residual_m_norm_max",
    "valid/attn/gd_residual_mu_valid_ratio",
    "valid/attn/gd_residual_write_strength_mean",
    "valid/attn/gd_residual_write_strength_max",
    "valid/attn/gd_residual_write_strength_p95",
    "valid/attn/gd_residual_uncapped_write_strength_mean",
    "valid/attn/gd_residual_uncapped_write_strength_max",
    "valid/attn/gd_residual_uncapped_write_strength_p95",
    "valid/attn/gd_residual_sum_zeta_mean",
    "valid/attn/gd_residual_sum_zeta_max",
    "valid/attn/gd_residual_sum_zeta_p95",
    "valid/attn/gd_residual_uncapped_sum_zeta_mean",
    "valid/attn/gd_residual_uncapped_sum_zeta_max",
    "valid/attn/gd_residual_uncapped_sum_zeta_p95",
    "valid/attn/gd_residual_write_strength_cap_hit_ratio",
    "valid/attn/gd_residual_write_strength_cap_active",
    "valid/attn/gd_residual_write_strength_effective_cap",
    "valid/attn/gd_residual_m_norm_cap_hit_ratio",
    "valid/attn/gd_residual_m_norm_cap_active",
    "valid/attn/gd_residual_m_norm_effective_cap",
    "valid/attn/gd_residual_update_norm_cap_active",
    "valid/attn/gd_residual_update_norm_effective_cap",
    "valid/attn/gd_residual_lambda_mean",
    "valid/attn/gd_residual_inject_ratio",
    "valid/attn/gd_residual_beta_mean",
    "valid/attn/gd_residual_beta_max",
    "valid/attn/gd_residual_beta_cap_hit_ratio",
    "valid/attn/gd_residual_beta_cap_active",
    "valid/attn/gd_residual_beta_effective_cap",
    "valid/attn/gd_residual_read_margin_top1_top2_mean",
    "valid/attn/gd_residual_read_margin_top1_top2_p05",
    "valid/attn/gd_residual_read_entropy_mean",
    "valid/attn/gd_residual_read_selected_mass_mean",
    "valid/attn/gd_residual_read_selected_mass_p05",
]


@dataclass(frozen=True)
class RunSpec:
    setting: str
    seed: int
    role: str
    source_status: str
    analysis_dir: str
    expected_controls: dict[str, Any]
    notes: str = ""


RUNS: list[RunSpec] = [
    RunSpec(
        "default",
        123,
        "baseline_good",
        "official",
        "flash-vqg-20260520-flash-capacity-decomposition-gd-cb64-r16-s123-2026-05-20-18-12-06",
        {"write_strength_cap": None, "read_topk": 2, "mu_min_count": 0.1},
    ),
    RunSpec(
        "default",
        124,
        "baseline_bad",
        "official",
        "flash-vqg-20260528-seed-stability-wave2-corrected-tmux-20260528T021416Z-cb64-r16-s124-2026-05-28-02-16-23",
        {"write_strength_cap": None, "read_topk": 2, "mu_min_count": 0.1},
    ),
    RunSpec(
        "default",
        125,
        "baseline_good",
        "official",
        "flash-vqg-20260528-seed-stability-wave2-corrected-tmux-20260528T021416Z-cb64-r16-s125-2026-05-28-02-16-23",
        {"write_strength_cap": None, "read_topk": 2, "mu_min_count": 0.1},
    ),
    RunSpec(
        "hard04",
        123,
        "static_trust_region_good",
        "exploratory",
        "flash-vqg-20260529-seed-instability-cb64-r16-s123-d123-b64-ga4-fp32-noearly4ep-wcap0p04-2026-05-29-09-47-13",
        {"write_strength_cap": 0.04, "write_strength_cap_final": None, "read_topk": 2},
    ),
    RunSpec(
        "hard04",
        124,
        "static_trust_region_bad",
        "exploratory",
        "flash-vqg-20260529-seed-instability-cb64-r16-s124-d123-b64-ga4-fp32-noearly4ep-wcap0p04-2026-05-29-05-35-44",
        {"write_strength_cap": 0.04, "write_strength_cap_final": None, "read_topk": 2},
    ),
    RunSpec(
        "hard04",
        125,
        "static_trust_region_good",
        "exploratory",
        "flash-vqg-20260529-seed-instability-cb64-r16-s125-d123-b64-ga4-fp32-noearly4ep-wcap0p04-hard04-s125-minconfirm-2026-05-30-02-09-46",
        {"write_strength_cap": 0.04, "write_strength_cap_final": None, "read_topk": 2},
    ),
    RunSpec(
        "caprel0406late",
        123,
        "late_release_good",
        "exploratory",
        "flash-vqg-20260529-seed-instability-cb64-r16-s123-d123-b64-ga4-fp32-noearly4ep-wcap0p04-wcapfinal0p06-wcaprel2820to8468-caprel0406late-s123-2026-05-30-15-50-41",
        {"write_strength_cap": 0.04, "write_strength_cap_final": 0.06, "release": "2820->8468"},
    ),
    RunSpec(
        "caprel0406late",
        124,
        "late_release_bad",
        "exploratory",
        "flash-vqg-20260529-seed-instability-cb64-r16-s124-d123-b64-ga4-fp32-noearly4ep-wcap0p04-wcapfinal0p06-wcaprel2820to8468-caprel0406late-s124-2026-05-30-11-56-57",
        {"write_strength_cap": 0.04, "write_strength_cap_final": 0.06, "release": "2820->8468"},
    ),
    RunSpec(
        "caprel0406late",
        125,
        "late_release_good",
        "exploratory",
        "flash-vqg-20260529-seed-instability-cb64-r16-s125-d123-b64-ga4-fp32-noearly4ep-wcap0p04-wcapfinal0p06-wcaprel2820to8468-caprel0406late-s125-2026-05-30-10-52-31",
        {"write_strength_cap": 0.04, "write_strength_cap_final": 0.06, "release": "2820->8468"},
    ),
    RunSpec(
        "cap0405",
        123,
        "conservative_release_good",
        "exploratory",
        "flash-vqg-20260529-seed-instability-cb64-r16-s123-d123-b64-ga4-fp32-noearly4ep-wcap0p04-wcapfinal0p05-wcaprel2820to8468-wcapevalscheduled-caprel0405late-sched-s123-2026-05-31-07-20-33",
        {"write_strength_cap": 0.04, "write_strength_cap_final": 0.05, "release": "2820->8468"},
    ),
    RunSpec(
        "cap0405",
        124,
        "conservative_release_bad",
        "exploratory",
        "flash-vqg-20260529-seed-instability-cb64-r16-s124-d123-b64-ga4-fp32-noearly4ep-wcap0p04-wcapfinal0p05-wcaprel2820to8468-wcapevalscheduled-caprel0405late-sched-s124-2026-05-31-07-20-33",
        {"write_strength_cap": 0.04, "write_strength_cap_final": 0.05, "release": "2820->8468"},
    ),
    RunSpec(
        "cap0405_beta0p16",
        123,
        "conservative_release_low_beta_good",
        "exploratory",
        "flash-vqg-20260529-seed-instability-cb64-r16-s123-d123-b64-ga4-fp32-noearly4ep-wcap0p04-wcapfinal0p05-wcaprel2820to8468-wcapevalscheduled-beta0p16-caprel0405-beta0p16-s123-2026-05-31-11-18-01",
        {"write_strength_cap": 0.04, "write_strength_cap_final": 0.05, "beta_init": 0.16},
    ),
    RunSpec(
        "cap0405_beta0p16",
        124,
        "conservative_release_low_beta_bad",
        "exploratory",
        "flash-vqg-20260529-seed-instability-cb64-r16-s124-d123-b64-ga4-fp32-noearly4ep-wcap0p04-wcapfinal0p05-wcaprel2820to8468-wcapevalscheduled-beta0p16-caprel0405-beta0p16-s124-2026-05-31-11-18-01",
        {"write_strength_cap": 0.04, "write_strength_cap_final": 0.05, "beta_init": 0.16},
    ),
    RunSpec(
        "cap0406_mcap8",
        123,
        "release_with_static_mcap_good",
        "exploratory",
        "flash-vqg-20260529-seed-instability-cb64-r16-s123-d123-b64-ga4-fp32-noearly4ep-wcap0p04-wcapfinal0p06-wcaprel2820to8468-wcapevalscheduled-mcap8-caprel0406-mcap8-s123-2026-05-31-20-37-06",
        {"write_strength_cap": 0.04, "write_strength_cap_final": 0.06, "m_norm_cap": 8.0},
    ),
    RunSpec(
        "cap0406_mcap8",
        124,
        "release_with_static_mcap_bad",
        "exploratory",
        "flash-vqg-20260529-seed-instability-cb64-r16-s124-d123-b64-ga4-fp32-noearly4ep-wcap0p04-wcapfinal0p06-wcaprel2820to8468-wcapevalscheduled-mcap8-caprel0406-mcap8-s124-2026-05-31-20-37-06",
        {"write_strength_cap": 0.04, "write_strength_cap_final": 0.06, "m_norm_cap": 8.0},
    ),
]


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if math.isnan(parsed):
        return None
    return parsed


def metric_to_column(metric: str) -> str:
    return metric.replace("/", "__").replace("-", "_")


def analysis_dir(spec: RunSpec) -> Path:
    return ANALYSIS_ROOT / spec.analysis_dir


def manifest_path(spec: RunSpec) -> Path:
    return GENERATED_ROOT / spec.analysis_dir / "manifest.json"


def history_path(spec: RunSpec) -> Path:
    root = analysis_dir(spec)
    matches = sorted(root.glob("*/data/history.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one history.csv under {root}, found {len(matches)}")
    return matches[0]


def load_history(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            value = to_float(row.get("value"))
            if value is None:
                continue
            rows.append(
                {
                    "metric": row["metric"],
                    "step": int(float(row["step"])),
                    "epoch": float(row["epoch"]),
                    "timestamp": row.get("timestamp", ""),
                    "value": value,
                }
            )
    return rows


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def final_value(rows: list[dict[str, Any]], metric: str) -> float | None:
    values = [row for row in rows if row["metric"] == metric]
    if not values:
        return None
    values.sort(key=lambda row: (row["step"], row["epoch"]))
    return values[-1]["value"]


def max_value(rows: list[dict[str, Any]], metric: str) -> float | None:
    values = [row["value"] for row in rows if row["metric"] == metric]
    return max(values) if values else None


def best_metric_row(rows: list[dict[str, Any]], metric: str) -> dict[str, Any] | None:
    values = [row for row in rows if row["metric"] == metric]
    if not values:
        return None
    return max(values, key=lambda row: row["value"])


def pivot_validation_rows(rows: list[dict[str, Any]], spec: RunSpec) -> list[dict[str, Any]]:
    by_step: dict[int, dict[str, Any]] = {}
    for row in rows:
        metric = row["metric"]
        if metric not in METRICS:
            continue
        current = by_step.setdefault(
            row["step"],
            {
                "setting": spec.setting,
                "seed": spec.seed,
                "role": spec.role,
                "step": row["step"],
                "epoch": row["epoch"],
                "timestamp": row["timestamp"],
            },
        )
        current[metric_to_column(metric)] = row["value"]
    return [by_step[step] for step in sorted(by_step)]


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fields: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    fields.append(key)
                    seen.add(key)
        fieldnames = fields
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def setting_summary(final_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_setting: dict[str, list[dict[str, Any]]] = {}
    for row in final_rows:
        by_setting.setdefault(str(row["setting"]), []).append(row)

    summaries: list[dict[str, Any]] = []
    for setting, rows in sorted(by_setting.items()):
        hard_values = [float(row["final_hard"]) for row in rows if row.get("final_hard") not in (None, "")]
        best_values = [float(row["best_hard"]) for row in rows if row.get("best_hard") not in (None, "")]
        m_norm_values = [
            float(row["max_m_norm_max_over_valid"])
            for row in rows
            if row.get("max_m_norm_max_over_valid") not in (None, "")
        ]
        summaries.append(
            {
                "setting": setting,
                "num_runs": len(rows),
                "seeds": ";".join(str(row["seed"]) for row in sorted(rows, key=lambda item: int(item["seed"]))),
                "final_hard_values": ";".join(f"{value:.6f}" for value in hard_values),
                "final_hard_mean": mean(hard_values),
                "final_hard_min": min(hard_values) if hard_values else None,
                "final_hard_max": max(hard_values) if hard_values else None,
                "final_hard_spread": (max(hard_values) - min(hard_values)) if len(hard_values) >= 2 else None,
                "best_hard_max": max(best_values) if best_values else None,
                "max_m_norm_max_over_valid": max(m_norm_values) if m_norm_values else None,
                "m_norm_redline_gt8": any(value > 8.0 for value in m_norm_values),
                "m_norm_fail_gt12": any(value > 12.0 for value in m_norm_values),
            }
        )
    return summaries


def classify_setting(summary: dict[str, Any], final_rows: list[dict[str, Any]]) -> dict[str, Any]:
    setting = str(summary["setting"])
    spread = to_float(str(summary.get("final_hard_spread")))
    mmax = to_float(str(summary.get("max_m_norm_max_over_valid")))
    rows = [row for row in final_rows if row["setting"] == setting]
    seed123 = next((row for row in rows if int(row["seed"]) == 123), None)
    seed124 = next((row for row in rows if int(row["seed"]) == 124), None)
    best_final_gaps = [float(row["best_final_gap"]) for row in rows if row.get("best_final_gap") not in (None, "")]
    max_gap = max(best_final_gaps) if best_final_gaps else None

    label = "inconclusive"
    evidence = ""
    recommendation = ""
    if setting == "default":
        label = "unstable_baseline"
        evidence = "三 seed final hard spread 大, s124 为低盆地."
        recommendation = "只作为不稳定基线."
    elif setting == "hard04":
        label = "stable_ceiling_tax"
        evidence = "spread 小, 但 s123/s125 相比 default good seed 有上限损失."
        recommendation = "保留为 trust-region 稳定基准, 不作为最终性能方案."
    elif setting == "caprel0406late":
        label = "low_spread_state_overrun"
        evidence = "spread 小, 但 s123 m_norm_max 超过 12 红线."
        recommendation = "release 思路保留, 但无 guard 的 0.06 release 不进 official."
    elif setting == "cap0405":
        label = "late_drift_without_mnorm_overrun"
        evidence = "s123 final hard 低且 best-final gap 大, 但 m_norm 未过 8."
        recommendation = "不要重跑同一 cap0405; 需要定位 write/readout/lambda 或 read 轨迹."
    elif setting == "cap0405_beta0p16":
        label = "partial_rescue_low_ceiling"
        evidence = "两 seed 接近但整体 hard 低于 hard04, 仍有 best-final gap."
        recommendation = "beta init 可作诊断, 不作为主线."
    elif setting == "cap0406_mcap8":
        label = "ineffective_static_mnorm_cap"
        evidence = "m_norm 未触发红线但 spread 仍大, 静态 m_norm_cap 不是有效 guard."
        recommendation = "不能把 m_norm_cap=8 等同 guarded release."

    return {
        "setting": setting,
        "failure_type": label,
        "num_runs": summary.get("num_runs"),
        "seeds": summary.get("seeds"),
        "final_hard_spread": spread,
        "max_best_final_gap": max_gap,
        "max_m_norm_max_over_valid": mmax,
        "m_norm_redline_gt8": summary.get("m_norm_redline_gt8"),
        "m_norm_fail_gt12": summary.get("m_norm_fail_gt12"),
        "seed123_final_hard": seed123.get("final_hard") if seed123 else None,
        "seed124_final_hard": seed124.get("final_hard") if seed124 else None,
        "evidence": evidence,
        "recommendation": recommendation,
    }


def collect() -> dict[str, Any]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    final_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []

    for spec in RUNS:
        hpath = history_path(spec)
        mpath = manifest_path(spec)
        rows = load_history(hpath)
        manifest = load_manifest(mpath)
        best = best_metric_row(rows, HARD)
        final_hard = final_value(rows, HARD)
        best_hard = best["value"] if best is not None else None
        best_step = best["step"] if best is not None else None
        summary: dict[str, Any] = {
            "setting": spec.setting,
            "seed": spec.seed,
            "role": spec.role,
            "source_status": spec.source_status,
            "analysis_dir": str(analysis_dir(spec)),
            "history_csv": str(hpath),
            "manifest_path": str(mpath),
            "launch_id": spec.analysis_dir,
            "sweep_id": manifest.get("sweep_id"),
            "expected_controls": json.dumps(spec.expected_controls, ensure_ascii=False, sort_keys=True),
            "final_hard": final_hard,
            "best_hard": best_hard,
            "best_hard_step": best_step,
            "best_final_gap": (best_hard - final_hard) if best_hard is not None and final_hard is not None else None,
            "final_valid_accuracy": final_value(rows, "valid/accuracy"),
            "final_valid_loss": final_value(rows, "valid/loss"),
            "final_m_norm_max": final_value(rows, "valid/attn/gd_residual_m_norm_max"),
            "max_m_norm_max_over_valid": max_value(rows, "valid/attn/gd_residual_m_norm_max"),
            "final_write_strength_mean": final_value(rows, "valid/attn/gd_residual_write_strength_mean"),
            "final_write_strength_cap_hit_ratio": final_value(
                rows, "valid/attn/gd_residual_write_strength_cap_hit_ratio"
            ),
            "final_effective_cap": final_value(rows, "valid/attn/gd_residual_write_strength_effective_cap"),
            "final_lambda_mean": final_value(rows, "valid/attn/gd_residual_lambda_mean"),
            "final_inject_ratio": final_value(rows, "valid/attn/gd_residual_inject_ratio"),
            "final_beta_mean": final_value(rows, "valid/attn/gd_residual_beta_mean"),
            "notes": spec.notes,
        }
        for metric in METRICS:
            if final_value(rows, metric) is None:
                missing_rows.append(
                    {
                        "setting": spec.setting,
                        "seed": spec.seed,
                        "metric": metric,
                        "history_csv": str(hpath),
                    }
                )
        final_rows.append(summary)
        curve_rows.extend(pivot_validation_rows(rows, spec))
        for source_type, path in [
            ("history_csv", hpath),
            ("manifest", mpath),
        ]:
            source_rows.append(
                {
                    "setting": spec.setting,
                    "seed": spec.seed,
                    "source_type": source_type,
                    "path": str(path),
                    "exists": path.exists(),
                    "size_bytes": path.stat().st_size if path.exists() else None,
                    "sha256": sha256(path),
                }
            )

    default_by_seed = {
        int(row["seed"]): float(row["final_hard"])
        for row in final_rows
        if row["setting"] == "default" and row.get("final_hard") is not None
    }
    hard04_by_seed = {
        int(row["seed"]): float(row["final_hard"])
        for row in final_rows
        if row["setting"] == "hard04" and row.get("final_hard") is not None
    }
    for row in final_rows:
        seed = int(row["seed"])
        final_hard = row.get("final_hard")
        if final_hard is not None and seed in default_by_seed:
            row["delta_vs_default_same_seed"] = float(final_hard) - default_by_seed[seed]
            row["ceiling_tax_vs_default_same_seed"] = default_by_seed[seed] - float(final_hard)
        if final_hard is not None and seed in hard04_by_seed:
            row["delta_vs_hard04_same_seed"] = float(final_hard) - hard04_by_seed[seed]

    setting_rows = setting_summary(final_rows)
    taxonomy_rows = [classify_setting(row, final_rows) for row in setting_rows]

    source_contexts = [
        ROOT / "tmp/20260529-seed-instability-full-cap/research-handoff.md",
        ROOT / "tmp/20260529-seed-instability-full-cap/research-handoff-full-log.md",
        ROOT / "tmp/20260529-seed-instability-full-cap/seed-instability-final-summary.csv",
        ROOT / "tmp/20260529-seed-instability-full-cap/seed-instability-step-curve-summary.csv",
        ROOT / "docs/plans/20260622-flash-vqg-seed-stability-roadmap.md",
    ]
    for path in source_contexts:
        source_rows.append(
            {
                "setting": "context",
                "seed": "",
                "source_type": "context",
                "path": str(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else None,
                "sha256": sha256(path),
            }
        )

    write_csv(ARTIFACT_DIR / "write_control_final_summary.csv", final_rows)
    write_csv(ARTIFACT_DIR / "write_control_setting_summary.csv", setting_rows)
    write_csv(ARTIFACT_DIR / "write_control_step_curves.csv", curve_rows)
    write_csv(ARTIFACT_DIR / "failure_taxonomy.csv", taxonomy_rows)
    write_csv(ARTIFACT_DIR / "missing_metrics.csv", missing_rows)
    write_csv(ARTIFACT_DIR / "source_manifest.csv", source_rows)

    metadata = {
        "experiment_id": "20260624-01-flash-vqg-write-control-failure-audit",
        "created_from": str(Path(__file__).relative_to(ROOT)),
        "artifact_dir": str(ARTIFACT_DIR),
        "num_runs": len(RUNS),
        "settings": sorted({spec.setting for spec in RUNS}),
        "primary_metric": HARD,
        "notes": [
            "This audit reads existing history/manifest files only.",
            "No new training was launched.",
            "m_norm_max > 8 is treated as warning; > 12 is treated as redline.",
        ],
    }
    (ARTIFACT_DIR / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (ARTIFACT_DIR / "README.md").write_text(
        "# Flash-VQG write-control failure audit artifact\n\n"
        "This directory contains a read-only audit of existing cb64-r16 write-control runs.\n\n"
        "- `write_control_final_summary.csv`: one row per run.\n"
        "- `write_control_setting_summary.csv`: grouped setting-level spread and state summary.\n"
        "- `write_control_step_curves.csv`: validation-step metric curves.\n"
        "- `failure_taxonomy.csv`: setting-level failure labels.\n"
        "- `missing_metrics.csv`: metrics absent from historical histories.\n"
        "- `source_manifest.csv`: source paths and sha256 hashes.\n",
        encoding="utf-8",
    )
    return {
        "final_rows": final_rows,
        "setting_rows": setting_rows,
        "taxonomy_rows": taxonomy_rows,
        "missing_rows": missing_rows,
        "source_rows": source_rows,
    }


def assert_close(actual: float | None, expected: float, name: str, tol: float = 5e-4) -> None:
    if actual is None or abs(actual - expected) > tol:
        raise AssertionError(f"{name}: expected {expected}, got {actual}")


def run_checks(payload: dict[str, Any]) -> None:
    rows = payload["final_rows"]
    by_key = {(row["setting"], int(row["seed"])): row for row in rows}
    expected = {
        ("default", 123): 0.968711,
        ("default", 124): 0.819797,
        ("default", 125): 0.987285,
        ("hard04", 123): 0.945039,
        ("hard04", 124): 0.963055,
        ("hard04", 125): 0.952605,
        ("caprel0406late", 123): 0.949371,
        ("caprel0406late", 124): 0.963004,
        ("caprel0406late", 125): 0.960484,
        ("cap0405", 123): 0.811,
        ("cap0405", 124): 0.960,
    }
    for key, value in expected.items():
        assert_close(to_float(str(by_key[key].get("final_hard"))), value, f"{key} final_hard", tol=2e-3)
    assert_close(
        to_float(str(by_key[("caprel0406late", 123)].get("final_m_norm_max"))),
        14.487579,
        "caprel0406late s123 final_m_norm",
        tol=2e-3,
    )
    assert_close(
        to_float(str(by_key[("caprel0406late", 123)].get("max_m_norm_max_over_valid"))),
        15.735760,
        "caprel0406late s123 max_m_norm_over_valid",
        tol=2e-3,
    )
    cap0405_mnorm = to_float(str(by_key[("cap0405", 123)].get("max_m_norm_max_over_valid")))
    if cap0405_mnorm is None or cap0405_mnorm >= 8.0:
        raise AssertionError(f"cap0405 s123 m_norm expected < 8, got {cap0405_mnorm}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="run anchor checks after collection")
    args = parser.parse_args()
    payload = collect()
    if args.check:
        run_checks(payload)
    print(f"wrote {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
