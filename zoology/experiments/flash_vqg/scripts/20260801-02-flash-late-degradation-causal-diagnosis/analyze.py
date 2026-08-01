#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

LOCAL_DIR = Path(__file__).resolve().parent
if str(LOCAL_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_DIR))

from causal_common import (
    ARMS,
    GATE_MODES,
    SEEDS,
    atomic_write_json,
    load_json,
    run_root,
    utc_now,
)
import experiment


ENDPOINT = "valid/mqar_case/accuracy-1024x256"
TRACE_KEYS = (
    "train/loss",
    "attn/gd_residual_write_q_entropy",
    "attn/gd_residual_read_entropy",
    "attn/gd_residual_m_norm_mean",
    "attn/gd_residual_update_norm_mean",
    "attn/gd_residual_injection_warmup_factor",
)


def _finite_float(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def telemetry_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def validation_curve(path: Path) -> list[dict[str, Any]]:
    latest_train: dict[str, Any] = {}
    curve = []
    for row in telemetry_rows(path):
        if "train/loss" in row:
            latest_train = row
        if ENDPOINT not in row:
            continue
        record = {
            "validation_index": len(curve) + 1,
            "log_step": int(row["log_step"]),
            "epoch": int(row.get("epoch", -1)),
            "endpoint_1024x256": float(row[ENDPOINT]),
            "valid_accuracy": _finite_float(row.get("valid/accuracy")),
            "valid_loss": _finite_float(row.get("valid/loss")),
        }
        for key in TRACE_KEYS:
            record[key] = _finite_float(latest_train.get(key))
        curve.append(record)
    if not curve:
        raise RuntimeError(f"No validation rows found in {path}.")
    return curve


def screen_summary(
    arm: str,
    seed: int,
    gate_mode: str = "fixed",
) -> dict[str, Any]:
    result_file = experiment.result_path(arm, seed, "screen", gate_mode)
    result = load_json(result_file)
    if result.get("status") != "completed":
        raise RuntimeError(f"Screen result is incomplete: {result_file}")
    curve = validation_curve(Path(result["telemetry"]["path"]))
    epoch1 = [row for row in curve if row["epoch"] == 0]
    if not epoch1:
        raise RuntimeError(f"Epoch-1 validation is missing: {result_file}")
    start, terminal = epoch1[-1], curve[-1]
    summary = {
        "arm": arm,
        "seed": seed,
        "gate_mode": gate_mode,
        "result_path": str(result_file.resolve()),
        "epoch1_endpoint": start["endpoint_1024x256"],
        "step1232_endpoint": terminal["endpoint_1024x256"],
        "retention": terminal["endpoint_1024x256"] - start["endpoint_1024x256"],
        "terminal_valid_loss": terminal["valid_loss"],
        "terminal_train_loss": terminal["train/loss"],
        "curve": curve,
        "wall_clock_sec": result["wall_clock_sec"],
    }
    path = run_root() / "analysis" / "screens" / f"{arm}-s{seed}-{gate_mode}.json"
    atomic_write_json(path, summary)
    return summary


def classify_effect(effect: float, candidate_retention: float) -> str:
    if effect <= -0.05:
        return "strong_cause"
    if effect >= -0.02 and candidate_retention >= -0.02:
        return "stable"
    return "gray"


def paired_effect(
    baseline: str,
    candidate: str,
    seed: int,
    gate_mode: str = "fixed",
) -> dict[str, Any]:
    base = screen_summary(baseline, seed, gate_mode)
    target = screen_summary(candidate, seed, gate_mode)
    effect = target["retention"] - base["retention"]
    return {
        "baseline": baseline,
        "candidate": candidate,
        "seed": seed,
        "gate_mode": gate_mode,
        "baseline_retention": base["retention"],
        "candidate_retention": target["retention"],
        "factor_effect": effect,
        "classification": classify_effect(effect, target["retention"]),
    }


def aggregate_effect(
    baseline: str,
    candidate: str,
    seeds: list[int],
    gate_mode: str = "fixed",
) -> dict[str, Any]:
    rows = [paired_effect(baseline, candidate, seed, gate_mode) for seed in seeds]
    effects = [row["factor_effect"] for row in rows]
    strong = sum(row["classification"] == "strong_cause" for row in rows)
    stable = sum(row["classification"] == "stable" for row in rows)
    mean = sum(effects) / len(effects)
    if strong == len(rows) and mean <= -0.05:
        decision = "confirmed_cause"
    elif len(rows) == 3 and strong >= 2 and mean <= -0.05:
        decision = "confirmed_cause"
    elif stable == len(rows):
        decision = "no_evidence"
    else:
        decision = "inconclusive"
    payload = {
        "baseline": baseline,
        "candidate": candidate,
        "gate_mode": gate_mode,
        "seeds": seeds,
        "rows": rows,
        "mean_factor_effect": mean,
        "decision": decision,
    }
    name = f"{baseline}-vs-{candidate}-{'-'.join(map(str, seeds))}-{gate_mode}.json"
    atomic_write_json(run_root() / "analysis" / "comparisons" / name, payload)
    return payload


def formal_summary(arms: list[str], seeds: list[int]) -> dict[str, Any]:
    rows = []
    for arm in arms:
        for seed in seeds:
            path = experiment.result_path(arm, seed, "formal", "fixed")
            result = load_json(path)
            for role in ("best", "last"):
                checkpoint = result[f"{role}_checkpoint"]
                metrics = checkpoint["metrics"]
                rows.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "role": role,
                        "epoch": checkpoint["epoch"],
                        "endpoint_1024x256": metrics.get(ENDPOINT),
                        "valid_accuracy": metrics.get("valid/accuracy"),
                        "valid_loss": metrics.get("valid/loss"),
                        "wall_clock_sec": result["wall_clock_sec"],
                    }
                )
    payload = {
        "status": "completed",
        "created_at_utc": utc_now(),
        "arms": arms,
        "seeds": seeds,
        "rows": rows,
    }
    atomic_write_json(run_root() / "analysis" / "formal-summary.json", payload)
    return payload


def write_curve_csv(summaries: list[dict[str, Any]]) -> Path:
    rows = []
    for summary in summaries:
        for point in summary["curve"]:
            rows.append(
                {
                    "arm": summary["arm"],
                    "seed": summary["seed"],
                    "gate_mode": summary["gate_mode"],
                    **point,
                }
            )
    path = run_root() / "analysis" / "validation-curves.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", choices=ARMS, required=True)
    parser.add_argument("--seeds", nargs="+", choices=SEEDS, type=int, required=True)
    parser.add_argument("--gate-mode", choices=GATE_MODES, default="fixed")
    args = parser.parse_args()
    summaries = [
        screen_summary(arm, seed, args.gate_mode)
        for arm in args.arms
        for seed in args.seeds
    ]
    path = write_curve_csv(summaries)
    print(json.dumps({"status": "completed", "curve_csv": str(path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
