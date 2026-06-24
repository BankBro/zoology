#!/usr/bin/env python3
"""Validate early-window trace config and smoke outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path("/home/lyj/mnt/project/zoology")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from zoology.config import TrainConfig


REQUIRED_METRICS = {
    "attn/gd_residual_write_strength_effective_cap",
    "attn/gd_residual_write_strength_scheduled_cap",
    "attn/gd_residual_write_strength_cap_release_progress",
    "attn/gd_residual_write_strength_cap_hit_ratio",
    "attn/gd_residual_update_norm_mean",
    "attn/gd_residual_update_norm_p95",
    "attn/gd_residual_update_norm_max",
    "attn/gd_residual_m_norm_mean",
    "attn/gd_residual_m_norm_max",
    "attn/gd_residual_lambda_mean",
    "attn/gd_residual_inject_ratio",
    "attn/gd_residual_read_margin_top1_top2_mean",
    "attn/gd_residual_read_entropy_mean",
    "attn/gd_residual_read_selected_mass_mean",
    "attn/gd_residual_read_candidate_churn_mean",
    "attn/gd_residual_read_candidate_top1_flip_rate",
}

REQUIRED_TRACE_FIELDS = {
    "run_id",
    "global_step",
    "valid_batch_idx",
    "sample_idx",
    "input_hash",
    "target_hash",
    "layer_idx",
    "head_idx",
    "query_pos",
    "block_idx",
    "token_idx",
    "read_topk",
    "topk_candidate_ids",
    "topk_scores",
    "topk_probs",
    "margin_top1_top2",
    "entropy",
    "selected_mass",
}


def _parse_ints(raw: str) -> list[int]:
    values = [part.strip() for part in str(raw).split(",") if part.strip()]
    return [int(value) for value in values]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_metrics_yaml(path: Path) -> set[str]:
    metrics: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("- "):
                metrics.add(line[2:].strip())
    return metrics


def _validate_metrics_yaml(path: Path) -> dict[str, Any]:
    metrics = _load_metrics_yaml(path)
    missing = sorted(REQUIRED_METRICS - metrics)
    return {
        "path": str(path),
        "metric_count": len(metrics),
        "missing": missing,
        "passed": not missing,
    }


def _validate_train_config_support() -> dict[str, Any]:
    fields = getattr(TrainConfig, "model_fields", None)
    if fields is None:
        fields = getattr(TrainConfig, "__fields__", {})
    has_field = "read_trace_train_steps" in fields
    return {
        "has_read_trace_train_steps": has_field,
        "passed": bool(has_field),
    }


def _validate_generated_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "checked": False, "passed": True}
    text = path.read_text(encoding="utf-8")
    checks = {
        "contains_read_trace_train_steps": "read_trace_train_steps" in text,
        "contains_builder_args": "_builder_args" in text or "configs = build_configs(" in text,
    }
    return {
        "path": str(path),
        "checked": True,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _validate_trace_dir(trace_dir: Path | None, required_steps: list[int]) -> dict[str, Any]:
    if trace_dir is None:
        return {"trace_dir": None, "checked": False, "passed": True}
    early_metrics_path = trace_dir / "early_window_metrics.jsonl"
    result: dict[str, Any] = {
        "trace_dir": str(trace_dir),
        "checked": True,
        "early_metrics_path": str(early_metrics_path),
        "early_metrics_exists": early_metrics_path.exists(),
        "step_checks": [],
    }
    if not early_metrics_path.exists():
        result["passed"] = False
        return result

    early_rows = _read_jsonl(early_metrics_path)
    seen_steps = {int(row.get("train_step")) for row in early_rows if row.get("train_step") is not None}
    result["early_metrics_rows"] = len(early_rows)
    result["early_metrics_steps"] = sorted(seen_steps)
    result["missing_early_metric_steps"] = sorted(set(required_steps) - seen_steps)

    required_metric_suffixes = {
        "early_window/attn/gd_residual_remote_read_topk_effective",
        "early_window/attn/gd_residual_update_norm_mean",
        "early_window/attn/gd_residual_update_norm_p95",
        "early_window/attn/gd_residual_update_norm_max",
    }
    metric_keys = set().union(*(row.keys() for row in early_rows)) if early_rows else set()
    result["missing_early_metric_keys"] = sorted(required_metric_suffixes - metric_keys)

    all_steps_passed = True
    for step in required_steps:
        trace_path = trace_dir / f"train_step_{step}" / "read_trace.jsonl"
        step_result: dict[str, Any] = {
            "train_step": int(step),
            "trace_path": str(trace_path),
            "exists": trace_path.exists(),
            "records": 0,
            "missing_fields": [],
        }
        if trace_path.exists():
            rows = _read_jsonl(trace_path)
            step_result["records"] = len(rows)
            if rows:
                missing_fields = sorted(REQUIRED_TRACE_FIELDS - set(rows[0].keys()))
                step_result["missing_fields"] = missing_fields
                step_result["global_steps"] = sorted(
                    {int(row["global_step"]) for row in rows if row.get("global_step") is not None}
                )
                step_result["passed"] = (
                    not missing_fields
                    and len(rows) > 0
                    and step_result["global_steps"] == [int(step)]
                )
            else:
                step_result["passed"] = False
        else:
            step_result["passed"] = False
        all_steps_passed = all_steps_passed and bool(step_result["passed"])
        result["step_checks"].append(step_result)

    result["passed"] = (
        all_steps_passed
        and not result["missing_early_metric_steps"]
        and not result["missing_early_metric_keys"]
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-yaml", type=Path, default=SCRIPT_DIR / "metrics.yaml")
    parser.add_argument("--trace-dir", type=Path, default=None)
    parser.add_argument("--generated-config", type=Path, default=None)
    parser.add_argument("--required-steps", type=str, default="0,1,2")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=SCRIPT_DIR / "outputs" / "config-runtime-smoke.json",
    )
    args = parser.parse_args()

    required_steps = _parse_ints(args.required_steps)
    report = {
        "metrics_yaml": _validate_metrics_yaml(args.metrics_yaml),
        "train_config_support": _validate_train_config_support(),
        "generated_config": _validate_generated_config(args.generated_config),
        "trace": _validate_trace_dir(args.trace_dir, required_steps),
    }
    report["passed"] = all(
        section.get("passed", False)
        for section in report.values()
        if isinstance(section, dict)
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
