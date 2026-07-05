#!/usr/bin/env python3
from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get("ZOOLOGY_REPO_ROOT", "/home/lyj/mnt/project/zoology")).resolve()
SOURCE_SCRIPT = (
    REPO_ROOT
    / "zoology/experiments/flash_vqg/scripts/"
    / "20260704-01-flash-vqg-default-dropout-read-support-write-confidence-screen/"
    / "read_support_write_confidence_screen.py"
)
EXPERIMENT_ID = "20260706-01-flash-vqg-default-dropout-r16-mstate-control"
ARTIFACT_DIR = REPO_ROOT / "docs/artifacts" / EXPERIMENT_ID
METRICS_YAML = SCRIPT_DIR / "metrics.yaml"
GRAD_ACCUMULATION_STEPS = 4
DEFAULT_MAX_TRAIN_STEPS = 704
PASS_HARD_ACCURACY = 0.85
PASS_GAP = 0.04


def _load_source():
    spec = importlib.util.spec_from_file_location("mstate_control_base", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load source script: {SOURCE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASEWRAP = _load_source()
BASEMOD = BASEWRAP.BASEMOD

TARGETS = (
    "fixed-r16-baseline",
    "r16-update-softcap0p5",
    "r16-mnorm-cap6",
    "r16-update-softcap0p5-injwarm512",
)


def _optimizer_to_train_forward_steps(optimizer_step: int) -> int:
    return int(optimizer_step) * GRAD_ACCUMULATION_STEPS


_COMMON_BASE = {
    **BASEWRAP._COMMON_BASE,
    "description": "",
    "fox_remote_read_topk": 16,
    "fox_remote_read_topk_initial": None,
    "fox_remote_read_topk_final": None,
    "fox_remote_read_topk_release_start_train_steps": 0,
    "fox_remote_read_topk_release_end_train_steps": 0,
    "fox_remote_read_topk_schedule": "linear_int",
    "fox_remote_read_topk_eval_policy": "scheduled",
    "fox_gd_residual_dense_read_chunked": False,
    "fox_gd_residual_m_norm_cap": None,
    "fox_gd_residual_update_norm_cap": None,
    "fox_gd_residual_update_norm_cap_final": None,
    "fox_gd_residual_update_norm_cap_release_start_train_steps": 0,
    "fox_gd_residual_update_norm_cap_release_end_train_steps": 0,
    "fox_gd_residual_update_norm_cap_eval_policy": "final",
    "fox_gd_residual_update_norm_cap_schedule": "linear",
    "fox_gd_residual_update_norm_softcap": None,
    "fox_gd_residual_update_norm_softcap_mode": "none",
    "fox_gd_residual_injection_warmup_start_train_steps": 0,
    "fox_gd_residual_injection_warmup_end_train_steps": 0,
    "fox_gd_residual_injection_warmup_eval_policy": "scheduled",
    "fox_gd_residual_injection_softcap_ratio": None,
    "fox_gd_residual_injection_softcap_mode": "none",
    "fox_gd_residual_read_confidence_gate_mode": "none",
    "fox_gd_residual_read_confidence_margin_ref": 0.5,
    "fox_gd_residual_read_confidence_temp": 0.25,
    "fox_gd_residual_read_confidence_floor": 0.25,
    "fox_gd_residual_read_softmargin_mode": "none",
    "fox_gd_residual_read_softmargin_tau_max": 3.0,
    "fox_gd_residual_read_softmargin_margin_ref": 0.5,
    "fox_gd_residual_read_softmargin_temp": 0.25,
    "fox_gd_residual_write_strength_mode": "renorm_topk",
    "warmup_start_optimizer_step": 0,
    "warmup_end_optimizer_step": 0,
    "max_train_steps_override": None,
}


def _variant(**updates: Any) -> dict[str, Any]:
    spec = dict(_COMMON_BASE)
    spec.update(updates)
    return spec


VARIANTS: dict[str, dict[str, Any]] = {
    "fixed-r16-baseline": _variant(
        description="default-dropout fixed read_topk=16 baseline, no M_state control",
    ),
    "r16-update-softcap0p5": _variant(
        description="read_topk=16 with smooth_p4 M_state update_norm softcap=0.5",
        fox_gd_residual_update_norm_softcap=0.5,
        fox_gd_residual_update_norm_softcap_mode="smooth_p4",
    ),
    "r16-mnorm-cap6": _variant(
        description="read_topk=16 with hard M_state norm cap=6.0 diagnostic",
        fox_gd_residual_m_norm_cap=6.0,
    ),
    "r16-update-softcap0p5-injwarm512": _variant(
        description=(
            "read_topk=16 with smooth_p4 update_norm softcap=0.5 and "
            "residual injection warmup 0->512 optimizer steps"
        ),
        fox_gd_residual_update_norm_softcap=0.5,
        fox_gd_residual_update_norm_softcap_mode="smooth_p4",
        fox_gd_residual_injection_warmup_start_train_steps=_optimizer_to_train_forward_steps(0),
        fox_gd_residual_injection_warmup_end_train_steps=_optimizer_to_train_forward_steps(512),
        fox_gd_residual_injection_warmup_eval_policy="final",
        warmup_start_optimizer_step=0,
        warmup_end_optimizer_step=512,
    ),
}

_FLASH_KEYS = tuple(
    dict.fromkeys(
        BASEWRAP._FLASH_KEYS
        + (
            "fox_gd_residual_m_norm_cap",
            "fox_gd_residual_update_norm_cap",
            "fox_gd_residual_update_norm_cap_final",
            "fox_gd_residual_update_norm_cap_release_start_train_steps",
            "fox_gd_residual_update_norm_cap_release_end_train_steps",
            "fox_gd_residual_update_norm_cap_eval_policy",
            "fox_gd_residual_update_norm_cap_schedule",
            "fox_gd_residual_update_norm_softcap",
            "fox_gd_residual_update_norm_softcap_mode",
            "fox_gd_residual_injection_warmup_start_train_steps",
            "fox_gd_residual_injection_warmup_end_train_steps",
            "fox_gd_residual_injection_warmup_eval_policy",
            "fox_gd_residual_injection_softcap_ratio",
            "fox_gd_residual_injection_softcap_mode",
            "fox_gd_residual_read_confidence_gate_mode",
            "fox_gd_residual_read_confidence_margin_ref",
            "fox_gd_residual_read_confidence_temp",
            "fox_gd_residual_read_confidence_floor",
            "fox_gd_residual_read_softmargin_mode",
            "fox_gd_residual_read_softmargin_tau_max",
            "fox_gd_residual_read_softmargin_margin_ref",
            "fox_gd_residual_read_softmargin_temp",
            "fox_gd_residual_write_strength_mode",
        )
    )
)

_ORIGINAL_BUILD_CONFIG = BASEMOD.BASE.build_config
_ORIGINAL_FLASH_SETTINGS = BASEMOD.BASE._flash_vqg_settings
_ORIGINAL_RUN_TRAIN = BASEWRAP._ORIGINAL_RUN_TRAIN
_ORIGINAL_RUN_COLLECT = BASEWRAP._ORIGINAL_RUN_COLLECT


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return BASEWRAP._json_default(value)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    keys.append(key)
                    seen.add(key)
        fieldnames = keys
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _dedupe_adjacent(values: list[str]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        if not deduped or deduped[-1] != value:
            deduped.append(value)
    return deduped


def _find_nested_key(payload: Any, key: str) -> Any:
    if isinstance(payload, dict):
        if key in payload:
            return payload[key]
        for value in payload.values():
            found = _find_nested_key(value, key)
            if found not in (None, ""):
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _find_nested_key(value, key)
            if found not in (None, ""):
                return found
    return ""


def _parse_numeric_series(text: str, metric: str) -> list[str]:
    return _dedupe_adjacent(re.findall(rf"{re.escape(metric)}=([-+0-9.eE]+)", text))


def _parse_final_metrics(log_path: Path) -> dict[str, Any]:
    if not log_path.exists():
        return {}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    accuracy_1024 = _parse_numeric_series(text, "valid/mqar_case/accuracy-1024x256")
    valid_accuracy = _parse_numeric_series(text, "valid/accuracy")
    valid_loss = _parse_numeric_series(text, "valid/loss")
    final_1024 = _float_or_none(accuracy_1024[-1]) if accuracy_1024 else None
    best_1024 = max((_float_or_none(value) for value in accuracy_1024), default=None)
    valid_acc_floats = [value for value in (_float_or_none(v) for v in valid_accuracy) if value is not None]
    return {
        "final_1024x256_accuracy": "" if final_1024 is None else final_1024,
        "best_1024x256_accuracy": "" if best_1024 is None else best_1024,
        "best_final_1024x256_gap": (
            "" if best_1024 is None or final_1024 is None else best_1024 - final_1024
        ),
        "final_valid_accuracy": valid_accuracy[-1] if valid_accuracy else "",
        "best_valid_accuracy": max(valid_acc_floats) if valid_acc_floats else "",
        "final_valid_loss": valid_loss[-1] if valid_loss else "",
        "n_validation_summaries": len(accuracy_1024),
        "n_validation_summary_lines": len(re.findall(r"valid/mqar_case/accuracy-1024x256=", text)),
    }


def _variant_config(variant: str) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(f"Unsupported variant: {variant}")
    return VARIANTS[variant]


def _disable_read_trace(config: Any) -> None:
    BASEWRAP._disable_read_trace(config)
    config.read_trace_enabled = False
    config.read_trace_train_steps = []
    config.read_trace_valid_batches = []
    config.read_trace_output_dir = None
    config.read_churn_probe_enabled = False
    config.read_churn_probe_valid_batches = []
    config.train_inline_event_trace_enabled = False
    config.train_inline_event_trace_steps = []
    config.train_inline_event_trace_output_dir = None


def _patch_identity() -> None:
    compat_variants = dict(VARIANTS)
    compat_variants["default-r2"] = _variant(description="compatibility alias")
    compat_variants["fixed-r2-baseline"] = _variant(description="compatibility alias")
    BASEMOD.SCRIPT_DIR = SCRIPT_DIR
    BASEMOD.EXPERIMENT_ID = EXPERIMENT_ID
    BASEMOD.ARTIFACT_DIR = ARTIFACT_DIR
    BASEMOD.METRICS_YAML = METRICS_YAML
    BASEMOD.TARGETS = TARGETS
    BASEMOD.VARIANTS = compat_variants
    BASEMOD.TRACE_TRAIN_STEPS = []
    BASEMOD.DEFAULT_CAPTURE_STEPS = ""
    BASEMOD.DEFAULT_MAX_TRAIN_STEPS = DEFAULT_MAX_TRAIN_STEPS
    BASEMOD.BASE.SCRIPT_DIR = SCRIPT_DIR
    BASEMOD.BASE.EXPERIMENT_ID = EXPERIMENT_ID
    BASEMOD.BASE.ARTIFACT_DIR = ARTIFACT_DIR
    BASEMOD.BASE.TARGETS = tuple(list(TARGETS) + ["default-r2", "fixed-r2-baseline"])
    BASEMOD.BASE.VARIANTS = compat_variants
    BASEMOD.BASE.METRICS_YAML = METRICS_YAML
    BASEMOD.BASE.EXPECTED_TOTAL_OPTIMIZER_STEPS = BASEMOD.BASE.EXPECTED_STEPS_PER_EPOCH


def _patch_support() -> None:
    def build_config(*args: Any, **kwargs: Any):
        variant = kwargs.get("variant")
        if variant is None and len(args) >= 3:
            variant = args[2]
        spec = _variant_config(str(variant))
        build_kwargs = dict(kwargs)
        build_kwargs["target"] = "fixed-r2-baseline"
        build_kwargs["variant"] = "fixed-r2-baseline"
        config = _ORIGINAL_BUILD_CONFIG(*args, **build_kwargs)
        machine_name = build_kwargs.get("machine_name", "unknown")
        config.run_id = f"{EXPERIMENT_ID}-{variant}-s124-d123-b64ga4-{machine_name}"
        config.launch_id = f"fvqg-{EXPERIMENT_ID}-{machine_name}-{variant}"
        config.metrics_white_list = list(config.metrics_white_list or [])
        for key in _FLASH_KEYS:
            BASEMOD.BASE._set_flash_vqg_kwarg(config, key, spec.get(key))
        _disable_read_trace(config)
        return config

    def flash_vqg_settings(config: Any) -> dict[str, Any]:
        settings = _ORIGINAL_FLASH_SETTINGS(config)
        for key in _FLASH_KEYS:
            settings[key] = BASEMOD._flash_setting(config, key)
        return settings

    def variant_settings_match(settings: dict[str, Any], variant: str) -> bool:
        spec = _variant_config(variant)
        if int(float(settings.get("num_codebook_vectors", -1))) != 64:
            return False
        if int(float(settings.get("fox_remote_read_topk", -1))) != 16:
            return False
        if int(float(settings.get("fox_gd_residual_write_topk", -1))) != 4:
            return False
        for key in _FLASH_KEYS:
            expected = spec.get(key)
            actual = settings.get(key)
            if expected is None:
                if actual not in (None, ""):
                    return False
            elif isinstance(expected, str):
                if str(actual) != expected:
                    return False
            else:
                if actual is None or abs(float(actual) - float(expected)) >= 1e-12:
                    return False
        return True

    BASEMOD.BASE.build_config = build_config
    BASEMOD.BASE._flash_vqg_settings = flash_vqg_settings
    BASEMOD.BASE._variant_settings_match = variant_settings_match


_patch_limiter_support = _patch_support


def _variant_gap_rows(run_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, str]]] = {}
    for row in run_rows:
        if row.get("status") != "completed":
            continue
        grouped.setdefault(str(row.get("variant", "")), {})[str(row.get("machine", ""))] = row
    rows: list[dict[str, Any]] = []
    comparison_metrics = {
        "read_selected_mass": "gd_residual_read_selected_mass_mean",
        "read_entropy": "gd_residual_read_entropy_mean",
        "read_margin_top1_top2": "gd_residual_read_margin_top1_top2_mean",
        "update_norm_mean": "gd_residual_update_norm_mean",
        "update_norm_p95": "gd_residual_update_norm_p95",
        "update_norm_max": "gd_residual_update_norm_max",
        "update_softcap_hit_ratio": "gd_residual_update_norm_softcap_hit_ratio",
        "update_softcap_scale_mean": "gd_residual_update_norm_softcap_scale_mean",
        "update_softcap_scale_min": "gd_residual_update_norm_softcap_scale_min",
        "update_softcap_scale_p05": "gd_residual_update_norm_softcap_scale_p05",
        "m_norm_mean": "gd_residual_m_norm_mean",
        "m_norm_max": "gd_residual_m_norm_max",
        "m_norm_cap_hit_ratio": "gd_residual_m_norm_cap_hit_ratio",
        "lambda_mean": "gd_residual_lambda_mean",
        "inject_ratio": "gd_residual_inject_ratio",
        "injection_warmup_factor": "gd_residual_injection_warmup_factor",
        "write_strength_mean": "gd_residual_write_strength_mean",
        "raw_topk_mass_mean": "gd_residual_raw_topk_mass_mean",
        "write_top1_mass_mean": "gd_residual_write_top1_mass_mean",
        "vq_entropy": "vq_c_entropy",
        "vq_usage_max": "vq_c_usage_max",
    }
    for target in TARGETS:
        spec = VARIANTS[target]
        machines = grouped.get(target, {})
        r2080 = machines.get("2080ti", {})
        r3090 = machines.get("3090", {})
        f2080 = _float_or_none(r2080.get("final_1024x256_accuracy"))
        f3090 = _float_or_none(r3090.get("final_1024x256_accuracy"))
        acc2080 = _float_or_none(r2080.get("final_valid_accuracy"))
        acc3090 = _float_or_none(r3090.get("final_valid_accuracy"))
        loss2080 = _float_or_none(r2080.get("final_valid_loss"))
        loss3090 = _float_or_none(r3090.get("final_valid_loss"))
        gap = abs(f2080 - f3090) if f2080 is not None and f3090 is not None else None
        completed_pair = set(machines) >= {"2080ti", "3090"}
        passes = (
            ""
            if f2080 is None or f3090 is None or gap is None
            else f2080 >= PASS_HARD_ACCURACY and f3090 >= PASS_HARD_ACCURACY and gap <= PASS_GAP
        )
        metrics2080 = _parse_last_validation_metrics(Path(str(r2080.get("log_path", "")))) if r2080 else {}
        metrics3090 = _parse_last_validation_metrics(Path(str(r3090.get("log_path", "")))) if r3090 else {}
        row: dict[str, Any] = {
            "variant": target,
            "description": spec["description"],
            "completed_machines": ",".join(sorted(machines)),
            "completed_pair": completed_pair,
            "final_1024x256_2080ti": f2080,
            "final_1024x256_3090": f3090,
            "final_gap": gap,
            "final_gap_percentage_points": None if gap is None else gap * 100.0,
            "final_accuracy_2080ti": acc2080,
            "final_accuracy_3090": acc3090,
            "loss_2080ti": loss2080,
            "loss_3090": loss3090,
            "passes_screen": passes,
            "read_topk": spec.get("fox_remote_read_topk"),
            "write_topk": 4,
            "update_norm_softcap": spec.get("fox_gd_residual_update_norm_softcap"),
            "update_norm_softcap_mode": spec.get("fox_gd_residual_update_norm_softcap_mode"),
            "m_norm_cap": spec.get("fox_gd_residual_m_norm_cap"),
            "injection_warmup_start_train_steps": spec.get("fox_gd_residual_injection_warmup_start_train_steps"),
            "injection_warmup_end_train_steps": spec.get("fox_gd_residual_injection_warmup_end_train_steps"),
            "warmup_end_optimizer_step": spec.get("warmup_end_optimizer_step", ""),
        }
        for label, key in comparison_metrics.items():
            value2080 = _float_or_none(metrics2080.get(key))
            value3090 = _float_or_none(metrics3090.get(key))
            row[f"{label}_2080ti"] = value2080
            row[f"{label}_3090"] = value3090
            row[f"{label}_abs_diff"] = (
                None if value2080 is None or value3090 is None else abs(value2080 - value3090)
            )
        rows.append(row)
    return rows


_FINAL_METRIC_KEYS = (
    "gd_residual_remote_read_topk_effective",
    "gd_residual_read_selected_mass_mean",
    "gd_residual_read_entropy_mean",
    "gd_residual_read_margin_top1_top2_mean",
    "gd_residual_update_norm_mean",
    "gd_residual_update_norm_p95",
    "gd_residual_update_norm_max",
    "gd_residual_update_norm_softcap_hit_ratio",
    "gd_residual_update_norm_softcap_scale_mean",
    "gd_residual_update_norm_softcap_scale_min",
    "gd_residual_update_norm_softcap_scale_p05",
    "gd_residual_update_norm_softcap_effective_cap",
    "gd_residual_m_norm_mean",
    "gd_residual_m_norm_max",
    "gd_residual_m_norm_cap_hit_ratio",
    "gd_residual_lambda_mean",
    "gd_residual_inject_ratio",
    "gd_residual_injection_warmup_factor",
    "gd_residual_write_strength_mean",
    "gd_residual_raw_topk_mass_mean",
    "gd_residual_write_top1_mass_mean",
    "gd_residual_write_q_entropy_mean",
    "gd_residual_write_q_top1_mean",
)


def _tail_text(path: Path, max_bytes: int = 2 * 1024 * 1024) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as f:
        size = path.stat().st_size
        if size > max_bytes:
            f.seek(size - max_bytes)
        return f.read().decode("utf-8", errors="replace")


def _parse_last_validation_metrics(log_path: Path) -> dict[str, str]:
    text = _tail_text(log_path)
    metrics: dict[str, str] = {}
    for key in _FINAL_METRIC_KEYS:
        matches = re.findall(rf"valid/attn/{re.escape(key)}=([^,\]\s]+)", text)
        metrics[key] = matches[-1] if matches else ""
    for key in (
        "c_entropy",
        "c_usage_max",
        "write_top1_mass_mean",
        "write_entropy_mean",
    ):
        matches = re.findall(rf"valid/vq/{re.escape(key)}=([^,\]\s]+)", text)
        metrics[f"vq_{key}"] = matches[-1] if matches else ""
    return metrics


def _mechanism_metrics_rows(run_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in run_rows:
        if row.get("status") != "completed":
            continue
        target = str(row.get("variant", ""))
        if target not in VARIANTS:
            continue
        metrics = _parse_last_validation_metrics(Path(str(row.get("log_path", ""))))
        out: dict[str, Any] = {
            "variant": target,
            "machine": row.get("machine", ""),
            "final_valid_loss": row.get("final_valid_loss", ""),
            "final_valid_accuracy": row.get("final_valid_accuracy", ""),
            "best_valid_accuracy": row.get("best_valid_accuracy", ""),
            "final_1024x256_accuracy": row.get("final_1024x256_accuracy", ""),
            "best_1024x256_accuracy": row.get("best_1024x256_accuracy", ""),
            "log_path": row.get("log_path", ""),
        }
        out.update(metrics)
        rows.append(out)
    return rows


def _latest_queue_rows(outputs_dir: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    completed: list[dict[str, str]] = []
    invalid: list[dict[str, str]] = []
    status_rank = {"pending": 0, "train-started": 1, "completed": 2, "failed": 3}
    for status_path in sorted(outputs_dir.glob("*/queue-status.tsv")):
        rows_by_key: dict[tuple[str, str], dict[str, str]] = {}
        with status_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                row = dict(row)
                row["status_path"] = str(status_path)
                key = (str(row.get("queue", "")), str(row.get("target", "")))
                status = str(row.get("status", ""))
                previous = rows_by_key.get(key)
                previous_status = str(previous.get("status", "")) if previous else ""
                current_rank = status_rank.get(status, -1)
                previous_rank = status_rank.get(previous_status, -1)
                if previous is None or current_rank >= previous_rank:
                    rows_by_key[key] = row
        for row in rows_by_key.values():
            if row.get("status") == "completed":
                completed.append(row)
            else:
                invalid.append(row)
    return completed, invalid


def _duration_seconds(started_at: str, finished_at: str) -> float | str:
    try:
        return (datetime.fromisoformat(finished_at) - datetime.fromisoformat(started_at)).total_seconds()
    except Exception:
        return ""


def _queue_run_rows(outputs_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    completed_queue_rows, invalid_queue_rows = _latest_queue_rows(outputs_dir)
    run_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    for row in invalid_queue_rows:
        invalid_rows.append(
            {
                "queue": row.get("queue", ""),
                "machine": row.get("machine", ""),
                "target": row.get("target", ""),
                "variant": row.get("variant", ""),
                "status": row.get("status", ""),
                "log_path": row.get("log", ""),
                "config_json": row.get("config_json", ""),
                "result_json": row.get("result_json", ""),
                "status_path": row.get("status_path", ""),
            }
        )
    for row in completed_queue_rows:
        target = str(row.get("target", ""))
        variant = str(row.get("variant", target))
        machine = str(row.get("machine", ""))
        log_path = Path(str(row.get("log", "")))
        config_path = Path(str(row.get("config_json", "")))
        result_path = Path(str(row.get("result_json", "")))
        config_payload = _read_json(config_path) if config_path.exists() else {}
        result_payload = _read_json(result_path) if result_path.exists() else {}
        config_model = config_payload.get("model") or {}
        variant_spec = result_payload.get("variant_spec") or VARIANTS.get(variant, {})
        metrics = _parse_final_metrics(log_path)
        duration_seconds = _duration_seconds(str(row.get("started_at", "")), str(row.get("finished_at", "")))
        run_rows.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "machine": machine,
                "queue": row.get("queue", ""),
                "target": target,
                "variant": variant,
                "variant_description": variant_spec.get("description", ""),
                "gpu": row.get("gpu", ""),
                "status": row.get("status", ""),
                "started_at": row.get("started_at", ""),
                "finished_at": row.get("finished_at", ""),
                "duration_seconds": duration_seconds,
                "duration_minutes": (
                    float(duration_seconds) / 60.0 if duration_seconds != "" else ""
                ),
                "run_id": config_payload.get("run_id", ""),
                "launch_id": config_payload.get("launch_id", ""),
                "embed_dropout": (config_payload.get("model") or {}).get("embed_dropout", ""),
                "resid_dropout": (config_payload.get("model") or {}).get("resid_dropout", ""),
                "drop_path": (config_payload.get("model") or {}).get("drop_path", ""),
                "max_epochs": config_payload.get("max_epochs", ""),
                "max_train_steps": config_payload.get("max_train_steps", ""),
                "gradient_accumulation_steps": config_payload.get("gradient_accumulation_steps", ""),
                "num_codebook_vectors": _find_nested_key(config_model, "num_codebook_vectors"),
                "fox_remote_read_topk": _find_nested_key(config_model, "fox_remote_read_topk"),
                "fox_gd_residual_rank": _find_nested_key(config_model, "fox_gd_residual_rank"),
                "fox_gd_residual_write_topk": _find_nested_key(config_model, "fox_gd_residual_write_topk"),
                "fox_gd_residual_update_norm_softcap": _find_nested_key(
                    config_model, "fox_gd_residual_update_norm_softcap"
                ),
                "fox_gd_residual_update_norm_softcap_mode": _find_nested_key(
                    config_model, "fox_gd_residual_update_norm_softcap_mode"
                ),
                "fox_gd_residual_m_norm_cap": _find_nested_key(config_model, "fox_gd_residual_m_norm_cap"),
                "fox_gd_residual_injection_warmup_start_train_steps": _find_nested_key(
                    config_model, "fox_gd_residual_injection_warmup_start_train_steps"
                ),
                "fox_gd_residual_injection_warmup_end_train_steps": _find_nested_key(
                    config_model, "fox_gd_residual_injection_warmup_end_train_steps"
                ),
                "read_trace_enabled": config_payload.get("read_trace_enabled", ""),
                "read_trace_train_steps": config_payload.get("read_trace_train_steps", ""),
                "train_inline_event_trace_enabled": config_payload.get(
                    "train_inline_event_trace_enabled", ""
                ),
                "final_valid_loss": metrics.get("final_valid_loss", ""),
                "final_valid_accuracy": metrics.get("final_valid_accuracy", ""),
                "best_valid_accuracy": metrics.get("best_valid_accuracy", ""),
                "final_1024x256_accuracy": metrics.get("final_1024x256_accuracy", ""),
                "best_1024x256_accuracy": metrics.get("best_1024x256_accuracy", ""),
                "best_final_1024x256_gap": metrics.get("best_final_1024x256_gap", ""),
                "n_validation_summaries": metrics.get("n_validation_summaries", ""),
                "n_validation_summary_lines": metrics.get("n_validation_summary_lines", ""),
                "log_path": str(log_path),
                "log_sha256": _sha256(log_path) if log_path.exists() else "",
                "result_json": str(result_path),
                "result_sha256": _sha256(result_path) if result_path.exists() else "",
                "config_json": str(config_path),
                "config_sha256": _sha256(config_path) if config_path.exists() else "",
                "zoology_commit": ((result_payload.get("env") or {}).get("zoology_commit", "")),
                "flash_vqg_commit": ((result_payload.get("env") or {}).get("flash_vqg_commit", "")),
            }
        )
    return run_rows, invalid_rows


def _preflight_rows(outputs_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    cache_rows: list[dict[str, Any]] = []
    init_rows: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    for path in sorted(outputs_dir.glob("*/cache-hash-*.json")):
        payload = _read_json(path)
        cache = payload.get("cache") or {}
        cache_rows.append(
            {
                "machine": payload.get("machine_name", ""),
                "target": payload.get("target", ""),
                "variant": payload.get("variant", ""),
                "file_count": cache.get("file_count", ""),
                "combined_content_sha256": cache.get("combined_content_sha256", ""),
                "expected_combined_content_sha256": cache.get(
                    "expected_combined_content_sha256", ""
                ),
                "match_expected": cache.get("match_expected", ""),
                "path": str(path),
                "sha256": _sha256(path),
            }
        )
    for path in sorted(outputs_dir.glob("*/init-verify.json")):
        payload = _read_json(path)
        init = payload.get("init_checkpoint") or {}
        init_rows.append(
            {
                "machine": payload.get("machine_name", ""),
                "checkpoint": init.get("checkpoint", ""),
                "expected_model_state_sha256": init.get("expected_model_state_sha256", ""),
                "embedded_model_state_sha256": init.get("embedded_model_state_sha256", ""),
                "actual_model_state_sha256": init.get("actual_model_state_sha256", ""),
                "match_expected": init.get("match_expected", ""),
                "match_embedded": init.get("match_embedded", ""),
                "path": str(path),
                "sha256": _sha256(path),
            }
        )
    for path in sorted(outputs_dir.glob("*/preflight/batch-order-*.json")):
        payload = _read_json(path)
        batch = payload.get("batch_order") or {}
        batch_rows.append(
            {
                "machine": payload.get("machine_name", ""),
                "target": payload.get("target", ""),
                "variant": payload.get("variant", ""),
                "num_batches": batch.get("num_batches", ""),
                "sha256": batch.get("sha256", ""),
                "first_16": json.dumps(batch.get("first_16", []), ensure_ascii=False),
                "read_trace_enabled": payload.get("read_trace_enabled", ""),
                "train_inline_event_trace_enabled": payload.get("train_inline_event_trace_enabled", ""),
                "path": str(path),
                "file_sha256": _sha256(path),
            }
        )
    return cache_rows, init_rows, batch_rows


def _source_manifest_rows(outputs_dir: Path, run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in run_rows:
        for kind, key in (("log", "log_path"), ("config", "config_json"), ("result", "result_json")):
            path = Path(str(row.get(key, "")))
            rows.append(
                {
                    "kind": kind,
                    "machine": row.get("machine", ""),
                    "variant": row.get("variant", ""),
                    "path": str(path),
                    "exists": path.exists(),
                    "sha256": _sha256(path) if path.exists() else "",
                    "submitted_to_git": False,
                }
            )
    for pattern, kind in (
        ("*/queue-status.tsv", "queue_status"),
        ("*/target-manifest.txt", "target_manifest"),
        ("*/init-verify.json", "init_verify"),
        ("*/cache-hash-*.json", "cache_hash"),
        ("*/preflight/*.json", "batch_preflight"),
        ("*/preflight-*.json", "config_preflight"),
    ):
        for path in sorted(outputs_dir.glob(pattern)):
            rows.append(
                {
                    "kind": kind,
                    "machine": "2080ti" if "2080ti" in str(path) else ("3090" if "3090" in str(path) else ""),
                    "variant": "",
                    "path": str(path),
                    "exists": path.exists(),
                    "sha256": _sha256(path) if path.exists() else "",
                    "submitted_to_git": False,
                }
            )
    return rows


def _completed_timestamp(path: Path) -> str:
    if not path.exists():
        return datetime.now().astimezone().isoformat(timespec="seconds")
    return datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec="seconds")


def _repair_queue_status_for_collect(outputs_dir: Path) -> int:
    repaired = 0
    for result_path in sorted(outputs_dir.glob("*/results/*.json")):
        if not result_path.exists():
            continue
        try:
            result = _read_json(result_path)
        except Exception:
            continue
        target = str(result.get("target") or result_path.stem)
        variant = str(result.get("variant") or target)
        machine = str(result.get("machine_name") or "")
        queue_dir = result_path.parents[1]
        status_path = queue_dir / "queue-status.tsv"
        config_path = queue_dir / "configs" / f"{target}.json"
        log_path = queue_dir / "logs" / f"{target}.log"
        if not status_path.exists() or not log_path.exists() or not config_path.exists():
            continue
        log_tail = _tail_text(log_path)
        if "TrainingInterrupted" in log_tail or "Traceback" in log_tail:
            continue
        done_marker = f"[done] target={target} train_status=0" in log_tail
        final_marker = "valid/mqar_case/accuracy-1024x256=" in log_tail
        if not done_marker and not final_marker:
            continue
        with status_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        target_rows = [row for row in rows if row.get("target") == target]
        if any(row.get("status") == "completed" for row in target_rows):
            continue
        if not target_rows:
            continue
        latest = target_rows[-1]
        if str(latest.get("status", "")).startswith(("train-failed", "stopped", "failed")):
            continue
        repaired_row = dict(latest)
        repaired_row["machine"] = machine or repaired_row.get("machine", "")
        repaired_row["target"] = target
        repaired_row["variant"] = variant
        repaired_row["status"] = "completed"
        repaired_row["log"] = str(log_path)
        repaired_row["config_json"] = str(config_path)
        repaired_row["result_json"] = str(result_path)
        repaired_row["finished_at"] = repaired_row.get("finished_at") or _completed_timestamp(log_path)
        fieldnames = [
            "queue",
            "machine",
            "target",
            "variant",
            "gpu",
            "pid",
            "status",
            "log",
            "config_json",
            "result_json",
            "started_at",
            "finished_at",
        ]
        with status_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
            writer.writerow({key: repaired_row.get(key, "") for key in fieldnames})
        repaired += 1
    return repaired


def _write_execution_status_from_runs(artifact_dir: Path, run_rows: list[dict[str, str]]) -> None:
    rows: list[dict[str, Any]] = []
    for row in run_rows:
        rows.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "machine": row.get("machine", ""),
                "target": row.get("target", ""),
                "variant": row.get("variant", ""),
                "queue": row.get("queue", ""),
                "status": row.get("status", ""),
                "result_json": row.get("result_json", ""),
                "config_json": row.get("config_json", ""),
                "log_path": row.get("log_path", ""),
                "started_at": row.get("started_at", ""),
                "finished_at": row.get("finished_at", ""),
                "zoology_commit": row.get("zoology_commit", ""),
                "flash_vqg_commit": row.get("flash_vqg_commit", ""),
            }
        )
    _write_csv(artifact_dir / "execution-status-summary.csv", rows)


def _write_variant_decision_from_runs(artifact_dir: Path, run_rows: list[dict[str, str]]) -> None:
    completed: dict[str, set[str]] = {target: set() for target in TARGETS}
    hard: dict[tuple[str, str], float] = {}
    for row in run_rows:
        variant = str(row.get("variant", ""))
        machine = str(row.get("machine", ""))
        if row.get("status") == "completed" and variant in completed:
            completed[variant].add(machine)
            value = _float_or_none(row.get("final_1024x256_accuracy"))
            if value is not None:
                hard[(variant, machine)] = value
    rows: list[dict[str, Any]] = []
    for target in TARGETS:
        machines = sorted(machine for machine in completed.get(target, set()) if machine)
        f2080 = hard.get((target, "2080ti"))
        f3090 = hard.get((target, "3090"))
        gap = abs(f2080 - f3090) if f2080 is not None and f3090 is not None else None
        passes = (
            f2080 is not None
            and f3090 is not None
            and f2080 >= PASS_HARD_ACCURACY
            and f3090 >= PASS_HARD_ACCURACY
            and gap is not None
            and gap <= PASS_GAP
        )
        rows.append(
            {
                "variant": target,
                "machines_completed": ",".join(machines),
                "completed_pair": set(machines) >= {"2080ti", "3090"},
                "final_1024x256_2080ti": f2080,
                "final_1024x256_3090": f3090,
                "final_gap_percentage_points": None if gap is None else gap * 100.0,
                "passes_screen": passes if gap is not None else "",
                "decision": (
                    "same_seed_pair_rerun_candidate"
                    if passes
                    else "do_not_promote; inspect mechanism metrics or redesign"
                ),
            }
        )
    _write_csv(artifact_dir / "variant-decision-summary.csv", rows)


def run_train(args: Any) -> int:
    spec = _variant_config(args.variant)
    requested_max_train_steps = args.max_train_steps
    override_max_train_steps = spec.get("max_train_steps_override")
    if (
        override_max_train_steps is not None
        and requested_max_train_steps is not None
        and int(requested_max_train_steps) < int(override_max_train_steps)
    ):
        max_train_steps = requested_max_train_steps
    else:
        max_train_steps = override_max_train_steps or requested_max_train_steps
    config = BASEMOD.BASE.build_config(
        target=args.target,
        machine_name=args.machine_name,
        variant=args.variant,
        logger_backend=args.logger_backend,
        trace_output_dir=args.trace_output_dir,
        max_epochs=args.max_epochs,
        max_train_steps=max_train_steps,
        max_validation_batches=args.max_validation_batches,
    )
    BASEMOD._apply_run_suffix(config, args.run_suffix)
    cache_payload = BASEMOD.BASE._hash_cache_for_config(config.data)
    if not cache_payload["match_expected"]:
        raise RuntimeError("MQAR cache content hash does not match canonical hash.")
    init_payload = BASEMOD.BASE._verify_init_checkpoint(args.init_checkpoint)
    if not init_payload["match_expected"] or not init_payload["match_embedded"]:
        raise RuntimeError("Init checkpoint tensor hash does not match canonical hash.")
    print(
        "pretrain_data_guard=PASS "
        f"cache_files={cache_payload['file_count']} "
        f"cache_sha256={cache_payload['combined_content_sha256']} "
        f"init_sha256={init_payload['actual_model_state_sha256']} "
        "read_trace_enabled=false train_inline_event_trace_enabled=false"
    )
    args.max_train_steps = max_train_steps
    return _ORIGINAL_RUN_TRAIN(args)


def run_collect(args: Any) -> int:
    outputs_dir = args.outputs_dir if args.outputs_dir.is_absolute() else (SCRIPT_DIR / args.outputs_dir)
    artifact_dir = args.artifact_dir if args.artifact_dir.is_absolute() else (SCRIPT_DIR / args.artifact_dir)
    args.outputs_dir = outputs_dir
    args.artifact_dir = artifact_dir
    repaired_queue_rows = _repair_queue_status_for_collect(outputs_dir)
    try:
        code = _ORIGINAL_RUN_COLLECT(args)
    except Exception:
        code = 1
    queue_run_rows, queue_invalid_rows = _queue_run_rows(outputs_dir)
    if queue_run_rows:
        run_rows = queue_run_rows
        invalid_rows = queue_invalid_rows
        _write_csv(artifact_dir / "run-summary.csv", run_rows)
        _write_csv(artifact_dir / "invalid-runs.csv", invalid_rows)
    else:
        run_rows = _read_csv(artifact_dir / "run-summary.csv")
        invalid_rows = _read_csv(artifact_dir / "invalid-runs.csv")
    cache_rows, init_rows, batch_rows = _preflight_rows(outputs_dir)
    gap_rows = _variant_gap_rows(run_rows)
    mechanism_rows = _mechanism_metrics_rows(run_rows)
    source_rows = _source_manifest_rows(outputs_dir, run_rows)
    _write_csv(artifact_dir / "cross-machine-comparison.csv", gap_rows)
    _write_csv(artifact_dir / "mechanism-metrics-summary.csv", mechanism_rows)
    _write_csv(artifact_dir / "cache-init-preflight-summary.csv", cache_rows + init_rows)
    _write_csv(artifact_dir / "batch-order-summary.csv", batch_rows)
    _write_csv(artifact_dir / "source-manifest.csv", source_rows)
    _write_execution_status_from_runs(artifact_dir, run_rows)
    _write_variant_decision_from_runs(artifact_dir, run_rows)
    readme = (
        f"# {EXPERIMENT_ID}\n\n"
        "本 artifact 收尾 default-dropout fixed-r16 M_state control paired 1ep screen. "
        "本轮固定 `read_topk=16`, `write_topk=4`, `embed_dropout=0.1`, canonical cache/init/batch order, "
        "并关闭 read trace, train inline event trace 和 shadow dense read. "
        "测试项包括 baseline, smooth update softcap=0.5, hard M_state norm cap=6.0, "
        "以及 update softcap + residual injection warmup 0->512 optimizer steps.\n\n"
        "核心文件:\n\n"
        "- `run-summary.csv`: per-run final/best metrics.\n"
        "- `cross-machine-comparison.csv`: 2080ti/3090 final hard gap by variant.\n"
        "- `mechanism-metrics-summary.csv`: final validation residual memory/read/write metrics parsed from logs.\n"
        "- `early-window-summary.csv`: train-step scalar metrics if available.\n"
        "- `cache-init-preflight-summary.csv`: cache/init hash evidence.\n"
        "- `source-manifest.csv`: mirrored lightweight raw evidence.\n"
    )
    (artifact_dir / "README.md").write_text(readme, encoding="utf-8")
    metadata_path = artifact_dir / "metadata.json"
    metadata = _read_json(metadata_path) if metadata_path.exists() else {}
    completed_run_count = sum(1 for row in run_rows if row.get("status") == "completed")
    metadata.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "variants": VARIANTS,
            "formal_targets": TARGETS,
            "gradient_accumulation_steps": GRAD_ACCUMULATION_STEPS,
            "max_train_steps": DEFAULT_MAX_TRAIN_STEPS,
            "trace_mode": "read_trace_disabled",
            "screen_pass_rule": "both machines final 1024x256 >= 0.85 and gap <= 0.04",
            "update_softcap_formula": "scale=(1+(update_norm/cap)^4)^(-1/4), applied to zeta before M_state write",
            "m_norm_cap_note": "hard diagnostic Frobenius norm cap on per-code M_state",
            "injection_warmup_note": "optimizer step 512 maps to train-forward step 2048 with grad_accumulation_steps=4",
            "cross_machine_rows": len(gap_rows),
            "mechanism_metrics_rows": len(mechanism_rows),
            "repaired_queue_status_rows_for_collect": repaired_queue_rows,
        }
    )
    metadata["summary"] = {
        "run_count": len(run_rows),
        "completed_count": completed_run_count,
        "invalid_count": len(invalid_rows),
        "comparison_count": len(gap_rows),
    }
    _save_json(metadata_path, metadata)
    return 0 if queue_run_rows else code


def main() -> int:
    _patch_identity()
    _patch_support()
    BASEMOD.run_train = run_train
    BASEMOD.run_collect = run_collect
    os.environ["FLASH_VQG_READ_TRACE_MODE"] = "disabled"
    return BASEMOD.main()


if __name__ == "__main__":
    raise SystemExit(main())
