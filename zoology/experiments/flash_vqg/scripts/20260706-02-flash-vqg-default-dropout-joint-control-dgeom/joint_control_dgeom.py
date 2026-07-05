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
EXPERIMENT_ID = "20260706-02-flash-vqg-default-dropout-joint-control-dgeom"
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

FORMAL_TARGETS = (
    "r16-update-softcap0p5-injwarm512-rerun",
    "r8-update-softcap0p5-injwarm512",
    "r4-update-softcap0p5-injwarm512",
    "r2-update-softcap0p5-injwarm512",
    "r16-injwarm512-only",
)

DGEOM_TARGETS = (
    "dgeom-r16-update-softcap0p5-injwarm512",
    "dgeom-r16-injwarm512-only",
    "dgeom-r4-update-softcap0p5-injwarm512",
)

TARGETS = (
    *FORMAL_TARGETS,
    *DGEOM_TARGETS,
)

DGEOM_TRACE_TRAIN_STEPS = (0, 64, 256, 512, 703)


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
    "run_kind": "formal",
    "d_geometry_trace_train_steps": [],
    "fox_gd_residual_d_geometry_trace_enabled": False,
    "fox_gd_residual_d_geometry_max_pairs_per_group": 4096,
    "fox_gd_residual_d_geometry_top_hotspots": 64,
}


def _variant(**updates: Any) -> dict[str, Any]:
    spec = dict(_COMMON_BASE)
    spec.update(updates)
    return spec


VARIANTS: dict[str, dict[str, Any]] = {
    "r16-update-softcap0p5-injwarm512-rerun": _variant(
        description=(
            "formal rerun: read_topk=16 with smooth_p4 update_norm softcap=0.5 and "
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
    "r8-update-softcap0p5-injwarm512": _variant(
        description="formal: read_topk=8 with update softcap=0.5 and residual injection warmup 0->512 optimizer steps",
        fox_remote_read_topk=8,
        fox_gd_residual_update_norm_softcap=0.5,
        fox_gd_residual_update_norm_softcap_mode="smooth_p4",
        fox_gd_residual_injection_warmup_start_train_steps=_optimizer_to_train_forward_steps(0),
        fox_gd_residual_injection_warmup_end_train_steps=_optimizer_to_train_forward_steps(512),
        fox_gd_residual_injection_warmup_eval_policy="final",
        warmup_start_optimizer_step=0,
        warmup_end_optimizer_step=512,
    ),
    "r4-update-softcap0p5-injwarm512": _variant(
        description="formal: read_topk=4 with update softcap=0.5 and residual injection warmup 0->512 optimizer steps",
        fox_remote_read_topk=4,
        fox_gd_residual_update_norm_softcap=0.5,
        fox_gd_residual_update_norm_softcap_mode="smooth_p4",
        fox_gd_residual_injection_warmup_start_train_steps=_optimizer_to_train_forward_steps(0),
        fox_gd_residual_injection_warmup_end_train_steps=_optimizer_to_train_forward_steps(512),
        fox_gd_residual_injection_warmup_eval_policy="final",
        warmup_start_optimizer_step=0,
        warmup_end_optimizer_step=512,
    ),
    "r2-update-softcap0p5-injwarm512": _variant(
        description="formal: read_topk=2 with update softcap=0.5 and residual injection warmup 0->512 optimizer steps",
        fox_remote_read_topk=2,
        fox_gd_residual_update_norm_softcap=0.5,
        fox_gd_residual_update_norm_softcap_mode="smooth_p4",
        fox_gd_residual_injection_warmup_start_train_steps=_optimizer_to_train_forward_steps(0),
        fox_gd_residual_injection_warmup_end_train_steps=_optimizer_to_train_forward_steps(512),
        fox_gd_residual_injection_warmup_eval_policy="final",
        warmup_start_optimizer_step=0,
        warmup_end_optimizer_step=512,
    ),
    "r16-injwarm512-only": _variant(
        description="formal ablation: read_topk=16 with residual injection warmup only, no update softcap",
        fox_gd_residual_injection_warmup_start_train_steps=_optimizer_to_train_forward_steps(0),
        fox_gd_residual_injection_warmup_end_train_steps=_optimizer_to_train_forward_steps(512),
        fox_gd_residual_injection_warmup_eval_policy="final",
        warmup_start_optimizer_step=0,
        warmup_end_optimizer_step=512,
    ),
    "dgeom-r16-update-softcap0p5-injwarm512": _variant(
        description=(
            "diagnostic only: D-direction geometry trace for read_topk=16 joint control; "
            "not used for formal pass/fail"
        ),
        run_kind="d_geometry",
        d_geometry_trace_train_steps=list(DGEOM_TRACE_TRAIN_STEPS),
        fox_gd_residual_update_norm_softcap=0.5,
        fox_gd_residual_update_norm_softcap_mode="smooth_p4",
        fox_gd_residual_injection_warmup_start_train_steps=_optimizer_to_train_forward_steps(0),
        fox_gd_residual_injection_warmup_end_train_steps=_optimizer_to_train_forward_steps(512),
        fox_gd_residual_injection_warmup_eval_policy="final",
        fox_gd_residual_d_geometry_trace_enabled=True,
        warmup_start_optimizer_step=0,
        warmup_end_optimizer_step=512,
    ),
    "dgeom-r16-injwarm512-only": _variant(
        description="diagnostic only: D-direction geometry trace for read_topk=16 injection-warmup-only ablation",
        run_kind="d_geometry",
        d_geometry_trace_train_steps=list(DGEOM_TRACE_TRAIN_STEPS),
        fox_gd_residual_injection_warmup_start_train_steps=_optimizer_to_train_forward_steps(0),
        fox_gd_residual_injection_warmup_end_train_steps=_optimizer_to_train_forward_steps(512),
        fox_gd_residual_injection_warmup_eval_policy="final",
        fox_gd_residual_d_geometry_trace_enabled=True,
        warmup_start_optimizer_step=0,
        warmup_end_optimizer_step=512,
    ),
    "dgeom-r4-update-softcap0p5-injwarm512": _variant(
        description=(
            "diagnostic only: D-direction geometry trace for read_topk=4 joint control; "
            "not used for formal pass/fail"
        ),
        run_kind="d_geometry",
        d_geometry_trace_train_steps=list(DGEOM_TRACE_TRAIN_STEPS),
        fox_remote_read_topk=4,
        fox_gd_residual_update_norm_softcap=0.5,
        fox_gd_residual_update_norm_softcap_mode="smooth_p4",
        fox_gd_residual_injection_warmup_start_train_steps=_optimizer_to_train_forward_steps(0),
        fox_gd_residual_injection_warmup_end_train_steps=_optimizer_to_train_forward_steps(512),
        fox_gd_residual_injection_warmup_eval_policy="final",
        fox_gd_residual_d_geometry_trace_enabled=True,
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
            "fox_gd_residual_d_geometry_trace_enabled",
            "fox_gd_residual_d_geometry_max_pairs_per_group",
            "fox_gd_residual_d_geometry_top_hotspots",
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


def _enable_d_geometry_trace(config: Any, spec: dict[str, Any], trace_output_dir: Any) -> None:
    steps = [int(step) for step in spec.get("d_geometry_trace_train_steps", [])]
    if not steps:
        raise ValueError("D-geometry diagnostic requires non-empty d_geometry_trace_train_steps.")
    if trace_output_dir is None:
        trace_output_dir = SCRIPT_DIR / "outputs" / "_preflight_d_geometry_trace" / str(config.launch_id)
    config.train_inline_event_trace_enabled = True
    config.train_inline_event_trace_steps = steps
    config.train_inline_event_trace_output_dir = str(trace_output_dir)


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
        if spec.get("run_kind") == "d_geometry":
            _enable_d_geometry_trace(config, spec, build_kwargs.get("trace_output_dir"))
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
        if int(float(settings.get("fox_remote_read_topk", -1))) != int(spec.get("fox_remote_read_topk", -1)):
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
            "run_kind": spec.get("run_kind", "formal"),
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
            "passes_screen": passes if spec.get("run_kind", "formal") == "formal" else "",
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


def _iter_d_geometry_records(outputs_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(outputs_dir.glob("*/traces/*/train_inline_step_*/micro_*/d_geometry_trace.jsonl")):
        rel = path.relative_to(outputs_dir)
        queue_dir = str(rel.parts[0]) if len(rel.parts) > 0 else ""
        variant = str(rel.parts[2]) if len(rel.parts) > 2 else ""
        machine = "2080ti" if "2080ti" in queue_dir else ("3090" if "3090" in queue_dir else "")
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                payload["machine"] = machine
                payload["variant"] = variant
                payload["queue_dir"] = queue_dir
                payload["trace_path"] = str(path)
                records.append(payload)
    return records


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:
        return None
    return out


def _mean(values: list[float]) -> float | str:
    return "" if not values else sum(values) / len(values)


def _max_value(values: list[float]) -> float | str:
    return "" if not values else max(values)


def _min_value(values: list[float]) -> float | str:
    return "" if not values else min(values)


def _group_d_geometry(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in records:
        key = (
            row.get("variant", ""),
            row.get("machine", ""),
            row.get("optimizer_step", ""),
            row.get("layer_idx", ""),
            row.get("head_idx", ""),
            row.get("code_idx", ""),
        )
        grouped.setdefault(key, []).append(row)
    metric_keys = (
        "pair_abs_cos_mean",
        "pair_abs_cos_p95",
        "pair_abs_cos_max",
        "signed_cos_mean",
        "resultant_length",
        "effective_rank",
        "condition_number",
        "update_norm_mean",
        "update_norm_p95",
        "update_norm_max",
        "write_strength_mean",
        "raw_topk_mass_mean",
        "addr_z_norm_mean",
    )
    rows: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items()):
        variant, machine, optimizer_step, layer_idx, head_idx, code_idx = key
        out: dict[str, Any] = {
            "variant": variant,
            "machine": machine,
            "optimizer_step": optimizer_step,
            "layer_idx": layer_idx,
            "head_idx": head_idx,
            "code_idx": code_idx,
            "num_groups": len(items),
            "num_events_sum": sum(int(row.get("num_events") or 0) for row in items),
            "num_pairs_sampled_sum": sum(int(row.get("num_pairs_sampled") or 0) for row in items),
        }
        for metric in metric_keys:
            values = [value for value in (_as_float(row.get(metric)) for row in items) if value is not None]
            out[f"{metric}_mean"] = _mean(values)
            out[f"{metric}_max"] = _max_value(values)
            out[f"{metric}_min"] = _min_value(values)
        rows.append(out)
    return rows


def _d_geometry_cross_machine_rows(group_rows: list[dict[str, Any]], run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hard_by_variant_machine: dict[tuple[str, str], float | None] = {}
    for row in run_rows:
        hard_by_variant_machine[
            (str(row.get("variant", "")), str(row.get("machine", "")))
        ] = _float_or_none(row.get("final_1024x256_accuracy"))
    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for row in group_rows:
        key = (
            row.get("variant", ""),
            row.get("optimizer_step", ""),
            row.get("layer_idx", ""),
            row.get("head_idx", ""),
            row.get("code_idx", ""),
        )
        grouped.setdefault(key, {})[str(row.get("machine", ""))] = row
    metrics = (
        "pair_abs_cos_p95_mean",
        "pair_abs_cos_max_mean",
        "effective_rank_mean",
        "condition_number_mean",
        "update_norm_p95_mean",
        "update_norm_max_mean",
        "write_strength_mean_mean",
        "raw_topk_mass_mean_mean",
    )
    rows: list[dict[str, Any]] = []
    for key, machines in sorted(grouped.items()):
        if not {"2080ti", "3090"} <= set(machines):
            continue
        variant, optimizer_step, layer_idx, head_idx, code_idx = key
        r2080 = machines["2080ti"]
        r3090 = machines["3090"]
        hard2080 = hard_by_variant_machine.get((str(variant), "2080ti"))
        hard3090 = hard_by_variant_machine.get((str(variant), "3090"))
        row: dict[str, Any] = {
            "variant": variant,
            "optimizer_step": optimizer_step,
            "layer_idx": layer_idx,
            "head_idx": head_idx,
            "code_idx": code_idx,
            "final_1024x256_2080ti": hard2080,
            "final_1024x256_3090": hard3090,
            "higher_final_machine": (
                "" if hard2080 is None or hard3090 is None else ("2080ti" if hard2080 >= hard3090 else "3090")
            ),
            "num_events_sum_2080ti": r2080.get("num_events_sum", ""),
            "num_events_sum_3090": r3090.get("num_events_sum", ""),
        }
        for metric in metrics:
            v2080 = _as_float(r2080.get(metric))
            v3090 = _as_float(r3090.get(metric))
            row[f"{metric}_2080ti"] = v2080
            row[f"{metric}_3090"] = v3090
            row[f"{metric}_abs_diff"] = (
                "" if v2080 is None or v3090 is None else abs(v2080 - v3090)
            )
        rows.append(row)
    return rows


def _d_geometry_hotspot_rows(group_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hotspot_specs = (
        ("high_pair_abs_cos_p95", "pair_abs_cos_p95_mean", True),
        ("high_update_norm_p95", "update_norm_p95_mean", True),
        ("high_condition_number", "condition_number_mean", True),
        ("low_effective_rank", "effective_rank_mean", False),
    )
    rows: list[dict[str, Any]] = []
    by_vm_step: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in group_rows:
        key = (row.get("variant", ""), row.get("machine", ""), row.get("optimizer_step", ""))
        by_vm_step.setdefault(key, []).append(row)
    for (variant, machine, optimizer_step), items in sorted(by_vm_step.items()):
        for hotspot_name, metric, descending in hotspot_specs:
            ranked = [
                row for row in items if _as_float(row.get(metric)) is not None
            ]
            ranked.sort(key=lambda row: float(_as_float(row.get(metric)) or 0.0), reverse=descending)
            for rank, row in enumerate(ranked[:16], start=1):
                out = {
                    "variant": variant,
                    "machine": machine,
                    "optimizer_step": optimizer_step,
                    "hotspot_metric": hotspot_name,
                    "rank": rank,
                    "metric_value": row.get(metric, ""),
                    "layer_idx": row.get("layer_idx", ""),
                    "head_idx": row.get("head_idx", ""),
                    "code_idx": row.get("code_idx", ""),
                    "num_events_sum": row.get("num_events_sum", ""),
                    "pair_abs_cos_p95_mean": row.get("pair_abs_cos_p95_mean", ""),
                    "update_norm_p95_mean": row.get("update_norm_p95_mean", ""),
                    "condition_number_mean": row.get("condition_number_mean", ""),
                    "effective_rank_mean": row.get("effective_rank_mean", ""),
                }
                rows.append(out)
    return rows


def _write_d_geometry_artifacts(artifact_dir: Path, outputs_dir: Path, run_rows: list[dict[str, Any]]) -> dict[str, Any]:
    records = _iter_d_geometry_records(outputs_dir)
    raw_fieldnames = [
        "variant",
        "machine",
        "optimizer_step",
        "micro_step",
        "layer_idx",
        "block_idx",
        "block_len",
        "head_idx",
        "code_idx",
        "num_events",
        "pair_total_count",
        "pair_sample_count",
        "pair_abs_cos_mean",
        "pair_abs_cos_p50",
        "pair_abs_cos_p90",
        "pair_abs_cos_p95",
        "pair_abs_cos_max",
        "signed_cos_mean",
        "signed_cos_std",
        "adjacent_cos_mean",
        "adjacent_abs_cos_mean",
        "resultant_length",
        "effective_rank",
        "condition_number",
        "update_norm_mean",
        "update_norm_p95",
        "update_norm_max",
        "write_strength_mean",
        "write_strength_p95",
        "raw_topk_mass_mean",
        "addr_z_norm_mean",
        "addr_z_norm_p05",
        "addr_z_norm_p95",
        "addr_z_norm_min",
        "trace_path",
    ]
    _write_csv(artifact_dir / "d-geometry-summary.csv", records, raw_fieldnames)
    group_rows = _group_d_geometry(records)
    cross_rows = _d_geometry_cross_machine_rows(group_rows, run_rows)
    hotspot_rows = _d_geometry_hotspot_rows(group_rows)
    _write_csv(artifact_dir / "d-geometry-by-code-head.csv", group_rows)
    _write_csv(artifact_dir / "d-geometry-cross-machine.csv", cross_rows)
    _write_csv(artifact_dir / "d-geometry-hotspot-summary.csv", hotspot_rows)
    readme = (
        "# D-geometry 诊断说明\n\n"
        "这些文件只来自 diagnostic targets, 不参与 formal pass/fail 判定. "
        "诊断对象是 gd_residual_v1 中真正用于 M_state write 的归一化方向 "
        "`D_pack = normalize((K - codebook) @ addr_proj)`. "
        "统计项按 layer/head/code 聚合 pairwise cosine, effective rank, condition number 和 update_norm.\n\n"
        "- `d-geometry-summary.csv`: 原始 trace JSONL 的扁平表.\n"
        "- `d-geometry-by-code-head.csv`: 按 variant/machine/step/layer/head/code 聚合.\n"
        "- `d-geometry-cross-machine.csv`: 2080ti vs 3090 同组指标差异.\n"
        "- `d-geometry-hotspot-summary.csv`: 高相关, 高 update_norm, 高 condition number, 低 effective rank hotspot.\n"
    )
    (artifact_dir / "d-geometry-readme.md").write_text(readme, encoding="utf-8")
    return {
        "d_geometry_raw_rows": len(records),
        "d_geometry_group_rows": len(group_rows),
        "d_geometry_cross_machine_rows": len(cross_rows),
        "d_geometry_hotspot_rows": len(hotspot_rows),
    }


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
                "fox_gd_residual_d_geometry_trace_enabled": _find_nested_key(
                    config_model, "fox_gd_residual_d_geometry_trace_enabled"
                ),
                "fox_gd_residual_d_geometry_max_pairs_per_group": _find_nested_key(
                    config_model, "fox_gd_residual_d_geometry_max_pairs_per_group"
                ),
                "fox_gd_residual_d_geometry_top_hotspots": _find_nested_key(
                    config_model, "fox_gd_residual_d_geometry_top_hotspots"
                ),
                "read_trace_enabled": config_payload.get("read_trace_enabled", ""),
                "read_trace_train_steps": config_payload.get("read_trace_train_steps", ""),
                "train_inline_event_trace_enabled": config_payload.get(
                    "train_inline_event_trace_enabled", ""
                ),
                "train_inline_event_trace_steps": config_payload.get(
                    "train_inline_event_trace_steps", ""
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
        ("*/traces/*/train_inline_step_*/micro_*/d_geometry_trace.jsonl", "d_geometry_trace"),
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
        f"read_trace_enabled={bool(config.read_trace_enabled)} "
        f"train_inline_event_trace_enabled={bool(config.train_inline_event_trace_enabled)}"
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
    d_geometry_counts = _write_d_geometry_artifacts(artifact_dir, outputs_dir, run_rows)
    _write_csv(artifact_dir / "cross-machine-comparison.csv", gap_rows)
    _write_csv(artifact_dir / "variant-summary.csv", gap_rows)
    _write_csv(artifact_dir / "mechanism-metrics-summary.csv", mechanism_rows)
    _write_csv(artifact_dir / "cache-init-preflight-summary.csv", cache_rows + init_rows)
    _write_csv(artifact_dir / "batch-order-summary.csv", batch_rows)
    _write_csv(artifact_dir / "source-manifest.csv", source_rows)
    _write_execution_status_from_runs(artifact_dir, run_rows)
    _write_variant_decision_from_runs(artifact_dir, run_rows)
    readme = (
        f"# {EXPERIMENT_ID}\n\n"
        "本 artifact 收尾 default-dropout joint control paired 1ep screen 和 D-direction geometry 诊断. "
        "Formal runs 固定 `write_topk=4`, `embed_dropout=0.1`, canonical cache/init/batch order, "
        "并关闭 read trace, hash probe, train inline event trace 和 D-geometry trace. "
        "Diagnostic D-geometry targets 单独开启 train-inline scalar trace, 不参与 formal pass/fail 判定.\n\n"
        "核心文件:\n\n"
        "- `run-summary.csv`: per-run final/best metrics.\n"
        "- `cross-machine-comparison.csv`: 2080ti/3090 final hard gap by variant.\n"
        "- `mechanism-metrics-summary.csv`: final validation residual memory/read/write metrics parsed from logs.\n"
        "- `d-geometry-summary.csv`: D_pack pairwise cosine/rank/update_norm diagnostic rows.\n"
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
            "targets": TARGETS,
            "formal_training_targets": FORMAL_TARGETS,
            "d_geometry_targets": DGEOM_TARGETS,
            "gradient_accumulation_steps": GRAD_ACCUMULATION_STEPS,
            "max_train_steps": DEFAULT_MAX_TRAIN_STEPS,
            "trace_mode": "formal trace disabled; D-geometry diagnostic trace enabled only on dgeom targets",
            "screen_pass_rule": "both machines final 1024x256 >= 0.85 and gap <= 0.04",
            "update_softcap_formula": "scale=(1+(update_norm/cap)^4)^(-1/4), applied to zeta before M_state write",
            "injection_warmup_note": "optimizer step 512 maps to train-forward step 2048 with grad_accumulation_steps=4",
            "d_geometry_note": "D_pack=normalize((K-codebook)@addr_proj); optimizer step 703 is the last in-epoch trace point before 704 total updates",
            "cross_machine_rows": len(gap_rows),
            "mechanism_metrics_rows": len(mechanism_rows),
            "repaired_queue_status_rows_for_collect": repaired_queue_rows,
            **d_geometry_counts,
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
