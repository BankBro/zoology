#!/usr/bin/env python3
from __future__ import annotations

import csv
from datetime import datetime
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
    / "20260701-02-flash-vqg-default-dropout-amplifier-trace/default_dropout_amplifier_trace.py"
)
EXPERIMENT_ID = "20260703-04-flash-vqg-safe-limiter-readk2-readk4-screen"
ARTIFACT_DIR = REPO_ROOT / "docs/artifacts" / EXPERIMENT_ID
METRICS_YAML = SCRIPT_DIR / "metrics.yaml"
GRAD_ACCUMULATION_STEPS = 4
TRACE_STEPS: list[int] = []
DEFAULT_MAX_TRAIN_STEPS = 704


def _load_source():
    spec = importlib.util.spec_from_file_location("safe_limiter_default_dropout_base", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load source script: {SOURCE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASEMOD = _load_source()
TARGETS = (
    "r2-baseline",
    "r2-safe-inj-ratio1p0",
    "r2-safe-inj-ratio2p0",
    "r2-updatecap-0p5to0p8-linear512",
    "r2-updatecap-0p5to1p0-linear512",
    "r4-baseline",
    "r4-safe-inj-ratio1p0",
    "r4-safe-inj-ratio2p0",
    "r4-updatecap-0p5to0p8-linear512",
    "r4-updatecap-0p5to1p0-linear512",
)


def _optimizer_to_train_forward_steps(optimizer_step: int) -> int:
    return int(optimizer_step) * GRAD_ACCUMULATION_STEPS


_COMMON_BASE = {
    **BASEMOD._R2,
    "description": "",
    "embed_dropout": BASEMOD.DEFAULT_EMBED_DROPOUT,
    "residual_norm_mode": None,
    "fox_gd_residual_update_norm_cap": None,
    "fox_gd_residual_update_norm_cap_final": None,
    "fox_gd_residual_update_norm_cap_release_start_train_steps": 0,
    "fox_gd_residual_update_norm_cap_release_end_train_steps": 0,
    "fox_gd_residual_update_norm_cap_eval_policy": "final",
    "fox_gd_residual_update_norm_cap_schedule": "linear",
    "fox_gd_residual_update_norm_softcap": None,
    "fox_gd_residual_update_norm_softcap_mode": "none",
    "fox_gd_residual_injection_softcap_ratio": None,
    "fox_gd_residual_injection_softcap_mode": "none",
    "fox_gd_residual_injection_warmup_start_train_steps": 0,
    "fox_gd_residual_injection_warmup_end_train_steps": 0,
    "fox_gd_residual_injection_warmup_eval_policy": "scheduled",
    "warmup_start_optimizer_step": 0,
    "warmup_end_optimizer_step": 0,
}


def _variant(read_topk: int, **updates: Any) -> dict[str, Any]:
    spec = dict(_COMMON_BASE)
    spec.update(
        {
            "fox_remote_read_topk": int(read_topk),
            "fox_remote_read_topk_initial": None,
            "fox_remote_read_topk_final": None,
            "fox_remote_read_topk_release_start_train_steps": 0,
            "fox_remote_read_topk_release_end_train_steps": 0,
            "fox_remote_read_topk_schedule": "linear_int",
            "fox_remote_read_topk_eval_policy": "scheduled",
            "fox_gd_residual_dense_read_chunked": False,
        }
    )
    spec.update(updates)
    return spec


VARIANTS: dict[str, dict[str, Any]] = {
    "r2-baseline": _variant(
        2,
        description="default-dropout fixed read_topk=2 baseline, no trace and no limiter",
    ),
    "r2-safe-inj-ratio1p0": _variant(
        2,
        description="read_topk=2 with safe residual injection limit ratio=1.0",
        fox_gd_residual_injection_softcap_ratio=1.0,
        fox_gd_residual_injection_softcap_mode="safe_smooth_p4",
    ),
    "r2-safe-inj-ratio2p0": _variant(
        2,
        description="read_topk=2 with safe residual injection limit ratio=2.0",
        fox_gd_residual_injection_softcap_ratio=2.0,
        fox_gd_residual_injection_softcap_mode="safe_smooth_p4",
    ),
    "r2-updatecap-0p5to0p8-linear512": _variant(
        2,
        description="read_topk=2 with M_state update_norm hard cap released 0.5->0.8 over 512 optimizer steps",
        fox_gd_residual_update_norm_cap=0.5,
        fox_gd_residual_update_norm_cap_final=0.8,
        fox_gd_residual_update_norm_cap_release_start_train_steps=_optimizer_to_train_forward_steps(0),
        fox_gd_residual_update_norm_cap_release_end_train_steps=_optimizer_to_train_forward_steps(512),
        fox_gd_residual_update_norm_cap_eval_policy="final",
        fox_gd_residual_update_norm_cap_schedule="linear",
    ),
    "r2-updatecap-0p5to1p0-linear512": _variant(
        2,
        description="read_topk=2 with M_state update_norm hard cap released 0.5->1.0 over 512 optimizer steps",
        fox_gd_residual_update_norm_cap=0.5,
        fox_gd_residual_update_norm_cap_final=1.0,
        fox_gd_residual_update_norm_cap_release_start_train_steps=_optimizer_to_train_forward_steps(0),
        fox_gd_residual_update_norm_cap_release_end_train_steps=_optimizer_to_train_forward_steps(512),
        fox_gd_residual_update_norm_cap_eval_policy="final",
        fox_gd_residual_update_norm_cap_schedule="linear",
    ),
    "r4-baseline": _variant(
        4,
        description="default-dropout fixed read_topk=4 baseline, no trace and no limiter",
    ),
    "r4-safe-inj-ratio1p0": _variant(
        4,
        description="read_topk=4 with safe residual injection limit ratio=1.0",
        fox_gd_residual_injection_softcap_ratio=1.0,
        fox_gd_residual_injection_softcap_mode="safe_smooth_p4",
    ),
    "r4-safe-inj-ratio2p0": _variant(
        4,
        description="read_topk=4 with safe residual injection limit ratio=2.0",
        fox_gd_residual_injection_softcap_ratio=2.0,
        fox_gd_residual_injection_softcap_mode="safe_smooth_p4",
    ),
    "r4-updatecap-0p5to0p8-linear512": _variant(
        4,
        description="read_topk=4 with M_state update_norm hard cap released 0.5->0.8 over 512 optimizer steps",
        fox_gd_residual_update_norm_cap=0.5,
        fox_gd_residual_update_norm_cap_final=0.8,
        fox_gd_residual_update_norm_cap_release_start_train_steps=_optimizer_to_train_forward_steps(0),
        fox_gd_residual_update_norm_cap_release_end_train_steps=_optimizer_to_train_forward_steps(512),
        fox_gd_residual_update_norm_cap_eval_policy="final",
        fox_gd_residual_update_norm_cap_schedule="linear",
    ),
    "r4-updatecap-0p5to1p0-linear512": _variant(
        4,
        description="read_topk=4 with M_state update_norm hard cap released 0.5->1.0 over 512 optimizer steps",
        fox_gd_residual_update_norm_cap=0.5,
        fox_gd_residual_update_norm_cap_final=1.0,
        fox_gd_residual_update_norm_cap_release_start_train_steps=_optimizer_to_train_forward_steps(0),
        fox_gd_residual_update_norm_cap_release_end_train_steps=_optimizer_to_train_forward_steps(512),
        fox_gd_residual_update_norm_cap_eval_policy="final",
        fox_gd_residual_update_norm_cap_schedule="linear",
    ),
}

_FLASH_KEYS = (
    "fox_remote_read_topk",
    "fox_remote_read_topk_initial",
    "fox_remote_read_topk_final",
    "fox_remote_read_topk_release_start_train_steps",
    "fox_remote_read_topk_release_end_train_steps",
    "fox_remote_read_topk_schedule",
    "fox_remote_read_topk_eval_policy",
    "fox_gd_residual_dense_read_chunked",
    "fox_gd_residual_update_norm_cap",
    "fox_gd_residual_update_norm_cap_final",
    "fox_gd_residual_update_norm_cap_release_start_train_steps",
    "fox_gd_residual_update_norm_cap_release_end_train_steps",
    "fox_gd_residual_update_norm_cap_eval_policy",
    "fox_gd_residual_update_norm_cap_schedule",
    "fox_gd_residual_update_norm_softcap",
    "fox_gd_residual_update_norm_softcap_mode",
    "fox_gd_residual_injection_softcap_ratio",
    "fox_gd_residual_injection_softcap_mode",
    "fox_gd_residual_injection_warmup_start_train_steps",
    "fox_gd_residual_injection_warmup_end_train_steps",
    "fox_gd_residual_injection_warmup_eval_policy",
)

_ORIGINAL_BUILD_CONFIG = BASEMOD.BASE.build_config
_ORIGINAL_FLASH_SETTINGS = BASEMOD.BASE._flash_vqg_settings
_ORIGINAL_RUN_TRAIN = BASEMOD.run_train
_ORIGINAL_RUN_COLLECT = BASEMOD.run_collect


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return BASEMOD._json_default(value)


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


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _variant_config(variant: str) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(f"Unsupported variant: {variant}")
    return VARIANTS[variant]


def _disable_read_trace(config: Any) -> None:
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
    compat_variants["default-r2"] = _variant(
        2,
        description="compatibility alias used internally by the default-dropout wrapper",
    )
    compat_variants["fixed-r2-baseline"] = _variant(
        2,
        description="compatibility alias used internally by the train-read-topk wrapper",
    )
    BASEMOD.SCRIPT_DIR = SCRIPT_DIR
    BASEMOD.EXPERIMENT_ID = EXPERIMENT_ID
    BASEMOD.ARTIFACT_DIR = ARTIFACT_DIR
    BASEMOD.METRICS_YAML = METRICS_YAML
    BASEMOD.TARGETS = TARGETS
    BASEMOD.VARIANTS = compat_variants
    BASEMOD.TRACE_TRAIN_STEPS = list(TRACE_STEPS)
    BASEMOD.DEFAULT_CAPTURE_STEPS = ""
    BASEMOD.DEFAULT_MAX_TRAIN_STEPS = DEFAULT_MAX_TRAIN_STEPS
    BASEMOD.BASE.SCRIPT_DIR = SCRIPT_DIR
    BASEMOD.BASE.EXPERIMENT_ID = EXPERIMENT_ID
    BASEMOD.BASE.ARTIFACT_DIR = ARTIFACT_DIR
    BASEMOD.BASE.TARGETS = tuple(list(TARGETS) + ["default-r2", "fixed-r2-baseline"])
    BASEMOD.BASE.VARIANTS = compat_variants
    BASEMOD.BASE.METRICS_YAML = METRICS_YAML
    BASEMOD.BASE.EXPECTED_TOTAL_OPTIMIZER_STEPS = BASEMOD.BASE.EXPECTED_STEPS_PER_EPOCH


def _patch_limiter_support() -> None:
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
        if int(float(settings.get("fox_remote_read_topk", -1))) != int(spec.get("fox_remote_read_topk")):
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


def _variant_gap_rows(run_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, str]]] = {}
    for row in run_rows:
        if row.get("status") != "completed":
            continue
        grouped.setdefault(str(row.get("variant", "")), {})[str(row.get("machine", ""))] = row
    rows: list[dict[str, Any]] = []
    for target in TARGETS:
        spec = VARIANTS[target]
        machines = grouped.get(target, {})
        r2080 = machines.get("2080ti", {})
        r3090 = machines.get("3090", {})
        f2080 = _float_or_none(r2080.get("final_1024x256_accuracy"))
        f3090 = _float_or_none(r3090.get("final_1024x256_accuracy"))
        gap = abs(f2080 - f3090) if f2080 is not None and f3090 is not None else None
        rows.append(
            {
                "variant": target,
                "description": spec["description"],
                "completed_machines": ",".join(sorted(machines)),
                "completed_pair": set(machines) >= {"2080ti", "3090"},
                "final_1024x256_2080ti": f2080,
                "final_1024x256_3090": f3090,
                "final_gap": gap,
                "final_gap_percentage_points": None if gap is None else gap * 100.0,
                "final_within_4pp": "" if gap is None else gap <= 0.04,
                "passes_screen": (
                    ""
                    if f2080 is None or f3090 is None or gap is None
                    else f2080 >= 0.82 and f3090 >= 0.82 and gap <= 0.04
                ),
                "read_topk": spec.get("fox_remote_read_topk"),
                "update_cap_initial": spec.get("fox_gd_residual_update_norm_cap"),
                "update_cap_final": spec.get("fox_gd_residual_update_norm_cap_final"),
                "update_cap_schedule": spec.get("fox_gd_residual_update_norm_cap_schedule"),
                "update_cap_release_end_train_steps": spec.get(
                    "fox_gd_residual_update_norm_cap_release_end_train_steps"
                ),
                "injection_softcap_ratio": spec.get("fox_gd_residual_injection_softcap_ratio"),
                "injection_softcap_mode": spec.get("fox_gd_residual_injection_softcap_mode"),
                "warmup_start_optimizer_step": spec.get("warmup_start_optimizer_step", ""),
                "warmup_end_optimizer_step": spec.get("warmup_end_optimizer_step", ""),
            }
        )
    return rows


def _latest_by_variant_machine_step(rows: list[dict[str, str]], step: int) -> dict[tuple[str, str], dict[str, str]]:
    out: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        try:
            row_step = int(float(row.get("train_step", "")))
        except (TypeError, ValueError):
            continue
        if row_step == step:
            out[(str(row.get("target", "")), str(row.get("machine", "")))] = row
    return out


_FINAL_LIMITER_KEYS = (
    "gd_residual_inject_ratio",
    "gd_residual_injection_warmup_factor",
    "gd_residual_injection_softcap_active",
    "gd_residual_injection_softcap_effective_ratio",
    "gd_residual_injection_softcap_mode_safe_smooth_p4",
    "gd_residual_injection_softcap_hit_ratio",
    "gd_residual_injection_softcap_scale_mean",
    "gd_residual_injection_softcap_scale_min",
    "gd_residual_injection_softcap_scale_p05",
    "gd_residual_injection_softcap_pre_ratio_mean",
    "gd_residual_injection_softcap_pre_ratio_min",
    "gd_residual_injection_softcap_pre_ratio_p05",
    "gd_residual_update_norm_cap_active",
    "gd_residual_update_norm_effective_cap",
    "gd_residual_update_norm_cap_hit_ratio",
    "gd_residual_update_norm_softcap_hit_ratio",
    "gd_residual_update_norm_softcap_scale_mean",
    "gd_residual_update_norm_softcap_scale_min",
    "gd_residual_update_norm_softcap_scale_p05",
    "gd_residual_update_norm_mean",
    "gd_residual_update_norm_max",
    "gd_residual_update_norm_p95",
    "gd_residual_m_norm_mean",
    "gd_residual_m_norm_max",
    "gd_residual_lambda_mean",
    "gd_residual_remote_read_topk_effective",
)


def _parse_last_validation_limiter_metrics(log_path: Path) -> dict[str, str]:
    text = _tail_text(log_path, max_bytes=2 * 1024 * 1024)
    metrics: dict[str, str] = {}
    for key in _FINAL_LIMITER_KEYS:
        matches = re.findall(rf"valid/attn/{re.escape(key)}=([^,\]\s]+)", text)
        metrics[key] = matches[-1] if matches else ""
    return metrics


def _limiter_metrics_rows(
    early_rows: list[dict[str, str]],
    run_rows: list[dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    by704 = _latest_by_variant_machine_step(early_rows, 704)
    rows: list[dict[str, Any]] = []
    run_by_variant_machine: dict[tuple[str, str], dict[str, str]] = {}
    if run_rows:
        for row in run_rows:
            run_by_variant_machine[(str(row.get("variant", "")), str(row.get("machine", "")))] = row
    for target in TARGETS:
        spec = VARIANTS[target]
        for machine in ("2080ti", "3090"):
            row = by704.get((target, machine), {})
            run_row = run_by_variant_machine.get((target, machine), {})
            final_metrics = _parse_last_validation_limiter_metrics(Path(str(run_row.get("log_path", ""))))
            out: dict[str, Any] = {
                "variant": target,
                "machine": machine,
                "read_topk": spec.get("fox_remote_read_topk"),
                "update_cap_initial": spec.get("fox_gd_residual_update_norm_cap"),
                "update_cap_final": spec.get("fox_gd_residual_update_norm_cap_final"),
                "injection_softcap_ratio": spec.get("fox_gd_residual_injection_softcap_ratio"),
                "injection_softcap_mode": spec.get("fox_gd_residual_injection_softcap_mode"),
                "warmup_end_optimizer_step": spec.get("warmup_end_optimizer_step", ""),
                "loss_step704": row.get("loss", ""),
                "final_valid_loss": run_row.get("final_valid_loss", ""),
                "final_1024x256_accuracy": run_row.get("final_1024x256_accuracy", ""),
                "log_path": run_row.get("log_path", ""),
            }
            for key in _FINAL_LIMITER_KEYS:
                out[key] = row.get(key, "") or final_metrics.get(key, "")
            rows.append(out)
    return rows


def _tail_text(path: Path, max_bytes: int = 256 * 1024) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as f:
        size = path.stat().st_size
        if size > max_bytes:
            f.seek(size - max_bytes)
        return f.read().decode("utf-8", errors="replace")


def _completed_timestamp(path: Path) -> str:
    if not path.exists():
        return datetime.now().astimezone().isoformat(timespec="seconds")
    return datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec="seconds")


def _repair_queue_status_for_collect(outputs_dir: Path) -> int:
    """Make collect robust to stopped idle wrappers.

    Some queues were intentionally stopped while sleeping after a target had already
    written result/log evidence. The base collector only accepts a `completed`
    queue-status row, so add a collect-only completed row when the run evidence is
    already present. Interrupted duplicate runs are not repaired.
    """
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
        rows: list[dict[str, str]] = []
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
                "queue": row.get("queue", ""),
                "wrapper_status": row.get("status", ""),
                "effective_status": row.get("status", ""),
                "execution_caveat": "rebuilt_from_run_summary",
                "result_json": row.get("result_json", ""),
                "result_exists": bool(row.get("result_json")),
                "config_json": row.get("config_json", ""),
                "config_exists": bool(row.get("config_json")),
                "log_path": row.get("log_path", ""),
                "log_exists": bool(row.get("log_path")),
                "log_done_marker": "",
                "hash_probe_json": "",
                "hash_probe_exists": False,
                "started_at": row.get("started_at", ""),
                "finished_at": row.get("finished_at", ""),
                "zoology_commit": row.get("zoology_commit", ""),
                "flash_vqg_commit": row.get("flash_vqg_commit", ""),
            }
        )
    _write_csv(artifact_dir / "execution-status-summary.csv", rows)


def _write_variant_decision_from_runs(artifact_dir: Path, run_rows: list[dict[str, str]]) -> None:
    rows: list[dict[str, Any]] = []
    completed: dict[str, set[str]] = {target: set() for target in TARGETS}
    for row in run_rows:
        if row.get("status") == "completed" and row.get("variant") in completed:
            completed[str(row["variant"])].add(str(row.get("machine", "")))
    for target in TARGETS:
        machines = sorted(machine for machine in completed.get(target, set()) if machine)
        spec = VARIANTS[target]
        rows.append(
            {
                "target": target,
                "machines_completed": ",".join(machines),
                "completed_pair": set(machines) >= {"2080ti", "3090"},
                "embed_dropout": spec.get("embed_dropout", ""),
                "read_topk": spec.get("fox_remote_read_topk", ""),
                "decision": (
                    "usable_pair_diagnostic"
                    if set(machines) >= {"2080ti", "3090"}
                    else "incomplete_pair_do_not_interpret"
                ),
            }
        )
    _write_csv(artifact_dir / "variant-decision-summary.csv", rows)


def run_train(args: Any) -> int:
    config = BASEMOD.BASE.build_config(
        target=args.target,
        machine_name=args.machine_name,
        variant=args.variant,
        logger_backend=args.logger_backend,
        trace_output_dir=args.trace_output_dir,
        max_epochs=args.max_epochs,
        max_train_steps=args.max_train_steps,
        max_validation_batches=args.max_validation_batches,
    )
    BASEMOD._apply_run_suffix(config, args.run_suffix)

    cache_payload = BASEMOD.BASE._hash_cache_for_config(config.data)
    if not cache_payload["match_expected"]:
        raise RuntimeError(
            "MQAR cache content hash does not match canonical hash; stop before training."
        )
    init_payload = BASEMOD.BASE._verify_init_checkpoint(args.init_checkpoint)
    if not init_payload["match_expected"] or not init_payload["match_embedded"]:
        raise RuntimeError(
            "Init checkpoint tensor hash does not match canonical hash; stop before training."
        )

    print(
        "pretrain_data_guard=PASS "
        f"cache_files={cache_payload['file_count']} "
        f"cache_sha256={cache_payload['combined_content_sha256']} "
        f"init_sha256={init_payload['actual_model_state_sha256']}"
    )
    return _ORIGINAL_RUN_TRAIN(args)


def run_collect(args: Any) -> int:
    repaired_queue_rows = _repair_queue_status_for_collect(args.outputs_dir)
    code = _ORIGINAL_RUN_COLLECT(args)
    artifact_dir = args.artifact_dir
    run_rows = _read_csv(artifact_dir / "run-summary.csv")
    early_rows = _read_csv(artifact_dir / "early-window-summary.csv")
    gap_rows = _variant_gap_rows(run_rows)
    limiter_rows = _limiter_metrics_rows(early_rows, run_rows)
    _write_csv(artifact_dir / "cross-machine-comparison.csv", gap_rows)
    _write_csv(artifact_dir / "limiter-metrics-summary.csv", limiter_rows)
    _write_execution_status_from_runs(artifact_dir, run_rows)
    _write_variant_decision_from_runs(artifact_dir, run_rows)
    invalid_rows = _read_csv(artifact_dir / "invalid-runs.csv")
    completed_run_count = sum(1 for row in run_rows if row.get("status") == "completed")
    completed_variant_count = len(
        {
            row.get("variant", "")
            for row in run_rows
            if row.get("status") == "completed" and row.get("variant")
        }
    )
    readme = (
        f"# {EXPERIMENT_ID}\n\n"
        "本 artifact 收尾 default-dropout safe limiter read_topk=2/4 1ep screen. "
        "本轮保持 `write_topk=4`, `embed_dropout=0.1`, canonical cache/init, "
        "并明确关闭 read trace, train inline event trace 和 shadow dense read. "
        "测试项包括 baseline, safe residual injection limit ratio=1.0/2.0, "
        "以及 M_state update_norm hard cap 从 0.5 线性释放到 0.8/1.0 over 512 optimizer steps.\n\n"
        "核心文件:\n\n"
        "- `run-summary.csv`: per-run final/best metrics.\n"
        "- `cross-machine-comparison.csv`: 2080ti/3090 final hard gap by variant.\n"
        "- `limiter-metrics-summary.csv`: step704/final limiter hit/scale and residual metrics.\n"
        "- `early-window-summary.csv`: train-step scalar metrics.\n"
        "- `cache-init-preflight-summary.csv`: cache/init hash evidence.\n"
        "- `source-manifest.csv`: mirrored lightweight raw evidence.\n"
    )
    (artifact_dir / "README.md").write_text(readme, encoding="utf-8")
    metadata_path = artifact_dir / "metadata.json"
    metadata = _read_json(metadata_path) if metadata_path.exists() else {}
    metadata.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "variants": VARIANTS,
            "gradient_accumulation_steps": GRAD_ACCUMULATION_STEPS,
            "safe_injection_formula": "scale=(1+(ratio/cap)^4)^(-1/4), computed from detached finite-clamped norms",
            "scheduled_update_cap": "hard cap releases linearly from 0.5 to final value over 512 optimizer steps",
            "trace_mode": "read_trace_disabled",
            "cross_machine_rows": len(gap_rows),
            "limiter_metrics_rows": len(limiter_rows),
            "execution_status_rows": len(run_rows),
            "effective_completed_runs": completed_run_count,
            "execution_status_source": (
                "rebuilt from run-summary.csv after collect; invalid-runs.csv keeps stopped "
                "or duplicate raw rows that are not counted as official completed runs"
            ),
            "repaired_queue_status_rows_for_collect": repaired_queue_rows,
            "diagnostic_note": "safe limiters are default-off candidate stabilization controls, not promoted defaults.",
        }
    )
    summary = metadata.get("summary") if isinstance(metadata.get("summary"), dict) else {}
    summary.update(
        {
            "run_count": len(run_rows),
            "completed_count": completed_run_count,
            "comparison_count": len(gap_rows),
            "invalid_count": len(invalid_rows),
            "variant_count": completed_variant_count,
        }
    )
    metadata["summary"] = summary
    _save_json(metadata_path, metadata)
    return code


def main() -> int:
    _patch_identity()
    _patch_limiter_support()
    BASEMOD.run_train = run_train
    BASEMOD.run_collect = run_collect
    os.environ["FLASH_VQG_READ_TRACE_MODE"] = "disabled"
    return BASEMOD.main()


if __name__ == "__main__":
    raise SystemExit(main())
