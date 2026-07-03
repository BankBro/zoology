#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib.util
import json
import os
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
EXPERIMENT_ID = "20260703-03-flash-vqg-softcap-stability-screen"
ARTIFACT_DIR = REPO_ROOT / "docs/artifacts" / EXPERIMENT_ID
METRICS_YAML = SCRIPT_DIR / "metrics.yaml"
GRAD_ACCUMULATION_STEPS = 4
TRACE_STEPS: list[int] = []
DEFAULT_MAX_TRAIN_STEPS = 704


def _load_source():
    spec = importlib.util.spec_from_file_location("softcap_default_dropout_base", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load source script: {SOURCE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASEMOD = _load_source()
TARGETS = (
    "baseline-r2-no-trace",
    "inject-softcap0p5-r2",
    "inject-softcap0p5-linear512-r2",
    "update-softcap0p5-r2",
    "update-softcap0p5-linear512-r2",
)


def _optimizer_to_train_forward_steps(optimizer_step: int) -> int:
    return int(optimizer_step) * GRAD_ACCUMULATION_STEPS


_R2_BASE = {
    **BASEMOD._R2,
    "description": "",
    "embed_dropout": BASEMOD.DEFAULT_EMBED_DROPOUT,
    "residual_norm_mode": None,
    "fox_gd_residual_update_norm_cap": None,
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


def _variant(**updates: Any) -> dict[str, Any]:
    spec = dict(_R2_BASE)
    spec.update(updates)
    return spec


VARIANTS: dict[str, dict[str, Any]] = {
    "baseline-r2-no-trace": _variant(
        description="default-dropout fixed read_topk=2 baseline, no trace and no softcap",
    ),
    "inject-softcap0p5-r2": _variant(
        description="default-dropout fixed read_topk=2 with injection smooth_p4 softcap ratio=0.5",
        fox_gd_residual_injection_softcap_ratio=0.5,
        fox_gd_residual_injection_softcap_mode="smooth_p4",
    ),
    "inject-softcap0p5-linear512-r2": _variant(
        description="injection smooth_p4 softcap ratio=0.5 plus residual injection linear warmup 0->512 optimizer steps",
        fox_gd_residual_injection_softcap_ratio=0.5,
        fox_gd_residual_injection_softcap_mode="smooth_p4",
        warmup_start_optimizer_step=0,
        warmup_end_optimizer_step=512,
        fox_gd_residual_injection_warmup_start_train_steps=_optimizer_to_train_forward_steps(0),
        fox_gd_residual_injection_warmup_end_train_steps=_optimizer_to_train_forward_steps(512),
    ),
    "update-softcap0p5-r2": _variant(
        description="default-dropout fixed read_topk=2 with M_state update_norm smooth_p4 softcap=0.5",
        fox_gd_residual_update_norm_softcap=0.5,
        fox_gd_residual_update_norm_softcap_mode="smooth_p4",
    ),
    "update-softcap0p5-linear512-r2": _variant(
        description="M_state update_norm smooth_p4 softcap=0.5 plus residual injection linear warmup 0->512 optimizer steps",
        fox_gd_residual_update_norm_softcap=0.5,
        fox_gd_residual_update_norm_softcap_mode="smooth_p4",
        warmup_start_optimizer_step=0,
        warmup_end_optimizer_step=512,
        fox_gd_residual_injection_warmup_start_train_steps=_optimizer_to_train_forward_steps(0),
        fox_gd_residual_injection_warmup_end_train_steps=_optimizer_to_train_forward_steps(512),
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
        description="compatibility alias used internally by the default-dropout wrapper",
    )
    compat_variants["fixed-r2-baseline"] = _variant(
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


def _patch_softcap_support() -> None:
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
        if int(float(settings.get("fox_remote_read_topk", -1))) != 2:
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
                "update_softcap": spec.get("fox_gd_residual_update_norm_softcap"),
                "update_softcap_mode": spec.get("fox_gd_residual_update_norm_softcap_mode"),
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


def _softcap_metrics_rows(early_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by704 = _latest_by_variant_machine_step(early_rows, 704)
    rows: list[dict[str, Any]] = []
    keys = (
        "gd_residual_inject_ratio",
        "gd_residual_injection_softcap_hit_ratio",
        "gd_residual_injection_softcap_scale_mean",
        "gd_residual_injection_softcap_scale_min",
        "gd_residual_injection_softcap_pre_ratio_mean",
        "gd_residual_update_norm_softcap_hit_ratio",
        "gd_residual_update_norm_softcap_scale_mean",
        "gd_residual_update_norm_softcap_scale_min",
        "gd_residual_update_norm_max",
        "gd_residual_update_norm_p95",
        "gd_residual_m_norm_max",
        "gd_residual_lambda_mean",
    )
    for target in TARGETS:
        spec = VARIANTS[target]
        for machine in ("2080ti", "3090"):
            row = by704.get((target, machine), {})
            out: dict[str, Any] = {
                "variant": target,
                "machine": machine,
                "update_softcap": spec.get("fox_gd_residual_update_norm_softcap"),
                "injection_softcap_ratio": spec.get("fox_gd_residual_injection_softcap_ratio"),
                "warmup_end_optimizer_step": spec.get("warmup_end_optimizer_step", ""),
                "loss_step704": row.get("loss", ""),
            }
            for key in keys:
                out[key] = row.get(key, "")
            rows.append(out)
    return rows


def run_collect(args: Any) -> int:
    code = _ORIGINAL_RUN_COLLECT(args)
    artifact_dir = args.artifact_dir
    run_rows = _read_csv(artifact_dir / "run-summary.csv")
    early_rows = _read_csv(artifact_dir / "early-window-summary.csv")
    gap_rows = _variant_gap_rows(run_rows)
    softcap_rows = _softcap_metrics_rows(early_rows)
    _write_csv(artifact_dir / "cross-machine-comparison.csv", gap_rows)
    _write_csv(artifact_dir / "softcap-metrics-summary.csv", softcap_rows)
    readme = (
        f"# {EXPERIMENT_ID}\n\n"
        "本 artifact 收尾 default-dropout smooth cap 1ep stability screen. "
        "本轮保持 `read_topk=2`, `write_topk=4`, `embed_dropout=0.1`, canonical cache/init, "
        "并明确关闭 read trace. Smooth cap 使用 `scale=(1+(x/cap)^4)^(-1/4)`, cap=0.5.\n\n"
        "核心文件:\n\n"
        "- `run-summary.csv`: per-run final/best metrics.\n"
        "- `cross-machine-comparison.csv`: 2080ti/3090 final hard gap by variant.\n"
        "- `softcap-metrics-summary.csv`: step704 softcap hit/scale and residual metrics.\n"
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
            "softcap_formula": "scale=(1+(x/cap)^4)^(-1/4)",
            "softcap_cap": 0.5,
            "trace_mode": "read_trace_disabled",
            "cross_machine_rows": len(gap_rows),
            "softcap_metrics_rows": len(softcap_rows),
            "diagnostic_note": "smooth_p4 softcaps are default-off candidate stabilization controls, not promoted defaults.",
        }
    )
    _save_json(metadata_path, metadata)
    return code


def main() -> int:
    _patch_identity()
    _patch_softcap_support()
    BASEMOD.run_collect = run_collect
    BASEMOD.BASE.run_collect = run_collect
    os.environ["FLASH_VQG_READ_TRACE_MODE"] = "disabled"
    return BASEMOD.main()


if __name__ == "__main__":
    raise SystemExit(main())
