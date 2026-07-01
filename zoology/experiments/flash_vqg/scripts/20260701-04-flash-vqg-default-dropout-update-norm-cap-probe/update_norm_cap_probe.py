#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
EXPERIMENT_ID = "20260701-04-flash-vqg-default-dropout-update-norm-cap-probe"
ARTIFACT_DIR = REPO_ROOT / "docs/artifacts" / EXPERIMENT_ID
TRACE_STEPS = [0, 16, 64, 128, 256, 384, 512, 704]


def _load_source():
    spec = importlib.util.spec_from_file_location("default_dropout_update_norm_cap_base", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load source script: {SOURCE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASEMOD = _load_source()

_R2_BASE = {
    "kind": "fixed",
    "fox_remote_read_topk": 2,
    "fox_remote_read_topk_initial": None,
    "fox_remote_read_topk_final": None,
    "fox_remote_read_topk_release_start_train_steps": 0,
    "fox_remote_read_topk_release_end_train_steps": 0,
    "fox_remote_read_topk_schedule": "linear_int",
    "fox_remote_read_topk_eval_policy": "scheduled",
    "fox_gd_residual_dense_read_chunked": False,
    "embed_dropout": 0.1,
    "residual_norm_mode": None,
}


def _variant(description: str, update_norm_cap: float | None) -> dict[str, Any]:
    spec = dict(_R2_BASE)
    spec.update(
        {
            "description": description,
            "fox_gd_residual_update_norm_cap": update_norm_cap,
        }
    )
    return spec


TARGETS = ("baseline-r2", "ucap0p5-r2", "ucap0p25-r2")
VARIANTS: dict[str, dict[str, Any]] = {
    "baseline-r2": _variant("default-dropout fixed read_topk=2, no update_norm_cap", None),
    "ucap0p5-r2": _variant("default-dropout fixed read_topk=2, update_norm_cap=0.5", 0.5),
    "ucap0p25-r2": _variant("default-dropout fixed read_topk=2, update_norm_cap=0.25", 0.25),
}


BASEMOD.SCRIPT_DIR = SCRIPT_DIR
BASEMOD.EXPERIMENT_ID = EXPERIMENT_ID
BASEMOD.ARTIFACT_DIR = ARTIFACT_DIR
BASEMOD.TRACE_TRAIN_STEPS = list(TRACE_STEPS)
BASEMOD.DEFAULT_CAPTURE_STEPS = ",".join(str(step) for step in TRACE_STEPS)
BASEMOD.DEFAULT_MAX_EPOCHS = 1
BASEMOD.DEFAULT_MAX_TRAIN_STEPS = 704
BASEMOD.TARGETS = TARGETS
BASEMOD.VARIANTS = VARIANTS
BASEMOD.BASE.EXPECTED_TOTAL_OPTIMIZER_STEPS = BASEMOD.BASE.EXPECTED_STEPS_PER_EPOCH
BASEMOD._patch_base()

_ORIGINAL_BUILD_CONFIG = BASEMOD.BASE.build_config
_ORIGINAL_FLASH_SETTINGS = BASEMOD.BASE._flash_vqg_settings
_ORIGINAL_VARIANT_MATCH = BASEMOD.BASE._variant_settings_match
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


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _patch_update_cap_support() -> None:
    def build_config(
        *,
        target: str,
        machine_name: str,
        variant: str,
        logger_backend: str,
        trace_output_dir: Path,
        max_epochs: int,
        max_train_steps: int | None,
        max_validation_batches: int | None,
    ):
        config = _ORIGINAL_BUILD_CONFIG(
            target=target,
            machine_name=machine_name,
            variant=variant,
            logger_backend=logger_backend,
            trace_output_dir=trace_output_dir,
            max_epochs=max_epochs,
            max_train_steps=max_train_steps,
            max_validation_batches=max_validation_batches,
        )
        spec = BASEMOD._variant_config(variant)
        BASEMOD.BASE._set_flash_vqg_kwarg(
            config,
            "fox_gd_residual_update_norm_cap",
            spec.get("fox_gd_residual_update_norm_cap"),
        )
        return config

    def flash_vqg_settings(config: Any) -> dict[str, Any]:
        settings = _ORIGINAL_FLASH_SETTINGS(config)
        settings["fox_gd_residual_update_norm_cap"] = BASEMOD._flash_setting(
            config, "fox_gd_residual_update_norm_cap"
        )
        return settings

    def variant_settings_match(settings: dict[str, Any], variant: str) -> bool:
        if not _ORIGINAL_VARIANT_MATCH(settings, variant):
            return False
        expected = BASEMOD._variant_config(variant).get("fox_gd_residual_update_norm_cap")
        actual = settings.get("fox_gd_residual_update_norm_cap")
        if expected is None:
            return actual in (None, "")
        return actual is not None and abs(float(actual) - float(expected)) < 1e-12

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
                "update_norm_cap": spec.get("fox_gd_residual_update_norm_cap"),
                "completed_machines": ",".join(sorted(machines)),
                "completed_pair": set(machines) >= {"2080ti", "3090"},
                "final_1024x256_2080ti": f2080,
                "final_1024x256_3090": f3090,
                "final_gap": gap,
                "final_gap_percentage_points": None if gap is None else gap * 100.0,
                "final_within_4pp": "" if gap is None else gap <= 0.04,
                "read_topk": spec.get("fox_remote_read_topk"),
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


def _cap_metrics_rows(early_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by704 = _latest_by_variant_machine_step(early_rows, 704)
    rows: list[dict[str, Any]] = []
    for target in TARGETS:
        spec = VARIANTS[target]
        for machine in ("2080ti", "3090"):
            row = by704.get((target, machine), {})
            rows.append(
                {
                    "variant": target,
                    "machine": machine,
                    "update_norm_cap": spec.get("fox_gd_residual_update_norm_cap"),
                    "loss_step704": row.get("loss", ""),
                    "update_norm_cap_active": row.get("layer_1/attn/gd_residual_update_norm_cap_active", ""),
                    "update_norm_effective_cap": row.get("layer_1/attn/gd_residual_update_norm_effective_cap", ""),
                    "update_norm_cap_hit_ratio": row.get("layer_1/attn/gd_residual_update_norm_cap_hit_ratio", ""),
                    "update_norm_p95": row.get("gd_residual_update_norm_p95", ""),
                    "update_norm_max": row.get("gd_residual_update_norm_max", ""),
                    "m_norm_max": row.get("gd_residual_m_norm_max", ""),
                    "lambda_mean": row.get("gd_residual_lambda_mean", ""),
                    "inject_ratio": row.get("gd_residual_inject_ratio", ""),
                }
            )
    return rows


def run_collect(args: argparse.Namespace) -> int:
    code = _ORIGINAL_RUN_COLLECT(args)
    artifact_dir = args.artifact_dir
    run_rows = _read_csv(artifact_dir / "run-summary.csv")
    early_rows = _read_csv(artifact_dir / "early-window-summary.csv")
    gap_rows = _variant_gap_rows(run_rows)
    cap_rows = _cap_metrics_rows(early_rows)
    _write_csv(artifact_dir / "variant-gap-summary.csv", gap_rows)
    _write_csv(artifact_dir / "cap-metrics-summary.csv", cap_rows)
    readme = (
        f"# {EXPERIMENT_ID}\n\n"
        "本 artifact 收尾 default-dropout update_norm_cap diagnostic probe. "
        "本轮只测试现有 hard update cap 是否能缓解 default-r2 跨机器 1ep 分叉, "
        "不写 official MQAR ledger.\n\n"
        "共同配置: `seed=124`, `data_seed=123`, `cb64-r16`, `read_topk=2`, "
        "`write_topk=4`, canonical MQAR cache, seed124 canonical init, "
        "`embed_dropout=0.1`, `resid_dropout=0.0`, `drop_path=0.0`, `max_train_steps=704`.\n\n"
        "核心文件:\n\n"
        "- `run-summary.csv`: per-run final/best metrics.\n"
        "- `variant-gap-summary.csv`: 2080ti/3090 final hard gap by variant.\n"
        "- `cap-metrics-summary.csv`: step704 cap hit, update norm, M norm, lambda/inject metrics.\n"
        "- `early-window-summary.csv`: train-step eval read/write scalar metrics.\n"
        "- `read-trace-cross-machine-summary.csv`: 2080ti/3090 read support match summary.\n"
        "- `hash-probe-comparison-summary.csv`: train-mode forward/backward hash comparison.\n"
        "- `first-mismatch-summary.csv`: first cross-machine mismatch by target.\n"
        "- `cache-init-preflight-summary.csv`: cache/init hash evidence.\n"
        "- `queue-summary.csv`: queue status.\n"
        "- `source-manifest.csv`: mirrored lightweight raw evidence.\n\n"
        "注意: `update_norm_cap` 使用 detached scale, 是 diagnostic hard cap, 不是最终机制方案.\n"
    )
    (artifact_dir / "README.md").write_text(readme, encoding="utf-8")
    metadata_path = artifact_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    metadata.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "variant_gap_rows": len(gap_rows),
            "cap_metrics_rows": len(cap_rows),
            "variants": VARIANTS,
            "diagnostic_note": "update_norm_cap uses detached scale; do not treat as final deployment fix.",
        }
    )
    _save_json(metadata_path, metadata)
    return code


_patch_update_cap_support()
BASEMOD.run_collect = run_collect


if __name__ == "__main__":
    raise SystemExit(BASEMOD.main())
