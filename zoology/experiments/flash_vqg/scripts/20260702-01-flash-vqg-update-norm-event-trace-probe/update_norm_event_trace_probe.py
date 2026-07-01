#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get("ZOOLOGY_REPO_ROOT", "/home/lyj/mnt/project/zoology")).resolve()
SOURCE_SCRIPT = (
    REPO_ROOT
    / "zoology/experiments/flash_vqg/scripts/"
    / "20260701-04-flash-vqg-default-dropout-update-norm-cap-probe/update_norm_cap_probe.py"
)
EXPERIMENT_ID = "20260702-01-flash-vqg-update-norm-event-trace-probe"
ARTIFACT_DIR = REPO_ROOT / "docs/artifacts" / EXPERIMENT_ID
TRACE_STEPS = [0, 1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 704]


def _load_source():
    spec = importlib.util.spec_from_file_location("update_norm_event_trace_base", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load source script: {SOURCE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASEMOD = _load_source()

TARGETS = ("baseline-r2", "ucap0p5-r2")
VARIANTS: dict[str, dict[str, Any]] = {
    "baseline-r2": {
        **BASEMOD._R2_BASE,
        "description": "default-dropout fixed read_topk=2, no update_norm_cap, update-event trace enabled",
        "fox_gd_residual_update_norm_cap": None,
        "fox_gd_residual_update_event_trace_enabled": True,
        "fox_gd_residual_update_event_trace_topk": 64,
        "fox_gd_residual_update_event_trace_hypothetical_cap": 0.5,
    },
    "ucap0p5-r2": {
        **BASEMOD._R2_BASE,
        "description": "default-dropout fixed read_topk=2, update_norm_cap=0.5, update-event trace enabled",
        "fox_gd_residual_update_norm_cap": 0.5,
        "fox_gd_residual_update_event_trace_enabled": True,
        "fox_gd_residual_update_event_trace_topk": 64,
        "fox_gd_residual_update_event_trace_hypothetical_cap": 0.5,
    },
}


BASEMOD.SCRIPT_DIR = SCRIPT_DIR
BASEMOD.EXPERIMENT_ID = EXPERIMENT_ID
BASEMOD.ARTIFACT_DIR = ARTIFACT_DIR
BASEMOD.TRACE_STEPS = list(TRACE_STEPS)
BASEMOD.BASEMOD.SCRIPT_DIR = SCRIPT_DIR
BASEMOD.BASEMOD.EXPERIMENT_ID = EXPERIMENT_ID
BASEMOD.BASEMOD.ARTIFACT_DIR = ARTIFACT_DIR
BASEMOD.BASEMOD.TRACE_TRAIN_STEPS = list(TRACE_STEPS)
BASEMOD.BASEMOD.DEFAULT_CAPTURE_STEPS = ",".join(str(step) for step in TRACE_STEPS)
BASEMOD.BASEMOD.DEFAULT_MAX_EPOCHS = 1
BASEMOD.BASEMOD.DEFAULT_MAX_TRAIN_STEPS = 704
BASEMOD.BASEMOD.TARGETS = TARGETS
BASEMOD.BASEMOD.VARIANTS = VARIANTS
BASEMOD.BASEMOD.BASE.EXPECTED_TOTAL_OPTIMIZER_STEPS = BASEMOD.BASEMOD.BASE.EXPECTED_STEPS_PER_EPOCH
BASEMOD.BASEMOD._patch_base()
BASEMOD.TARGETS = TARGETS
BASEMOD.VARIANTS = VARIANTS

_ORIGINAL_BUILD_CONFIG = BASEMOD._ORIGINAL_BUILD_CONFIG
_ORIGINAL_FLASH_SETTINGS = BASEMOD._ORIGINAL_FLASH_SETTINGS
_ORIGINAL_VARIANT_MATCH = BASEMOD._ORIGINAL_VARIANT_MATCH
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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _mean(values: list[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    values = sorted(values)
    idx = max(0, min(len(values) - 1, int(round(0.95 * (len(values) - 1)))))
    return float(values[idx])


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _machine_target_from_trace(path: Path, outputs_dir: Path) -> tuple[str, str]:
    rel = path.relative_to(outputs_dir / "traces")
    return rel.parts[0], rel.parts[1]


def _variant_config(variant: str) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(f"Unsupported variant: {variant}")
    return VARIANTS[variant]


def _patch_event_trace_support() -> None:
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
        spec = _variant_config(variant)
        for key in (
            "fox_gd_residual_update_norm_cap",
            "fox_gd_residual_update_event_trace_enabled",
            "fox_gd_residual_update_event_trace_topk",
            "fox_gd_residual_update_event_trace_hypothetical_cap",
        ):
            BASEMOD.BASEMOD.BASE._set_flash_vqg_kwarg(config, key, spec.get(key))
        config.read_trace_train_steps = list(TRACE_STEPS)
        return config

    def flash_vqg_settings(config: Any) -> dict[str, Any]:
        settings = _ORIGINAL_FLASH_SETTINGS(config)
        for key in (
            "fox_gd_residual_update_norm_cap",
            "fox_gd_residual_update_event_trace_enabled",
            "fox_gd_residual_update_event_trace_topk",
            "fox_gd_residual_update_event_trace_hypothetical_cap",
        ):
            settings[key] = BASEMOD.BASEMOD._flash_setting(config, key)
        return settings

    def variant_settings_match(settings: dict[str, Any], variant: str) -> bool:
        if not _ORIGINAL_VARIANT_MATCH(settings, variant):
            return False
        spec = _variant_config(variant)
        for key in (
            "fox_gd_residual_update_norm_cap",
            "fox_gd_residual_update_event_trace_enabled",
            "fox_gd_residual_update_event_trace_topk",
            "fox_gd_residual_update_event_trace_hypothetical_cap",
        ):
            expected = spec.get(key)
            actual = settings.get(key)
            if isinstance(expected, bool):
                if bool(actual) != expected:
                    return False
            elif expected is None:
                if actual not in (None, ""):
                    return False
            else:
                if actual is None or abs(float(actual) - float(expected)) >= 1e-12:
                    return False
        return True

    BASEMOD.BASEMOD.BASE.build_config = build_config
    BASEMOD.BASEMOD.BASE._flash_vqg_settings = flash_vqg_settings
    BASEMOD.BASEMOD.BASE._variant_settings_match = variant_settings_match


def _collect_update_event_traces(outputs_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    raw_rows: list[dict[str, Any]] = []
    for path in sorted((outputs_dir / "traces").glob("**/train_step_*/update_event_trace.jsonl")):
        machine, target = _machine_target_from_trace(path.parents[1], outputs_dir)
        for record in _read_jsonl(path):
            row = dict(record)
            row["experiment_id"] = EXPERIMENT_ID
            row["machine"] = machine
            row["target"] = target
            row["trace_path"] = str(path)
            raw_rows.append(row)

    summary_rows: list[dict[str, Any]] = []
    step_groups: dict[tuple[str, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    layer_groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        target = str(row.get("target", ""))
        machine = str(row.get("machine", ""))
        step = int(row.get("global_step", -1))
        layer = int(row.get("layer_idx", -1))
        step_groups[(target, machine, step, layer)].append(row)
        layer_groups[(target, machine, layer)].append(row)

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        updates = [float(row["update_norm_uncapped"]) for row in rows if row.get("update_norm_uncapped") not in (None, "")]
        err_norms = [float(row["err_norm"]) for row in rows if row.get("err_norm") not in (None, "")]
        actual_hits = [1.0 if _boolish(row.get("actual_cap_hit")) else 0.0 for row in rows]
        hyp_hits = [1.0 if _boolish(row.get("hypothetical_cap_hit")) else 0.0 for row in rows]
        actual_scales = [float(row["actual_cap_scale"]) for row in rows if row.get("actual_cap_scale") not in (None, "")]
        hyp_scales = [float(row["hypothetical_cap_scale"]) for row in rows if row.get("hypothetical_cap_scale") not in (None, "")]
        return {
            "records": len(rows),
            "update_norm_max": max(updates) if updates else None,
            "update_norm_mean": _mean(updates),
            "update_norm_p95": _p95(updates),
            "err_norm_mean": _mean(err_norms),
            "err_norm_p95": _p95(err_norms),
            "actual_cap_hit_ratio": _mean(actual_hits),
            "hypothetical_cap_hit_ratio": _mean(hyp_hits),
            "actual_cap_scale_mean": _mean(actual_scales),
            "hypothetical_cap_scale_mean": _mean(hyp_scales),
        }

    for (target, machine, step, layer), rows in sorted(step_groups.items()):
        summary_rows.append(
            {
                "target": target,
                "machine": machine,
                "train_step": step,
                "layer_idx": layer,
                **summarize(rows),
            }
        )

    layer_summary_rows: list[dict[str, Any]] = []
    for (target, machine, layer), rows in sorted(layer_groups.items()):
        layer_summary_rows.append(
            {
                "target": target,
                "machine": machine,
                "layer_idx": layer,
                **summarize(rows),
            }
        )

    cross_rows: list[dict[str, Any]] = []
    by_key = {(r["target"], r["machine"], int(r["train_step"]), int(r["layer_idx"])): r for r in summary_rows}
    for target in TARGETS:
        for step in TRACE_STEPS:
            for layer in (0, 1):
                r2080 = by_key.get((target, "2080ti", step, layer))
                r3090 = by_key.get((target, "3090", step, layer))
                if not r2080 or not r3090:
                    continue
                max2080 = _float_or_none(r2080.get("update_norm_max"))
                max3090 = _float_or_none(r3090.get("update_norm_max"))
                hyp2080 = _float_or_none(r2080.get("hypothetical_cap_hit_ratio"))
                hyp3090 = _float_or_none(r3090.get("hypothetical_cap_hit_ratio"))
                cross_rows.append(
                    {
                        "target": target,
                        "train_step": step,
                        "layer_idx": layer,
                        "records_2080ti": r2080.get("records", ""),
                        "records_3090": r3090.get("records", ""),
                        "update_norm_max_2080ti": max2080,
                        "update_norm_max_3090": max3090,
                        "update_norm_max_abs_gap": (
                            None if max2080 is None or max3090 is None else abs(max2080 - max3090)
                        ),
                        "hypothetical_cap_hit_ratio_2080ti": hyp2080,
                        "hypothetical_cap_hit_ratio_3090": hyp3090,
                        "hypothetical_cap_hit_ratio_abs_gap": (
                            None if hyp2080 is None or hyp3090 is None else abs(hyp2080 - hyp3090)
                        ),
                    }
                )

    top_rows = sorted(
        raw_rows,
        key=lambda row: float(row.get("update_norm_uncapped") or 0.0),
        reverse=True,
    )[:512]
    return summary_rows, layer_summary_rows, cross_rows, top_rows


def run_collect(args: argparse.Namespace) -> int:
    code = _ORIGINAL_RUN_COLLECT(args)
    artifact_dir = args.artifact_dir
    event_rows, event_layer_rows, event_cross_rows, event_top_rows = _collect_update_event_traces(args.outputs_dir)
    _write_csv(artifact_dir / "update-event-step-summary.csv", event_rows)
    _write_csv(artifact_dir / "update-event-trace-summary.csv", event_layer_rows)
    _write_csv(artifact_dir / "update-event-cross-machine-summary.csv", event_cross_rows)
    _write_csv(artifact_dir / "update-event-top.csv", event_top_rows)

    readme = (
        f"# {EXPERIMENT_ID}\n\n"
        "本 artifact 收尾 default-dropout update_norm event trace diagnostic. "
        "本轮只比较 `baseline-r2` 与 `ucap0p5-r2`, 用事件级 trace 观察 top residual update, "
        "不把 hard cap 当作最终机制方案, 不写 official MQAR ledger.\n\n"
        "共同配置: `seed=124`, `data_seed=123`, `cb64-r16`, `read_topk=2`, "
        "`write_topk=4`, canonical MQAR cache, seed124 canonical init, "
        "`embed_dropout=0.1`, `resid_dropout=0.0`, `drop_path=0.0`, `max_train_steps=704`.\n\n"
        "注意: `update_event_trace.jsonl` 来自指定 optimizer step 上的 fixed validation batch eval forward, "
        "用于观察同一训练进度下的机制快照, 不是直接记录实际 training minibatch.\n\n"
        "核心文件:\n\n"
        "- `run-summary.csv`: per-run final/best metrics.\n"
        "- `variant-gap-summary.csv`: 2080ti/3090 final hard gap by variant.\n"
        "- `cap-metrics-summary.csv`: step704 cap hit, update norm, M norm, lambda/inject metrics.\n"
        "- `early-window-summary.csv`: train-step eval read/write scalar metrics.\n"
        "- `update-event-step-summary.csv`: per-step/layer top update event summary.\n"
        "- `update-event-trace-summary.csv`: per-machine/variant/layer event aggregate.\n"
        "- `update-event-cross-machine-summary.csv`: paired event aggregate comparison.\n"
        "- `update-event-top.csv`: global top 512 event rows.\n"
        "- `read-trace-cross-machine-summary.csv`: 2080ti/3090 read support match summary.\n"
        "- `hash-probe-comparison-summary.csv`: train-mode forward/backward hash comparison.\n"
        "- `cache-init-preflight-summary.csv`: cache/init hash evidence.\n"
        "- `source-manifest.csv`: mirrored lightweight raw evidence.\n"
    )
    (artifact_dir / "README.md").write_text(readme, encoding="utf-8")
    metadata_path = artifact_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    metadata.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "variants": VARIANTS,
            "trace_steps": TRACE_STEPS,
            "update_event_step_summary_rows": len(event_rows),
            "update_event_trace_summary_rows": len(event_layer_rows),
            "update_event_cross_machine_rows": len(event_cross_rows),
            "update_event_top_rows": len(event_top_rows),
            "trace_scope_note": "update event traces are fixed validation-batch eval forwards at selected optimizer steps, not actual training minibatch writes.",
            "diagnostic_note": "update_norm_cap and hypothetical cap are diagnostic controls; do not treat hard cap as final deployment fix.",
        }
    )
    _save_json(metadata_path, metadata)
    return code


_patch_event_trace_support()
BASEMOD.run_collect = run_collect
BASEMOD.BASEMOD.run_collect = run_collect


if __name__ == "__main__":
    raise SystemExit(BASEMOD.BASEMOD.main())
