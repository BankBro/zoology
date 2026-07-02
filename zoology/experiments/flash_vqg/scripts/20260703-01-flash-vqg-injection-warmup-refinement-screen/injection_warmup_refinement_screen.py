#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
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
    / "20260701-04-flash-vqg-default-dropout-update-norm-cap-probe/update_norm_cap_probe.py"
)
EXPERIMENT_ID = "20260703-01-flash-vqg-injection-warmup-refinement-screen"
ARTIFACT_DIR = REPO_ROOT / "docs/artifacts" / EXPERIMENT_ID
METRICS_YAML = SCRIPT_DIR / "metrics.yaml"
TRACE_STEPS = [0, 1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 704]
GRAD_ACCUMULATION_STEPS = 4


def _load_source():
    spec = importlib.util.spec_from_file_location("injection_warmup_base", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load source script: {SOURCE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASEMOD = _load_source()

TARGETS = (
    "inj-warmup-linear704-r2",
    "inj-warmup-linear1024-r2",
    "inj-warmup-silent32-linear704-r2",
)

_R2_BASE = {
    **BASEMOD._R2_BASE,
    "fox_gd_residual_update_norm_cap": None,
    "fox_gd_residual_update_event_trace_enabled": False,
    "fox_gd_residual_update_event_trace_topk": 64,
    "fox_gd_residual_update_event_trace_hypothetical_cap": None,
    "fox_gd_residual_injection_warmup_start_train_steps": 0,
    "fox_gd_residual_injection_warmup_end_train_steps": 0,
    "fox_gd_residual_injection_warmup_eval_policy": "scheduled",
}


def _optimizer_to_train_forward_steps(optimizer_step: int) -> int:
    return int(optimizer_step) * GRAD_ACCUMULATION_STEPS


VARIANTS: dict[str, dict[str, Any]] = {
    "inj-warmup-linear704-r2": {
        **_R2_BASE,
        "description": "default-dropout fixed read_topk=2, residual injection linearly warms from optimizer step 0 to 704",
        "warmup_start_optimizer_step": 0,
        "warmup_end_optimizer_step": 704,
        "fox_gd_residual_injection_warmup_start_train_steps": _optimizer_to_train_forward_steps(0),
        "fox_gd_residual_injection_warmup_end_train_steps": _optimizer_to_train_forward_steps(704),
    },
    "inj-warmup-linear1024-r2": {
        **_R2_BASE,
        "description": "default-dropout fixed read_topk=2, residual injection linearly warms from optimizer step 0 to 1024",
        "warmup_start_optimizer_step": 0,
        "warmup_end_optimizer_step": 1024,
        "fox_gd_residual_injection_warmup_start_train_steps": _optimizer_to_train_forward_steps(0),
        "fox_gd_residual_injection_warmup_end_train_steps": _optimizer_to_train_forward_steps(1024),
    },
    "inj-warmup-silent32-linear704-r2": {
        **_R2_BASE,
        "description": "default-dropout fixed read_topk=2, residual injection silent to optimizer step 32 then linearly warms to 704",
        "warmup_start_optimizer_step": 32,
        "warmup_end_optimizer_step": 704,
        "fox_gd_residual_injection_warmup_start_train_steps": _optimizer_to_train_forward_steps(32),
        "fox_gd_residual_injection_warmup_end_train_steps": _optimizer_to_train_forward_steps(704),
    },
}


BASEMOD.SCRIPT_DIR = SCRIPT_DIR
BASEMOD.EXPERIMENT_ID = EXPERIMENT_ID
BASEMOD.ARTIFACT_DIR = ARTIFACT_DIR
BASEMOD.TRACE_STEPS = list(TRACE_STEPS)
BASEMOD.BASEMOD.SCRIPT_DIR = SCRIPT_DIR
BASEMOD.BASEMOD.EXPERIMENT_ID = EXPERIMENT_ID
BASEMOD.BASEMOD.ARTIFACT_DIR = ARTIFACT_DIR
BASEMOD.BASEMOD.METRICS_YAML = METRICS_YAML
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
_ORIGINAL_RUN_PREFLIGHT = BASEMOD.BASEMOD.run_preflight
_ORIGINAL_RUN_COLLECT = BASEMOD._ORIGINAL_RUN_COLLECT


def _variant_config(variant: str) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(f"Unsupported variant: {variant}")
    return VARIANTS[variant]


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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _batch_order_hash(dataloader: Any) -> dict[str, Any]:
    sampler = getattr(dataloader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(0)
    order = list(iter(sampler)) if sampler is not None else list(range(len(dataloader)))
    hasher = hashlib.sha256()
    for item in order:
        hasher.update(int(item).to_bytes(8, "little", signed=True))
    return {
        "num_batches": len(order),
        "sha256": hasher.hexdigest(),
        "first_16": order[:16],
    }


def _patch_injection_warmup_support() -> None:
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
            "fox_gd_residual_injection_warmup_start_train_steps",
            "fox_gd_residual_injection_warmup_end_train_steps",
            "fox_gd_residual_injection_warmup_eval_policy",
        ):
            BASEMOD.BASEMOD.BASE._set_flash_vqg_kwarg(config, key, spec.get(key))
        config.read_trace_train_steps = list(TRACE_STEPS)
        config.train_inline_event_trace_enabled = False
        config.train_inline_event_trace_steps = []
        config.train_inline_event_trace_output_dir = None
        return config

    def flash_vqg_settings(config: Any) -> dict[str, Any]:
        settings = _ORIGINAL_FLASH_SETTINGS(config)
        for key in (
            "fox_gd_residual_update_norm_cap",
            "fox_gd_residual_injection_warmup_start_train_steps",
            "fox_gd_residual_injection_warmup_end_train_steps",
            "fox_gd_residual_injection_warmup_eval_policy",
        ):
            settings[key] = BASEMOD.BASEMOD._flash_setting(config, key)
        return settings

    def variant_settings_match(settings: dict[str, Any], variant: str) -> bool:
        if not _ORIGINAL_VARIANT_MATCH(settings, variant):
            return False
        spec = _variant_config(variant)
        for key in (
            "fox_gd_residual_update_norm_cap",
            "fox_gd_residual_injection_warmup_start_train_steps",
            "fox_gd_residual_injection_warmup_end_train_steps",
            "fox_gd_residual_injection_warmup_eval_policy",
        ):
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

    BASEMOD.BASEMOD.BASE.build_config = build_config
    BASEMOD.BASEMOD.BASE._flash_vqg_settings = flash_vqg_settings
    BASEMOD.BASEMOD.BASE._variant_settings_match = variant_settings_match


def run_preflight(args: argparse.Namespace) -> int:
    code = _ORIGINAL_RUN_PREFLIGHT(args)
    if args.output_json is None:
        return code
    path = Path(args.output_json)
    payload = _read_json(path)
    config = BASEMOD.BASEMOD.BASE.build_config(
        target=args.target,
        machine_name=args.machine_name,
        variant=args.variant,
        logger_backend="none",
        trace_output_dir=SCRIPT_DIR / "outputs/preflight-traces" / args.machine_name / args.target,
        max_epochs=args.max_epochs,
        max_train_steps=args.max_train_steps,
        max_validation_batches=args.max_validation_batches,
    )
    BASEMOD.BASEMOD._apply_run_suffix(config, args.run_suffix)
    train_loader, _ = BASEMOD.BASEMOD.BASE.prepare_data(config.data)
    batch_order = _batch_order_hash(train_loader)
    payload["batch_order"] = batch_order
    payload["batch_order_sha256"] = batch_order["sha256"]
    payload["batch_order_first_16"] = batch_order["first_16"]
    payload["batch_order_num_batches"] = batch_order["num_batches"]
    _save_json(path, payload)
    return code


def _machine_target_from_trace(path: Path, outputs_dir: Path) -> tuple[str, str]:
    rel = path.relative_to(outputs_dir / "traces")
    return rel.parts[0], rel.parts[1]


def _collect_injection_warmup_summary(outputs_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((outputs_dir / "traces").glob("**/early_window_metrics.jsonl")):
        machine, target = _machine_target_from_trace(path, outputs_dir)
        spec = _variant_config(target)
        for record in _read_jsonl(path):
            step = record.get("train_step", "")
            layer0 = _float_or_none(
                record.get("early_window/layer_0/attn/gd_residual_injection_warmup_factor")
            )
            layer1 = _float_or_none(
                record.get("early_window/layer_1/attn/gd_residual_injection_warmup_factor")
            )
            factor = _float_or_none(record.get("early_window/attn/gd_residual_injection_warmup_factor"))
            rows.append(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "machine": machine,
                    "target": target,
                    "train_step": step,
                    "warmup_start_optimizer_step": spec.get("warmup_start_optimizer_step", ""),
                    "warmup_end_optimizer_step": spec.get("warmup_end_optimizer_step", ""),
                    "warmup_start_train_forward_step": spec.get(
                        "fox_gd_residual_injection_warmup_start_train_steps", ""
                    ),
                    "warmup_end_train_forward_step": spec.get(
                        "fox_gd_residual_injection_warmup_end_train_steps", ""
                    ),
                    "warmup_eval_policy": spec.get(
                        "fox_gd_residual_injection_warmup_eval_policy", ""
                    ),
                    "factor_mean": factor,
                    "factor_layer0": layer0,
                    "factor_layer1": layer1,
                    "inject_ratio_mean": record.get("early_window/attn/gd_residual_inject_ratio", ""),
                    "inject_ratio_layer0": record.get("early_window/layer_0/attn/gd_residual_inject_ratio", ""),
                    "inject_ratio_layer1": record.get("early_window/layer_1/attn/gd_residual_inject_ratio", ""),
                    "lambda_mean": record.get("early_window/attn/gd_residual_lambda_mean", ""),
                    "loss": record.get("loss", ""),
                }
            )
    return rows


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
        b2080 = _float_or_none(r2080.get("best_1024x256_accuracy"))
        b3090 = _float_or_none(r3090.get("best_1024x256_accuracy"))
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
                "best_1024x256_2080ti": b2080,
                "best_1024x256_3090": b3090,
                "read_topk": spec.get("fox_remote_read_topk"),
                "warmup_start_optimizer_step": spec.get("warmup_start_optimizer_step", ""),
                "warmup_end_optimizer_step": spec.get("warmup_end_optimizer_step", ""),
                "warmup_start_train_forward_step": spec.get(
                    "fox_gd_residual_injection_warmup_start_train_steps", ""
                ),
                "warmup_end_train_forward_step": spec.get(
                    "fox_gd_residual_injection_warmup_end_train_steps", ""
                ),
            }
        )
    return rows


def run_collect(args: argparse.Namespace) -> int:
    code = _ORIGINAL_RUN_COLLECT(args)
    artifact_dir = args.artifact_dir
    warmup_rows = _collect_injection_warmup_summary(args.outputs_dir)
    run_rows = _read_csv(artifact_dir / "run-summary.csv")
    gap_rows = _variant_gap_rows(run_rows)
    _write_csv(artifact_dir / "injection-warmup-summary.csv", warmup_rows)
    _write_csv(artifact_dir / "variant-gap-summary.csv", gap_rows)

    readme = (
        f"# {EXPERIMENT_ID}\n\n"
        "本 artifact 收尾 default-dropout residual injection warmup refinement 1ep screen. "
        "本轮只控制 residual correction 注入到 `O_base` 的强度, 不改变 `M_state` build/write/read, "
        "不改变 dropout 协议, 不写 official MQAR ledger.\n\n"
        "共同配置: `seed=124`, `data_seed=123`, `cb64-r16`, `read_topk=2`, "
        "`write_topk=4`, canonical MQAR cache, seed124 canonical init, "
        "`embed_dropout=0.1`, `resid_dropout=0.0`, `drop_path=0.0`, "
        "`max_train_steps=704` optimizer steps.\n\n"
        "Warmup step 说明: Flash-VQG 内部使用 train-forward counter. "
        "本轮 `gradient_accumulation_steps=4`, 所以 optimizer step 704 对应 train-forward step 2816, "
        "optimizer step 1024 对应 train-forward step 4096, optimizer step 32 对应 train-forward step 128.\n\n"
        "核心文件:\n\n"
        "- `run-summary.csv`: per-run final/best metrics.\n"
        "- `variant-gap-summary.csv`: 2080ti/3090 final hard gap by variant.\n"
        "- `injection-warmup-summary.csv`: warmup factor, inject ratio, lambda and loss by train step.\n"
        "- `early-window-summary.csv`: train-step eval read/write scalar metrics.\n"
        "- `read-trace-cross-machine-summary.csv`: 2080ti/3090 read support match summary.\n"
        "- `hash-probe-comparison-summary.csv`: train-mode forward/backward hash comparison.\n"
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
            "trace_steps": TRACE_STEPS,
            "gradient_accumulation_steps": GRAD_ACCUMULATION_STEPS,
            "optimizer_to_train_forward_mapping": "train_forward_step = optimizer_step * gradient_accumulation_steps",
            "injection_warmup_summary_rows": len(warmup_rows),
            "variant_gap_rows": len(gap_rows),
            "diagnostic_note": "residual injection warmup controls O_res_added only; M_state build/write/read semantics are unchanged.",
        }
    )
    _save_json(metadata_path, metadata)
    return code


_patch_injection_warmup_support()
BASEMOD.run_preflight = run_preflight
BASEMOD.BASEMOD.run_preflight = run_preflight
BASEMOD.run_collect = run_collect
BASEMOD.BASEMOD.run_collect = run_collect


if __name__ == "__main__":
    raise SystemExit(BASEMOD.BASEMOD.main())
