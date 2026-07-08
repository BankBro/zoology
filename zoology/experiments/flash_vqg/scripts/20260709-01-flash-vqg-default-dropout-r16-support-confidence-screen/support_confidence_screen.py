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
SOURCE = (
    REPO_ROOT
    / "zoology/experiments/flash_vqg/scripts/"
    / "20260708-01-flash-vqg-r8-r16-fixed-init-three-seed-repeat/"
    / "fixed_init_three_seed_repeat.py"
)
EXPERIMENT_ID = "20260709-01-flash-vqg-default-dropout-r16-support-confidence-screen"
ARTIFACT_DIR = REPO_ROOT / "docs/artifacts" / EXPERIMENT_ID
METRICS_YAML = SCRIPT_DIR / "metrics.yaml"
SEEDS = (125, 124)
VARIANT_NAMES = (
    "baseline-r16-joint",
    "read-gate-r16",
    "read-softmargin-r16",
    "read-gate-softmargin-r16",
)
GRAD_ACCUMULATION_STEPS = 4
DEFAULT_MAX_TRAIN_STEPS = 704
PASS_HARD_ACCURACY = 0.85
PASS_GAP = 0.04


def _load_source():
    spec = importlib.util.spec_from_file_location("fixed_init_support_conf_base", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load source script: {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_source()
BASEMOD = BASE.BASEMOD
_ORIGINAL_BASE_RUN_COLLECT = BASE.run_collect


def _optimizer_to_train_forward_steps(optimizer_step: int) -> int:
    return int(optimizer_step) * GRAD_ACCUMULATION_STEPS


def _target(seed: int, variant_name: str) -> str:
    return f"s{seed}-{variant_name}"


TARGETS = tuple(_target(seed, variant_name) for seed in SEEDS for variant_name in VARIANT_NAMES)


def _common_variant(description: str, **updates: Any) -> dict[str, Any]:
    spec = BASE.BASE._variant(
        description=description,
        fox_remote_read_topk=16,
        fox_gd_residual_update_norm_softcap=0.5,
        fox_gd_residual_update_norm_softcap_mode="smooth_p4",
        fox_gd_residual_injection_warmup_start_train_steps=_optimizer_to_train_forward_steps(0),
        fox_gd_residual_injection_warmup_end_train_steps=_optimizer_to_train_forward_steps(512),
        fox_gd_residual_injection_warmup_eval_policy="final",
        warmup_start_optimizer_step=0,
        warmup_end_optimizer_step=512,
        run_kind="formal",
        d_geometry_trace_train_steps=[],
        fox_gd_residual_d_geometry_trace_enabled=False,
        fox_gd_residual_read_confidence_gate_mode="none",
        fox_gd_residual_read_confidence_margin_ref=0.5,
        fox_gd_residual_read_confidence_temp=0.25,
        fox_gd_residual_read_confidence_floor=0.25,
        fox_gd_residual_read_softmargin_mode="none",
        fox_gd_residual_read_softmargin_tau_max=3.0,
        fox_gd_residual_read_softmargin_margin_ref=0.5,
        fox_gd_residual_read_softmargin_temp=0.25,
    )
    spec.update(updates)
    return spec


VARIANT_SPECS: dict[str, dict[str, Any]] = {
    "baseline-r16-joint": _common_variant(
        "baseline: read_topk=16 with update softcap=0.5 and injection warmup 0->512 optimizer steps"
    ),
    "read-gate-r16": _common_variant(
        "read confidence gate: baseline plus margin_sigmoid residual injection gate",
        fox_gd_residual_read_confidence_gate_mode="margin_sigmoid",
    ),
    "read-softmargin-r16": _common_variant(
        "read softmargin: baseline plus topk_mass_temperature smoothing of residual read weights",
        fox_gd_residual_read_softmargin_mode="topk_mass_temperature",
    ),
    "read-gate-softmargin-r16": _common_variant(
        "combined read confidence: margin_sigmoid gate plus topk_mass_temperature softmargin",
        fox_gd_residual_read_confidence_gate_mode="margin_sigmoid",
        fox_gd_residual_read_softmargin_mode="topk_mass_temperature",
    ),
}


VARIANTS: dict[str, dict[str, Any]] = {}
for seed in SEEDS:
    for variant_name in VARIANT_NAMES:
        target = _target(seed, variant_name)
        spec = dict(VARIANT_SPECS[variant_name])
        spec.update(
            {
                "training_seed": seed,
                "data_seed": 123,
                "repeat": 1,
                "read_topk": 16,
                "fixed_init_seed": 124,
                "support_variant": variant_name,
            }
        )
        VARIANTS[target] = spec


_FLASH_EXTRA_KEYS = (
    "fox_gd_residual_read_confidence_gate_mode",
    "fox_gd_residual_read_confidence_margin_ref",
    "fox_gd_residual_read_confidence_temp",
    "fox_gd_residual_read_confidence_floor",
    "fox_gd_residual_read_softmargin_mode",
    "fox_gd_residual_read_softmargin_tau_max",
    "fox_gd_residual_read_softmargin_margin_ref",
    "fox_gd_residual_read_softmargin_temp",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:
        return None
    return out


def _int_or_none(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _target_meta(target: str) -> dict[str, Any]:
    spec = VARIANTS[target]
    return {
        "training_seed": int(spec["training_seed"]),
        "data_seed": int(spec["data_seed"]),
        "read_topk": int(spec["read_topk"]),
        "repeat": int(spec["repeat"]),
        "fixed_init_seed": int(spec["fixed_init_seed"]),
        "support_variant": str(spec["support_variant"]),
    }


def _compat_variants() -> dict[str, dict[str, Any]]:
    compat = dict(VARIANTS)
    compat["default-r2"] = BASE.BASE._variant(description="compatibility alias")
    compat["fixed-r2-baseline"] = BASE.BASE._variant(description="compatibility alias")
    return compat


def _patch_identity() -> None:
    compat_variants = _compat_variants()
    BASE.EXPERIMENT_ID = EXPERIMENT_ID
    BASE.ARTIFACT_DIR = ARTIFACT_DIR
    BASE.METRICS_YAML = METRICS_YAML
    BASE.FORMAL_TARGETS = TARGETS
    BASE.DGEOM_TARGETS = ()
    BASE.TARGETS = TARGETS
    BASE.VARIANTS = compat_variants
    BASE.DEFAULT_MAX_TRAIN_STEPS = DEFAULT_MAX_TRAIN_STEPS
    BASE.BASEMOD.SCRIPT_DIR = SCRIPT_DIR
    BASE.BASEMOD.EXPERIMENT_ID = EXPERIMENT_ID
    BASE.BASEMOD.ARTIFACT_DIR = ARTIFACT_DIR
    BASE.BASEMOD.METRICS_YAML = METRICS_YAML
    BASE.BASEMOD.TARGETS = TARGETS
    BASE.BASEMOD.VARIANTS = compat_variants
    BASE.BASEMOD.TRACE_TRAIN_STEPS = []
    BASE.BASEMOD.DEFAULT_CAPTURE_STEPS = ""
    BASE.BASEMOD.DEFAULT_MAX_TRAIN_STEPS = DEFAULT_MAX_TRAIN_STEPS
    BASE.BASEMOD.BASE.SCRIPT_DIR = SCRIPT_DIR
    BASE.BASEMOD.BASE.EXPERIMENT_ID = EXPERIMENT_ID
    BASE.BASEMOD.BASE.ARTIFACT_DIR = ARTIFACT_DIR
    BASE.BASEMOD.BASE.TARGETS = tuple(list(TARGETS) + ["default-r2", "fixed-r2-baseline"])
    BASE.BASEMOD.BASE.VARIANTS = compat_variants
    BASE.BASEMOD.BASE.METRICS_YAML = METRICS_YAML


def _patch_support() -> None:
    compat_variants = _compat_variants()
    BASE.VARIANTS = compat_variants
    BASE.TARGETS = TARGETS
    BASE.FORMAL_TARGETS = TARGETS
    BASE.DGEOM_TARGETS = ()
    BASE._patch_support()

    original_build_config = BASE.BASEMOD.BASE.build_config
    original_flash_settings = BASE.BASEMOD.BASE._flash_vqg_settings

    def build_config(*args: Any, **kwargs: Any):
        variant = kwargs.get("variant")
        if variant is None and len(args) >= 3:
            variant = args[2]
        variant = str(variant)
        config = original_build_config(*args, **kwargs)
        if variant in VARIANTS:
            spec = VARIANTS[variant]
            seed = int(spec["training_seed"])
            data_seed = int(spec["data_seed"])
            variant_name = str(spec["support_variant"])
            machine_name = kwargs.get("machine_name", "unknown")
            config.seed = seed
            config.data.seed = data_seed
            config.run_id = f"{EXPERIMENT_ID}-s{seed}-{variant_name}-fixedinit-s124-d{data_seed}-b64ga4-{machine_name}"
            config.launch_id = f"fvqg-{EXPERIMENT_ID}-{machine_name}-s{seed}-{variant_name}"
            for key in _FLASH_EXTRA_KEYS:
                BASE.BASEMOD.BASE._set_flash_vqg_kwarg(config, key, spec.get(key))
        return config

    def flash_vqg_settings(config: Any) -> dict[str, Any]:
        settings = original_flash_settings(config)
        for key in _FLASH_EXTRA_KEYS:
            settings[key] = BASE.BASEMOD._flash_setting(config, key)
        return settings

    BASE.BASEMOD.BASE.build_config = build_config
    BASE.BASEMOD.BASE._flash_vqg_settings = flash_vqg_settings
    BASE.BASEMOD.BASE._variant_settings_match = _variant_settings_match


_patch_limiter_support = _patch_support


def _variant_settings_match(settings: dict[str, Any], variant: str) -> bool:
    if variant not in VARIANTS:
        return True
    spec = VARIANTS[variant]
    checks: dict[str, Any] = {
        "num_codebook_vectors": 64,
        "fox_remote_read_topk": 16,
        "fox_gd_residual_write_topk": 4,
        "fox_gd_residual_update_norm_softcap": 0.5,
        "fox_gd_residual_update_norm_softcap_mode": "smooth_p4",
        "fox_gd_residual_injection_warmup_start_train_steps": _optimizer_to_train_forward_steps(0),
        "fox_gd_residual_injection_warmup_end_train_steps": _optimizer_to_train_forward_steps(512),
        "fox_gd_residual_injection_warmup_eval_policy": "final",
    }
    for key in _FLASH_EXTRA_KEYS:
        checks[key] = spec.get(key)
    for key, expected in checks.items():
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


def _add_target_meta(row: dict[str, Any]) -> dict[str, Any]:
    target = str(row.get("target") or row.get("variant") or "")
    if target in VARIANTS:
        row.update(_target_meta(target))
    return row


def _summarize_seed_variant(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for variant_name in VARIANT_NAMES:
            target = _target(seed, variant_name)
            group = {
                str(row.get("machine")): row
                for row in run_rows
                if row.get("status") == "completed" and row.get("target") == target
            }
            r2080 = group.get("2080ti", {})
            r3090 = group.get("3090", {})
            f2080 = _float_or_none(r2080.get("final_1024x256_accuracy"))
            f3090 = _float_or_none(r3090.get("final_1024x256_accuracy"))
            acc2080 = _float_or_none(r2080.get("final_valid_accuracy"))
            acc3090 = _float_or_none(r3090.get("final_valid_accuracy"))
            loss2080 = _float_or_none(r2080.get("final_valid_loss"))
            loss3090 = _float_or_none(r3090.get("final_valid_loss"))
            gap = abs(f2080 - f3090) if f2080 is not None and f3090 is not None else None
            rows.append(
                {
                    "training_seed": seed,
                    "support_variant": variant_name,
                    "target": target,
                    "completed_pair": f2080 is not None and f3090 is not None,
                    "final_1024x256_2080ti": f2080,
                    "final_1024x256_3090": f3090,
                    "gap": gap,
                    "gap_percentage_points": None if gap is None else gap * 100.0,
                    "final_accuracy_2080ti": acc2080,
                    "final_accuracy_3090": acc3090,
                    "loss_2080ti": loss2080,
                    "loss_3090": loss3090,
                    "passes_pair_screen": (
                        f2080 is not None
                        and f3090 is not None
                        and f2080 >= PASS_HARD_ACCURACY
                        and f3090 >= PASS_HARD_ACCURACY
                        and gap is not None
                        and gap <= PASS_GAP
                    ),
                }
            )
    return rows


def _summarize_variant_across_seeds(pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant_name in VARIANT_NAMES:
        subset = [row for row in pair_rows if row.get("support_variant") == variant_name]
        passed = sum(str(row.get("passes_pair_screen")) == "True" for row in subset)
        gaps = [_float_or_none(row.get("gap_percentage_points")) for row in subset]
        gaps = [gap for gap in gaps if gap is not None]
        values = []
        for row in subset:
            for key in ("final_1024x256_2080ti", "final_1024x256_3090"):
                value = _float_or_none(row.get(key))
                if value is not None:
                    values.append(value)
        rows.append(
            {
                "support_variant": variant_name,
                "completed_pairs": sum(str(row.get("completed_pair")) == "True" for row in subset),
                "expected_pairs": len(SEEDS),
                "passed_pairs": passed,
                "mean_gap_percentage_points": None if not gaps else sum(gaps) / len(gaps),
                "max_gap_percentage_points": None if not gaps else max(gaps),
                "min_1024x256": None if not values else min(values),
                "max_1024x256": None if not values else max(values),
                "all_values_above_0p85": len(values) == len(SEEDS) * 2 and all(value >= PASS_HARD_ACCURACY for value in values),
                "passes_two_seed_screen": passed == len(SEEDS),
            }
        )
    return rows


def _write_formal_ledger(artifact_dir: Path, run_rows: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for row in run_rows:
        rows.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "machine": row.get("machine", ""),
                "gpu": row.get("gpu", ""),
                "target": row.get("target", ""),
                "support_variant": row.get("support_variant", ""),
                "training_seed": row.get("training_seed", ""),
                "status": row.get("status", ""),
                "started_at": row.get("started_at", ""),
                "finished_at": row.get("finished_at", ""),
                "duration_minutes": row.get("duration_minutes", ""),
                "dtype_policy": "torch.float32 matmul precision highest; TF32 matmul disabled",
                "log_path": row.get("log_path", ""),
                "result_json": row.get("result_json", ""),
                "config_json": row.get("config_json", ""),
            }
        )
    _write_csv(artifact_dir / "formal-ledger.csv", rows)


def _write_master_summary(artifact_dir: Path, outputs_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for path in sorted(outputs_dir.glob("*/master-status.tsv")):
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                row = dict(row)
                if row.get("mode") != "formal":
                    continue
                row["status_path"] = str(path)
                rows.append(row)
    _write_csv(artifact_dir / "master-status-summary.csv", rows)


def _is_formal_row(row: dict[str, Any]) -> bool:
    haystack = " ".join(
        str(row.get(key, ""))
        for key in ("run_id", "launch_id", "log_path", "result_json", "config_json")
    )
    return "-formal" in haystack


def _post_collect(args: Any, code: int) -> int:
    artifact_dir = args.artifact_dir if args.artifact_dir.is_absolute() else (SCRIPT_DIR / args.artifact_dir)
    outputs_dir = args.outputs_dir if args.outputs_dir.is_absolute() else (SCRIPT_DIR / args.outputs_dir)
    run_rows = []
    for raw_row in _read_csv(artifact_dir / "run-summary.csv"):
        row = _add_target_meta(dict(raw_row))
        if not _is_formal_row(row):
            continue
        row["experiment_id"] = EXPERIMENT_ID
        run_rows.append(row)
    pair_rows = _summarize_seed_variant(run_rows)
    _write_csv(artifact_dir / "run-summary.csv", run_rows)
    _write_csv(artifact_dir / "cross-machine-comparison.csv", pair_rows)
    _write_csv(artifact_dir / "variant-seed-summary.csv", pair_rows)
    _write_csv(artifact_dir / "variant-summary.csv", _summarize_variant_across_seeds(pair_rows))
    _write_formal_ledger(artifact_dir, run_rows)
    _write_master_summary(artifact_dir, outputs_dir)

    readme = (
        f"# {EXPERIMENT_ID}\n\n"
        "Formal-only support-confidence screen for default-dropout cb64-r16. "
        "All runs use the same seed124 canonical init checkpoint; `s124/s125` "
        "refer to training RNG seed only. Heavy traces are disabled.\n\n"
        "Core files:\n\n"
        "- `run-summary.csv`: per-run status and final/best metrics.\n"
        "- `cross-machine-comparison.csv`: seed/variant aligned 2080ti vs 3090 comparison.\n"
        "- `variant-summary.csv`: two-seed pass/fail summary by support-confidence variant.\n"
        "- `mechanism-metrics-summary.csv`: residual read/write/state scalar metrics.\n"
        "- `formal-ledger.csv`: formal MQAR ledger.\n"
    )
    (artifact_dir / "README.md").write_text(readme, encoding="utf-8")

    metadata_path = artifact_dir / "metadata.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    payload.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "targets": TARGETS,
            "training_seeds": SEEDS,
            "support_variants": VARIANT_NAMES,
            "read_topk": 16,
            "fixed_init_seed": 124,
            "data_seed": 123,
            "fixed_init_checkpoint": (
                "zoology/experiments/flash_vqg/scripts/"
                "20260630-03-flash-vqg-s124-fixed-r4-4ep-confirm/"
                "outputs/canonical-init/cb64r16-s124-init.pt"
            ),
            "screen_rule": "each seed/variant pair passes when both machines final 1024x256 >= 0.85 and gap <= 0.04",
            "trace_policy": "formal heavy read_trace, hash_probe, train inline event trace, and D-geometry trace disabled",
        }
    )
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return code


def run_collect(args: Any) -> int:
    code = _ORIGINAL_BASE_RUN_COLLECT(args)
    return _post_collect(args, code)


def main() -> int:
    _patch_identity()
    _patch_support()
    BASE.BASEMOD.run_train = BASE.BASE.run_train
    BASE.BASEMOD.run_collect = run_collect
    BASE.run_collect = run_collect
    os.environ["FLASH_VQG_READ_TRACE_MODE"] = "disabled"
    return BASE.BASEMOD.main()


if __name__ == "__main__":
    raise SystemExit(main())
