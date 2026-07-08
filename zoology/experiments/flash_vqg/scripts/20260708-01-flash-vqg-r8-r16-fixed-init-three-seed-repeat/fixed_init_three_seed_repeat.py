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
    / "20260707-01-flash-vqg-r8-r16-joint-repro/"
    / "r8_r16_joint_repro.py"
)
EXPERIMENT_ID = "20260708-01-flash-vqg-r8-r16-fixed-init-three-seed-repeat"
ARTIFACT_DIR = REPO_ROOT / "docs/artifacts" / EXPERIMENT_ID
METRICS_YAML = SCRIPT_DIR / "metrics.yaml"
SEEDS = (123, 124, 125)
READ_TOPKS = (8, 16)
REPEATS = (1, 2)
GRAD_ACCUMULATION_STEPS = 4
DEFAULT_MAX_TRAIN_STEPS = 704
PASS_HARD_ACCURACY = 0.85
PASS_GAP = 0.04


def _load_source():
    spec = importlib.util.spec_from_file_location("r8_r16_repro_base", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load source script: {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_source()
BASEMOD = BASE.BASEMOD
_ORIGINAL_BASE_RUN_COLLECT = BASE.run_collect


def _target(seed: int, read_topk: int, repeat: int) -> str:
    return f"s{seed}-r{read_topk}-rep{repeat}"


TARGETS = tuple(
    _target(seed, read_topk, repeat)
    for seed in SEEDS
    for read_topk in READ_TOPKS
    for repeat in REPEATS
)


def _optimizer_to_train_forward_steps(optimizer_step: int) -> int:
    return int(optimizer_step) * GRAD_ACCUMULATION_STEPS


def _variant(seed: int, read_topk: int, repeat: int) -> dict[str, Any]:
    spec = BASE._variant(
        description=(
            f"fixed seed124 init; training_seed={seed}; read_topk={read_topk}; "
            f"repeat={repeat}; update softcap=0.5; injection warmup 0->512 optimizer steps"
        ),
        fox_remote_read_topk=read_topk,
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
    )
    spec.update(
        {
            "training_seed": seed,
            "data_seed": 123,
            "repeat": repeat,
            "read_topk": read_topk,
            "fixed_init_seed": 124,
        }
    )
    return spec


VARIANTS: dict[str, dict[str, Any]] = {
    _target(seed, read_topk, repeat): _variant(seed, read_topk, repeat)
    for seed in SEEDS
    for read_topk in READ_TOPKS
    for repeat in REPEATS
}


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


def _target_meta(target: str) -> dict[str, int]:
    spec = VARIANTS[target]
    return {
        "training_seed": int(spec["training_seed"]),
        "data_seed": int(spec["data_seed"]),
        "read_topk": int(spec["read_topk"]),
        "repeat": int(spec["repeat"]),
        "fixed_init_seed": int(spec["fixed_init_seed"]),
    }


def _patch_identity() -> None:
    compat_variants = dict(VARIANTS)
    compat_variants["default-r2"] = BASE._variant(description="compatibility alias")
    compat_variants["fixed-r2-baseline"] = BASE._variant(description="compatibility alias")
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
    compat_variants = dict(VARIANTS)
    compat_variants["default-r2"] = BASE._variant(description="compatibility alias")
    compat_variants["fixed-r2-baseline"] = BASE._variant(description="compatibility alias")
    BASE.VARIANTS = compat_variants
    BASE.TARGETS = TARGETS
    BASE.FORMAL_TARGETS = TARGETS
    BASE.DGEOM_TARGETS = ()
    BASE._patch_support()

    original_build_config = BASE.BASEMOD.BASE.build_config

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
            read_topk = int(spec["read_topk"])
            repeat = int(spec["repeat"])
            machine_name = kwargs.get("machine_name", "unknown")
            config.seed = seed
            config.data.seed = data_seed
            config.run_id = (
                f"{EXPERIMENT_ID}-s{seed}-r{read_topk}-rep{repeat}"
                f"-fixedinit-s124-d{data_seed}-b64ga4-{machine_name}"
            )
            config.launch_id = (
                f"fvqg-{EXPERIMENT_ID}-{machine_name}-"
                f"s{seed}-r{read_topk}-rep{repeat}"
            )
        return config

    BASE.BASEMOD.BASE.build_config = build_config
    BASE.BASEMOD.BASE._variant_settings_match = _variant_settings_match


_patch_limiter_support = _patch_support


def _variant_settings_match(settings: dict[str, Any], variant: str) -> bool:
    if variant not in VARIANTS:
        return True
    spec = VARIANTS[variant]
    checks = {
        "num_codebook_vectors": 64,
        "fox_remote_read_topk": int(spec["fox_remote_read_topk"]),
        "fox_gd_residual_write_topk": 4,
        "fox_gd_residual_update_norm_softcap": 0.5,
    }
    for key, expected in checks.items():
        actual = settings.get(key)
        if actual is None or abs(float(actual) - float(expected)) >= 1e-12:
            return False
    return str(settings.get("fox_gd_residual_update_norm_softcap_mode")) == "smooth_p4"


def _add_target_meta(row: dict[str, Any]) -> dict[str, Any]:
    target = str(row.get("target") or row.get("variant") or "")
    if target in VARIANTS:
        row.update(_target_meta(target))
    return row


def _summarize_variant_seed_repeat(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for read_topk in READ_TOPKS:
            subset = [
                row for row in run_rows
                if row.get("status") == "completed"
                and _int_or_none(row.get("training_seed")) == seed
                and _int_or_none(row.get("read_topk")) == read_topk
            ]
            values = [
                _float_or_none(row.get("final_1024x256_accuracy"))
                for row in subset
                if _float_or_none(row.get("final_1024x256_accuracy")) is not None
            ]
            min_value = min(values) if values else None
            max_value = max(values) if values else None
            gap = None if min_value is None or max_value is None else max_value - min_value
            rows.append(
                {
                    "training_seed": seed,
                    "read_topk": read_topk,
                    "completed_runs": len(subset),
                    "expected_runs": 4,
                    "values": ",".join(f"{value:.6f}" for value in values),
                    "min_1024x256": min_value,
                    "max_1024x256": max_value,
                    "gap": gap,
                    "gap_percentage_points": None if gap is None else gap * 100.0,
                    "all_above_0p85": len(values) == 4 and all(value >= PASS_HARD_ACCURACY for value in values),
                    "passes_screen": len(values) == 4 and all(value >= PASS_HARD_ACCURACY for value in values) and gap is not None and gap <= PASS_GAP,
                }
            )
    return rows


def _summarize_cross_machine(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for read_topk in READ_TOPKS:
            for repeat in REPEATS:
                group = {
                    str(row.get("machine")): row
                    for row in run_rows
                    if row.get("status") == "completed"
                    and _int_or_none(row.get("training_seed")) == seed
                    and _int_or_none(row.get("read_topk")) == read_topk
                    and _int_or_none(row.get("repeat")) == repeat
                }
                r2080 = group.get("2080ti", {})
                r3090 = group.get("3090", {})
                f2080 = _float_or_none(r2080.get("final_1024x256_accuracy"))
                f3090 = _float_or_none(r3090.get("final_1024x256_accuracy"))
                gap = abs(f2080 - f3090) if f2080 is not None and f3090 is not None else None
                rows.append(
                    {
                        "training_seed": seed,
                        "read_topk": read_topk,
                        "repeat": repeat,
                        "completed_pair": f2080 is not None and f3090 is not None,
                        "final_1024x256_2080ti": f2080,
                        "final_1024x256_3090": f3090,
                        "gap": gap,
                        "gap_percentage_points": None if gap is None else gap * 100.0,
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


def _summarize_within_machine(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for read_topk in READ_TOPKS:
            for machine in ("2080ti", "3090"):
                subset = [
                    row for row in run_rows
                    if row.get("status") == "completed"
                    and row.get("machine") == machine
                    and _int_or_none(row.get("training_seed")) == seed
                    and _int_or_none(row.get("read_topk")) == read_topk
                ]
                values = [
                    _float_or_none(row.get("final_1024x256_accuracy"))
                    for row in subset
                    if _float_or_none(row.get("final_1024x256_accuracy")) is not None
                ]
                spread = (max(values) - min(values)) if len(values) >= 2 else None
                rows.append(
                    {
                        "training_seed": seed,
                        "read_topk": read_topk,
                        "machine": machine,
                        "completed_repeats": len(values),
                        "values": ",".join(f"{value:.6f}" for value in values),
                        "repeat_spread": spread,
                        "repeat_spread_percentage_points": None if spread is None else spread * 100.0,
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
                "training_seed": row.get("training_seed", ""),
                "read_topk": row.get("read_topk", ""),
                "repeat": row.get("repeat", ""),
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
    _write_csv(artifact_dir / "run-summary.csv", run_rows)
    _write_csv(artifact_dir / "variant-seed-repeat-summary.csv", _summarize_variant_seed_repeat(run_rows))
    _write_csv(artifact_dir / "cross-machine-comparison.csv", _summarize_cross_machine(run_rows))
    _write_csv(artifact_dir / "within-machine-repeat-summary.csv", _summarize_within_machine(run_rows))
    _write_formal_ledger(artifact_dir, run_rows)
    _write_master_summary(artifact_dir, outputs_dir)

    readme = (
        f"# {EXPERIMENT_ID}\n\n"
        "Fixed-init training-seed repeat for R8/R16 joint control. "
        "All runs use the same seed124 canonical init checkpoint; `s123/s124/s125` "
        "refer to training RNG seed only, not model initialization seed.\n\n"
        "Core files:\n\n"
        "- `run-summary.csv`: per-run status and final/best metrics.\n"
        "- `variant-seed-repeat-summary.csv`: four-run summary for each seed/read_topk.\n"
        "- `cross-machine-comparison.csv`: repeat-aligned 2080ti vs 3090 comparisons.\n"
        "- `within-machine-repeat-summary.csv`: same-machine repeat spread.\n"
        "- `mechanism-metrics-summary.csv`: residual read/write/state metrics.\n"
        "- `formal-ledger.csv`: formal MQAR ledger for completed/failed formal runs.\n"
    )
    (artifact_dir / "README.md").write_text(readme, encoding="utf-8")

    metadata_path = artifact_dir / "metadata.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    payload.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "targets": TARGETS,
            "training_seeds": SEEDS,
            "read_topks": READ_TOPKS,
            "repeats": REPEATS,
            "fixed_init_seed": 124,
            "data_seed": 123,
            "fixed_init_checkpoint": (
                "zoology/experiments/flash_vqg/scripts/"
                "20260630-03-flash-vqg-s124-fixed-r4-4ep-confirm/"
                "outputs/canonical-init/cb64r16-s124-init.pt"
            ),
            "stability_rule": "4 final 1024x256 values >= 0.85 and max-min gap <= 0.04 for each seed/read_topk",
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
