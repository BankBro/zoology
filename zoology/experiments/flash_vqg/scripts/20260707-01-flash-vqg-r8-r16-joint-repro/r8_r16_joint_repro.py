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
    / "20260706-02-flash-vqg-default-dropout-joint-control-dgeom/"
    / "joint_control_dgeom.py"
)
EXPERIMENT_ID = "20260707-01-flash-vqg-r8-r16-joint-repro"
ARTIFACT_DIR = REPO_ROOT / "docs/artifacts" / EXPERIMENT_ID
METRICS_YAML = SCRIPT_DIR / "metrics.yaml"
TARGETS = (
    "r8-update-softcap0p5-injwarm512-rerun",
    "r16-update-softcap0p5-injwarm512-rerun",
)
GRAD_ACCUMULATION_STEPS = 4
DEFAULT_MAX_TRAIN_STEPS = 704
PASS_HARD_ACCURACY = 0.85
PASS_GAP = 0.04


def _load_source():
    spec = importlib.util.spec_from_file_location("joint_control_dgeom_base", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load source script: {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_source()
BASEMOD = BASE.BASEMOD


def _optimizer_to_train_forward_steps(optimizer_step: int) -> int:
    return int(optimizer_step) * GRAD_ACCUMULATION_STEPS


def _variant(**updates: Any) -> dict[str, Any]:
    spec = dict(BASE._COMMON_BASE)
    spec.update(
        {
            "description": "",
            "fox_remote_read_topk": 16,
            "fox_gd_residual_update_norm_softcap": None,
            "fox_gd_residual_update_norm_softcap_mode": "none",
            "fox_gd_residual_injection_warmup_start_train_steps": 0,
            "fox_gd_residual_injection_warmup_end_train_steps": 0,
            "fox_gd_residual_injection_warmup_eval_policy": "scheduled",
            "warmup_start_optimizer_step": 0,
            "warmup_end_optimizer_step": 0,
            "run_kind": "formal",
            "d_geometry_trace_train_steps": [],
            "fox_gd_residual_d_geometry_trace_enabled": False,
        }
    )
    spec.update(updates)
    return spec


VARIANTS: dict[str, dict[str, Any]] = {
    "r8-update-softcap0p5-injwarm512-rerun": _variant(
        description=(
            "formal rerun: read_topk=8 with smooth_p4 update_norm softcap=0.5 "
            "and residual injection warmup 0->512 optimizer steps"
        ),
        fox_remote_read_topk=8,
        fox_gd_residual_update_norm_softcap=0.5,
        fox_gd_residual_update_norm_softcap_mode="smooth_p4",
        fox_gd_residual_injection_warmup_start_train_steps=_optimizer_to_train_forward_steps(0),
        fox_gd_residual_injection_warmup_end_train_steps=_optimizer_to_train_forward_steps(512),
        fox_gd_residual_injection_warmup_eval_policy="final",
        warmup_start_optimizer_step=0,
        warmup_end_optimizer_step=512,
    ),
    "r16-update-softcap0p5-injwarm512-rerun": _variant(
        description=(
            "formal rerun: read_topk=16 with smooth_p4 update_norm softcap=0.5 "
            "and residual injection warmup 0->512 optimizer steps"
        ),
        fox_remote_read_topk=16,
        fox_gd_residual_update_norm_softcap=0.5,
        fox_gd_residual_update_norm_softcap_mode="smooth_p4",
        fox_gd_residual_injection_warmup_start_train_steps=_optimizer_to_train_forward_steps(0),
        fox_gd_residual_injection_warmup_end_train_steps=_optimizer_to_train_forward_steps(512),
        fox_gd_residual_injection_warmup_eval_policy="final",
        warmup_start_optimizer_step=0,
        warmup_end_optimizer_step=512,
    ),
}


def _patch_identity() -> None:
    compat_variants = dict(VARIANTS)
    compat_variants["default-r2"] = _variant(description="compatibility alias")
    compat_variants["fixed-r2-baseline"] = _variant(description="compatibility alias")
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
    compat_variants["default-r2"] = _variant(description="compatibility alias")
    compat_variants["fixed-r2-baseline"] = _variant(description="compatibility alias")
    BASE.VARIANTS = compat_variants
    BASE.TARGETS = TARGETS
    BASE.FORMAL_TARGETS = TARGETS
    BASE.DGEOM_TARGETS = ()
    BASE._patch_support()


_patch_limiter_support = _patch_support


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


def _variant_stability_rows(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in TARGETS:
        subset = [row for row in run_rows if row.get("variant") == variant and row.get("status") == "completed"]
        values = [
            _float_or_none(row.get("final_1024x256_accuracy"))
            for row in subset
            if _float_or_none(row.get("final_1024x256_accuracy")) is not None
        ]
        machines = ",".join(str(row.get("machine", "")) for row in subset)
        gpus = ",".join(str(row.get("gpu", "")) for row in subset)
        min_value = min(values) if values else None
        max_value = max(values) if values else None
        gap = None if min_value is None or max_value is None else max_value - min_value
        rows.append(
            {
                "variant": variant,
                "read_topk": VARIANTS[variant].get("fox_remote_read_topk"),
                "completed_runs": len(subset),
                "machines": machines,
                "gpus": gpus,
                "final_1024x256_values": ",".join(f"{value:.6f}" for value in values),
                "final_1024x256_min": min_value,
                "final_1024x256_max": max_value,
                "final_gap": gap,
                "final_gap_percentage_points": None if gap is None else gap * 100.0,
                "all_above_0p85": bool(values) and len(values) == 3 and all(value >= PASS_HARD_ACCURACY for value in values),
                "passes_stability_screen": bool(values)
                and len(values) == 3
                and all(value >= PASS_HARD_ACCURACY for value in values)
                and gap is not None
                and gap <= PASS_GAP,
            }
        )
    return rows


def _write_master_summary(artifact_dir: Path, outputs_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for path in sorted(outputs_dir.glob("*/master-status.tsv")):
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                row = dict(row)
                row["status_path"] = str(path)
                rows.append(row)
    _write_csv(artifact_dir / "master-status-summary.csv", rows)


def _post_collect(args: Any, code: int) -> int:
    artifact_dir = args.artifact_dir if args.artifact_dir.is_absolute() else (SCRIPT_DIR / args.artifact_dir)
    outputs_dir = args.outputs_dir if args.outputs_dir.is_absolute() else (SCRIPT_DIR / args.outputs_dir)
    run_rows = _read_csv(artifact_dir / "run-summary.csv")
    stability_rows = _variant_stability_rows(run_rows)
    _write_csv(artifact_dir / "variant-stability-summary.csv", stability_rows)
    _write_csv(artifact_dir / "batch-summary.csv", stability_rows)
    _write_master_summary(artifact_dir, outputs_dir)
    readme = (
        f"# {EXPERIMENT_ID}\n\n"
        "本 artifact 收尾 r8/r16 joint control same-seed 三卡复跑. "
        "每个 variant 在 2080ti GPU0, 2080ti GPU1, 3090 GPU0 上各跑一次 1ep formal run. "
        "formal runs 关闭 D-geometry, read trace, hash probe 和 train-inline event trace.\n\n"
        "核心文件:\n\n"
        "- `run-summary.csv`: 每个 run 的 final/best metrics.\n"
        "- `variant-stability-summary.csv`: 每个 variant 的三卡 max-min gap 和 pass/fail.\n"
        "- `mechanism-metrics-summary.csv`: final validation residual read/write/state metrics.\n"
        "- `cache-init-preflight-summary.csv`: cache/init hash evidence.\n"
        "- `batch-order-summary.csv`: batch order hash evidence.\n"
        "- `master-status-summary.csv`: batch 自动接续状态.\n"
    )
    (artifact_dir / "README.md").write_text(readme, encoding="utf-8")
    metadata_path = artifact_dir / "metadata.json"
    if metadata_path.exists():
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        payload = {}
    payload.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "targets": TARGETS,
            "formal_training_targets": TARGETS,
            "d_geometry_targets": [],
            "parallel_rerun_design": {
                "r8_runs": ["2080ti-gpu0", "2080ti-gpu1", "3090-gpu0"],
                "r16_runs": ["2080ti-gpu0", "2080ti-gpu1", "3090-gpu0"],
                "batch_order": ["r8", "r16"],
            },
            "stability_rule": "three final 1024x256 values >= 0.85 and max-min gap <= 0.04",
        }
    )
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return code


def run_collect(args: Any) -> int:
    code = BASE.run_collect(args)
    return _post_collect(args, code)


def main() -> int:
    _patch_identity()
    _patch_support()
    BASE.BASEMOD.run_train = BASE.run_train
    BASE.BASEMOD.run_collect = run_collect
    BASE.run_collect = run_collect
    os.environ["FLASH_VQG_READ_TRACE_MODE"] = "disabled"
    return BASE.BASEMOD.main()


if __name__ == "__main__":
    raise SystemExit(main())
