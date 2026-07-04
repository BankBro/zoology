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
    / "20260703-04-flash-vqg-safe-limiter-readk2-readk4-screen/safe_limiter_readk_screen.py"
)
EXPERIMENT_ID = "20260704-01-flash-vqg-default-dropout-read-support-write-confidence-screen"
ARTIFACT_DIR = REPO_ROOT / "docs/artifacts" / EXPERIMENT_ID
METRICS_YAML = SCRIPT_DIR / "metrics.yaml"
GRAD_ACCUMULATION_STEPS = 4
DEFAULT_MAX_TRAIN_STEPS = 704


def _load_source():
    spec = importlib.util.spec_from_file_location("read_support_write_confidence_base", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load source script: {SOURCE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASEWRAP = _load_source()
BASEMOD = BASEWRAP.BASEMOD

TARGETS = (
    "baseline-r2",
    "baseline-r4",
    "fixed-r8",
    "fixed-r16",
    "fixed-r64",
    "fixed-r32",
    "sched16to2-linear512",
    "write-mass-r2",
    "write-mass-injwarm512-r2",
)

FINAL_BASE_TARGETS = (
    "baseline-r2",
    "baseline-r4",
    "fixed-r8",
    "fixed-r16",
    "write-mass-r2",
    "write-mass-injwarm512-r2",
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
    "fox_gd_residual_write_strength_mode": "renorm_topk",
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
    "baseline-r2": _variant(2, description="default-dropout fixed read_topk=2 baseline"),
    "baseline-r4": _variant(4, description="default-dropout fixed read_topk=4 baseline"),
    "fixed-r8": _variant(8, description="default-dropout fixed read_topk=8"),
    "fixed-r16": _variant(16, description="default-dropout fixed read_topk=16"),
    "fixed-r64": _variant(
        64,
        description="default-dropout fixed read_topk=64 with chunked dense read",
        fox_gd_residual_dense_read_chunked=True,
    ),
    "fixed-r32": _variant(32, description="default-dropout fixed read_topk=32"),
    "sched16to2-linear512": dict(
        _COMMON_BASE,
        description="default-dropout scheduled read_topk 16->2 over 512 optimizer steps",
        fox_remote_read_topk=None,
        fox_remote_read_topk_initial=16,
        fox_remote_read_topk_final=2,
        fox_remote_read_topk_release_start_train_steps=_optimizer_to_train_forward_steps(0),
        fox_remote_read_topk_release_end_train_steps=_optimizer_to_train_forward_steps(512),
        fox_remote_read_topk_schedule="linear_int",
        fox_remote_read_topk_eval_policy="final",
        fox_gd_residual_dense_read_chunked=False,
    ),
    "write-mass-r2": _variant(
        2,
        description="read_topk=2 with topk_mass_scaled GD residual write strength",
        fox_gd_residual_write_strength_mode="topk_mass_scaled",
    ),
    "write-mass-injwarm512-r2": _variant(
        2,
        description="read_topk=2 with topk_mass_scaled write strength and 0->512 injection warmup",
        fox_gd_residual_write_strength_mode="topk_mass_scaled",
        fox_gd_residual_injection_warmup_start_train_steps=_optimizer_to_train_forward_steps(0),
        fox_gd_residual_injection_warmup_end_train_steps=_optimizer_to_train_forward_steps(512),
        fox_gd_residual_injection_warmup_eval_policy="final",
        warmup_start_optimizer_step=0,
        warmup_end_optimizer_step=512,
    ),
}

_FLASH_KEYS = tuple(dict.fromkeys(BASEWRAP._FLASH_KEYS + ("fox_gd_residual_write_strength_mode",)))
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
    BASEWRAP._disable_read_trace(config)


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
                "passes_screen": (
                    ""
                    if f2080 is None or f3090 is None or gap is None
                    else f2080 >= 0.82 and f3090 >= 0.82 and gap <= 0.04
                ),
                "read_topk": spec.get("fox_remote_read_topk"),
                "read_topk_initial": spec.get("fox_remote_read_topk_initial"),
                "read_topk_final": spec.get("fox_remote_read_topk_final"),
                "dense_read_chunked": spec.get("fox_gd_residual_dense_read_chunked"),
                "write_strength_mode": spec.get("fox_gd_residual_write_strength_mode"),
                "warmup_end_optimizer_step": spec.get("warmup_end_optimizer_step", ""),
            }
        )
    return rows


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
        raise RuntimeError("MQAR cache content hash does not match canonical hash.")
    init_payload = BASEMOD.BASE._verify_init_checkpoint(args.init_checkpoint)
    if not init_payload["match_expected"] or not init_payload["match_embedded"]:
        raise RuntimeError("Init checkpoint tensor hash does not match canonical hash.")
    print(
        "pretrain_data_guard=PASS "
        f"cache_files={cache_payload['file_count']} "
        f"cache_sha256={cache_payload['combined_content_sha256']} "
        f"init_sha256={init_payload['actual_model_state_sha256']}"
    )
    return _ORIGINAL_RUN_TRAIN(args)


def run_collect(args: Any) -> int:
    code = _ORIGINAL_RUN_COLLECT(args)
    artifact_dir = args.artifact_dir
    run_rows = _read_csv(artifact_dir / "run-summary.csv")
    gap_rows = _variant_gap_rows(run_rows)
    _write_csv(artifact_dir / "cross-machine-comparison.csv", gap_rows)
    readme = (
        f"# {EXPERIMENT_ID}\n\n"
        "本 artifact 收尾 default-dropout read-support / write-confidence 1ep screen. "
        "本轮保持 canonical cache/init/batch order, `embed_dropout=0.1`, `write_topk=4`, "
        "并关闭 read trace, train inline event trace 和 shadow dense read.\n\n"
        "核心文件:\n\n"
        "- `run-summary.csv`: per-run final/best metrics.\n"
        "- `cross-machine-comparison.csv`: 2080ti/3090 final hard gap by variant.\n"
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
            "trace_mode": "read_trace_disabled",
            "screen_pass_rule": "both machines final 1024x256 >= 0.82 and gap <= 0.04",
            "cross_machine_rows": len(gap_rows),
        }
    )
    _save_json(metadata_path, metadata)
    return code


def main() -> int:
    _patch_identity()
    _patch_support()
    BASEMOD.run_train = run_train
    BASEMOD.run_collect = run_collect
    os.environ["FLASH_VQG_READ_TRACE_MODE"] = "disabled"
    return BASEMOD.main()


if __name__ == "__main__":
    raise SystemExit(main())
