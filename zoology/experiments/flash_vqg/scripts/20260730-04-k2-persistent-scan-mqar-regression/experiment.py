#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

LOCAL_DIR = Path(__file__).resolve().parent
if str(LOCAL_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_DIR))

import torch

from common import (
    BUILDERS,
    EXPECTED_CACHE_HASH,
    EXPECTED_FLASH_COMMIT,
    EXPECTED_INIT_FILE_HASH,
    EXPECTED_INIT_STATE_HASH,
    EXPECTED_PARAMETERS,
    EXPERIMENT_ID,
    FLASH_ROOT,
    PYTHON,
    REPO_ROOT,
    SEEDS,
    VARIANTS,
    atomic_write_json,
    descriptor,
    generated_root,
    run_root,
    run_tag,
    sha256_file,
    stable_json_sha256,
    training_descriptors,
    utc_now,
)


BASE_PATH = REPO_ROOT / "zoology/experiments/flash_vqg/scripts/20260729-02-mqar-deterministic-selected-read-regression/experiment.py"


def _load_base():
    spec = importlib.util.spec_from_file_location("k2_mqar_formal_base", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load base experiment: {BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_base()
_BASE_BUILD_CONFIG = BASE.build_config


def run_id(variant: str, seed: int, phase: str) -> str:
    precision = "fp32" if phase == "diagnostic_fp32" else "bf16"
    return f"{variant}-s{seed}-{precision}-b64ga4-{phase}"


def checkpoint_root(phase: str) -> Path:
    return run_root() / "checkpoints" / phase


def result_path(variant: str, seed: int, phase: str) -> Path:
    return run_root() / "training" / phase / run_id(variant, seed, phase) / "result.json"


def source_identity(variant: str) -> dict[str, str]:
    identity = BASE.source_identity(variant)
    identity["block_len"] = "64"
    return identity


def build_config(variant: str, seed: int, phase: str):
    descriptor(variant, seed)
    if phase not in {"smoke", "screen", "formal", "diagnostic_fp32"}:
        raise ValueError(f"Unsupported phase: {phase}.")
    base_phase = "smoke" if phase == "smoke" else "formal"
    config = _BASE_BUILD_CONFIG(variant, seed, base_phase)
    if phase == "smoke":
        config.resume_stop_after_optimizer_step = None
    elif phase in {"screen", "diagnostic_fp32"}:
        config.max_epochs = 1
        config.max_train_steps = None
        config.validations_per_epoch = 4
    if phase == "diagnostic_fp32":
        config.precision = "float32"
    config.checkpoint.root_dir = str(checkpoint_root(phase))
    config.launch_id = f"{EXPERIMENT_ID}-{run_tag()}-{phase}"
    config.sweep_id = EXPERIMENT_ID
    config.run_id = run_id(variant, seed, phase)
    config.training_telemetry_path = str(
        run_root() / "training" / phase / config.run_id / "telemetry.jsonl"
    )
    config.resume_identity = source_identity(variant)
    config.resume_identity["precision"] = "fp32" if phase == "diagnostic_fp32" else "bf16"
    kwargs = BASE.BASE._find_flash_kwargs(config.model)
    kwargs.update(
        {
            "block_len": 64,
            "local_num_blocks": 2,
            "fox_gd_residual_rank": 16,
            "fox_gd_residual_write_topk": 4,
            "fox_remote_read_topk": 16,
            "fox_gd_residual_remat_mode": "post_phase1",
            "fox_gd_residual_builder": BUILDERS[variant],
            "fox_gd_residual_persistent_tile_blocks": 8,
            "fox_gd_residual_grouped_chunk_backend": "triton",
            "fox_gd_residual_selected_read_backend": "triton_remat",
            "fox_gd_residual_selected_read_backward_backend": "triton_deterministic",
            "fox_gd_residual_triton_input_policy": "fp32_boundary",
        }
    )
    return config


def model_audit(config: Any) -> dict[str, Any]:
    audit = BASE.model_audit(config)
    kwargs = BASE.BASE._find_flash_kwargs(config.model)
    audit.update(
        {
            "builder": kwargs.get("fox_gd_residual_builder"),
            "tile_blocks": kwargs.get("fox_gd_residual_persistent_tile_blocks"),
            "selected_backward": kwargs.get("fox_gd_residual_selected_read_backward_backend"),
        }
    )
    return audit


def runtime_audit(states: dict[str, dict[str, Any]], variant: str) -> dict[str, Any]:
    audits = [
        state.get("fox_gd_residual_triton_runtime_audit")
        for state in states.values()
        if state.get("fox_gd_residual_triton_runtime_audit") is not None
    ]
    fallback_keys = (
        "grouped_fallbacks",
        "selected_fallbacks",
        "grouped_recompute_fallbacks",
        "selected_recompute_fallbacks",
        "persistent_fallbacks",
    )
    fallbacks = sum(int(audit.get(key, 0)) for audit in audits for key in fallback_keys)
    selected = sum(int(audit.get("selected_calls", 0)) for audit in audits)
    recompute = sum(int(audit.get("selected_recompute_calls", 0)) for audit in audits)
    persistent = sum(int(audit.get("persistent_calls", 0)) for audit in audits)
    expects_persistent = BUILDERS[variant] == "persistent_scan_triton"
    core_dtypes = [audit["actual_core_dtype"] for audit in audits if "actual_core_dtype" in audit]
    dtype_passed = all(value == "float32" for value in core_dtypes) and (
        bool(core_dtypes) or expects_persistent
    )
    passed = (
        bool(audits)
        and selected > 0
        and recompute > 0
        and fallbacks == 0
        and ((persistent > 0) == expects_persistent)
        and dtype_passed
    )
    return {
        "passed": passed,
        "modules": len(audits),
        "logical_selected_calls": selected,
        "selected_recompute_calls": recompute,
        "persistent_calls": persistent,
        "core_dtype_evidence": core_dtypes or ["persistent_fp32_contract"],
        "fallbacks": fallbacks,
    }


def _config_jobs() -> list[dict[str, Any]]:
    jobs = []
    for row in training_descriptors():
        for phase in ("formal",):
            jobs.append({**row, "phase": phase})
    for variant in VARIANTS:
        for phase in ("smoke", "screen"):
            jobs.append({**descriptor(variant, 123), "phase": phase})
        jobs.append({**descriptor(variant, 123), "phase": "diagnostic_fp32"})
    return jobs


def preflight() -> dict[str, Any]:
    BASE.configure_numerics()
    env = BASE.environment_metadata()
    jobs = []
    for row in _config_jobs():
        config = build_config(row["variant"], row["seed"], row["phase"])
        BASE.write_resolved_config(config)
        jobs.append({**row, "audit": model_audit(config), "precision": config.precision})
    p0 = build_config("p0-a1-block64", 123, "formal")
    k2 = build_config("k2-persistent-p8", 123, "formal")
    differences = BASE.config_differences(p0, k2)
    cache = BASE.BASE._cache_content_hash(p0.data)
    checks = {
        "python": Path(sys.executable).resolve() == PYTHON.resolve(),
        "versions": (env["python"], env["torch"], env["torch_cuda"], env["triton"], env["fla"])
        == ("3.12.11", "2.6.0+cu118", "11.8", "3.2.0", "0.4.2"),
        "cuda": env["cuda_available"],
        "gpu": env["gpu_name"] == "NVIDIA GeForce RTX 3090",
        "visible_gpu": env["cuda_visible_devices"] == "0",
        "gpu_free": env["gpu_used_bytes"] is not None and env["gpu_used_bytes"] < 1024**3,
        "cache": cache.get("combined_content_sha256") == EXPECTED_CACHE_HASH,
        "init_file": sha256_file(BASE.init_path()) == EXPECTED_INIT_FILE_HASH,
        "flash_commit": env["flash_commit"] == EXPECTED_FLASH_COMMIT,
        "source_clean": not env["zoology_status"] and not env["flash_status"],
        "branch": bool(re.fullmatch(r"\d{8}-\d{6}-k2-persistent-mqar-regression", env["zoology_branch"])),
        "single_variable": len(differences) == 1 and differences[0].endswith("fox_gd_residual_builder"),
    }
    job_checks = []
    for row in jobs:
        audit = row["audit"]
        job_checks.append(
            audit["trainable_parameters"] == EXPECTED_PARAMETERS
            and audit["state_sha256"] == EXPECTED_INIT_STATE_HASH
            and audit["strict"]
            and audit["block_len"] == 64
            and audit["local_num_blocks"] == 2
            and audit["rank"] == 16
            and audit["read_topk"] == 16
            and audit["write_topk"] == 4
            and audit["input_policy"] == "fp32_boundary"
            and audit["grouped_backend"] == "triton"
            and audit["selected_backend"] == "triton_remat"
            and audit["selected_backward"] == "triton_deterministic"
            and audit["remat_mode"] == "post_phase1"
            and audit["builder"] == BUILDERS[row["variant"]]
            and audit["tile_blocks"] == 8
            and row["precision"]
            == ("float32" if row["phase"] == "diagnostic_fp32" else "amp_bfloat16")
        )
    checks["jobs"] = all(job_checks)
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "run_tag": run_tag(),
        "status": "passed" if all(checks.values()) else "failed",
        "recorded_at_utc": utc_now(),
        "environment": env,
        "cache": cache,
        "config_differences": differences,
        "checks": checks,
        "jobs": jobs,
    }
    path = run_root() / "preflight.json"
    atomic_write_json(path, payload)
    if payload["status"] != "passed":
        raise RuntimeError(f"Preflight failed: {path}")
    return payload


def _nested_state_hash(value: Any) -> str:
    digest = hashlib.sha256()

    def update(item: Any, path: str) -> None:
        digest.update(path.encode("utf-8") + b"\0")
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode("ascii") + b"\0")
            digest.update(str(tuple(tensor.shape)).encode("ascii") + b"\0")
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=lambda entry: str(entry)):
                update(item[key], f"{path}.{key}")
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                update(child, f"{path}[{index}]")
        else:
            digest.update(repr(item).encode("utf-8"))

    update(value, "root")
    return digest.hexdigest()


def _triton_config_record(config: Any) -> dict[str, Any]:
    return {
        "kwargs": dict(config.kwargs),
        "num_warps": config.num_warps,
        "num_stages": config.num_stages,
        "num_ctas": getattr(config, "num_ctas", None),
    }


def gate_autotune_snapshot() -> dict[str, Any]:
    from fla.modules import fused_norm_gate

    result = {}
    for name in ("layer_norm_gated_fwd_kernel", "layer_norm_gated_bwd_kernel"):
        autotuner = getattr(fused_norm_gate, name).fn
        best = getattr(autotuner, "best_config", None)
        result[name] = {
            "best_config": None if best is None else _triton_config_record(best),
            "cache_size": len(autotuner.cache),
        }
    return result


def run_training(variant: str, seed: int, phase: str) -> int:
    BASE.configure_numerics()
    config = build_config(variant, seed, phase)
    resolved = BASE.write_resolved_config(config)
    output = result_path(variant, seed, phase)
    checkpoint_dir = BASE.checkpoint_run_dir(config)
    started_at, started = utc_now(), time.perf_counter()
    try:
        from zoology.train import train

        train(config)
        status, error, return_code = "completed", None, 0
    except BaseException as exc:
        status, error, return_code = "failed", f"{type(exc).__name__}: {exc}", 1
        traceback.print_exc()
    result: dict[str, Any] = {
        **descriptor(variant, seed),
        "experiment_id": EXPERIMENT_ID,
        "run_tag": run_tag(),
        "phase": phase,
        "train_precision": "fp32" if phase == "diagnostic_fp32" else "bf16",
        "status": status,
        "error": error,
        "started_at_utc": started_at,
        "ended_at_utc": utc_now(),
        "wall_clock_sec": time.perf_counter() - started,
        "resolved_config_path": str(resolved.resolve()),
        "resolved_config_sha256": sha256_file(resolved),
        "normalized_config_sha256": stable_json_sha256(BASE.normalized_config(config, mask_variant=False)),
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "telemetry": BASE.telemetry_summary(Path(config.training_telemetry_path)),
    }
    if status == "completed":
        expected_epoch = 4 if phase == "formal" else 1
        for role in ("last", "best"):
            result[f"{role}_checkpoint"] = BASE.checkpoint_metadata(checkpoint_dir / f"{role}.pt")
        resume = torch.load(checkpoint_dir / "resume.pt", map_location="cpu", weights_only=False)
        optimizer_dtypes = sorted(
            {
                str(value.dtype)
                for state in resume["optimizer_state_dict"]["state"].values()
                for value in state.values()
                if torch.is_tensor(value) and value.is_floating_point()
            }
        )
        result["resume_audit"] = {
            "optimizer_step": int(resume["optimizer_step"]),
            "grad_scaler_skips": int(resume["grad_scaler_skips"]),
            "runtime_state": resume["model_runtime_state"],
            "model_state_sha256": _nested_state_hash(resume["model_state_dict"]),
            "optimizer_state_sha256": _nested_state_hash(resume["optimizer_state_dict"]),
            "model_state_dtypes": sorted(
                {str(value.dtype) for value in resume["model_state_dict"].values() if torch.is_tensor(value) and value.is_floating_point()}
            ),
            "optimizer_state_dtypes": optimizer_dtypes,
            "grad_scaler_enabled": bool(resume["grad_scaler_state_dict"]),
        }
        result["runtime_audit"] = runtime_audit(result["resume_audit"]["runtime_state"], variant)
        result["gate_autotune"] = gate_autotune_snapshot()
        valid = (
            result["last_checkpoint"]["epoch"] == expected_epoch
            and result["last_checkpoint"]["finite_metrics"]
            and result["resume_audit"]["grad_scaler_skips"] == 0
            and result["resume_audit"]["model_state_dtypes"] == ["torch.float32"]
            and result["resume_audit"]["optimizer_state_dtypes"] == ["torch.float32"]
            and result["runtime_audit"]["passed"]
        )
        if not valid:
            result["status"] = "failed"
            result["error"] = "Completed training failed checkpoint, dtype or runtime audit."
            return_code = 1
    atomic_write_json(output, result)
    print(json.dumps({"status": result["status"], "result": str(output)}, ensure_ascii=False))
    return return_code


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    train = sub.add_parser("train")
    train.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    train.add_argument("--seed", choices=SEEDS, type=int, required=True)
    train.add_argument(
        "--phase",
        choices=("smoke", "screen", "formal", "diagnostic_fp32"),
        required=True,
    )
    args = parser.parse_args()
    if args.command == "preflight":
        preflight()
        return 0
    return run_training(args.variant, args.seed, args.phase)


if __name__ == "__main__":
    raise SystemExit(main())
