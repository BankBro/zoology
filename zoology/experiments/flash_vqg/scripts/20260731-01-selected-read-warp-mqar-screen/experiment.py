#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
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
    BACKENDS,
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
    utc_now,
)


BASE_DIR = REPO_ROOT / "zoology/experiments/flash_vqg/scripts/20260730-04-k2-persistent-scan-mqar-regression"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_upstream():
    current_common = sys.modules.get("common")
    base_common = _load_module("selected_warp_k2_common", BASE_DIR / "common.py")
    sys.modules["common"] = base_common
    try:
        return _load_module("selected_warp_k2_experiment", BASE_DIR / "experiment.py")
    finally:
        if current_common is None:
            sys.modules.pop("common", None)
        else:
            sys.modules["common"] = current_common


UPSTREAM = _load_upstream()
BASE = UPSTREAM.BASE


def run_id(variant: str, seed: int, phase: str) -> str:
    return f"{variant}-s{seed}-bf16-b64ga4-{phase}"


def checkpoint_root(phase: str) -> Path:
    return run_root() / "checkpoints" / phase


def result_path(variant: str, seed: int, phase: str) -> Path:
    return run_root() / "training" / phase / run_id(variant, seed, phase) / "result.json"


def source_identity(variant: str) -> dict[str, str]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "run_tag": run_tag(),
        "machine": "3090",
        "variant": variant,
        "selected_backward": BACKENDS[variant],
        "precision": "bf16",
        "zoology_commit": BASE.git_value(REPO_ROOT, "rev-parse", "HEAD"),
        "flash_commit": BASE.git_value(FLASH_ROOT, "rev-parse", "HEAD"),
        "cache_content_sha256": EXPECTED_CACHE_HASH,
        "init_file_sha256": sha256_file(BASE.init_path()),
    }


def build_config(variant: str, seed: int, phase: str):
    descriptor(variant, seed)
    if phase not in {"smoke", "screen"}:
        raise ValueError(f"Unsupported phase: {phase}.")
    base_phase = "smoke" if phase == "smoke" else "formal"
    previous_tag = os.environ.get("MQAR_K2_PERSISTENT_RUN_TAG")
    os.environ["MQAR_K2_PERSISTENT_RUN_TAG"] = run_tag()
    try:
        config = UPSTREAM._BASE_BUILD_CONFIG("p0-a1-block64", seed, base_phase)
    finally:
        if previous_tag is None:
            os.environ.pop("MQAR_K2_PERSISTENT_RUN_TAG", None)
        else:
            os.environ["MQAR_K2_PERSISTENT_RUN_TAG"] = previous_tag
    if phase == "smoke":
        config.resume_stop_after_optimizer_step = None
    else:
        config.max_epochs = 1
        config.max_train_steps = None
        config.validations_per_epoch = 4
    config.checkpoint.root_dir = str(checkpoint_root(phase))
    config.launch_id = f"{EXPERIMENT_ID}-{run_tag()}-{phase}"
    config.sweep_id = EXPERIMENT_ID
    config.run_id = run_id(variant, seed, phase)
    config.training_telemetry_path = str(
        run_root() / "training" / phase / config.run_id / "telemetry.jsonl"
    )
    config.resume_identity = source_identity(variant)
    kwargs = BASE.BASE._find_flash_kwargs(config.model)
    kwargs.update(
        {
            "block_len": 64,
            "local_num_blocks": 2,
            "fox_gd_residual_rank": 16,
            "fox_gd_residual_write_topk": 4,
            "fox_remote_read_topk": 16,
            "fox_gd_residual_remat_mode": "post_phase1",
            "fox_gd_residual_builder": "grouped_chunk_torch_ref",
            "fox_gd_residual_grouped_chunk_backend": "triton",
            "fox_gd_residual_selected_read_backend": "triton_remat",
            "fox_gd_residual_selected_read_backward_backend": BACKENDS[variant],
            "fox_gd_residual_selected_read_chunk_size": 8192,
            "fox_gd_residual_triton_input_policy": "fp32_boundary",
        }
    )
    return config


def serialize_config(config: Any) -> dict[str, Any]:
    from zoology.checkpoints import serialize_train_config

    return serialize_train_config(config)


def write_resolved_config(config: Any) -> Path:
    path = generated_root() / f"{config.run_id}.json"
    atomic_write_json(path, serialize_config(config))
    return path


def _flatten(payload: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(payload, dict):
        result = {}
        for key, value in payload.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten(value, child))
        return result
    if isinstance(payload, list):
        result = {}
        for index, value in enumerate(payload):
            result.update(_flatten(value, f"{prefix}[{index}]"))
        return result
    return {prefix: payload}


def normalized_config(config: Any) -> dict[str, Any]:
    payload = serialize_config(config)
    payload["run_id"] = "<run>"
    payload["launch_id"] = "<launch>"
    payload["checkpoint"]["root_dir"] = "<checkpoint>"
    payload["training_telemetry_path"] = "<telemetry>"
    payload["resume_identity"] = "<identity>"
    return payload


def config_differences(left: Any, right: Any) -> list[str]:
    left_flat = _flatten(normalized_config(left))
    right_flat = _flatten(normalized_config(right))
    keys = sorted(set(left_flat) | set(right_flat))
    return [key for key in keys if left_flat.get(key) != right_flat.get(key)]


def model_audit(config: Any) -> dict[str, Any]:
    audit = BASE.model_audit(config)
    kwargs = BASE.BASE._find_flash_kwargs(config.model)
    audit.update(
        {
            "builder": kwargs.get("fox_gd_residual_builder"),
            "selected_backward": kwargs.get(
                "fox_gd_residual_selected_read_backward_backend"
            ),
            "selected_chunk_size": kwargs.get(
                "fox_gd_residual_selected_read_chunk_size"
            ),
        }
    )
    return audit


def runtime_audit(states: dict[str, dict[str, Any]]) -> dict[str, Any]:
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
    core_dtypes = [audit["actual_core_dtype"] for audit in audits if "actual_core_dtype" in audit]
    passed = (
        bool(audits)
        and selected > 0
        and recompute > 0
        and persistent == 0
        and fallbacks == 0
        and bool(core_dtypes)
        and all(value == "float32" for value in core_dtypes)
    )
    return {
        "passed": passed,
        "modules": len(audits),
        "logical_selected_calls": selected,
        "selected_recompute_calls": recompute,
        "persistent_calls": persistent,
        "core_dtype_evidence": core_dtypes,
        "fallbacks": fallbacks,
    }


def preflight() -> dict[str, Any]:
    BASE.configure_numerics()
    env = BASE.environment_metadata()
    jobs = []
    for variant in VARIANTS:
        for phase in ("smoke", "screen"):
            config = build_config(variant, 123, phase)
            write_resolved_config(config)
            jobs.append(
                {
                    **descriptor(variant, 123),
                    "phase": phase,
                    "audit": model_audit(config),
                    "precision": config.precision,
                }
            )
    baseline = build_config(VARIANTS[0], 123, "screen")
    differences = {
        variant: config_differences(baseline, build_config(variant, 123, "screen"))
        for variant in VARIANTS[1:]
    }
    cache = BASE.BASE._cache_content_hash(baseline.data)
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
        "branch": bool(
            re.fullmatch(r"20260731-\d{6}-selected-read-warp-mqar-screen", env["zoology_branch"])
        ),
        "single_variable": all(
            len(rows) == 1
            and rows[0].endswith("fox_gd_residual_selected_read_backward_backend")
            for rows in differences.values()
        ),
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
            and audit["remat_mode"] == "post_phase1"
            and audit["builder"] == "grouped_chunk_torch_ref"
            and audit["selected_backward"] == BACKENDS[row["variant"]]
            and audit["selected_chunk_size"] == 8192
            and row["precision"] == "amp_bfloat16"
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
        digest.update(path.encode() + b"\0")
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode() + b"\0")
            digest.update(str(tuple(tensor.shape)).encode() + b"\0")
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=lambda entry: str(entry)):
                update(item[key], f"{path}.{key}")
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                update(child, f"{path}[{index}]")
        else:
            digest.update(repr(item).encode())

    update(value, "root")
    return digest.hexdigest()


def run_training(variant: str, seed: int, phase: str) -> int:
    BASE.configure_numerics()
    config = build_config(variant, seed, phase)
    resolved = write_resolved_config(config)
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
        "status": status,
        "error": error,
        "started_at_utc": started_at,
        "ended_at_utc": utc_now(),
        "wall_clock_sec": time.perf_counter() - started,
        "resolved_config_path": str(resolved.resolve()),
        "resolved_config_sha256": sha256_file(resolved),
        "normalized_config_sha256": stable_json_sha256(normalized_config(config)),
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "telemetry": BASE.telemetry_summary(Path(config.training_telemetry_path)),
    }
    if status == "completed":
        for role in ("last", "best"):
            result[f"{role}_checkpoint"] = BASE.checkpoint_metadata(
                checkpoint_dir / f"{role}.pt"
            )
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
                {
                    str(value.dtype)
                    for value in resume["model_state_dict"].values()
                    if torch.is_tensor(value) and value.is_floating_point()
                }
            ),
            "optimizer_state_dtypes": optimizer_dtypes,
            "grad_scaler_enabled": bool(resume["grad_scaler_state_dict"]),
        }
        result["runtime_audit"] = runtime_audit(result["resume_audit"]["runtime_state"])
        result["gate_autotune"] = UPSTREAM.gate_autotune_snapshot()
        valid = (
            result["last_checkpoint"]["epoch"] == 1
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
    print(json.dumps({"status": result["status"], "result": str(output)}))
    return return_code


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    train = sub.add_parser("train")
    train.add_argument("--variant", choices=VARIANTS, required=True)
    train.add_argument("--seed", choices=SEEDS, type=int, required=True)
    train.add_argument("--phase", choices=("smoke", "screen"), required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        preflight()
        return 0
    return run_training(args.variant, args.seed, args.phase)


if __name__ == "__main__":
    raise SystemExit(main())
