#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

LOCAL_DIR = Path(__file__).resolve().parent
if str(LOCAL_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_DIR))

os.environ.setdefault("TRITON_F32_DEFAULT", "ieee")
os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
os.environ.setdefault("TORCH_DETERMINISTIC", "0")
os.environ.setdefault("GDN_KERNEL_DTYPE", "bfloat16")

import torch

from common import (
    ARMS,
    CANONICAL,
    EXPECTED_CACHE_HASH,
    EXPECTED_FLASH_COMMIT,
    EXPECTED_INIT,
    EXPERIMENT_ID,
    FASTEST,
    FLASH_ROOT,
    GDN,
    PYTHON,
    REPO_ROOT,
    SEEDS,
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


BASE_SCRIPT = (
    REPO_ROOT
    / "zoology/experiments/flash_vqg/scripts"
    / "20260724-01-flash-vqg-gd-residual-efficiency"
    / "efficiency_benchmark.py"
)
SMOKE_TRAIN_ORDER = (0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 4, 4)
SMOKE_VALID_ORDER = tuple(segment for segment in range(8) for _ in range(3))


def _load_base():
    spec = importlib.util.spec_from_file_location("fastest_gdn_mqar_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load base module: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_base()


def git_value(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        text=True,
        capture_output=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def init_path(arm: str) -> Path:
    value = BASE.GDN_CANONICAL_INIT if arm == GDN else BASE.CANONICAL_INIT
    return Path(value).resolve()


def state_dict_hash(state_dict: dict[str, torch.Tensor]) -> str:
    return BASE._state_dict_hash(state_dict)


def run_id(arm: str, seed: int, phase: str) -> str:
    return f"{arm}-s{seed}-bf16-b64ga4-{phase}"


def checkpoint_root(phase: str) -> Path:
    return run_root() / "checkpoints" / phase


def checkpoint_run_dir(config: Any) -> Path:
    return Path(config.checkpoint.root_dir) / str(config.launch_id) / str(config.run_id)


def result_path(arm: str, seed: int, phase: str) -> Path:
    return run_root() / "training" / phase / run_id(arm, seed, phase) / "result.json"


def source_identity(arm: str) -> dict[str, str]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "run_tag": run_tag(),
        "machine": "3090",
        "arm": arm,
        "precision": "bf16",
        "zoology_commit": git_value(REPO_ROOT, "rev-parse", "HEAD"),
        "flash_commit": git_value(FLASH_ROOT, "rev-parse", "HEAD"),
        "cache_content_sha256": EXPECTED_CACHE_HASH,
        "init_file_sha256": sha256_file(init_path(arm)),
    }


def _apply_common_flash(kwargs: dict[str, Any]) -> None:
    kwargs.update(
        {
            "block_len": 64,
            "local_num_blocks": 2,
            "fox_gd_residual_rank": 16,
            "fox_gd_residual_write_topk": 4,
            "fox_remote_read_topk": 16,
            "fox_gd_residual_remat_mode": "post_phase1",
            "fox_gd_residual_grouped_chunk_backend": "triton",
            "fox_gd_residual_selected_read_backend": "triton_remat",
            "fox_gd_residual_selected_read_chunk_size": 8192,
            "fox_gd_residual_triton_input_policy": "fp32_boundary",
            "fox_gd_residual_selected_read_input_policy": "fp32_boundary",
            "fox_gd_residual_compute_read_observations_when_disabled": True,
            "vq_compute_metrics_when_disabled": True,
            "fox_gd_residual_checkpoint_preserve_rng_state": True,
        }
    )


def _apply_fastest(kwargs: dict[str, Any]) -> None:
    kwargs.update(
        {
            "fox_gd_residual_builder": "persistent_scan_triton",
            "fox_gd_residual_persistent_tile_blocks": 8,
            "fox_gd_residual_persistent_host_empty_check": False,
            "fox_gd_residual_persistent_backward_backend": "fixed_slot_vjp",
            "fox_gd_residual_selected_read_backward_backend": (
                "triton_state_owner_r1a_s1_w2"
            ),
            "fox_gd_residual_geometry_backend": "head_grouped",
            "fox_gd_residual_selected_read_forward_backend": "hoisted_w2",
            "fox_gd_residual_scan_read_fusion_backend": "off",
        }
    )


def _apply_canonical(kwargs: dict[str, Any]) -> None:
    kwargs.update(
        {
            "fox_gd_residual_builder": "grouped_chunk_torch_ref",
            "fox_gd_residual_persistent_backward_backend": "autograd",
            "fox_gd_residual_selected_read_backward_backend": (
                "triton_deterministic_s1_head"
            ),
            "fox_gd_residual_geometry_backend": "event_gemv",
            "fox_gd_residual_selected_read_forward_backend": "query_w8",
            "fox_gd_residual_scan_read_fusion_backend": "off",
        }
    )


def _base_config(arm: str):
    flash = BASE._build_flash_config("core", "triton", "triton_remat")
    if arm == GDN:
        return BASE._build_gdn_config(flash.data)
    kwargs = BASE._find_flash_kwargs(flash.model)
    _apply_common_flash(kwargs)
    _apply_fastest(kwargs) if arm == FASTEST else _apply_canonical(kwargs)
    return flash


def _configure_training(config: Any, arm: str, seed: int, phase: str) -> None:
    config.seed = int(seed)
    config.data.seed = 123
    config.data.batch_size = (64, 16)
    config.gradient_accumulation_steps = 4
    config.validations_per_epoch = 3 if phase == "smoke" else 4
    config.max_epochs = 4 if phase == "formal" else 1
    config.max_train_steps = 3 if phase == "smoke" else None
    config.max_validation_batches = None
    config.early_stopping_metric = None
    config.early_stopping_threshold = None
    config.precision = "amp_bfloat16"
    config.resume_enabled = True
    config.resume_stop_after_optimizer_step = 1 if phase == "smoke" else None
    config.max_grad_scaler_skips = 0
    config.max_consecutive_grad_scaler_skips = 0
    config.resume_identity = source_identity(arm)
    config.training_runtime_initial_state = {}


def _configure_io(config: Any, arm: str, seed: int, phase: str) -> None:
    if phase == "smoke":
        config.data.train_batch_segment_order = list(SMOKE_TRAIN_ORDER)
        config.data.test_batch_segment_order = list(SMOKE_VALID_ORDER)
    else:
        config.data.train_batch_segment_order = None
        config.data.test_batch_segment_order = None
    config.metrics_white_list = []
    config.logger.backend = "none"
    config.checkpoint.enabled = True
    config.checkpoint.save_best = True
    config.checkpoint.save_last = True
    config.checkpoint.best_metric = "valid/accuracy"
    config.checkpoint.best_mode = "max"
    config.checkpoint.root_dir = str(checkpoint_root(phase))
    config.init_checkpoint_path = str(init_path(arm))
    config.init_checkpoint_strict = True
    config.launch_id = f"{EXPERIMENT_ID}-{run_tag()}-{phase}"
    config.sweep_id = EXPERIMENT_ID
    config.run_id = run_id(arm, seed, phase)
    config.training_telemetry_path = str(
        result_path(arm, seed, phase).parent / "telemetry.jsonl"
    )


def build_config(arm: str, seed: int, phase: str):
    descriptor(arm, seed)
    if phase not in {"smoke", "screen", "formal"}:
        raise ValueError(f"Unsupported phase: {phase}.")
    config = _base_config(arm)
    _configure_training(config, arm, seed, phase)
    _configure_io(config, arm, seed, phase)
    return config


def serialize_config(config: Any) -> dict[str, Any]:
    from zoology.checkpoints import serialize_train_config

    return serialize_train_config(config)


def normalized_config(config: Any) -> dict[str, Any]:
    payload = serialize_config(config)
    payload["run_id"] = "<run>"
    payload["launch_id"] = "<launch>"
    payload["checkpoint"]["root_dir"] = "<checkpoint>"
    payload["resume_path"] = None
    payload["training_telemetry_path"] = "<telemetry>"
    payload["resume_identity"] = "<identity>"
    return payload


def write_resolved_config(config: Any) -> Path:
    path = generated_root() / f"{config.run_id}.json"
    atomic_write_json(path, serialize_config(config))
    return path


def configure_numerics() -> None:
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checks = {
        "TRITON_F32_DEFAULT": "ieee",
        "NVIDIA_TF32_OVERRIDE": "0",
        "GDN_KERNEL_DTYPE": "bfloat16",
    }
    for key, expected in checks.items():
        if os.environ.get(key) != expected:
            raise RuntimeError(f"{key} must be {expected}.")


def environment_metadata() -> dict[str, Any]:
    import fla
    import triton

    available = torch.cuda.is_available()
    return {
        "sys_executable": sys.executable,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton": triton.__version__,
        "fla": fla.__version__,
        "cuda_available": available,
        "gpu_name": torch.cuda.get_device_name(0) if available else None,
        "gpu_capability": list(torch.cuda.get_device_capability(0)) if available else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_used_bytes": int(torch.cuda.device_memory_used()) if available else None,
        "zoology_commit": git_value(REPO_ROOT, "rev-parse", "HEAD"),
        "flash_commit": git_value(FLASH_ROOT, "rev-parse", "HEAD"),
        "zoology_branch": git_value(REPO_ROOT, "branch", "--show-current"),
        "flash_branch": git_value(FLASH_ROOT, "branch", "--show-current"),
        "zoology_status": git_value(REPO_ROOT, "status", "--short"),
        "flash_status": git_value(FLASH_ROOT, "status", "--short"),
    }


def model_audit(config: Any, arm: str) -> dict[str, Any]:
    model = BASE.LanguageModel(config.model)
    payload = torch.load(init_path(arm), map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(payload["model_state_dict"], strict=True)
    result = {
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "state_sha256": state_dict_hash(model.state_dict()),
        "strict": not incompatible.missing_keys and not incompatible.unexpected_keys,
        "active_state_capacity": 131072,
    }
    if arm != GDN:
        kwargs = BASE._find_flash_kwargs(config.model)
        result["flash_kwargs"] = {
            key: kwargs.get(key)
            for key in (
                "block_len",
                "local_num_blocks",
                "fox_gd_residual_rank",
                "fox_gd_residual_write_topk",
                "fox_remote_read_topk",
                "fox_gd_residual_remat_mode",
                "fox_gd_residual_builder",
                "fox_gd_residual_selected_read_backward_backend",
                "fox_gd_residual_persistent_backward_backend",
                "fox_gd_residual_geometry_backend",
                "fox_gd_residual_selected_read_forward_backend",
                "fox_gd_residual_selected_read_input_policy",
            )
        }
    return result


def _audit_job(arm: str, seed: int, phase: str) -> dict[str, Any]:
    config = build_config(arm, seed, phase)
    resolved = write_resolved_config(config)
    audit = model_audit(config, arm)
    expected = EXPECTED_INIT["gdn" if arm == GDN else "flash"]
    checks = {
        "params": audit["trainable_parameters"] == expected["params"],
        "state": audit["state_sha256"] == expected["state"],
        "strict": audit["strict"],
        "capacity": audit["active_state_capacity"] == 131072,
        "precision": config.precision == "amp_bfloat16",
        "batch": tuple(config.data.batch_size) == (64, 16),
        "ga": int(config.gradient_accumulation_steps) == 4,
    }
    return {
        **descriptor(arm, seed),
        "phase": phase,
        "resolved_config_path": str(resolved.resolve()),
        "audit": audit,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _audit_flash_contract(jobs: list[dict[str, Any]]) -> bool:
    by_arm = {row["arm"]: row["audit"]["flash_kwargs"] for row in jobs if row["phase"] == "formal" and row["seed"] == 123 and row["arm"] != GDN}
    fastest = by_arm[FASTEST]
    canonical = by_arm[CANONICAL]
    return all(
        (
            fastest["fox_gd_residual_builder"] == "persistent_scan_triton",
            fastest["fox_gd_residual_persistent_backward_backend"] == "fixed_slot_vjp",
            fastest["fox_gd_residual_geometry_backend"] == "head_grouped",
            fastest["fox_gd_residual_selected_read_forward_backend"] == "hoisted_w2",
            canonical["fox_gd_residual_builder"] == "grouped_chunk_torch_ref",
            canonical["fox_gd_residual_selected_read_backward_backend"] == "triton_deterministic_s1_head",
        )
    )


def preflight() -> dict[str, Any]:
    configure_numerics()
    env = environment_metadata()
    jobs = [
        _audit_job(row["arm"], row["seed"], phase)
        for phase in ("smoke", "screen", "formal")
        for row in training_descriptors(phase)
    ]
    sample = build_config(FASTEST, 123, "formal")
    cache = BASE._cache_content_hash(sample.data)
    checks = {
        "python": Path(sys.executable).resolve() == PYTHON.resolve(),
        "versions": (env["python"], env["torch"], env["torch_cuda"], env["triton"], env["fla"]) == ("3.12.11", "2.6.0+cu118", "11.8", "3.2.0", "0.4.2"),
        "cuda": env["cuda_available"],
        "gpu": env["gpu_name"] == "NVIDIA GeForce RTX 3090",
        "visible_gpu": env["cuda_visible_devices"] == "0",
        "gpu_free": env["gpu_used_bytes"] is not None and env["gpu_used_bytes"] < 1024**3,
        "cache": cache.get("combined_content_sha256") == EXPECTED_CACHE_HASH,
        "flash_commit": env["flash_commit"] == EXPECTED_FLASH_COMMIT,
        "source_clean": not env["zoology_status"] and not env["flash_status"],
        "branch": bool(re.fullmatch(r"20260801-\d{6}-fastest-flash-vs-gdn-mqar", env["zoology_branch"])),
        "init_files": all(sha256_file(init_path(arm)) == EXPECTED_INIT["gdn" if arm == GDN else "flash"]["file"] for arm in ARMS),
        "jobs": all(row["passed"] for row in jobs),
        "flash_contract": _audit_flash_contract(jobs),
    }
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "run_tag": run_tag(),
        "status": "passed" if all(checks.values()) else "failed",
        "recorded_at_utc": utc_now(),
        "environment": env,
        "cache": cache,
        "checks": checks,
        "jobs": jobs,
    }
    atomic_write_json(run_root() / "preflight.json", payload)
    if payload["status"] != "passed":
        raise RuntimeError(f"Preflight failed: {run_root() / 'preflight.json'}")
    return payload


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metrics = payload.get("metrics") or {}
    return {
        "path": str(path.resolve()),
        "file_sha256": sha256_file(path),
        "model_state_sha256": state_dict_hash(payload["model_state_dict"]),
        "epoch": int(payload.get("epoch", -1)) + 1,
        "metrics": metrics,
        "finite_metrics": all(
            math.isfinite(float(value))
            for value in metrics.values()
            if isinstance(value, (int, float))
        ),
    }


def _nested_hash(value: Any) -> str:
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


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def telemetry_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path.resolve()), "records": 0}
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if "train/optimizer_step_wall_sec" in row:
            rows[int(row["log_step"])] = row
    successful = [row for row in rows.values() if not int(row.get("train/optimizer_step_skipped", 0))]
    times = [float(row["train/optimizer_step_wall_sec"]) for row in successful[1:]]
    return {
        "path": str(path.resolve()),
        "records": len(rows),
        "successful_records": len(successful),
        "optimizer_step_wall_sec_p50": _percentile(times, 0.5),
        "optimizer_step_wall_sec_p90": _percentile(times, 0.9),
        "peak_allocated_mib": max((float(row.get("train/peak_allocated_mib", 0)) for row in rows.values()), default=0),
        "peak_reserved_mib": max((float(row.get("train/peak_reserved_mib", 0)) for row in rows.values()), default=0),
    }


def _runtime_audits(states: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        state["fox_gd_residual_triton_runtime_audit"]
        for state in states.values()
        if state.get("fox_gd_residual_triton_runtime_audit") is not None
    ]


def gate_autotune_snapshot() -> dict[str, Any]:
    from fla.modules import fused_norm_gate

    payload = {}
    for name in ("layer_norm_gated_fwd_kernel", "layer_norm_gated_bwd_kernel"):
        autotuner = getattr(fused_norm_gate, name).fn
        best = getattr(autotuner, "best_config", None)
        payload[name] = {
            "best_config": None
            if best is None
            else {
                "kwargs": dict(best.kwargs),
                "num_warps": best.num_warps,
                "num_stages": best.num_stages,
            },
            "cache_size": len(autotuner.cache),
        }
    return payload


def runtime_audit(states: dict[str, dict[str, Any]], arm: str) -> dict[str, Any]:
    if arm == GDN:
        return {"passed": True, "modules": 0, "fallbacks": 0}
    audits = _runtime_audits(states)
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
    persistent_expected = arm == FASTEST
    passed = bool(audits) and selected > 0 and recompute > 0 and fallbacks == 0
    passed = passed and ((persistent > 0) == persistent_expected)
    return {
        "passed": passed,
        "modules": len(audits),
        "selected_calls": selected,
        "selected_recompute_calls": recompute,
        "persistent_calls": persistent,
        "fallbacks": fallbacks,
    }


def _resume_audit(path: Path, arm: str) -> dict[str, Any]:
    resume = torch.load(path, map_location="cpu", weights_only=False)
    optimizer_dtypes = sorted(
        {
            str(value.dtype)
            for state in resume["optimizer_state_dict"]["state"].values()
            for value in state.values()
            if torch.is_tensor(value) and value.is_floating_point()
        }
    )
    model_dtypes = sorted(
        {
            str(value.dtype)
            for value in resume["model_state_dict"].values()
            if torch.is_tensor(value) and value.is_floating_point()
        }
    )
    return {
        "optimizer_step": int(resume["optimizer_step"]),
        "grad_scaler_skips": int(resume["grad_scaler_skips"]),
        "runtime_state": resume["model_runtime_state"],
        "model_state_dtypes": model_dtypes,
        "optimizer_state_dtypes": optimizer_dtypes,
        "model_state_sha256": _nested_hash(resume["model_state_dict"]),
        "optimizer_state_sha256": _nested_hash(resume["optimizer_state_dict"]),
        "runtime_audit": runtime_audit(resume["model_runtime_state"], arm),
    }


def _completed_result(result: dict[str, Any], checkpoint_dir: Path) -> bool:
    arm, phase = result["arm"], result["phase"]
    for role in ("last", "best"):
        result[f"{role}_checkpoint"] = checkpoint_metadata(checkpoint_dir / f"{role}.pt")
    result["resume_audit"] = _resume_audit(checkpoint_dir / "resume.pt", arm)
    expected_epoch = 4 if phase == "formal" else 1
    expected_steps = 3 if phase == "smoke" else None
    resume = result["resume_audit"]
    checks = {
        "epoch": result["last_checkpoint"]["epoch"] == expected_epoch,
        "metrics": result["last_checkpoint"]["finite_metrics"],
        "scaler": resume["grad_scaler_skips"] == 0,
        "model_dtype": resume["model_state_dtypes"] == ["torch.float32"],
        "optimizer_dtype": resume["optimizer_state_dtypes"] == ["torch.float32"],
        "runtime": resume["runtime_audit"]["passed"],
        "steps": expected_steps is None or resume["optimizer_step"] == expected_steps,
    }
    result["completion_checks"] = checks
    return all(checks.values())


def run_training(arm: str, seed: int, phase: str) -> int:
    configure_numerics()
    config = build_config(arm, seed, phase)
    resolved = write_resolved_config(config)
    output = result_path(arm, seed, phase)
    checkpoint_dir = checkpoint_run_dir(config)
    started_at, started = utc_now(), time.perf_counter()
    try:
        from zoology.train import TrainingInterrupted, train

        train(config)
        status, error, return_code = "completed", None, 0
    except TrainingInterrupted as exc:
        status, error, return_code = "controlled_stop", f"{type(exc).__name__}: {exc}", 75
    except BaseException as exc:
        status, error, return_code = "failed", f"{type(exc).__name__}: {exc}", 1
        traceback.print_exc()
    result = {
        **descriptor(arm, seed),
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
        "telemetry": telemetry_summary(Path(config.training_telemetry_path)),
    }
    if status == "completed" and not _completed_result(result, checkpoint_dir):
        result["status"] = "failed"
        result["error"] = "Completed training failed checkpoint, dtype, or runtime audit."
        return_code = 1
    if status == "completed":
        result["gate_autotune"] = gate_autotune_snapshot()
    atomic_write_json(output, result)
    print(json.dumps({"status": result["status"], "result": str(output)}, ensure_ascii=False))
    return return_code


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    train = sub.add_parser("train")
    train.add_argument("--arm", choices=ARMS, required=True)
    train.add_argument("--seed", choices=SEEDS, type=int, required=True)
    train.add_argument("--phase", choices=("smoke", "screen", "formal"), required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        preflight()
        return 0
    return run_training(args.arm, args.seed, args.phase)


if __name__ == "__main__":
    raise SystemExit(main())
