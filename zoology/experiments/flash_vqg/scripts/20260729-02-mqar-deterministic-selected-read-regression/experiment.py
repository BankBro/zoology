#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

import torch

from common import (
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


BASE_SCRIPT = (
    REPO_ROOT
    / "zoology/experiments/flash_vqg/scripts"
    / "20260724-01-flash-vqg-gd-residual-efficiency"
    / "efficiency_benchmark.py"
)
SMOKE_TRAIN_SEGMENT_ORDER = (0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 4, 4)
SMOKE_VALID_SEGMENT_ORDER = tuple(segment for segment in range(8) for _ in range(3))


def _load_base():
    spec = importlib.util.spec_from_file_location("mqar_remat_base", BASE_SCRIPT)
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


def init_path() -> Path:
    return Path(BASE.CANONICAL_INIT).resolve()


def state_dict_hash(state_dict: dict[str, torch.Tensor]) -> str:
    return BASE._state_dict_hash(state_dict)


def run_id(variant: str, seed: int, phase: str) -> str:
    return f"{variant}-s{seed}-bf16-b64ga4-{phase}"


def checkpoint_root(phase: str) -> Path:
    return run_root() / "checkpoints" / phase


def checkpoint_run_dir(config: Any) -> Path:
    return Path(config.checkpoint.root_dir) / str(config.launch_id) / str(config.run_id)


def source_identity(variant: str) -> dict[str, str]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "run_tag": run_tag(),
        "machine": "3090",
        "variant": variant,
        "remat_mode": VARIANTS[variant],
        "precision": "bf16",
        "zoology_commit": git_value(REPO_ROOT, "rev-parse", "HEAD"),
        "flash_commit": git_value(FLASH_ROOT, "rev-parse", "HEAD"),
        "cache_content_sha256": EXPECTED_CACHE_HASH,
        "init_file_sha256": sha256_file(init_path()),
    }


def build_config(variant: str, seed: int, phase: str):
    descriptor(variant, seed)
    if phase not in {"smoke", "formal"}:
        raise ValueError(f"Unsupported phase: {phase}.")
    config = BASE._build_flash_config("core", "triton", "triton_remat")
    config.seed = int(seed)
    config.data.seed = 123
    config.data.batch_size = (64, 16)
    config.gradient_accumulation_steps = 4
    config.validations_per_epoch = 4 if phase == "formal" else 3
    config.max_epochs = 4 if phase == "formal" else 1
    config.max_train_steps = None if phase == "formal" else 3
    config.max_validation_batches = None
    config.early_stopping_metric = None
    config.early_stopping_threshold = None
    config.precision = "amp_bfloat16"
    config.resume_enabled = True
    config.resume_stop_after_optimizer_step = 1 if phase == "smoke" else None
    config.max_grad_scaler_skips = 0
    config.max_consecutive_grad_scaler_skips = 0
    config.resume_identity = source_identity(variant)
    config.training_runtime_initial_state = {}
    if phase == "smoke":
        config.data.train_batch_segment_order = list(SMOKE_TRAIN_SEGMENT_ORDER)
        config.data.test_batch_segment_order = list(SMOKE_VALID_SEGMENT_ORDER)
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
    config.init_checkpoint_path = str(init_path())
    config.init_checkpoint_strict = True
    config.launch_id = f"{EXPERIMENT_ID}-{run_tag()}-{phase}"
    config.sweep_id = EXPERIMENT_ID
    config.run_id = run_id(variant, seed, phase)
    config.training_telemetry_path = str(
        run_root() / "training" / phase / config.run_id / "telemetry.jsonl"
    )
    kwargs = BASE._find_flash_kwargs(config.model)
    kwargs["fox_gd_residual_triton_input_policy"] = "fp32_boundary"
    kwargs["fox_gd_residual_remat_mode"] = VARIANTS[variant]
    return config


def serialize_config(config: Any) -> dict[str, Any]:
    from zoology.checkpoints import serialize_train_config

    return serialize_train_config(config)


def normalized_config(config: Any, *, mask_variant: bool) -> dict[str, Any]:
    payload = serialize_config(config)
    payload["run_id"] = "<run>"
    payload["launch_id"] = "<launch>"
    payload["checkpoint"]["root_dir"] = "<checkpoint>"
    payload["resume_path"] = None
    payload["training_telemetry_path"] = "<telemetry>"
    identity = payload.get("resume_identity") or {}
    identity["variant"] = "<variant>"
    identity["remat_mode"] = "<remat>" if mask_variant else identity.get("remat_mode")
    return payload


def resolved_config_path(config: Any) -> Path:
    return generated_root() / f"{config.run_id}.json"


def write_resolved_config(config: Any) -> Path:
    path = resolved_config_path(config)
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


def config_differences(left: Any, right: Any) -> list[str]:
    left_flat = _flatten(normalized_config(left, mask_variant=True))
    right_flat = _flatten(normalized_config(right, mask_variant=True))
    keys = sorted(set(left_flat) | set(right_flat))
    return [key for key in keys if left_flat.get(key) != right_flat.get(key)]


def configure_numerics() -> None:
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if os.environ.get("TRITON_F32_DEFAULT") != "ieee":
        raise RuntimeError("TRITON_F32_DEFAULT must be ieee.")
    if os.environ.get("NVIDIA_TF32_OVERRIDE") != "0":
        raise RuntimeError("NVIDIA_TF32_OVERRIDE must be 0.")


def environment_metadata() -> dict[str, Any]:
    import fla
    import triton

    cuda_available = torch.cuda.is_available()
    return {
        "sys_executable": sys.executable,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton": triton.__version__,
        "fla": fla.__version__,
        "cuda_available": cuda_available,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "gpu_capability": list(torch.cuda.get_device_capability(0)) if cuda_available else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_used_bytes": int(torch.cuda.device_memory_used()) if cuda_available else None,
        "zoology_commit": git_value(REPO_ROOT, "rev-parse", "HEAD"),
        "flash_commit": git_value(FLASH_ROOT, "rev-parse", "HEAD"),
        "zoology_branch": git_value(REPO_ROOT, "branch", "--show-current"),
        "flash_branch": git_value(FLASH_ROOT, "branch", "--show-current"),
        "zoology_status": git_value(REPO_ROOT, "status", "--short"),
        "flash_status": git_value(FLASH_ROOT, "status", "--short"),
    }


def selected_read_test_gate() -> dict[str, Any]:
    command = [
        str(PYTHON),
        "-m",
        "pytest",
        "-q",
        str(FLASH_ROOT / "tests/test_fox_gd_residual_v1.py"),
        "-k",
        "selected_read",
    ]
    result = subprocess.run(
        command,
        cwd=FLASH_ROOT,
        text=True,
        capture_output=True,
        env=os.environ.copy(),
    )
    output = "\n".join(
        value.strip() for value in (result.stdout, result.stderr) if value.strip()
    )
    return {
        "command": command,
        "return_code": int(result.returncode),
        "passed": result.returncode == 0 and "3 passed" in output,
        "output_tail": "\n".join(output.splitlines()[-20:]),
    }


def model_audit(config: Any) -> dict[str, Any]:
    model = BASE.LanguageModel(config.model)
    payload = torch.load(init_path(), map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(payload["model_state_dict"], strict=True)
    kwargs = BASE._find_flash_kwargs(config.model)
    return {
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "state_sha256": state_dict_hash(model.state_dict()),
        "strict": not incompatible.missing_keys and not incompatible.unexpected_keys,
        "block_len": kwargs.get("block_len"),
        "local_num_blocks": kwargs.get("local_num_blocks"),
        "rank": kwargs.get("fox_gd_residual_rank"),
        "read_topk": kwargs.get("fox_remote_read_topk"),
        "write_topk": kwargs.get("fox_gd_residual_write_topk"),
        "input_policy": kwargs.get("fox_gd_residual_triton_input_policy"),
        "grouped_backend": kwargs.get("fox_gd_residual_grouped_chunk_backend"),
        "selected_backend": kwargs.get("fox_gd_residual_selected_read_backend"),
        "remat_mode": kwargs.get("fox_gd_residual_remat_mode"),
    }


def preflight() -> dict[str, Any]:
    configure_numerics()
    env = environment_metadata()
    source_tests = selected_read_test_gate()
    jobs = []
    for row in training_descriptors():
        for phase in ("smoke", "formal"):
            config = build_config(row["variant"], row["seed"], phase)
            write_resolved_config(config)
            audit = model_audit(config)
            jobs.append({**row, "phase": phase, "audit": audit})
    a0 = build_config("a0-fixed-off", 124, "formal")
    a1 = build_config("a1-fixed-post-phase1", 124, "formal")
    differences = config_differences(a0, a1)
    cache = BASE._cache_content_hash(a0.data)
    common_checks = {
        "python": Path(sys.executable).resolve() == PYTHON.resolve(),
        "python_version": env["python"] == "3.12.11",
        "torch": env["torch"] == "2.6.0+cu118",
        "cuda": env["torch_cuda"] == "11.8",
        "triton": env["triton"] == "3.2.0",
        "fla": env["fla"] == "0.4.2",
        "cuda_available": env["cuda_available"],
        "gpu": env["gpu_name"] == "NVIDIA GeForce RTX 3090",
        "visible_gpu": env["cuda_visible_devices"] == "0",
        "gpu_free": env["gpu_used_bytes"] is not None and env["gpu_used_bytes"] < 1024**3,
        "cache": cache.get("combined_content_sha256") == EXPECTED_CACHE_HASH,
        "init_file": sha256_file(init_path()) == EXPECTED_INIT_FILE_HASH,
        "flash_commit": env["flash_commit"] == EXPECTED_FLASH_COMMIT,
        "zoology_clean": not env["zoology_status"],
        "flash_clean": not env["flash_status"],
        "zoology_branch": bool(
            re.fullmatch(
                r"\d{8}-\d{6}-mqar-deterministic-selected-read-regression",
                env["zoology_branch"],
            )
        ),
        "single_variable": len(differences) == 1 and differences[0].endswith("fox_gd_residual_remat_mode"),
        "selected_read_tests": source_tests["passed"],
    }
    job_checks = []
    for row in jobs:
        audit = row["audit"]
        job_checks.append(
            audit["trainable_parameters"] == EXPECTED_PARAMETERS
            and audit["state_sha256"] == EXPECTED_INIT_STATE_HASH
            and audit["strict"]
            and audit["block_len"] == 32
            and audit["local_num_blocks"] == 2
            and audit["rank"] == 16
            and audit["read_topk"] == 16
            and audit["write_topk"] == 4
            and audit["input_policy"] == "fp32_boundary"
            and audit["grouped_backend"] == "triton"
            and audit["selected_backend"] == "triton_remat"
            and audit["remat_mode"] == row["remat_mode"]
        )
    checks = {**common_checks, "jobs": all(job_checks)}
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "run_tag": run_tag(),
        "status": "passed" if all(checks.values()) else "failed",
        "recorded_at_utc": utc_now(),
        "environment": env,
        "cache": cache,
        "selected_read_tests": source_tests,
        "config_differences": differences,
        "checks": checks,
        "jobs": jobs,
    }
    path = run_root() / "preflight.json"
    atomic_write_json(path, payload)
    if payload["status"] != "passed":
        raise RuntimeError(f"Preflight failed: {path}")
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


def percentile(values: list[float], quantile: float) -> float | None:
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
    stable = successful[1:]
    times = [float(row["train/optimizer_step_wall_sec"]) for row in stable]
    return {
        "path": str(path.resolve()),
        "records": len(rows),
        "successful_records": len(successful),
        "optimizer_step_wall_sec_p50": percentile(times, 0.5),
        "optimizer_step_wall_sec_p90": percentile(times, 0.9),
        "peak_allocated_mib": max((float(row.get("train/peak_allocated_mib", 0)) for row in rows.values()), default=0),
        "peak_reserved_mib": max((float(row.get("train/peak_reserved_mib", 0)) for row in rows.values()), default=0),
    }


def runtime_audit(states: dict[str, dict[str, Any]], remat_mode: str) -> dict[str, Any]:
    audits = [
        state.get("fox_gd_residual_triton_runtime_audit")
        for state in states.values()
        if state.get("fox_gd_residual_triton_runtime_audit") is not None
    ]
    fallbacks = sum(
        int(audit.get(key, 0))
        for audit in audits
        for key in (
            "grouped_fallbacks",
            "selected_fallbacks",
            "grouped_recompute_fallbacks",
            "selected_recompute_fallbacks",
        )
    )
    recompute_calls = sum(
        int(audit.get("selected_recompute_calls", 0)) for audit in audits
    )
    logical_calls = sum(int(audit.get("selected_calls", 0)) for audit in audits)
    passed = (
        bool(audits)
        and logical_calls > 0
        and fallbacks == 0
        and all(audit.get("actual_core_dtype") == "float32" for audit in audits)
        and ((recompute_calls > 0) == (remat_mode == "post_phase1"))
    )
    return {
        "passed": passed,
        "modules": len(audits),
        "logical_selected_calls": logical_calls,
        "selected_recompute_calls": recompute_calls,
        "fallbacks": fallbacks,
    }


def result_path(variant: str, seed: int, phase: str) -> Path:
    return run_root() / "training" / phase / run_id(variant, seed, phase) / "result.json"


def run_training(variant: str, seed: int, phase: str) -> int:
    configure_numerics()
    config = build_config(variant, seed, phase)
    resolved = write_resolved_config(config)
    output = result_path(variant, seed, phase)
    checkpoint_dir = checkpoint_run_dir(config)
    started_at = utc_now()
    started = time.perf_counter()
    try:
        from zoology.train import TrainingInterrupted, train

        train(config)
        status, error, return_code = "completed", None, 0
    except TrainingInterrupted as exc:
        status, error, return_code = "controlled_stop", f"{type(exc).__name__}: {exc}", 75
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
        "normalized_config_sha256": stable_json_sha256(normalized_config(config, mask_variant=False)),
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "resume_path": str((checkpoint_dir / "resume.pt").resolve()),
        "telemetry": telemetry_summary(Path(config.training_telemetry_path)),
    }
    if status == "completed":
        expected_epoch = 4 if phase == "formal" else 1
        for role in ("last", "best"):
            result[f"{role}_checkpoint"] = checkpoint_metadata(checkpoint_dir / f"{role}.pt")
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
            "model_state_dtypes": sorted(
                {str(value.dtype) for value in resume["model_state_dict"].values() if torch.is_tensor(value) and value.is_floating_point()}
            ),
            "optimizer_state_dtypes": optimizer_dtypes,
            "grad_scaler_enabled": bool(resume["grad_scaler_state_dict"]),
        }
        result["runtime_audit"] = runtime_audit(
            result["resume_audit"]["runtime_state"],
            VARIANTS[variant],
        )
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
            result["error"] = "Completed training failed checkpoint or dtype audit."
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
    train.add_argument("--phase", choices=("smoke", "formal"), required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        preflight()
        return 0
    return run_training(args.variant, args.seed, args.phase)


if __name__ == "__main__":
    raise SystemExit(main())
