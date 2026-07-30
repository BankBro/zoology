#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
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
os.environ.setdefault("GDN_KERNEL_DTYPE", "float32")

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
    SEED,
    VARIANTS,
    atomic_write_json,
    generated_root,
    run_root,
    run_tag,
    sha256_file,
    stable_json_sha256,
    utc_now,
)


BASE_SCRIPT = (
    REPO_ROOT
    / "zoology/experiments/flash_vqg/scripts"
    / "20260724-01-flash-vqg-gd-residual-efficiency/efficiency_benchmark.py"
)


def load_base():
    spec = importlib.util.spec_from_file_location("block64_remat_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load base module: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_base()


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


def nested_state_hash(value: Any) -> str:
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


def run_id(variant: str, phase: str) -> str:
    return f"{variant}-s{SEED}-fp32-b64ga4-{phase}"


def result_path(variant: str, phase: str) -> Path:
    return run_root() / "training" / phase / run_id(variant, phase) / "result.json"


def source_identity(variant: str) -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "run_tag": run_tag(),
        "variant": variant,
        "remat_mode": VARIANTS[variant],
        "seed": str(SEED),
        "precision": "fp32",
        "zoology_commit": git_value(REPO_ROOT, "rev-parse", "HEAD"),
        "flash_commit": git_value(FLASH_ROOT, "rev-parse", "HEAD"),
        "cache_content_sha256": EXPECTED_CACHE_HASH,
        "init_file_sha256": sha256_file(init_path()),
    }


def build_config(variant: str, phase: str):
    if variant not in VARIANTS or phase not in {"smoke", "screen"}:
        raise ValueError(f"Unsupported job: {variant}, {phase}.")
    config = BASE._build_flash_config("core", "triton", "triton_remat")
    config.seed = SEED
    config.data.seed = 123
    config.data.batch_size = (64, 16)
    config.gradient_accumulation_steps = 4
    config.validations_per_epoch = 3 if phase == "smoke" else 4
    config.max_epochs = 1
    config.max_train_steps = 3 if phase == "smoke" else None
    config.max_validation_batches = None
    config.early_stopping_metric = None
    config.early_stopping_threshold = None
    config.precision = "float32"
    config.resume_enabled = True
    config.resume_stop_after_optimizer_step = None
    config.resume_identity = source_identity(variant)
    config.training_runtime_initial_state = {}
    config.metrics_white_list = []
    config.logger.backend = "none"
    config.checkpoint.enabled = True
    config.checkpoint.save_best = True
    config.checkpoint.save_last = True
    config.checkpoint.best_metric = "valid/accuracy"
    config.checkpoint.best_mode = "max"
    config.checkpoint.root_dir = str(run_root() / "checkpoints" / phase)
    config.init_checkpoint_path = str(init_path())
    config.init_checkpoint_strict = True
    config.launch_id = f"{EXPERIMENT_ID}-{run_tag()}-{phase}"
    config.sweep_id = EXPERIMENT_ID
    config.run_id = run_id(variant, phase)
    config.training_telemetry_path = str(result_path(variant, phase).with_name("telemetry.jsonl"))
    kwargs = BASE._find_flash_kwargs(config.model)
    kwargs.update(
        {
            "block_len": 64,
            "fox_gd_residual_write_topk": 4,
            "fox_remote_read_topk": 16,
            "fox_gd_residual_remat_mode": VARIANTS[variant],
            "fox_gd_residual_selected_read_backward_backend": "triton_deterministic",
            "fox_gd_residual_triton_input_policy": "fp32_boundary",
        }
    )
    return config


def serialize_config(config: Any) -> dict[str, Any]:
    from zoology.checkpoints import serialize_train_config

    return serialize_train_config(config)


def write_config(config: Any) -> Path:
    path = generated_root() / f"{config.run_id}.json"
    atomic_write_json(path, serialize_config(config))
    return path


def normalized_config(config: Any) -> dict[str, Any]:
    payload = serialize_config(config)
    payload["run_id"] = "<run>"
    payload["launch_id"] = "<launch>"
    payload["checkpoint"]["root_dir"] = "<checkpoint>"
    payload["training_telemetry_path"] = "<telemetry>"
    payload["resume_identity"]["variant"] = "<variant>"
    payload["resume_identity"]["remat_mode"] = "<remat>"
    return payload


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten(child, name))
        return result
    if isinstance(value, list):
        result = {}
        for index, child in enumerate(value):
            result.update(flatten(child, f"{prefix}[{index}]"))
        return result
    return {prefix: value}


def config_differences(left: Any, right: Any) -> list[str]:
    left_flat = flatten(normalized_config(left))
    right_flat = flatten(normalized_config(right))
    return sorted(
        key
        for key in set(left_flat) | set(right_flat)
        if left_flat.get(key) != right_flat.get(key)
    )


def configure_numerics() -> None:
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if os.environ.get("TRITON_F32_DEFAULT") != "ieee":
        raise RuntimeError("TRITON_F32_DEFAULT must be ieee.")
    if os.environ.get("NVIDIA_TF32_OVERRIDE") != "0":
        raise RuntimeError("NVIDIA_TF32_OVERRIDE must be 0.")


def model_audit(config: Any) -> dict[str, Any]:
    model = BASE.LanguageModel(config.model)
    payload = torch.load(init_path(), map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(payload["model_state_dict"], strict=True)
    kwargs = BASE._find_flash_kwargs(config.model)
    return {
        "parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "state_sha256": state_dict_hash(model.state_dict()),
        "strict": not incompatible.missing_keys and not incompatible.unexpected_keys,
        "block_len": kwargs["block_len"],
        "write_topk": kwargs["fox_gd_residual_write_topk"],
        "read_topk": kwargs["fox_remote_read_topk"],
        "remat_mode": kwargs["fox_gd_residual_remat_mode"],
        "selected_backward": kwargs["fox_gd_residual_selected_read_backward_backend"],
        "grouped_backend": kwargs["fox_gd_residual_grouped_chunk_backend"],
    }


def environment() -> dict[str, Any]:
    import fla
    import triton

    available = torch.cuda.is_available()
    return {
        "python_executable": sys.executable,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "fla": fla.__version__,
        "cuda_available": available,
        "gpu": torch.cuda.get_device_name(0) if available else None,
        "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_used_bytes": int(torch.cuda.device_memory_used()) if available else None,
        "zoology_commit": git_value(REPO_ROOT, "rev-parse", "HEAD"),
        "flash_commit": git_value(FLASH_ROOT, "rev-parse", "HEAD"),
        "zoology_branch": git_value(REPO_ROOT, "branch", "--show-current"),
        "zoology_status": git_value(REPO_ROOT, "status", "--short"),
        "flash_status": git_value(FLASH_ROOT, "status", "--short"),
    }


def preflight() -> dict[str, Any]:
    configure_numerics()
    env = environment()
    configs = {name: build_config(name, "screen") for name in VARIANTS}
    audits = {name: model_audit(config) for name, config in configs.items()}
    for config in configs.values():
        write_config(config)
    cache = BASE._cache_content_hash(configs["a0-block64"].data)
    differences = config_differences(configs["a0-block64"], configs["a1-block64"])
    checks = {
        "python": Path(sys.executable).resolve() == PYTHON.resolve(),
        "versions": (env["python"], env["torch"], env["cuda"], env["triton"], env["fla"])
        == ("3.12.11", "2.6.0+cu118", "11.8", "3.2.0", "0.4.2"),
        "gpu": env["cuda_available"] and env["gpu"] == "NVIDIA GeForce RTX 2080 Ti",
        "visible_gpu": env["visible_gpu"] == "1",
        "gpu_free": env["gpu_used_bytes"] is not None and env["gpu_used_bytes"] < 1024**3,
        "flash_commit": env["flash_commit"] == EXPECTED_FLASH_COMMIT,
        "source_clean": not env["zoology_status"] and not env["flash_status"],
        "branch": env["zoology_branch"] == "20260730-144531-a1-block64-remat-quality-canary",
        "cache": cache["combined_content_sha256"] == EXPECTED_CACHE_HASH,
        "init_file": sha256_file(init_path()) == EXPECTED_INIT_FILE_HASH,
        "audits": all(
            row["parameters"] == EXPECTED_PARAMETERS
            and row["state_sha256"] == EXPECTED_INIT_STATE_HASH
            and row["strict"]
            and row["block_len"] == 64
            and row["write_topk"] == 4
            and row["read_topk"] == 16
            and row["selected_backward"] == "triton_deterministic"
            and row["grouped_backend"] == "triton"
            and row["remat_mode"] == VARIANTS[name]
            for name, row in audits.items()
        ),
        "single_variable": len(differences) == 1
        and differences[0].endswith("fox_gd_residual_remat_mode"),
    }
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "run_tag": run_tag(),
        "status": "passed" if all(checks.values()) else "failed",
        "recorded_at_utc": utc_now(),
        "environment": env,
        "cache": cache,
        "audits": audits,
        "config_differences": differences,
        "checks": checks,
    }
    atomic_write_json(run_root() / "preflight.json", payload)
    if payload["status"] != "passed":
        raise RuntimeError(f"Preflight failed: {checks}")
    return payload


def checkpoint_dir(config: Any) -> Path:
    return Path(config.checkpoint.root_dir) / str(config.launch_id) / str(config.run_id)


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metrics = payload.get("metrics") or {}
    return {
        "path": str(path.resolve()),
        "file_sha256": sha256_file(path),
        "model_state_sha256": state_dict_hash(payload["model_state_dict"]),
        "epoch": int(payload.get("epoch", -1)) + 1,
        "metrics": metrics,
        "finite": all(math.isfinite(float(value)) for value in metrics.values()),
    }


def telemetry(path: Path) -> dict[str, Any]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if "train/loss" not in row:
            continue
        rows.append(
            {
                "step": int(row["log_step"]),
                "loss": float(row["train/loss"]),
                "wall_seconds": float(row.get("train/optimizer_step_wall_sec", 0.0)),
                "skipped": int(row.get("train/optimizer_step_skipped", 0)),
                "peak_allocated_mib": float(row.get("train/peak_allocated_mib", 0.0)),
                "peak_reserved_mib": float(row.get("train/peak_reserved_mib", 0.0)),
            }
        )
    return {"path": str(path.resolve()), "records": rows}


def runtime_audit(states: dict[str, dict[str, Any]], remat_mode: str) -> dict[str, Any]:
    audits = [
        state.get("fox_gd_residual_triton_runtime_audit")
        for state in states.values()
        if state.get("fox_gd_residual_triton_runtime_audit") is not None
    ]
    logical = sum(int(audit.get("selected_calls", 0)) for audit in audits)
    recompute = sum(int(audit.get("selected_recompute_calls", 0)) for audit in audits)
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
    passed = bool(audits) and logical > 0 and fallbacks == 0 and (
        (recompute > 0) == (remat_mode == "post_phase1")
    )
    return {
        "passed": passed,
        "modules": len(audits),
        "logical_selected_calls": logical,
        "selected_recompute_calls": recompute,
        "fallbacks": fallbacks,
    }


def triton_config_record(config: Any) -> dict[str, Any]:
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
            "best_config": None if best is None else triton_config_record(best),
            "cache": [
                {"key": repr(key), "config": triton_config_record(config)}
                for key, config in autotuner.cache.items()
            ],
        }
    return result


def run_training(variant: str, phase: str) -> int:
    configure_numerics()
    config = build_config(variant, phase)
    resolved = write_config(config)
    output = result_path(variant, phase)
    started_at, started = utc_now(), time.perf_counter()
    try:
        from zoology.train import train

        train(config)
        status, error, code = "completed", None, 0
    except BaseException as exc:
        status, error, code = "failed", f"{type(exc).__name__}: {exc}", 1
        traceback.print_exc()
    result: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "run_tag": run_tag(),
        "variant": variant,
        "remat_mode": VARIANTS[variant],
        "phase": phase,
        "seed": SEED,
        "precision": "fp32",
        "status": status,
        "error": error,
        "started_at_utc": started_at,
        "ended_at_utc": utc_now(),
        "wall_clock_sec": time.perf_counter() - started,
        "resolved_config": str(resolved.resolve()),
        "resolved_config_sha256": sha256_file(resolved),
        "normalized_config_sha256": stable_json_sha256(serialize_config(config)),
    }
    if status == "completed":
        root = checkpoint_dir(config)
        result["last_checkpoint"] = checkpoint_metadata(root / "last.pt")
        result["best_checkpoint"] = checkpoint_metadata(root / "best.pt")
        resume = torch.load(root / "resume.pt", map_location="cpu", weights_only=False)
        result["resume"] = {
            "path": str((root / "resume.pt").resolve()),
            "optimizer_step": int(resume["optimizer_step"]),
            "model_state_sha256": nested_state_hash(resume["model_state_dict"]),
            "optimizer_state_sha256": nested_state_hash(resume["optimizer_state_dict"]),
            "runtime_state": resume["model_runtime_state"],
        }
        result["runtime_audit"] = runtime_audit(
            result["resume"]["runtime_state"],
            VARIANTS[variant],
        )
        result["telemetry"] = telemetry(Path(config.training_telemetry_path))
        result["gate_autotune"] = gate_autotune_snapshot()
        valid = (
            result["last_checkpoint"]["epoch"] == 1
            and result["last_checkpoint"]["finite"]
            and result["runtime_audit"]["passed"]
            and all(row["skipped"] == 0 for row in result["telemetry"]["records"])
        )
        if not valid:
            result["status"], result["error"], code = (
                "failed",
                "Completed training failed checkpoint, telemetry or runtime audit.",
                1,
            )
    atomic_write_json(output, result)
    print(json.dumps({"status": result["status"], "result": str(output)}))
    return code


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    train = sub.add_parser("train")
    train.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    train.add_argument("--phase", choices=("smoke", "screen"), required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        preflight()
        return 0
    return run_training(args.variant, args.phase)


if __name__ == "__main__":
    raise SystemExit(main())
