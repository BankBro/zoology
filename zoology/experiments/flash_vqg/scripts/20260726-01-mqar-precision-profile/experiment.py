#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

LOCAL_SCRIPT_DIR = Path(__file__).resolve().parent
if str(LOCAL_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_SCRIPT_DIR))

os.environ.setdefault("TRITON_F32_DEFAULT", "ieee")
os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
os.environ.setdefault("TORCH_DETERMINISTIC", "0")

import torch

from common import (
    EXPERIMENT_ID,
    FLASH_ROOT,
    GDN_KERNEL_DTYPE,
    MACHINES,
    PRECISION_CONFIG,
    PYTHON,
    REPO_ROOT,
    SCRIPT_DIR,
    atomic_write_json,
    machine_name,
    output_root,
    sha256_file,
    stable_json_sha256,
    training_descriptors,
    utc_now,
)


BASE_SCRIPT = (
    REPO_ROOT
    / "zoology/experiments/flash_vqg/scripts/"
    / "20260724-01-flash-vqg-gd-residual-efficiency/efficiency_benchmark.py"
)
GENERATED_ROOT = REPO_ROOT / "zoology/experiments/flash_vqg/generated"
EXPECTED_CACHE_HASH = "d9098e876a036b8cb90a7186174fd827e0f5b422482266772850069c905bd8c8"
EXPECTED_INIT = {
    "flash": {
        "file": "26bf2cb0b44c8a32cfd44977f609e4314ab53dfa6b6a0e8abfbffd10191ec878",
        "state": "2a1107bf22d0804ed485ab94bdc7af8004ef7b892a60c2967f842ba0f4b4efb0",
        "params": 1_160_390,
    },
    "gdn": {
        "file": "a4e76e7776bdc83a582c2613cd7d9782100a9148aa119763ecaaeeb8273f7b71",
        "state": "bdba0c19b2530c72c3ae7dd6bd708901c2369f6d3e1da9d850ea8347d5ea60a6",
        "params": 1_335_942,
    },
}
SMOKE_TRAIN_SEGMENT_ORDER = (0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 4, 4)
SMOKE_VALID_SEGMENT_ORDER = tuple(
    segment for segment in range(8) for _ in range(3)
)
FLASH_RUNTIME_MODULE = "backbone.layers.1.sequence_mixer.mixer.attn"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_module(BASE_SCRIPT, "mqar_precision_efficiency_base")


def git_value(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        text=True,
        capture_output=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def state_dict_hash(state_dict: dict[str, torch.Tensor]) -> str:
    return BASE._state_dict_hash(state_dict)


def init_path(model: str) -> Path:
    value = BASE.CANONICAL_INIT if model == "flash" else BASE.GDN_CANONICAL_INIT
    return Path(value).resolve()


def descriptor_id(model: str, seed: int, precision: str) -> str:
    return f"{machine_name()}-{model}-s{seed}-{precision}"


def launch_id(phase: str) -> str:
    return f"{EXPERIMENT_ID}-{machine_name()}-{phase}"


def run_id(model: str, seed: int, precision: str, phase: str) -> str:
    return f"{model}-s{seed}-{precision}-b64ga4-{phase}"


def checkpoint_root(phase: str) -> Path:
    return output_root() / "checkpoints" / phase


def checkpoint_run_dir(config: Any) -> Path:
    return (
        Path(config.checkpoint.root_dir)
        / str(config.launch_id)
        / str(config.run_id)
    )


def configure_numerics(precision: str) -> None:
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if os.environ.get("TRITON_F32_DEFAULT") != "ieee":
        raise RuntimeError("TRITON_F32_DEFAULT must be ieee.")
    if os.environ.get("NVIDIA_TF32_OVERRIDE") != "0":
        raise RuntimeError("NVIDIA_TF32_OVERRIDE must be 0.")
    expected_gdn = GDN_KERNEL_DTYPE[precision]
    if os.environ.get("GDN_KERNEL_DTYPE") != expected_gdn:
        raise RuntimeError(
            f"GDN_KERNEL_DTYPE must be {expected_gdn} for {precision}."
        )


def _source_identity(model: str, precision: str) -> dict[str, str]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "machine": machine_name(),
        "model": model,
        "precision": precision,
        "zoology_commit": git_value(REPO_ROOT, "rev-parse", "HEAD"),
        "flash_commit": git_value(FLASH_ROOT, "rev-parse", "HEAD"),
        "zoology_branch": git_value(REPO_ROOT, "branch", "--show-current"),
        "flash_branch": git_value(FLASH_ROOT, "branch", "--show-current"),
        "cache_content_sha256": EXPECTED_CACHE_HASH,
        "init_file_sha256": sha256_file(init_path(model)),
    }


def build_config(model: str, seed: int, precision: str, phase: str):
    machine = machine_name()
    if model not in {"flash", "gdn"}:
        raise ValueError(f"Unsupported model: {model}")
    if precision not in MACHINES[machine]["train_precisions"]:
        raise ValueError(f"Unsupported precision on {machine}: {precision}")
    if seed not in {123, 124, 125}:
        raise ValueError(f"Unsupported seed: {seed}")
    if phase not in {"smoke", "stress", "formal"}:
        raise ValueError(f"Unsupported phase: {phase}")

    flash = BASE._build_flash_config("core", "triton", "triton_remat")
    gdn = BASE._build_gdn_config(flash.data)
    config = flash if model == "flash" else gdn
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
    config.precision = PRECISION_CONFIG[precision]
    config.resume_enabled = True
    config.resume_stop_after_optimizer_step = 1 if phase == "smoke" else None
    config.max_grad_scaler_skips = 2 if precision == "fp16" else 0
    config.max_consecutive_grad_scaler_skips = 2 if precision == "fp16" else 0
    config.resume_identity = _source_identity(model, precision)
    config.training_runtime_initial_state = {}
    if phase != "formal":
        config.data.train_batch_segment_order = list(SMOKE_TRAIN_SEGMENT_ORDER)
        config.data.test_batch_segment_order = list(SMOKE_VALID_SEGMENT_ORDER)
    else:
        config.data.train_batch_segment_order = None
        config.data.test_batch_segment_order = None
    if phase == "stress":
        if model != "flash":
            raise ValueError("Stress smoke is Flash-only.")
        config.training_runtime_initial_state = {
            FLASH_RUNTIME_MODULE: {
                "fox_gd_residual_train_forward_count": 2048,
            }
        }

    config.metrics_white_list = []
    config.logger.backend = "none"
    config.checkpoint.enabled = True
    config.checkpoint.save_best = True
    config.checkpoint.save_last = True
    config.checkpoint.best_metric = "valid/accuracy"
    config.checkpoint.best_mode = "max"
    config.checkpoint.root_dir = str(checkpoint_root(phase))
    config.init_checkpoint_path = str(init_path(model))
    config.init_checkpoint_strict = True
    config.launch_id = launch_id(phase)
    config.sweep_id = EXPERIMENT_ID
    config.run_id = run_id(model, seed, precision, phase)
    config.training_telemetry_path = str(
        output_root()
        / "training"
        / phase
        / config.run_id
        / "telemetry.jsonl"
    )
    if model == "flash":
        kwargs = BASE._find_flash_kwargs(config.model)
        kwargs["fox_gd_residual_triton_input_policy"] = "fp32_boundary"
    return config


def resolved_config_path(config: Any) -> Path:
    return GENERATED_ROOT / str(config.launch_id) / f"{config.run_id}.json"


def write_resolved_config(config: Any) -> Path:
    from zoology.checkpoints import serialize_train_config

    path = resolved_config_path(config)
    atomic_write_json(path, serialize_train_config(config))
    return path


def normalized_config_hash(config: Any) -> str:
    from zoology.checkpoints import serialize_train_config

    payload = serialize_train_config(config)
    payload["run_id"] = "<run>"
    payload["launch_id"] = "<launch>"
    payload["checkpoint"]["root_dir"] = "<checkpoint>"
    payload["resume_path"] = None
    payload["training_telemetry_path"] = "<telemetry>"
    if "resume_identity" in payload:
        payload["resume_identity"]["machine"] = "<machine>"
    return stable_json_sha256(payload)


def _model_audit(config: Any, model_name: str) -> dict[str, Any]:
    model = BASE.LanguageModel(config.model)
    payload = torch.load(init_path(model_name), map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(payload["model_state_dict"], strict=True)
    runtime_modules = [
        name
        for name, module in model.named_modules()
        if getattr(module, "get_training_runtime_state", None) is not None
    ]
    kwargs = BASE._find_flash_kwargs(config.model) if model_name == "flash" else {}
    return {
        "trainable_params": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "state_sha256": state_dict_hash(model.state_dict()),
        "strict": not incompatible.missing_keys and not incompatible.unexpected_keys,
        "runtime_modules": runtime_modules,
        "flash_boundary": kwargs.get("fox_gd_residual_triton_input_policy"),
        "flash_grouped_backend": kwargs.get("fox_gd_residual_grouped_chunk_backend"),
        "flash_read_backend": kwargs.get("fox_gd_residual_selected_read_backend"),
    }


def config_audit(config: Any, descriptor: dict[str, Any], phase: str) -> dict[str, Any]:
    model = descriptor["model"]
    precision = descriptor["train_precision"]
    audit = _model_audit(config, model)
    checks = {
        "batch": tuple(config.data.batch_size) == (64, 16),
        "gradient_accumulation": config.gradient_accumulation_steps == 4,
        "epochs": config.max_epochs == (4 if phase == "formal" else 1),
        "validations": config.validations_per_epoch == (4 if phase == "formal" else 3),
        "early_stopping_off": config.early_stopping_metric is None,
        "precision": config.precision == PRECISION_CONFIG[precision],
        "resume": config.resume_enabled,
        "init_file": sha256_file(init_path(model)) == EXPECTED_INIT[model]["file"],
        "init_state": audit["state_sha256"] == EXPECTED_INIT[model]["state"],
        "params": audit["trainable_params"] == EXPECTED_INIT[model]["params"],
        "strict": audit["strict"],
    }
    if model == "flash":
        checks.update(
            {
                "boundary": audit["flash_boundary"] == "fp32_boundary",
                "grouped": audit["flash_grouped_backend"] == "triton",
                "selected": audit["flash_read_backend"] == "triton_remat",
                "runtime_module": FLASH_RUNTIME_MODULE in audit["runtime_modules"],
            }
        )
    return {
        **descriptor,
        "phase": phase,
        "run_id": config.run_id,
        "normalized_config_sha256": normalized_config_hash(config),
        "checks": checks,
        "passed": all(checks.values()),
        "model_audit": audit,
    }


def environment_metadata() -> dict[str, Any]:
    import fla
    import triton

    return {
        "sys_executable": sys.executable,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton": triton.__version__,
        "fla": fla.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "zoology_commit": git_value(REPO_ROOT, "rev-parse", "HEAD"),
        "flash_commit": git_value(FLASH_ROOT, "rev-parse", "HEAD"),
        "zoology_status": git_value(REPO_ROOT, "status", "--short"),
        "flash_status": git_value(FLASH_ROOT, "status", "--short"),
    }


def preflight() -> dict[str, Any]:
    machine = machine_name()
    os.environ["GDN_KERNEL_DTYPE"] = "float32"
    configure_numerics("fp32")
    env = environment_metadata()
    base_config = build_config("flash", 123, "fp32", "formal")
    cache = BASE._cache_content_hash(base_config.data)
    jobs = []
    for descriptor in training_descriptors(machine):
        for phase in ("smoke", "formal"):
            config = build_config(
                descriptor["model"],
                descriptor["seed"],
                descriptor["train_precision"],
                phase,
            )
            write_resolved_config(config)
            jobs.append(config_audit(config, descriptor, phase))
        if descriptor["model"] == "flash":
            stress = build_config(
                "flash",
                descriptor["seed"],
                descriptor["train_precision"],
                "stress",
            )
            write_resolved_config(stress)
            jobs.append(config_audit(stress, descriptor, "stress"))
    allowed_zoology_status = [
        line for line in env["zoology_status"].splitlines() if line != "?? sources/"
    ]
    checks = {
        "python": Path(sys.executable).resolve() == PYTHON.resolve(),
        "python_version": env["python"] == "3.12.11",
        "torch": env["torch"] == "2.6.0+cu118",
        "cuda": env["torch_cuda"] == "11.8",
        "triton": env["triton"] == "3.2.0",
        "fla": env["fla"] == "0.4.2",
        "cuda_available": env["cuda_available"],
        "gpu": env["gpu_name"] == MACHINES[machine]["gpu_name"],
        "visible_gpu": env["cuda_visible_devices"] == MACHINES[machine]["visible_gpu"],
        "cache": cache.get("combined_content_sha256") == EXPECTED_CACHE_HASH,
        "zoology_clean": not allowed_zoology_status,
        "flash_clean": not env["flash_status"],
        "zoology_branch": env["zoology_branch"] == "flash-vqg",
        "flash_branch": (
            env["flash_branch"] == "20260428-gd-residual-v1-sync"
        ),
        "jobs": all(row["passed"] for row in jobs),
    }
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "machine": machine,
        "status": "passed" if all(checks.values()) else "failed",
        "recorded_at_utc": utc_now(),
        "environment": env,
        "cache": cache,
        "checks": checks,
        "jobs": jobs,
    }
    path = output_root() / "preflight.json"
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
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def telemetry_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path.resolve()), "records": 0}
    by_step: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if "train/optimizer_step_wall_sec" in row:
                by_step[int(row["log_step"])] = row
    successful = [
        row
        for row in by_step.values()
        if int(row.get("train/optimizer_step_skipped", 0)) == 0
    ]
    stable = successful[1:]
    times = [float(row["train/optimizer_step_wall_sec"]) for row in stable]
    return {
        "path": str(path.resolve()),
        "records": len(by_step),
        "successful_records": len(successful),
        "stable_records_excluding_first": len(stable),
        "optimizer_step_wall_sec_p50": percentile(times, 0.50),
        "optimizer_step_wall_sec_p90": percentile(times, 0.90),
        "peak_allocated_mib": max(
            (float(row.get("train/peak_allocated_mib", 0.0)) for row in by_step.values()),
            default=0.0,
        ),
        "peak_reserved_mib": max(
            (float(row.get("train/peak_reserved_mib", 0.0)) for row in by_step.values()),
            default=0.0,
        ),
    }


def train_result_path(model: str, seed: int, precision: str, phase: str) -> Path:
    return (
        output_root()
        / "training"
        / phase
        / run_id(model, seed, precision, phase)
        / "result.json"
    )


def run_training(model: str, seed: int, precision: str, phase: str) -> int:
    os.environ["GDN_KERNEL_DTYPE"] = GDN_KERNEL_DTYPE[precision]
    configure_numerics(precision)
    config = build_config(model, seed, precision, phase)
    resolved = write_resolved_config(config)
    out = train_result_path(model, seed, precision, phase)
    checkpoint_dir = checkpoint_run_dir(config)
    started_at = utc_now()
    started = time.perf_counter()
    try:
        from zoology.train import TrainingInterrupted, train

        train(config)
        status = "completed"
        error = None
        return_code = 0
    except TrainingInterrupted as exc:
        status = "controlled_stop"
        error = f"{type(exc).__name__}: {exc}"
        return_code = 75
    except BaseException as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        return_code = 1
        traceback.print_exc()
    result: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "machine": machine_name(),
        "model": model,
        "seed": seed,
        "train_precision": precision,
        "gdn_kernel_dtype": GDN_KERNEL_DTYPE[precision],
        "phase": phase,
        "status": status,
        "error": error,
        "started_at_utc": started_at,
        "ended_at_utc": utc_now(),
        "wall_clock_sec": time.perf_counter() - started,
        "resolved_config_path": str(resolved.resolve()),
        "resolved_config_sha256": sha256_file(resolved),
        "normalized_config_sha256": normalized_config_hash(config),
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "resume_path": str((checkpoint_dir / "resume.pt").resolve()),
        "telemetry": telemetry_summary(Path(config.training_telemetry_path)),
    }
    if status == "completed":
        expected_epoch = 4 if phase == "formal" else 1
        for role in ("last", "best"):
            result[f"{role}_checkpoint"] = checkpoint_metadata(
                checkpoint_dir / f"{role}.pt"
            )
        if result["last_checkpoint"]["epoch"] != expected_epoch:
            status = "failed"
            result["status"] = status
            result["error"] = "Unexpected last checkpoint epoch."
            return_code = 1
        resume_payload = torch.load(
            checkpoint_dir / "resume.pt",
            map_location="cpu",
            weights_only=False,
        )
        optimizer_dtypes = sorted(
            {
                str(value.dtype)
                for state in resume_payload["optimizer_state_dict"]["state"].values()
                for value in state.values()
                if torch.is_tensor(value) and value.is_floating_point()
            }
        )
        result["resume_audit"] = {
            "epoch_idx": resume_payload["epoch_idx"],
            "next_train_batch_idx": resume_payload["next_train_batch_idx"],
            "optimizer_step": resume_payload["optimizer_step"],
            "optimizer_attempt_step": resume_payload["optimizer_attempt_step"],
            "grad_scaler_skips": resume_payload["grad_scaler_skips"],
            "runtime_state": resume_payload["model_runtime_state"],
            "model_state_dtypes": sorted(
                {
                    str(value.dtype)
                    for value in resume_payload["model_state_dict"].values()
                    if torch.is_tensor(value) and value.is_floating_point()
                }
            ),
            "optimizer_state_dtypes": optimizer_dtypes,
            "grad_scaler_enabled": bool(
                resume_payload["grad_scaler_state_dict"]
            ),
        }
        result["telemetry"] = telemetry_summary(
            Path(config.training_telemetry_path)
        )
    atomic_write_json(out, result)
    print(json.dumps({"status": status, "result": str(out)}, ensure_ascii=False))
    return return_code


def write_matrix_manifest() -> Path:
    machine = machine_name()
    rows = training_descriptors(machine)
    path = output_root() / "manifests" / "training-matrix.json"
    atomic_write_json(path, rows)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    sub.add_parser("write-matrix")
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--model", choices=("flash", "gdn"), required=True)
    train_parser.add_argument("--seed", type=int, required=True)
    train_parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), required=True)
    train_parser.add_argument("--phase", choices=("smoke", "stress", "formal"), required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        preflight()
        return 0
    if args.command == "write-matrix":
        print(write_matrix_manifest())
        return 0
    return run_training(args.model, args.seed, args.precision, args.phase)


if __name__ == "__main__":
    raise SystemExit(main())
