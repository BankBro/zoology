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
os.environ.setdefault("GDN_KERNEL_DTYPE", "bfloat16")

import torch

from causal_common import (
    ARMS,
    ARM_SPECS,
    EXPECTED_CACHE_HASH,
    EXPECTED_FLASH_COMMIT,
    EXPECTED_INIT_FILE_HASH,
    EXPECTED_INIT_STATE_HASH,
    EXPECTED_PARAMETERS,
    EXPERIMENT_ID,
    FLASH_ROOT,
    GATE_MODES,
    PHASES,
    PYTHON,
    REPO_ROOT,
    SEEDS,
    ZOOLOGY_BASE_COMMIT,
    arm_spec,
    atomic_write_json,
    descriptor,
    generated_root,
    run_root,
    run_tag,
    sha256_file,
    stable_json_sha256,
    utc_now,
)


UPSTREAM_DIR = (
    REPO_ROOT
    / "zoology/experiments/flash_vqg/scripts"
    / "20260801-01-fastest-flash-vs-gdn-mqar"
)
SMOKE_TRAIN_ORDER = (0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 4, 4)
SMOKE_VALID_ORDER = tuple(segment for segment in range(8) for _ in range(3))


def _load_upstream():
    path = UPSTREAM_DIR / "experiment.py"
    spec = importlib.util.spec_from_file_location("late_degradation_upstream", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load upstream experiment: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


UPSTREAM = _load_upstream()
BASE = UPSTREAM.BASE


def git_value(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        text=True,
        capture_output=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def git_is_ancestor(root: Path, ancestor: str, descendant: str = "HEAD") -> bool:
    result = subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", ancestor, descendant],
        capture_output=True,
    )
    return result.returncode == 0


def init_path() -> Path:
    return Path(BASE.CANONICAL_INIT).resolve()


def run_id(arm: str, seed: int, phase: str, gate_mode: str) -> str:
    return f"{arm}-s{seed}-bf16-b64ga4-{gate_mode}-{phase}"


def result_path(arm: str, seed: int, phase: str, gate_mode: str = "fixed") -> Path:
    return run_root() / "training" / phase / run_id(arm, seed, phase, gate_mode) / "result.json"


def checkpoint_root(phase: str, gate_mode: str) -> Path:
    return run_root() / "checkpoints" / phase / gate_mode


def checkpoint_run_dir(config: Any) -> Path:
    return Path(config.checkpoint.root_dir) / str(config.launch_id) / str(config.run_id)


def source_identity(arm: str, gate_mode: str) -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "run_tag": run_tag(),
        "arm": arm,
        "arm_spec": arm_spec(arm),
        "gate_mode": gate_mode,
        "zoology_commit": git_value(REPO_ROOT, "rev-parse", "HEAD"),
        "flash_commit": git_value(FLASH_ROOT, "rev-parse", "HEAD"),
        "cache_content_sha256": EXPECTED_CACHE_HASH,
        "init_file_sha256": sha256_file(init_path()),
    }


def _base_config(arm: str):
    family = arm_spec(arm)["family"]
    upstream_arm = UPSTREAM.FASTEST if family == "fastest" else UPSTREAM.CANONICAL
    config = UPSTREAM._base_config(upstream_arm)
    kwargs = BASE._find_flash_kwargs(config.model)
    spec = arm_spec(arm)
    kwargs.update(
        {
            "block_len": spec["block_len"],
            "local_num_blocks": spec["local_num_blocks"],
            "fox_gd_residual_selected_read_backward_backend": spec["selected_backward"],
            "fox_gd_residual_selected_read_chunk_size": spec["selected_chunk"],
        }
    )
    return config


def _configure_training(
    config: Any,
    arm: str,
    seed: int,
    phase: str,
    gate_mode: str,
) -> None:
    config.seed = int(seed)
    config.data.seed = 123
    config.data.batch_size = (64, 16)
    config.gradient_accumulation_steps = 4
    config.validations_per_epoch = 3 if phase == "smoke" else 4
    config.max_epochs = 1 if phase == "smoke" else 4
    config.max_train_steps = 3 if phase == "smoke" else (1232 if phase == "screen" else None)
    config.max_validation_batches = None
    config.early_stopping_metric = None
    config.early_stopping_threshold = None
    config.precision = "amp_bfloat16"
    config.resume_enabled = True
    config.resume_stop_after_optimizer_step = None
    config.max_grad_scaler_skips = 0
    config.max_consecutive_grad_scaler_skips = 0
    config.resume_identity = source_identity(arm, gate_mode)
    config.training_runtime_initial_state = {}


def _configure_io(
    config: Any,
    arm: str,
    seed: int,
    phase: str,
    gate_mode: str,
) -> None:
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
    config.checkpoint.root_dir = str(checkpoint_root(phase, gate_mode))
    config.init_checkpoint_path = str(init_path())
    config.init_checkpoint_strict = True
    config.launch_id = f"{EXPERIMENT_ID}-{run_tag()}-{gate_mode}-{phase}"
    config.sweep_id = EXPERIMENT_ID
    config.run_id = run_id(arm, seed, phase, gate_mode)
    config.training_telemetry_path = str(
        result_path(arm, seed, phase, gate_mode).parent / "telemetry.jsonl"
    )


def build_config(arm: str, seed: int, phase: str, gate_mode: str = "fixed"):
    descriptor(arm, seed, gate_mode)
    if phase not in PHASES:
        raise ValueError(f"Unsupported phase: {phase}.")
    config = _base_config(arm)
    _configure_training(config, arm, seed, phase, gate_mode)
    _configure_io(config, arm, seed, phase, gate_mode)
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
    expected = {"TRITON_F32_DEFAULT": "ieee", "NVIDIA_TF32_OVERRIDE": "0"}
    for key, value in expected.items():
        if os.environ.get(key) != value:
            raise RuntimeError(f"{key} must be {value}.")


def configure_gate_bwd_runtime(gate_mode: str) -> None:
    if gate_mode == "default":
        return
    if gate_mode != "fixed":
        raise ValueError(f"Unsupported gate mode: {gate_mode}.")
    import triton
    from fla.modules import fused_norm_gate

    autotuner = fused_norm_gate.layer_norm_gated_bwd_kernel.fn
    autotuner.configs = [
        triton.Config({"BT": 64}, num_warps=4, num_stages=2)
    ]
    autotuner.cache.clear()


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


def scientific_contract(config: Any) -> dict[str, Any]:
    kwargs = BASE._find_flash_kwargs(config.model)
    keys = (
        "block_len",
        "local_num_blocks",
        "fox_gd_residual_remat_mode",
        "fox_gd_residual_builder",
        "fox_gd_residual_selected_read_backward_backend",
        "fox_gd_residual_selected_read_chunk_size",
        "fox_gd_residual_persistent_backward_backend",
        "fox_gd_residual_geometry_backend",
        "fox_gd_residual_selected_read_forward_backend",
        "fox_gd_residual_triton_input_policy",
        "fox_gd_residual_selected_read_input_policy",
    )
    return {key: kwargs.get(key) for key in keys}


def _contract_checks(arm: str, contract: dict[str, Any]) -> dict[str, bool]:
    spec = arm_spec(arm)
    family = spec["family"]
    return {
        "block": contract["block_len"] == spec["block_len"],
        "local": contract["local_num_blocks"] == spec["local_num_blocks"],
        "backward": contract["fox_gd_residual_selected_read_backward_backend"] == spec["selected_backward"],
        "chunk": contract["fox_gd_residual_selected_read_chunk_size"] == spec["selected_chunk"],
        "remat": contract["fox_gd_residual_remat_mode"] == "post_phase1",
        "builder": contract["fox_gd_residual_builder"] == ("persistent_scan_triton" if family == "fastest" else "grouped_chunk_torch_ref"),
        "input": contract["fox_gd_residual_triton_input_policy"] == "fp32_boundary",
        "selected_input": contract["fox_gd_residual_selected_read_input_policy"] == "fp32_boundary",
    }


def _model_audit(config: Any) -> dict[str, Any]:
    model = BASE.LanguageModel(config.model)
    payload = torch.load(init_path(), map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(payload["model_state_dict"], strict=True)
    return {
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "state_sha256": BASE._state_dict_hash(model.state_dict()),
        "strict": not incompatible.missing_keys and not incompatible.unexpected_keys,
    }


def _gate_config_audit() -> dict[str, Any]:
    from fla.modules import fused_norm_gate

    configs = fused_norm_gate.layer_norm_gated_bwd_kernel.fn.configs
    rows = [
        {
            "kwargs": dict(config.kwargs),
            "num_warps": config.num_warps,
            "num_stages": config.num_stages,
        }
        for config in configs
    ]
    expected = [{"kwargs": {"BT": 64}, "num_warps": 4, "num_stages": 2}]
    return {"configs": rows, "passed": rows == expected}


def preflight() -> dict[str, Any]:
    configure_numerics()
    configure_gate_bwd_runtime("fixed")
    env = environment_metadata()
    configs = {arm: build_config(arm, 123, "screen") for arm in ARMS}
    jobs = []
    for arm, config in configs.items():
        resolved = write_resolved_config(config)
        contract = scientific_contract(config)
        checks = _contract_checks(arm, contract)
        jobs.append(
            {
                "arm": arm,
                "resolved_config": str(resolved.resolve()),
                "contract": contract,
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    sample = configs["ctrl-bridge"]
    cache = BASE._cache_content_hash(sample.data)
    model = _model_audit(sample)
    gate = _gate_config_audit()
    checks = {
        "python": Path(sys.executable).resolve() == PYTHON.resolve(),
        "versions": (env["python"], env["torch"], env["torch_cuda"], env["triton"], env["fla"]) == ("3.12.11", "2.6.0+cu118", "11.8", "3.2.0", "0.4.2"),
        "cuda": env["cuda_available"],
        "gpu": env["gpu_name"] == "NVIDIA GeForce RTX 3090",
        "visible_gpu": env["cuda_visible_devices"] == "0",
        "gpu_free": env["gpu_used_bytes"] is not None and env["gpu_used_bytes"] < 1024**3,
        "cache": cache.get("combined_content_sha256") == EXPECTED_CACHE_HASH,
        "flash_commit": env["flash_commit"] == EXPECTED_FLASH_COMMIT,
        "zoology_base": git_is_ancestor(REPO_ROOT, ZOOLOGY_BASE_COMMIT),
        "source_clean": not env["zoology_status"] and not env["flash_status"],
        "init_file": sha256_file(init_path()) == EXPECTED_INIT_FILE_HASH,
        "model": model == {"trainable_parameters": EXPECTED_PARAMETERS, "state_sha256": EXPECTED_INIT_STATE_HASH, "strict": True},
        "gate": gate["passed"],
        "jobs": all(row["passed"] for row in jobs),
    }
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "run_tag": run_tag(),
        "status": "passed" if all(checks.values()) else "failed",
        "recorded_at_utc": utc_now(),
        "environment": env,
        "cache": cache,
        "model": model,
        "gate": gate,
        "checks": checks,
        "jobs": jobs,
    }
    atomic_write_json(run_root() / "preflight.json", payload)
    if payload["status"] != "passed":
        raise RuntimeError(f"Preflight failed: {run_root() / 'preflight.json'}")
    return payload


def _runtime_arm(arm: str) -> str:
    return UPSTREAM.FASTEST if arm_spec(arm)["family"] == "fastest" else UPSTREAM.CANONICAL


def _completion_result(result: dict[str, Any], config: Any) -> bool:
    checkpoint_dir = checkpoint_run_dir(config)
    for role in ("last", "best"):
        result[f"{role}_checkpoint"] = UPSTREAM.checkpoint_metadata(checkpoint_dir / f"{role}.pt")
    resume = UPSTREAM._resume_audit(checkpoint_dir / "resume.pt", _runtime_arm(result["arm"]))
    result["resume_audit"] = resume
    expected_steps = {"smoke": 3, "screen": 1232, "formal": 2816}[result["phase"]]
    checks = {
        "metrics": result["last_checkpoint"]["finite_metrics"],
        "steps": resume["optimizer_step"] == expected_steps,
        "scaler": resume["grad_scaler_skips"] == 0,
        "model_dtype": resume["model_state_dtypes"] == ["torch.float32"],
        "optimizer_dtype": resume["optimizer_state_dtypes"] == ["torch.float32"],
        "runtime": resume["runtime_audit"]["passed"],
    }
    result["completion_checks"] = checks
    return all(checks.values())


def run_training(arm: str, seed: int, phase: str, gate_mode: str) -> int:
    configure_numerics()
    configure_gate_bwd_runtime(gate_mode)
    config = build_config(arm, seed, phase, gate_mode)
    resolved = write_resolved_config(config)
    output = result_path(arm, seed, phase, gate_mode)
    started_at, started = utc_now(), time.perf_counter()
    try:
        from zoology.train import train

        train(config)
        status, error, return_code = "completed", None, 0
    except BaseException as exc:
        status, error, return_code = "failed", f"{type(exc).__name__}: {exc}", 1
        traceback.print_exc()
    result = {
        **descriptor(arm, seed, gate_mode),
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
        "checkpoint_dir": str(checkpoint_run_dir(config).resolve()),
        "telemetry": UPSTREAM.telemetry_summary(Path(config.training_telemetry_path)),
    }
    if status == "completed" and not _completion_result(result, config):
        result["status"] = "failed"
        result["error"] = "Completed training failed checkpoint, dtype, or runtime audit."
        return_code = 1
    if result["status"] == "completed":
        result["gate_autotune"] = UPSTREAM.gate_autotune_snapshot()
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
    train.add_argument("--phase", choices=PHASES, required=True)
    train.add_argument("--gate-mode", choices=GATE_MODES, default="fixed")
    args = parser.parse_args()
    if args.command == "preflight":
        preflight()
        return 0
    return run_training(args.arm, args.seed, args.phase, args.gate_mode)


if __name__ == "__main__":
    raise SystemExit(main())
