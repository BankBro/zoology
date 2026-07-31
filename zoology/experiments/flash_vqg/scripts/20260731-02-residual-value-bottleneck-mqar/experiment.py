#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

LOCAL_DIR = Path(__file__).resolve().parent
if str(LOCAL_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_DIR))

import torch

from common import (
    BASELINE,
    EXPECTED_CACHE_HASH,
    EXPECTED_FLASH_COMMIT,
    EXPECTED_INIT_FILE_HASH,
    EXPECTED_INIT_STATE_HASH,
    EXPECTED_PARAMETERS,
    EXPECTED_STATE_HASHES,
    EXPERIMENT_ID,
    FLASH_ROOT,
    PYTHON,
    REPO_ROOT,
    SEEDS,
    VARIANT_DIMS,
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


BASE_DIR = REPO_ROOT / "zoology/experiments/flash_vqg/scripts/20260731-01-selected-read-warp-mqar-screen"


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
    base_common = _load_module("residual_value_selected_common", BASE_DIR / "common.py")
    sys.modules["common"] = base_common
    try:
        return _load_module("residual_value_selected_experiment", BASE_DIR / "experiment.py")
    finally:
        if current_common is None:
            sys.modules.pop("common", None)
        else:
            sys.modules["common"] = current_common


UPSTREAM = _load_upstream()
BASE = UPSTREAM.BASE
SHARED_CANONICAL_INIT = Path(
    os.environ.get(
        "MQAR_CANONICAL_INIT",
        "/home/lyj/mnt/project/zoology/zoology/experiments/flash_vqg/scripts/"
        "20260630-03-flash-vqg-s124-fixed-r4-4ep-confirm/outputs/canonical-init/"
        "cb64r16-s124-init.pt",
    )
).resolve()
BASE.BASE.CANONICAL_INIT = SHARED_CANONICAL_INIT
SHARED_CACHE_DIR = Path(
    os.environ.get(
        "MQAR_CACHE_DIR",
        "/home/lyj/mnt/project/zoology/data/flash_vqg",
    )
).resolve()


def run_id(variant: str, seed: int, phase: str) -> str:
    return f"{variant}-s{seed}-bf16-b64ga4-{phase}"


def checkpoint_root(phase: str) -> Path:
    return run_root() / "checkpoints" / phase


def result_path(variant: str, seed: int, phase: str) -> Path:
    return run_root() / "training" / phase / run_id(variant, seed, phase) / "result.json"


def canonical_init_path() -> Path:
    return Path(BASE.init_path()).resolve()


def init_path(variant: str) -> Path:
    if variant == BASELINE:
        return canonical_init_path()
    return run_root() / "derived-init" / f"{variant}.pt"


def _projection_keys(state: dict[str, torch.Tensor]) -> list[str]:
    return sorted(key for key in state if key.endswith("fox_gd_residual_value_proj"))


def _build_derived_init(config: Any, variant: str) -> Path:
    path = init_path(variant)
    if path.is_file():
        return path
    payload = torch.load(canonical_init_path(), map_location="cpu", weights_only=False)
    model = BASE.BASE.LanguageModel(config.model)
    incompatible = model.load_state_dict(payload["model_state_dict"], strict=False)
    candidate_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    projection_keys = _projection_keys(candidate_state)
    if sorted(incompatible.missing_keys) != projection_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected derived-init incompatibility: {incompatible}")
    for key, value in payload["model_state_dict"].items():
        if not torch.equal(candidate_state[key], value):
            raise RuntimeError(f"Canonical initialization changed at {key}.")
    derived = copy.deepcopy(payload)
    derived["model_state_dict"] = candidate_state
    derived["residual_value_bottleneck"] = {
        "variant": variant,
        "residual_value_dim": VARIANT_DIMS[variant],
        "canonical_init": str(canonical_init_path()),
        "canonical_init_sha256": sha256_file(canonical_init_path()),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(derived, path)
    return path


def source_identity(variant: str) -> dict[str, Any]:
    path = init_path(variant)
    return {
        **descriptor(variant, 123),
        "experiment_id": EXPERIMENT_ID,
        "run_tag": run_tag(),
        "zoology_commit": BASE.git_value(REPO_ROOT, "rev-parse", "HEAD"),
        "flash_commit": BASE.git_value(FLASH_ROOT, "rev-parse", "HEAD"),
        "cache_content_sha256": EXPECTED_CACHE_HASH,
        "init_file_sha256": sha256_file(path),
    }


def build_config(variant: str, seed: int, phase: str):
    descriptor(variant, seed)
    if seed != 123 or phase not in {"smoke", "screen"}:
        raise ValueError(f"Unsupported Q0 job: {variant}, {seed}, {phase}.")
    previous = os.environ.get("MQAR_SELECTED_WARP_RUN_TAG")
    os.environ["MQAR_SELECTED_WARP_RUN_TAG"] = run_tag()
    try:
        config = UPSTREAM.build_config("s1-head8192", seed, phase)
    finally:
        if previous is None:
            os.environ.pop("MQAR_SELECTED_WARP_RUN_TAG", None)
        else:
            os.environ["MQAR_SELECTED_WARP_RUN_TAG"] = previous
    kwargs = BASE.BASE._find_flash_kwargs(config.model)
    kwargs["fox_gd_residual_value_dim"] = str(VARIANT_DIMS[variant])
    kwargs["fox_gd_residual_value_proj_init_seed"] = "1729"
    config.data.cache_dir = str(SHARED_CACHE_DIR)
    path = canonical_init_path() if variant == BASELINE else _build_derived_init(config, variant)
    config.init_checkpoint_path = str(path)
    config.checkpoint.root_dir = str(checkpoint_root(phase))
    config.launch_id = f"{EXPERIMENT_ID}-{run_tag()}-{phase}"
    config.sweep_id = EXPERIMENT_ID
    config.run_id = run_id(variant, seed, phase)
    config.training_telemetry_path = str(
        run_root() / "training" / phase / config.run_id / "telemetry.jsonl"
    )
    config.resume_identity = source_identity(variant)
    return config


def serialize_config(config: Any) -> dict[str, Any]:
    from zoology.checkpoints import serialize_train_config

    return serialize_train_config(config)


def write_resolved_config(config: Any) -> Path:
    path = generated_root() / f"{config.run_id}.json"
    atomic_write_json(path, serialize_config(config))
    return path


def normalized_config(config: Any) -> dict[str, Any]:
    payload = serialize_config(config)
    payload["run_id"] = "<run>"
    payload["launch_id"] = "<launch>"
    payload["checkpoint"]["root_dir"] = "<checkpoint>"
    payload["training_telemetry_path"] = "<telemetry>"
    payload["resume_identity"] = "<identity>"
    payload["init_checkpoint_path"] = "<variant-init>"
    return payload


def _value_differences(left: Any, right: Any) -> list[str]:
    left_flat = UPSTREAM._flatten(normalized_config(left))
    right_flat = UPSTREAM._flatten(normalized_config(right))
    keys = sorted(set(left_flat) | set(right_flat))
    return [key for key in keys if left_flat.get(key) != right_flat.get(key)]


def model_audit(config: Any, variant: str) -> dict[str, Any]:
    payload = torch.load(init_path(variant), map_location="cpu", weights_only=False)
    canonical = torch.load(canonical_init_path(), map_location="cpu", weights_only=False)
    model = BASE.BASE.LanguageModel(config.model)
    incompatible = model.load_state_dict(payload["model_state_dict"], strict=True)
    state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    projection_keys = _projection_keys(state)
    projection_parameters = sum(state[key].numel() for key in projection_keys)
    common_equal = all(torch.equal(state[key], value) for key, value in canonical["model_state_dict"].items())
    orthogonal_error = 0.0
    for key in projection_keys:
        projection = state[key].float()
        gram = torch.einsum("hvu,hvw->huw", projection, projection)
        identity = torch.eye(projection.size(-1)).expand(projection.size(0), -1, -1)
        orthogonal_error = max(orthogonal_error, float((gram - identity).abs().max()))
    kwargs = BASE.BASE._find_flash_kwargs(config.model)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "trainable_parameters": trainable,
        "expected_parameters": 1_160_390 + projection_parameters,
        "projection_parameters": projection_parameters,
        "projection_keys": projection_keys,
        "projection_orthogonal_max_abs": orthogonal_error,
        "common_state_equal": common_equal,
        "state_sha256": BASE.state_dict_hash(state),
        "strict": not incompatible.missing_keys and not incompatible.unexpected_keys,
        "block_len": kwargs.get("block_len"),
        "local_num_blocks": kwargs.get("local_num_blocks"),
        "rank": kwargs.get("fox_gd_residual_rank"),
        "read_topk": kwargs.get("fox_remote_read_topk"),
        "write_topk": kwargs.get("fox_gd_residual_write_topk"),
        "residual_value_dim": int(kwargs.get("fox_gd_residual_value_dim")),
        "builder": kwargs.get("fox_gd_residual_builder"),
        "selected_backward": kwargs.get("fox_gd_residual_selected_read_backward_backend"),
        "input_policy": kwargs.get("fox_gd_residual_triton_input_policy"),
        "remat_mode": kwargs.get("fox_gd_residual_remat_mode"),
    }


def runtime_audit(states: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return UPSTREAM.runtime_audit(states)


def _job_passed(row: dict[str, Any]) -> bool:
    audit = row["audit"]
    expected_projection_keys = 0 if row["variant"] == BASELINE else 1
    return (
        audit["trainable_parameters"] == audit["expected_parameters"]
        and audit["trainable_parameters"] == EXPECTED_PARAMETERS[row["variant"]]
        and audit["state_sha256"] == EXPECTED_STATE_HASHES[row["variant"]]
        and len(audit["projection_keys"]) == expected_projection_keys
        and audit["projection_orthogonal_max_abs"] <= 1e-5
        and audit["common_state_equal"]
        and audit["strict"]
        and audit["block_len"] == 64
        and audit["local_num_blocks"] == 2
        and audit["rank"] == 16
        and audit["read_topk"] == 16
        and audit["write_topk"] == 4
        and audit["residual_value_dim"] == VARIANT_DIMS[row["variant"]]
        and audit["builder"] == "grouped_chunk_torch_ref"
        and audit["selected_backward"] == "triton_deterministic_s1_head"
        and audit["input_policy"] == "fp32_boundary"
        and audit["remat_mode"] == "post_phase1"
        and row["precision"] == "amp_bfloat16"
    )


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
                    "precision": config.precision,
                    "audit": model_audit(config, variant),
                }
            )
    baseline = build_config(BASELINE, 123, "screen")
    differences = {
        variant: _value_differences(baseline, build_config(variant, 123, "screen"))
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
        "canonical_init": sha256_file(canonical_init_path()) == EXPECTED_INIT_FILE_HASH,
        "canonical_state": jobs[0]["audit"]["state_sha256"] == EXPECTED_INIT_STATE_HASH,
        "flash_commit": env["flash_commit"] == EXPECTED_FLASH_COMMIT,
        "source_clean": not env["zoology_status"] and not env["flash_status"],
        "branch": bool(
            re.fullmatch(r"20260731-\d{6}-residual-value-bottleneck-mqar", env["zoology_branch"])
        ),
        "single_variable": all(
            len(rows) == 1
            and rows[0].endswith("kwargs.fox_gd_residual_value_dim")
            for rows in differences.values()
        ),
        "jobs": all(_job_passed(row) for row in jobs),
    }
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


def run_training(variant: str, seed: int, phase: str) -> int:
    replacements = {
        "EXPERIMENT_ID": EXPERIMENT_ID,
        "build_config": build_config,
        "descriptor": descriptor,
        "result_path": result_path,
        "run_tag": run_tag,
        "runtime_audit": runtime_audit,
        "write_resolved_config": write_resolved_config,
    }
    original = {name: getattr(UPSTREAM, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(UPSTREAM, name, value)
        return UPSTREAM.run_training(variant, seed, phase)
    finally:
        for name, value in original.items():
            setattr(UPSTREAM, name, value)


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
