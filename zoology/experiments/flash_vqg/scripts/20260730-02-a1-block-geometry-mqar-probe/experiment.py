#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent / "20260730-01-a1-acceleration-mqar-probe"
REPO_ROOT = Path("/home/lyj/mnt/project/zoology")
FLASH_ROOT = Path("/home/lyj/mnt/project/Flash-VQG")
EXPERIMENT_ID = "20260730-02-a1-block-geometry-mqar-probe"
EXPECTED_FLASH_COMMIT = "60a18b2"
SCALE = 4
VARIANTS = ("a1-reference", "a1-block128", "a1-block128-k2r8")


def _load_base_experiment():
    previous_common = sys.modules.pop("common", None)
    sys.path.insert(0, str(BASE_DIR))
    spec = importlib.util.spec_from_file_location("a1_geometry_base_experiment", BASE_DIR / "experiment.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    sys.path.remove(str(BASE_DIR))
    if previous_common is None:
        sys.modules.pop("common", None)
    else:
        sys.modules["common"] = previous_common
    return module


BASE = _load_base_experiment()
PYTHON = BASE.PYTHON


def run_tag() -> str:
    value = os.environ.get("MQAR_A1_GEOMETRY_RUN_TAG", "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise RuntimeError("MQAR_A1_GEOMETRY_RUN_TAG must be a non-empty safe name.")
    return value


def run_root() -> Path:
    return SCRIPT_DIR / "outputs" / "2080ti" / run_tag()


def generated_root() -> Path:
    return REPO_ROOT / "zoology/experiments/flash_vqg/generated" / f"{EXPERIMENT_ID}-{run_tag()}"


def git_value(root: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(root), *args], text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else ""


def source_identity(variant: str) -> dict[str, str]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "run_tag": run_tag(),
        "variant": variant,
        "seed": "123",
        "geometry_scale": str(SCALE if variant != "a1-reference" else 1),
        "zoology_commit": git_value(REPO_ROOT, "rev-parse", "HEAD"),
        "flash_commit": git_value(FLASH_ROOT, "rev-parse", "HEAD"),
    }


def _scale_segment(segment: Any, *, scale_examples: bool) -> None:
    segment.input_seq_len *= SCALE
    segment.num_kv_pairs *= SCALE
    if scale_examples:
        if segment.num_examples % SCALE:
            raise ValueError("Training example count must be divisible by geometry scale.")
        segment.num_examples //= SCALE


def _scale_candidate_config(config: Any) -> None:
    for segment in config.data.train_configs:
        _scale_segment(segment, scale_examples=True)
    for segment in config.data.test_configs:
        _scale_segment(segment, scale_examples=False)
    config.data.batch_size = (16, 4)
    configs = config.model.sequence_mixer.kwargs["configs"]
    for child in configs:
        if child["name"].endswith("BaseConv"):
            child["kwargs"]["l_max"] = 4096


def run_id(variant: str, phase: str) -> str:
    scale = SCALE if variant != "a1-reference" else 1
    return f"{variant}-scale{scale}-s123-fp32-{phase}"


def result_path(variant: str, phase: str) -> Path:
    return run_root() / "training" / phase / run_id(variant, phase) / "result.json"


def build_config(variant: str, phase: str):
    if variant not in VARIANTS:
        raise ValueError(f"Unsupported variant: {variant}")
    config = copy.deepcopy(BASE.build_config(variant, phase))
    if variant != "a1-reference":
        _scale_candidate_config(config)
    config.resume_identity = source_identity(variant)
    config.checkpoint.root_dir = str(run_root() / "checkpoints" / phase)
    config.launch_id = f"{EXPERIMENT_ID}-{run_tag()}-{phase}"
    config.sweep_id = EXPERIMENT_ID
    config.run_id = run_id(variant, phase)
    config.training_telemetry_path = str(result_path(variant, phase).with_name("telemetry.jsonl"))
    return config


def write_config(config: Any) -> Path:
    path = generated_root() / f"{config.run_id}.json"
    BASE.atomic_write_json(path, BASE.serialize_config(config))
    return path


def train_tokens(config: Any) -> int:
    return sum(row.num_examples * row.input_seq_len for row in config.data.train_configs)


def train_batches(config: Any) -> int:
    batch = int(config.data.batch_size[0])
    return sum(row.num_examples for row in config.data.train_configs) // batch


def geometry_rows(config: Any) -> list[dict[str, int]]:
    block_len = BASE.BASE._find_flash_kwargs(config.model)["block_len"]
    return [
        {
            "blocks": row.input_seq_len // block_len,
            "tokens_per_pair": row.input_seq_len // row.num_kv_pairs,
        }
        for row in config.data.train_configs
    ]


def prepare_data() -> None:
    from zoology.data.utils import prepare_data as build_data

    for variant in ("a1-reference", "a1-block128"):
        train_loader, test_loader = build_data(build_config(variant, "screen").data)
        del train_loader, test_loader


def preflight() -> dict[str, Any]:
    BASE.configure_numerics()
    configs = {name: build_config(name, "screen") for name in VARIANTS}
    audits = {name: BASE.model_audit(config) for name, config in configs.items()}
    caches = {name: BASE.BASE._cache_content_hash(config.data) for name, config in configs.items()}
    for config in configs.values():
        write_config(config)
    reference = configs["a1-reference"]
    checks = {
        "python": Path(sys.executable).resolve() == PYTHON.resolve(),
        "gpu": torch.cuda.is_available() and torch.cuda.get_device_name(0) == "NVIDIA GeForce RTX 2080 Ti",
        "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES") == "1",
        "gpu_free": torch.cuda.is_available() and int(torch.cuda.device_memory_used()) < 1024**3,
        "source_clean": not git_value(REPO_ROOT, "status", "--short") and not git_value(FLASH_ROOT, "status", "--short"),
        "branch": git_value(REPO_ROOT, "branch", "--show-current") == "20260730-055000-a1-acceleration-mqar-probe",
        "flash_commit": git_value(FLASH_ROOT, "rev-parse", "--short", "HEAD") == EXPECTED_FLASH_COMMIT,
        "same_tokens": all(train_tokens(config) == train_tokens(reference) for config in configs.values()),
        "same_batches": all(train_batches(config) == train_batches(reference) for config in configs.values()),
        "same_blocks": all(geometry_rows(config) == geometry_rows(reference) for config in configs.values()),
        "model_audit": all(row["parameters"] == 1_160_390 and row["strict"] for row in audits.values()),
    }
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "run_tag": run_tag(),
        "status": "passed" if all(checks.values()) else "failed",
        "source": source_identity("preflight"),
        "checks": checks,
        "train_tokens": {name: train_tokens(config) for name, config in configs.items()},
        "train_batches": {name: train_batches(config) for name, config in configs.items()},
        "geometry": {name: geometry_rows(config) for name, config in configs.items()},
        "audits": audits,
        "caches": caches,
    }
    BASE.atomic_write_json(run_root() / "preflight.json", payload)
    if payload["status"] != "passed":
        raise RuntimeError(f"Preflight failed: {checks}")
    return payload


def checkpoint_dir(config: Any) -> Path:
    return Path(config.checkpoint.root_dir) / str(config.launch_id) / str(config.run_id)


def run_training(variant: str, phase: str) -> int:
    BASE.configure_numerics()
    config = build_config(variant, phase)
    resolved = write_config(config)
    started = time.perf_counter()
    try:
        from zoology.train import train

        train(config)
        status, error, code = "completed", None, 0
    except BaseException as exc:
        status, error, code = "failed", f"{type(exc).__name__}: {exc}", 1
        traceback.print_exc()
    result = {
        "experiment_id": EXPERIMENT_ID,
        "run_tag": run_tag(),
        "variant": variant,
        "phase": phase,
        "status": status,
        "error": error,
        "wall_clock_sec": time.perf_counter() - started,
        "resolved_config": str(resolved.resolve()),
        "resolved_config_sha256": BASE.sha256_file(resolved),
    }
    if status == "completed":
        root = checkpoint_dir(config)
        result["last_checkpoint"] = BASE.checkpoint_metadata(root / "last.pt")
        result["best_checkpoint"] = BASE.checkpoint_metadata(root / "best.pt")
        if not result["last_checkpoint"]["finite"]:
            result["status"], result["error"], code = "failed", "Non-finite checkpoint metrics.", 1
    BASE.atomic_write_json(result_path(variant, phase), result)
    print(json.dumps({"status": result["status"], "result": str(result_path(variant, phase))}))
    return code


def standard_accuracy(variant: str) -> float:
    result = BASE.load_json(result_path(variant, "screen"))
    shape = "1024x256" if variant == "a1-reference" else "4096x1024"
    return float(result["last_checkpoint"]["metrics"][f"valid/mqar_case/accuracy-{shape}"])


def summarize() -> dict[str, Any]:
    accuracy = {name: standard_accuracy(name) for name in VARIANTS}
    reference = accuracy["a1-reference"]
    deltas = {name: value - reference for name, value in accuracy.items() if name != "a1-reference"}
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "run_tag": run_tag(),
        "status": "passed" if any(value >= -0.02 for value in deltas.values()) else "quality_rejected",
        "standard_accuracy": accuracy,
        "standard_delta": deltas,
        "threshold": -0.02,
    }
    BASE.atomic_write_json(run_root() / "summary.json", payload)
    print(json.dumps(payload, sort_keys=True))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare-data")
    sub.add_parser("preflight")
    train = sub.add_parser("train")
    train.add_argument("--variant", choices=VARIANTS, required=True)
    train.add_argument("--phase", choices=("smoke", "screen"), required=True)
    sub.add_parser("summarize")
    args = parser.parse_args()
    if args.command == "prepare-data":
        prepare_data()
        return 0
    if args.command == "preflight":
        preflight()
        return 0
    if args.command == "summarize":
        return 0 if summarize()["status"] == "passed" else 2
    return run_training(args.variant, args.phase)


if __name__ == "__main__":
    raise SystemExit(main())
