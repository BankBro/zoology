#!/usr/bin/env python3
"""Preflight checks for the 1-epoch repro screen experiment."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get("ZOOLOGY_REPO_ROOT", "/home/lyj/mnt/project/zoology")).resolve()
EXPERIMENT_ID = "20260625-02-flash-vqg-1epoch-repro-screen"
BUILDER_PATH = (
    REPO_ROOT
    / "zoology/experiments/flash_vqg/scripts/20260425-gd-residual-v1-mqar/config_builder.py"
)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from zoology.data.utils import prepare_data


def _load_builder_module():
    spec = importlib.util.spec_from_file_location("gd_residual_v1_config_builder", BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load config builder from {BUILDER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def _nvidia_smi_query() -> dict[str, Any]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return {"available": False}
    rows = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        rows.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "driver_version": parts[2],
                "memory_total_mib": int(parts[3]),
            }
        )
    return {"available": True, "gpus": rows}


def _base_args(*, logger_backend: str, metrics_yaml: Path, trace_output_dir: Path) -> Namespace:
    return Namespace(
        launch_id_prefix="fvqg-20260625-02-rscreen-preflight",
        backend="torch",
        logger_backend=logger_backend,
        dmodels="128",
        learning_rates="1e-3",
        train_batch_order="global_shuffle",
        seed_values="123",
        data_seed=123,
        num_codebook_vectors="64",
        fox_remote_path_backend="torch",
        fox_remote_read_topk_values="2",
        fox_remote_formula="gd_residual_v1",
        fox_gd_residual_rank=16,
        fox_gd_residual_write_topk=4,
        fox_gd_residual_builder="grouped_chunk_torch_ref",
        fox_gd_residual_pack_mode="semivec_ref",
        fox_gd_residual_chunk_size=64,
        fox_gd_residual_mu_min_count=0.1,
        fox_gd_residual_beta_init=0.5,
        fox_gd_residual_lambda_init=0.05,
        vq_score_mode="codebook_dot",
        vq_weight_mode="dense_softmax",
        vq_update_mode="grad",
        vq_softmax_tau=0.25,
        vq_topk=4,
        gradient_accumulation_steps=4,
        train_batch_size=64,
        eval_batch_size=16,
        metrics_white_list=None,
        metrics_white_list_file=str(metrics_yaml),
        cache_dir="./data/flash_vqg",
        project="flash_vqg_1epoch_repro_screen",
        entity="scu-mclab",
        max_epochs=1,
        max_train_steps=None,
        max_validation_batches=None,
        validations_per_epoch=4,
        disable_early_stopping="true",
        read_churn_probe_enabled="true",
        read_churn_probe_valid_batches="441",
        read_churn_probe_max_samples=16,
        read_churn_probe_query_only="true",
        read_trace_enabled="true",
        read_trace_valid_batches="441",
        read_trace_max_samples=4,
        read_trace_query_only="true",
        read_trace_max_queries_per_sample=8,
        read_trace_output_dir=str(trace_output_dir),
        read_trace_train_steps="0,64,130,203,352,448,704",
        experiment_mode="rscreen_preflight",
        run_id="rscreen-preflight",
    )


def _config_contract(config) -> dict[str, Any]:
    train_dataloader, _ = prepare_data(config.data)
    num_batches = len(train_dataloader)
    accum_steps = int(config.gradient_accumulation_steps)
    num_optimizer_steps = (num_batches + accum_steps - 1) // accum_steps
    return {
        "train_batches": int(num_batches),
        "gradient_accumulation_steps": accum_steps,
        "num_optimizer_steps": int(num_optimizer_steps),
        "passed": num_batches == 2815 and accum_steps == 4 and num_optimizer_steps == 704,
    }


def _env_snapshot(machine_name: str) -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count()
    nvidia_smi = _nvidia_smi_query()
    runtime_ready = bool(cuda_available and device_count > 0 and nvidia_smi.get("available"))
    return {
        "experiment_id": EXPERIMENT_ID,
        "machine_name": machine_name,
        "hostname": platform.node(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_branch": _git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_commit": _git_value(["rev-parse", "--short", "HEAD"]),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": cuda_available,
        "device_count": device_count,
        "runtime_ready": runtime_ready,
        "torch_deterministic": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": bool(getattr(torch.backends.cudnn, "deterministic", False)),
        "cudnn_benchmark": bool(getattr(torch.backends.cudnn, "benchmark", False)),
        "env": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "SWANLAB_MODE": os.environ.get("SWANLAB_MODE"),
        },
        "nvidia_smi": nvidia_smi,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine-name", type=str, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--mode", choices=["smoke", "train"], default="train")
    args = parser.parse_args()

    metrics_yaml = SCRIPT_DIR / "metrics.yaml"
    trace_output_dir = SCRIPT_DIR / "outputs" / "preflight-traces" / args.machine_name / args.mode
    builder_args = _base_args(
        logger_backend="none" if args.mode == "smoke" else "swanlab",
        metrics_yaml=metrics_yaml,
        trace_output_dir=trace_output_dir,
    )
    builder_module = _load_builder_module()
    if args.mode == "smoke":
        configs = builder_module.build_gd_residual_v1_smoke_configs(builder_args)
    else:
        configs = builder_module.build_gd_residual_v1_train_configs(builder_args)
    if len(configs) != 1:
        raise RuntimeError(f"Expected 1 config, got {len(configs)}")
    config = configs[0]

    payload = {
        "env": _env_snapshot(args.machine_name),
        "contract": _config_contract(config),
        "run_id": getattr(config, "run_id", None),
        "max_epochs": getattr(config, "max_epochs", None),
        "max_train_steps": getattr(config, "max_train_steps", None),
        "validations_per_epoch": getattr(config, "validations_per_epoch", None),
        "read_trace_train_steps": list(getattr(config, "read_trace_train_steps", [])),
        "trace_output_dir": str(getattr(config, "read_trace_output_dir", "")),
    }
    payload["passed"] = bool(payload["contract"]["passed"] and payload["env"]["runtime_ready"])

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
