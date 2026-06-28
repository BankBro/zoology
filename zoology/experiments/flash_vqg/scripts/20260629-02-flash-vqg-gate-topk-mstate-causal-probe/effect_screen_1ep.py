#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get("ZOOLOGY_REPO_ROOT", "/home/lyj/mnt/project/zoology")).resolve()
FLASH_VQG_ROOT = Path(os.environ.get("FLASH_VQG_ROOT", "/home/lyj/mnt/project/Flash-VQG")).resolve()
EXPERIMENT_ID = "20260629-02-flash-vqg-gate-topk-mstate-causal-probe"
NO_DROPOUT_SCRIPT = (
    REPO_ROOT
    / "zoology/experiments/flash_vqg/scripts"
    / "20260628-02-flash-vqg-no-dropout-4ep-confirm/no_dropout_4ep.py"
)
FIRST_PROBE_SCRIPT = (
    REPO_ROOT
    / "zoology/experiments/flash_vqg/scripts"
    / "20260627-03-flash-vqg-first-divergence-probe/probe_first_divergence.py"
)
DEFAULT_INIT_CHECKPOINT = (
    REPO_ROOT
    / "zoology/experiments/flash_vqg/scripts"
    / "20260627-02-flash-vqg-canonical-init-lock-screen/outputs/canonical-init/cb64r16-s123-init.pt"
)

if str(FLASH_VQG_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(FLASH_VQG_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from zoology.checkpoints import serialize_train_config  # noqa: E402
from zoology.train import train  # noqa: E402


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


NO_DROPOUT = _load_module(NO_DROPOUT_SCRIPT, "flash_vqg_no_dropout_4ep_lib")
FIRST_PROBE = _load_module(FIRST_PROBE_SCRIPT, "flash_vqg_first_probe_lib")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    return str(value)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default)
        + "\n",
        encoding="utf-8",
    )


def _git_value(repo: Path, args: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def _env_snapshot(machine_name: str) -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    return {
        "experiment_id": EXPERIMENT_ID,
        "machine_name": machine_name,
        "utc_time": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch_version": str(torch.__version__),
        "torch_cuda": None if torch.version.cuda is None else str(torch.version.cuda),
        "cuda_available": cuda_available,
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "cuda_device_capability": torch.cuda.get_device_capability(0) if cuda_available else None,
        "zoology_branch": _git_value(REPO_ROOT, ["branch", "--show-current"]),
        "zoology_commit": _git_value(REPO_ROOT, ["rev-parse", "--short", "HEAD"]),
        "flash_vqg_branch": _git_value(FLASH_VQG_ROOT, ["branch", "--show-current"]),
        "flash_vqg_commit": _git_value(FLASH_VQG_ROOT, ["rev-parse", "--short", "HEAD"]),
    }


def _set_flash_vqg_kwarg(config: Any, key: str, value: Any) -> None:
    flash_kwargs = FIRST_PROBE._find_flash_vqg_kwargs(config.model)
    if not flash_kwargs:
        raise KeyError("Could not find Flash-VQG mixer kwargs in config.model.")
    flash_kwargs[key] = value


def _apply_overrides(config: Any, args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if args.fox_gate_logf_constant_f is not None:
        value = float(args.fox_gate_logf_constant_f)
        _set_flash_vqg_kwarg(config, "fox_gate_logf_constant_f", value)
        overrides["fox_gate_logf_constant_f"] = value
    if args.fox_remote_read_topk is not None:
        value = int(args.fox_remote_read_topk)
        _set_flash_vqg_kwarg(config, "fox_remote_read_topk", value)
        overrides["fox_remote_read_topk"] = value
    if args.fox_gd_residual_residual_norm_mode is not None:
        value = str(args.fox_gd_residual_residual_norm_mode)
        _set_flash_vqg_kwarg(config, "fox_gd_residual_residual_norm_mode", value)
        overrides["fox_gd_residual_residual_norm_mode"] = value
    return overrides


def _build_config(args: argparse.Namespace):
    config = NO_DROPOUT.build_config(
        target="no-dropout-4ep-s123-r1",
        machine_name=args.machine_name,
        variant="no-dropout",
        logger_backend=args.logger_backend,
        trace_output_dir=args.output_dir / "read-trace",
        max_epochs=1,
        max_train_steps=args.max_train_steps,
        max_validation_batches=args.max_validation_batches,
    )
    config.launch_id = f"fvqg-{EXPERIMENT_ID}-1ep-{args.variant_name}-{args.machine_name}"
    config.run_id = f"{EXPERIMENT_ID}-1ep-{args.variant_name}-{args.machine_name}"
    config.init_checkpoint_path = str(args.init_checkpoint)
    config.init_checkpoint_strict = True
    config.init_checkpoint_source_launch_id = "canonical-init-2080ti"
    config.init_checkpoint_source_run_id = "initlock-cb64r16-default-s123-r1-d123-b64ga4-2080ti"
    config.checkpoint.enabled = False
    config.checkpoint.save_best = False
    config.checkpoint.save_last = False
    config.checkpoint.save_config_json = False
    overrides = _apply_overrides(config, args)
    return config, overrides


def run_train(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("torch.cuda is unavailable inside this container.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config, overrides = _build_config(args)
    _save_json(args.output_dir / "config.json", serialize_train_config(config))
    result = train(config)
    _save_json(
        args.output_dir / "result.json",
        {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "screen": "1ep",
            "machine_name": args.machine_name,
            "variant": args.variant_name,
            "config_overrides": overrides,
            "init_checkpoint": str(args.init_checkpoint),
            "environment": _env_snapshot(args.machine_name),
            "train_result": result,
        },
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"{EXPERIMENT_ID} 1ep effect screen")
    parser.add_argument("--machine-name", required=True)
    parser.add_argument("--variant-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, default=DEFAULT_INIT_CHECKPOINT)
    parser.add_argument("--logger-backend", choices=["none", "swanlab", "wandb"], default="none")
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument("--max-validation-batches", type=int)
    parser.add_argument("--fox-gate-logf-constant-f", type=float)
    parser.add_argument("--fox-remote-read-topk", type=int)
    parser.add_argument("--fox-gd-residual-residual-norm-mode", choices=["rmsnorm", "raw", "zero"])
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    os.chdir(REPO_ROOT)
    return run_train(args)


if __name__ == "__main__":
    raise SystemExit(main())
