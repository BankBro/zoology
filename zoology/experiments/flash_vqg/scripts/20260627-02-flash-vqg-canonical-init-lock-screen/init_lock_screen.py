#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
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
FLASH_VQG_ROOT = Path(os.environ.get("FLASH_VQG_ROOT", "/home/lyj/mnt/project/Flash-VQG")).resolve()
EXPERIMENT_ID = "20260627-02-flash-vqg-canonical-init-lock-screen"
METRICS_YAML = (
    REPO_ROOT
    / "zoology/experiments/flash_vqg/scripts/20260625-02-flash-vqg-1epoch-repro-screen/metrics.yaml"
)
BUILDER_PATH = (
    REPO_ROOT
    / "zoology/experiments/flash_vqg/scripts/20260425-gd-residual-v1-mqar/config_builder.py"
)

if str(FLASH_VQG_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(FLASH_VQG_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from zoology.checkpoints import serialize_train_config
from zoology.data.utils import prepare_data
from zoology.model import LanguageModel
from zoology.train import train
from zoology.utils import set_determinism


def _git_value(repo: Path, args: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def _load_builder_module():
    spec = importlib.util.spec_from_file_location("gd_residual_v1_config_builder", BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load config builder from {BUILDER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _base_args(
    *,
    target: str,
    machine_name: str,
    logger_backend: str,
    trace_output_dir: Path,
    max_epochs: int,
    max_train_steps: int | None,
    max_validation_batches: int | None,
) -> Namespace:
    if target not in {"default-s123-r1", "default-s123-r2"}:
        raise ValueError(f"Unsupported target: {target}")
    repeat = "r1" if target.endswith("-r1") else "r2"
    return Namespace(
        launch_id_prefix=f"fvqg-20260627-02-initlock-{machine_name}-{target}",
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
        fox_gd_residual_addr_eps=1e-6,
        fox_gd_residual_den_eps=1e-6,
        fox_gd_residual_rho_eps=1e-12,
        fox_gd_residual_addr_init_rng_mode="global",
        fox_gd_residual_addr_init_seed=None,
        fox_gd_residual_beta_init=0.5,
        fox_gd_residual_beta_cap=None,
        fox_gd_residual_beta_cap_final=None,
        fox_gd_residual_beta_cap_release_start_train_steps=0,
        fox_gd_residual_beta_cap_release_end_train_steps=0,
        fox_gd_residual_beta_cap_eval_policy="final",
        fox_gd_residual_beta_control_mode="hard_cap",
        fox_gd_residual_beta_sigmoid_temp=1.0,
        fox_gd_residual_beta_low=None,
        fox_gd_residual_beta_high=None,
        fox_gd_residual_beta_low_final=None,
        fox_gd_residual_beta_high_final=None,
        fox_gd_residual_beta_band_release_start_train_steps=0,
        fox_gd_residual_beta_band_release_end_train_steps=0,
        fox_gd_residual_beta_band_eval_policy="final",
        fox_gd_residual_beta_band_schedule="smoothstep",
        fox_gd_residual_lambda_init=0.05,
        fox_gd_residual_lambda_floor=0.0,
        fox_gd_residual_write_strength_mode="renorm_topk",
        fox_gd_residual_write_strength_cap=None,
        fox_gd_residual_write_strength_cap_mode="hard",
        fox_gd_residual_write_strength_cap_until_train_steps=0,
        fox_gd_residual_write_strength_cap_final=None,
        fox_gd_residual_write_strength_cap_release_start_train_steps=0,
        fox_gd_residual_write_strength_cap_release_end_train_steps=0,
        fox_gd_residual_write_strength_cap_eval_policy="final",
        fox_gd_residual_write_budget=None,
        fox_gd_residual_write_budget_final=None,
        fox_gd_residual_write_budget_release_start_train_steps=0,
        fox_gd_residual_write_budget_release_end_train_steps=0,
        fox_gd_residual_write_budget_eval_policy="final",
        fox_gd_residual_write_budget_schedule="smoothstep",
        fox_gd_residual_write_total_cap=None,
        fox_gd_residual_write_total_cap_final=None,
        fox_gd_residual_write_total_cap_release_start_train_steps=0,
        fox_gd_residual_write_total_cap_release_end_train_steps=0,
        fox_gd_residual_write_total_cap_eval_policy="final",
        fox_gd_residual_write_total_cap_schedule="smoothstep",
        fox_gd_residual_write_q_alpha=1.0,
        fox_gd_residual_m_norm_cap=None,
        fox_gd_residual_update_norm_cap=None,
        fox_gd_residual_norm_with_gain=False,
        fox_gd_residual_use_separate_addr_codebook=False,
        vq_score_mode="codebook_dot",
        vq_weight_mode="dense_softmax",
        vq_update_mode="grad",
        vq_softmax_tau=0.25,
        codebook_init_rng_mode="global",
        codebook_init_seed=None,
        vq_topk=4,
        gradient_accumulation_steps=4,
        train_batch_size=64,
        eval_batch_size=16,
        metrics_white_list=None,
        metrics_white_list_file=str(METRICS_YAML),
        cache_dir="./data/flash_vqg",
        project="flash_vqg_init_lock_screen",
        entity="scu-mclab",
        max_epochs=max_epochs,
        max_train_steps=max_train_steps,
        max_validation_batches=max_validation_batches,
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
        experiment_mode=f"initlock_cb64r16_default_s123_{repeat}_d123_b64ga4_{machine_name}",
        run_id=f"initlock-cb64r16-default-s123-{repeat}-d123-b64ga4-{machine_name}",
    )


def build_config(
    *,
    target: str,
    machine_name: str,
    logger_backend: str,
    trace_output_dir: Path,
    max_epochs: int,
    max_train_steps: int | None,
    max_validation_batches: int | None,
):
    builder = _load_builder_module()
    args = _base_args(
        target=target,
        machine_name=machine_name,
        logger_backend=logger_backend,
        trace_output_dir=trace_output_dir,
        max_epochs=max_epochs,
        max_train_steps=max_train_steps,
        max_validation_batches=max_validation_batches,
    )
    configs = builder.build_gd_residual_v1_train_configs(args)
    if len(configs) != 1:
        raise RuntimeError(f"Expected one config, got {len(configs)}")
    config = configs[0]
    config.launch_id = args.launch_id_prefix
    return config


def tensor_hash(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().cpu().contiguous()
    return hashlib.sha256(cpu.numpy().tobytes()).hexdigest()


def state_dict_hashes(state_dict: dict[str, torch.Tensor]) -> tuple[str, dict[str, str]]:
    digest = hashlib.sha256()
    per_tensor: dict[str, str] = {}
    for key in sorted(state_dict):
        value = state_dict[key]
        if not torch.is_tensor(value):
            continue
        value_hash = tensor_hash(value)
        per_tensor[key] = value_hash
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(b"\0")
        digest.update(value_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), per_tensor


def env_snapshot(machine_name: str) -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "machine_name": machine_name,
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch_version": str(torch.__version__),
        "torch_cuda": None if torch.version.cuda is None else str(torch.version.cuda),
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "zoology_branch": _git_value(REPO_ROOT, ["branch", "--show-current"]),
        "zoology_commit": _git_value(REPO_ROOT, ["rev-parse", "--short", "HEAD"]),
        "flash_vqg_branch": _git_value(FLASH_VQG_ROOT, ["branch", "--show-current"]),
        "flash_vqg_commit": _git_value(FLASH_VQG_ROOT, ["rev-parse", "--short", "HEAD"]),
    }


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_init(args: argparse.Namespace) -> int:
    config = build_config(
        target="default-s123-r1",
        machine_name=args.machine_name,
        logger_backend="none",
        trace_output_dir=SCRIPT_DIR / "outputs" / "init-traces" / args.machine_name,
        max_epochs=1,
        max_train_steps=None,
        max_validation_batches=None,
    )
    set_determinism(config.seed, deterministic=False)
    model = LanguageModel(config.model)
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    state_hash, per_tensor = state_dict_hashes(state)
    payload = {
        "model_state_dict": state,
        "model_state_sha256": state_hash,
        "per_tensor_sha256": per_tensor,
        "config": serialize_train_config(config),
        "env": env_snapshot(args.machine_name),
        "run_id": config.run_id,
        "launch_id": config.launch_id,
        "experiment_id": EXPERIMENT_ID,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    meta = {
        "checkpoint": str(args.output),
        "model_state_sha256": state_hash,
        "num_tensors": len(per_tensor),
        "env": payload["env"],
        "run_id": config.run_id,
        "launch_id": config.launch_id,
    }
    if args.meta_json:
        save_json(args.meta_json, meta)
    print(json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def verify_init(args: argparse.Namespace) -> int:
    payload = torch.load(args.checkpoint, map_location="cpu")
    state = payload["model_state_dict"]
    state_hash, per_tensor = state_dict_hashes(state)
    expected = payload.get("model_state_sha256")
    ok = expected == state_hash
    mismatched_keys = [
        key
        for key, value in payload.get("per_tensor_sha256", {}).items()
        if per_tensor.get(key) != value
    ]
    result = {
        "checkpoint": str(args.checkpoint),
        "expected_model_state_sha256": expected,
        "actual_model_state_sha256": state_hash,
        "match": bool(ok and not mismatched_keys),
        "mismatched_keys": mismatched_keys,
        "num_tensors": len(per_tensor),
        "env": env_snapshot(args.machine_name),
    }
    if args.output_json:
        save_json(args.output_json, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["match"] else 1


def preflight(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("torch.cuda is unavailable inside this container.")
    config = build_config(
        target=args.target,
        machine_name=args.machine_name,
        logger_backend="none",
        trace_output_dir=SCRIPT_DIR / "outputs" / "preflight-traces" / args.machine_name / args.target,
        max_epochs=1,
        max_train_steps=None,
        max_validation_batches=None,
    )
    train_loader, _ = prepare_data(config.data)
    train_batches = len(train_loader)
    accum = int(config.gradient_accumulation_steps)
    optim_steps = (train_batches + accum - 1) // accum
    result = {
        "env": env_snapshot(args.machine_name),
        "target": args.target,
        "run_id": config.run_id,
        "launch_id": config.launch_id,
        "train_batches": train_batches,
        "gradient_accumulation_steps": accum,
        "num_optimizer_steps": optim_steps,
        "cache_dir": config.data.cache_dir,
        "passed": train_batches == 2815 and accum == 4 and optim_steps == 704,
    }
    if args.output_json:
        save_json(args.output_json, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


def export_config(args: argparse.Namespace) -> int:
    config = build_config(
        target=args.target,
        machine_name=args.machine_name,
        logger_backend="none",
        trace_output_dir=SCRIPT_DIR / "outputs" / "probe-traces" / args.machine_name / args.target,
        max_epochs=1,
        max_train_steps=None,
        max_validation_batches=None,
    )
    payload = (
        "# -*- coding: utf-8 -*-\n"
        "from zoology.checkpoints import _load_data_segment_config\n"
        "from zoology.config import CheckpointConfig, DataConfig, LoggerConfig, ModelConfig, TrainConfig\n\n"
        f"_payload = {repr(serialize_train_config(config))}\n"
        "_data_payload = _payload['data']\n"
        "_data = DataConfig(\n"
        "    train_configs=[_load_data_segment_config(item) for item in _data_payload['train_configs']],\n"
        "    test_configs=[_load_data_segment_config(item) for item in _data_payload['test_configs']],\n"
        "    **{k: v for k, v in _data_payload.items() if k not in {'train_configs', 'test_configs'}},\n"
        ")\n"
        "configs = [TrainConfig(\n"
        "    data=_data,\n"
        "    model=ModelConfig.model_validate(_payload['model']),\n"
        "    logger=LoggerConfig.model_validate(_payload.get('logger', {})),\n"
        "    checkpoint=CheckpointConfig.model_validate(_payload.get('checkpoint', {})),\n"
        "    **{k: v for k, v in _payload.items() if k not in {'data', 'model', 'logger', 'checkpoint'}},\n"
        ")]\n"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


def run_train(args: argparse.Namespace) -> int:
    config = build_config(
        target=args.target,
        machine_name=args.machine_name,
        logger_backend=args.logger_backend,
        trace_output_dir=args.trace_output_dir,
        max_epochs=args.max_epochs,
        max_train_steps=args.max_train_steps,
        max_validation_batches=args.max_validation_batches,
    )
    config.init_checkpoint_path = str(args.init_checkpoint)
    config.init_checkpoint_strict = True
    config.init_checkpoint_source_launch_id = "canonical-init-2080ti"
    config.init_checkpoint_source_run_id = "initlock-cb64r16-default-s123-r1-d123-b64ga4-2080ti"
    if args.output_config_json:
        save_json(args.output_config_json, serialize_train_config(config))
    result = train(config)
    if args.output_result_json:
        save_json(
            args.output_result_json,
            {
                "machine_name": args.machine_name,
                "target": args.target,
                "init_checkpoint": str(args.init_checkpoint),
                "train_result": result,
                "env": env_snapshot(args.machine_name),
            },
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("make-init")
    p.add_argument("--machine-name", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--meta-json", type=Path)
    p.set_defaults(func=make_init)

    p = sub.add_parser("verify-init")
    p.add_argument("--machine-name", required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-json", type=Path)
    p.set_defaults(func=verify_init)

    p = sub.add_parser("preflight")
    p.add_argument("--machine-name", required=True)
    p.add_argument("--target", choices=["default-s123-r1", "default-s123-r2"], default="default-s123-r1")
    p.add_argument("--output-json", type=Path)
    p.set_defaults(func=preflight)

    p = sub.add_parser("export-config")
    p.add_argument("--machine-name", required=True)
    p.add_argument("--target", choices=["default-s123-r1", "default-s123-r2"], default="default-s123-r1")
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(func=export_config)

    p = sub.add_parser("train")
    p.add_argument("--machine-name", required=True)
    p.add_argument("--target", choices=["default-s123-r1", "default-s123-r2"], required=True)
    p.add_argument("--init-checkpoint", type=Path, required=True)
    p.add_argument("--trace-output-dir", type=Path, required=True)
    p.add_argument("--output-config-json", type=Path)
    p.add_argument("--output-result-json", type=Path)
    p.add_argument("--logger-backend", choices=["none", "swanlab", "wandb"], default="none")
    p.add_argument("--max-epochs", type=int, default=1)
    p.add_argument("--max-train-steps", type=int)
    p.add_argument("--max-validation-batches", type=int)
    p.set_defaults(func=run_train)

    args = parser.parse_args()
    os.chdir(REPO_ROOT)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
