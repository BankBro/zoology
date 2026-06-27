#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import random
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from einops import rearrange

from zoology.checkpoints import load_checkpoint_payload, resolve_checkpoint_path
from zoology.config import CheckpointConfig, LoggerConfig
from zoology.data.utils import prepare_continuous_data, prepare_data
from zoology.model import ContinuousInputModel, LanguageModel
from zoology.utils import set_determinism


DEFAULT_CONFIG = (
    "zoology/experiments/flash_vqg/generated/"
    "fvqg-20260626-01-cache-sync-rerun-3090-default-s123-r1-2026-06-26-05-42-44/"
    "launch_configs.py"
)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _copy_config(config):
    return config.model_copy(deep=True) if hasattr(config, "model_copy") else config.copy(deep=True)


def _load_config(config_path: Path, index: int):
    spec = importlib.util.spec_from_file_location("early_step_probe_config", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load config module from {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    configs = getattr(module, "configs")
    if not configs:
        raise RuntimeError(f"No configs found in {config_path}")
    return _copy_config(configs[index])


def _sha_update_text(hasher: "hashlib._Hash", value: str) -> None:
    encoded = value.encode("utf-8")
    hasher.update(len(encoded).to_bytes(8, "little"))
    hasher.update(encoded)


def _hash_tensor(tensor: torch.Tensor) -> str:
    hasher = hashlib.sha256()
    detached = tensor.detach().cpu().contiguous()
    _sha_update_text(hasher, str(detached.dtype))
    _sha_update_text(hasher, ",".join(str(dim) for dim in detached.shape))
    hasher.update(detached.numpy().tobytes())
    return hasher.hexdigest()


def _update_tensor_hash(hasher: "hashlib._Hash", name: str, tensor: torch.Tensor) -> None:
    detached = tensor.detach().cpu().contiguous()
    _sha_update_text(hasher, name)
    _sha_update_text(hasher, str(detached.dtype))
    _sha_update_text(hasher, ",".join(str(dim) for dim in detached.shape))
    hasher.update(detached.numpy().tobytes())


def _hash_state_dict(state_dict: dict[str, torch.Tensor]) -> str:
    hasher = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        _update_tensor_hash(hasher, name, tensor)
    return hasher.hexdigest()


def _hash_model_params(model: torch.nn.Module) -> str:
    return _hash_state_dict(model.state_dict())


def _hash_model_grads(model: torch.nn.Module) -> str:
    hasher = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        grad = parameter.grad
        if grad is None:
            continue
        _update_tensor_hash(hasher, name, grad)
    return hasher.hexdigest()


def _hash_optimizer_state(optimizer: optim.Optimizer) -> str:
    hasher = hashlib.sha256()
    state = optimizer.state_dict()
    _sha_update_text(hasher, json.dumps(state["param_groups"], sort_keys=True, default=_json_default))
    for param_idx, values in sorted(state["state"].items(), key=lambda item: str(item[0])):
        _sha_update_text(hasher, str(param_idx))
        for key, value in sorted(values.items()):
            _sha_update_text(hasher, str(key))
            if isinstance(value, torch.Tensor):
                _update_tensor_hash(hasher, str(key), value)
            else:
                _sha_update_text(hasher, repr(value))
    return hasher.hexdigest()


def _hash_cache_file(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    hasher = hashlib.sha256()
    for key in sorted(payload):
        value = payload[key]
        _sha_update_text(hasher, key)
        if isinstance(value, torch.Tensor):
            _update_tensor_hash(hasher, key, value)
        else:
            _sha_update_text(hasher, json.dumps(value, sort_keys=True, default=_json_default))
    return {
        "path": str(path),
        "name": path.name,
        "bytes": path.stat().st_size,
        "content_sha256": hasher.hexdigest(),
    }


def _hash_cache_dir(cache_dir: Path) -> dict[str, Any]:
    files = sorted(cache_dir.glob("data_*.pt"))
    hasher = hashlib.sha256()
    items = []
    for path in files:
        item = _hash_cache_file(path)
        items.append(item)
        _sha_update_text(hasher, item["name"])
        _sha_update_text(hasher, item["content_sha256"])
    return {
        "cache_dir": str(cache_dir),
        "file_count": len(items),
        "combined_content_sha256": hasher.hexdigest(),
        "files": items,
    }


def _git_commit(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _git_status(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "status", "--short"],
            cwd=repo,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None


def _environment_payload(repo: Path) -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    return {
        "utc_time": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": cuda_available,
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "cuda_device_capability": torch.cuda.get_device_capability(0) if cuda_available else None,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "env": {
            key: os.environ.get(key)
            for key in [
                "CUDA_VISIBLE_DEVICES",
                "TORCH_DETERMINISTIC",
                "CUBLAS_WORKSPACE_CONFIG",
                "PYTHONHASHSEED",
            ]
        },
        "git_commit": _git_commit(repo),
        "git_status_short": _git_status(repo),
    }


def _configure_runtime(args: argparse.Namespace) -> None:
    if args.disable_cudnn_tf32:
        torch.backends.cudnn.allow_tf32 = False
    if args.disable_matmul_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
    if args.float32_matmul_precision:
        torch.set_float32_matmul_precision(args.float32_matmul_precision)


def _set_dense_teacher_runtime(model: torch.nn.Module, targets: torch.Tensor, input_type: str) -> None:
    if input_type != "discrete":
        return
    runtime = {"teacher_target_mask": (targets != -100).detach()}

    def setter(module):
        setter_fn = getattr(module, "set_dense_teacher_runtime", None)
        if setter_fn is not None:
            setter_fn(runtime)

    model.apply(setter)


def _clear_dense_teacher_runtime(model: torch.nn.Module) -> None:
    def clearer(module):
        clearer_fn = getattr(module, "clear_dense_teacher_runtime", None)
        if clearer_fn is not None:
            clearer_fn()

    model.apply(clearer)


def _compute_loss(model: torch.nn.Module, input_type: str, loss_type: str, inputs, targets, loss_fn):
    if input_type == "continuous":
        all_embeddings = model.backbone.embeddings.word_embeddings.weight
        vocab_size = all_embeddings.shape[0]
        value_embeddings = all_embeddings[vocab_size // 2:]
        outputs = model(inputs)
        num_kv_pairs = targets.shape[1]
        outputs = outputs[:, -num_kv_pairs:]
        outputs_flat = outputs.reshape(-1, outputs.size(-1))
        targets_flat = targets.reshape(-1)
        if loss_type != "ce":
            raise NotImplementedError("continuous probe only supports ce loss")
        logits = outputs_flat @ value_embeddings.T
        loss = loss_fn(logits, targets_flat)
        preds = logits.argmax(dim=-1).view(targets.shape)
        return loss, preds, logits

    if loss_type == "ce":
        logits = model(inputs)
        loss = loss_fn(rearrange(logits, "... c -> (...) c"), targets.flatten())
        preds = logits.argmax(dim=-1)
        return loss, preds, logits
    if loss_type == "mse":
        embeddings = model(inputs, return_embeddings=True)
        target_embeds = model.backbone.embeddings.word_embeddings(targets)
        mask = (targets != -100).unsqueeze(-1)
        loss = torch.nn.functional.mse_loss(
            embeddings[mask.expand_as(embeddings)].view(-1, embeddings.size(-1)),
            target_embeds[mask.expand_as(target_embeds)].view(-1, target_embeds.size(-1)),
        )
        logits = embeddings @ model.backbone.embeddings.word_embeddings.weight.T
        preds = logits.argmax(dim=-1)
        return loss, preds, logits
    if loss_type == "ce_embed":
        embeddings = model(inputs, return_embeddings=True)
        value_embeddings = model.backbone.embeddings.word_embeddings.weight
        flat_embeds = rearrange(embeddings, "b s d -> (b s) d")
        flat_targets = targets.flatten()
        mask = flat_targets != -100
        logits_for_loss = flat_embeds[mask] @ value_embeddings.T
        loss = loss_fn(logits_for_loss, flat_targets[mask])
        logits = embeddings @ value_embeddings.T
        preds = logits.argmax(dim=-1)
        return loss, preds, logits
    raise ValueError(f"Unsupported loss_type: {loss_type}")


def _auxiliary_loss(model: torch.nn.Module) -> torch.Tensor | int:
    losses = []

    def collect(module):
        getter = getattr(module, "get_auxiliary_loss", None)
        if getter is not None:
            losses.append(getter())

    model.apply(collect)
    return sum(losses) if losses else 0


def _scalar_metrics(model: torch.nn.Module) -> dict[str, float]:
    metrics: dict[str, float] = {}

    def collect(module):
        getter = getattr(module, "get_scalar_metrics", None)
        if getter is None:
            return
        values = getter()
        if not values:
            return
        for key, value in values.items():
            metrics[str(key)] = float(value)

    model.apply(collect)
    return metrics


def _batch_order_hash(dataloader) -> dict[str, Any]:
    sampler = getattr(dataloader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(0)
    order = list(iter(sampler)) if sampler is not None else list(range(len(dataloader)))
    hasher = hashlib.sha256()
    for item in order:
        hasher.update(int(item).to_bytes(8, "little", signed=True))
    return {
        "num_batches": len(order),
        "sha256": hasher.hexdigest(),
        "first_16": order[:16],
    }


def _prepare_config(args: argparse.Namespace, repo: Path):
    config = _load_config((repo / args.config).resolve(), args.config_index)
    config = _copy_config(config)
    config.run_id = args.run_id
    config.launch_id = args.launch_id
    config.logger = LoggerConfig(backend="none", project_name=None, entity=None)
    config.checkpoint = CheckpointConfig(enabled=False)
    config.read_churn_probe_enabled = False
    config.read_trace_enabled = False
    config.read_trace_train_steps = []
    config.read_trace_output_dir = None
    if args.init_checkpoint_path is not None:
        config.init_checkpoint_path = args.init_checkpoint_path
        config.init_checkpoint_strict = True
    return config


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    repo = Path(args.repo).resolve()
    os.chdir(repo)
    _configure_runtime(args)

    if args.deterministic:
        os.environ.setdefault("TORCH_DETERMINISTIC", "1")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(args.pre_seed)
    np.random.seed(args.pre_seed)
    torch.manual_seed(args.pre_seed)

    config = _prepare_config(args, repo)
    set_determinism(config.seed, deterministic=args.deterministic)

    if config.input_type == "continuous":
        model = ContinuousInputModel(config.model)
        train_dataloader, test_dataloader = prepare_continuous_data(
            config.data,
            embeddings=model.backbone.embeddings.word_embeddings.weight.detach(),
        )
    else:
        model = LanguageModel(config.model)
        train_dataloader, test_dataloader = prepare_data(config.data)

    if config.init_checkpoint_path is not None:
        resolved_init_checkpoint = resolve_checkpoint_path(config.init_checkpoint_path, which="best")
        payload = load_checkpoint_payload(resolved_init_checkpoint, map_location="cpu")
        model.load_state_dict(payload["model_state_dict"], strict=bool(config.init_checkpoint_strict))

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    records: list[dict[str, Any]] = []
    records.append({
        "stage": "after_model_to_device_before_optimizer_step",
        "optimizer_step": 0,
        "micro_step": 0,
        "model_params_sha256": _hash_model_params(model),
    })

    sampler = getattr(train_dataloader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(0)

    accum_steps = int(config.gradient_accumulation_steps)
    optimizer.zero_grad()
    accum_loss = 0.0
    optimizer_step = 0
    micro_step = 0
    first_forward_done = False

    for inputs, targets, slices in train_dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        _set_dense_teacher_runtime(model, targets, config.input_type)
        try:
            loss, preds, logits = _compute_loss(
                model,
                config.input_type,
                config.loss_type,
                inputs,
                targets,
                loss_fn,
            )
        finally:
            _clear_dense_teacher_runtime(model)

        if config.input_type == "discrete":
            aux = _auxiliary_loss(model)
            if aux:
                loss = loss + aux

        if not first_forward_done:
            records.append({
                "stage": "first_microbatch_forward_before_backward",
                "optimizer_step": int(optimizer_step),
                "micro_step": int(micro_step),
                "loss": float(loss.detach().cpu().item()),
                "inputs_sha256": _hash_tensor(inputs),
                "targets_sha256": _hash_tensor(targets),
                "logits_sha256": _hash_tensor(logits),
                "preds_sha256": _hash_tensor(preds),
                "slice0": slices[0] if slices else {},
                "scalar_metrics": _scalar_metrics(model),
            })
            first_forward_done = True

        effective_accum = accum_steps
        (loss / effective_accum).backward()
        accum_loss += float(loss.detach().cpu().item())
        micro_step += 1

        records.append({
            "stage": "after_microbatch_backward",
            "optimizer_step": int(optimizer_step),
            "micro_step": int(micro_step),
            "loss": float(loss.detach().cpu().item()),
            "grad_sha256": _hash_model_grads(model),
            "model_params_sha256": _hash_model_params(model),
            "scalar_metrics": _scalar_metrics(model),
        })

        if micro_step % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            optimizer_step += 1
            records.append({
                "stage": "after_optimizer_step",
                "optimizer_step": int(optimizer_step),
                "micro_step": int(micro_step),
                "avg_loss": float(accum_loss / accum_steps),
                "model_params_sha256": _hash_model_params(model),
                "optimizer_state_sha256": _hash_optimizer_state(optimizer),
                "scalar_metrics": _scalar_metrics(model),
            })
            accum_loss = 0.0
            if optimizer_step >= args.max_optimizer_steps:
                break

    cache_dir = Path(config.data.cache_dir or "")
    cache_payload = _hash_cache_dir((repo / cache_dir).resolve()) if config.data.cache_dir else None
    return {
        "schema_version": 1,
        "probe": {
            "name": args.probe_name,
            "run_id": args.run_id,
            "launch_id": args.launch_id,
            "config": str((repo / args.config).resolve()),
            "config_index": args.config_index,
            "device": str(device),
            "deterministic": args.deterministic,
            "disable_cudnn_tf32": args.disable_cudnn_tf32,
            "disable_matmul_tf32": args.disable_matmul_tf32,
            "max_optimizer_steps": args.max_optimizer_steps,
        },
        "environment": _environment_payload(repo),
        "config_summary": {
            "seed": config.seed,
            "data_seed": config.data.seed,
            "run_id": config.run_id,
            "model_name": config.model.name,
            "batch_size": config.data.batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "loss_type": config.loss_type,
            "train_batch_order": config.data.train_batch_order,
            "cache_dir": config.data.cache_dir,
            "train_segments": [
                segment.model_dump() if hasattr(segment, "model_dump") else dict(segment)
                for segment in config.data.train_configs
            ],
        },
        "cache": cache_payload,
        "batch_order": _batch_order_hash(train_dataloader),
        "records": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--config-index", type=int, default=0)
    parser.add_argument("--probe-name", required=True)
    parser.add_argument("--run-id", default="early-step-hash-probe")
    parser.add_argument("--launch-id", default="20260627-01-flash-vqg-early-step-hash-probe")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--init-checkpoint-path", default=None)
    parser.add_argument("--max-optimizer-steps", type=int, default=2)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--disable-cudnn-tf32", action="store_true")
    parser.add_argument("--disable-matmul-tf32", action="store_true")
    parser.add_argument("--float32-matmul-precision", default=None)
    parser.add_argument("--pre-seed", type=int, default=99991)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_probe(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default)
        f.write("\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
