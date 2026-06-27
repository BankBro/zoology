#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import platform
import random
import socket
import subprocess
import sys
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from einops import rearrange

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get("ZOOLOGY_REPO_ROOT", SCRIPT_DIR.parents[4])).resolve()
FLASH_VQG_ROOT = Path(
    os.environ.get("FLASH_VQG_ROOT", "/home/lyj/mnt/project/Flash-VQG")
).resolve()
EXPERIMENT_ID = "20260627-03-flash-vqg-first-divergence-probe"
PREVIOUS_EXPERIMENT_ID = "20260627-02-flash-vqg-canonical-init-lock-screen"
PREVIOUS_RUNNER = (
    REPO_ROOT
    / "zoology/experiments/flash_vqg/scripts"
    / PREVIOUS_EXPERIMENT_ID
    / "init_lock_screen.py"
)
DEFAULT_INIT_CHECKPOINT = (
    REPO_ROOT
    / "zoology/experiments/flash_vqg/scripts"
    / PREVIOUS_EXPERIMENT_ID
    / "outputs/canonical-init/cb64r16-s123-init.pt"
)
EXPECTED_CACHE_COMBINED_SHA256 = (
    "d9098e876a036b8cb90a7186174fd827e0f5b422482266772850069c905bd8c8"
)
EXPECTED_INIT_STATE_SHA256 = (
    "dd0b7bb57b8bbb1fd2bd6dbcfcaa82d609c318b9b1c9ead0e89032f17bf16edf"
)

if str(FLASH_VQG_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(FLASH_VQG_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from zoology.checkpoints import (  # noqa: E402
    load_checkpoint_payload,
    resolve_checkpoint_path,
    serialize_train_config,
)
from zoology.config import CheckpointConfig, LoggerConfig  # noqa: E402
from zoology.data.utils import prepare_continuous_data, prepare_data  # noqa: E402
from zoology.model import ContinuousInputModel, LanguageModel  # noqa: E402
from zoology.utils import set_determinism  # noqa: E402


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


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default)
        + "\n",
        encoding="utf-8",
    )


def _sha_update_text(hasher: Any, value: str) -> None:
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


def _hash_tensor_bytes(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().cpu().contiguous()
    return hashlib.sha256(cpu.numpy().tobytes()).hexdigest()


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


def _update_tensor_hash(hasher: Any, name: str, tensor: torch.Tensor) -> None:
    detached = tensor.detach().cpu().contiguous()
    _sha_update_text(hasher, name)
    _sha_update_text(hasher, str(detached.dtype))
    _sha_update_text(hasher, ",".join(str(dim) for dim in detached.shape))
    hasher.update(detached.numpy().tobytes())


def _hash_state_dict(state_dict: dict[str, torch.Tensor]) -> tuple[str, dict[str, str]]:
    hasher = hashlib.sha256()
    per_tensor: dict[str, str] = {}
    for name, tensor in sorted(state_dict.items()):
        if not torch.is_tensor(tensor):
            continue
        tensor_hash = _hash_tensor_bytes(tensor)
        per_tensor[name] = tensor_hash
        hasher.update(name.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(str(tuple(tensor.shape)).encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(str(tensor.dtype).encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(tensor_hash.encode("ascii"))
        hasher.update(b"\n")
    return hasher.hexdigest(), per_tensor


def _hash_model_params(model: torch.nn.Module) -> str:
    return _hash_state_dict(model.state_dict())[0]


def _hash_model_grads(model: torch.nn.Module) -> str:
    hasher = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        _sha_update_text(hasher, name)
        if parameter.grad is None:
            _sha_update_text(hasher, "NONE")
        else:
            _sha_update_text(hasher, _hash_tensor(parameter.grad))
    return hasher.hexdigest()


def _hash_optimizer_state(optimizer: optim.Optimizer) -> str:
    hasher = hashlib.sha256()
    state = optimizer.state_dict()
    _sha_update_text(
        hasher,
        json.dumps(state["param_groups"], sort_keys=True, default=_json_default),
    )
    for param_idx, values in sorted(state["state"].items(), key=lambda item: str(item[0])):
        _sha_update_text(hasher, str(param_idx))
        for key, value in sorted(values.items()):
            _sha_update_text(hasher, str(key))
            if isinstance(value, torch.Tensor):
                _sha_update_text(hasher, _hash_tensor(value))
            else:
                _sha_update_text(hasher, repr(value))
    return hasher.hexdigest()


def _tensor_summary(
    tensor: torch.Tensor,
    *,
    include_hash: bool = True,
    sample_count: int = 8,
) -> dict[str, Any]:
    detached = tensor.detach()
    cpu = detached.cpu().contiguous()
    flat = cpu.reshape(-1)
    result: dict[str, Any] = {
        "shape": list(cpu.shape),
        "dtype": str(cpu.dtype),
        "numel": int(cpu.numel()),
    }
    if include_hash:
        result["sha256"] = _hash_tensor(cpu)
    if cpu.numel() > 0:
        f32 = flat.float()
        result.update(
            {
                "mean": float(f32.mean().item()),
                "std": float(f32.std(unbiased=False).item()),
                "min": float(f32.min().item()),
                "max": float(f32.max().item()),
                "l2_norm": float(torch.linalg.vector_norm(f32).item()),
                "sample": [
                    _json_default(value)
                    for value in flat[: max(0, sample_count)].detach().cpu()
                ],
            }
        )
    return result


def _git_value(repo: Path, args: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def _env_snapshot(machine_name: str, variant: str | None = None) -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    return {
        "experiment_id": EXPERIMENT_ID,
        "machine_name": machine_name,
        "variant": variant,
        "utc_time": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch_version": torch.__version__,
        "torch_cuda": None if torch.version.cuda is None else str(torch.version.cuda),
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": cuda_available,
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "cuda_device_capability": (
            torch.cuda.get_device_capability(0) if cuda_available else None
        ),
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
        "zoology_branch": _git_value(REPO_ROOT, ["branch", "--show-current"]),
        "zoology_commit": _git_value(REPO_ROOT, ["rev-parse", "--short", "HEAD"]),
        "zoology_status_short": _git_value(REPO_ROOT, ["status", "--short"]),
        "flash_vqg_branch": _git_value(FLASH_VQG_ROOT, ["branch", "--show-current"]),
        "flash_vqg_commit": _git_value(FLASH_VQG_ROOT, ["rev-parse", "--short", "HEAD"]),
        "flash_vqg_status_short": _git_value(FLASH_VQG_ROOT, ["status", "--short"]),
    }


def _load_previous_runner():
    spec = importlib.util.spec_from_file_location("init_lock_screen_20260627_02", PREVIOUS_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load previous runner: {PREVIOUS_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _copy_config(config: Any) -> Any:
    if hasattr(config, "model_copy"):
        return config.model_copy(deep=True)
    return config.copy(deep=True)


def _walk_config_objects(value: Any):
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk_config_objects(child)
        return
    if isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_config_objects(child)
        return
    kwargs = getattr(value, "kwargs", None)
    if isinstance(kwargs, dict):
        yield from _walk_config_objects(kwargs)
    for attr in ("sequence_mixer", "state_mixer"):
        child = getattr(value, attr, None)
        if child is not None:
            yield from _walk_config_objects(child)


def _set_model_kwarg(model_config: Any, key: str, value: Any, *, require: bool = True) -> int:
    count = 0
    for node in _walk_config_objects(model_config):
        if isinstance(node, dict) and key in node:
            node[key] = value
            count += 1
    if require and count <= 0:
        raise KeyError(f"Did not find model kwarg `{key}` in config.")
    return count


def _find_flash_vqg_kwargs(model_config: Any) -> dict[str, Any]:
    for node in _walk_config_objects(model_config):
        if isinstance(node, dict) and node.get("fox_remote_formula") == "gd_residual_v1":
            return node
    return {}


def _build_probe_config(
    *,
    target: str,
    machine_name: str,
    variant: str,
    max_optimizer_steps: int,
) -> Any:
    previous = _load_previous_runner()
    config = previous.build_config(
        target=target,
        machine_name=machine_name,
        logger_backend="none",
        trace_output_dir=SCRIPT_DIR / "outputs" / "trace-disabled" / machine_name / variant,
        max_epochs=1,
        max_train_steps=max_optimizer_steps,
        max_validation_batches=1,
    )
    config = _copy_config(config)
    config.launch_id = f"fvqg-{EXPERIMENT_ID}-{machine_name}-{target}-{variant}"
    config.run_id = f"{EXPERIMENT_ID}-{machine_name}-{target}-{variant}"
    config.logger = LoggerConfig(backend="none", project_name=None, entity=None)
    config.checkpoint = CheckpointConfig(
        enabled=False,
        save_best=False,
        save_last=False,
        save_config_json=False,
    )
    config.read_churn_probe_enabled = False
    config.read_trace_enabled = False
    config.read_trace_train_steps = []
    config.read_trace_output_dir = None
    _set_model_kwarg(config.model, "enable_layer_metrics", True, require=False)
    _set_model_kwarg(config.model, "fox_phase2_metrics_mode", "full", require=False)
    if variant == "ref-gd":
        _set_model_kwarg(config.model, "fox_gd_residual_builder", "grouped_chunk_torch_ref")
        _set_model_kwarg(config.model, "fox_gd_residual_pack_mode", "loop_ref")
    return config


def _configure_runtime_for_variant(variant: str) -> None:
    if variant != "strict-fp32":
        return
    os.environ.setdefault("TORCH_DETERMINISTIC", "1")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _parse_int_set(value: str | None) -> set[int]:
    if value is None or not str(value).strip():
        return set()
    items: set[int] = set()
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        items.add(int(part))
    return items


def _cache_items_for_config(data_config: Any) -> list[dict[str, Any]]:
    cache_dir_raw = data_config.cache_dir
    if not cache_dir_raw:
        raise ValueError("config.data.cache_dir is empty.")
    cache_dir = Path(cache_dir_raw)
    if not cache_dir.is_absolute():
        cache_dir = REPO_ROOT / cache_dir
    max_seed = 2**32
    np.random.seed(int(data_config.seed))
    train_seeds = np.random.randint(0, max_seed // 2, size=len(data_config.train_configs))
    test_seeds = np.random.randint(max_seed // 2, max_seed, size=len(data_config.test_configs))

    items: list[dict[str, Any]] = []
    for role, configs, seeds in (
        ("train", data_config.train_configs, train_seeds),
        ("test", data_config.test_configs, test_seeds),
    ):
        for idx, (segment_config, seed) in enumerate(zip(configs, seeds)):
            segment_payload = segment_config.model_dump()
            cache_hash = hashlib.md5(
                json.dumps(
                    {**segment_payload, "_seed": int(seed)},
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
            path = cache_dir / f"data_{cache_hash}.pt"
            items.append(
                {
                    "role": role,
                    "segment_idx": idx,
                    "seed": int(seed),
                    "path": str(path),
                    "name": path.name,
                    "exists": path.exists(),
                    "segment": segment_payload,
                }
            )
    return items


def _hash_cache_for_config(data_config: Any) -> dict[str, Any]:
    items = _cache_items_for_config(data_config)
    combined = hashlib.sha256()
    hashed_items: list[dict[str, Any]] = []
    missing = [item for item in items if not item["exists"]]
    if missing:
        raise FileNotFoundError(
            "Missing required MQAR cache files: "
            + ", ".join(item["path"] for item in missing[:8])
        )
    for item in items:
        file_hash = _hash_cache_file(Path(item["path"]))
        full_item = {**item, **file_hash}
        hashed_items.append(full_item)
        _sha_update_text(combined, item["name"])
        _sha_update_text(combined, file_hash["content_sha256"])
    return {
        "cache_dir": str(Path(data_config.cache_dir)),
        "file_count": len(hashed_items),
        "combined_content_sha256": combined.hexdigest(),
        "expected_combined_content_sha256": EXPECTED_CACHE_COMBINED_SHA256,
        "match_expected": combined.hexdigest() == EXPECTED_CACHE_COMBINED_SHA256,
        "files": hashed_items,
    }


def _verify_init_checkpoint(path: Path) -> dict[str, Any]:
    resolved = resolve_checkpoint_path(path, which="best")
    payload = load_checkpoint_payload(resolved, map_location="cpu")
    state = payload["model_state_dict"]
    actual, per_tensor = _hash_state_dict(state)
    embedded = payload.get("model_state_sha256")
    return {
        "checkpoint": str(resolved),
        "expected_model_state_sha256": EXPECTED_INIT_STATE_SHA256,
        "embedded_model_state_sha256": embedded,
        "actual_model_state_sha256": actual,
        "match_expected": actual == EXPECTED_INIT_STATE_SHA256,
        "match_embedded": embedded == actual,
        "num_tensors": len(per_tensor),
    }


def _batch_order_hash(dataloader: Any) -> dict[str, Any]:
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


def _set_dense_teacher_runtime(model: torch.nn.Module, targets: torch.Tensor, input_type: str) -> None:
    if input_type != "discrete":
        return
    runtime = {"teacher_target_mask": (targets != -100).detach()}

    def setter(module: torch.nn.Module) -> None:
        setter_fn = getattr(module, "set_dense_teacher_runtime", None)
        if setter_fn is not None:
            setter_fn(runtime)

    model.apply(setter)


def _clear_dense_teacher_runtime(model: torch.nn.Module) -> None:
    def clearer(module: torch.nn.Module) -> None:
        clearer_fn = getattr(module, "clear_dense_teacher_runtime", None)
        if clearer_fn is not None:
            clearer_fn()

    model.apply(clearer)


def _compute_loss(
    model: torch.nn.Module,
    input_type: str,
    loss_type: str,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    loss_fn: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if input_type == "continuous":
        all_embeddings = model.backbone.embeddings.word_embeddings.weight
        vocab_size = all_embeddings.shape[0]
        value_embeddings = all_embeddings[vocab_size // 2 :]
        outputs = model(inputs)
        num_kv_pairs = targets.shape[1]
        outputs = outputs[:, -num_kv_pairs:]
        outputs_flat = outputs.reshape(-1, outputs.size(-1))
        targets_flat = targets.reshape(-1)
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

    def collect(module: torch.nn.Module) -> None:
        getter = getattr(module, "get_auxiliary_loss", None)
        if getter is not None:
            losses.append(getter())

    model.apply(collect)
    return sum(losses) if losses else 0


def _scalar_metrics(model: torch.nn.Module) -> dict[str, float]:
    metrics: dict[str, float] = {}

    def collect(module: torch.nn.Module) -> None:
        getter = getattr(module, "get_scalar_metrics", None)
        if getter is None:
            return
        values = getter()
        if not values:
            return
        for key, value in values.items():
            with suppress(Exception):
                metrics[str(key)] = float(value)

    model.apply(collect)
    return metrics


def _first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


class ForwardCapture:
    def __init__(self, model: torch.nn.Module, *, sample_count: int = 4):
        self.sample_count = int(sample_count)
        self._stage: str | None = None
        self._records: list[dict[str, Any]] = []
        self._start_idx = 0
        self._handles = []
        for name, module in model.named_modules():
            if self._should_capture(name):
                self._handles.append(module.register_forward_hook(self._make_hook(name)))

    @staticmethod
    def _should_capture(name: str) -> bool:
        if name == "backbone.embeddings":
            return True
        if name == "backbone.ln_f":
            return True
        suffixes = (
            ".norm1",
            ".norm2",
            ".sequence_mixer",
            ".sequence_mixer.mixer",
            ".state_mixer",
        )
        return any(name.endswith(suffix) for suffix in suffixes)

    def _make_hook(self, name: str):
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            if self._stage is None:
                return
            tensor = _first_tensor(output)
            if tensor is None:
                return
            self._records.append(
                {
                    "stage": self._stage,
                    "module": name,
                    **_tensor_summary(
                        tensor,
                        include_hash=True,
                        sample_count=self.sample_count,
                    ),
                }
            )

        return hook

    def begin(self, stage: str) -> None:
        self._stage = stage
        self._start_idx = len(self._records)

    def end(self) -> list[dict[str, Any]]:
        records = self._records[self._start_idx :]
        self._stage = None
        self._start_idx = len(self._records)
        return records

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def _set_shadow_dense_runtime_flags(
    model: torch.nn.Module,
    *,
    enabled: bool,
    chunk_size: int,
) -> int:
    count = 0
    seen: set[int] = set()
    for module in model.modules():
        candidates = [getattr(module, "config", None)]
        attn = getattr(module, "attn", None)
        if attn is not None:
            candidates.append(getattr(attn, "config", None))
        for cfg in candidates:
            if cfg is None or id(cfg) in seen:
                continue
            seen.add(id(cfg))
            if not hasattr(cfg, "fox_remote_formula"):
                continue
            setattr(cfg, "_fox_gd_residual_shadow_dense_read_metrics", bool(enabled))
            setattr(cfg, "_fox_gd_residual_shadow_dense_read_chunk_size", int(chunk_size))
            count += 1
    return count


def _prediction_margin_summary(
    logits: torch.Tensor,
    preds: torch.Tensor,
    targets: torch.Tensor,
    *,
    max_positions: int = 16,
) -> dict[str, Any]:
    mask = targets != -100
    valid = mask.nonzero(as_tuple=False)
    result: dict[str, Any] = {
        "query_positions": int(valid.size(0)),
        "accuracy": None,
        "samples": [],
    }
    if valid.numel() == 0:
        return result
    result["accuracy"] = float((preds[mask] == targets[mask]).float().mean().item())
    samples = []
    for row in valid[:max_positions]:
        b = int(row[0].item())
        t = int(row[1].item())
        target = int(targets[b, t].item())
        scores = logits[b, t].detach().float()
        top = torch.topk(scores, k=2)
        best_other = top.values[1] if int(top.indices[0].item()) == target else top.values[0]
        correct = scores[target]
        samples.append(
            {
                "batch_idx": b,
                "token_idx": t,
                "target": target,
                "pred": int(preds[b, t].item()),
                "correct_logit": float(correct.item()),
                "best_other_logit": float(best_other.item()),
                "margin": float((correct - best_other).item()),
            }
        )
    result["samples"] = samples
    return result


def _config_summary(config: Any, variant: str) -> dict[str, Any]:
    flash_kwargs = _find_flash_vqg_kwargs(config.model)
    return {
        "variant": variant,
        "seed": int(config.seed),
        "data_seed": int(config.data.seed),
        "run_id": config.run_id,
        "launch_id": config.launch_id,
        "batch_size": config.data.batch_size,
        "train_batch_order": config.data.train_batch_order,
        "gradient_accumulation_steps": int(config.gradient_accumulation_steps),
        "learning_rate": float(config.learning_rate),
        "weight_decay": float(config.weight_decay),
        "loss_type": config.loss_type,
        "cache_dir": config.data.cache_dir,
        "flash_vqg": {
            key: flash_kwargs.get(key)
            for key in [
                "fox_remote_read_topk",
                "fox_remote_formula",
                "fox_gd_residual_rank",
                "fox_gd_residual_write_topk",
                "fox_gd_residual_builder",
                "fox_gd_residual_pack_mode",
                "fox_gd_residual_chunk_size",
                "enable_layer_metrics",
                "fox_phase2_metrics_mode",
                "attn_backend",
                "fox_remote_path_backend",
            ]
        },
        "train_segments": [
            segment.model_dump() if hasattr(segment, "model_dump") else dict(segment)
            for segment in config.data.train_configs
        ],
        "test_segments": [
            segment.model_dump() if hasattr(segment, "model_dump") else dict(segment)
            for segment in config.data.test_configs
        ],
    }


def run_cache_hash(args: argparse.Namespace) -> int:
    config = _build_probe_config(
        target=args.target,
        machine_name=args.machine_name,
        variant="baseline",
        max_optimizer_steps=1,
    )
    payload = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "machine_name": args.machine_name,
        "target": args.target,
        "environment": _env_snapshot(args.machine_name),
        "cache": _hash_cache_for_config(config.data),
    }
    if args.output_json:
        _save_json(args.output_json, payload)
    print(json.dumps(payload["cache"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if payload["cache"]["match_expected"] else 1


def run_verify_init(args: argparse.Namespace) -> int:
    payload = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "machine_name": args.machine_name,
        "environment": _env_snapshot(args.machine_name),
        "init_checkpoint": _verify_init_checkpoint(args.checkpoint),
    }
    if args.output_json:
        _save_json(args.output_json, payload)
    print(json.dumps(payload["init_checkpoint"], ensure_ascii=False, indent=2, sort_keys=True))
    init = payload["init_checkpoint"]
    return 0 if init["match_expected"] and init["match_embedded"] else 1


def run_probe(args: argparse.Namespace) -> int:
    if args.max_optimizer_steps <= 0:
        raise ValueError("--max-optimizer-steps must be positive.")
    os.chdir(REPO_ROOT)
    _configure_runtime_for_variant(args.variant)

    random.seed(args.pre_seed)
    np.random.seed(args.pre_seed)
    torch.manual_seed(args.pre_seed)

    config = _build_probe_config(
        target=args.target,
        machine_name=args.machine_name,
        variant=args.variant,
        max_optimizer_steps=args.max_optimizer_steps,
    )
    config.init_checkpoint_path = str(args.init_checkpoint)
    config.init_checkpoint_strict = True
    set_determinism(config.seed, deterministic=args.variant == "strict-fp32")

    cache_payload = _hash_cache_for_config(config.data)
    if not cache_payload["match_expected"]:
        raise RuntimeError(
            "MQAR cache content hash does not match canonical hash; stop before training."
        )
    init_payload = _verify_init_checkpoint(args.init_checkpoint)
    if not init_payload["match_expected"] or not init_payload["match_embedded"]:
        raise RuntimeError(
            "Init checkpoint tensor hash does not match canonical hash; stop before training."
        )

    if config.input_type == "continuous":
        model = ContinuousInputModel(config.model)
        train_dataloader, _test_dataloader = prepare_continuous_data(
            config.data,
            embeddings=model.backbone.embeddings.word_embeddings.weight.detach(),
        )
    else:
        model = LanguageModel(config.model)
        train_dataloader, _test_dataloader = prepare_data(config.data)

    resolved_init = resolve_checkpoint_path(args.init_checkpoint, which="best")
    checkpoint_payload = load_checkpoint_payload(resolved_init, map_location="cpu")
    model.load_state_dict(
        checkpoint_payload["model_state_dict"],
        strict=bool(config.init_checkpoint_strict),
    )
    after_load_hash = _hash_model_params(model)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false.")
    model.to(device)
    shadow_flag_count = _set_shadow_dense_runtime_flags(
        model,
        enabled=args.variant == "shadow-read",
        chunk_size=args.shadow_dense_chunk_size,
    )
    model.train()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    capture_steps = _parse_int_set(args.capture_optimizer_steps)
    capture = ForwardCapture(model, sample_count=args.capture_sample_count)
    records: list[dict[str, Any]] = [
        {
            "record_type": "state_hash",
            "stage": "after_init_checkpoint_load_before_to_device",
            "optimizer_step": 0,
            "micro_step": 0,
            "model_params_sha256": after_load_hash,
        },
        {
            "record_type": "state_hash",
            "stage": "after_model_to_device_before_optimizer_step",
            "optimizer_step": 0,
            "micro_step": 0,
            "model_params_sha256": _hash_model_params(model),
        },
    ]

    sampler = getattr(train_dataloader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(0)

    accum_steps = int(config.gradient_accumulation_steps)
    optimizer.zero_grad()
    accum_loss = 0.0
    optimizer_step = 0
    micro_step = 0
    completed_optimizer_steps = 0

    try:
        for inputs, targets, slices in train_dataloader:
            window_micro_idx = micro_step % accum_steps
            capture_this_forward = optimizer_step in capture_steps and window_micro_idx == 0
            stage = f"forward_before_backward_step{optimizer_step}_micro{micro_step}"
            inputs = inputs.to(device)
            targets = targets.to(device)
            _set_dense_teacher_runtime(model, targets, config.input_type)
            if capture_this_forward:
                capture.begin(stage)
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
            module_records = capture.end() if capture_this_forward else []

            aux_loss_value: torch.Tensor | int = 0
            if config.input_type == "discrete":
                aux_loss_value = _auxiliary_loss(model)
                if aux_loss_value:
                    loss = loss + aux_loss_value

            if capture_this_forward:
                records.append(
                    {
                        "record_type": "forward",
                        "stage": stage,
                        "optimizer_step": int(optimizer_step),
                        "micro_step": int(micro_step),
                        "window_micro_idx": int(window_micro_idx),
                        "loss": float(loss.detach().cpu().item()),
                        "aux_loss": (
                            float(aux_loss_value.detach().cpu().item())
                            if isinstance(aux_loss_value, torch.Tensor)
                            else float(aux_loss_value)
                        ),
                        "inputs_sha256": _hash_tensor(inputs),
                        "targets_sha256": _hash_tensor(targets),
                        "logits_sha256": _hash_tensor(logits),
                        "preds_sha256": _hash_tensor(preds),
                        "inputs": _tensor_summary(
                            inputs,
                            include_hash=False,
                            sample_count=args.capture_sample_count,
                        ),
                        "targets": _tensor_summary(
                            targets,
                            include_hash=False,
                            sample_count=args.capture_sample_count,
                        ),
                        "logits": _tensor_summary(
                            logits,
                            include_hash=False,
                            sample_count=args.capture_sample_count,
                        ),
                        "slice0": slices[0] if slices else {},
                        "prediction_margins": _prediction_margin_summary(
                            logits,
                            preds,
                            targets,
                            max_positions=args.margin_sample_positions,
                        ),
                        "scalar_metrics": _scalar_metrics(model),
                        "module_records": module_records,
                    }
                )

            (loss / accum_steps).backward()
            accum_loss += float(loss.detach().cpu().item())
            micro_step += 1

            if micro_step <= args.hash_micro_steps:
                records.append(
                    {
                        "record_type": "backward",
                        "stage": "after_microbatch_backward",
                        "optimizer_step": int(optimizer_step),
                        "micro_step": int(micro_step),
                        "loss": float(loss.detach().cpu().item()),
                        "grad_sha256": _hash_model_grads(model),
                        "model_params_sha256": _hash_model_params(model),
                        "scalar_metrics": _scalar_metrics(model),
                    }
                )

            if micro_step % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                optimizer_step += 1
                completed_optimizer_steps = optimizer_step
                records.append(
                    {
                        "record_type": "optimizer_step",
                        "stage": "after_optimizer_step",
                        "optimizer_step": int(optimizer_step),
                        "micro_step": int(micro_step),
                        "avg_loss": float(accum_loss / accum_steps),
                        "model_params_sha256": _hash_model_params(model),
                        "optimizer_state_sha256": _hash_optimizer_state(optimizer),
                        "scalar_metrics": _scalar_metrics(model),
                    }
                )
                accum_loss = 0.0
                if optimizer_step >= args.max_optimizer_steps:
                    break
    finally:
        capture.close()

    payload = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "probe": {
            "machine_name": args.machine_name,
            "variant": args.variant,
            "target": args.target,
            "device": str(device),
            "max_optimizer_steps": int(args.max_optimizer_steps),
            "hash_micro_steps": int(args.hash_micro_steps),
            "capture_optimizer_steps": sorted(capture_steps),
            "shadow_dense_flag_count": int(shadow_flag_count),
            "init_checkpoint": str(resolved_init),
        },
        "environment": _env_snapshot(args.machine_name, args.variant),
        "config_summary": _config_summary(config, args.variant),
        "serialized_config": serialize_train_config(config),
        "cache": cache_payload,
        "init_checkpoint": init_payload,
        "batch_order": _batch_order_hash(train_dataloader),
        "records": records,
        "completed_optimizer_steps": int(completed_optimizer_steps),
    }
    _save_json(args.output_json, payload)
    print(f"wrote {args.output_json}")
    return 0


def _iter_hash_rows(payload: dict[str, Any], source: str):
    probe = payload.get("probe", {})
    machine = probe.get("machine_name") or payload.get("machine_name")
    variant = probe.get("variant")
    target = probe.get("target")

    cache = payload.get("cache") or {}
    if cache.get("combined_content_sha256"):
        yield {
            "source": source,
            "machine": machine,
            "variant": variant,
            "target": target,
            "stage": "preflight",
            "optimizer_step": "",
            "micro_step": "",
            "field": "cache_combined_content_sha256",
            "module": "",
            "sha256": cache["combined_content_sha256"],
        }

    init = payload.get("init_checkpoint") or {}
    if init.get("actual_model_state_sha256"):
        yield {
            "source": source,
            "machine": machine,
            "variant": variant,
            "target": target,
            "stage": "preflight",
            "optimizer_step": "",
            "micro_step": "",
            "field": "init_model_state_sha256",
            "module": "",
            "sha256": init["actual_model_state_sha256"],
        }

    batch_order = payload.get("batch_order") or {}
    if batch_order.get("sha256"):
        yield {
            "source": source,
            "machine": machine,
            "variant": variant,
            "target": target,
            "stage": "preflight",
            "optimizer_step": "",
            "micro_step": "",
            "field": "batch_order_sha256",
            "module": "",
            "sha256": batch_order["sha256"],
        }

    for record in payload.get("records", []):
        base = {
            "source": source,
            "machine": machine,
            "variant": variant,
            "target": target,
            "stage": record.get("stage"),
            "optimizer_step": record.get("optimizer_step", ""),
            "micro_step": record.get("micro_step", ""),
        }
        for field in (
            "model_params_sha256",
            "grad_sha256",
            "optimizer_state_sha256",
            "inputs_sha256",
            "targets_sha256",
            "logits_sha256",
            "preds_sha256",
        ):
            if record.get(field):
                yield {**base, "field": field, "module": "", "sha256": record[field]}
        for module_record in record.get("module_records", []) or []:
            if module_record.get("sha256"):
                yield {
                    **base,
                    "field": "module_output_sha256",
                    "module": module_record.get("module", ""),
                    "sha256": module_record["sha256"],
                }


def run_compare(args: argparse.Namespace) -> int:
    rows: list[dict[str, Any]] = []
    for path in args.inputs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(_iter_hash_rows(payload, str(path)))

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["variant"],
            row["target"],
            row["stage"],
            str(row["optimizer_step"]),
            str(row["micro_step"]),
            row["field"],
            row["module"],
        )
        groups.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for key, members in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        values = {str(member["source"]): member["sha256"] for member in members}
        unique = sorted(set(values.values()))
        summary_rows.append(
            {
                "variant": key[0],
                "target": key[1],
                "stage": key[2],
                "optimizer_step": key[3],
                "micro_step": key[4],
                "field": key[5],
                "module": key[6],
                "source_count": len(members),
                "unique_sha256_count": len(unique),
                "all_match": len(unique) <= 1,
                "values_json": json.dumps(values, sort_keys=True),
            }
        )

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "variant",
                    "target",
                    "stage",
                    "optimizer_step",
                    "micro_step",
                    "field",
                    "module",
                    "source_count",
                    "unique_sha256_count",
                    "all_match",
                    "values_json",
                ],
            )
            writer.writeheader()
            writer.writerows(summary_rows)

    mismatch_rows = [row for row in summary_rows if not row["all_match"]]
    payload = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "input_count": len(args.inputs),
        "hash_row_count": len(rows),
        "comparison_row_count": len(summary_rows),
        "mismatch_row_count": len(mismatch_rows),
        "first_mismatch": mismatch_rows[0] if mismatch_rows else None,
        "all_match": not mismatch_rows,
    }
    if args.output_json:
        _save_json(args.output_json, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("cache-hash")
    p.add_argument("--machine-name", required=True)
    p.add_argument("--target", choices=["default-s123-r1", "default-s123-r2"], default="default-s123-r1")
    p.add_argument("--output-json", type=Path)
    p.set_defaults(func=run_cache_hash)

    p = sub.add_parser("verify-init")
    p.add_argument("--machine-name", required=True)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_INIT_CHECKPOINT)
    p.add_argument("--output-json", type=Path)
    p.set_defaults(func=run_verify_init)

    p = sub.add_parser("probe")
    p.add_argument("--machine-name", required=True)
    p.add_argument("--variant", choices=["baseline", "strict-fp32", "shadow-read", "ref-gd"], required=True)
    p.add_argument("--target", choices=["default-s123-r1", "default-s123-r2"], default="default-s123-r1")
    p.add_argument("--init-checkpoint", type=Path, default=DEFAULT_INIT_CHECKPOINT)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--max-optimizer-steps", type=int, default=1)
    p.add_argument("--hash-micro-steps", type=int, default=4)
    p.add_argument("--capture-optimizer-steps", default="0")
    p.add_argument("--capture-sample-count", type=int, default=4)
    p.add_argument("--margin-sample-positions", type=int, default=16)
    p.add_argument("--shadow-dense-chunk-size", type=int, default=8)
    p.add_argument("--pre-seed", type=int, default=99991)
    p.set_defaults(func=run_probe)

    p = sub.add_parser("compare")
    p.add_argument("--inputs", type=Path, nargs="+", required=True)
    p.add_argument("--output-csv", type=Path)
    p.add_argument("--output-json", type=Path)
    p.set_defaults(func=run_compare)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    os.chdir(REPO_ROOT)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
