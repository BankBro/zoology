#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# FLA otherwise enables Triton TF32 implicitly on Ampere even when PyTorch's
# matmul TF32 flags are disabled. Set this before importing torch/FLA modules.
os.environ.setdefault("TRITON_F32_DEFAULT", "ieee")

import numpy as np
import torch
import torch.nn as nn


EXPERIMENT_ID = "20260724-01-flash-vqg-gd-residual-efficiency"
REPO_ROOT = Path(os.environ.get("ZOOLOGY_REPO_ROOT", "/home/lyj/mnt/project/zoology")).resolve()
FLASH_ROOT = Path(os.environ.get("FLASH_VQG_ROOT", "/home/lyj/mnt/project/Flash-VQG")).resolve()
SCRIPT_DIR = Path(__file__).resolve().parent
SUPPORT_SCRIPT = (
    REPO_ROOT
    / "zoology/experiments/flash_vqg/scripts/"
    / "20260709-01-flash-vqg-default-dropout-r16-support-confidence-screen/"
    / "support_confidence_screen.py"
)
GDN_BUILDER = (
    REPO_ROOT
    / "zoology/experiments/flash_vqg/scripts/20260526-gdn-expanded-k/config_builder.py"
)
CANONICAL_INIT = (
    REPO_ROOT
    / "zoology/experiments/flash_vqg/scripts/"
    / "20260630-03-flash-vqg-s124-fixed-r4-4ep-confirm/"
    / "outputs/canonical-init/cb64r16-s124-init.pt"
)
GDN_CANONICAL_INIT = SCRIPT_DIR / "outputs/canonical-init/gdnxk-h2-ek4-ev4-s124-init.pt"
EXPECTED_CACHE_HASH = "d9098e876a036b8cb90a7186174fd827e0f5b422482266772850069c905bd8c8"
EXPECTED_INIT_HASH = "2a1107bf22d0804ed485ab94bdc7af8004ef7b892a60c2967f842ba0f4b4efb0"
EXPECTED_BATCH_ORDER_HASH = "fb11b66aca3ad686a85f9623c9ae6769bb4a799fdaa0d952c0af518f0dfcc320"
FLASH_TARGET = "s124-baseline-r16-joint"
TRAIN_SHAPE = (64, 256)
EVAL_SHAPE = (16, 1024)

if str(FLASH_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(FLASH_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from zoology.data.utils import prepare_data  # noqa: E402
from zoology.experiments.flash_vqg.flash_vqg_suite import build_configs  # noqa: E402
from zoology.model import LanguageModel  # noqa: E402
from zoology.utils import set_determinism  # noqa: E402


_SUPPORT_MODULE: Any | None = None
_GDN_MODULE: Any | None = None


def _load_file_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _support_module():
    global _SUPPORT_MODULE
    if _SUPPORT_MODULE is None:
        module = _load_file_module(SUPPORT_SCRIPT, "efficiency_support_config")
        module._patch_identity()
        module._patch_support()
        _SUPPORT_MODULE = module
    return _SUPPORT_MODULE


def _gdn_module():
    global _GDN_MODULE
    if _GDN_MODULE is None:
        _GDN_MODULE = _load_file_module(GDN_BUILDER, "efficiency_gdn_builder")
    return _GDN_MODULE


def _walk_config(value: Any):
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk_config(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_config(child)
    else:
        kwargs = getattr(value, "kwargs", None)
        if isinstance(kwargs, dict):
            yield from _walk_config(kwargs)
        for attr in ("sequence_mixer", "state_mixer"):
            child = getattr(value, attr, None)
            if child is not None:
                yield from _walk_config(child)


def _find_flash_kwargs(model_config: Any) -> dict[str, Any]:
    for node in _walk_config(model_config):
        if isinstance(node, dict) and node.get("fox_remote_formula") == "gd_residual_v1":
            return node
    raise KeyError("Could not find gd_residual_v1 mixer kwargs.")


def _build_flash_config(
    metrics_mode: str,
    grouped_chunk_backend: str = "torch",
    selected_read_backend: str = "materialized",
):
    module = _support_module()
    config = module.BASE.BASEMOD.BASE.build_config(
        target=FLASH_TARGET,
        machine_name="efficiency-audit",
        variant=FLASH_TARGET,
        logger_backend="none",
        trace_output_dir=SCRIPT_DIR / "outputs/traces",
        max_epochs=1,
        max_train_steps=None,
        max_validation_batches=None,
    )
    config.checkpoint.enabled = False
    config.init_checkpoint_path = str(CANONICAL_INIT)
    config.init_checkpoint_strict = True
    kwargs = _find_flash_kwargs(config.model)
    kwargs["fox_gd_residual_grouped_chunk_backend"] = str(grouped_chunk_backend)
    kwargs["fox_gd_residual_selected_read_backend"] = str(selected_read_backend)
    if metrics_mode == "core":
        kwargs["enable_layer_metrics"] = False
        kwargs["fox_phase2_metrics_mode"] = "off"
        config.metrics_white_list = []
    return config


def _build_gdn_config(data_config: Any):
    configs = build_configs(
        sweep_id=f"{EXPERIMENT_ID}-gdn-baseline",
        flash_backend="torch",
        logger_backend="none",
        include_gdn=True,
        dmodels=[128],
        learning_rates=[1e-3],
        if_remote_enabled=True,
        local_num_blocks=2,
        train_batch_order="global_shuffle",
        seed_values=[124],
        data_seed=123,
        gradient_accumulation_steps=4,
        train_batch_size=64,
        eval_batch_size=16,
        cache_dir=str(REPO_ROOT / "data/flash_vqg"),
        max_epochs=1,
        validations_per_epoch=4,
        early_stopping_metric=None,
        early_stopping_threshold=None,
        metrics_white_list=[],
    )
    base = [config for config in configs if config.model.name == "gated_delta_net"]
    if len(base) != 1:
        raise RuntimeError(f"Expected one base GDN config, got {len(base)}")
    config = _gdn_module()._apply_gdn_expanded_k_hparams(
        base[0],
        num_heads=2,
        expand_k=4,
        expand_v=4,
        use_gate=False,
        use_short_conv=True,
        conv_size=4,
    )
    config.data = copy.deepcopy(data_config)
    config.run_id = "gdnxk-h2-ek4-ev4-usegate0-efficiency-audit"
    config.checkpoint.enabled = False
    return config


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, torch.dtype):
        return str(value)
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in keys} for row in rows)


def _sha_text(hasher: Any, value: str) -> None:
    encoded = value.encode("utf-8")
    hasher.update(len(encoded).to_bytes(8, "little"))
    hasher.update(encoded)


def _tensor_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _state_dict_hash(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        value = state_dict[key]
        if not torch.is_tensor(value):
            continue
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(b"\0")
        digest.update(_tensor_hash(value).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _cache_items(data_config: Any) -> list[dict[str, Any]]:
    cache_dir = Path(data_config.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = REPO_ROOT / cache_dir
    np.random.seed(int(data_config.seed))
    max_seed = 2**32
    train_seeds = np.random.randint(0, max_seed // 2, size=len(data_config.train_configs))
    test_seeds = np.random.randint(max_seed // 2, max_seed, size=len(data_config.test_configs))
    out = []
    for role, configs, seeds in (
        ("train", data_config.train_configs, train_seeds),
        ("test", data_config.test_configs, test_seeds),
    ):
        for index, (segment, seed) in enumerate(zip(configs, seeds)):
            payload = segment.model_dump()
            md5 = hashlib.md5(
                json.dumps({**payload, "_seed": int(seed)}, sort_keys=True).encode("utf-8")
            ).hexdigest()
            out.append(
                {
                    "role": role,
                    "segment_index": index,
                    "seed": int(seed),
                    "path": str(cache_dir / f"data_{md5}.pt"),
                    "segment": payload,
                }
            )
    return out


def _cache_content_hash(data_config: Any) -> dict[str, Any]:
    combined = hashlib.sha256()
    rows = []
    for item in _cache_items(data_config):
        path = Path(item["path"])
        if not path.exists():
            raise FileNotFoundError(f"Missing cache file: {path}")
        payload = torch.load(path, map_location="cpu")
        digest = hashlib.sha256()
        for key in sorted(payload):
            value = payload[key]
            _sha_text(digest, key)
            if torch.is_tensor(value):
                # Preserve the canonical cache-hash domain separation used by
                # the existing cross-machine preflight helper.
                _sha_text(digest, key)
                _sha_text(digest, str(value.dtype))
                _sha_text(digest, ",".join(str(dim) for dim in value.shape))
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
            else:
                _sha_text(digest, json.dumps(value, sort_keys=True, default=_json_default))
        content_hash = digest.hexdigest()
        _sha_text(combined, path.name)
        _sha_text(combined, content_hash)
        rows.append({**item, "bytes": path.stat().st_size, "content_sha256": content_hash})
    actual = combined.hexdigest()
    return {
        "file_count": len(rows),
        "combined_content_sha256": actual,
        "expected": EXPECTED_CACHE_HASH,
        "match": actual == EXPECTED_CACHE_HASH,
        "files": rows,
    }


def _batch_order_hash(dataloader: Any) -> dict[str, Any]:
    sampler = dataloader.sampler
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(0)
    order = list(iter(sampler))
    digest = hashlib.sha256()
    for item in order:
        digest.update(int(item).to_bytes(8, "little", signed=True))
    actual = digest.hexdigest()
    return {
        "num_batches": len(order),
        "sha256": actual,
        "expected": EXPECTED_BATCH_ORDER_HASH,
        "match": actual == EXPECTED_BATCH_ORDER_HASH,
        "first_16": order[:16],
    }


def _select_batch(dataloader: Any, shape: tuple[int, int]):
    sampler = getattr(dataloader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(0)
    for ordinal, batch in enumerate(dataloader):
        inputs, targets, slices = batch
        if tuple(inputs.shape) == shape and tuple(targets.shape) == shape:
            return inputs.contiguous(), targets.contiguous(), slices, ordinal
    raise RuntimeError(f"Could not find canonical batch with shape {shape}.")


def _git_value(repo: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def _environment() -> dict[str, Any]:
    available = torch.cuda.is_available()
    return {
        "utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": available,
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_name": torch.cuda.get_device_name(0) if available else None,
        "cuda_capability": torch.cuda.get_device_capability(0) if available else None,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "gdn_kernel_dtype": os.environ.get("GDN_KERNEL_DTYPE"),
        "triton_f32_default": os.environ.get("TRITON_F32_DEFAULT"),
        "zoology_branch": _git_value(REPO_ROOT, "branch", "--show-current"),
        "zoology_commit": _git_value(REPO_ROOT, "rev-parse", "HEAD"),
        "flash_branch": _git_value(FLASH_ROOT, "branch", "--show-current"),
        "flash_commit": _git_value(FLASH_ROOT, "rev-parse", "HEAD"),
    }


def _load_gdn_init(model: nn.Module, path: Path) -> str:
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    state_hash = _state_dict_hash(model.state_dict())
    if payload.get("model_state_sha256") != state_hash:
        raise RuntimeError("GDN init checkpoint embedded hash does not match its model state.")
    return state_hash


def _model_and_hash(model_name: str, flash_config: Any, gdn_config: Any, gdn_init: Path):
    config = flash_config if model_name == "flash" else gdn_config
    set_determinism(124, deterministic=False)
    model = LanguageModel(config.model)
    if model_name == "flash":
        payload = torch.load(CANONICAL_INIT, map_location="cpu")
        model.load_state_dict(payload["model_state_dict"], strict=True)
        state_hash = _state_dict_hash(model.state_dict())
    else:
        state_hash = _load_gdn_init(model, gdn_init)
    return model, config, state_hash


def _model_metadata(model: nn.Module, name: str, state_hash: str) -> dict[str, Any]:
    params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    capacity = 2 * 64 * 64 * 16 if name == "flash" else 2 * 256 * 256
    return {
        "name": name,
        "trainable_parameters": params,
        "active_state_capacity": capacity,
        "init_model_state_sha256": state_hash,
        "expected_init_model_state_sha256": EXPECTED_INIT_HASH if name == "flash" else None,
        "init_hash_match": state_hash == EXPECTED_INIT_HASH if name == "flash" else True,
    }


def _set_dense_teacher(model: nn.Module, targets: torch.Tensor) -> None:
    runtime = {"teacher_target_mask": (targets != -100).detach()}
    for module in model.modules():
        setter = getattr(module, "set_dense_teacher_runtime", None)
        if setter is not None:
            setter(runtime)


def _clear_dense_teacher(model: nn.Module) -> None:
    for module in model.modules():
        clearer = getattr(module, "clear_dense_teacher_runtime", None)
        if clearer is not None:
            clearer()


def _auxiliary_loss(model: nn.Module, device: torch.device) -> torch.Tensor:
    values = []
    for module in model.modules():
        getter = getattr(module, "get_auxiliary_loss", None)
        if getter is not None:
            values.append(getter())
    return sum(values, torch.zeros((), device=device))


def _collect_metrics(model: nn.Module) -> dict[str, float]:
    metrics = {}
    for module in model.modules():
        getter = getattr(module, "get_scalar_metrics", None)
        if getter is not None:
            metrics.update({str(key): float(value) for key, value in getter().items()})
    return metrics


class _CudaRanges:
    def __init__(self):
        self.events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = defaultdict(list)

    @contextmanager
    def range(self, name: str):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.profiler.record_function(f"efficiency/{name}"):
            yield
        end.record()
        self.events[name].append((start, end))

    def milliseconds(self) -> dict[str, float]:
        return {
            name: sum(start.elapsed_time(end) for start, end in pairs)
            for name, pairs in self.events.items()
        }


def _forward_loss(
    model: LanguageModel,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    ranges: _CudaRanges,
    formal: bool,
    include_auxiliary: bool,
):
    with ranges.range("backbone"):
        hidden = model.backbone(inputs)
    with ranges.range("lm_head"):
        logits = model.lm_head(hidden)
    with ranges.range("cross_entropy"):
        loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.flatten())
    if formal:
        with ranges.range("argmax"):
            logits.argmax(dim=-1)
    if include_auxiliary:
        with ranges.range("auxiliary_loss"):
            loss = loss + _auxiliary_loss(model, inputs.device)
    return loss


def _train_iteration(
    model: LanguageModel,
    optimizer: torch.optim.Optimizer,
    cpu_batch: tuple[torch.Tensor, torch.Tensor],
    formal: bool,
):
    ranges = _CudaRanges()
    wall_start = time.perf_counter()
    loss_sync_ms = 0.0
    with ranges.range("zero_grad"):
        optimizer.zero_grad()
    for _ in range(4):
        with ranges.range("h2d"):
            inputs = cpu_batch[0].to("cuda")
            targets = cpu_batch[1].to("cuda")
        _set_dense_teacher(model, targets)
        try:
            loss = _forward_loss(model, inputs, targets, ranges, formal, True)
        finally:
            _clear_dense_teacher(model)
        with ranges.range("backward"):
            (loss / 4).backward()
        if formal:
            sync_start = time.perf_counter()
            loss.detach().cpu().item()
            loss_sync_ms += (time.perf_counter() - sync_start) * 1000.0
    with ranges.range("optimizer_step"):
        optimizer.step()
    metrics_start = time.perf_counter()
    metrics = _collect_metrics(model) if formal else {}
    metrics_wall_ms = (time.perf_counter() - metrics_start) * 1000.0
    torch.cuda.synchronize()
    record = ranges.milliseconds()
    record.update(
        {
            "cuda_total_ms": sum(ranges.milliseconds().values()),
            "wall_ms": (time.perf_counter() - wall_start) * 1000.0,
            "loss_sync_wall_ms": loss_sync_ms,
            "metrics_wall_ms": metrics_wall_ms,
            "metric_count": len(metrics),
        }
    )
    return record


def _eval_iteration(
    model: LanguageModel,
    cpu_batch: tuple[torch.Tensor, torch.Tensor],
    formal: bool,
):
    ranges = _CudaRanges()
    wall_start = time.perf_counter()
    with ranges.range("h2d"):
        inputs = cpu_batch[0].to("cuda")
        targets = cpu_batch[1].to("cuda")
    _set_dense_teacher(model, targets)
    try:
        with torch.no_grad():
            _forward_loss(model, inputs, targets, ranges, formal, False)
    finally:
        _clear_dense_teacher(model)
    metrics_start = time.perf_counter()
    metrics = _collect_metrics(model) if formal else {}
    metrics_wall_ms = (time.perf_counter() - metrics_start) * 1000.0
    torch.cuda.synchronize()
    record = ranges.milliseconds()
    record.update(
        {
            "cuda_total_ms": sum(ranges.milliseconds().values()),
            "wall_ms": (time.perf_counter() - wall_start) * 1000.0,
            "metrics_wall_ms": metrics_wall_ms,
            "metric_count": len(metrics),
        }
    )
    return record


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q, method="linear"))


def _summaries(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    numeric_keys = sorted(
        {key for record in records for key, value in record.items() if isinstance(value, (int, float))}
    )
    rows = []
    for key in numeric_keys:
        values = [float(record[key]) for record in records if isinstance(record.get(key), (int, float))]
        rows.append(
            {
                "metric": key,
                "count": len(values),
                "mean": statistics.fmean(values),
                "p50": _percentile(values, 50),
                "p90": _percentile(values, 90),
                "min": min(values),
                "max": max(values),
            }
        )
    return rows


def _saved_tensor_hook(rows: list[dict[str, Any]]):
    def pack(tensor: torch.Tensor):
        rows.append(
            {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "numel": tensor.numel(),
                "bytes": tensor.numel() * tensor.element_size(),
                "requires_grad": bool(tensor.requires_grad),
            }
        )
        return tensor

    return pack


def _profiler_context(output_dir: Path, *, detailed: bool):
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    return torch.profiler.profile(
        activities=activities,
        record_shapes=detailed,
        profile_memory=detailed,
        with_stack=detailed,
    )


def _write_profiler(prof: Any, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    trace = output_dir / "trace.json.gz"
    prof.export_chrome_trace(str(trace))
    paths = {"trace": str(trace)}
    for key in ("cpu_time_total", "cuda_time_total", "cuda_memory_usage"):
        path = output_dir / f"profiler-{key}.txt"
        path.write_text(prof.key_averages().table(sort_by=key, row_limit=200), encoding="utf-8")
        paths[key] = str(path)
    return paths


def _dump_memory_snapshot(output_dir: Path) -> tuple[str | None, str | None]:
    snapshot = output_dir / "memory-snapshot.pickle"
    try:
        torch.cuda.memory._dump_snapshot(str(snapshot))
        return str(snapshot), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _configure_numerics() -> None:
    os.environ["GDN_KERNEL_DTYPE"] = "float32"
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _preflight(args: argparse.Namespace) -> int:
    _configure_numerics()
    flash = _build_flash_config("formal")
    cache = _cache_content_hash(flash.data)
    train_loader, test_loader = prepare_data(flash.data)
    batch_order = _batch_order_hash(train_loader)
    train = _select_batch(train_loader, TRAIN_SHAPE)
    valid = _select_batch(test_loader, EVAL_SHAPE)
    init_payload = torch.load(CANONICAL_INIT, map_location="cpu")
    init_hash = _state_dict_hash(init_payload["model_state_dict"])
    gdn = _build_gdn_config(flash.data)
    set_determinism(124, deterministic=False)
    flash_model = LanguageModel(flash.model)
    set_determinism(124, deterministic=False)
    gdn_model = LanguageModel(gdn.model)
    gdn_init_hash = _load_gdn_init(gdn_model, args.gdn_init)
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "environment": _environment(),
        "gpu_gate": bool(torch.cuda.is_available()),
        "cache": cache,
        "batch_order": batch_order,
        "init": {
            "path": str(CANONICAL_INIT),
            "actual": init_hash,
            "expected": EXPECTED_INIT_HASH,
            "match": init_hash == EXPECTED_INIT_HASH,
            "embedded": init_payload.get("model_state_sha256"),
        },
        "gdn_init": {
            "path": str(args.gdn_init),
            "actual": gdn_init_hash,
            "embedded": torch.load(args.gdn_init, map_location="cpu").get("model_state_sha256"),
        },
        "fixed_batches": {
            "train": {
                "shape": list(train[0].shape),
                "ordinal": train[3],
                "inputs_sha256": _tensor_hash(train[0]),
                "targets_sha256": _tensor_hash(train[1]),
            },
            "eval": {
                "shape": list(valid[0].shape),
                "ordinal": valid[3],
                "inputs_sha256": _tensor_hash(valid[0]),
                "targets_sha256": _tensor_hash(valid[1]),
            },
        },
        "models": {
            "flash": _model_metadata(flash_model, "flash", init_hash),
            "gdn": _model_metadata(gdn_model, "gdn", gdn_init_hash),
        },
        "flash_settings": _find_flash_kwargs(flash.model),
    }
    payload["passed"] = all(
        (
            payload["gpu_gate"],
            cache["match"],
            batch_order["match"],
            payload["init"]["match"],
            payload["init"]["embedded"] == init_hash,
            payload["gdn_init"]["embedded"] == gdn_init_hash,
            payload["models"]["flash"]["trainable_parameters"] == 1160390,
            payload["models"]["gdn"]["trainable_parameters"] == 1335942,
        )
    )
    _write_json(args.output, payload)
    print(json.dumps({"passed": payload["passed"], "output": str(args.output)}, sort_keys=True))
    return 0 if payload["passed"] else 1


def _prepare_run(args: argparse.Namespace):
    _configure_numerics()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the target container.")
    flash = _build_flash_config(
        args.metrics_mode,
        args.flash_grouped_chunk_backend,
        args.flash_selected_read_backend,
    )
    gdn = _build_gdn_config(flash.data)
    model, config, state_hash = _model_and_hash(args.model, flash, gdn, args.gdn_init)
    train_loader, test_loader = prepare_data(flash.data)
    selected = _select_batch(train_loader, TRAIN_SHAPE) if args.phase == "train" else _select_batch(test_loader, EVAL_SHAPE)
    model = model.to("cuda")
    model.train(args.phase == "train")
    optimizer = None
    if args.phase == "train":
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    metadata = {
        "experiment_id": EXPERIMENT_ID,
        "environment": _environment(),
        "model": _model_metadata(model, args.model, state_hash),
        "phase": args.phase,
        "metrics_mode": args.metrics_mode,
        "run_kind": args.run_kind,
        "flash_implementation": {
            "grouped_chunk_backend": args.flash_grouped_chunk_backend,
            "selected_read_backend": args.flash_selected_read_backend,
        },
        "repeat_id": args.repeat_id,
        "warmup": args.warmup,
        "active": args.active,
        "batch": {
            "shape": list(selected[0].shape),
            "ordinal": selected[3],
            "inputs_sha256": _tensor_hash(selected[0]),
            "targets_sha256": _tensor_hash(selected[1]),
            "valid_targets": int((selected[1] != -100).sum().item()),
        },
        "resolved_run_id": config.run_id,
    }
    return model, optimizer, (selected[0], selected[1]), metadata


def _make_gdn_init(args: argparse.Namespace) -> int:
    if args.output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite existing GDN init: {args.output}")
    _configure_numerics()
    flash = _build_flash_config("core")
    gdn = _build_gdn_config(flash.data)
    set_determinism(124, deterministic=False)
    model = LanguageModel(gdn.model)
    state_hash = _state_dict_hash(model.state_dict())
    payload = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "model": "gdnxk-h2-ek4-ev4-usegate0",
        "seed": 124,
        "model_state_sha256": state_hash,
        "model_state_dict": model.state_dict(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    _write_json(
        args.output.with_suffix(".json"),
        {
            "checkpoint": str(args.output),
            "bytes": args.output.stat().st_size,
            "model_state_sha256": state_hash,
        },
    )
    print(json.dumps({"checkpoint": str(args.output), "model_state_sha256": state_hash}, sort_keys=True))
    return 0


def _flash_model_from_config(config: Any) -> LanguageModel:
    set_determinism(124, deterministic=False)
    model = LanguageModel(config.model)
    payload = torch.load(CANONICAL_INIT, map_location="cpu")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to("cuda")


def _cpu_tensor_map(values: Iterable[tuple[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {name: value.detach().float().cpu().clone() for name, value in values}


def _run_flash_equivalence_variant(
    config: Any,
    train_batch: tuple[torch.Tensor, torch.Tensor],
    eval_batch: tuple[torch.Tensor, torch.Tensor],
) -> dict[str, Any]:
    model = _flash_model_from_config(config)
    eval_inputs = eval_batch[0].to("cuda")
    eval_targets = eval_batch[1].to("cuda")
    model.eval()
    torch.manual_seed(20260724)
    torch.cuda.manual_seed_all(20260724)
    _set_dense_teacher(model, eval_targets)
    try:
        with torch.no_grad():
            eval_hidden = model.backbone(eval_inputs)
            eval_logits = model.lm_head(eval_hidden)
            eval_loss = nn.functional.cross_entropy(
                eval_logits.reshape(-1, eval_logits.size(-1)),
                eval_targets.flatten(),
            )
    finally:
        _clear_dense_teacher(model)
    eval_payload = {
        "hidden": eval_hidden.detach().float().cpu(),
        "loss": float(eval_loss.detach().cpu().item()),
    }
    del eval_hidden, eval_logits, eval_loss, eval_inputs, eval_targets

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    train_inputs = train_batch[0].to("cuda")
    train_targets = train_batch[1].to("cuda")
    torch.manual_seed(20260724)
    torch.cuda.manual_seed_all(20260724)
    optimizer.zero_grad(set_to_none=True)
    loss_values = []
    last_hidden = None
    for _ in range(4):
        _set_dense_teacher(model, train_targets)
        try:
            hidden = model.backbone(train_inputs)
            logits = model.lm_head(hidden)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                train_targets.flatten(),
            )
            loss = loss + _auxiliary_loss(model, train_inputs.device)
        finally:
            _clear_dense_teacher(model)
        (loss / 4).backward()
        loss_values.append(loss.detach())
        last_hidden = hidden.detach()
    gradients = _cpu_tensor_map(
        (name, parameter.grad)
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    )
    optimizer.step()
    parameters = _cpu_tensor_map(model.named_parameters())
    optimizer_tensors = {}
    for name, parameter in model.named_parameters():
        for state_name, value in optimizer.state.get(parameter, {}).items():
            if torch.is_tensor(value):
                optimizer_tensors[f"{name}/{state_name}"] = value.detach().float().cpu().clone()
    torch.cuda.synchronize()
    train_payload = {
        "hidden": last_hidden.float().cpu(),
        "losses": torch.stack(loss_values).float().cpu(),
        "gradients": gradients,
        "parameters": parameters,
        "optimizer": optimizer_tensors,
    }
    del model, optimizer, train_inputs, train_targets, hidden, logits, loss, last_hidden
    torch.cuda.empty_cache()
    return {"eval": eval_payload, "train": train_payload}


def _tensor_comparison(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    reference_f32 = reference.float()
    candidate_f32 = candidate.float()
    difference = candidate_f32 - reference_f32
    reference_norm = reference_f32.norm().clamp_min(1e-24)
    return {
        "max_abs": float(difference.abs().max().item()) if difference.numel() else 0.0,
        "relative_l2": float((difference.norm() / reference_norm).item()),
    }


def _tensor_map_comparison(
    reference: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
) -> dict[str, Any]:
    reference_keys = set(reference)
    candidate_keys = set(candidate)
    rows = []
    for key in sorted(reference_keys & candidate_keys):
        rows.append({"name": key, **_tensor_comparison(reference[key], candidate[key])})
    return {
        "missing_from_candidate": sorted(reference_keys - candidate_keys),
        "missing_from_reference": sorted(candidate_keys - reference_keys),
        "max_abs": max((row["max_abs"] for row in rows), default=0.0),
        "max_relative_l2": max((row["relative_l2"] for row in rows), default=0.0),
        "worst_max_abs": sorted(rows, key=lambda row: row["max_abs"], reverse=True)[:10],
        "worst_relative_l2": sorted(
            rows,
            key=lambda row: row["relative_l2"],
            reverse=True,
        )[:10],
    }


def _equivalence(args: argparse.Namespace) -> int:
    _configure_numerics()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the target container.")
    reference_config = _build_flash_config("core", "torch", "materialized")
    candidate_config = _build_flash_config("core", "triton", "triton_remat")
    train_loader, test_loader = prepare_data(reference_config.data)
    train = _select_batch(train_loader, TRAIN_SHAPE)
    valid = _select_batch(test_loader, EVAL_SHAPE)
    train_batch = (train[0], train[1])
    eval_batch = (valid[0], valid[1])

    reference = _run_flash_equivalence_variant(
        reference_config,
        train_batch,
        eval_batch,
    )
    candidate = _run_flash_equivalence_variant(
        candidate_config,
        train_batch,
        eval_batch,
    )
    comparisons = {
        "eval_hidden": _tensor_comparison(
            reference["eval"]["hidden"], candidate["eval"]["hidden"]
        ),
        "eval_loss_abs": abs(reference["eval"]["loss"] - candidate["eval"]["loss"]),
        "train_hidden": _tensor_comparison(
            reference["train"]["hidden"], candidate["train"]["hidden"]
        ),
        "train_losses": _tensor_comparison(
            reference["train"]["losses"], candidate["train"]["losses"]
        ),
        "gradients": _tensor_map_comparison(
            reference["train"]["gradients"], candidate["train"]["gradients"]
        ),
        "parameters_after_step": _tensor_map_comparison(
            reference["train"]["parameters"], candidate["train"]["parameters"]
        ),
        "optimizer_after_step": _tensor_map_comparison(
            reference["train"]["optimizer"], candidate["train"]["optimizer"]
        ),
    }
    passed = all(
        (
            comparisons["eval_hidden"]["max_abs"] <= 1e-5,
            comparisons["eval_hidden"]["relative_l2"] <= 1e-5,
            comparisons["eval_loss_abs"] <= 1e-6,
            comparisons["train_hidden"]["max_abs"] <= 2e-5,
            comparisons["train_hidden"]["relative_l2"] <= 1e-5,
            comparisons["train_losses"]["max_abs"] <= 1e-5,
            comparisons["gradients"]["max_relative_l2"] <= 1e-4,
            comparisons["gradients"]["max_abs"] <= 2e-5,
            comparisons["parameters_after_step"]["max_abs"] <= 2e-6,
            not comparisons["gradients"]["missing_from_candidate"],
            not comparisons["gradients"]["missing_from_reference"],
        )
    )
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "environment": _environment(),
        "reference": {
            "grouped_chunk_backend": "torch",
            "selected_read_backend": "materialized",
        },
        "candidate": {
            "grouped_chunk_backend": "triton",
            "selected_read_backend": "triton_remat",
        },
        "fixed_batches": {
            "train_inputs_sha256": _tensor_hash(train[0]),
            "train_targets_sha256": _tensor_hash(train[1]),
            "eval_inputs_sha256": _tensor_hash(valid[0]),
            "eval_targets_sha256": _tensor_hash(valid[1]),
        },
        "comparisons": comparisons,
        "passed": passed,
    }
    _write_json(args.output, payload)
    print(json.dumps({"output": str(args.output), "passed": passed, "comparisons": comparisons}))
    return 0 if passed else 1


def _run(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, optimizer, batch, metadata = _prepare_run(args)
    formal = args.metrics_mode == "formal"
    run_one = (
        (lambda: _train_iteration(model, optimizer, batch, formal))
        if args.phase == "train"
        else (lambda: _eval_iteration(model, batch, formal))
    )
    for _ in range(args.warmup):
        run_one()
    history_enabled = args.run_kind in {"memory", "profile"}
    if history_enabled:
        torch.cuda.memory._record_memory_history(
            enabled="all",
            context="all",
            stacks="python",
            max_entries=100000,
        )
    torch.cuda.reset_peak_memory_stats()
    persistent = {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
    }
    saved_tensors: list[dict[str, Any]] = []
    profiler_paths = {}
    profile_ctx = (
        _profiler_context(args.output_dir, detailed=args.run_kind == "profile")
        if args.run_kind in {"profile", "op-profile"}
        else nullcontext(None)
    )
    saved_ctx = (
        torch.autograd.graph.saved_tensors_hooks(_saved_tensor_hook(saved_tensors), lambda tensor: tensor)
        if args.run_kind == "profile" and args.phase == "train"
        else nullcontext()
    )
    records = []
    with profile_ctx as prof, saved_ctx:
        for _ in range(args.active):
            records.append(run_one())
            if prof is not None:
                prof.step()
    # Capture the last forward's scalar diagnostics outside the timed region.
    # This is intentionally opt-in because get_scalar_metrics() synchronizes
    # each scalar to the host and is itself one of the costs under audit.
    last_scalar_metrics = _collect_metrics(model) if args.capture_scalar_metrics else {}
    if prof is not None:
        profiler_paths = _write_profiler(prof, args.output_dir)
    snapshot_path = None
    snapshot_error = None
    if args.run_kind in {"memory", "profile"}:
        snapshot_path, snapshot_error = _dump_memory_snapshot(args.output_dir)
    if history_enabled:
        torch.cuda.memory._record_memory_history(enabled=None)
    memory = {
        "persistent": persistent,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "snapshot": snapshot_path,
        "snapshot_error": snapshot_error,
    }
    summary = {
        **metadata,
        "records": records,
        "summaries": _summaries(records),
        "memory": memory,
        "profiler": profiler_paths,
        "saved_tensors": saved_tensors,
        "last_scalar_metrics": last_scalar_metrics,
    }
    output = args.output_dir / "summary.json"
    _write_json(output, summary)
    _write_csv(args.output_dir / "records.csv", records)
    _write_csv(args.output_dir / "summary.csv", summary["summaries"])
    if saved_tensors:
        _write_csv(args.output_dir / "saved-tensors.csv", saved_tensors)
    print(json.dumps({"output": str(output), "memory": memory}, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark Flash-VQG and the frozen same-scale GDN.")
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--output", type=Path, required=True)
    preflight.add_argument("--gdn-init", type=Path, default=GDN_CANONICAL_INIT)
    preflight.set_defaults(func=_preflight)

    make_init = sub.add_parser("make-gdn-init")
    make_init.add_argument("--output", type=Path, default=GDN_CANONICAL_INIT)
    make_init.add_argument("--force", action="store_true")
    make_init.set_defaults(func=_make_gdn_init)

    equivalence = sub.add_parser("equivalence")
    equivalence.add_argument("--output", type=Path, required=True)
    equivalence.set_defaults(func=_equivalence)

    run = sub.add_parser("run")
    run.add_argument("--model", choices=("flash", "gdn"), required=True)
    run.add_argument("--phase", choices=("train", "eval"), required=True)
    run.add_argument("--metrics-mode", choices=("core", "formal"), default="core")
    run.add_argument(
        "--run-kind",
        choices=("timing", "memory", "profile", "op-profile"),
        required=True,
    )
    run.add_argument("--warmup", type=int, default=5)
    run.add_argument("--active", type=int, default=10)
    run.add_argument("--repeat-id", type=int, default=0)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--gdn-init", type=Path, default=GDN_CANONICAL_INIT)
    run.add_argument(
        "--flash-grouped-chunk-backend",
        choices=("torch", "triton"),
        default="torch",
    )
    run.add_argument(
        "--flash-selected-read-backend",
        choices=("materialized", "triton_remat"),
        default="materialized",
    )
    run.add_argument(
        "--capture-scalar-metrics",
        action="store_true",
        help="Capture the last forward's scalar metrics after the timed region.",
    )
    run.set_defaults(func=_run)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if hasattr(args, "warmup") and (args.warmup < 0 or args.active <= 0):
        raise ValueError("warmup must be non-negative and active must be positive.")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
