#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import platform
import re
import socket
import subprocess
import sys
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get("ZOOLOGY_REPO_ROOT", "/home/lyj/mnt/project/zoology")).resolve()
FLASH_VQG_ROOT = Path(os.environ.get("FLASH_VQG_ROOT", "/home/lyj/mnt/project/Flash-VQG")).resolve()
EXPERIMENT_ID = "20260629-03-flash-vqg-dense-read-confirm"
PREVIOUS_EXPERIMENT_ID = "20260627-02-flash-vqg-canonical-init-lock-screen"
PREVIOUS_SCRIPT_DIR = REPO_ROOT / "zoology/experiments/flash_vqg/scripts" / PREVIOUS_EXPERIMENT_ID
DEFAULT_INIT_CHECKPOINT = PREVIOUS_SCRIPT_DIR / "outputs/canonical-init/cb64r16-s123-init.pt"
EXPECTED_CACHE_COMBINED_SHA256 = (
    "d9098e876a036b8cb90a7186174fd827e0f5b422482266772850069c905bd8c8"
)
EXPECTED_INIT_STATE_SHA256 = (
    "dd0b7bb57b8bbb1fd2bd6dbcfcaa82d609c318b9b1c9ead0e89032f17bf16edf"
)
METRICS_YAML = (
    REPO_ROOT
    / "zoology/experiments/flash_vqg/scripts/20260625-02-flash-vqg-1epoch-repro-screen/metrics.yaml"
)
BUILDER_PATH = (
    REPO_ROOT
    / "zoology/experiments/flash_vqg/scripts/20260425-gd-residual-v1-mqar/config_builder.py"
)
ARTIFACT_DIR = REPO_ROOT / "docs/artifacts" / EXPERIMENT_ID
TARGETS = ("dense-read-4ep-s123-r1", "dense-read-4ep-s123-r2")
VARIANT = "dense-read"
EXPECTED_NUM_CODEBOOK_VECTORS = 64
EXPECTED_REMOTE_READ_TOPK = 64
EXPECTED_DENSE_READ_CHUNKED = True
DEFAULT_MAX_EPOCHS = 4
EXPECTED_STEPS_PER_EPOCH = 704
EXPECTED_TOTAL_OPTIMIZER_STEPS = EXPECTED_STEPS_PER_EPOCH * DEFAULT_MAX_EPOCHS
SOURCE_HOST_BY_MACHINE = {
    "2080ti": "mclab-2080ti",
    "3090": "mclab-3090",
}

if str(FLASH_VQG_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(FLASH_VQG_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from zoology.checkpoints import serialize_train_config  # noqa: E402
from zoology.data.utils import prepare_data  # noqa: E402
from zoology.model import LanguageModel  # noqa: E402
from zoology.train import train  # noqa: E402
from zoology.utils import set_determinism  # noqa: E402


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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    if fieldnames is None:
        fieldnames = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    fieldnames.append(key)
                    seen.add(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _dedupe_adjacent(values: list[str]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        if not deduped or value != deduped[-1]:
            deduped.append(value)
    return deduped


def _sha_update_text(hasher: Any, value: str) -> None:
    encoded = value.encode("utf-8")
    hasher.update(len(encoded).to_bytes(8, "little"))
    hasher.update(encoded)


def _update_tensor_hash(hasher: Any, name: str, tensor: torch.Tensor) -> None:
    detached = tensor.detach().cpu().contiguous()
    _sha_update_text(hasher, name)
    _sha_update_text(hasher, str(detached.dtype))
    _sha_update_text(hasher, ",".join(str(dim) for dim in detached.shape))
    hasher.update(detached.numpy().tobytes())


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


def _cache_items_for_config(data_config: Any) -> list[dict[str, Any]]:
    cache_dir_raw = data_config.cache_dir
    if not cache_dir_raw:
        raise ValueError("config.data.cache_dir is empty.")
    cache_dir = Path(cache_dir_raw)
    if not cache_dir.is_absolute():
        cache_dir = REPO_ROOT / cache_dir
    max_seed = 2**32
    import numpy as np

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


def _tensor_hash(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().cpu().contiguous()
    return hashlib.sha256(cpu.numpy().tobytes()).hexdigest()


def _state_dict_hashes(state_dict: dict[str, torch.Tensor]) -> tuple[str, dict[str, str]]:
    digest = hashlib.sha256()
    per_tensor: dict[str, str] = {}
    for key in sorted(state_dict):
        value = state_dict[key]
        if not torch.is_tensor(value):
            continue
        value_hash = _tensor_hash(value)
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


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


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


def _find_flash_vqg_kwargs(model_config: Any) -> dict[str, Any]:
    for node in _walk_config_objects(model_config):
        if isinstance(node, dict) and node.get("fox_remote_formula") == "gd_residual_v1":
            return node
    return {}


def _set_flash_vqg_kwarg(config: Any, key: str, value: Any) -> None:
    flash_kwargs = _find_flash_vqg_kwargs(config.model)
    if not flash_kwargs:
        raise KeyError("Could not find Flash-VQG gd_residual_v1 mixer kwargs.")
    flash_kwargs[key] = value


def _find_nested_key(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for child in value.values():
            found = _find_nested_key(child, key)
            if found != "":
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_nested_key(child, key)
            if found != "":
                return found
    return ""


def _flash_vqg_settings(config: Any) -> dict[str, Any]:
    flash_kwargs = _find_flash_vqg_kwargs(config.model)
    return {
        "num_codebook_vectors": flash_kwargs.get("num_codebook_vectors", ""),
        "fox_remote_read_topk": flash_kwargs.get("fox_remote_read_topk", ""),
        "fox_gd_residual_dense_read_chunked": flash_kwargs.get(
            "fox_gd_residual_dense_read_chunked", ""
        ),
        "fox_gd_residual_rank": flash_kwargs.get("fox_gd_residual_rank", ""),
        "fox_gd_residual_write_topk": flash_kwargs.get("fox_gd_residual_write_topk", ""),
    }


def _base_args(
    *,
    target: str,
    machine_name: str,
    variant: str,
    logger_backend: str,
    trace_output_dir: Path,
    max_epochs: int,
    max_train_steps: int | None,
    max_validation_batches: int | None,
) -> Namespace:
    if target not in TARGETS:
        raise ValueError(f"Unsupported target: {target}")
    repeat = "r1" if target.endswith("-r1") else "r2"
    return Namespace(
        launch_id_prefix=f"fvqg-{EXPERIMENT_ID}-{machine_name}-{target}",
        backend="torch",
        logger_backend=logger_backend,
        dmodels="128",
        learning_rates="1e-3",
        train_batch_order="global_shuffle",
        seed_values="123",
        data_seed=123,
        num_codebook_vectors=str(EXPECTED_NUM_CODEBOOK_VECTORS),
        fox_remote_path_backend="torch",
        fox_remote_read_topk_values=str(EXPECTED_REMOTE_READ_TOPK),
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
        project="flash_vqg_dense_read_4ep_confirm",
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
        read_trace_train_steps="0,64,130,203,352,448,704,1408,2112,2816",
        experiment_mode=f"{EXPERIMENT_ID}_{variant}_s123_{repeat}_d123_b64ga4_{machine_name}",
        run_id=f"{EXPERIMENT_ID}-{variant}-s123-{repeat}-d123-b64ga4-{machine_name}",
    )


def build_config(
    *,
    target: str,
    machine_name: str,
    variant: str,
    logger_backend: str,
    trace_output_dir: Path,
    max_epochs: int,
    max_train_steps: int | None,
    max_validation_batches: int | None,
):
    if variant != VARIANT:
        raise ValueError(f"Unsupported variant: {variant}")
    builder = _load_builder_module()
    args = _base_args(
        target=target,
        machine_name=machine_name,
        variant=variant,
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
    config.model.embed_dropout = 0.0
    config.model.resid_dropout = 0.0
    config.model.drop_path = 0.0
    _set_flash_vqg_kwarg(config, "fox_remote_read_topk", EXPECTED_REMOTE_READ_TOPK)
    _set_flash_vqg_kwarg(config, "fox_gd_residual_dense_read_chunked", EXPECTED_DENSE_READ_CHUNKED)
    return config


def env_snapshot(machine_name: str) -> dict[str, Any]:
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
        "zoology_branch": _git_value(REPO_ROOT, ["branch", "--show-current"]),
        "zoology_commit": _git_value(REPO_ROOT, ["rev-parse", "--short", "HEAD"]),
        "flash_vqg_branch": _git_value(FLASH_VQG_ROOT, ["branch", "--show-current"]),
        "flash_vqg_commit": _git_value(FLASH_VQG_ROOT, ["rev-parse", "--short", "HEAD"]),
    }


def _hash_cache_for_config(data_config: Any) -> dict[str, Any]:
    metadata = _cache_items_for_config(data_config)
    combined = hashlib.sha256()
    files: list[dict[str, Any]] = []
    missing = [item for item in metadata if not item["exists"]]
    if missing:
        raise FileNotFoundError(
            "Missing required MQAR cache files: "
            + ", ".join(item["path"] for item in missing[:8])
        )
    for item in metadata:
        file_hash = _hash_cache_file(Path(item["path"]))
        row = {**item, **file_hash}
        files.append(row)
        _sha_update_text(combined, item["name"])
        _sha_update_text(combined, file_hash["content_sha256"])
    combined_hash = combined.hexdigest()
    return {
        "file_count": len(files),
        "combined_content_sha256": combined_hash,
        "expected_combined_content_sha256": EXPECTED_CACHE_COMBINED_SHA256,
        "match_expected": combined_hash == EXPECTED_CACHE_COMBINED_SHA256,
        "files": files,
    }


def _verify_init_checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    state = payload["model_state_dict"]
    state_hash, per_tensor = _state_dict_hashes(state)
    embedded_hash = payload.get("model_state_sha256")
    embedded_per_tensor = payload.get("per_tensor_sha256", {})
    mismatched_keys = [
        key for key, value in embedded_per_tensor.items() if per_tensor.get(key) != value
    ]
    return {
        "checkpoint": str(path),
        "expected_model_state_sha256": EXPECTED_INIT_STATE_SHA256,
        "embedded_model_state_sha256": embedded_hash,
        "actual_model_state_sha256": state_hash,
        "match_expected": state_hash == EXPECTED_INIT_STATE_SHA256,
        "match_embedded": embedded_hash == state_hash and not mismatched_keys,
        "mismatched_keys": mismatched_keys,
        "num_tensors": len(per_tensor),
    }


def _parse_final_metrics(log_path: Path) -> dict[str, Any]:
    if not log_path.exists():
        return {}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    raw_accuracy_1024 = re.findall(r"valid/mqar_case/accuracy-1024x256=([0-9.]+)", text)
    raw_valid_accuracy = re.findall(r"valid/accuracy=([0-9.]+)", text)
    raw_valid_loss = re.findall(r"valid/loss=([0-9.]+)", text)
    # tqdm can write the same validation summary twice. Adjacent de-duplication
    # keeps final/best metrics stable while making the count reflect events.
    accuracy_1024 = _dedupe_adjacent(raw_accuracy_1024)
    valid_accuracy = _dedupe_adjacent(raw_valid_accuracy)
    valid_loss = _dedupe_adjacent(raw_valid_loss)
    final_1024 = float(accuracy_1024[-1]) if accuracy_1024 else None
    best_1024 = max(float(value) for value in accuracy_1024) if accuracy_1024 else None
    return {
        "final_1024x256_accuracy": accuracy_1024[-1] if accuracy_1024 else "",
        "best_1024x256_accuracy": best_1024 if best_1024 is not None else "",
        "best_final_1024x256_gap": (
            best_1024 - final_1024 if best_1024 is not None and final_1024 is not None else ""
        ),
        "final_valid_accuracy": valid_accuracy[-1] if valid_accuracy else "",
        "best_valid_accuracy": max(float(value) for value in valid_accuracy) if valid_accuracy else "",
        "final_valid_loss": valid_loss[-1] if valid_loss else "",
        "n_validation_summaries": len(accuracy_1024),
        "n_validation_summary_lines": len(raw_accuracy_1024),
    }


def _machine_from_path(path: Path) -> str:
    parts = path.parts
    for machine in ("2080ti", "3090"):
        if machine in parts:
            return machine
        if any(part.startswith(f"{machine}-") for part in parts):
            return machine
    return ""


def _source_and_mirror_paths(path: Path) -> tuple[str, str, str, str]:
    machine = _machine_from_path(path)
    source_host = SOURCE_HOST_BY_MACHINE.get(machine, "")
    mirror_path = str(path)
    if machine == "3090":
        return machine, source_host, f"{source_host}:{path}", mirror_path
    if machine == "2080ti":
        return machine, source_host, str(path), mirror_path
    return machine, source_host, str(path), mirror_path


def run_cache_hash(args: argparse.Namespace) -> int:
    config = build_config(
        target=args.target,
        machine_name=args.machine_name,
        variant=args.variant,
        logger_backend="none",
        trace_output_dir=SCRIPT_DIR / "outputs/preflight-traces" / args.machine_name / args.target,
        max_epochs=DEFAULT_MAX_EPOCHS,
        max_train_steps=None,
        max_validation_batches=None,
    )
    payload = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "machine_name": args.machine_name,
        "target": args.target,
        "variant": args.variant,
        "environment": env_snapshot(args.machine_name),
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
        "environment": env_snapshot(args.machine_name),
        "init_checkpoint": _verify_init_checkpoint(args.checkpoint),
    }
    if args.output_json:
        _save_json(args.output_json, payload)
    print(json.dumps(payload["init_checkpoint"], ensure_ascii=False, indent=2, sort_keys=True))
    init = payload["init_checkpoint"]
    return 0 if init["match_expected"] and init["match_embedded"] else 1


def run_preflight(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("torch.cuda is unavailable inside this container.")
    config = build_config(
        target=args.target,
        machine_name=args.machine_name,
        variant=args.variant,
        logger_backend="none",
        trace_output_dir=SCRIPT_DIR / "outputs/preflight-traces" / args.machine_name / args.target,
        max_epochs=args.max_epochs,
        max_train_steps=None,
        max_validation_batches=None,
    )
    train_loader, _ = prepare_data(config.data)
    train_batches = len(train_loader)
    accum = int(config.gradient_accumulation_steps)
    optim_steps_per_epoch = (train_batches + accum - 1) // accum
    max_epochs = int(config.max_epochs)
    total_optimizer_steps = optim_steps_per_epoch * max_epochs
    flash_settings = _flash_vqg_settings(config)
    result = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "machine_name": args.machine_name,
        "target": args.target,
        "variant": args.variant,
        "env": env_snapshot(args.machine_name),
        "run_id": config.run_id,
        "launch_id": config.launch_id,
        "train_batches": train_batches,
        "gradient_accumulation_steps": accum,
        "max_epochs": max_epochs,
        "num_optimizer_steps_per_epoch": optim_steps_per_epoch,
        "total_optimizer_steps": total_optimizer_steps,
        "expected_optimizer_steps_per_epoch": EXPECTED_STEPS_PER_EPOCH,
        "expected_total_optimizer_steps": EXPECTED_TOTAL_OPTIMIZER_STEPS,
        "cache_dir": config.data.cache_dir,
        "embed_dropout": config.model.embed_dropout,
        "resid_dropout": config.model.resid_dropout,
        "drop_path": config.model.drop_path,
        **flash_settings,
        "passed": (
            train_batches == 2815
            and accum == 4
            and max_epochs == DEFAULT_MAX_EPOCHS
            and optim_steps_per_epoch == EXPECTED_STEPS_PER_EPOCH
            and total_optimizer_steps == EXPECTED_TOTAL_OPTIMIZER_STEPS
            and config.model.embed_dropout == 0.0
            and config.model.resid_dropout == 0.0
            and config.model.drop_path == 0.0
            and flash_settings["num_codebook_vectors"] == EXPECTED_NUM_CODEBOOK_VECTORS
            and flash_settings["fox_remote_read_topk"] == EXPECTED_REMOTE_READ_TOPK
            and flash_settings["fox_gd_residual_dense_read_chunked"]
            == EXPECTED_DENSE_READ_CHUNKED
        ),
    }
    if args.output_json:
        _save_json(args.output_json, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


def run_train(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("torch.cuda is unavailable inside this container.")
    config = build_config(
        target=args.target,
        machine_name=args.machine_name,
        variant=args.variant,
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
        _save_json(args.output_config_json, serialize_train_config(config))
    result = train(config)
    if args.output_result_json:
        _save_json(
            args.output_result_json,
            {
                "schema_version": 1,
                "experiment_id": EXPERIMENT_ID,
                "machine_name": args.machine_name,
                "target": args.target,
                "variant": args.variant,
                "init_checkpoint": str(args.init_checkpoint),
                "train_result": result,
                "env": env_snapshot(args.machine_name),
            },
        )
    return 0


def run_config_summary(args: argparse.Namespace) -> int:
    config = build_config(
        target=args.target,
        machine_name=args.machine_name,
        variant=args.variant,
        logger_backend="none",
        trace_output_dir=SCRIPT_DIR / "outputs/config-traces" / args.machine_name / args.target,
        max_epochs=DEFAULT_MAX_EPOCHS,
        max_train_steps=None,
        max_validation_batches=None,
    )
    payload = serialize_train_config(config)
    flash_settings = _flash_vqg_settings(config)
    if args.output_json:
        _save_json(args.output_json, payload)
    print(
        json.dumps(
            {
                "run_id": config.run_id,
                "launch_id": config.launch_id,
                "embed_dropout": config.model.embed_dropout,
                "resid_dropout": config.model.resid_dropout,
                "drop_path": config.model.drop_path,
                "max_epochs": config.max_epochs,
                **flash_settings,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def run_collect(args: argparse.Namespace) -> int:
    outputs_dir = args.outputs_dir
    artifact_dir = args.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, Any]] = []
    queue_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    status_rank = {"pending": 0, "started": 1, "completed": 2}
    for status_path in sorted(outputs_dir.glob("*/queue-status.tsv")):
        rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        with status_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                row = dict(row)
                row["status_path"] = str(status_path)
                key = (row.get("queue", ""), row.get("target", ""))
                prev = rows_by_key.get(key)
                current_status = str(row.get("status", ""))
                prev_status = str(prev.get("status", "")) if prev else ""
                current_rank = status_rank.get(current_status, 99 if current_status.startswith("failed") else -1)
                prev_rank = status_rank.get(prev_status, 99 if prev_status.startswith("failed") else -1)
                if prev is None or current_rank >= prev_rank:
                    rows_by_key[key] = row
        for row in rows_by_key.values():
            queue_rows.append(row)
            if row.get("status") != "completed":
                invalid_rows.append(row)
                continue
            log_path = Path(str(row.get("log", "")))
            result_path = Path(str(row.get("result_json", "")))
            config_path = Path(str(row.get("config_json", "")))
            machine = str(row.get("machine", ""))
            target = str(row.get("target", ""))
            metrics = _parse_final_metrics(log_path)
            config_payload = _read_json(config_path) if config_path.exists() else {}
            result_payload = _read_json(result_path) if result_path.exists() else {}
            config_model = config_payload.get("model") or {}
            duration_seconds = ""
            started_at = row.get("started_at", "")
            finished_at = row.get("finished_at", "")
            try:
                start = datetime.fromisoformat(str(started_at))
                finish = datetime.fromisoformat(str(finished_at))
                duration_seconds = (finish - start).total_seconds()
            except Exception:
                pass
            run_rows.append(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "machine": machine,
                    "queue": row.get("queue", ""),
                    "target": target,
                    "variant": row.get("variant", ""),
                    "gpu": row.get("gpu", ""),
                    "status": row.get("status", ""),
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "duration_seconds": duration_seconds,
                    "duration_minutes": float(duration_seconds) / 60.0 if duration_seconds != "" else "",
                    "run_id": config_payload.get("run_id", ""),
                    "launch_id": config_payload.get("launch_id", ""),
                    "embed_dropout": (config_payload.get("model") or {}).get("embed_dropout", ""),
                    "resid_dropout": (config_payload.get("model") or {}).get("resid_dropout", ""),
                    "drop_path": (config_payload.get("model") or {}).get("drop_path", ""),
                    "max_epochs": config_payload.get("max_epochs", ""),
                    "num_codebook_vectors": _find_nested_key(config_model, "num_codebook_vectors"),
                    "fox_remote_read_topk": _find_nested_key(config_model, "fox_remote_read_topk"),
                    "fox_gd_residual_dense_read_chunked": _find_nested_key(
                        config_model, "fox_gd_residual_dense_read_chunked"
                    ),
                    "fox_gd_residual_rank": _find_nested_key(
                        config_model, "fox_gd_residual_rank"
                    ),
                    "fox_gd_residual_write_topk": _find_nested_key(
                        config_model, "fox_gd_residual_write_topk"
                    ),
                    "final_valid_loss": metrics.get("final_valid_loss", ""),
                    "final_valid_accuracy": metrics.get("final_valid_accuracy", ""),
                    "best_valid_accuracy": metrics.get("best_valid_accuracy", ""),
                    "final_1024x256_accuracy": metrics.get("final_1024x256_accuracy", ""),
                    "best_1024x256_accuracy": metrics.get("best_1024x256_accuracy", ""),
                    "best_final_1024x256_gap": metrics.get("best_final_1024x256_gap", ""),
                    "n_validation_summaries": metrics.get("n_validation_summaries", ""),
                    "n_validation_summary_lines": metrics.get("n_validation_summary_lines", ""),
                    "log_path": str(log_path),
                    "log_sha256": _sha256(log_path) if log_path.exists() else "",
                    "result_json": str(result_path),
                    "result_sha256": _sha256(result_path) if result_path.exists() else "",
                    "config_json": str(config_path),
                    "config_sha256": _sha256(config_path) if config_path.exists() else "",
                    "zoology_commit": ((result_payload.get("env") or {}).get("zoology_commit", "")),
                    "flash_vqg_commit": ((result_payload.get("env") or {}).get("flash_vqg_commit", "")),
                }
            )

    cache_rows: list[dict[str, Any]] = []
    init_rows: list[dict[str, Any]] = []
    for path in sorted(outputs_dir.glob("*/cache-hash.json")):
        payload = _read_json(path)
        cache = payload.get("cache") or {}
        cache_rows.append(
            {
                "machine": payload.get("machine_name", ""),
                "target": payload.get("target", ""),
                "variant": payload.get("variant", ""),
                "file_count": cache.get("file_count", ""),
                "combined_content_sha256": cache.get("combined_content_sha256", ""),
                "expected_combined_content_sha256": cache.get("expected_combined_content_sha256", ""),
                "match_expected": cache.get("match_expected", ""),
                "path": str(path),
                "sha256": _sha256(path),
            }
        )
    for path in sorted(outputs_dir.glob("*/init-verify.json")):
        payload = _read_json(path)
        init = payload.get("init_checkpoint") or {}
        init_rows.append(
            {
                "machine": payload.get("machine_name", ""),
                "checkpoint": init.get("checkpoint", ""),
                "expected_model_state_sha256": init.get("expected_model_state_sha256", ""),
                "embedded_model_state_sha256": init.get("embedded_model_state_sha256", ""),
                "actual_model_state_sha256": init.get("actual_model_state_sha256", ""),
                "match_expected": init.get("match_expected", ""),
                "match_embedded": init.get("match_embedded", ""),
                "path": str(path),
                "sha256": _sha256(path),
            }
        )

    comparison_rows: list[dict[str, Any]] = []
    ref_values = [
        row for row in run_rows if row.get("machine") == "2080ti" and row.get("final_1024x256_accuracy") not in ("", None)
    ]
    if ref_values:
        ref = ref_values[0]
        ref_acc = float(ref["final_1024x256_accuracy"])
        for row in run_rows:
            if row is ref or row.get("final_1024x256_accuracy") in ("", None):
                continue
            acc = float(row["final_1024x256_accuracy"])
            gap = abs(ref_acc - acc)
            comparison_rows.append(
                {
                    "reference_machine": ref.get("machine", ""),
                    "reference_target": ref.get("target", ""),
                    "reference_1024x256_accuracy": ref_acc,
                    "candidate_machine": row.get("machine", ""),
                    "candidate_target": row.get("target", ""),
                    "candidate_1024x256_accuracy": acc,
                    "absolute_gap": gap,
                    "gap_percentage_points": gap * 100.0,
                    "within_4pp": gap <= 0.04,
                }
            )

    source_rows: list[dict[str, Any]] = []
    source_candidates: list[Path] = []
    source_candidates.extend(sorted(outputs_dir.glob("*/queue-status.tsv")))
    source_candidates.extend(sorted(outputs_dir.glob("*/cache-hash.json")))
    source_candidates.extend(sorted(outputs_dir.glob("*/init-verify.json")))
    source_candidates.extend(sorted(outputs_dir.glob("*/preflight.json")))
    source_candidates.extend(sorted(outputs_dir.glob("*/configs/*.json")))
    source_candidates.extend(sorted(outputs_dir.glob("*/results/*.json")))
    source_candidates.extend(sorted(outputs_dir.glob("*/logs/*.log")))
    source_candidates.extend(sorted(outputs_dir.glob("*.nohup.log")))
    source_candidates.extend(sorted(outputs_dir.glob("*.setsid.log")))
    source_candidates.extend(sorted(outputs_dir.glob("*.sha256")))
    for path in source_candidates:
        machine, source_host, source_path, mirror_path = _source_and_mirror_paths(path)
        source_rows.append(
            {
                "machine": machine,
                "source_host": source_host,
                "source_path": source_path,
                "mirror_path": mirror_path,
                "bytes": path.stat().st_size if path.exists() else "",
                "sha256": _sha256(path) if path.exists() else "",
                "mirrored_to_main_workspace": True,
            }
        )

    _write_csv(artifact_dir / "run-summary.csv", run_rows)
    _write_csv(artifact_dir / "cross-machine-comparison.csv", comparison_rows)
    _write_csv(artifact_dir / "cache-init-preflight-summary.csv", cache_rows + init_rows)
    queue_fieldnames = [
        "queue",
        "machine",
        "target",
        "variant",
        "gpu",
        "pid",
        "status",
        "log",
        "config_json",
        "result_json",
        "started_at",
        "finished_at",
        "status_path",
    ]
    _write_csv(artifact_dir / "queue-summary.csv", queue_rows, fieldnames=queue_fieldnames)
    _write_csv(artifact_dir / "invalid-runs.csv", invalid_rows, fieldnames=queue_fieldnames)
    _write_csv(artifact_dir / "source-manifest.csv", source_rows)
    _save_json(
        artifact_dir / "metadata.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "status": "collected",
            "ledger": "not written; diagnostic/exploratory screen",
            "outputs_dir": str(outputs_dir),
            "artifact_dir": str(artifact_dir),
            "summary": {
                "run_count": len(run_rows),
                "completed_count": sum(1 for row in run_rows if row.get("status") == "completed"),
                "invalid_count": len(invalid_rows),
                "cache_hash_records": len(cache_rows),
                "init_verify_records": len(init_rows),
                "comparison_count": len(comparison_rows),
                "all_1024x256_within_4pp": (
                    all(str(row.get("within_4pp")) == "True" or row.get("within_4pp") is True for row in comparison_rows)
                    if comparison_rows
                    else False
                ),
            },
            "expected_cache_combined_sha256": EXPECTED_CACHE_COMBINED_SHA256,
            "expected_init_state_sha256": EXPECTED_INIT_STATE_SHA256,
            "expected_num_codebook_vectors": EXPECTED_NUM_CODEBOOK_VECTORS,
            "expected_fox_remote_read_topk": EXPECTED_REMOTE_READ_TOPK,
            "expected_fox_gd_residual_dense_read_chunked": EXPECTED_DENSE_READ_CHUNKED,
            "zoology_branch": _git_value(REPO_ROOT, ["branch", "--show-current"]),
            "zoology_commit": _git_value(REPO_ROOT, ["rev-parse", "--short", "HEAD"]),
            "flash_vqg_branch": _git_value(FLASH_VQG_ROOT, ["branch", "--show-current"]),
            "flash_vqg_commit": _git_value(FLASH_VQG_ROOT, ["rev-parse", "--short", "HEAD"]),
        },
    )
    readme = (
        f"# {EXPERIMENT_ID}\n\n"
        "本 artifact 收尾 dense-read 4 epoch confirm. 本轮是 diagnostic / confirm screen, 不写 official MQAR ledger.\n\n"
        "关键配置: `fox_remote_read_topk=64`, `num_codebook_vectors=64`, "
        "`fox_gd_residual_dense_read_chunked=True`, no-dropout, canonical cache/init.\n\n"
        "## 文件\n\n"
        "- `run-summary.csv`: per-run final metrics.\n"
        "- `cross-machine-comparison.csv`: 以 2080ti run 为参考的 1024x256 gap.\n"
        "- `cache-init-preflight-summary.csv`: cache content hash 与 init state hash preflight.\n"
        "- `queue-summary.csv`: queue 状态.\n"
        "- `invalid-runs.csv`: failed/interrupted/pending run.\n"
        "- `source-manifest.csv`: raw evidence 路径和 sha256.\n"
        "- `metadata.json`: 收尾元数据.\n"
        "\n"
        "注: `n_validation_summaries` 对 tqdm 相邻重复 summary 做了去重; "
        "`n_validation_summary_lines` 保留原始日志匹配行数. `best_*` 是日志观测到的 best validation metric, "
        "不是 saved-best checkpoint 复评.\n"
    )
    (artifact_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"artifact_dir": str(artifact_dir), "run_count": len(run_rows)}, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("cache-hash")
    p.add_argument("--machine-name", required=True)
    p.add_argument("--target", choices=TARGETS, default=TARGETS[0])
    p.add_argument("--variant", choices=[VARIANT], default=VARIANT)
    p.add_argument("--output-json", type=Path)
    p.set_defaults(func=run_cache_hash)

    p = sub.add_parser("verify-init")
    p.add_argument("--machine-name", required=True)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_INIT_CHECKPOINT)
    p.add_argument("--output-json", type=Path)
    p.set_defaults(func=run_verify_init)

    p = sub.add_parser("preflight")
    p.add_argument("--machine-name", required=True)
    p.add_argument("--target", choices=TARGETS, default=TARGETS[0])
    p.add_argument("--variant", choices=[VARIANT], default=VARIANT)
    p.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    p.add_argument("--output-json", type=Path)
    p.set_defaults(func=run_preflight)

    p = sub.add_parser("config-summary")
    p.add_argument("--machine-name", required=True)
    p.add_argument("--target", choices=TARGETS, default=TARGETS[0])
    p.add_argument("--variant", choices=[VARIANT], default=VARIANT)
    p.add_argument("--output-json", type=Path)
    p.set_defaults(func=run_config_summary)

    p = sub.add_parser("train")
    p.add_argument("--machine-name", required=True)
    p.add_argument("--target", choices=TARGETS, required=True)
    p.add_argument("--variant", choices=[VARIANT], default=VARIANT)
    p.add_argument("--init-checkpoint", type=Path, required=True)
    p.add_argument("--trace-output-dir", type=Path, required=True)
    p.add_argument("--output-config-json", type=Path)
    p.add_argument("--output-result-json", type=Path)
    p.add_argument("--logger-backend", choices=["none", "swanlab", "wandb"], default="none")
    p.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    p.add_argument("--max-train-steps", type=int)
    p.add_argument("--max-validation-batches", type=int)
    p.set_defaults(func=run_train)

    p = sub.add_parser("collect")
    p.add_argument("--outputs-dir", type=Path, default=SCRIPT_DIR / "outputs")
    p.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    p.set_defaults(func=run_collect)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    os.chdir(REPO_ROOT)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
