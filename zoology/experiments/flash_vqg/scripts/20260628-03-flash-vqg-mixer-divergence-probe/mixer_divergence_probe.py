#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import random
import socket
import subprocess
import sys
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get("ZOOLOGY_REPO_ROOT", "/home/lyj/mnt/project/zoology")).resolve()
FLASH_VQG_ROOT = Path(os.environ.get("FLASH_VQG_ROOT", "/home/lyj/mnt/project/Flash-VQG")).resolve()
EXPERIMENT_ID = "20260628-03-flash-vqg-mixer-divergence-probe"
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
EXPECTED_CACHE_COMBINED_SHA256 = (
    "d9098e876a036b8cb90a7186174fd827e0f5b422482266772850069c905bd8c8"
)
EXPECTED_INIT_STATE_SHA256 = (
    "dd0b7bb57b8bbb1fd2bd6dbcfcaa82d609c318b9b1c9ead0e89032f17bf16edf"
)
BASE_TARGET = "no-dropout-4ep-s123-r1"
TARGET = "mixer-probe-s123-r1"
VARIANT = "no-dropout"
DEFAULT_TRACE_STEPS = "0,1,4,16,64,130,203,352,448,704"
ARTIFACT_DIR = REPO_ROOT / "docs/artifacts" / EXPERIMENT_ID

if str(FLASH_VQG_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(FLASH_VQG_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from zoology.checkpoints import load_checkpoint_payload, resolve_checkpoint_path, serialize_train_config  # noqa: E402
from zoology.config import CheckpointConfig, LoggerConfig  # noqa: E402
from zoology.data.utils import prepare_data  # noqa: E402
from zoology.model import LanguageModel  # noqa: E402
from zoology.utils import set_determinism  # noqa: E402


TRACE_ORDER = [
    "phase1/q_all",
    "phase1/k_all",
    "phase1/v_all",
    "phase1/g_raw_all",
    "phase1/K_q_all",
    "phase1/Delta_all",
    "phase1/W_all",
    "state_build/logf_all",
    "state_build/beta_all",
    "state_build/G_state",
    "state_build/L_state",
    "state_build/M_state",
    "phase2/S_far",
    "phase2/O_base",
    "phase2_read/top_idx",
    "phase2_read/top_scores",
    "phase2_read/top_probs",
    "phase2_read/omega_sel",
    "phase2_read/read_selected_mass",
    "phase2_read/u_res",
    "phase2/O_res_added",
    "phase2/Out_f32",
    "output_proj/O_heads",
    "output_proj/o_heads",
    "output_proj/res",
    "forward/logits",
    "forward/preds",
    "forward/loss",
]


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FIRST_PROBE = _load_module(FIRST_PROBE_SCRIPT, "flash_vqg_first_probe_lib")
NO_DROPOUT = _load_module(NO_DROPOUT_SCRIPT, "flash_vqg_no_dropout_lib")


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


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
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


def _parse_int_set(value: str | None) -> set[int]:
    if value is None or not str(value).strip():
        return set()
    return {int(part.strip()) for part in str(value).split(",") if part.strip()}


def _build_config(machine_name: str, max_optimizer_steps: int):
    config = NO_DROPOUT.build_config(
        target=BASE_TARGET,
        machine_name=machine_name,
        variant=VARIANT,
        logger_backend="none",
        trace_output_dir=SCRIPT_DIR / "outputs/legacy-read-trace-disabled" / machine_name,
        max_epochs=1,
        max_train_steps=max_optimizer_steps,
        max_validation_batches=1,
    )
    config.launch_id = f"fvqg-{EXPERIMENT_ID}-{machine_name}-{TARGET}"
    config.run_id = f"{EXPERIMENT_ID}-{machine_name}-{TARGET}"
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
    config.model.embed_dropout = 0.0
    config.model.resid_dropout = 0.0
    config.model.drop_path = 0.0
    FIRST_PROBE._set_model_kwarg(config.model, "enable_layer_metrics", True, require=False)
    FIRST_PROBE._set_model_kwarg(config.model, "fox_phase2_metrics_mode", "full", require=False)
    return config


def _env_snapshot(machine_name: str) -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    return {
        "experiment_id": EXPERIMENT_ID,
        "machine_name": machine_name,
        "utc_time": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "torch_version": torch.__version__,
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


def _set_mixer_trace_runtime(model: torch.nn.Module, runtime: dict[str, Any]) -> int:
    count = 0

    def setter(module: torch.nn.Module) -> None:
        nonlocal count
        setter_fn = getattr(module, "set_mixer_trace_runtime", None)
        if setter_fn is not None:
            setter_fn(runtime)
            count += 1

    model.apply(setter)
    return count


def _clear_mixer_trace_runtime(model: torch.nn.Module) -> None:
    def clearer(module: torch.nn.Module) -> None:
        clearer_fn = getattr(module, "clear_mixer_trace_runtime", None)
        if clearer_fn is not None:
            clearer_fn()

    model.apply(clearer)


def _forward_record(
    *,
    optimizer_step: int,
    micro_step: int,
    window_micro_idx: int,
    loss: torch.Tensor,
    aux_loss_value: torch.Tensor | int,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    logits: torch.Tensor,
    preds: torch.Tensor,
    slices: list[dict[str, Any]],
    mixer_trace_records: list[dict[str, Any]],
    sample_count: int,
) -> dict[str, Any]:
    return {
        "record_type": "forward",
        "optimizer_step": int(optimizer_step),
        "micro_step": int(micro_step),
        "window_micro_idx": int(window_micro_idx),
        "loss": float(loss.detach().cpu().item()),
        "aux_loss": (
            float(aux_loss_value.detach().cpu().item())
            if isinstance(aux_loss_value, torch.Tensor)
            else float(aux_loss_value)
        ),
        "inputs_sha256": FIRST_PROBE._hash_tensor(inputs),
        "targets_sha256": FIRST_PROBE._hash_tensor(targets),
        "logits_sha256": FIRST_PROBE._hash_tensor(logits),
        "preds_sha256": FIRST_PROBE._hash_tensor(preds),
        "inputs": FIRST_PROBE._tensor_summary(inputs, include_hash=False, sample_count=sample_count),
        "targets": FIRST_PROBE._tensor_summary(targets, include_hash=False, sample_count=sample_count),
        "logits": FIRST_PROBE._tensor_summary(logits, include_hash=False, sample_count=sample_count),
        "slice0": slices[0] if slices else {},
        "prediction_margins": FIRST_PROBE._prediction_margin_summary(
            logits,
            preds,
            targets,
            max_positions=16,
        ),
        "scalar_metrics": {},
        "mixer_trace_records": mixer_trace_records,
    }


def run_preflight(args: argparse.Namespace) -> int:
    config = _build_config(args.machine_name, max_optimizer_steps=args.max_optimizer_steps)
    cache = FIRST_PROBE._hash_cache_for_config(config.data)
    init = FIRST_PROBE._verify_init_checkpoint(args.init_checkpoint)
    if config.input_type != "discrete":
        raise ValueError("This probe expects discrete MQAR config.")
    _model = LanguageModel(config.model)
    train_dataloader, _ = prepare_data(config.data)
    batch_order = FIRST_PROBE._batch_order_hash(train_dataloader)
    payload = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "machine_name": args.machine_name,
        "target": TARGET,
        "max_optimizer_steps": int(args.max_optimizer_steps),
        "environment": _env_snapshot(args.machine_name),
        "cache": cache,
        "init_checkpoint": init,
        "batch_order": batch_order,
        "serialized_config": serialize_train_config(config),
        "match": {
            "cache": cache.get("combined_content_sha256") == EXPECTED_CACHE_COMBINED_SHA256,
            "init": init.get("actual_model_state_sha256") == EXPECTED_INIT_STATE_SHA256,
        },
    }
    output = args.output_json or (SCRIPT_DIR / "outputs" / args.machine_name / "preflight.json")
    _save_json(output, payload)
    if not payload["match"]["cache"] or not payload["match"]["init"]:
        raise RuntimeError("Preflight hash mismatch.")
    print(f"wrote {output}")
    return 0


def run_probe(args: argparse.Namespace) -> int:
    if args.max_optimizer_steps <= 0:
        raise ValueError("--max-optimizer-steps must be positive.")
    os.chdir(REPO_ROOT)
    random.seed(args.pre_seed)
    np.random.seed(args.pre_seed)
    torch.manual_seed(args.pre_seed)

    config = _build_config(args.machine_name, max_optimizer_steps=args.max_optimizer_steps)
    config.init_checkpoint_path = str(args.init_checkpoint)
    config.init_checkpoint_strict = True
    set_determinism(config.seed, deterministic=False)

    cache = FIRST_PROBE._hash_cache_for_config(config.data)
    init = FIRST_PROBE._verify_init_checkpoint(args.init_checkpoint)
    if cache.get("combined_content_sha256") != EXPECTED_CACHE_COMBINED_SHA256:
        raise RuntimeError("MQAR cache content hash mismatch.")
    if init.get("actual_model_state_sha256") != EXPECTED_INIT_STATE_SHA256:
        raise RuntimeError("Init checkpoint tensor hash mismatch.")

    if config.input_type != "discrete":
        raise ValueError("This probe expects discrete MQAR config.")
    model = LanguageModel(config.model)
    train_dataloader, _ = prepare_data(config.data)
    resolved_init = resolve_checkpoint_path(args.init_checkpoint, which="best")
    checkpoint_payload = load_checkpoint_payload(resolved_init, map_location="cpu")
    model.load_state_dict(checkpoint_payload["model_state_dict"], strict=True)
    after_load_hash = FIRST_PROBE._hash_model_params(model)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    model.to(device)
    model.train()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    trace_steps = _parse_int_set(args.trace_optimizer_steps)
    sampler = getattr(train_dataloader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(0)

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
            "model_params_sha256": FIRST_PROBE._hash_model_params(model),
        },
    ]
    trace_setter_count = 0
    accum_steps = int(config.gradient_accumulation_steps)
    optimizer.zero_grad()
    accum_loss = 0.0
    optimizer_step = 0
    micro_step = 0
    completed_optimizer_steps = 0

    try:
        for inputs, targets, slices in train_dataloader:
            window_micro_idx = micro_step % accum_steps
            trace_this_forward = optimizer_step in trace_steps and window_micro_idx == 0
            runtime = {
                "enabled": trace_this_forward,
                "experiment_id": EXPERIMENT_ID,
                "run_id": config.run_id,
                "machine": args.machine_name,
                "optimizer_step": int(optimizer_step),
                "micro_step": int(micro_step),
                "window_micro_idx": int(window_micro_idx),
                "layer_idx": int(args.layer_idx),
                "sample_count": int(args.trace_sample_count),
                "records": [],
            }
            inputs = inputs.to(device)
            targets = targets.to(device)
            FIRST_PROBE._set_dense_teacher_runtime(model, targets, config.input_type)
            if trace_this_forward:
                trace_setter_count = max(trace_setter_count, _set_mixer_trace_runtime(model, runtime))
            try:
                loss, preds, logits = FIRST_PROBE._compute_loss(
                    model,
                    config.input_type,
                    config.loss_type,
                    inputs,
                    targets,
                    loss_fn,
                )
            finally:
                FIRST_PROBE._clear_dense_teacher_runtime(model)
                _clear_mixer_trace_runtime(model)

            aux_loss_value: torch.Tensor | int = 0
            if config.input_type == "discrete":
                aux_loss_value = FIRST_PROBE._auxiliary_loss(model)
                if aux_loss_value:
                    loss = loss + aux_loss_value

            if trace_this_forward:
                trace_records = list(runtime.get("records") or [])
                records.append(
                    _forward_record(
                        optimizer_step=optimizer_step,
                        micro_step=micro_step,
                        window_micro_idx=window_micro_idx,
                        loss=loss,
                        aux_loss_value=aux_loss_value,
                        inputs=inputs,
                        targets=targets,
                        logits=logits,
                        preds=preds,
                        slices=slices,
                        mixer_trace_records=trace_records,
                        sample_count=args.trace_sample_count,
                    )
                )

            (loss / accum_steps).backward()
            accum_loss += float(loss.detach().cpu().item())
            micro_step += 1

            if micro_step <= args.hash_micro_steps:
                records.append(
                    {
                        "record_type": "backward",
                        "optimizer_step": int(optimizer_step),
                        "micro_step": int(micro_step),
                        "loss": float(loss.detach().cpu().item()),
                        "grad_sha256": FIRST_PROBE._hash_model_grads(model),
                        "model_params_sha256": FIRST_PROBE._hash_model_params(model),
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
                        "optimizer_step": int(optimizer_step),
                        "micro_step": int(micro_step),
                        "avg_loss": float(accum_loss / accum_steps),
                        "model_params_sha256": FIRST_PROBE._hash_model_params(model),
                        "optimizer_state_sha256": FIRST_PROBE._hash_optimizer_state(optimizer),
                    }
                )
                accum_loss = 0.0
                if optimizer_step >= args.max_optimizer_steps:
                    break

    finally:
        _clear_mixer_trace_runtime(model)

    output = args.output_json or (SCRIPT_DIR / "outputs" / args.machine_name / "probe.json")
    payload = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "probe": {
            "machine_name": args.machine_name,
            "target": TARGET,
            "variant": VARIANT,
            "device": str(device),
            "layer_idx": int(args.layer_idx),
            "max_optimizer_steps": int(args.max_optimizer_steps),
            "trace_optimizer_steps": sorted(trace_steps),
            "trace_setter_count": int(trace_setter_count),
            "init_checkpoint": str(resolved_init),
        },
        "environment": _env_snapshot(args.machine_name),
        "config_summary": {
            "seed": config.seed,
            "data_seed": config.data.seed,
            "batch_size": config.data.batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "embed_dropout": config.model.embed_dropout,
            "resid_dropout": config.model.resid_dropout,
            "drop_path": config.model.drop_path,
        },
        "serialized_config": serialize_train_config(config),
        "cache": cache,
        "init_checkpoint": init,
        "batch_order": FIRST_PROBE._batch_order_hash(train_dataloader),
        "records": records,
        "completed_optimizer_steps": int(completed_optimizer_steps),
    }
    _save_json(output, payload)
    print(f"wrote {output}")
    return 0


def _iter_trace_rows(payload: dict[str, Any], source: Path) -> Iterable[dict[str, Any]]:
    probe = payload.get("probe") or {}
    machine = probe.get("machine_name")
    for record in payload.get("records") or []:
        if record.get("record_type") != "forward":
            continue
        base = {
            "machine": machine,
            "target": probe.get("target"),
            "variant": probe.get("variant"),
            "optimizer_step": record.get("optimizer_step"),
            "micro_step": record.get("micro_step"),
            "window_micro_idx": record.get("window_micro_idx"),
            "source_json": str(source),
        }
        for item in record.get("mixer_trace_records") or []:
            yield {
                **base,
                "layer_idx": item.get("layer_idx"),
                "trace_name": item.get("trace_name"),
                "sha256": item.get("sha256"),
                "dtype": item.get("dtype"),
                "shape": json.dumps(item.get("shape"), separators=(",", ":")),
                "numel": item.get("numel"),
                "mean": item.get("mean"),
                "std": item.get("std"),
                "min": item.get("min"),
                "max": item.get("max"),
                "l2_norm": item.get("l2_norm"),
                "sample": json.dumps(item.get("sample"), separators=(",", ":"), default=_json_default),
            }
        yield {
            **base,
            "layer_idx": "",
            "trace_name": "forward/logits",
            "sha256": record.get("logits_sha256"),
            "dtype": record.get("logits", {}).get("dtype"),
            "shape": json.dumps(record.get("logits", {}).get("shape"), separators=(",", ":")),
            "numel": record.get("logits", {}).get("numel"),
            "mean": record.get("logits", {}).get("mean"),
            "std": record.get("logits", {}).get("std"),
            "min": record.get("logits", {}).get("min"),
            "max": record.get("logits", {}).get("max"),
            "l2_norm": record.get("logits", {}).get("l2_norm"),
            "sample": json.dumps(record.get("logits", {}).get("sample"), separators=(",", ":"), default=_json_default),
        }
        yield {
            **base,
            "layer_idx": "",
            "trace_name": "forward/preds",
            "sha256": record.get("preds_sha256"),
            "dtype": "",
            "shape": "",
            "numel": "",
            "mean": "",
            "std": "",
            "min": "",
            "max": "",
            "l2_norm": "",
            "sample": "",
        }
        yield {
            **base,
            "layer_idx": "",
            "trace_name": "forward/loss",
            "sha256": f"{float(record.get('loss')):.12g}",
            "dtype": "float",
            "shape": "[]",
            "numel": 1,
            "mean": record.get("loss"),
            "std": 0,
            "min": record.get("loss"),
            "max": record.get("loss"),
            "l2_norm": abs(float(record.get("loss"))),
            "sample": json.dumps([record.get("loss")], separators=(",", ":")),
        }


def _trace_rank(row: dict[str, Any]) -> tuple[int, int, int, str]:
    name = str(row.get("trace_name", ""))
    return (
        int(row.get("optimizer_step") or 0),
        int(row.get("micro_step") or 0),
        TRACE_ORDER.index(name) if name in TRACE_ORDER else 999,
        name,
    )


def _compare_trace_rows(trace_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str, str], dict[str, dict[str, Any]]] = {}
    for row in trace_rows:
        key = (
            str(row.get("optimizer_step", "")),
            str(row.get("micro_step", "")),
            str(row.get("layer_idx", "")),
            str(row.get("trace_name", "")),
        )
        by_key.setdefault(key, {})[str(row.get("machine"))] = row
    rows: list[dict[str, Any]] = []
    for key, values in by_key.items():
        ref = values.get("2080ti")
        cand = values.get("3090")
        if ref is None or cand is None:
            continue
        ref_mean = _float_or_none(ref.get("mean"))
        cand_mean = _float_or_none(cand.get("mean"))
        ref_l2 = _float_or_none(ref.get("l2_norm"))
        cand_l2 = _float_or_none(cand.get("l2_norm"))
        rows.append(
            {
                "optimizer_step": key[0],
                "micro_step": key[1],
                "layer_idx": key[2],
                "trace_name": key[3],
                "sha256_match": str(ref.get("sha256")) == str(cand.get("sha256")),
                "shape_match": str(ref.get("shape")) == str(cand.get("shape")),
                "dtype_match": str(ref.get("dtype")) == str(cand.get("dtype")),
                "ref_sha256": ref.get("sha256"),
                "candidate_sha256": cand.get("sha256"),
                "ref_mean": ref.get("mean"),
                "candidate_mean": cand.get("mean"),
                "abs_mean_gap": "" if ref_mean is None or cand_mean is None else abs(ref_mean - cand_mean),
                "ref_l2_norm": ref.get("l2_norm"),
                "candidate_l2_norm": cand.get("l2_norm"),
                "abs_l2_norm_gap": "" if ref_l2 is None or cand_l2 is None else abs(ref_l2 - cand_l2),
                "ref_sample": ref.get("sample"),
                "candidate_sample": cand.get("sample"),
            }
        )
    return sorted(rows, key=_trace_rank)


def _float_or_none(value: Any) -> float | None:
    with suppress(Exception):
        if value == "":
            return None
        return float(value)
    return None


def run_collect(args: argparse.Namespace) -> int:
    output_dir = args.outputs_dir
    payload_paths = sorted(output_dir.glob("*/probe.json"))
    payloads = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in payload_paths]
    trace_rows = [row for path, payload in payloads for row in _iter_trace_rows(payload, path)]
    comparison_rows = _compare_trace_rows(trace_rows)
    mismatch_rows = [row for row in comparison_rows if str(row.get("sha256_match")) == "False"]
    preflight_rows = []
    source_rows = []
    for path in sorted(output_dir.glob("*/*.json")):
        source_rows.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
        if path.name == "preflight.json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            preflight_rows.append(
                {
                    "machine": payload.get("machine_name"),
                    "cache_match": payload.get("match", {}).get("cache"),
                    "init_match": payload.get("match", {}).get("init"),
                    "cache_sha256": payload.get("cache", {}).get("combined_content_sha256"),
                    "init_sha256": payload.get("init_checkpoint", {}).get("actual_model_state_sha256"),
                    "batch_order_sha256": payload.get("batch_order", {}).get("sha256"),
                    "zoology_commit": payload.get("environment", {}).get("zoology_commit"),
                    "flash_vqg_commit": payload.get("environment", {}).get("flash_vqg_commit"),
                }
            )

    artifact_dir = args.artifact_dir
    _write_csv(artifact_dir / "trace-summary.csv", trace_rows)
    _write_csv(artifact_dir / "cross-machine-trace-comparison.csv", comparison_rows)
    _write_csv(artifact_dir / "preflight-summary.csv", preflight_rows)
    _write_csv(artifact_dir / "source-manifest.csv", source_rows)
    metadata = {
        "experiment_id": EXPERIMENT_ID,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "payload_count": len(payloads),
        "trace_row_count": len(trace_rows),
        "comparison_row_count": len(comparison_rows),
        "mismatch_row_count": len(mismatch_rows),
        "first_mismatch": mismatch_rows[0] if mismatch_rows else None,
    }
    _save_json(artifact_dir / "metadata.json", metadata)
    readme = (
        f"# {EXPERIMENT_ID} artifact\n\n"
        "This artifact contains the layer-1 Flash-VQG mixer divergence probe summaries.\n\n"
        "- `trace-summary.csv`: per-machine trace hashes and summaries.\n"
        "- `cross-machine-trace-comparison.csv`: 2080ti vs 3090 joined trace comparison.\n"
        "- `preflight-summary.csv`: cache/init/batch-order/code preflight evidence.\n"
        "- `source-manifest.csv`: raw JSON evidence hashes.\n"
        "- `metadata.json`: first mismatch metadata.\n"
    )
    (artifact_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=EXPERIMENT_ID)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("preflight")
    p.add_argument("--machine-name", required=True)
    p.add_argument("--init-checkpoint", type=Path, default=DEFAULT_INIT_CHECKPOINT)
    p.add_argument("--max-optimizer-steps", type=int, default=1)
    p.add_argument("--output-json", type=Path)
    p.set_defaults(func=run_preflight)

    p = sub.add_parser("probe")
    p.add_argument("--machine-name", required=True)
    p.add_argument("--init-checkpoint", type=Path, default=DEFAULT_INIT_CHECKPOINT)
    p.add_argument("--device", default=None)
    p.add_argument("--max-optimizer-steps", type=int, default=1)
    p.add_argument("--trace-optimizer-steps", default=DEFAULT_TRACE_STEPS)
    p.add_argument("--layer-idx", type=int, default=1)
    p.add_argument("--trace-sample-count", type=int, default=8)
    p.add_argument("--hash-micro-steps", type=int, default=4)
    p.add_argument("--pre-seed", type=int, default=777)
    p.add_argument("--output-json", type=Path)
    p.set_defaults(func=run_probe)

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
