#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import random
import re
import statistics
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from einops import rearrange

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get("ZOOLOGY_REPO_ROOT", "/home/lyj/mnt/project/zoology")).resolve()
BASE_SCRIPT = (
    REPO_ROOT
    / "zoology/experiments/flash_vqg/scripts/20260630-04-flash-vqg-default-dropout-fixed-r4-1ep-screen/default_dropout_fixed_r4_1ep.py"
)
METRICS_YAML = (
    REPO_ROOT
    / "zoology/experiments/flash_vqg/scripts/20260625-01-flash-vqg-early-window-trace/metrics.yaml"
)
EXPERIMENT_ID = "20260701-02-flash-vqg-default-dropout-amplifier-trace"
ARTIFACT_DIR = REPO_ROOT / "docs/artifacts" / EXPERIMENT_ID
DEFAULT_INIT_CHECKPOINT = (
    REPO_ROOT
    / "zoology/experiments/flash_vqg/scripts/20260630-03-flash-vqg-s124-fixed-r4-4ep-confirm/outputs/canonical-init/cb64r16-s124-init.pt"
)
DEFAULT_MAX_EPOCHS = 1
DEFAULT_EMBED_DROPOUT = 0.1
DEFAULT_RESID_DROPOUT = 0.0
DEFAULT_DROP_PATH = 0.0
TRACE_TRAIN_STEPS = [0, 1, 4, 16, 64, 128]
DEFAULT_MAX_TRAIN_STEPS = 128
DEFAULT_CAPTURE_STEPS = "0,1,4,16,64,128"
FIRST_PROBE_SCRIPT = (
    REPO_ROOT
    / "zoology/experiments/flash_vqg/scripts/20260627-03-flash-vqg-first-divergence-probe/probe_first_divergence.py"
)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PRIOR = _load_module(BASE_SCRIPT, "default_dropout_fixed_r4_1ep_base")
FIRST = _load_module(FIRST_PROBE_SCRIPT, "flash_vqg_first_divergence_lib")
BASE = PRIOR.BASE
ORIGINAL_BASE_ARGS = PRIOR.ORIGINAL_BASE_ARGS
ORIGINAL_BUILD_CONFIG = PRIOR.ORIGINAL_BUILD_CONFIG


def _variant_from(base: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    merged = dict(base)
    merged.update(overrides)
    return merged


_R2 = {
    "kind": "fixed",
    "description": "fixed train-time read_topk=2",
    "fox_remote_read_topk": 2,
    "fox_remote_read_topk_initial": None,
    "fox_remote_read_topk_final": None,
    "fox_remote_read_topk_release_start_train_steps": 0,
    "fox_remote_read_topk_release_end_train_steps": 0,
    "fox_remote_read_topk_schedule": "linear_int",
    "fox_remote_read_topk_eval_policy": "scheduled",
    "fox_gd_residual_dense_read_chunked": False,
}
_R4 = {
    "kind": "fixed",
    "description": "fixed train-time read_topk=4",
    "fox_remote_read_topk": 4,
    "fox_remote_read_topk_initial": None,
    "fox_remote_read_topk_final": None,
    "fox_remote_read_topk_release_start_train_steps": 0,
    "fox_remote_read_topk_release_end_train_steps": 0,
    "fox_remote_read_topk_schedule": "linear_int",
    "fox_remote_read_topk_eval_policy": "scheduled",
    "fox_gd_residual_dense_read_chunked": False,
}

TARGETS = ("default-r4", "dropout005-r4", "default-r2")
VARIANTS: dict[str, dict[str, Any]] = {
    "default-r4": _variant_from(
        _R4,
        description="default-dropout fixed read_topk=4",
        embed_dropout=DEFAULT_EMBED_DROPOUT,
        residual_norm_mode=None,
    ),
    "dropout005-r4": _variant_from(
        _R4,
        description="fixed read_topk=4 with embed_dropout=0.05 boundary control",
        embed_dropout=0.05,
        residual_norm_mode=None,
    ),
    "default-r2": _variant_from(
        _R2,
        description="default-dropout fixed read_topk=2",
        embed_dropout=DEFAULT_EMBED_DROPOUT,
        residual_norm_mode=None,
    ),
}


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
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    if fieldnames is None:
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    keys.append(key)
                    seen.add(key)
        fieldnames = keys
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _variant_config(variant: str) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(f"Unsupported variant: {variant}")
    return VARIANTS[variant]


def _apply_run_suffix(config: Any, suffix: str | None) -> None:
    if suffix is None:
        return
    suffix = str(suffix).strip()
    if not suffix:
        return
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", suffix).strip("-")
    if not safe:
        return
    config.run_id = f"{config.run_id}-{safe}"
    if getattr(config, "launch_id", None):
        config.launch_id = f"{config.launch_id}-{safe}"


def _read_expected_init_hash() -> str:
    if os.environ.get("EXPECTED_INIT_STATE_SHA256"):
        return str(os.environ["EXPECTED_INIT_STATE_SHA256"])
    return str(getattr(PRIOR, "EXPECTED_INIT_STATE_SHA256", "2a1107bf22d0804ed485ab94bdc7af8004ef7b892a60c2967f842ba0f4b4efb0"))


def _patch_base() -> None:
    BASE.SCRIPT_DIR = SCRIPT_DIR
    BASE.EXPERIMENT_ID = EXPERIMENT_ID
    BASE.ARTIFACT_DIR = ARTIFACT_DIR
    BASE.TARGETS = TARGETS
    BASE.VARIANTS = VARIANTS
    BASE.METRICS_YAML = METRICS_YAML
    BASE.DEFAULT_INIT_CHECKPOINT = DEFAULT_INIT_CHECKPOINT
    BASE.DEFAULT_MAX_EPOCHS = DEFAULT_MAX_EPOCHS
    BASE.EXPECTED_INIT_STATE_SHA256 = _read_expected_init_hash()
    BASE.EXPECTED_TOTAL_OPTIMIZER_STEPS = BASE.EXPECTED_STEPS_PER_EPOCH * DEFAULT_MAX_EPOCHS

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
    ):
        args = ORIGINAL_BASE_ARGS(
            target=target,
            machine_name=machine_name,
            variant=variant,
            logger_backend=logger_backend,
            trace_output_dir=trace_output_dir,
            max_epochs=max_epochs,
            max_train_steps=max_train_steps,
            max_validation_batches=max_validation_batches,
        )
        args.seed_values = "124"
        args.project = "flash_vqg_default_dropout_r2_r4_overnight"
        args.metrics_white_list_file = str(METRICS_YAML)
        args.read_trace_train_steps = ",".join(str(step) for step in TRACE_TRAIN_STEPS)
        args.experiment_mode = f"{EXPERIMENT_ID}_{variant}_s124_d123_b64ga4_{machine_name}"
        args.run_id = f"{EXPERIMENT_ID}-{variant}-s124-d123-b64ga4-{machine_name}"
        args.launch_id_prefix = f"fvqg-{EXPERIMENT_ID}-{machine_name}-{target}"
        return args

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
        spec = _variant_config(variant)
        config = ORIGINAL_BUILD_CONFIG(
            target=target,
            machine_name=machine_name,
            variant=variant,
            logger_backend=logger_backend,
            trace_output_dir=trace_output_dir,
            max_epochs=max_epochs,
            max_train_steps=max_train_steps,
            max_validation_batches=max_validation_batches,
        )
        config.model.embed_dropout = float(spec.get("embed_dropout", DEFAULT_EMBED_DROPOUT))
        config.model.resid_dropout = DEFAULT_RESID_DROPOUT
        config.model.drop_path = DEFAULT_DROP_PATH
        config.read_trace_train_steps = list(TRACE_TRAIN_STEPS)
        config.metrics_white_list = list(config.metrics_white_list or [])
        residual_norm_mode = spec.get("residual_norm_mode")
        if residual_norm_mode is not None:
            BASE._set_flash_vqg_kwarg(config, "fox_gd_residual_residual_norm_mode", str(residual_norm_mode))
        return config

    BASE._base_args = _base_args
    BASE.build_config = build_config


_patch_base()


def _flash_setting(config: Any, key: str) -> Any:
    return BASE._find_nested_key(BASE.serialize_train_config(config).get("model") or {}, key)


def _configure_hash_probe_globals() -> None:
    FIRST.EXPERIMENT_ID = EXPERIMENT_ID
    FIRST.DEFAULT_INIT_CHECKPOINT = DEFAULT_INIT_CHECKPOINT
    FIRST.EXPECTED_INIT_STATE_SHA256 = _read_expected_init_hash()


def _build_hash_probe_config(
    *,
    target: str,
    machine_name: str,
    max_optimizer_steps: int,
) -> Any:
    config = BASE.build_config(
        target=target,
        machine_name=machine_name,
        variant=target,
        logger_backend="none",
        trace_output_dir=SCRIPT_DIR / "outputs" / "hash-trace-disabled" / machine_name / target,
        max_epochs=1,
        max_train_steps=max_optimizer_steps,
        max_validation_batches=1,
    )
    config.logger = FIRST.LoggerConfig(backend="none", project_name=None, entity=None)
    config.checkpoint = FIRST.CheckpointConfig(
        enabled=False,
        save_best=False,
        save_last=False,
        save_config_json=False,
    )
    config.read_churn_probe_enabled = False
    config.read_trace_enabled = False
    config.read_trace_train_steps = []
    config.read_trace_output_dir = None
    BASE._set_flash_vqg_kwarg(config, "enable_layer_metrics", True)
    BASE._set_flash_vqg_kwarg(config, "fox_phase2_metrics_mode", "full")
    config.init_checkpoint_path = str(DEFAULT_INIT_CHECKPOINT)
    config.init_checkpoint_strict = True
    return config


def _hash_probe_config_summary(config: Any, target: str) -> dict[str, Any]:
    payload = BASE.serialize_train_config(config)
    flash_kwargs = BASE._find_flash_vqg_kwargs(config.model)
    return {
        "target": target,
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
        "embed_dropout": config.model.embed_dropout,
        "resid_dropout": config.model.resid_dropout,
        "drop_path": config.model.drop_path,
        "flash_vqg": {
            key: flash_kwargs.get(key)
            for key in [
                "fox_remote_read_topk",
                "fox_gd_residual_write_topk",
                "fox_gd_residual_rank",
                "vq_weight_mode",
                "enable_layer_metrics",
                "fox_phase2_metrics_mode",
                "fox_remote_formula",
            ]
        },
        "serialized_model_keys": sorted((payload.get("model") or {}).keys()),
    }


def _hash_prediction_margin_summary(
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


def _compute_hash_probe_loss(
    model: torch.nn.Module,
    input_type: str,
    loss_type: str,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    loss_fn: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if input_type == "ce":
        input_type = "discrete"
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
    raise ValueError(f"Unsupported hash-probe loss_type={loss_type!r} input_type={input_type!r}")


def run_hash_probe(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("torch.cuda is unavailable inside this container.")
    _configure_hash_probe_globals()
    random.seed(args.pre_seed)
    np.random.seed(args.pre_seed)
    torch.manual_seed(args.pre_seed)

    config = _build_hash_probe_config(
        target=args.target,
        machine_name=args.machine_name,
        max_optimizer_steps=args.max_optimizer_steps,
    )
    BASE.set_determinism(config.seed)
    cache_payload = FIRST._hash_cache_for_config(config.data)
    if not cache_payload["match_expected"]:
        raise RuntimeError("MQAR cache content hash does not match canonical hash; stop before training.")
    init_payload = FIRST._verify_init_checkpoint(args.init_checkpoint)
    if not init_payload["match_expected"] or not init_payload["match_embedded"]:
        raise RuntimeError("Init checkpoint tensor hash does not match canonical hash; stop before training.")

    if config.input_type == "continuous":
        model = FIRST.ContinuousInputModel(config.model)
        train_dataloader, _ = FIRST.prepare_continuous_data(
            config.data,
            embeddings=model.backbone.embeddings.word_embeddings.weight.detach(),
        )
    else:
        model = FIRST.LanguageModel(config.model)
        train_dataloader, _ = FIRST.prepare_data(config.data)

    resolved_init = FIRST.resolve_checkpoint_path(args.init_checkpoint, which="best")
    checkpoint_payload = FIRST.load_checkpoint_payload(resolved_init, map_location="cpu")
    model.load_state_dict(checkpoint_payload["model_state_dict"], strict=True)
    after_load_hash = FIRST._hash_model_params(model)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    capture_steps = FIRST._parse_int_set(args.capture_optimizer_steps)
    capture = FIRST.ForwardCapture(model, sample_count=args.capture_sample_count)
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
            "model_params_sha256": FIRST._hash_model_params(model),
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
            FIRST._set_dense_teacher_runtime(model, targets, config.input_type)
            if capture_this_forward:
                capture.begin(stage)
            try:
                loss, preds, logits = _compute_hash_probe_loss(
                    model,
                    config.input_type,
                    config.loss_type,
                    inputs,
                    targets,
                    loss_fn,
                )
            finally:
                FIRST._clear_dense_teacher_runtime(model)
            module_records = capture.end() if capture_this_forward else []
            aux_loss_value: torch.Tensor | int = 0
            if config.input_type == "discrete":
                aux_loss_value = FIRST._auxiliary_loss(model)
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
                        "inputs_sha256": FIRST._hash_tensor(inputs),
                        "targets_sha256": FIRST._hash_tensor(targets),
                        "logits_sha256": FIRST._hash_tensor(logits),
                        "preds_sha256": FIRST._hash_tensor(preds),
                        "inputs": FIRST._tensor_summary(inputs, include_hash=False, sample_count=args.capture_sample_count),
                        "targets": FIRST._tensor_summary(targets, include_hash=False, sample_count=args.capture_sample_count),
                        "logits": FIRST._tensor_summary(logits, include_hash=False, sample_count=args.capture_sample_count),
                        "slice0": slices[0] if slices else {},
                        "prediction_margins": _hash_prediction_margin_summary(
                            logits,
                            preds,
                            targets,
                            max_positions=args.margin_sample_positions,
                        ),
                        "scalar_metrics": FIRST._scalar_metrics(model),
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
                        "grad_sha256": FIRST._hash_model_grads(model),
                        "model_params_sha256": FIRST._hash_model_params(model),
                        "scalar_metrics": FIRST._scalar_metrics(model),
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
                        "model_params_sha256": FIRST._hash_model_params(model),
                        "optimizer_state_sha256": FIRST._hash_optimizer_state(optimizer),
                        "scalar_metrics": FIRST._scalar_metrics(model),
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
            "variant": args.target,
            "target": args.target,
            "device": str(device),
            "max_optimizer_steps": int(args.max_optimizer_steps),
            "hash_micro_steps": int(args.hash_micro_steps),
            "capture_optimizer_steps": sorted(capture_steps),
            "init_checkpoint": str(resolved_init),
        },
        "environment": FIRST._env_snapshot(args.machine_name, args.target),
        "config_summary": _hash_probe_config_summary(config, args.target),
        "serialized_config": BASE.serialize_train_config(config),
        "cache": cache_payload,
        "init_checkpoint": init_payload,
        "batch_order": FIRST._batch_order_hash(train_dataloader),
        "records": records,
        "completed_optimizer_steps": int(completed_optimizer_steps),
    }
    if args.output_json:
        _save_json(args.output_json, payload)
    print(f"wrote {args.output_json}")
    return 0


def run_preflight(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("torch.cuda is unavailable inside this container.")
    config = BASE.build_config(
        target=args.target,
        machine_name=args.machine_name,
        variant=args.variant,
        logger_backend="none",
        trace_output_dir=SCRIPT_DIR / "outputs/preflight-traces" / args.machine_name / args.target,
        max_epochs=args.max_epochs,
        max_train_steps=args.max_train_steps,
        max_validation_batches=args.max_validation_batches,
    )
    _apply_run_suffix(config, args.run_suffix)
    train_loader, _ = BASE.prepare_data(config.data)
    train_batches = len(train_loader)
    accum = int(config.gradient_accumulation_steps)
    steps_per_epoch = (train_batches + accum - 1) // accum
    spec = _variant_config(args.variant)
    expected_steps = int(args.max_train_steps) if args.max_train_steps is not None else steps_per_epoch * int(args.max_epochs)
    flash_settings = BASE._flash_vqg_settings(config)
    payload = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "machine_name": args.machine_name,
        "target": args.target,
        "variant": args.variant,
        "variant_spec": spec,
        "env": BASE.env_snapshot(args.machine_name),
        "run_id": config.run_id,
        "launch_id": config.launch_id,
        "train_batches": train_batches,
        "gradient_accumulation_steps": accum,
        "optimizer_steps_per_epoch": steps_per_epoch,
        "max_epochs": int(config.max_epochs),
        "max_train_steps": config.max_train_steps,
        "expected_effective_optimizer_steps": expected_steps,
        "read_trace_train_steps": list(config.read_trace_train_steps),
        "cache_dir": config.data.cache_dir,
        "embed_dropout": config.model.embed_dropout,
        "resid_dropout": config.model.resid_dropout,
        "drop_path": config.model.drop_path,
        "fox_gd_residual_residual_norm_mode": _flash_setting(config, "fox_gd_residual_residual_norm_mode"),
        **flash_settings,
    }
    payload["passed"] = (
        train_batches == 2815
        and accum == 4
        and steps_per_epoch == BASE.EXPECTED_STEPS_PER_EPOCH
        and int(config.max_epochs) == int(args.max_epochs)
        and config.max_train_steps == args.max_train_steps
        and abs(float(config.model.embed_dropout) - float(spec.get("embed_dropout", DEFAULT_EMBED_DROPOUT))) < 1e-12
        and abs(float(config.model.resid_dropout) - DEFAULT_RESID_DROPOUT) < 1e-12
        and abs(float(config.model.drop_path) - DEFAULT_DROP_PATH) < 1e-12
        and flash_settings["num_codebook_vectors"] == BASE.EXPECTED_NUM_CODEBOOK_VECTORS
        and BASE._variant_settings_match(flash_settings, args.variant)
    )
    if args.output_json:
        _save_json(args.output_json, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default))
    return 0 if payload["passed"] else 1


def run_config_summary(args: argparse.Namespace) -> int:
    config = BASE.build_config(
        target=args.target,
        machine_name=args.machine_name,
        variant=args.variant,
        logger_backend="none",
        trace_output_dir=SCRIPT_DIR / "outputs/config-traces" / args.machine_name / args.target,
        max_epochs=args.max_epochs,
        max_train_steps=args.max_train_steps,
        max_validation_batches=args.max_validation_batches,
    )
    _apply_run_suffix(config, args.run_suffix)
    payload = BASE.serialize_train_config(config)
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
                "max_train_steps": config.max_train_steps,
                **BASE._flash_vqg_settings(config),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
    )
    return 0


def run_train(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("torch.cuda is unavailable inside this container.")
    config = BASE.build_config(
        target=args.target,
        machine_name=args.machine_name,
        variant=args.variant,
        logger_backend=args.logger_backend,
        trace_output_dir=args.trace_output_dir,
        max_epochs=args.max_epochs,
        max_train_steps=args.max_train_steps,
        max_validation_batches=args.max_validation_batches,
    )
    _apply_run_suffix(config, args.run_suffix)
    config.init_checkpoint_path = str(args.init_checkpoint)
    config.init_checkpoint_strict = True
    config.init_checkpoint_source_launch_id = "canonical-init-2080ti"
    config.init_checkpoint_source_run_id = "initlock-cb64r16-default-s124-r1-d123-b64ga4-2080ti"
    if args.output_config_json:
        _save_json(args.output_config_json, BASE.serialize_train_config(config))
    result = BASE.train(config)
    if args.output_result_json:
        _save_json(
            args.output_result_json,
            {
                "schema_version": 1,
                "experiment_id": EXPERIMENT_ID,
                "machine_name": args.machine_name,
                "target": args.target,
                "variant": args.variant,
                "variant_spec": _variant_config(args.variant),
                "init_checkpoint": str(args.init_checkpoint),
                "train_result": result,
                "env": BASE.env_snapshot(args.machine_name),
            },
        )
    return 0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _mean(values: list[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def _p05(values: list[float]) -> float | None:
    if not values:
        return None
    values = sorted(values)
    idx = max(0, min(len(values) - 1, int(round(0.05 * (len(values) - 1)))))
    return float(values[idx])


def _stable_key(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _machine_target_from_trace(path: Path, outputs_dir: Path) -> tuple[str, str]:
    rel = path.relative_to(outputs_dir / "traces")
    return rel.parts[0], rel.parts[1]


def _collect_early_window(outputs_dir: Path) -> list[dict[str, Any]]:
    selected = [
        "loss",
        "early_window/attn/gd_residual_read_candidate_retention_mean",
        "early_window/attn/gd_residual_read_candidate_churn_mean",
        "early_window/attn/gd_residual_read_candidate_top1_flip_rate",
        "early_window/attn/gd_residual_read_margin_top1_top2_mean",
        "early_window/attn/gd_residual_read_margin_top1_top2_p05",
        "early_window/attn/gd_residual_read_entropy_mean",
        "early_window/attn/gd_residual_read_selected_mass_mean",
        "early_window/attn/gd_residual_read_selected_mass_p05",
        "early_window/attn/gd_residual_lambda_mean",
        "early_window/attn/gd_residual_inject_ratio",
        "early_window/attn/gd_residual_m_norm_mean",
        "early_window/attn/gd_residual_m_norm_max",
        "early_window/attn/gd_residual_update_norm_mean",
        "early_window/attn/gd_residual_update_norm_p95",
        "early_window/attn/gd_residual_update_norm_max",
        "early_window/attn/gd_residual_write_strength_mean",
        "early_window/attn/gd_residual_write_strength_p95",
        "early_window/attn/gd_residual_write_strength_max",
        "early_window/attn/gd_residual_sum_zeta_mean",
        "early_window/attn/gd_residual_sum_zeta_p95",
        "early_window/attn/gd_residual_sum_zeta_max",
        "early_window/attn/gd_residual_raw_topk_mass_mean",
        "early_window/attn/gd_residual_raw_topk_mass_p05",
        "early_window/attn/gd_residual_write_top1_mass_mean",
        "early_window/attn/gd_residual_write_q_entropy_mean",
        "early_window/attn/gd_residual_write_q_top1_mean",
    ]
    rows: list[dict[str, Any]] = []
    for path in sorted((outputs_dir / "traces").glob("**/early_window_metrics.jsonl")):
        machine, target = _machine_target_from_trace(path, outputs_dir)
        for row in _read_jsonl(path):
            out = {
                "experiment_id": EXPERIMENT_ID,
                "machine": machine,
                "target": target,
                "run_id": row.get("run_id", ""),
                "train_step": row.get("train_step", ""),
                "valid_batches": row.get("valid_batches", ""),
                "source_path": str(path),
            }
            for key in selected:
                out[key.replace("early_window/attn/", "").replace("early_window/", "")] = row.get(key, "")
            for key, value in row.items():
                if re.match(r"early_window/layer_[01]/attn/gd_residual_(read_|inject|lambda|m_norm|update_norm)", str(key)):
                    out[key.replace("early_window/", "")] = value
            rows.append(out)
    return rows


def _collect_read_trace(outputs_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    raw_rows_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for path in sorted((outputs_dir / "traces").glob("**/train_step_*/read_trace.jsonl")):
        machine, target = _machine_target_from_trace(path.parents[1], outputs_dir)
        records = _read_jsonl(path)
        if not records:
            continue
        margins = [float(r["margin_top1_top2"]) for r in records if r.get("margin_top1_top2") is not None]
        entropy = [float(r["entropy"]) for r in records if r.get("entropy") is not None]
        mass = [float(r["selected_mass"]) for r in records if r.get("selected_mass") is not None]
        top1_ids = [tuple(r.get("topk_candidate_ids") or [None])[0] for r in records if r.get("topk_candidate_ids")]
        first = records[0]
        train_step = int(first.get("global_step", -1))
        summary_rows.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "machine": machine,
                "target": target,
                "run_id": first.get("run_id", ""),
                "train_step": train_step,
                "valid_batch_idx": first.get("valid_batch_idx", ""),
                "records": len(records),
                "read_topk": first.get("read_topk", ""),
                "margin_top1_top2_mean": _mean(margins),
                "margin_top1_top2_p05": _p05(margins),
                "entropy_mean": _mean(entropy),
                "selected_mass_mean": _mean(mass),
                "selected_mass_p05": _p05(mass),
                "unique_top1_ids": len(set(top1_ids)),
                "trace_path": str(path),
                "trace_sha256": _sha256(path),
            }
        )
        for r in records:
            key = (
                target,
                int(r.get("global_step", -1)),
                _stable_key(r.get("input_hash")),
                _stable_key(r.get("target_hash")),
                int(r.get("layer_idx", -1)),
                int(r.get("head_idx", -1)),
                int(r.get("block_idx", -1)),
                int(r.get("token_idx", -1)),
            )
            raw_rows_by_key[(machine, *key)] = r

    cross_rows: list[dict[str, Any]] = []
    keys = sorted({tuple(key) for machine, *key in raw_rows_by_key.keys()})
    for key in keys:
        r2080 = raw_rows_by_key.get(("2080ti", *key))
        r3090 = raw_rows_by_key.get(("3090", *key))
        if not r2080 or not r3090:
            continue
        ids2080 = [int(v) for v in (r2080.get("topk_candidate_ids") or [])]
        ids3090 = [int(v) for v in (r3090.get("topk_candidate_ids") or [])]
        overlap = len(set(ids2080) & set(ids3090))
        denom = max(1, min(len(ids2080), len(ids3090)))
        cross_rows.append(
            {
                "target": key[0],
                "train_step": key[1],
                "layer_idx": key[4],
                "head_idx": key[5],
                "block_idx": key[6],
                "token_idx": key[7],
                "input_hash": key[2],
                "target_hash": key[3],
                "top1_match": bool(ids2080 and ids3090 and ids2080[0] == ids3090[0]),
                "topk_exact_match": ids2080 == ids3090,
                "topk_overlap_ratio": overlap / denom,
                "ids_2080ti": json.dumps(ids2080, separators=(",", ":")),
                "ids_3090": json.dumps(ids3090, separators=(",", ":")),
            }
        )

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in cross_rows:
        grouped.setdefault((str(row["target"]), int(row["train_step"])), []).append(row)
    cross_summary: list[dict[str, Any]] = []
    for (target, train_step), rows in sorted(grouped.items()):
        cross_summary.append(
            {
                "target": target,
                "train_step": train_step,
                "records": len(rows),
                "top1_match_rate": _mean([1.0 if row["top1_match"] else 0.0 for row in rows]),
                "topk_exact_match_rate": _mean([1.0 if row["topk_exact_match"] else 0.0 for row in rows]),
                "topk_overlap_ratio_mean": _mean([float(row["topk_overlap_ratio"]) for row in rows]),
            }
        )
    return summary_rows, cross_rows, cross_summary


def _iter_hash_probe_rows(payload: dict[str, Any], source: str) -> Iterable[dict[str, Any]]:
    probe = payload.get("probe", {})
    machine = probe.get("machine_name")
    target = probe.get("target")
    cache = payload.get("cache") or {}
    if cache.get("combined_content_sha256"):
        yield {
            "source": source,
            "machine": machine,
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


def _hash_probe_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    stage = str(row.get("stage", ""))
    field = str(row.get("field", ""))
    module = str(row.get("module", ""))
    stage_rank = {
        "preflight": 0,
        "after_init_checkpoint_load_before_to_device": 1,
        "after_model_to_device_before_optimizer_step": 2,
    }.get(stage, 10)
    if stage.startswith("forward_before_backward"):
        stage_rank = 3
    elif stage == "after_microbatch_backward":
        stage_rank = 4
    elif stage == "after_optimizer_step":
        stage_rank = 5

    def as_int(value: Any) -> int:
        try:
            return int(value)
        except Exception:
            return -1

    field_rank = {
        "cache_combined_content_sha256": 0,
        "init_model_state_sha256": 1,
        "batch_order_sha256": 2,
        "model_params_sha256": 3,
        "inputs_sha256": 4,
        "targets_sha256": 5,
        "module_output_sha256": 6,
        "logits_sha256": 7,
        "preds_sha256": 8,
        "grad_sha256": 9,
        "optimizer_state_sha256": 10,
    }.get(field, 99)
    module_rank = {
        "backbone.embeddings": 0,
        "backbone.layers.0.dropout1": 1,
        "backbone.layers.0.drop_path1": 2,
        "backbone.layers.0.norm1": 3,
        "backbone.layers.0.sequence_mixer.mixer": 4,
        "backbone.layers.0.sequence_mixer": 5,
        "backbone.layers.0.dropout2": 6,
        "backbone.layers.0.drop_path2": 7,
        "backbone.layers.0.norm2": 8,
        "backbone.layers.0.state_mixer": 9,
        "backbone.layers.1.dropout1": 10,
        "backbone.layers.1.drop_path1": 11,
        "backbone.layers.1.norm1": 12,
        "backbone.layers.1.sequence_mixer.mixer": 13,
        "backbone.layers.1.sequence_mixer": 14,
        "backbone.layers.1.dropout2": 15,
        "backbone.layers.1.drop_path2": 16,
        "backbone.layers.1.norm2": 17,
        "backbone.layers.1.state_mixer": 18,
        "backbone.ln_f": 19,
    }.get(module, 999)
    return (
        str(row.get("target", "")),
        stage_rank,
        as_int(row.get("optimizer_step")),
        as_int(row.get("micro_step")),
        field_rank,
        module_rank,
        field,
        module,
    )


def _collect_hash_probes(outputs_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    paths = sorted((outputs_dir / "hash-probes").glob("*/*/hash_probe.json"))
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = _read_json(path)
        rows.extend(_iter_hash_probe_rows(payload, str(path)))
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
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
        values = {str(member["machine"]): member["sha256"] for member in members}
        unique = sorted(set(values.values()))
        summary_rows.append(
            {
                "target": key[0],
                "stage": key[1],
                "optimizer_step": key[2],
                "micro_step": key[3],
                "field": key[4],
                "module": key[5],
                "machine_count": len(members),
                "unique_sha256_count": len(unique),
                "all_match": len(unique) <= 1,
                "values_json": json.dumps(values, sort_keys=True),
            }
        )
    summary_rows = sorted(summary_rows, key=_hash_probe_sort_key)
    first_rows: list[dict[str, Any]] = []
    for target in sorted({str(row.get("target", "")) for row in summary_rows}):
        mismatch = [row for row in summary_rows if str(row.get("target")) == target and str(row.get("all_match")) == "False"]
        first = mismatch[0] if mismatch else {}
        first_rows.append(
            {
                "target": target,
                "comparison_rows": len([row for row in summary_rows if str(row.get("target")) == target]),
                "mismatch_rows": len(mismatch),
                "first_mismatch_stage": first.get("stage", ""),
                "first_mismatch_optimizer_step": first.get("optimizer_step", ""),
                "first_mismatch_micro_step": first.get("micro_step", ""),
                "first_mismatch_field": first.get("field", ""),
                "first_mismatch_module": first.get("module", ""),
            }
        )
    return summary_rows, first_rows


def run_collect(args: argparse.Namespace) -> int:
    code = BASE.run_collect(args)
    artifact_dir = args.artifact_dir
    outputs_dir = args.outputs_dir
    early_rows = _collect_early_window(outputs_dir)
    trace_rows, cross_rows, cross_summary = _collect_read_trace(outputs_dir)
    hash_rows, first_rows = _collect_hash_probes(outputs_dir)
    _write_csv(artifact_dir / "early-window-summary.csv", early_rows)
    _write_csv(artifact_dir / "read-trace-summary.csv", trace_rows)
    _write_csv(artifact_dir / "read-trace-cross-machine.csv", cross_rows)
    _write_csv(artifact_dir / "read-trace-cross-machine-summary.csv", cross_summary)
    _write_csv(artifact_dir / "hash-probe-comparison-summary.csv", hash_rows)
    _write_csv(artifact_dir / "first-mismatch-summary.csv", first_rows)
    readme = (
        f"# {EXPERIMENT_ID}\n\n"
        "本 artifact 收尾 default-dropout amplifier trace diagnostic. "
        "本轮只定位放大链路, 不测试稳定化方案, 不写 official MQAR ledger.\n\n"
        "共同配置: `seed=124`, `data_seed=123`, `cb64-r16`, `write_topk=4`, "
        "canonical MQAR cache, seed124 canonical init, `resid_dropout=0`, `drop_path=0`.\n\n"
        "核心文件:\n\n"
        "- `run-summary.csv`: per-run final/best metrics.\n"
        "- `variant-summary.csv`: per-variant cross-machine summary.\n"
        "- `early-window-summary.csv`: train-step eval read/write scalar metrics.\n"
        "- `read-trace-summary.csv`: fixed sample read trace aggregate.\n"
        "- `read-trace-cross-machine-summary.csv`: 2080ti/3090 trace support match summary.\n"
        "- `hash-probe-comparison-summary.csv`: train-mode forward/backward hash comparison.\n"
        "- `first-mismatch-summary.csv`: first cross-machine mismatch by target.\n"
        "- `cache-init-preflight-summary.csv`: cache/init hash evidence.\n"
        "- `queue-summary.csv`: queue status.\n"
        "- `source-manifest.csv`: mirrored lightweight raw evidence.\n"
    )
    (artifact_dir / "README.md").write_text(readme, encoding="utf-8")
    metadata_path = artifact_dir / "metadata.json"
    metadata = _read_json(metadata_path) if metadata_path.exists() else {}
    metadata.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "early_window_rows": len(early_rows),
            "read_trace_summary_rows": len(trace_rows),
            "read_trace_cross_machine_rows": len(cross_rows),
            "read_trace_cross_machine_summary_rows": len(cross_summary),
            "hash_probe_comparison_rows": len(hash_rows),
            "first_mismatch_rows": len(first_rows),
        }
    )
    _save_json(metadata_path, metadata)
    return code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("cache-hash")
    p.add_argument("--machine-name", required=True)
    p.add_argument("--target", choices=TARGETS, default=TARGETS[0])
    p.add_argument("--variant", choices=TARGETS, default=TARGETS[0])
    p.add_argument("--output-json", type=Path)
    p.set_defaults(func=BASE.run_cache_hash)

    p = sub.add_parser("verify-init")
    p.add_argument("--machine-name", required=True)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_INIT_CHECKPOINT)
    p.add_argument("--output-json", type=Path)
    p.set_defaults(func=BASE.run_verify_init)

    p = sub.add_parser("preflight")
    p.add_argument("--machine-name", required=True)
    p.add_argument("--target", choices=TARGETS, default=TARGETS[0])
    p.add_argument("--variant", choices=TARGETS, default=TARGETS[0])
    p.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    p.add_argument("--max-train-steps", type=int, default=DEFAULT_MAX_TRAIN_STEPS)
    p.add_argument("--max-validation-batches", type=int)
    p.add_argument("--run-suffix")
    p.add_argument("--output-json", type=Path)
    p.set_defaults(func=run_preflight)

    p = sub.add_parser("config-summary")
    p.add_argument("--machine-name", required=True)
    p.add_argument("--target", choices=TARGETS, default=TARGETS[0])
    p.add_argument("--variant", choices=TARGETS, default=TARGETS[0])
    p.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    p.add_argument("--max-train-steps", type=int, default=DEFAULT_MAX_TRAIN_STEPS)
    p.add_argument("--max-validation-batches", type=int)
    p.add_argument("--run-suffix")
    p.add_argument("--output-json", type=Path)
    p.set_defaults(func=run_config_summary)

    p = sub.add_parser("hash-probe")
    p.add_argument("--machine-name", required=True)
    p.add_argument("--target", choices=TARGETS, required=True)
    p.add_argument("--init-checkpoint", type=Path, default=DEFAULT_INIT_CHECKPOINT)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--device")
    p.add_argument("--max-optimizer-steps", type=int, default=DEFAULT_MAX_TRAIN_STEPS)
    p.add_argument("--hash-micro-steps", type=int, default=4)
    p.add_argument("--capture-optimizer-steps", default=DEFAULT_CAPTURE_STEPS)
    p.add_argument("--capture-sample-count", type=int, default=4)
    p.add_argument("--margin-sample-positions", type=int, default=16)
    p.add_argument("--pre-seed", type=int, default=99991)
    p.set_defaults(func=run_hash_probe)

    p = sub.add_parser("train")
    p.add_argument("--machine-name", required=True)
    p.add_argument("--target", choices=TARGETS, required=True)
    p.add_argument("--variant", choices=TARGETS, required=True)
    p.add_argument("--init-checkpoint", type=Path, required=True)
    p.add_argument("--trace-output-dir", type=Path, required=True)
    p.add_argument("--output-config-json", type=Path)
    p.add_argument("--output-result-json", type=Path)
    p.add_argument("--logger-backend", choices=["none", "swanlab", "wandb"], default="none")
    p.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    p.add_argument("--max-train-steps", type=int, default=DEFAULT_MAX_TRAIN_STEPS)
    p.add_argument("--max-validation-batches", type=int)
    p.add_argument("--run-suffix")
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
    BASE.EXPECTED_INIT_STATE_SHA256 = _read_expected_init_hash()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
