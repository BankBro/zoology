from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import socket
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

_THIS_FILE = Path(__file__).resolve()
for _parent in _THIS_FILE.parents:
    if (_parent / "zoology").is_dir() and (_parent / "docs").is_dir():
        sys.path.insert(0, str(_parent))
        break

from zoology.checkpoints import load_checkpoint_payload
from zoology.data.utils import prepare_data
from zoology.model import LanguageModel
from zoology.train import Trainer
from zoology.config import (
    CheckpointConfig,
    DataConfig,
    DataSegmentConfig,
    LoggerConfig,
    ModelConfig,
    TrainConfig,
)


EXPERIMENT_ID = "20260629-04-flash-vqg-eval-read-topk-sweep"
DEFAULT_TOPKS = [1, 2, 4, 8, 16, 32, 64]


class NullLogger:
    def log_config(self, config: TrainConfig):
        return None

    def log_model(self, model, config: TrainConfig):
        return None

    def log(self, metrics: dict, *, step: int | None = None):
        return None

    def finish(self):
        return None

    def get_summary(self) -> dict[str, Any]:
        return {"backend": "none", "enabled": False}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _data_segment_registry() -> dict[str, type[DataSegmentConfig]]:
    from zoology.data.circuits import (
        CumulativeMajorityConfig,
        CumulativeParityConfig,
        MajorityConfig,
        ParityConfig,
        VocabMajorityConfig,
    )
    from zoology.data.compositional_mqar import CompositionalMQARConfig
    from zoology.data.forgetting_mqar import ForgettingMQARConfig
    from zoology.data.multiquery_ar import MQARConfig
    from zoology.data.stacked_mqar import ContinuousMQARConfig

    return {
        "multiquery_ar": MQARConfig,
        "continuous_mqar": ContinuousMQARConfig,
        "compositional_mqar": CompositionalMQARConfig,
        "forgetting_mqar": ForgettingMQARConfig,
        "parity": ParityConfig,
        "majority": MajorityConfig,
        "vocab_majority": VocabMajorityConfig,
        "cumulative_parity": CumulativeParityConfig,
        "cumulative_majority": CumulativeMajorityConfig,
    }


def _load_data_segment_config(payload: dict[str, Any] | DataSegmentConfig) -> DataSegmentConfig:
    if isinstance(payload, DataSegmentConfig):
        return payload
    if not isinstance(payload, dict):
        raise TypeError(f"data segment payload must be dict, got {type(payload).__name__}")
    segment_name = payload.get("name")
    if segment_name is None:
        return DataSegmentConfig.model_validate(payload)
    registry = _data_segment_registry()
    if segment_name not in registry:
        raise ValueError(f"unknown data segment `{segment_name}`; known: {sorted(registry)}")
    return registry[segment_name].model_validate(payload)


def load_train_config_from_json(config_path: Path) -> TrainConfig:
    payload = _load_json(config_path)
    data_payload = payload["data"]
    data_config = DataConfig(
        train_configs=[
            _load_data_segment_config(item) for item in data_payload["train_configs"]
        ],
        test_configs=[
            _load_data_segment_config(item) for item in data_payload["test_configs"]
        ],
        **{
            k: v
            for k, v in data_payload.items()
            if k not in {"train_configs", "test_configs"}
        },
    )
    return TrainConfig(
        data=data_config,
        model=ModelConfig.model_validate(payload["model"]),
        logger=LoggerConfig.model_validate(payload.get("logger", {})),
        checkpoint=CheckpointConfig.model_validate(payload.get("checkpoint", {})),
        **{
            k: v
            for k, v in payload.items()
            if k not in {"data", "model", "logger", "checkpoint"}
        },
    )


def _iter_flash_vqg_modules(model):
    for module in model.modules():
        attn = getattr(module, "attn", None)
        config = getattr(attn, "config", None)
        if config is not None and hasattr(config, "fox_remote_read_topk"):
            yield module, attn, config


def set_eval_read_topk(model, topk: int) -> int:
    updated = 0
    for module, _attn, config in _iter_flash_vqg_modules(model):
        module.fox_remote_read_topk = int(topk)
        module.fox_remote_read_topk_initial = None
        module.fox_remote_read_topk_final = None
        config.fox_remote_read_topk = int(topk)
        config.fox_remote_read_topk_initial = None
        config.fox_remote_read_topk_final = None
        updated += 1
    if updated <= 0:
        raise RuntimeError("No Flash-VQG modules with fox_remote_read_topk were found.")
    return updated


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _flatten_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {str(k): _json_safe(v) for k, v in metrics.items()}


def _read_existing_keys(records_path: Path) -> set[tuple[str, str, int, str]]:
    if not records_path.exists():
        return set()
    keys: set[tuple[str, str, int, str]] = set()
    with records_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") != "completed":
                continue
            keys.add(
                (
                    str(row.get("checkpoint_id")),
                    str(row.get("checkpoint_kind")),
                    int(row.get("eval_read_topk")),
                    str(row.get("eval_machine")),
                )
            )
    return keys


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _read_checkpoint_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _resolve_checkpoint_path(row: dict[str, str], kind: str) -> Path:
    key = f"{kind}_checkpoint"
    raw = row.get(key) or ""
    if not raw:
        raw = str(Path(row["run_dir"]) / f"{kind}.pt")
    return Path(raw).expanduser()


def _build_row_metadata(
    *,
    row: dict[str, str],
    checkpoint_path: Path,
    train_config_path: Path,
    checkpoint_kind: str,
    eval_machine: str,
    topk: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    payload = load_checkpoint_payload(checkpoint_path, map_location="cpu")
    saved_metrics = payload.get("metrics") if isinstance(payload, dict) else {}
    if not isinstance(saved_metrics, dict):
        saved_metrics = {}
    return {
        "experiment_id": EXPERIMENT_ID,
        "checkpoint_id": row["checkpoint_id"],
        "checkpoint_source_machine": row["source_machine"],
        "checkpoint_source_host": row.get("source_host"),
        "checkpoint_kind": checkpoint_kind,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "train_config_path": str(train_config_path.resolve()),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "checkpoint_epoch": payload.get("epoch"),
        "checkpoint_saved_valid_accuracy": saved_metrics.get("valid/accuracy"),
        "checkpoint_saved_1024x256": saved_metrics.get(
            "valid/mqar_case/accuracy-1024x256"
        ),
        "eval_machine": eval_machine,
        "eval_host": socket.gethostname(),
        "eval_read_topk": int(topk),
        "device_arg": args.device,
    }


def _evaluate_one(
    *,
    row: dict[str, str],
    checkpoint_kind: str,
    topk: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint_path = _resolve_checkpoint_path(row, checkpoint_kind)
    train_config_path = Path(row.get("train_config", "") or checkpoint_path.parent / "train_config.json")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    if not train_config_path.exists():
        raise FileNotFoundError(f"train_config.json not found: {train_config_path}")

    metadata = _build_row_metadata(
        row=row,
        checkpoint_path=checkpoint_path,
        train_config_path=train_config_path,
        checkpoint_kind=checkpoint_kind,
        eval_machine=args.eval_machine,
        topk=topk,
        args=args,
    )
    config = load_train_config_from_json(train_config_path)
    config.logger = LoggerConfig(backend="none")
    config.checkpoint.enabled = False
    if args.max_validation_batches is not None:
        config.max_validation_batches = int(args.max_validation_batches)

    train_dataloader, test_dataloader = prepare_data(config.data)
    model = LanguageModel(config.model)
    payload = load_checkpoint_payload(checkpoint_path, map_location="cpu")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()
    updated_modules = set_eval_read_topk(model, int(topk))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = _utc_now_iso()
    start = time.perf_counter()
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        input_type=config.input_type,
        max_epochs=1,
        max_train_steps=0,
        max_validation_batches=config.max_validation_batches,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        validations_per_epoch=1,
        early_stopping_metric=None,
        early_stopping_threshold=None,
        slice_keys=config.slice_keys,
        loss_type=config.loss_type,
        read_churn_probe_enabled=False,
        read_trace_enabled=False,
        run_id=f"{row['checkpoint_id']}-{checkpoint_kind}-eval-topk{topk}",
        device=device,
        logger=NullLogger(),
        checkpoint_manager=None,
    )
    trainer.loss_fn = torch.nn.CrossEntropyLoss()
    metrics = trainer.test(epoch_idx=0)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory_bytes = int(torch.cuda.max_memory_allocated(device))
    else:
        peak_memory_bytes = None
    finished = _utc_now_iso()
    elapsed = time.perf_counter() - start

    flat_metrics = _flatten_metrics(metrics)
    effective_topk = flat_metrics.get("valid/attn/gd_residual_remote_read_topk_effective")
    if effective_topk is not None and abs(float(effective_topk) - float(topk)) > 1e-6:
        raise RuntimeError(
            "effective read topk mismatch: "
            f"expected {topk}, got {effective_topk}"
        )

    return {
        **metadata,
        "status": "completed",
        "started_at": started,
        "finished_at": finished,
        "duration_seconds": elapsed,
        "updated_flash_vqg_modules": updated_modules,
        "peak_memory_bytes": peak_memory_bytes,
        "max_validation_batches": config.max_validation_batches,
        "metrics": flat_metrics,
        "valid_loss": flat_metrics.get("valid/loss"),
        "valid_accuracy": flat_metrics.get("valid/accuracy"),
        "valid_mqar_case_accuracy_1024x256": flat_metrics.get(
            "valid/mqar_case/accuracy-1024x256"
        ),
        "valid_input_seq_len_accuracy_1024": flat_metrics.get(
            "valid/input_seq_len/accuracy-1024"
        ),
        "valid_num_kv_pairs_accuracy_256": flat_metrics.get(
            "valid/num_kv_pairs/accuracy-256"
        ),
        "valid_effective_read_topk": effective_topk,
        "valid_read_selected_mass_mean": flat_metrics.get(
            "valid/attn/gd_residual_read_selected_mass_mean"
        ),
        "valid_read_selected_mass_p05": flat_metrics.get(
            "valid/attn/gd_residual_read_selected_mass_p05"
        ),
    }


def run(args: argparse.Namespace) -> int:
    records_path = Path(args.output_dir) / "eval-records.jsonl"
    status_path = Path(args.output_dir) / "status.json"
    checkpoint_rows = _read_checkpoint_rows(Path(args.checkpoint_manifest))
    topks = [int(item) for item in str(args.topks).split(",") if item.strip()]
    kinds = [item.strip() for item in str(args.checkpoint_kinds).split(",") if item.strip()]
    if any(kind not in {"best", "last"} for kind in kinds):
        raise ValueError("--checkpoint-kinds must contain only best,last")

    existing_keys = _read_existing_keys(records_path) if args.resume else set()
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device was requested but torch.cuda.is_available() is false.")

    selected_rows = [
        row for row in checkpoint_rows
        if not args.source_machine or row.get("source_machine") == args.source_machine
    ]
    total = len(selected_rows) * len(kinds) * len(topks)
    completed = 0
    failed = 0
    skipped = 0
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    status = {
        "experiment_id": EXPERIMENT_ID,
        "eval_machine": args.eval_machine,
        "started_at": _utc_now_iso(),
        "checkpoint_manifest": str(Path(args.checkpoint_manifest).resolve()),
        "topks": topks,
        "checkpoint_kinds": kinds,
        "total": total,
        "completed": completed,
        "failed": failed,
        "skipped": skipped,
    }
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

    for row in selected_rows:
        for checkpoint_kind in kinds:
            for topk in topks:
                key = (row["checkpoint_id"], checkpoint_kind, int(topk), args.eval_machine)
                if key in existing_keys:
                    skipped += 1
                    continue
                try:
                    result = _evaluate_one(
                        row=row,
                        checkpoint_kind=checkpoint_kind,
                        topk=int(topk),
                        args=args,
                        device=device,
                    )
                    completed += 1
                    _append_jsonl(records_path, result)
                except BaseException as exc:
                    failed += 1
                    failure = {
                        "experiment_id": EXPERIMENT_ID,
                        "checkpoint_id": row.get("checkpoint_id"),
                        "checkpoint_source_machine": row.get("source_machine"),
                        "checkpoint_kind": checkpoint_kind,
                        "eval_machine": args.eval_machine,
                        "eval_read_topk": int(topk),
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(limit=20),
                        "finished_at": _utc_now_iso(),
                    }
                    _append_jsonl(records_path, failure)
                    if not args.continue_on_error:
                        raise
                status.update(
                    {
                        "completed": completed,
                        "failed": failed,
                        "skipped": skipped,
                        "updated_at": _utc_now_iso(),
                    }
                )
                status_path.write_text(
                    json.dumps(status, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

    status.update(
        {
            "finished_at": _utc_now_iso(),
            "completed": completed,
            "failed": failed,
            "skipped": skipped,
        }
    )
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if failed == 0 else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate dense-read checkpoints under fixed eval read topk.")
    parser.add_argument("--checkpoint-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eval-machine", required=True)
    parser.add_argument("--source-machine", default="")
    parser.add_argument("--topks", default=",".join(str(k) for k in DEFAULT_TOPKS))
    parser.add_argument("--checkpoint-kinds", default="best,last")
    parser.add_argument("--device", default=os.environ.get("DEVICE", "auto"))
    parser.add_argument("--max-validation-batches", type=int, default=None)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--continue-on-error", action="store_true", default=False)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
