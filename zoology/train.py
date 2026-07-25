import argparse
import hashlib
import random
import json
import signal
import time
from collections import defaultdict
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, Union
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from einops import rearrange

from zoology.data.utils import prepare_data, prepare_continuous_data
from zoology.config import CheckpointConfig, TrainConfig
from zoology.checkpoints import (
    load_checkpoint_payload,
    resolve_checkpoint_path,
    serialize_train_config,
)
from zoology.experiments.flash_vqg.manifest import update_manifest_for_run
from zoology.model import LanguageModel, ContinuousInputModel
from zoology.logger import LoggerProtocol, build_logger
from zoology.utils import set_determinism
from zoology.metrics import compute_mse, compute_ce_with_embeddings


class TrainingInterrupted(RuntimeError):
    pass


def _format_training_error(exc: BaseException) -> str:
    message = str(exc).strip()
    if message:
        return f"{type(exc).__name__}: {message}"
    return type(exc).__name__


def _build_signal_interrupt_handler(signal_name: str):
    def _handler(_signum, _frame):
        raise TrainingInterrupted(f"收到 {signal_name} 信号, 终止训练.")

    return _handler


def _install_training_signal_handlers() -> dict[int, signal.Handlers]:
    previous_handlers: dict[int, signal.Handlers] = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(
            signum,
            _build_signal_interrupt_handler(signal.Signals(signum).name),
        )
    return previous_handlers


def _restore_training_signal_handlers(previous_handlers: dict[int, signal.Handlers]) -> None:
    for signum, handler in previous_handlers.items():
        signal.signal(signum, handler)


def _config_identity(config: TrainConfig) -> dict[str, str]:
    serialized = serialize_train_config(config)
    encoded = json.dumps(serialized, sort_keys=True, default=str).encode("utf-8")
    return {
        "train_config_sha256": hashlib.sha256(encoded).hexdigest(),
        **{str(key): str(value) for key, value in config.resume_identity.items()},
    }


def _capture_rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _collect_model_runtime_state(model: nn.Module) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, module in model.named_modules():
        getter = getattr(module, "get_training_runtime_state", None)
        if getter is not None:
            result[name] = getter()
    return result


def _restore_model_runtime_state(
    model: nn.Module,
    state: dict[str, dict[str, Any]],
) -> None:
    modules = dict(model.named_modules())
    missing = sorted(set(state) - set(modules))
    if missing:
        raise KeyError(f"Resume runtime modules are missing: {missing}")
    for name, value in state.items():
        loader = getattr(modules[name], "load_training_runtime_state", None)
        if loader is None:
            raise TypeError(f"Module no longer supports training runtime state: {name}")
        loader(value)


class CheckpointManager:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.checkpoint_config: CheckpointConfig = config.checkpoint
        self.enabled = self.checkpoint_config.enabled
        self.best_value = None

        launch_dir = config.launch_id if config.launch_id is not None else "manual"
        self.run_dir = Path(self.checkpoint_config.root_dir) / launch_dir / config.run_id
        self.best_path = self.run_dir / "best.pt"
        self.last_path = self.run_dir / "last.pt"
        self.resume_path = (
            Path(config.resume_path)
            if config.resume_path is not None
            else self.run_dir / "resume.pt"
        )
        self.config_path = self.run_dir / "train_config.json"
        self.best_metric = self._resolve_best_metric()
        self.best_mode = self.checkpoint_config.best_mode

    def _resolve_best_metric(self):
        if self.checkpoint_config.best_metric is not None:
            return self.checkpoint_config.best_metric
        if self.config.early_stopping_metric is not None:
            return self.config.early_stopping_metric
        return "valid/accuracy"

    def setup(self):
        if not self.enabled:
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if self.checkpoint_config.save_config_json:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(serialize_train_config(self.config), f, ensure_ascii=False, indent=2)

    def _serialize_model(self, model: nn.Module):
        return {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        }

    def _build_payload(self, model: nn.Module, epoch_idx: int, metrics: dict):
        return {
            "model_state_dict": self._serialize_model(model),
            "epoch": epoch_idx,
            "metrics": metrics,
            "run_id": self.config.run_id,
            "launch_id": self.config.launch_id,
            "sweep_id": self.config.sweep_id,
            "model_name": self.config.model.name,
        }

    def _atomic_save(self, payload: dict, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp_path)
        tmp_path.replace(path)

    def save_resume(self, payload: dict[str, Any]) -> None:
        if not self.config.resume_enabled:
            return
        if not self.enabled:
            raise RuntimeError("resume_enabled requires checkpoint.enabled=true.")
        self._atomic_save(payload, self.resume_path)

    def load_resume(self) -> dict[str, Any] | None:
        if not self.config.resume_enabled or not self.resume_path.exists():
            return None
        payload = torch.load(self.resume_path, map_location="cpu", weights_only=False)
        if int(payload.get("format_version", -1)) != 1:
            raise RuntimeError("Unsupported resume checkpoint format version.")
        expected = _config_identity(self.config)
        if payload.get("identity") != expected:
            raise RuntimeError(
                f"Resume identity mismatch: expected={expected}, "
                f"actual={payload.get('identity')}"
            )
        return payload

    def _is_better(self, current_value):
        if self.best_value is None:
            return True
        if self.best_mode == "min":
            return current_value < self.best_value
        return current_value > self.best_value

    def save_epoch(self, model: nn.Module, epoch_idx: int, metrics: dict):
        if not self.enabled:
            return

        payload = self._build_payload(model=model, epoch_idx=epoch_idx, metrics=metrics)

        if self.checkpoint_config.save_last:
            self._atomic_save(payload, self.last_path)

        if not self.checkpoint_config.save_best:
            return

        if self.best_metric not in metrics:
            raise KeyError(
                f"Best checkpoint metric `{self.best_metric}` was not found in validation metrics: "
                f"{sorted(metrics.keys())}"
            )

        current_value = metrics[self.best_metric]
        if self._is_better(current_value):
            self.best_value = current_value
            self._atomic_save(payload, self.best_path)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        input_type: str = "discrete",
        max_epochs: int = 100,
        max_train_steps: int | None = None,
        max_validation_batches: int | None = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.1,
        gradient_accumulation_steps: int = 1,
        validations_per_epoch: int = 1,
        precision: str = "float32",
        training_runtime_initial_state: dict[str, dict[str, Any]] | None = None,
        resume_stop_after_optimizer_step: int | None = None,
        max_grad_scaler_skips: int = 0,
        max_consecutive_grad_scaler_skips: int = 0,
        training_telemetry_path: str | None = None,
        early_stopping_metric: str = None,
        early_stopping_threshold: float = None,
        loss_type: str = "ce",
        slice_keys: List[str] = [],
        read_churn_probe_enabled: bool = False,
        read_churn_probe_valid_batches: List[int] = [0],
        read_churn_probe_max_samples: int = 16,
        read_churn_probe_query_only: bool = True,
        read_trace_enabled: bool = False,
        read_trace_valid_batches: List[int] = [0],
        read_trace_max_samples: int = 4,
        read_trace_query_only: bool = True,
        read_trace_max_queries_per_sample: int = 8,
        read_trace_output_dir: str | None = None,
        read_trace_train_steps: List[int] | None = None,
        train_inline_event_trace_enabled: bool = False,
        train_inline_event_trace_steps: List[int] | None = None,
        train_inline_event_trace_output_dir: str | None = None,
        run_id: str | None = None,
        device: Union[str, int] = "cuda",
        logger: LoggerProtocol = None,
        checkpoint_manager: CheckpointManager = None,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.input_type = input_type
        self.logger = logger
        self.checkpoint_manager = checkpoint_manager

        self.device = device
        self.max_epochs = max_epochs
        self.max_train_steps = None if max_train_steps is None else int(max_train_steps)
        if self.max_train_steps is not None and self.max_train_steps < 0:
            raise ValueError("max_train_steps must be non-negative or None.")
        self.max_validation_batches = (
            None if max_validation_batches is None else int(max_validation_batches)
        )
        if self.max_validation_batches is not None and self.max_validation_batches <= 0:
            raise ValueError("max_validation_batches must be positive or None.")
        self._max_train_steps_reached = False
        self.early_stopping_metric = early_stopping_metric
        self.early_stopping_threshold = early_stopping_threshold
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.validations_per_epoch = int(validations_per_epoch)
        if self.validations_per_epoch <= 0:
            raise ValueError("validations_per_epoch must be a positive integer.")
        self.precision = str(precision)
        if self.precision not in {"float32", "amp_float16", "amp_bfloat16"}:
            raise ValueError(f"Unsupported training precision: {self.precision}")
        self.training_runtime_initial_state = dict(
            training_runtime_initial_state or {}
        )
        self.resume_stop_after_optimizer_step = (
            None
            if resume_stop_after_optimizer_step is None
            else int(resume_stop_after_optimizer_step)
        )
        self.max_grad_scaler_skips = int(max_grad_scaler_skips)
        self.max_consecutive_grad_scaler_skips = int(
            max_consecutive_grad_scaler_skips
        )
        if self.max_grad_scaler_skips < 0 or self.max_consecutive_grad_scaler_skips < 0:
            raise ValueError("GradScaler skip limits must be non-negative.")
        self.training_telemetry_path = (
            Path(training_telemetry_path)
            if training_telemetry_path is not None
            else None
        )
        self.slice_keys = slice_keys
        self.loss_type = loss_type
        self.global_step = 0
        self.optimizer_step = 0
        self.optimizer_attempt_step = 0
        self.grad_scaler_skips = 0
        self.consecutive_grad_scaler_skips = 0
        self._completed_validations: set[tuple[int, int]] = set()
        self._resume_epoch_idx = 0
        self._resume_next_train_batch_idx = 0
        self.read_churn_probe_enabled = bool(read_churn_probe_enabled)
        self.read_churn_probe_valid_batches = {
            int(idx) for idx in (read_churn_probe_valid_batches or [])
        }
        self.read_churn_probe_max_samples = int(read_churn_probe_max_samples)
        self.read_churn_probe_query_only = bool(read_churn_probe_query_only)
        self._read_churn_probe_prev_top_idx_by_key: dict = {}
        self.read_trace_enabled = bool(read_trace_enabled)
        self.read_trace_valid_batches = {
            int(idx) for idx in (read_trace_valid_batches or [])
        }
        self.read_trace_max_samples = int(read_trace_max_samples)
        self.read_trace_query_only = bool(read_trace_query_only)
        self.read_trace_max_queries_per_sample = int(read_trace_max_queries_per_sample)
        self.read_trace_output_dir = (
            Path(read_trace_output_dir)
            if read_trace_output_dir is not None and str(read_trace_output_dir).strip()
            else None
        )
        self.read_trace_train_steps = {
            int(step) for step in (read_trace_train_steps or [])
        }
        if any(step < 0 for step in self.read_trace_train_steps):
            raise ValueError("read_trace_train_steps must contain non-negative integers.")
        self._completed_read_trace_train_steps: set[int] = set()
        self.early_window_metrics_path = (
            self.read_trace_output_dir / "early_window_metrics.jsonl"
            if self.read_trace_output_dir is not None and self.read_trace_train_steps
            else None
        )
        self.train_inline_event_trace_enabled = bool(train_inline_event_trace_enabled)
        self.train_inline_event_trace_steps = {
            int(step) for step in (train_inline_event_trace_steps or [])
        }
        if any(step < 0 for step in self.train_inline_event_trace_steps):
            raise ValueError("train_inline_event_trace_steps must contain non-negative integers.")
        self.train_inline_event_trace_output_dir = (
            Path(train_inline_event_trace_output_dir)
            if train_inline_event_trace_output_dir is not None
            and str(train_inline_event_trace_output_dir).strip()
            else None
        )
        self.run_id = run_id

    def _autocast_context(self):
        if self.precision == "float32":
            return nullcontext()
        device_type = torch.device(self.device).type
        dtype = (
            torch.float16
            if self.precision == "amp_float16"
            else torch.bfloat16
        )
        return torch.autocast(device_type=device_type, dtype=dtype)

    def _step_optimizer(self) -> bool:
        self.optimizer_attempt_step += 1
        if not self.scaler.is_enabled():
            self.optimizer.step()
            self.consecutive_grad_scaler_skips = 0
            return True

        scale_before = float(self.scaler.get_scale())
        self.scaler.step(self.optimizer)
        self.scaler.update()
        skipped = float(self.scaler.get_scale()) < scale_before
        if not skipped:
            self.consecutive_grad_scaler_skips = 0
            return True

        self.grad_scaler_skips += 1
        self.consecutive_grad_scaler_skips += 1
        if self.grad_scaler_skips > self.max_grad_scaler_skips:
            raise FloatingPointError("GradScaler total skip limit exceeded.")
        if (
            self.consecutive_grad_scaler_skips
            > self.max_consecutive_grad_scaler_skips
        ):
            raise FloatingPointError("GradScaler consecutive skip limit exceeded.")
        return False

    def _build_resume_payload(
        self,
        *,
        epoch_idx: int,
        next_train_batch_idx: int,
    ) -> dict[str, Any]:
        best_value = None
        if self.checkpoint_manager is not None:
            best_value = self.checkpoint_manager.best_value
        return {
            "format_version": 1,
            "identity": _config_identity(self.checkpoint_manager.config),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "grad_scaler_state_dict": self.scaler.state_dict(),
            "rng_state": _capture_rng_state(),
            "model_runtime_state": _collect_model_runtime_state(self.model),
            "epoch_idx": int(epoch_idx),
            "next_train_batch_idx": int(next_train_batch_idx),
            "optimizer_step": int(self.optimizer_step),
            "optimizer_attempt_step": int(self.optimizer_attempt_step),
            "global_step": int(self.global_step),
            "grad_scaler_skips": int(self.grad_scaler_skips),
            "consecutive_grad_scaler_skips": int(
                self.consecutive_grad_scaler_skips
            ),
            "completed_validations": sorted(
                [int(epoch), int(step)]
                for epoch, step in self._completed_validations
            ),
            "checkpoint_best_value": best_value,
        }

    def _save_resume(
        self,
        *,
        epoch_idx: int,
        next_train_batch_idx: int,
        allow_controlled_stop: bool,
    ) -> None:
        if (
            self.checkpoint_manager is None
            or not self.checkpoint_manager.config.resume_enabled
        ):
            return
        payload = self._build_resume_payload(
            epoch_idx=epoch_idx,
            next_train_batch_idx=next_train_batch_idx,
        )
        self.checkpoint_manager.save_resume(payload)
        if (
            allow_controlled_stop
            and self.resume_stop_after_optimizer_step is not None
            and self.optimizer_step == self.resume_stop_after_optimizer_step
        ):
            raise TrainingInterrupted(
                "Controlled stop after saving the requested optimizer step."
            )

    def _load_resume(self) -> None:
        if self.checkpoint_manager is None:
            return
        payload = self.checkpoint_manager.load_resume()
        if payload is None:
            return
        self.model.load_state_dict(payload["model_state_dict"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.scheduler.load_state_dict(payload["scheduler_state_dict"])
        self.scaler.load_state_dict(payload["grad_scaler_state_dict"])
        _restore_model_runtime_state(self.model, payload["model_runtime_state"])
        self._resume_epoch_idx = int(payload["epoch_idx"])
        self._resume_next_train_batch_idx = int(payload["next_train_batch_idx"])
        self.optimizer_step = int(payload["optimizer_step"])
        self.optimizer_attempt_step = int(payload["optimizer_attempt_step"])
        self.global_step = int(payload["global_step"])
        self.grad_scaler_skips = int(payload["grad_scaler_skips"])
        self.consecutive_grad_scaler_skips = int(
            payload["consecutive_grad_scaler_skips"]
        )
        self._completed_validations = {
            (int(epoch), int(step))
            for epoch, step in payload["completed_validations"]
        }
        self.checkpoint_manager.best_value = payload["checkpoint_best_value"]
        _restore_rng_state(payload["rng_state"])

    def _set_dense_teacher_runtime(self, targets: torch.Tensor) -> None:
        if self.input_type != "discrete":
            return
        runtime = {
            "teacher_target_mask": (targets != -100).detach(),
        }

        def setter(module):
            setter_fn = getattr(module, "set_dense_teacher_runtime", None)
            if setter_fn is not None:
                setter_fn(runtime)

        self.model.apply(setter)

    def _clear_dense_teacher_runtime(self) -> None:
        def clearer(module):
            clearer_fn = getattr(module, "clear_dense_teacher_runtime", None)
            if clearer_fn is not None:
                clearer_fn()

        self.model.apply(clearer)

    def _set_read_candidate_probe_runtime(
        self,
        *,
        batch_idx: int,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        epoch_idx: int,
        trace_output_dir: Path | None = None,
        global_step: int | None = None,
        force_trace_enabled: bool | None = None,
    ) -> bool:
        churn_enabled = self.read_churn_probe_enabled and batch_idx in self.read_churn_probe_valid_batches
        trace_enabled = (
            bool(force_trace_enabled)
            if force_trace_enabled is not None
            else self.read_trace_enabled and batch_idx in self.read_trace_valid_batches
        )
        if not churn_enabled and not trace_enabled:
            return False
        if self.input_type != "discrete":
            return False
        max_samples = min(
            max(self.read_churn_probe_max_samples if churn_enabled else 0, self.read_trace_max_samples if trace_enabled else 0),
            int(targets.size(0)),
        )
        if max_samples <= 0:
            return False
        churn_query_mask = (targets[:max_samples] != -100).detach()
        if not self.read_churn_probe_query_only:
            churn_query_mask = torch.ones_like(churn_query_mask, dtype=torch.bool)
        trace_query_mask = (targets[:max_samples] != -100).detach()
        if not self.read_trace_query_only:
            trace_query_mask = torch.ones_like(trace_query_mask, dtype=torch.bool)

        input_hashes: list[str] = []
        target_hashes: list[str] = []
        for sample_idx in range(max_samples):
            input_hashes.append(
                hashlib.sha1(inputs[sample_idx].detach().cpu().numpy().tobytes()).hexdigest()
            )
            target_hashes.append(
                hashlib.sha1(targets[sample_idx].detach().cpu().numpy().tobytes()).hexdigest()
            )

        runtime = {
            "enabled": True,
            "churn_enabled": bool(churn_enabled),
            "trace_enabled": bool(trace_enabled),
            "valid_batch_idx": int(batch_idx),
            "epoch_idx": int(epoch_idx),
            "global_step": int(self.global_step if global_step is None else global_step),
            "max_samples": int(max_samples),
            "query_mask": churn_query_mask,
            "trace_query_mask": trace_query_mask,
            "trace_max_samples": int(self.read_trace_max_samples),
            "trace_max_queries_per_sample": int(self.read_trace_max_queries_per_sample),
            "trace_output_dir": (
                str(trace_output_dir)
                if trace_output_dir is not None
                else str(self.read_trace_output_dir)
                if self.read_trace_output_dir is not None
                else None
            ),
            "run_id": getattr(self, "run_id", None),
            "input_hashes": input_hashes,
            "target_hashes": target_hashes,
            "_prev_top_idx_by_key": self._read_churn_probe_prev_top_idx_by_key,
        }

        def setter(module):
            setter_fn = getattr(module, "set_read_candidate_probe_runtime", None)
            if setter_fn is not None:
                setter_fn(runtime)

        self.model.apply(setter)
        return True

    def _clear_read_candidate_probe_runtime(self) -> None:
        def clearer(module):
            clearer_fn = getattr(module, "clear_read_candidate_probe_runtime", None)
            if clearer_fn is not None:
                clearer_fn()

        self.model.apply(clearer)

    def _set_train_inline_event_trace_runtime(
        self,
        *,
        epoch_idx: int,
        optimizer_step_idx: int,
        train_batch_idx: int,
        micro_step_idx: int,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> bool:
        if not self.train_inline_event_trace_enabled:
            return False
        if optimizer_step_idx not in self.train_inline_event_trace_steps:
            return False
        if self.train_inline_event_trace_output_dir is None:
            return False
        if self.input_type != "discrete":
            return False

        trace_output_dir = (
            self.train_inline_event_trace_output_dir
            / f"train_inline_step_{int(optimizer_step_idx)}"
            / f"micro_{int(micro_step_idx)}"
        )
        max_samples = int(targets.size(0))
        input_hashes: list[str] = []
        target_hashes: list[str] = []
        for sample_idx in range(max_samples):
            input_hashes.append(
                hashlib.sha1(inputs[sample_idx].detach().cpu().numpy().tobytes()).hexdigest()
            )
            target_hashes.append(
                hashlib.sha1(targets[sample_idx].detach().cpu().numpy().tobytes()).hexdigest()
            )

        query_mask = (targets != -100).detach()
        runtime = {
            "enabled": True,
            "churn_enabled": False,
            "trace_enabled": True,
            "trace_phase": "train_inline",
            "epoch_idx": int(epoch_idx),
            "global_step": int(optimizer_step_idx),
            "optimizer_step": int(optimizer_step_idx),
            "train_batch_idx": int(train_batch_idx),
            "micro_step": int(micro_step_idx),
            "valid_batch_idx": None,
            "max_samples": int(max_samples),
            "query_mask": query_mask,
            "trace_query_mask": query_mask,
            "trace_max_samples": int(max_samples),
            "trace_max_queries_per_sample": -1,
            "trace_output_dir": str(trace_output_dir),
            "run_id": getattr(self, "run_id", None),
            "input_hashes": input_hashes,
            "target_hashes": target_hashes,
            "_prev_top_idx_by_key": {},
        }

        def setter(module):
            setter_fn = getattr(module, "set_read_candidate_probe_runtime", None)
            if setter_fn is not None:
                setter_fn(runtime)

        self.model.apply(setter)
        return True

    def compute_loss(self, inputs, targets, *, return_predictions: bool = True):
        if self.input_type == "continuous":
            
            all_embeddings = self.model.backbone.embeddings.word_embeddings.weight
            vocab_size = all_embeddings.shape[0]
            embed_dim = all_embeddings.shape[1]
            value_embeddings = all_embeddings[vocab_size // 2:]  # all values as candidates
            
            outputs = self.model(inputs)
            num_kv_pairs = targets.shape[1]
            outputs = outputs[:, -num_kv_pairs:]
            
            outputs_flat = outputs.reshape(-1, embed_dim)
            targets_flat = targets.reshape(-1)
            
            if self.loss_type == "mse":
                target_embeds = value_embeddings[targets_flat]
                loss, _ = compute_mse(outputs_flat, target_embeds)
            else:  # ce or ce_embed
                loss, _ = compute_ce_with_embeddings(
                    outputs_flat, targets_flat, value_embeddings
                )
            
            preds = None
            if return_predictions:
                logits = outputs_flat @ value_embeddings.T
                preds = logits.argmax(dim=-1).view(targets.shape)
            return loss, preds
        
        else: # discrete
            if self.loss_type == "ce":
                logits = self.model(inputs)
                loss = self.loss_fn(
                    rearrange(logits, "... c -> (...) c"), 
                    targets.flatten()
                )
                preds = logits.argmax(dim=-1) if return_predictions else None
                return loss, preds
            
            elif self.loss_type == "mse":
                embeddings = self.model(inputs, return_embeddings=True)
                target_embeds = self.model.backbone.embeddings.word_embeddings(targets)
                mask = (targets != -100).unsqueeze(-1)
                loss, _ = compute_mse(
                    embeddings[mask.expand_as(embeddings)].view(-1, embeddings.size(-1)),
                    target_embeds[mask.expand_as(target_embeds)].view(-1, target_embeds.size(-1)),
                )
                preds = None
                if return_predictions:
                    logits = embeddings @ self.model.backbone.embeddings.word_embeddings.weight.T
                    preds = logits.argmax(dim=-1)
                return loss, preds
            
            elif self.loss_type == "ce_embed":
                embeddings = self.model(inputs, return_embeddings=True)
                value_embeddings = self.model.backbone.embeddings.word_embeddings.weight
                flat_embeds = rearrange(embeddings, "b s d -> (b s) d")
                flat_targets = targets.flatten()
                mask = flat_targets != -100
                loss, _ = compute_ce_with_embeddings(
                    flat_embeds[mask], flat_targets[mask], value_embeddings,
                )
                preds = None
                if return_predictions:
                    logits = embeddings @ value_embeddings.T
                    preds = logits.argmax(dim=-1)
                return loss, preds

    def _collect_model_scalar_metrics(self) -> dict[str, float]:
        scalar_metrics: dict[str, float] = {}

        def collect(module):
            getter = getattr(module, "get_scalar_metrics", None)
            if getter is None:
                return
            module_metrics = getter()
            if not module_metrics:
                return
            for key, value in module_metrics.items():
                scalar_metrics[str(key)] = float(value)

        self.model.apply(collect)
        return scalar_metrics

    @staticmethod
    def _prefix_phase_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
        prefixed: dict[str, float] = {}
        for key, value in metrics.items():
            key = str(key)
            prefixed_key = key if key.startswith(prefix) else f"{prefix}{key}"
            prefixed[prefixed_key] = float(value)
        return prefixed

    @staticmethod
    def _json_safe_metric(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return float(value.detach().cpu().item())
            return value.detach().cpu().tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (float, int, str, bool)) or value is None:
            return value
        return str(value)

    def _append_early_window_metrics(self, payload: dict[str, Any]) -> None:
        if self.early_window_metrics_path is None:
            return
        self.early_window_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        safe_payload = {
            str(key): self._json_safe_metric(value)
            for key, value in payload.items()
        }
        with self.early_window_metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(safe_payload, ensure_ascii=False, sort_keys=True) + "\n")

    def _log_metrics(self, metrics: dict[str, Any]):
        self.logger.log(metrics, step=self.global_step)
        if self.training_telemetry_path is not None:
            self.training_telemetry_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "log_step": int(self.global_step),
                "recorded_at_utc": datetime.utcnow().isoformat() + "Z",
                **{
                    str(key): self._json_safe_metric(value)
                    for key, value in metrics.items()
                },
            }
            with self.training_telemetry_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
                )
        self.global_step += 1

    def _maybe_run_train_step_read_trace(self, *, epoch_idx: int) -> None:
        if not self.read_trace_train_steps:
            return
        train_step = int(self.optimizer_step)
        if train_step not in self.read_trace_train_steps:
            return
        if train_step in self._completed_read_trace_train_steps:
            return
        if not self.read_trace_enabled:
            return
        if not self.read_trace_valid_batches:
            return
        self._completed_read_trace_train_steps.add(train_step)
        self._run_train_step_read_trace(epoch_idx=epoch_idx, train_step=train_step)

    def _run_train_step_read_trace(self, *, epoch_idx: int, train_step: int) -> None:
        if self.input_type != "discrete":
            return
        if self.read_trace_output_dir is None:
            return

        was_training = self.model.training
        self.model.eval()
        target_batches = set(self.read_trace_valid_batches)
        trace_output_dir = self.read_trace_output_dir / f"train_step_{int(train_step)}"
        scalar_metric_buckets: dict[str, list[float]] = defaultdict(list)
        processed_batches = 0
        total_loss = 0.0
        total_examples = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets, slices) in enumerate(self.test_dataloader):
                if batch_idx not in target_batches:
                    continue
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self._set_dense_teacher_runtime(targets)
                read_probe_active = self._set_read_candidate_probe_runtime(
                    batch_idx=batch_idx,
                    inputs=inputs,
                    targets=targets,
                    epoch_idx=epoch_idx,
                    trace_output_dir=trace_output_dir,
                    global_step=train_step,
                    force_trace_enabled=True,
                )
                try:
                    with self._autocast_context():
                        loss, preds = self.compute_loss(inputs, targets)
                finally:
                    self._clear_dense_teacher_runtime()
                    if read_probe_active:
                        self._clear_read_candidate_probe_runtime()

                processed_batches += 1
                batch_size = int(targets.size(0))
                total_examples += batch_size
                total_loss += float(loss.detach().cpu().item()) * batch_size
                for key, value in self._collect_model_scalar_metrics().items():
                    scalar_metric_buckets[key].append(float(value))

                if target_batches.issubset(set(range(batch_idx + 1))):
                    break

        aggregated_scalar_metrics = {
            key: float(np.mean(values))
            for key, values in scalar_metric_buckets.items()
            if values
        }
        payload: dict[str, Any] = {
            "run_id": self.run_id,
            "epoch": int(epoch_idx),
            "train_step": int(train_step),
            "valid_batches": sorted(int(idx) for idx in target_batches),
            "processed_batches": int(processed_batches),
            "examples": int(total_examples),
            "loss": (total_loss / total_examples) if total_examples else None,
            "trace_output_dir": str(trace_output_dir),
        }
        payload.update(self._prefix_phase_metrics(aggregated_scalar_metrics, "early_window/"))
        self._append_early_window_metrics(payload)

        if was_training:
            self.model.train()

    def _validation_boundaries(self, num_optimizer_steps: int) -> set[int]:
        if self.validations_per_epoch <= 1:
            return set()
        boundaries = {
            max(1, round(num_optimizer_steps * validation_idx / self.validations_per_epoch))
            for validation_idx in range(1, self.validations_per_epoch)
        }
        return {boundary for boundary in boundaries if boundary < num_optimizer_steps}

    def train_epoch(
        self,
        epoch_idx: int,
        start_batch_idx: int = 0,
        validation_callback: Callable[[int, int], None] | None = None,
    ):
        self.model.train()
        sampler = getattr(self.train_dataloader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch_idx)

        accum_steps = self.gradient_accumulation_steps
        num_batches = len(self.train_dataloader)
        remainder = num_batches % accum_steps
        num_optimizer_steps = (num_batches + accum_steps - 1) // accum_steps
        validation_boundaries = self._validation_boundaries(num_optimizer_steps)
        # Index where the last (possibly partial) accumulation window begins
        partial_start = num_batches - remainder if remainder > 0 else num_batches
        start_batch_idx = int(start_batch_idx)
        if start_batch_idx < 0 or start_batch_idx > num_batches:
            raise ValueError(f"Invalid resume batch cursor: {start_batch_idx}")
        if start_batch_idx < num_batches and start_batch_idx % accum_steps != 0:
            raise ValueError("Resume batch cursor must be an optimizer boundary.")

        iterator = tqdm(
            self.train_dataloader,
            total=num_batches,
            desc=f"Train Epoch {epoch_idx}/{self.max_epochs}",
        )

        self.optimizer.zero_grad(set_to_none=True)
        accumulated_losses: list[torch.Tensor] = []
        optimizer_step_idx = start_batch_idx // accum_steps
        optimizer_window_started = None
        self._maybe_run_train_step_read_trace(epoch_idx=epoch_idx)

        for step_idx, (inputs, targets, slices) in enumerate(iterator):
            if step_idx < start_batch_idx:
                continue
            if optimizer_window_started is None:
                optimizer_window_started = time.perf_counter()
                if torch.cuda.is_available() and torch.device(self.device).type == "cuda":
                    torch.cuda.reset_peak_memory_stats()
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            current_optimizer_step = int(self.optimizer_step)
            micro_step_idx = int(step_idx % accum_steps)
            self._set_dense_teacher_runtime(targets)
            inline_trace_active = self._set_train_inline_event_trace_runtime(
                epoch_idx=epoch_idx,
                optimizer_step_idx=current_optimizer_step,
                train_batch_idx=step_idx,
                micro_step_idx=micro_step_idx,
                inputs=inputs,
                targets=targets,
            )
            try:
                with self._autocast_context():
                    loss, _ = self.compute_loss(
                        inputs,
                        targets,
                        return_predictions=False,
                    )
                    if self.input_type == "discrete":
                        auxiliary_loss = []

                        def get_auxiliary_loss(module):
                            if hasattr(module, "get_auxiliary_loss"):
                                auxiliary_loss.append(module.get_auxiliary_loss())

                        self.model.apply(get_auxiliary_loss)
                        if auxiliary_loss:
                            loss = loss + sum(auxiliary_loss)
            finally:
                self._clear_dense_teacher_runtime()
                if inline_trace_active:
                    self._clear_read_candidate_probe_runtime()

            if not torch.isfinite(loss.detach()).all():
                raise FloatingPointError(
                    f"Non-finite training loss at epoch={epoch_idx}, batch={step_idx}."
                )

            # Use correct divisor for the last partial window
            effective_accum = remainder if step_idx >= partial_start else accum_steps
            scaled_loss = loss / effective_accum
            self.scaler.scale(scaled_loss).backward()
            accumulated_losses.append(loss.detach())

            is_accum_boundary = (step_idx + 1) % accum_steps == 0
            is_last_batch = (step_idx + 1) == num_batches

            if is_accum_boundary or is_last_batch:
                optimizer_succeeded = self._step_optimizer()
                self.optimizer.zero_grad(set_to_none=True)

                micro_count = effective_accum if is_last_batch and not is_accum_boundary else accum_steps
                avg_loss = float(torch.stack(accumulated_losses).sum().cpu().item() / micro_count)
                iterator.set_postfix({"loss": avg_loss})
                metrics = {
                    "train/loss": avg_loss,
                    "train/optimizer_step_skipped": int(not optimizer_succeeded),
                    "train/grad_scaler_scale": float(self.scaler.get_scale()),
                    "train/optimizer_step_wall_sec": (
                        time.perf_counter() - optimizer_window_started
                    ),
                    "train/precision": self.precision,
                    "epoch": epoch_idx,
                }
                if torch.cuda.is_available() and torch.device(self.device).type == "cuda":
                    metrics["train/peak_allocated_mib"] = (
                        torch.cuda.max_memory_allocated() / 1024**2
                    )
                    metrics["train/peak_reserved_mib"] = (
                        torch.cuda.max_memory_reserved() / 1024**2
                    )
                if slices:
                    mqar_case = slices[0].get("mqar_case")
                    if mqar_case is not None:
                        metrics[f"train/mqar_case/loss-{mqar_case}"] = avg_loss
                metrics.update(self._collect_model_scalar_metrics())
                if optimizer_succeeded:
                    self.optimizer_step += 1
                self._log_metrics(metrics)
                accumulated_losses.clear()
                optimizer_step_idx += 1
                self._resume_epoch_idx = epoch_idx
                self._resume_next_train_batch_idx = step_idx + 1
                if optimizer_succeeded:
                    self._maybe_run_train_step_read_trace(epoch_idx=epoch_idx)
                if (
                    self.max_train_steps is not None
                    and self.optimizer_step >= self.max_train_steps
                ):
                    self._max_train_steps_reached = True
                    return

                if (
                    validation_callback is not None
                    and optimizer_step_idx in validation_boundaries
                ):
                    validation_callback(optimizer_step_idx, step_idx + 1)
                    self.model.train()
                optimizer_window_started = None

    def test(self, epoch_idx: int):
        self.model.eval()
        test_loss = 0.0
        sample_weighted_loss = 0.0
        total_examples = 0
        processed_batches = 0
        results = []
        scalar_metric_buckets: dict[str, list[float]] = defaultdict(list)
        if torch.cuda.is_available() and torch.device(self.device).type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        with torch.no_grad(), tqdm(
            total=len(self.test_dataloader),
            desc=f"Valid Epoch {epoch_idx}/{self.max_epochs}",
            postfix={"loss": "-", "acc": "-"},
        ) as iterator:
            for batch_idx, (inputs, targets, slices) in enumerate(self.test_dataloader):
                if (
                    self.max_validation_batches is not None
                    and batch_idx >= self.max_validation_batches
                ):
                    break
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self._set_dense_teacher_runtime(targets)
                read_probe_active = self._set_read_candidate_probe_runtime(
                    batch_idx=batch_idx,
                    inputs=inputs,
                    targets=targets,
                    epoch_idx=epoch_idx,
                    force_trace_enabled=False if self.read_trace_train_steps else None,
                )
                try:
                    with self._autocast_context():
                        loss, preds = self.compute_loss(inputs, targets)
                finally:
                    self._clear_dense_teacher_runtime()
                    if read_probe_active:
                        self._clear_read_candidate_probe_runtime()
                test_loss += float(loss.detach().cpu().item())
                batch_examples = int(targets.size(0))
                sample_weighted_loss += float(loss.detach().cpu().item()) * batch_examples
                total_examples += batch_examples
                processed_batches += 1
                results.extend(compute_metrics(preds.cpu(), targets.cpu(), slices))
                for key, value in self._collect_model_scalar_metrics().items():
                    scalar_metric_buckets[key].append(float(value))
                iterator.update(1)

            results = pd.DataFrame(results)
            if processed_batches <= 0 or results.empty:
                raise RuntimeError("Validation produced no batches.")
            test_accuracy = results["accuracy"].mean()

            # logging and printing
            metrics = {
                "valid/loss": test_loss / processed_batches,
                "valid/loss_sample_weighted": sample_weighted_loss / total_examples,
                "valid/accuracy": test_accuracy.item(),
            }
            if torch.cuda.is_available() and torch.device(self.device).type == "cuda":
                metrics["valid/peak_allocated_mib"] = (
                    torch.cuda.max_memory_allocated() / 1024**2
                )
                metrics["valid/peak_reserved_mib"] = (
                    torch.cuda.max_memory_reserved() / 1024**2
                )

            # compute metrics for slices
            for key in self.slice_keys:
                acc_by_slice = results.groupby(key)["accuracy"].mean()
                for value, accuracy in acc_by_slice.items():
                    metrics[f"valid/{key}/accuracy-{value}"] = accuracy

            aggregated_scalar_metrics = {
                key: float(np.mean(values))
                for key, values in scalar_metric_buckets.items()
                if values
            }
            metrics.update(self._prefix_phase_metrics(aggregated_scalar_metrics, "valid/"))

            iterator.set_postfix(metrics)
            self._log_metrics({"epoch": epoch_idx, **metrics})
        return metrics

    def fit(self):
        self.model.to(self.device)
        if self.checkpoint_manager is not None:
            self.checkpoint_manager.setup()
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.max_epochs, eta_min=0.0
        )
        device_type = torch.device(self.device).type
        if self.precision != "float32" and device_type != "cuda":
            raise RuntimeError("AMP precision profiles require a CUDA device.")
        if self.precision == "amp_bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("amp_bfloat16 is not supported by this CUDA device.")
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.precision == "amp_float16",
        )
        if self.training_runtime_initial_state:
            _restore_model_runtime_state(
                self.model,
                self.training_runtime_initial_state,
            )
        self._load_resume()
        last_metrics = None
        last_epoch = None
        for epoch_idx in range(self._resume_epoch_idx, self.max_epochs):
            start_batch_idx = (
                self._resume_next_train_batch_idx
                if epoch_idx == self._resume_epoch_idx
                else 0
            )

            def validate_and_save_resume(
                optimizer_step_idx: int,
                next_train_batch_idx: int,
            ) -> None:
                key = (int(epoch_idx), int(optimizer_step_idx))
                if key in self._completed_validations:
                    return
                self.test(epoch_idx)
                self._completed_validations.add(key)
                self._save_resume(
                    epoch_idx=epoch_idx,
                    next_train_batch_idx=next_train_batch_idx,
                    allow_controlled_stop=True,
                )

            self.train_epoch(
                epoch_idx,
                start_batch_idx=start_batch_idx,
                validation_callback=validate_and_save_resume,
            )
            metrics = self.test(epoch_idx)
            last_metrics = metrics
            last_epoch = epoch_idx
            if self.checkpoint_manager is not None:
                self.checkpoint_manager.save_epoch(
                    model=self.model,
                    epoch_idx=epoch_idx,
                    metrics=metrics,
                )

            early_stopped = (self.early_stopping_metric is not None) and metrics[
                self.early_stopping_metric
            ] > self.early_stopping_threshold
            if early_stopped:
                print(
                    f"Early stopping triggered at epoch {epoch_idx} with "
                    f"{self.early_stopping_metric} {metrics[self.early_stopping_metric]} > {self.early_stopping_threshold}"
                )
            terminal = bool(early_stopped or self._max_train_steps_reached)
            if not terminal:
                self.scheduler.step()
            self._resume_epoch_idx = epoch_idx + 1
            self._resume_next_train_batch_idx = 0
            self._save_resume(
                epoch_idx=epoch_idx + 1,
                next_train_batch_idx=0,
                allow_controlled_stop=False,
            )
            if terminal:
                break

        return {
            "final_epoch": last_epoch,
            "final_metrics": last_metrics,
        }


def compute_metrics(
    preds: torch.Tensor, 
    targets: torch.Tensor, 
    slices: List[dict],
    ignore_index: int = -100,
):
    results = []
    for pred, target, slc in zip(preds, targets, slices):
        results.append(
            {
                "accuracy": (pred == target)[target != ignore_index].to(float).mean().item(),
                **slc
            }
        )
    return results


def train(config: TrainConfig):
    import os
    set_determinism(config.seed, deterministic=os.environ.get("TORCH_DETERMINISTIC", "0") == "1")
    checkpoint_manager = CheckpointManager(config)
    previous_signal_handlers = _install_training_signal_handlers()
    logger: LoggerProtocol | None = None
    try:
        logger = build_logger(config)
        logger.log_config(config)
        config.print()

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
            model.load_state_dict(
                payload["model_state_dict"],
                strict=bool(config.init_checkpoint_strict),
            )
            print(f"Loaded init checkpoint from {resolved_init_checkpoint}")

        logger.log_model(model, config=config)
        update_manifest_for_run(
            config=config,
            logger_summary=logger.get_summary(),
            status="running",
        )

        task = Trainer(
            model=model,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            input_type=config.input_type,
            max_epochs=config.max_epochs,
            max_train_steps=config.max_train_steps,
            max_validation_batches=config.max_validation_batches,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            validations_per_epoch=config.validations_per_epoch,
            precision=config.precision,
            training_runtime_initial_state=config.training_runtime_initial_state,
            resume_stop_after_optimizer_step=(
                config.resume_stop_after_optimizer_step
            ),
            max_grad_scaler_skips=config.max_grad_scaler_skips,
            max_consecutive_grad_scaler_skips=(
                config.max_consecutive_grad_scaler_skips
            ),
            training_telemetry_path=config.training_telemetry_path,
            early_stopping_metric=config.early_stopping_metric,
            early_stopping_threshold=config.early_stopping_threshold,
            slice_keys=config.slice_keys,
            loss_type=config.loss_type,
            read_churn_probe_enabled=config.read_churn_probe_enabled,
            read_churn_probe_valid_batches=config.read_churn_probe_valid_batches,
            read_churn_probe_max_samples=config.read_churn_probe_max_samples,
            read_churn_probe_query_only=config.read_churn_probe_query_only,
            read_trace_enabled=config.read_trace_enabled,
            read_trace_valid_batches=config.read_trace_valid_batches,
            read_trace_max_samples=config.read_trace_max_samples,
            read_trace_query_only=config.read_trace_query_only,
            read_trace_max_queries_per_sample=config.read_trace_max_queries_per_sample,
            read_trace_output_dir=config.read_trace_output_dir,
            read_trace_train_steps=config.read_trace_train_steps,
            train_inline_event_trace_enabled=config.train_inline_event_trace_enabled,
            train_inline_event_trace_steps=config.train_inline_event_trace_steps,
            train_inline_event_trace_output_dir=config.train_inline_event_trace_output_dir,
            run_id=config.run_id,
            device="cuda" if torch.cuda.is_available() else "cpu",
            logger=logger,
            checkpoint_manager=checkpoint_manager,
        )
        task.fit()
        update_manifest_for_run(
            config=config,
            logger_summary=logger.get_summary(),
            status="completed",
        )
    except BaseException as exc:
        if logger is not None:
            update_manifest_for_run(
                config=config,
                logger_summary=logger.get_summary(),
                status="failed",
                error=_format_training_error(exc),
            )
        raise
    finally:
        _restore_training_signal_handlers(previous_signal_handlers)
        if logger is not None:
            logger.finish()


if __name__ == "__main__":
    config = TrainConfig.from_cli()
    train(config)
