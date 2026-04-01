import argparse
import random
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Union
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
from zoology.checkpoints import serialize_train_config
from zoology.experiments.flash_vqg.manifest import update_manifest_for_run
from zoology.model import LanguageModel, ContinuousInputModel
from zoology.logger import LoggerProtocol, build_logger
from zoology.utils import set_determinism
from zoology.metrics import compute_mse, compute_ce_with_embeddings


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
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp_path)
        tmp_path.replace(path)

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
        learning_rate: float = 1e-3,
        weight_decay: float = 0.1,
        gradient_accumulation_steps: int = 1,
        early_stopping_metric: str = None,
        early_stopping_threshold: float = None,
        loss_type: str = "ce",
        slice_keys: List[str] = [],
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
        self.early_stopping_metric = early_stopping_metric
        self.early_stopping_threshold = early_stopping_threshold
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.slice_keys = slice_keys
        self.loss_type = loss_type
        self.global_step = 0

    def compute_loss(self, inputs, targets):
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
            
            logits = outputs_flat @ value_embeddings.T
            preds = (logits).argmax(dim=-1).view(targets.shape)
            return loss, preds
        
        else: # discrete
            if self.loss_type == "ce":
                logits = self.model(inputs)
                loss = self.loss_fn(
                    rearrange(logits, "... c -> (...) c"), 
                    targets.flatten()
                )
                preds = logits.argmax(dim=-1)
                return loss, preds
            
            elif self.loss_type == "mse":
                embeddings = self.model(inputs, return_embeddings=True)
                target_embeds = self.model.backbone.embeddings.word_embeddings(targets)
                mask = (targets != -100).unsqueeze(-1)
                loss, _ = compute_mse(
                    embeddings[mask.expand_as(embeddings)].view(-1, embeddings.size(-1)),
                    target_embeds[mask.expand_as(target_embeds)].view(-1, target_embeds.size(-1)),
                )
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

    def _log_metrics(self, metrics: dict[str, float | int]):
        self.logger.log(metrics, step=self.global_step)
        self.global_step += 1

    def train_epoch(self, epoch_idx: int):
        self.model.train()
        sampler = getattr(self.train_dataloader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch_idx)

        accum_steps = self.gradient_accumulation_steps
        num_batches = len(self.train_dataloader)
        remainder = num_batches % accum_steps
        # Index where the last (possibly partial) accumulation window begins
        partial_start = num_batches - remainder if remainder > 0 else num_batches

        iterator = tqdm(
            self.train_dataloader,
            total=num_batches,
            desc=f"Train Epoch {epoch_idx}/{self.max_epochs}",
        )

        self.optimizer.zero_grad()
        accum_loss = 0.0

        for step_idx, (inputs, targets, slices) in enumerate(iterator):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            loss, preds = self.compute_loss(inputs, targets)

            # Auxiliary losses (discrete mode only)
            if self.input_type == "discrete":
                auxiliary_loss = []
                def get_auxiliary_loss(module):
                    if hasattr(module, "get_auxiliary_loss"):
                        auxiliary_loss.append(module.get_auxiliary_loss())
                self.model.apply(get_auxiliary_loss)
                if auxiliary_loss:
                    loss = loss + sum(auxiliary_loss)

            # Use correct divisor for the last partial window
            effective_accum = remainder if step_idx >= partial_start else accum_steps
            (loss / effective_accum).backward()
            accum_loss += loss.item()

            is_accum_boundary = (step_idx + 1) % accum_steps == 0
            is_last_batch = (step_idx + 1) == num_batches

            if is_accum_boundary or is_last_batch:
                self.optimizer.step()
                self.optimizer.zero_grad()

                micro_count = effective_accum if is_last_batch and not is_accum_boundary else accum_steps
                avg_loss = accum_loss / micro_count
                iterator.set_postfix({"loss": avg_loss})
                metrics = {"train/loss": avg_loss, "epoch": epoch_idx}
                if slices:
                    mqar_case = slices[0].get("mqar_case")
                    if mqar_case is not None:
                        metrics[f"train/mqar_case/loss-{mqar_case}"] = avg_loss
                metrics.update(self._collect_model_scalar_metrics())
                self._log_metrics(metrics)
                accum_loss = 0.0

    def test(self, epoch_idx: int):
        self.model.eval()
        test_loss = 0
        results = []
        scalar_metric_buckets: dict[str, list[float]] = defaultdict(list)

        with torch.no_grad(), tqdm(
            total=len(self.test_dataloader),
            desc=f"Valid Epoch {epoch_idx}/{self.max_epochs}",
            postfix={"loss": "-", "acc": "-"},
        ) as iterator:
            for inputs, targets, slices in self.test_dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                loss, preds = self.compute_loss(inputs, targets)
                test_loss += loss / len(self.test_dataloader)
                results.extend(compute_metrics(preds.cpu(), targets.cpu(), slices))
                for key, value in self._collect_model_scalar_metrics().items():
                    scalar_metric_buckets[key].append(float(value))
                iterator.update(1)

            results = pd.DataFrame(results)
            test_accuracy = results["accuracy"].mean()

            # logging and printing
            metrics = {
                "valid/loss": test_loss.item(),
                "valid/accuracy": test_accuracy.item(),
            }

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
        last_metrics = None
        last_epoch = None
        for epoch_idx in range(self.max_epochs):
            self.train_epoch(epoch_idx)
            metrics = self.test(epoch_idx)
            last_metrics = metrics
            last_epoch = epoch_idx
            if self.checkpoint_manager is not None:
                self.checkpoint_manager.save_epoch(
                    model=self.model,
                    epoch_idx=epoch_idx,
                    metrics=metrics,
                )

            # early stopping
            if (self.early_stopping_metric is not None) and metrics[
                self.early_stopping_metric
            ] > self.early_stopping_threshold:
                print(
                    f"Early stopping triggered at epoch {epoch_idx} with "
                    f"{self.early_stopping_metric} {metrics[self.early_stopping_metric]} > {self.early_stopping_threshold}"
                )
                break

            self.scheduler.step()

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
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            early_stopping_metric=config.early_stopping_metric,
            early_stopping_threshold=config.early_stopping_threshold,
            slice_keys=config.slice_keys,
            loss_type=config.loss_type,
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
    except Exception as exc:
        if logger is not None:
            update_manifest_for_run(
                config=config,
                logger_summary=logger.get_summary(),
                status="failed",
                error=str(exc),
            )
        raise
    finally:
        if logger is not None:
            logger.finish()


if __name__ == "__main__":
    config = TrainConfig.from_cli()
    train(config)
