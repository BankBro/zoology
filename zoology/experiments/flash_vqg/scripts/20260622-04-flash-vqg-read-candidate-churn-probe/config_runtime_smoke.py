#!/usr/bin/env python3
"""Config-to-runtime smoke for read candidate churn probe."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path("/home/lyj/mnt/project/zoology")
FLASH_VQG_ROOT = Path("/home/lyj/mnt/project/Flash-VQG")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(FLASH_VQG_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(FLASH_VQG_ROOT / "src"))

from zoology.experiments.flash_vqg.flash_vqg_suite import build_configs
from zoology.experiments.flash_vqg.manifest import config_summary_from_config
from zoology.experiments.flash_vqg.metrics_white_list import derive_flash_metric_controls
from zoology.logger import NoOpLogger
from zoology.model import LanguageModel
from zoology.train import Trainer


class FixedValidationDataset(Dataset):
    def __init__(self, *, batch_size: int, seq_len: int, vocab_size: int):
        inputs = torch.arange(batch_size * seq_len, dtype=torch.long).view(batch_size, seq_len)
        inputs = inputs.remainder(vocab_size // 2)
        targets = torch.full_like(inputs, -100)
        targets[:, -4:] = inputs[:, -4:].remainder(vocab_size)
        self.items = [(inputs, targets, [{"mqar_case": "smoke"} for _ in range(batch_size)])]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default)
        + "\n",
        encoding="utf-8",
    )


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA smoke, but CUDA is not available.")
    return device


def _build_config():
    metrics_white_list = [
        "attn/gd_residual_remote_read_topk_effective",
        "attn/gd_residual_read_margin_top1_top2_mean",
        "attn/gd_residual_read_entropy_mean",
        "attn/gd_residual_read_selected_mass_mean",
        "attn/gd_residual_read_candidate_*",
        "valid/attn/gd_residual_read_candidate_*",
    ]
    configs = build_configs(
        sweep_id="read-candidate-churn-smoke",
        flash_backend="torch",
        logger_backend="none",
        include_gdn=False,
        block_len=4,
        local_num_blocks=1,
        dmodels=[64],
        learning_rates=[1e-4],
        if_remote_enabled=True,
        train_batch_orders=["sequential"],
        seed_values=[123],
        data_seed=123,
        num_codebook_vectors_values=[8],
        fox_remote_path_backend="torch",
        fox_remote_read_topk_values=[4],
        fox_remote_formula="gd_residual_v1",
        fox_gd_residual_rank=2,
        fox_gd_residual_write_topk=2,
        fox_gd_residual_builder="grouped_chunk_torch_ref",
        fox_gd_residual_pack_mode="semivec_ref",
        fox_gd_residual_chunk_size=2,
        fox_gd_residual_mu_min_count=0.1,
        fox_gd_residual_beta_init=0.5,
        fox_gd_residual_lambda_init=0.05,
        vq_score_mode="codebook_dot",
        vq_weight_mode="dense_softmax",
        vq_update_mode="grad",
        vq_softmax_tau=0.25,
        vq_topk=4,
        train_batch_size=2,
        eval_batch_size=2,
        gradient_accumulation_steps=1,
        validations_per_epoch=1,
        max_epochs=1,
        early_stopping_metric=None,
        early_stopping_threshold=None,
        cache_dir="./data/flash_vqg",
        metrics_white_list=metrics_white_list,
        read_churn_probe_enabled=True,
        read_churn_probe_valid_batches=[0],
        read_churn_probe_max_samples=2,
        read_churn_probe_query_only=True,
    )
    if len(configs) != 1:
        raise RuntimeError(f"expected one config, got {len(configs)}")
    return configs[0], metrics_white_list


def _run_validation_probe(config, device: torch.device) -> dict[str, Any]:
    torch.manual_seed(123)
    model = LanguageModel(config.model).to(device)
    dataset = FixedValidationDataset(batch_size=2, seq_len=16, vocab_size=config.model.vocab_size)
    loader = DataLoader(dataset, batch_size=None, shuffle=False)
    trainer = Trainer(
        model=model,
        train_dataloader=loader,
        test_dataloader=loader,
        input_type=config.input_type,
        max_epochs=1,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        gradient_accumulation_steps=1,
        validations_per_epoch=1,
        early_stopping_metric=None,
        early_stopping_threshold=None,
        slice_keys=[],
        loss_type=config.loss_type,
        read_churn_probe_enabled=True,
        read_churn_probe_valid_batches=[0],
        read_churn_probe_max_samples=2,
        read_churn_probe_query_only=True,
        device=device,
        logger=NoOpLogger(),
        checkpoint_manager=None,
    )
    trainer.loss_fn = torch.nn.CrossEntropyLoss()
    first = trainer.test(epoch_idx=0)
    second = trainer.test(epoch_idx=1)
    required_second = [
        "valid/attn/gd_residual_read_candidate_probe_count",
        "valid/attn/gd_residual_read_candidate_has_prev",
        "valid/attn/gd_residual_read_candidate_retention_mean",
        "valid/attn/gd_residual_read_candidate_churn_mean",
        "valid/attn/gd_residual_read_candidate_top1_flip_rate",
        "valid/attn/gd_residual_read_margin_top1_top2_mean",
        "valid/attn/gd_residual_read_entropy_mean",
        "valid/attn/gd_residual_read_selected_mass_mean",
    ]
    missing = [key for key in required_second if key not in second]
    checks = {
        "first_has_prev_zero": first.get("valid/attn/gd_residual_read_candidate_has_prev") == 0.0,
        "second_has_prev_one": second.get("valid/attn/gd_residual_read_candidate_has_prev") == 1.0,
        "second_probe_count_positive": second.get("valid/attn/gd_residual_read_candidate_probe_count", 0.0) > 0.0,
    }
    selected = {
        key: second[key]
        for key in sorted(second)
        if "gd_residual_read" in key
    }
    return {
        "passed": not missing and all(checks.values()),
        "missing_second_metrics": missing,
        "checks": checks,
        "first_read_metrics": {
            key: first[key]
            for key in sorted(first)
            if "gd_residual_read" in key
        },
        "second_read_metrics": selected,
    }


def run_smoke(output_dir: Path, device: torch.device) -> dict[str, Any]:
    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    config, metrics_white_list = _build_config()
    metric_controls = derive_flash_metric_controls(metrics_white_list)
    manifest_summary = config_summary_from_config(config)
    validation = _run_validation_probe(config, device)
    static_checks = {
        "train_config_probe_enabled": bool(config.read_churn_probe_enabled),
        "train_config_probe_batches": config.read_churn_probe_valid_batches == [0],
        "manifest_probe_enabled": manifest_summary.get("read_churn_probe_enabled") is True,
        "manifest_probe_batches": manifest_summary.get("read_churn_probe_valid_batches") == [0],
        "metric_controls_enable_layer_metrics": bool(metric_controls["enable_layer_metrics"]),
        "metric_controls_phase2_not_off": metric_controls["fox_phase2_metrics_mode"] != "off",
    }
    summary = {
        "status": "passed" if all(static_checks.values()) and validation["passed"] else "failed",
        "device": str(device),
        "output_dir": str(output_dir),
        "wall_clock_sec": time.time() - started,
        "run_id": config.run_id,
        "static_checks": static_checks,
        "metric_controls": metric_controls,
        "manifest_summary_subset": {
            key: manifest_summary.get(key)
            for key in (
                "fox_remote_read_topk",
                "read_churn_probe_enabled",
                "read_churn_probe_valid_batches",
                "read_churn_probe_max_samples",
                "read_churn_probe_query_only",
            )
        },
        "validation": validation,
    }
    _write_json(output_dir / "summary.json", summary)
    (output_dir / "README.md").write_text(
        "# Read candidate churn smoke output\n\n"
        f"- status: `{summary['status']}`\n"
        f"- device: `{device}`\n\n"
        "See `summary.json`.\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_smoke(output_dir=args.output_dir, device=_resolve_device(args.device))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
