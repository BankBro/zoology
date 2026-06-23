#!/usr/bin/env python3
"""Config-to-runtime smoke for pressure telemetry controls."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path("/home/lyj/mnt/project/zoology")
FLASH_VQG_ROOT = Path("/home/lyj/mnt/project/Flash-VQG")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(FLASH_VQG_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(FLASH_VQG_ROOT / "src"))

from zoology.experiments.flash_vqg.flash_vqg_suite import build_configs
from zoology.experiments.flash_vqg.metrics_white_list import derive_flash_metric_controls
from zoology.model import LanguageModel


@dataclass(frozen=True)
class SmokeCase:
    case_id: str
    kwargs: dict[str, Any]
    expected_config: dict[str, Any]
    required_metrics: tuple[str, ...]
    expected_metrics: dict[str, Any]
    forward_passes: int = 1


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


def _isclose(actual: Any, expected: Any) -> bool:
    if expected is None:
        return actual is None
    if isinstance(expected, bool):
        return bool(actual) is expected
    if isinstance(expected, int) and not isinstance(expected, bool):
        return int(actual) == expected
    if isinstance(expected, float):
        return actual is not None and math.isclose(float(actual), expected, rel_tol=1e-6, abs_tol=1e-8)
    return actual == expected


def _flash_kwargs(config) -> dict[str, Any]:
    mixer_cfg = config.model.sequence_mixer
    if mixer_cfg is None:
        raise RuntimeError("TrainConfig has no sequence_mixer.")
    for item in reversed(mixer_cfg.kwargs.get("configs", [])):
        if item.get("name") == "zoology.mixers.flash_vqg.FlashVQGMixer":
            return dict(item.get("kwargs", {}))
    raise RuntimeError("FlashVQGMixer config not found.")


def _first_flash_mixer(model: LanguageModel):
    for module in model.modules():
        if module.__class__.__name__ == "FlashVQGMixer":
            return module
    raise RuntimeError("FlashVQGMixer module not found.")


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA smoke, but CUDA is not available.")
    return device


def _base_build_kwargs(metrics_white_list: list[str]) -> dict[str, Any]:
    return {
        "sweep_id": "pressure-telemetry-smoke",
        "flash_backend": "torch",
        "logger_backend": "none",
        "include_gdn": False,
        "block_len": 4,
        "dmodels": [64],
        "learning_rates": [1e-4],
        "if_remote_enabled": True,
        "local_num_blocks": 1,
        "seed_values": [123],
        "data_seed": 123,
        "num_codebook_vectors_values": [8],
        "fox_remote_path_backend": "torch",
        "fox_remote_read_topk_values": [2],
        "fox_remote_formula": "gd_residual_v1",
        "fox_gd_residual_rank": 2,
        "fox_gd_residual_write_topk": 2,
        "fox_gd_residual_chunk_size": 2,
        "fox_gd_residual_builder": "grouped_chunk_torch_ref",
        "fox_gd_residual_pack_mode": "semivec_ref",
        "fox_gd_residual_mu_min_count": 0.1,
        "fox_gd_residual_beta_init": 0.5,
        "fox_gd_residual_lambda_init": 0.05,
        "vq_score_mode": "codebook_dot",
        "vq_weight_mode": "dense_softmax",
        "vq_update_mode": "grad",
        "vq_softmax_tau": 0.25,
        "vq_topk": 4,
        "train_batch_size": 2,
        "eval_batch_size": 2,
        "gradient_accumulation_steps": 1,
        "validations_per_epoch": 1,
        "max_epochs": 1,
        "early_stopping_metric": None,
        "early_stopping_threshold": None,
        "cache_dir": "./data/flash_vqg",
        "metrics_white_list": metrics_white_list,
        "read_churn_probe_enabled": False,
        "read_churn_probe_valid_batches": [],
        "read_trace_enabled": False,
        "read_trace_valid_batches": [],
    }


def _metrics_white_list() -> list[str]:
    return [
        "attn/gd_residual_remote_read_topk_effective",
        "attn/gd_residual_write_strength_mean",
        "attn/gd_residual_uncapped_write_strength_mean",
        "attn/gd_residual_sum_zeta_mean",
        "attn/gd_residual_uncapped_sum_zeta_mean",
        "attn/gd_residual_write_strength_cap_active",
        "attn/gd_residual_write_strength_effective_cap",
        "attn/gd_residual_write_strength_scheduled_cap",
        "attn/gd_residual_write_strength_cap_release_progress",
        "attn/gd_residual_write_strength_cap_hit_ratio",
        "attn/gd_residual_update_norm_mean",
        "attn/gd_residual_update_norm_p95",
        "attn/gd_residual_update_norm_max",
        "attn/gd_residual_update_norm_cap_hit_ratio",
        "attn/gd_residual_update_norm_cap_active",
        "attn/gd_residual_update_norm_effective_cap",
        "attn/gd_residual_m_norm_mean",
        "attn/gd_residual_m_norm_max",
        "attn/gd_residual_lambda_mean",
        "attn/gd_residual_inject_ratio",
        "attn/gd_residual_read_margin_top1_top2_mean",
        "attn/gd_residual_read_entropy_mean",
        "attn/gd_residual_read_selected_mass_mean",
    ]


def _common_required_metrics() -> tuple[str, ...]:
    return (
        "attn/gd_residual_remote_read_topk_effective",
        "attn/gd_residual_write_strength_mean",
        "attn/gd_residual_uncapped_write_strength_mean",
        "attn/gd_residual_sum_zeta_mean",
        "attn/gd_residual_uncapped_sum_zeta_mean",
        "attn/gd_residual_update_norm_mean",
        "attn/gd_residual_update_norm_p95",
        "attn/gd_residual_update_norm_max",
        "attn/gd_residual_update_norm_cap_hit_ratio",
        "attn/gd_residual_update_norm_cap_active",
        "attn/gd_residual_update_norm_effective_cap",
        "attn/gd_residual_write_strength_scheduled_cap",
        "attn/gd_residual_write_strength_cap_release_progress",
        "attn/gd_residual_m_norm_mean",
        "attn/gd_residual_m_norm_max",
        "attn/gd_residual_lambda_mean",
        "attn/gd_residual_inject_ratio",
    )


def _cases() -> list[SmokeCase]:
    common = _common_required_metrics()
    return [
        SmokeCase(
            case_id="hard04",
            kwargs={
                "fox_gd_residual_write_strength_cap": 0.04,
                "fox_gd_residual_write_strength_cap_mode": "hard",
            },
            expected_config={
                "fox_gd_residual_write_strength_cap": 0.04,
                "fox_gd_residual_write_strength_cap_mode": "hard",
                "fox_gd_residual_update_norm_cap": None,
            },
            required_metrics=common
            + (
                "attn/gd_residual_write_strength_cap_active",
                "attn/gd_residual_write_strength_effective_cap",
            ),
            expected_metrics={
                "attn/gd_residual_write_strength_cap_active": 1.0,
                "attn/gd_residual_write_strength_effective_cap": 0.04,
                "attn/gd_residual_write_strength_scheduled_cap": 0.04,
                "attn/gd_residual_write_strength_cap_release_progress": 0.0,
                "attn/gd_residual_update_norm_cap_active": 0.0,
                "attn/gd_residual_update_norm_effective_cap": 0.0,
            },
        ),
        SmokeCase(
            case_id="caprel0406late-progress",
            kwargs={
                "fox_gd_residual_write_strength_cap": 0.04,
                "fox_gd_residual_write_strength_cap_final": 0.06,
                "fox_gd_residual_write_strength_cap_release_start_train_steps": 1,
                "fox_gd_residual_write_strength_cap_release_end_train_steps": 3,
            },
            expected_config={
                "fox_gd_residual_write_strength_cap": 0.04,
                "fox_gd_residual_write_strength_cap_final": 0.06,
                "fox_gd_residual_write_strength_cap_release_start_train_steps": 1,
                "fox_gd_residual_write_strength_cap_release_end_train_steps": 3,
            },
            required_metrics=common
            + (
                "attn/gd_residual_write_strength_cap_active",
                "attn/gd_residual_write_strength_effective_cap",
            ),
            expected_metrics={
                "attn/gd_residual_write_strength_cap_active": 1.0,
                "attn/gd_residual_write_strength_effective_cap": 0.05,
                "attn/gd_residual_write_strength_scheduled_cap": 0.05,
                "attn/gd_residual_write_strength_cap_release_progress": 0.5,
                "attn/gd_residual_update_norm_cap_active": 0.0,
            },
            forward_passes=3,
        ),
        SmokeCase(
            case_id="update-norm-cap",
            kwargs={
                "fox_gd_residual_write_strength_cap": 0.04,
                "fox_gd_residual_update_norm_cap": 0.02,
            },
            expected_config={
                "fox_gd_residual_write_strength_cap": 0.04,
                "fox_gd_residual_update_norm_cap": 0.02,
            },
            required_metrics=common
            + (
                "attn/gd_residual_write_strength_cap_active",
                "attn/gd_residual_write_strength_effective_cap",
            ),
            expected_metrics={
                "attn/gd_residual_write_strength_cap_active": 1.0,
                "attn/gd_residual_write_strength_effective_cap": 0.04,
                "attn/gd_residual_update_norm_cap_active": 1.0,
                "attn/gd_residual_update_norm_effective_cap": 0.02,
            },
        ),
        SmokeCase(
            case_id="update-norm-cap-hit",
            kwargs={
                "fox_gd_residual_write_strength_cap": 0.04,
                "fox_gd_residual_update_norm_cap": 0.001,
            },
            expected_config={
                "fox_gd_residual_write_strength_cap": 0.04,
                "fox_gd_residual_update_norm_cap": 0.001,
            },
            required_metrics=common
            + (
                "attn/gd_residual_write_strength_cap_active",
                "attn/gd_residual_write_strength_effective_cap",
            ),
            expected_metrics={
                "attn/gd_residual_write_strength_cap_active": 1.0,
                "attn/gd_residual_write_strength_effective_cap": 0.04,
                "attn/gd_residual_update_norm_cap_active": 1.0,
                "attn/gd_residual_update_norm_effective_cap": 0.001,
            },
        ),
    ]


def _build_case_config(case: SmokeCase, metrics_white_list: list[str]):
    kwargs = _base_build_kwargs(metrics_white_list)
    kwargs.update(case.kwargs)
    configs = build_configs(**kwargs)
    if len(configs) != 1:
        raise RuntimeError(f"{case.case_id}: expected one TrainConfig, got {len(configs)}")
    return configs[0]


def _check_expected(actual: dict[str, Any], expected: dict[str, Any]) -> list[dict[str, Any]]:
    checks = []
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        checks.append(
            {
                "key": key,
                "expected": expected_value,
                "actual": actual_value,
                "passed": _isclose(actual_value, expected_value),
            }
        )
    return checks


def _runtime_smoke(config, case: SmokeCase, device: torch.device) -> dict[str, Any]:
    torch.manual_seed(123)
    model = LanguageModel(config.model).to(device)
    model.train()
    for step in range(case.forward_passes):
        torch.manual_seed(1000 + step)
        input_ids = torch.randint(
            low=0,
            high=min(config.model.vocab_size, 128),
            size=(2, 16),
            device=device,
            dtype=torch.long,
        )
        _ = model(input_ids)
    mixer = _first_flash_mixer(model)
    metrics = mixer.get_scalar_metrics()
    missing = [key for key in case.required_metrics if key not in metrics]
    metric_checks = _check_expected(metrics, case.expected_metrics)
    if case.case_id == "update-norm-cap-hit":
        hit_ratio = float(metrics.get("attn/gd_residual_update_norm_cap_hit_ratio", 0.0))
        metric_checks.append(
            {
                "key": "attn/gd_residual_update_norm_cap_hit_ratio",
                "expected": ">0",
                "actual": hit_ratio,
                "passed": hit_ratio > 0.0,
            }
        )
    selected = {
        key: metrics[key]
        for key in sorted(metrics)
        if key.startswith("attn/gd_residual_")
    }
    train_forward_count = int(
        getattr(mixer.attn, "_fox_gd_residual_train_forward_count")
        .detach()
        .cpu()
        .item()
    )
    finite_checks = [
        {
            "key": key,
            "value": value,
            "passed": isinstance(value, int | float) and math.isfinite(float(value)),
        }
        for key, value in selected.items()
    ]
    return {
        "passed": (
            not missing
            and all(item["passed"] for item in metric_checks)
            and all(item["passed"] for item in finite_checks)
            and train_forward_count >= case.forward_passes
        ),
        "missing_required_metrics": missing,
        "metric_checks": metric_checks,
        "finite_checks_failed": [
            item for item in finite_checks if not item["passed"]
        ],
        "train_forward_count": train_forward_count,
        "metrics": selected,
    }


def run_smoke(output_dir: Path, device: torch.device) -> dict[str, Any]:
    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_white_list = _metrics_white_list()
    metric_controls = derive_flash_metric_controls(metrics_white_list)
    case_results = []
    for case in _cases():
        case_dir = output_dir / case.case_id
        config = _build_case_config(case, metrics_white_list)
        flash_kwargs = _flash_kwargs(config)
        config_checks = _check_expected(flash_kwargs, case.expected_config)
        runtime = _runtime_smoke(config, case, device)
        result = {
            "case_id": case.case_id,
            "run_id": config.run_id,
            "passed": all(item["passed"] for item in config_checks) and runtime["passed"],
            "config_checks": config_checks,
            "runtime": runtime,
            "flash_kwargs_subset": {
                key: flash_kwargs.get(key)
                for key in sorted(case.expected_config)
            },
        }
        _write_json(case_dir / "summary.json", result)
        case_results.append(result)
    summary = {
        "status": "passed" if all(item["passed"] for item in case_results) else "failed",
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "output_dir": str(output_dir),
        "wall_clock_sec": time.time() - started,
        "metric_controls": metric_controls,
        "metrics_white_list": metrics_white_list,
        "cases": case_results,
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    summary = run_smoke(args.output_dir, _resolve_device(args.device))
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))
    if summary["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
