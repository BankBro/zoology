#!/usr/bin/env python3
"""Config-to-runtime smoke for Flash-VQG gd_residual_v1 controls."""

from __future__ import annotations

import argparse
import ast
import importlib.util
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
SCRIPT_DIR = Path(__file__).resolve().parent


if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(FLASH_VQG_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(FLASH_VQG_ROOT / "src"))

from zoology.experiments.flash_vqg.flash_vqg_suite import build_configs
from zoology.experiments.flash_vqg.run_flash_vqg_suite import _render_generated_config
from zoology.model import LanguageModel


@dataclass(frozen=True)
class SmokeCase:
    case_id: str
    kwargs: dict[str, Any]
    expected: dict[str, Any]
    required_metrics: tuple[str, ...]
    expected_runtime_metrics: dict[str, Any] | None = None


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


def _append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=_json_default))
        handle.write("\n")


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
    configs = mixer_cfg.kwargs.get("configs", [])
    for item in reversed(configs):
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


def _base_build_kwargs() -> dict[str, Any]:
    return {
        "sweep_id": "config-runtime-smoke",
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
        "fox_remote_formula": "gd_residual_v1",
        "fox_gd_residual_rank": 2,
        "fox_gd_residual_write_topk": 2,
        "fox_gd_residual_chunk_size": 2,
        "fox_gd_residual_builder": "grouped_chunk_torch_ref",
        "fox_gd_residual_pack_mode": "semivec_ref",
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
        "metrics_white_list": [],
    }


def _cases() -> list[SmokeCase]:
    common_metrics = (
        "attn/gd_residual_write_strength_mean",
        "attn/gd_residual_m_norm_mean",
        "attn/gd_residual_lambda_mean",
        "attn/gd_residual_inject_ratio",
        "attn/gd_residual_remote_read_topk_effective",
    )
    return [
        SmokeCase(
            case_id="readk2-baseline",
            kwargs={
                "fox_remote_read_topk_values": [2],
                "fox_gd_residual_write_q_alpha": 0.75,
            },
            expected={
                "fox_remote_read_topk": 2,
                "fox_gd_residual_write_q_alpha": 0.75,
            },
            required_metrics=common_metrics,
            expected_runtime_metrics={
                "attn/gd_residual_remote_read_topk_effective": 2.0,
            },
        ),
        SmokeCase(
            case_id="readk4",
            kwargs={
                "fox_remote_read_topk_values": [4],
                "fox_gd_residual_write_q_alpha": 0.80,
            },
            expected={
                "fox_remote_read_topk": 4,
                "fox_gd_residual_write_q_alpha": 0.80,
            },
            required_metrics=common_metrics,
            expected_runtime_metrics={
                "attn/gd_residual_remote_read_topk_effective": 4.0,
            },
        ),
        SmokeCase(
            case_id="readk4-to-readk2-schedule",
            kwargs={
                "fox_remote_read_topk_values": [2],
                "fox_remote_read_topk_initial": 4,
                "fox_remote_read_topk_final": 2,
                "fox_remote_read_topk_release_start_train_steps": 1,
                "fox_remote_read_topk_release_end_train_steps": 3,
                "fox_remote_read_topk_schedule": "linear_int",
                "fox_remote_read_topk_eval_policy": "scheduled",
            },
            expected={
                "fox_remote_read_topk": 2,
                "fox_remote_read_topk_initial": 4,
                "fox_remote_read_topk_final": 2,
                "fox_remote_read_topk_release_start_train_steps": 1,
                "fox_remote_read_topk_release_end_train_steps": 3,
                "fox_remote_read_topk_schedule": "linear_int",
                "fox_remote_read_topk_eval_policy": "scheduled",
            },
            required_metrics=common_metrics,
            expected_runtime_metrics={
                "attn/gd_residual_remote_read_topk_effective": 4.0,
            },
        ),
        SmokeCase(
            case_id="write-cap-004",
            kwargs={
                "fox_remote_read_topk_values": [2],
                "fox_gd_residual_write_strength_cap": 0.04,
                "fox_gd_residual_write_strength_cap_mode": "hard",
            },
            expected={
                "fox_remote_read_topk": 2,
                "fox_gd_residual_write_strength_cap": 0.04,
                "fox_gd_residual_write_strength_cap_mode": "hard",
            },
            required_metrics=common_metrics
            + (
                "attn/gd_residual_write_strength_cap_active",
                "attn/gd_residual_write_strength_effective_cap",
            ),
            expected_runtime_metrics={
                "attn/gd_residual_remote_read_topk_effective": 2.0,
            },
        ),
        SmokeCase(
            case_id="bounded-beta-orthogonal-addr",
            kwargs={
                "fox_remote_read_topk_values": [2],
                "fox_gd_residual_beta_control_mode": "bounded_sigmoid",
                "fox_gd_residual_beta_low": 0.05,
                "fox_gd_residual_beta_high": 0.75,
                "fox_gd_residual_addr_init_rng_mode": "local_burn",
                "fox_gd_residual_addr_init_seed": 321,
                "fox_gd_residual_addr_proj_orthogonal_init": True,
                "codebook_init_rng_mode": "local_burn",
                "codebook_init_seed": 321,
            },
            expected={
                "fox_remote_read_topk": 2,
                "fox_gd_residual_beta_control_mode": "bounded_sigmoid",
                "fox_gd_residual_beta_low": 0.05,
                "fox_gd_residual_beta_high": 0.75,
                "fox_gd_residual_addr_init_rng_mode": "local_burn",
                "fox_gd_residual_addr_init_seed": 321,
                "fox_gd_residual_addr_proj_orthogonal_init": True,
                "codebook_init_rng_mode": "local_burn",
                "codebook_init_seed": 321,
            },
            required_metrics=common_metrics
            + (
                "attn/gd_residual_beta_bounded_active",
                "attn/gd_residual_beta_effective_low",
                "attn/gd_residual_beta_effective_high",
            ),
            expected_runtime_metrics={
                "attn/gd_residual_remote_read_topk_effective": 2.0,
            },
        ),
    ]


def _build_case_config(case: SmokeCase):
    kwargs = _base_build_kwargs()
    kwargs.update(case.kwargs)
    configs = build_configs(**kwargs)
    if len(configs) != 1:
        raise RuntimeError(f"{case.case_id}: expected one TrainConfig, got {len(configs)}")
    return configs[0]


def _check_expected(actual: dict[str, Any], expected: dict[str, Any]) -> list[dict[str, Any]]:
    checks = []
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        passed = _isclose(actual_value, expected_value)
        checks.append(
            {
                "key": key,
                "expected": expected_value,
                "actual": actual_value,
                "passed": passed,
            }
        )
    return checks


def _runtime_smoke(config, case: SmokeCase, device: torch.device) -> dict[str, Any]:
    torch.manual_seed(123)
    model = LanguageModel(config.model).to(device)
    model.train()
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
    metric_checks = _check_expected(metrics, case.expected_runtime_metrics or {})
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
    return {
        "passed": not missing and all(item["passed"] for item in metric_checks) and train_forward_count >= 1,
        "missing_required_metrics": missing,
        "metric_checks": metric_checks,
        "train_forward_count": train_forward_count,
        "metrics": selected,
    }


def _render_call_keyword_sets() -> list[set[str]]:
    source_path = REPO_ROOT / "zoology/experiments/flash_vqg/run_flash_vqg_suite.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    keyword_sets: list[set[str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "_render_generated_config":
            continue
        keyword_sets.append({keyword.arg for keyword in node.keywords if keyword.arg is not None})
    return keyword_sets


def _render_generated_config_text(case: SmokeCase) -> str:
    expected = case.expected
    return _render_generated_config(
        sweep_id="config-runtime-smoke-generated",
        backend="torch",
        logger_backend="none",
        include_gdn=False,
        block_lens=[4],
        paired_block_local_values=None,
        dmodels=[64],
        learning_rates=[1e-4],
        if_remote_enabled_values=[True],
        local_num_blocks_values=[1],
        train_batch_orders=["sequential"],
        seed_values=[123],
        data_seed=123,
        num_codebook_vectors_values=[8],
        num_codebook_vectors_map=None,
        fox_remote_path_backend="torch",
        fox_remote_read_topk_values=[expected.get("fox_remote_read_topk", 2)],
        fox_remote_read_topk_initial=expected.get("fox_remote_read_topk_initial"),
        fox_remote_read_topk_final=expected.get("fox_remote_read_topk_final"),
        fox_remote_read_topk_release_start_train_steps=expected.get(
            "fox_remote_read_topk_release_start_train_steps", 0
        ),
        fox_remote_read_topk_release_end_train_steps=expected.get(
            "fox_remote_read_topk_release_end_train_steps", 0
        ),
        fox_remote_read_topk_schedule=expected.get(
            "fox_remote_read_topk_schedule", "linear_int"
        ),
        fox_remote_read_topk_eval_policy=expected.get(
            "fox_remote_read_topk_eval_policy", "scheduled"
        ),
        fox_remote_formula="gd_residual_v1",
        fox_clr_rank=4,
        fox_clr_use_den_residual=True,
        fox_clr_remat_mode="off",
        fox_clr_residual_update_mode="additive",
        fox_clr_residual_forget_mode="global",
        fox_clr_state_write_topk=4,
        fox_clr_delta_target_mode="residual_to_coarse",
        fox_gd_residual_rank=2,
        fox_gd_residual_write_topk=2,
        fox_gd_residual_builder="grouped_chunk_torch_ref",
        fox_gd_residual_pack_mode="semivec_ref",
        fox_gd_residual_chunk_size=2,
        fox_gd_residual_mu_min_count=1.0,
        fox_gd_residual_addr_eps=1e-6,
        fox_gd_residual_den_eps=1e-6,
        fox_gd_residual_rho_eps=1e-12,
        fox_gd_residual_addr_init_rng_mode=expected.get(
            "fox_gd_residual_addr_init_rng_mode", "global"
        ),
        fox_gd_residual_addr_init_seed=expected.get("fox_gd_residual_addr_init_seed"),
        fox_gd_residual_beta_init=0.5,
        fox_gd_residual_beta_cap=None,
        fox_gd_residual_beta_cap_final=None,
        fox_gd_residual_beta_cap_release_start_train_steps=0,
        fox_gd_residual_beta_cap_release_end_train_steps=0,
        fox_gd_residual_beta_cap_eval_policy="final",
        fox_gd_residual_beta_control_mode=expected.get(
            "fox_gd_residual_beta_control_mode", "hard_cap"
        ),
        fox_gd_residual_beta_sigmoid_temp=1.0,
        fox_gd_residual_beta_low=expected.get("fox_gd_residual_beta_low"),
        fox_gd_residual_beta_high=expected.get("fox_gd_residual_beta_high"),
        fox_gd_residual_beta_low_final=None,
        fox_gd_residual_beta_high_final=None,
        fox_gd_residual_beta_band_release_start_train_steps=0,
        fox_gd_residual_beta_band_release_end_train_steps=0,
        fox_gd_residual_beta_band_eval_policy="final",
        fox_gd_residual_beta_band_schedule="smoothstep",
        fox_gd_residual_lambda_init=0.05,
        fox_gd_residual_lambda_floor=0.0,
        fox_gd_residual_write_strength_mode="renorm_topk",
        fox_gd_residual_write_strength_cap=expected.get("fox_gd_residual_write_strength_cap"),
        fox_gd_residual_write_strength_cap_mode=expected.get(
            "fox_gd_residual_write_strength_cap_mode", "hard"
        ),
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
        fox_gd_residual_write_q_alpha=expected.get("fox_gd_residual_write_q_alpha", 1.0),
        fox_gd_residual_m_norm_cap=None,
        fox_gd_residual_update_norm_cap=None,
        fox_gd_residual_norm_with_gain=False,
        fox_gd_residual_use_separate_addr_codebook=False,
        fox_gd_residual_addr_proj_orthogonal_init=bool(
            expected.get("fox_gd_residual_addr_proj_orthogonal_init", False)
        ),
        vq_score_mode="codebook_dot",
        vq_weight_mode="dense_softmax",
        vq_update_mode="grad",
        vq_softmax_tau=0.25,
        codebook_init_rng_mode=expected.get("codebook_init_rng_mode", "global"),
        codebook_init_seed=expected.get("codebook_init_seed"),
        vq_topk=4,
        gradient_accumulation_steps=1,
        train_batch_size=2,
        eval_batch_size=2,
        cache_dir="./data/flash_vqg",
        wandb_project="flash-vqg",
        wandb_entity="",
        max_epochs=1,
        metrics_white_list=[],
        validations_per_epoch=1,
        early_stopping_metric=None,
        early_stopping_threshold=None,
    )


def _render_generated_config_for_case(output_dir: Path, case: SmokeCase) -> dict[str, Any]:
    """Exercise generated-config rendering without launching training."""

    generated_dir = output_dir / "generated_configs"
    generated_dir.mkdir(parents=True, exist_ok=True)
    generated_path = generated_dir / f"{case.case_id}.py"
    text = _render_generated_config_text(case)
    generated_path.write_text(text + "\n", encoding="utf-8")

    marker_keys = (
        "fox_remote_read_topk_values",
        "fox_remote_read_topk_initial",
        "fox_remote_read_topk_final",
        "fox_remote_read_topk_release_start_train_steps",
        "fox_remote_read_topk_release_end_train_steps",
        "fox_remote_read_topk_schedule",
        "fox_remote_read_topk_eval_policy",
        "fox_gd_residual_write_q_alpha",
        "fox_gd_residual_addr_proj_orthogonal_init",
    )
    markers = {key: f"{key}=" in text for key in marker_keys}
    keyword_sets = _render_call_keyword_sets()
    main_call_keywords = {key: any(key in item for item in keyword_sets) for key in marker_keys}

    # Import the generated config to ensure Python syntax and effective configs are valid.
    module_name = f"generated_smoke_config_{case.case_id.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, generated_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    generated_configs = list(getattr(module, "configs"))
    generated_flash_kwargs = _flash_kwargs(generated_configs[0])

    checks = _check_expected(generated_flash_kwargs, case.expected)
    key_checks_passed = all(item["passed"] for item in checks)
    marker_checks_passed = all(markers.values())
    main_call_passed = all(main_call_keywords.values())
    return {
        "passed": key_checks_passed and marker_checks_passed and main_call_passed,
        "generated_path": str(generated_path),
        "markers": markers,
        "main_call_keywords": main_call_keywords,
        "render_call_count": len(keyword_sets),
        "checks": checks,
    }


def run_smoke(output_dir: Path, device: torch.device, skip_generated: bool) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_path = output_dir / "cases.jsonl"
    if cases_path.exists():
        cases_path.unlink()

    all_results = []
    for case in _cases():
        started = time.time()
        result: dict[str, Any] = {
            "case_id": case.case_id,
            "status": "unknown",
            "started_at_unix": started,
        }
        try:
            config = _build_case_config(case)
            flash_kwargs = _flash_kwargs(config)
            static_checks = _check_expected(flash_kwargs, case.expected)
            runtime = _runtime_smoke(config, case, device)
            generated = (
                {"skipped": True, "passed": True}
                if skip_generated
                else _render_generated_config_for_case(output_dir, case)
            )
            passed = (
                all(item["passed"] for item in static_checks)
                and runtime["passed"]
                and bool(generated["passed"])
            )
            result.update(
                {
                    "status": "passed" if passed else "failed",
                    "static_checks": static_checks,
                    "runtime": runtime,
                    "generated_config": generated,
                    "run_id": config.run_id,
                }
            )
        except Exception as exc:  # noqa: BLE001 - smoke should report all failures.
            result.update(
                {
                    "status": "failed",
                    "error": repr(exc),
                }
            )
        result["ended_at_unix"] = time.time()
        result["wall_clock_sec"] = result["ended_at_unix"] - started
        all_results.append(result)
        _append_jsonl(cases_path, result)

    passed_count = sum(item["status"] == "passed" for item in all_results)
    summary = {
        "status": "passed" if passed_count == len(all_results) else "failed",
        "passed": passed_count,
        "failed": len(all_results) - passed_count,
        "num_cases": len(all_results),
        "device": str(device),
        "output_dir": str(output_dir),
        "cases": all_results,
    }
    _write_json(output_dir / "summary.json", summary)
    (output_dir / "README.md").write_text(
        "# Config-to-runtime smoke output\n\n"
        f"- status: `{summary['status']}`\n"
        f"- cases: `{passed_count}/{len(all_results)}` passed\n"
        f"- device: `{device}`\n\n"
        "See `summary.json` and `cases.jsonl`.\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument(
        "--skip-generated-config-check",
        action="store_true",
        help="Skip run_flash_vqg_suite generated-config path check.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = _resolve_device(args.device)
    summary = run_smoke(
        output_dir=args.output_dir,
        device=device,
        skip_generated=bool(args.skip_generated_config_check),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
