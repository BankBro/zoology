from __future__ import annotations

import argparse
import json
import math
import os
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from einops import rearrange

from zoology.data.utils import prepare_data
from zoology.experiments.flash_vqg.flash_vqg_suite import build_configs
from zoology.model import LanguageModel
from zoology.train import compute_metrics
from zoology.utils import set_determinism


GD_DEBUG_METRICS = [
    "attn/gd_residual_debug_event_pack_wall_sec",
    "attn/gd_residual_debug_grouped_chunk_wall_sec",
    "attn/gd_residual_debug_phase2_residual_read_wall_sec",
    "attn/gd_residual_debug_event_count",
    "attn/gd_residual_debug_group_count",
    "attn/gd_residual_debug_max_events_per_group",
    "attn/gd_residual_debug_mean_events_per_group",
    "attn/gd_residual_debug_avg_events_per_group",
    "attn/gd_residual_debug_l_state_mean",
    "attn/gd_residual_debug_l_state_max",
    "attn/gd_residual_debug_l_state_frac_ge_0_1",
    "attn/gd_residual_debug_l_state_frac_ge_0_5",
    "attn/gd_residual_debug_l_state_frac_ge_1_0",
]

SHORT_RUN_METRICS = [
    "train/loss",
    "valid/loss",
    "valid/accuracy",
    "attn/nan_inf_count",
    "attn/den_min",
    "attn/o_remote_energy_ratio",
    "attn/gd_residual_inject_ratio",
    "attn/gd_residual_lambda_mean",
    "attn/gd_residual_write_strength_mean",
    "attn/gd_residual_m_norm_mean",
    "attn/gd_residual_m_norm_max",
    "attn/gd_residual_mu_valid_ratio",
    *GD_DEBUG_METRICS,
    "layer_*/attn/gd_residual_*",
    "valid/attn/nan_inf_count",
    "valid/attn/den_min",
    "valid/attn/o_remote_energy_ratio",
    "valid/attn/gd_residual_inject_ratio",
    "valid/attn/gd_residual_lambda_mean",
    "valid/attn/gd_residual_write_strength_mean",
    "valid/attn/gd_residual_m_norm_mean",
    "valid/attn/gd_residual_m_norm_max",
    "valid/attn/gd_residual_mu_valid_ratio",
    *[f"valid/{metric}" for metric in GD_DEBUG_METRICS],
    "valid/layer_*/attn/gd_residual_*",
]


@dataclass(frozen=True)
class VariantSpec:
    name: str
    if_remote_enabled: bool
    fox_remote_formula: str
    rank: int | None = None
    write_topk: int | None = None


VARIANTS: dict[str, VariantSpec] = {
    "gd_r16_wk4": VariantSpec("gd_r16_wk4", True, "gd_residual_v1", rank=16, write_topk=4),
    "a": VariantSpec("gd_r16_wk4", True, "gd_residual_v1", rank=16, write_topk=4),
    "gd_r16_wk2": VariantSpec("gd_r16_wk2", True, "gd_residual_v1", rank=16, write_topk=2),
    "b": VariantSpec("gd_r16_wk2", True, "gd_residual_v1", rank=16, write_topk=2),
    "gd_r8_wk4": VariantSpec("gd_r8_wk4", True, "gd_residual_v1", rank=8, write_topk=4),
    "c": VariantSpec("gd_r8_wk4", True, "gd_residual_v1", rank=8, write_topk=4),
    "local_only": VariantSpec("local_only", False, "legacy"),
    "legacy_fox": VariantSpec("legacy_fox", True, "legacy"),
}
MATRIX_VARIANTS = ["gd_r16_wk4", "gd_r16_wk2", "gd_r8_wk4", "local_only", "legacy_fox"]


def _env_bool(name: str, default: str = "0") -> bool:
    return str(os.environ.get(name, default)).strip().lower() in {"1", "true", "yes", "on"}


def _parse_csv_ints(raw: str) -> list[int]:
    values = [part.strip() for part in str(raw).split(",") if part.strip()]
    if not values:
        raise ValueError("CSV integer list must not be empty.")
    return [int(value) for value in values]


def _parse_read_topk(raw: str) -> int | None:
    normalized = str(raw).strip().lower()
    if normalized in {"dense", "none", "null"}:
        return None
    parsed = int(normalized)
    if parsed <= 0:
        raise ValueError("remote read topk must be positive or dense.")
    return parsed


def _read_topk_tag(value: int | None) -> str:
    return "dense" if value is None else f"top{int(value)}"


def resolve_variants(raw: str) -> list[VariantSpec]:
    names = [part.strip().lower() for part in str(raw).split(",") if part.strip()]
    if not names:
        raise ValueError("SHORT_RUN_VARIANT must not be empty.")
    if names == ["all"]:
        names = MATRIX_VARIANTS
    variants: list[VariantSpec] = []
    for name in names:
        if name not in VARIANTS:
            raise ValueError(f"unknown short-run variant: {name}")
        spec = VARIANTS[name]
        if spec.name not in [variant.name for variant in variants]:
            variants.append(spec)
    return variants


def _copy_model(value):
    return value.model_copy(deep=True) if hasattr(value, "model_copy") else value.copy(deep=True)


def _resize_data_budget(config, *, train_batches: int, batch_size: int, test_examples: int):
    config = _copy_model(config)
    train_per_segment = max(
        int(batch_size),
        math.ceil(int(train_batches) / max(len(config.data.train_configs), 1)) * int(batch_size),
    )
    config.data.train_configs = [
        _copy_model(segment).copy(update={"num_examples": train_per_segment})
        if not hasattr(segment, "model_copy")
        else segment.model_copy(update={"num_examples": train_per_segment})
        for segment in config.data.train_configs
    ]
    config.data.test_configs = [
        _copy_model(segment).copy(update={"num_examples": int(test_examples)})
        if not hasattr(segment, "model_copy")
        else segment.model_copy(update={"num_examples": int(test_examples)})
        for segment in config.data.test_configs
    ]
    return config


def build_short_run_config(args: argparse.Namespace, variant: VariantSpec, seed: int):
    read_topk = _parse_read_topk(args.remote_read_topk)
    gd_kwargs: dict[str, Any] = {}
    if variant.fox_remote_formula == "gd_residual_v1":
        gd_kwargs.update(
            fox_gd_residual_rank=int(variant.rank),
            fox_gd_residual_write_topk=int(variant.write_topk),
            fox_gd_residual_builder=str(args.builder),
            fox_gd_residual_pack_mode=str(args.pack_mode),
            fox_gd_residual_chunk_size=int(args.chunk_size),
            fox_gd_residual_mu_min_count=float(args.fox_gd_residual_mu_min_count),
            fox_gd_residual_beta_init=0.5,
            fox_gd_residual_lambda_init=0.05,
            vq_score_mode="codebook_dot",
            vq_weight_mode="dense_softmax",
            vq_update_mode="grad",
            vq_topk=max(int(variant.write_topk), 4),
        )

    configs = build_configs(
        sweep_id="gd-residual-v1-short-run",
        flash_backend="torch",
        logger_backend="none",
        include_gdn=False,
        block_len=int(args.block_len),
        local_num_blocks=int(args.local_num_blocks),
        dmodels=[int(args.d_model)],
        learning_rates=[float(args.learning_rate)],
        if_remote_enabled=bool(variant.if_remote_enabled),
        train_batch_order=str(args.train_batch_order),
        seed_values=[int(seed)],
        data_seed=int(args.data_seed),
        num_codebook_vectors_values=[int(args.num_codebook_vectors)],
        fox_remote_path_backend="torch",
        fox_remote_read_topk_values=[read_topk],
        fox_remote_formula=variant.fox_remote_formula,
        gradient_accumulation_steps=1,
        train_batch_size=int(args.batch_size),
        eval_batch_size=int(args.eval_batch_size),
        cache_dir=str(args.cache_dir),
        metrics_white_list=SHORT_RUN_METRICS,
        experiment_part="gd_residual_v1_mqar_readiness",
        experiment_mode="short_run",
        **gd_kwargs,
    )
    if len(configs) != 1:
        raise RuntimeError(f"expected one config for {variant.name}, got {len(configs)}")
    config = _resize_data_budget(
        configs[0],
        train_batches=int(args.train_batches),
        batch_size=int(args.batch_size),
        test_examples=int(args.test_examples_per_segment),
    )
    config.checkpoint.enabled = False
    config.run_id = (
        f"short-run-{variant.name}-s{int(seed)}-d{int(args.data_seed)}"
        f"-rread-{_read_topk_tag(read_topk)}-b{int(args.batch_size)}"
        f"-steps{int(args.train_batches)}"
    )
    return config


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _memory_snapshot(device: torch.device) -> dict[str, int | None]:
    if device.type != "cuda":
        return {"peak_allocated_bytes": None, "peak_reserved_bytes": None}
    return {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _collect_model_scalar_metrics(model: nn.Module) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for module in model.modules():
        getter = getattr(module, "get_scalar_metrics", None)
        if getter is None:
            continue
        for key, value in (getter() or {}).items():
            if math.isfinite(float(value)):
                metrics[str(key)] = float(value)
    return metrics


def _set_dense_teacher_runtime(model: nn.Module, targets: torch.Tensor) -> None:
    runtime = {"teacher_target_mask": (targets != -100).detach()}
    for module in model.modules():
        setter = getattr(module, "set_dense_teacher_runtime", None)
        if setter is not None:
            setter(runtime)


def _clear_dense_teacher_runtime(model: nn.Module) -> None:
    for module in model.modules():
        clearer = getattr(module, "clear_dense_teacher_runtime", None)
        if clearer is not None:
            clearer()


def _auxiliary_loss(model: nn.Module, device: torch.device) -> torch.Tensor:
    losses = []
    for module in model.modules():
        getter = getattr(module, "get_auxiliary_loss", None)
        if getter is not None:
            losses.append(getter())
    if not losses:
        return torch.zeros((), device=device)
    return sum(losses)


def _compute_loss(
    model: nn.Module,
    loss_fn: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    include_auxiliary_loss: bool,
):
    _set_dense_teacher_runtime(model, targets)
    try:
        logits = model(inputs)
        loss = loss_fn(rearrange(logits, "... c -> (...) c"), targets.flatten())
        if include_auxiliary_loss:
            loss = loss + _auxiliary_loss(model, inputs.device)
        preds = logits.argmax(dim=-1)
    finally:
        _clear_dense_teacher_runtime(model)
    return loss, preds


def _split_metrics(metrics: dict[str, float]) -> dict[str, dict[str, float]]:
    return {
        "gd_residual_metrics": {
            key: value
            for key, value in metrics.items()
            if "gd_residual" in key and not key.startswith("layer_") and "/layer_" not in key
        },
        "layer_metrics": {
            key: value
            for key, value in metrics.items()
            if key.startswith("layer_") or key.startswith("valid/layer_")
        },
        "event_diagnostics": {
            key: value
            for key, value in metrics.items()
            if "debug_event" in key or "events_per_group" in key
        },
        "l_state_diagnostics": {
            key: value for key, value in metrics.items() if "debug_l_state" in key
        },
    }


def _has_nan_or_inf(loss_value: float | None, metrics: dict[str, float]) -> bool:
    values = list(metrics.values())
    if loss_value is not None:
        values.append(float(loss_value))
    return any(not math.isfinite(float(value)) for value in values)


def _append_record(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _check_gd_init(model: nn.Module, variant: VariantSpec) -> dict[str, Any]:
    if variant.fox_remote_formula != "gd_residual_v1":
        return {"checked": False, "reason": "not_gd_residual_v1"}
    checks = []
    for module in model.modules():
        attn = getattr(module, "attn", None)
        beta_proj = getattr(attn, "fox_gd_residual_beta_proj", None)
        lambda_proj = getattr(attn, "fox_gd_residual_lambda_proj", None)
        if beta_proj is None or lambda_proj is None:
            continue
        beta = torch.sigmoid(beta_proj.bias.detach()).float().cpu()
        lam = torch.sigmoid(lambda_proj.bias.detach()).float().cpu()
        beta_ok = bool(torch.allclose(beta, torch.full_like(beta, 0.5), atol=1e-6, rtol=1e-6))
        lambda_ok = bool(torch.allclose(lam, torch.full_like(lam, 0.05), atol=1e-6, rtol=1e-6))
        checks.append(
            {
                "beta_mean": float(beta.mean().item()),
                "lambda_mean": float(lam.mean().item()),
                "beta_ok": beta_ok,
                "lambda_ok": lambda_ok,
            }
        )
    if not checks:
        raise RuntimeError("gd_residual_v1 init check found no FlashVQGMixer projections.")
    if not all(item["beta_ok"] and item["lambda_ok"] for item in checks):
        raise RuntimeError(f"gd_residual_v1 init check failed: {checks}")
    return {"checked": True, "layers": checks}


def _evaluate(
    *,
    model: nn.Module,
    dataloader,
    loss_fn: nn.Module,
    device: torch.device,
    max_batches: int,
) -> tuple[dict[str, float], float, float, float]:
    model.eval()
    loss_values: list[float] = []
    results: list[dict[str, Any]] = []
    metric_buckets: dict[str, list[float]] = {}
    _sync(device)
    start = time.perf_counter()
    with torch.no_grad():
        for batch_idx, (inputs, targets, slices) in enumerate(dataloader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            inputs = inputs.to(device)
            targets = targets.to(device)
            loss, preds = _compute_loss(
                model,
                loss_fn,
                inputs,
                targets,
                include_auxiliary_loss=False,
            )
            loss_values.append(float(loss.detach().cpu().item()))
            results.extend(compute_metrics(preds.cpu(), targets.cpu(), slices))
            for key, value in _collect_model_scalar_metrics(model).items():
                metric_buckets.setdefault(key, []).append(float(value))
    _sync(device)
    elapsed = time.perf_counter() - start
    valid_loss = float(sum(loss_values) / max(len(loss_values), 1))
    valid_accuracy = float(
        sum(float(row["accuracy"]) for row in results) / max(len(results), 1)
    )
    metrics = {
        f"valid/{key}": float(sum(values) / len(values))
        for key, values in metric_buckets.items()
        if values
    }
    return metrics, valid_loss, valid_accuracy, elapsed


def _append_valid_record(
    *,
    records_path: Path,
    model: nn.Module,
    dataloader,
    loss_fn: nn.Module,
    device: torch.device,
    max_batches: int,
    variant: VariantSpec,
    seed: int,
    run_id: str,
    step: int,
    phase: str,
    last_train_loss: float | None,
) -> tuple[dict[str, float], float, float, float, bool]:
    valid_metrics, valid_loss, valid_accuracy, valid_sec = _evaluate(
        model=model,
        dataloader=dataloader,
        loss_fn=loss_fn,
        device=device,
        max_batches=max_batches,
    )
    nan_or_inf = _has_nan_or_inf(valid_loss, valid_metrics)
    record = {
        "record_type": "valid",
        "phase": phase,
        "variant": variant.name,
        "seed": int(seed),
        "run_id": run_id,
        "step": int(step),
        "train_loss": last_train_loss,
        "valid_loss": valid_loss,
        "valid_accuracy": valid_accuracy,
        "valid_sec": valid_sec,
        "metrics_collect_sec": 0.0,
        "microbatch_sec": None,
        "step_sec": None,
        "memory": _memory_snapshot(device),
        "metrics": valid_metrics,
        "has_nan_or_inf": nan_or_inf,
        **_split_metrics(valid_metrics),
    }
    _append_record(records_path, record)
    return valid_metrics, valid_loss, valid_accuracy, valid_sec, nan_or_inf


def run_one(args: argparse.Namespace, variant: VariantSpec, seed: int, records_path: Path) -> dict[str, Any]:
    config = build_short_run_config(args, variant, seed)
    set_determinism(int(seed), deterministic=False)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    train_dataloader, test_dataloader = prepare_data(config.data)
    model = LanguageModel(config.model).to(device)
    init_check = _check_gd_init(model, variant)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    loss_fn = nn.CrossEntropyLoss()
    train_iter = iter(train_dataloader)

    status = "completed"
    last_train_loss: float | None = None
    last_metrics: dict[str, float] = {}
    train_records = 0
    valid_metrics: dict[str, float] = {}
    valid_loss = None
    valid_accuracy = None
    valid_sec = None

    (
        valid_metrics,
        valid_loss,
        valid_accuracy,
        valid_sec,
        nan_or_inf,
    ) = _append_valid_record(
        records_path=records_path,
        model=model,
        dataloader=test_dataloader,
        loss_fn=loss_fn,
        device=device,
        max_batches=int(args.eval_batches),
        variant=variant,
        seed=seed,
        run_id=config.run_id,
        step=0,
        phase="initial_valid",
        last_train_loss=None,
    )
    if nan_or_inf:
        status = "failed_nan_or_inf"
    else:
        model.train()

    for step_idx in range(int(args.train_batches)):
        if status != "completed":
            break
        try:
            inputs, targets, _slices = next(train_iter)
        except StopIteration:
            status = "stopped_no_more_train_batches"
            break

        inputs = inputs.to(device)
        targets = targets.to(device)
        _sync(device)
        total_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)

        forward_start = time.perf_counter()
        loss, _preds = _compute_loss(
            model,
            loss_fn,
            inputs,
            targets,
            include_auxiliary_loss=True,
        )
        _sync(device)
        forward_sec = time.perf_counter() - forward_start

        backward_start = time.perf_counter()
        loss.backward()
        _sync(device)
        backward_sec = time.perf_counter() - backward_start

        optimizer_start = time.perf_counter()
        optimizer.step()
        _sync(device)
        optimizer_sec = time.perf_counter() - optimizer_start

        metrics_start = time.perf_counter()
        scalar_metrics = _collect_model_scalar_metrics(model)
        _sync(device)
        metrics_collect_sec = time.perf_counter() - metrics_start

        train_loss = float(loss.detach().cpu().item())
        last_train_loss = train_loss
        last_metrics = scalar_metrics
        nan_or_inf = _has_nan_or_inf(train_loss, scalar_metrics)
        if nan_or_inf:
            status = "failed_nan_or_inf"
        record = {
            "record_type": "train",
            "variant": variant.name,
            "seed": int(seed),
            "run_id": config.run_id,
            "step": step_idx,
            "train_loss": train_loss,
            "valid_loss": None,
            "valid_accuracy": None,
            "forward_sec": forward_sec,
            "backward_sec": backward_sec,
            "optimizer_sec": optimizer_sec,
            "metrics_collect_sec": metrics_collect_sec,
            "microbatch_sec": time.perf_counter() - total_start,
            "step_sec": time.perf_counter() - total_start,
            "memory": _memory_snapshot(device),
            "metrics": scalar_metrics,
            "has_nan_or_inf": nan_or_inf,
            **_split_metrics(scalar_metrics),
        }
        _append_record(records_path, record)
        train_records += 1
        if nan_or_inf:
            break
        if (
            int(args.valid_every) > 0
            and train_records % int(args.valid_every) == 0
            and train_records < int(args.train_batches)
        ):
            (
                valid_metrics,
                valid_loss,
                valid_accuracy,
                valid_sec,
                nan_or_inf,
            ) = _append_valid_record(
                records_path=records_path,
                model=model,
                dataloader=test_dataloader,
                loss_fn=loss_fn,
                device=device,
                max_batches=int(args.eval_batches),
                variant=variant,
                seed=seed,
                run_id=config.run_id,
                step=train_records,
                phase="periodic_valid",
                last_train_loss=last_train_loss,
            )
            model.train()
            if nan_or_inf:
                status = "failed_nan_or_inf"
                break

    if status in {"completed", "stopped_no_more_train_batches"}:
        (
            valid_metrics,
            valid_loss,
            valid_accuracy,
            valid_sec,
            nan_or_inf,
        ) = _append_valid_record(
            records_path=records_path,
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device,
            max_batches=int(args.eval_batches),
            variant=variant,
            seed=seed,
            run_id=config.run_id,
            step=train_records,
            phase="final_valid",
            last_train_loss=last_train_loss,
        )
        if nan_or_inf:
            status = "failed_nan_or_inf"

    return {
        "variant": asdict(variant),
        "seed": int(seed),
        "run_id": config.run_id,
        "status": status,
        "train_records": train_records,
        "final_train_loss": last_train_loss,
        "valid_loss": valid_loss,
        "valid_accuracy": valid_accuracy,
        "valid_sec": valid_sec,
        "init_check": init_check,
        "memory": _memory_snapshot(device),
        "last_train_metrics": last_metrics,
        "valid_metrics": valid_metrics,
    }


def run_short_run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    summary_path = output_dir / "summary.json"
    if records_path.exists() and not bool(args.append):
        records_path.unlink()

    diagnostics_enabled = bool(args.enable_gd_diagnostics)
    old_diag = os.environ.get("FOX_GD_RESIDUAL_PROFILE_DIAGNOSTICS")
    os.environ["FOX_GD_RESIDUAL_PROFILE_DIAGNOSTICS"] = "1" if diagnostics_enabled else "0"
    variants = resolve_variants(args.variant)
    seeds = _parse_csv_ints(args.seeds)

    runs: list[dict[str, Any]] = []
    try:
        for variant in variants:
            for seed in seeds:
                try:
                    runs.append(run_one(args, variant, seed, records_path))
                except torch.cuda.OutOfMemoryError as exc:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    failure = _failure_record(variant, seed, exc)
                    runs.append(failure)
                    _append_record(records_path, failure)
                except BaseException as exc:
                    failure = _failure_record(variant, seed, exc)
                    runs.append(failure)
                    _append_record(records_path, failure)
    finally:
        if old_diag is None:
            os.environ.pop("FOX_GD_RESIDUAL_PROFILE_DIAGNOSTICS", None)
        else:
            os.environ["FOX_GD_RESIDUAL_PROFILE_DIAGNOSTICS"] = old_diag

    summary = {
        "output_dir": str(output_dir),
        "records_path": str(records_path),
        "diagnostics_enabled": diagnostics_enabled,
        "train_batches": int(args.train_batches),
        "valid_every": int(args.valid_every),
        "fox_gd_residual_mu_min_count": float(args.fox_gd_residual_mu_min_count),
        "seeds": seeds,
        "variants": [asdict(variant) for variant in variants],
        "runs": runs,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary_path={summary_path}")
    print(f"records_path={records_path}")
    return summary


def _failure_record(variant: VariantSpec, seed: int, exc: BaseException) -> dict[str, Any]:
    return {
        "record_type": "run_failure",
        "variant": variant.name,
        "seed": int(seed),
        "status": "failed",
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(limit=12),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run short MQAR gd_residual_v1 readiness checks.")
    parser.add_argument("--train-batches", type=int, default=int(os.environ.get("SHORT_RUN_TRAIN_BATCHES", "20")))
    parser.add_argument("--seeds", default=os.environ.get("SHORT_RUN_SEEDS", "123,124"))
    parser.add_argument("--variant", default=os.environ.get("SHORT_RUN_VARIANT", "gd_r16_wk4"))
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("SHORT_RUN_OUTPUT_DIR", "tmp/20260425-gd-residual-v1-short-run"),
    )
    parser.add_argument(
        "--enable-gd-diagnostics",
        action="store_true",
        default=_env_bool("PROFILE_ENABLE_GD_DIAGNOSTICS", "0"),
    )
    parser.add_argument("--append", action="store_true", default=False)
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("TRAIN_BATCH_SIZE", "8")))
    parser.add_argument("--eval-batch-size", type=int, default=int(os.environ.get("EVAL_BATCH_SIZE", "8")))
    parser.add_argument(
        "--test-examples-per-segment",
        type=int,
        default=int(os.environ.get("SHORT_RUN_TEST_EXAMPLES_PER_SEGMENT", os.environ.get("EVAL_BATCH_SIZE", "8"))),
    )
    parser.add_argument("--eval-batches", type=int, default=int(os.environ.get("SHORT_RUN_EVAL_BATCHES", "0")))
    parser.add_argument("--valid-every", type=int, default=int(os.environ.get("SHORT_RUN_VALID_EVERY", "0")))
    parser.add_argument("--d-model", type=int, default=int(os.environ.get("DMODEL", "128")))
    parser.add_argument(
        "--num-codebook-vectors",
        type=int,
        default=int(os.environ.get("NUM_CODEBOOK_VECTORS", "128")),
    )
    parser.add_argument("--block-len", type=int, default=int(os.environ.get("BLOCK_LEN", "32")))
    parser.add_argument("--local-num-blocks", type=int, default=int(os.environ.get("LOCAL_NUM_BLOCKS", "2")))
    parser.add_argument("--learning-rate", type=float, default=float(os.environ.get("LR", "1e-3")))
    parser.add_argument("--weight-decay", type=float, default=float(os.environ.get("WEIGHT_DECAY", "0.1")))
    parser.add_argument("--data-seed", type=int, default=int(os.environ.get("DATA_SEED", "123")))
    parser.add_argument("--remote-read-topk", default=os.environ.get("FOX_REMOTE_READ_TOPK", "2"))
    parser.add_argument(
        "--builder",
        choices=["token_step_ref", "grouped_chunk_torch_ref"],
        default=os.environ.get("FOX_GD_RESIDUAL_BUILDER", "grouped_chunk_torch_ref"),
    )
    parser.add_argument(
        "--pack-mode",
        choices=["loop_ref", "semivec_ref"],
        default=os.environ.get("FOX_GD_RESIDUAL_PACK_MODE", "semivec_ref"),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=int(os.environ.get("FOX_GD_RESIDUAL_CHUNK_SIZE", "64")),
    )
    parser.add_argument(
        "--fox-gd-residual-mu-min-count",
        type=float,
        default=float(os.environ.get("FOX_GD_RESIDUAL_MU_MIN_COUNT", "1.0")),
    )
    parser.add_argument("--cache-dir", default=os.environ.get("CACHE_DIR", "./data/flash_vqg"))
    parser.add_argument("--train-batch-order", default=os.environ.get("TRAIN_BATCH_ORDER", "global_shuffle"))
    parser.add_argument("--device", default=os.environ.get("SHORT_RUN_DEVICE", "auto"))
    return parser.parse_args()


if __name__ == "__main__":
    run_short_run(parse_args())
