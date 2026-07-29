#!/usr/bin/env python3
from __future__ import annotations

import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

LOCAL_DIR = Path(__file__).resolve().parent
if str(LOCAL_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_DIR))

os.environ.setdefault("TRITON_F32_DEFAULT", "ieee")
os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
os.environ.setdefault("TORCH_DETERMINISTIC", "0")

import torch
import torch.nn.functional as functional

from common import EXPERIMENT_ID, atomic_write_json, run_root, run_tag, utc_now
from experiment import BASE, build_config, configure_numerics, environment_metadata


CHECKPOINTS = (1, 16, 32)
SEED = 124


def _tensor_map(values) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().float().cpu().clone()
        for name, value in values
        if value is not None
    }


def _optimizer_map(model, optimizer) -> dict[str, torch.Tensor]:
    result = {}
    for name, parameter in model.named_parameters():
        for state_name, value in optimizer.state.get(parameter, {}).items():
            if torch.is_tensor(value):
                result[f"{name}/{state_name}"] = value.detach().float().cpu().clone()
    return result


def _runtime_summary(model) -> dict[str, Any]:
    result = {}
    for name, module in model.named_modules():
        getter = getattr(module, "get_training_runtime_state", None)
        if getter is None:
            continue
        state = getter() or {}
        if "fox_gd_residual_train_forward_count" not in state:
            continue
        result[name] = {
            "forward_count": int(state["fox_gd_residual_train_forward_count"]),
            "audit": state.get("fox_gd_residual_triton_runtime_audit") or {},
        }
    return result


def _run_variant(variant: str) -> dict[str, Any]:
    config = build_config(variant, SEED, "formal")
    BASE.set_determinism(SEED, deterministic=False)
    train_loader, _ = BASE.prepare_data(config.data)
    sampler = getattr(train_loader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(0)
    iterator = iter(train_loader)
    model = BASE._flash_model_from_config(config).train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    BASE.set_determinism(SEED, deterministic=False)
    snapshots = {}
    losses = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    for optimizer_step in range(1, max(CHECKPOINTS) + 1):
        optimizer.zero_grad(set_to_none=True)
        micro_losses = []
        last_hidden = None
        for _ in range(config.gradient_accumulation_steps):
            inputs_cpu, targets_cpu, _ = next(iterator)
            inputs = inputs_cpu.to("cuda")
            targets = targets_cpu.to("cuda")
            BASE._set_dense_teacher(model, targets)
            try:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    hidden = model.backbone(inputs)
                    logits = model.lm_head(hidden)
                    loss = functional.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        targets.flatten(),
                    )
                    loss = loss + BASE._auxiliary_loss(model, inputs.device)
            finally:
                BASE._clear_dense_teacher(model)
            (loss / config.gradient_accumulation_steps).backward()
            micro_losses.append(loss.detach().float())
            last_hidden = hidden.detach().float().cpu()
        gradients = None
        if optimizer_step == 1:
            gradients = _tensor_map(
                (name, parameter.grad)
                for name, parameter in model.named_parameters()
            )
        optimizer.step()
        step_loss = float(torch.stack(micro_losses).mean().cpu().item())
        losses.append(step_loss)
        if optimizer_step in CHECKPOINTS:
            snapshots[str(optimizer_step)] = {
                "loss": step_loss,
                "hidden": last_hidden if optimizer_step == 1 else None,
                "gradients": gradients,
                "parameters": _tensor_map(model.named_parameters()),
                "optimizer": _optimizer_map(model, optimizer),
                "runtime": _runtime_summary(model),
            }
        if optimizer_step in CHECKPOINTS:
            print(
                json.dumps(
                    {
                        "event": "trajectory_progress",
                        "variant": variant,
                        "step": optimizer_step,
                        "loss": step_loss,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    torch.cuda.synchronize()
    result = {
        "variant": variant,
        "losses": losses,
        "snapshots": snapshots,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
    }
    del model, optimizer, iterator
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _tensor_comparison(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    difference = candidate.float() - reference.float()
    reference_norm = reference.float().norm().clamp_min(1e-24)
    return {
        "allclose": bool(torch.allclose(candidate, reference, atol=1e-5, rtol=1e-4)),
        "max_abs": float(difference.abs().max().item()) if difference.numel() else 0.0,
        "relative_l2": float((difference.norm() / reference_norm).item()),
        "finite": bool(torch.isfinite(candidate).all() and torch.isfinite(reference).all()),
    }


def _map_comparison(reference: dict[str, torch.Tensor], candidate: dict[str, torch.Tensor]) -> dict[str, Any]:
    reference_keys, candidate_keys = set(reference), set(candidate)
    rows = [
        {"name": key, **_tensor_comparison(reference[key], candidate[key])}
        for key in sorted(reference_keys & candidate_keys)
    ]
    return {
        "allclose": all(row["allclose"] for row in rows)
        and reference_keys == candidate_keys,
        "finite": all(row["finite"] for row in rows),
        "missing_from_a1": sorted(reference_keys - candidate_keys),
        "missing_from_a0": sorted(candidate_keys - reference_keys),
        "max_abs": max((row["max_abs"] for row in rows), default=0.0),
        "max_relative_l2": max((row["relative_l2"] for row in rows), default=0.0),
        "worst_relative_l2": sorted(rows, key=lambda row: row["relative_l2"], reverse=True)[:10],
    }


def _runtime_checks(a0: dict[str, Any], a1: dict[str, Any]) -> dict[str, Any]:
    same_modules = set(a0) == set(a1) and bool(a0)
    forward_equal = same_modules and all(
        a0[name]["forward_count"] == a1[name]["forward_count"] for name in a0
    )
    a0_recompute = sum(int(value["audit"].get("selected_recompute_calls", 0)) for value in a0.values())
    a1_recompute = sum(int(value["audit"].get("selected_recompute_calls", 0)) for value in a1.values())
    fallbacks = sum(
        int(value["audit"].get(key, 0))
        for value in list(a0.values()) + list(a1.values())
        for key in (
            "grouped_fallbacks",
            "selected_fallbacks",
            "grouped_recompute_fallbacks",
            "selected_recompute_fallbacks",
        )
    )
    return {
        "same_modules": same_modules,
        "forward_counts_equal": forward_equal,
        "a0_selected_recompute_calls": a0_recompute,
        "a1_selected_recompute_calls": a1_recompute,
        "fallbacks": fallbacks,
        "passed": forward_equal and a0_recompute == 0 and a1_recompute > 0 and fallbacks == 0,
    }


def run() -> dict[str, Any]:
    configure_numerics()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    a0 = _run_variant("a0-off")
    a1 = _run_variant("a1-post-phase1")
    comparisons = {}
    for step in CHECKPOINTS:
        key = str(step)
        left, right = a0["snapshots"][key], a1["snapshots"][key]
        comparison = {
            "loss_abs": abs(left["loss"] - right["loss"]),
            "parameters": _map_comparison(left["parameters"], right["parameters"]),
            "optimizer": _map_comparison(left["optimizer"], right["optimizer"]),
            "runtime": _runtime_checks(left["runtime"], right["runtime"]),
        }
        if step == 1:
            comparison["hidden"] = _tensor_comparison(left["hidden"], right["hidden"])
            comparison["gradients"] = _map_comparison(left["gradients"], right["gradients"])
        comparisons[key] = comparison

    first = comparisons["1"]
    first_step_passed = (
        first["loss_abs"] <= 1e-5
        and first["hidden"]["allclose"]
        and first["gradients"]["allclose"]
        and first["parameters"]["allclose"]
        and first["optimizer"]["allclose"]
        and first["runtime"]["passed"]
    )
    later_diagnostic_passed = all(
        math.isfinite(comparisons[key]["loss_abs"])
        and comparisons[key]["parameters"]["finite"]
        and comparisons[key]["optimizer"]["finite"]
        and comparisons[key]["runtime"]["passed"]
        for key in ("16", "32")
    )
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "run_tag": run_tag(),
        "status": "passed" if first_step_passed and later_diagnostic_passed else "failed",
        "recorded_at_utc": utc_now(),
        "environment": environment_metadata(),
        "seed": SEED,
        "precision": "bf16",
        "gradient_accumulation_steps": 4,
        "checkpoints": list(CHECKPOINTS),
        "first_step_hard_gate": first_step_passed,
        "later_diagnostic_gate": later_diagnostic_passed,
        "a0": {key: value for key, value in a0.items() if key != "snapshots"},
        "a1": {key: value for key, value in a1.items() if key != "snapshots"},
        "comparisons": comparisons,
    }
    path = run_root() / "trajectory" / "summary.json"
    atomic_write_json(path, payload)
    print(json.dumps({"status": payload["status"], "summary": str(path)}, sort_keys=True))
    return payload


if __name__ == "__main__":
    result = run()
    raise SystemExit(0 if result["status"] == "passed" else 1)
