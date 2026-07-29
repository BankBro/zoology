#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

LOCAL_DIR = Path(__file__).resolve().parent
if str(LOCAL_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_DIR))

os.environ.setdefault("TRITON_F32_DEFAULT", "ieee")
os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
os.environ.setdefault("TORCH_DETERMINISTIC", "0")

import torch
import torch.nn.functional as functional

from common import PYTHON, VARIANTS, atomic_write_json, load_json, run_root, utc_now
from experiment import BASE, build_config, configure_numerics, environment_metadata


SEED = 124
LOCKSTEP_STEPS = 128
FRESH_STEPS = 32


def _capture_rng() -> tuple[torch.Tensor, list[torch.Tensor]]:
    return torch.random.get_rng_state(), torch.cuda.get_rng_state_all()


def _restore_rng(state: tuple[torch.Tensor, list[torch.Tensor]]) -> None:
    torch.random.set_rng_state(state[0])
    torch.cuda.set_rng_state_all(state[1])


def _model_and_optimizer(variant: str):
    config = build_config(variant, SEED, "formal")
    BASE.set_determinism(SEED, deterministic=False)
    model = BASE._flash_model_from_config(config).train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    return config, model, optimizer


def _batches(config: Any, count: int):
    BASE.set_determinism(SEED, deterministic=False)
    train_loader, _ = BASE.prepare_data(config.data)
    sampler = getattr(train_loader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(0)
    iterator = iter(train_loader)
    return [next(iterator)[:2] for _ in range(count)]


def _microbatch(model, config: Any, batch, rng_state):
    _restore_rng(rng_state)
    inputs_cpu, targets_cpu = batch
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
    return loss.detach(), hidden.detach(), logits.detach()


def _tensor_delta(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    if torch.equal(left, right):
        return {"equal": True, "max_abs": 0.0, "relative_l2": 0.0}
    left_f = left.detach().float()
    difference = right.detach().float() - left_f
    denominator = left_f.norm().clamp_min(1e-30)
    return {
        "equal": False,
        "max_abs": float(difference.abs().max().item()),
        "relative_l2": float((difference.norm() / denominator).item()),
    }


def _map_delta(
    left_values: Iterable[tuple[str, torch.Tensor | None]],
    right_values: Iterable[tuple[str, torch.Tensor | None]],
) -> dict[str, Any]:
    left = {name: value for name, value in left_values if value is not None}
    right = {name: value for name, value in right_values if value is not None}
    rows = []
    for name in sorted(set(left) & set(right)):
        row = {"name": name, **_tensor_delta(left[name], right[name])}
        if not row["equal"]:
            rows.append(row)
    rows.sort(key=lambda row: (row["max_abs"], row["relative_l2"]), reverse=True)
    return {
        "equal": not rows and set(left) == set(right),
        "different_tensor_count": len(rows),
        "missing_left": sorted(set(right) - set(left)),
        "missing_right": sorted(set(left) - set(right)),
        "worst": rows[:8],
    }


def _gradients(model):
    return ((name, parameter.grad) for name, parameter in model.named_parameters())


def _optimizer_tensors(model, optimizer):
    for name, parameter in model.named_parameters():
        for state_name, value in optimizer.state.get(parameter, {}).items():
            if torch.is_tensor(value):
                yield f"{name}/{state_name}", value


def _tensor_map_hash(values: Iterable[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(values, key=lambda item: item[0]):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


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


def _runtime_gate(a0: dict[str, Any], a1: dict[str, Any]) -> dict[str, Any]:
    same_modules = set(a0) == set(a1) and bool(a0)
    forward_equal = same_modules and all(
        a0[name]["forward_count"] == a1[name]["forward_count"] for name in a0
    )
    a0_recompute = sum(
        int(value["audit"].get("selected_recompute_calls", 0))
        for value in a0.values()
    )
    a1_recompute = sum(
        int(value["audit"].get("selected_recompute_calls", 0))
        for value in a1.values()
    )
    fallbacks = sum(
        int(value["audit"].get(key, 0))
        for value in (*a0.values(), *a1.values())
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


def _state_identity(model, optimizer) -> dict[str, str]:
    return {
        "model_state_sha256": BASE._state_dict_hash(model.state_dict()),
        "optimizer_state_sha256": _tensor_map_hash(
            _optimizer_tensors(model, optimizer)
        ),
    }


def _lockstep() -> dict[str, Any]:
    a0_config, a0_model, a0_optimizer = _model_and_optimizer("a0-fixed-off")
    a1_config, a1_model, a1_optimizer = _model_and_optimizer(
        "a1-fixed-post-phase1"
    )
    batches = _batches(
        a0_config,
        LOCKSTEP_STEPS * a0_config.gradient_accumulation_steps,
    )
    started = time.perf_counter()
    batch_index = 0
    first_divergence = None
    torch.cuda.reset_peak_memory_stats()

    for step in range(1, LOCKSTEP_STEPS + 1):
        a0_optimizer.zero_grad(set_to_none=True)
        a1_optimizer.zero_grad(set_to_none=True)
        for microbatch in range(1, a0_config.gradient_accumulation_steps + 1):
            rng_before = _capture_rng()
            a0_loss, a0_hidden, a0_logits = _microbatch(
                a0_model, a0_config, batches[batch_index], rng_before
            )
            rng_after = _capture_rng()
            a1_loss, a1_hidden, a1_logits = _microbatch(
                a1_model, a1_config, batches[batch_index], rng_before
            )
            _restore_rng(rng_after)
            comparison = {
                "loss": _tensor_delta(a0_loss, a1_loss),
                "hidden": _tensor_delta(a0_hidden, a1_hidden),
                "logits": _tensor_delta(a0_logits, a1_logits),
                "gradients": _map_delta(_gradients(a0_model), _gradients(a1_model)),
            }
            if not all(value["equal"] for value in comparison.values()):
                first_divergence = {
                    "step": step,
                    "microbatch": microbatch,
                    **comparison,
                }
                break
            batch_index += 1
        if first_divergence is not None:
            break
        a0_optimizer.step()
        a1_optimizer.step()
        parameters = _map_delta(
            a0_model.named_parameters(), a1_model.named_parameters()
        )
        optimizer = _map_delta(
            _optimizer_tensors(a0_model, a0_optimizer),
            _optimizer_tensors(a1_model, a1_optimizer),
        )
        if not parameters["equal"] or not optimizer["equal"]:
            first_divergence = {
                "step": step,
                "microbatch": None,
                "parameters": parameters,
                "optimizer": optimizer,
            }
            break
        if step in {1, 16, 32, 64, 128}:
            print(
                json.dumps(
                    {"event": "lockstep_progress", "step": step},
                    sort_keys=True,
                ),
                flush=True,
            )

    runtime = _runtime_gate(
        _runtime_summary(a0_model), _runtime_summary(a1_model)
    )
    identities = {
        "a0-fixed-off": _state_identity(a0_model, a0_optimizer),
        "a1-fixed-post-phase1": _state_identity(a1_model, a1_optimizer),
    }
    passed = (
        first_divergence is None
        and runtime["passed"]
        and identities["a0-fixed-off"] == identities["a1-fixed-post-phase1"]
    )
    payload = {
        "status": "passed" if passed else "failed",
        "steps": LOCKSTEP_STEPS,
        "gradient_accumulation_steps": a0_config.gradient_accumulation_steps,
        "first_divergence": first_divergence,
        "runtime": runtime,
        "identities": identities,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
    }
    del a0_model, a1_model, a0_optimizer, a1_optimizer, batches
    gc.collect()
    torch.cuda.empty_cache()
    return payload


def _single(variant: str, output: Path) -> dict[str, Any]:
    config, model, optimizer = _model_and_optimizer(variant)
    batches = _batches(config, FRESH_STEPS * config.gradient_accumulation_steps)
    batch_index = 0
    started = time.perf_counter()
    for step in range(1, FRESH_STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(config.gradient_accumulation_steps):
            rng_before = _capture_rng()
            _microbatch(model, config, batches[batch_index], rng_before)
            batch_index += 1
        optimizer.step()
    payload = {
        "status": "passed",
        "variant": variant,
        "steps": FRESH_STEPS,
        "identity": _state_identity(model, optimizer),
        "runtime": _runtime_summary(model),
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_write_json(output, payload)
    print(json.dumps({"event": "fresh_complete", "output": str(output)}))
    return payload


def _fresh_processes() -> dict[str, Any]:
    root = run_root() / "determinism" / "fresh"
    rows = []
    for variant in VARIANTS:
        for repeat in (1, 2):
            output = root / f"{variant}-repeat{repeat}.json"
            command = [
                str(PYTHON),
                str(Path(__file__).resolve()),
                "--single",
                "--variant",
                variant,
                "--output",
                str(output),
            ]
            subprocess.run(command, check=True, env=os.environ.copy())
            rows.append(load_json(output))
    comparisons = {}
    for variant in VARIANTS:
        selected = [row for row in rows if row["variant"] == variant]
        comparisons[variant] = {
            "repeat_count": len(selected),
            "identities_equal": selected[0]["identity"] == selected[1]["identity"],
            "identities": [row["identity"] for row in selected],
        }
    return {
        "status": (
            "passed"
            if all(value["identities_equal"] for value in comparisons.values())
            else "failed"
        ),
        "steps": FRESH_STEPS,
        "comparisons": comparisons,
        "runs": rows,
    }


def run() -> dict[str, Any]:
    configure_numerics()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    lockstep = _lockstep()
    path = run_root() / "determinism" / "summary.json"
    if lockstep["status"] != "passed":
        payload = {
            "status": "failed",
            "recorded_at_utc": utc_now(),
            "environment": environment_metadata(),
            "lockstep": lockstep,
            "fresh_processes": None,
        }
        atomic_write_json(path, payload)
        return payload
    fresh = _fresh_processes()
    payload = {
        "status": "passed" if fresh["status"] == "passed" else "failed",
        "recorded_at_utc": utc_now(),
        "environment": environment_metadata(),
        "lockstep": lockstep,
        "fresh_processes": fresh,
    }
    atomic_write_json(path, payload)
    print(json.dumps({"status": payload["status"], "summary": str(path)}))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--variant", choices=tuple(VARIANTS))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_numerics()
    if args.single:
        if args.variant is None or args.output is None:
            raise ValueError("--single requires --variant and --output.")
        _single(args.variant, args.output)
        return 0
    result = run()
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
