#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import re
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from common import (
    EXPERIMENT_ID,
    REPO_ROOT,
    VARIANTS,
    append_jsonl,
    atomic_write_json,
    build_config,
    configure_numerics,
    git_value,
    run_root,
    serialize_config,
    sha256_file,
    utc_now,
)

from zoology.train import Trainer


GATE_REPLAY_MODULE = (
    "backbone.layers.1.sequence_mixer.mixer.attn.output_gate_fused"
)


def _hash_bytes(*parts: bytes) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(len(part).to_bytes(8, "little"))
        digest.update(part)
    return digest.hexdigest()


def tensor_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    return _hash_bytes(str(value.dtype).encode(), str(tuple(value.shape)).encode(), raw)


def tensor_record(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach()
    result = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "numel": int(value.numel()),
        "sha256": tensor_hash(value),
    }
    if value.numel() and value.is_floating_point():
        f32 = value.float()
        result["max_abs"] = float(f32.abs().max().cpu().item())
        result["l2"] = float(torch.linalg.vector_norm(f32).cpu().item())
    if value.numel() <= 256:
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        result["raw_hex"] = raw.numpy().tobytes().hex()
    return result


def named_tensor_hash(items: list[tuple[str, torch.Tensor | None]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(items):
        digest.update(name.encode("utf-8") + b"\0")
        token = "NONE" if tensor is None else tensor_hash(tensor)
        digest.update(token.encode("ascii") + b"\n")
    return digest.hexdigest()


def named_tensor_records(
    items: list[tuple[str, torch.Tensor | None]],
) -> dict[str, Any]:
    return {
        name: None if tensor is None else tensor_record(tensor)
        for name, tensor in sorted(items)
    }


def model_items(model: torch.nn.Module) -> list[tuple[str, torch.Tensor]]:
    return [(name, tensor) for name, tensor in model.state_dict().items()]


def gradient_items(model: torch.nn.Module) -> list[tuple[str, torch.Tensor | None]]:
    return [(name, parameter.grad) for name, parameter in model.named_parameters()]


def optimizer_items(optimizer: torch.optim.Optimizer) -> list[tuple[str, torch.Tensor]]:
    state = optimizer.state_dict()
    items: list[tuple[str, torch.Tensor]] = []
    for param_id, values in sorted(state["state"].items(), key=lambda item: int(item[0])):
        for key, value in sorted(values.items()):
            if torch.is_tensor(value):
                items.append((f"state.{param_id}.{key}", value))
    return items


def optimizer_hash(optimizer: torch.optim.Optimizer) -> str:
    state = optimizer.state_dict()
    groups = json.dumps(state["param_groups"], sort_keys=True, default=str).encode()
    return _hash_bytes(groups, named_tensor_hash(optimizer_items(optimizer)).encode())


def rng_hash() -> str:
    parts = [
        pickle.dumps(random.getstate(), protocol=4),
        pickle.dumps(np.random.get_state(), protocol=4),
        torch.get_rng_state().numpy().tobytes(),
    ]
    if torch.cuda.is_available():
        parts.extend(state.cpu().numpy().tobytes() for state in torch.cuda.get_rng_state_all())
    return _hash_bytes(*parts)


def configure_gate_bwd_runtime(config_name: str) -> None:
    if config_name == "default":
        return
    match = re.fullmatch(r"bt(16|32|64)-w(2|4|8|16)", config_name)
    if match is None:
        raise ValueError(f"Unsupported gate backward config: {config_name}.")
    import triton
    from fla.modules import fused_norm_gate

    autotuner = fused_norm_gate.layer_norm_gated_bwd_kernel.fn
    autotuner.configs = [
        triton.Config(
            {"BT": int(match.group(1))},
            num_warps=int(match.group(2)),
        )
    ]
    autotuner.cache.clear()


def triton_config_record(config: Any) -> dict[str, Any]:
    return {
        "kwargs": dict(config.kwargs),
        "num_warps": config.num_warps,
        "num_stages": config.num_stages,
        "num_ctas": getattr(config, "num_ctas", None),
    }


def gate_autotune_snapshot() -> dict[str, Any]:
    from fla.modules import fused_norm_gate

    result = {}
    for name in ("layer_norm_gated_fwd_kernel", "layer_norm_gated_bwd_kernel"):
        autotuner = getattr(fused_norm_gate, name).fn
        best_config = getattr(autotuner, "best_config", None)
        result[name] = {
            "best_config": (
                None if best_config is None else triton_config_record(best_config)
            ),
            "cache": [
                {
                    "key": repr(key),
                    "config": triton_config_record(config),
                }
                for key, config in autotuner.cache.items()
            ],
        }
    return result


def output_records(value: Any, prefix: str = "output") -> dict[str, Any]:
    if torch.is_tensor(value):
        return {prefix: tensor_record(value)}
    if isinstance(value, (list, tuple)):
        result: dict[str, Any] = {}
        for index, child in enumerate(value):
            result.update(output_records(child, f"{prefix}.{index}"))
        return result
    if isinstance(value, dict):
        result = {}
        for key, child in sorted(value.items()):
            result.update(output_records(child, f"{prefix}.{key}"))
        return result
    return {}


def should_hook_module(name: str) -> bool:
    if name in {
        "backbone.embeddings",
        "backbone.ln_f",
        "lm_head",
        GATE_REPLAY_MODULE,
    }:
        return True
    parts = name.split(".")
    if len(parts) == 3 and parts[:2] == ["backbone", "layers"]:
        return True
    return len(parts) == 4 and parts[:2] == ["backbone", "layers"] and parts[3] in {
        "sequence_mixer",
        "state_mixer",
    }


class DiagnosticTrainer(Trainer):
    def __init__(
        self,
        *args: Any,
        diagnostic_trace_path: Path,
        detail_window: int | None,
        gate_replay_dir: Path | None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.diagnostic_trace_path = diagnostic_trace_path
        self.detail_window = detail_window
        self.gate_replay_dir = gate_replay_dir
        self._micro_in_window = 0
        self._pending_backward: dict[str, int] | None = None
        self._module_call_counts: dict[tuple[int, int, str], int] = defaultdict(int)
        self._hook_handles: list[Any] = []

    def _write(self, event: str, **payload: Any) -> None:
        append_jsonl(
            self.diagnostic_trace_path,
            {
                "event": event,
                "recorded_at_utc": utc_now(),
                "optimizer_step": int(self.optimizer_step),
                "window": int(self.optimizer_step) + 1,
                **payload,
            },
        )

    def _is_detail_window(self, window: int) -> bool:
        return self.detail_window is not None and window == self.detail_window

    def _state_payload(self, *, detail: bool) -> dict[str, Any]:
        model = model_items(self.model)
        gradients = gradient_items(self.model)
        result: dict[str, Any] = {
            "model_sha256": named_tensor_hash(model),
            "grad_sha256": named_tensor_hash(gradients),
            "optimizer_sha256": optimizer_hash(self.optimizer),
            "rng_sha256": rng_hash(),
        }
        if detail:
            result["model_tensors"] = named_tensor_records(model)
            result["grad_tensors"] = named_tensor_records(gradients)
            result["optimizer_tensors"] = named_tensor_records(
                optimizer_items(self.optimizer)
            )
        return result

    def _flush_pending_backward(self) -> None:
        if self._pending_backward is None:
            return
        context = self._pending_backward
        detail = self._is_detail_window(context["window"])
        self._write(
            "after_backward",
            window=context["window"],
            optimizer_step=context["optimizer_step"],
            micro_step=context["micro_step"],
            **self._state_payload(detail=detail),
        )
        self._pending_backward = None

    def compute_loss(self, inputs, targets, *, return_predictions: bool = True):
        if not self.model.training:
            return super().compute_loss(
                inputs,
                targets,
                return_predictions=return_predictions,
            )
        self._flush_pending_backward()
        window = int(self.optimizer_step) + 1
        micro_step = int(self._micro_in_window)
        rng_before = rng_hash()
        loss, predictions = super().compute_loss(
            inputs,
            targets,
            return_predictions=return_predictions,
        )
        self._write(
            "forward",
            window=window,
            micro_step=micro_step,
            input=tensor_record(inputs),
            target=tensor_record(targets),
            loss=tensor_record(loss.detach()),
            loss_value=float(loss.detach().cpu().item()),
            rng_before_sha256=rng_before,
            rng_after_sha256=rng_hash(),
        )
        self._pending_backward = {
            "window": window,
            "optimizer_step": int(self.optimizer_step),
            "micro_step": micro_step,
        }
        self._micro_in_window += 1
        return loss, predictions

    def _step_optimizer(self) -> bool:
        self._flush_pending_backward()
        window = int(self.optimizer_step) + 1
        detail = self._is_detail_window(window)
        self._write("before_optimizer", **self._state_payload(detail=detail))
        succeeded = super()._step_optimizer()
        self._write(
            "after_optimizer",
            optimizer_succeeded=bool(succeeded),
            **self._state_payload(detail=detail),
        )
        return succeeded

    def _log_metrics(self, metrics: dict[str, Any]):
        if "train/loss" in metrics:
            completed_window = int(self.optimizer_step)
            self._write(
                "after_zero_grad",
                window=completed_window,
                optimizer_step=completed_window - 1,
                train_loss=float(metrics["train/loss"]),
                **self._state_payload(
                    detail=self._is_detail_window(completed_window)
                ),
            )
            self._micro_in_window = 0
        return super()._log_metrics(metrics)

    def test(self, epoch_idx: int):
        self._write("before_validation", epoch=int(epoch_idx), rng_sha256=rng_hash())
        result = super().test(epoch_idx)
        self._write(
            "after_validation",
            epoch=int(epoch_idx),
            rng_sha256=rng_hash(),
            metrics=result,
        )
        return result

    def _module_hook(self, name: str):
        def hook(_module, inputs, output):
            window = int(self.optimizer_step) + 1
            if not self.model.training or not self._is_detail_window(window):
                return
            micro_step = int(self._micro_in_window)
            key = (window, micro_step, name)
            call_index = self._module_call_counts[key]
            self._module_call_counts[key] += 1
            self._write(
                "module_forward",
                window=window,
                micro_step=micro_step,
                module=name,
                call_index=call_index,
                inputs=output_records(inputs, "input"),
                outputs=output_records(output, "output"),
            )
            self._maybe_capture_gate_replay(
                name=name,
                module=_module,
                inputs=inputs,
                output=output,
                window=window,
                micro_step=micro_step,
                call_index=call_index,
            )

        return hook

    def _maybe_capture_gate_replay(
        self,
        *,
        name: str,
        module: torch.nn.Module,
        inputs: Any,
        output: Any,
        window: int,
        micro_step: int,
        call_index: int,
    ) -> None:
        if self.gate_replay_dir is None or name != GATE_REPLAY_MODULE:
            return
        if window != self.detail_window or micro_step != 0 or call_index != 0:
            return
        if not torch.is_tensor(output) or len(inputs) != 2:
            raise TypeError("Gate replay expects two tensor inputs and one tensor output.")
        path = self.gate_replay_dir / "layer1-window1-micro0.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        x, gate = inputs

        def save_replay(grad_output: torch.Tensor) -> torch.Tensor:
            payload = {
                "module": name,
                "window": window,
                "micro_step": micro_step,
                "x": x.detach().cpu(),
                "gate": gate.detach().cpu(),
                "output": output.detach().cpu(),
                "grad_output": grad_output.detach().cpu(),
                "weight": module.weight.detach().cpu(),
                "bias": None if module.bias is None else module.bias.detach().cpu(),
                "activation": module.activation,
                "eps": float(module.eps),
            }
            torch.save(payload, path)
            return grad_output

        output.register_hook(save_replay)

    def _install_module_hooks(self) -> None:
        if self.detail_window is None:
            return
        for name, module in self.model.named_modules():
            if should_hook_module(name):
                self._hook_handles.append(
                    module.register_forward_hook(self._module_hook(name))
                )

    def fit(self):
        self._install_module_hooks()
        try:
            return super().fit()
        finally:
            for handle in self._hook_handles:
                handle.remove()


def result_dir(label: str, variant: str) -> Path:
    return run_root() / "probes" / label / variant


def run_probe(
    *,
    variant: str,
    label: str,
    max_train_steps: int,
    detail_window: int | None,
    gate_bwd_config: str,
    capture_gate_replay: bool,
) -> int:
    configure_numerics()
    configure_gate_bwd_runtime(gate_bwd_config)
    config = build_config(
        variant,
        label=label,
        max_train_steps=max_train_steps,
    )
    output_dir = result_dir(label, variant)
    trace_path = output_dir / "trace.jsonl"
    if trace_path.exists():
        raise FileExistsError(f"Refusing to overwrite trace: {trace_path}")
    output_dir.mkdir(parents=True, exist_ok=False)
    gate_replay_dir = output_dir / "replay" if capture_gate_replay else None
    resolved_path = output_dir / "resolved_config.json"
    atomic_write_json(resolved_path, serialize_config(config))

    import zoology.train as train_module

    holder: dict[str, DiagnosticTrainer] = {}
    original_trainer = train_module.Trainer

    class BoundDiagnosticTrainer(DiagnosticTrainer):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(
                *args,
                diagnostic_trace_path=trace_path,
                detail_window=detail_window,
                gate_replay_dir=gate_replay_dir,
                **kwargs,
            )
            holder["trainer"] = self

    train_module.Trainer = BoundDiagnosticTrainer
    started = time.perf_counter()
    status = "completed"
    error = None
    try:
        train_module.train(config)
    except BaseException as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        train_module.Trainer = original_trainer

    trainer = holder.get("trainer")
    result = {
        "experiment_id": EXPERIMENT_ID,
        "run_tag": os.environ["MQAR_SEED124_DIAG_RUN_TAG"],
        "label": label,
        "variant": variant,
        "remat_mode": VARIANTS[variant],
        "seed": 124,
        "max_train_steps": max_train_steps,
        "detail_window": detail_window,
        "gate_bwd_config": gate_bwd_config,
        "gate_autotune": gate_autotune_snapshot(),
        "gate_replay": (
            []
            if gate_replay_dir is None
            else [
                {
                    "path": str(path.resolve()),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in sorted(gate_replay_dir.glob("*.pt"))
            ]
        ),
        "status": status,
        "error": error,
        "wall_clock_sec": time.perf_counter() - started,
        "trace_path": str(trace_path.resolve()),
        "trace_sha256": sha256_file(trace_path) if trace_path.exists() else None,
        "zoology_commit": git_value(REPO_ROOT, "rev-parse", "HEAD"),
        "flash_commit": git_value(
            Path(os.environ.get("FLASH_VQG_ROOT", "/home/lyj/mnt/project/Flash-VQG")),
            "rev-parse",
            "HEAD",
        ),
        "optimizer_step": None if trainer is None else int(trainer.optimizer_step),
        "global_step": None if trainer is None else int(trainer.global_step),
    }
    atomic_write_json(output_dir / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if status == "completed" else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--max-train-steps", type=int, required=True)
    parser.add_argument("--detail-window", type=int)
    parser.add_argument("--gate-bwd-config", default="default")
    parser.add_argument("--capture-gate-replay", action="store_true")
    args = parser.parse_args()
    return run_probe(
        variant=args.variant,
        label=args.label,
        max_train_steps=args.max_train_steps,
        detail_window=args.detail_window,
        gate_bwd_config=args.gate_bwd_config,
        capture_gate_replay=args.capture_gate_replay,
    )


if __name__ == "__main__":
    sys.exit(main())
