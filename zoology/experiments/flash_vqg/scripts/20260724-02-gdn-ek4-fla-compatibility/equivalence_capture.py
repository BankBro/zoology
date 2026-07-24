#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from common import (
    BASE,
    EXPERIMENT_ID,
    build_model_config,
    configure_numerics,
    environment_metadata,
    write_json,
)


def _cpu_tensor(rng: np.random.Generator, shape: tuple[int, ...], *, scale: float = 1.0):
    array = rng.standard_normal(shape, dtype=np.float32) * np.float32(scale)
    return torch.from_numpy(array.copy())


def _tensor_hash(tensor: torch.Tensor) -> str:
    return BASE._tensor_hash(tensor.detach().cpu().contiguous())


def _kernel_capture(args: argparse.Namespace) -> dict[str, Any]:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    rng = np.random.default_rng(2026072402)
    shape_qk = (1, 64, 2, 256)
    shape_v = (1, 64, 2, 256)
    q = _cpu_tensor(rng, shape_qk, scale=0.1).cuda().requires_grad_(True)
    k = _cpu_tensor(rng, shape_qk, scale=0.1).cuda().requires_grad_(True)
    v = _cpu_tensor(rng, shape_v, scale=0.1).cuda().requires_grad_(True)
    g_raw = _cpu_tensor(rng, (1, 64, 2), scale=0.1)
    g = (-torch.nn.functional.softplus(g_raw)).cuda().requires_grad_(True)
    beta_raw = _cpu_tensor(rng, (1, 64, 2), scale=0.1)
    beta = beta_raw.sigmoid().cuda().requires_grad_(True)

    output, final_state = chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    loss = output.float().square().mean() + 0.01 * final_state.float().square().mean()
    loss.backward()
    torch.cuda.synchronize()
    tensors = {
        "output": output.detach().cpu(),
        "final_state": final_state.detach().cpu(),
        "grad/q": q.grad.detach().cpu(),
        "grad/k": k.grad.detach().cpu(),
        "grad/v": v.grad.detach().cpu(),
        "grad/g": g.grad.detach().cpu(),
        "grad/beta": beta.grad.detach().cpu(),
    }
    return {
        "kind": "kernel",
        "loss": float(loss.detach().cpu()),
        "input_hashes": {
            "q": _tensor_hash(q),
            "k": _tensor_hash(k),
            "v": _tensor_hash(v),
            "g": _tensor_hash(g),
            "beta": _tensor_hash(beta),
        },
        "tensors": tensors,
    }


def _model_capture(args: argparse.Namespace) -> dict[str, Any]:
    from zoology.data.utils import prepare_data
    from zoology.utils import set_determinism

    _, flash = build_model_config(args.model)
    gdn = BASE._build_gdn_config(flash.data)
    model, _, state_hash = BASE._model_and_hash(
        args.model, flash, gdn, BASE.GDN_CANONICAL_INIT
    )
    train_loader, _ = prepare_data(flash.data)
    inputs_cpu, targets_cpu, _, ordinal = BASE._select_batch(train_loader, BASE.TRAIN_SHAPE)
    inputs_cpu = inputs_cpu[:2].contiguous()
    targets_cpu = targets_cpu[:2].contiguous()
    model = model.cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    optimizer.zero_grad(set_to_none=True)
    inputs = inputs_cpu.cuda()
    targets = targets_cpu.cuda()
    set_determinism(2026072402, deterministic=False)
    BASE._set_dense_teacher(model, targets)
    try:
        logits = model(inputs)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.flatten()
        )
        loss = loss + BASE._auxiliary_loss(model, inputs.device)
    finally:
        BASE._clear_dense_teacher(model)
    loss.backward()
    gradients = {
        name: parameter.grad.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    optimizer.step()
    torch.cuda.synchronize()
    parameters = {
        name: parameter.detach().cpu() for name, parameter in model.named_parameters()
    }
    return {
        "kind": "model",
        "model": args.model,
        "loss": float(loss.detach().cpu()),
        "init_state_hash": state_hash,
        "batch": {
            "ordinal": ordinal,
            "shape": list(inputs_cpu.shape),
            "inputs_sha256": _tensor_hash(inputs_cpu),
            "targets_sha256": _tensor_hash(targets_cpu),
        },
        "tensors": {
            "logits": logits.detach().cpu(),
            **{f"grad/{key}": value for key, value in gradients.items()},
            **{f"param_after/{key}": value for key, value in parameters.items()},
        },
    }


def capture(args: argparse.Namespace) -> int:
    configure_numerics()
    if not torch.cuda.is_available():
        raise RuntimeError("目标容器中 CUDA 不可用.")
    payload = _kernel_capture(args) if args.kind == "kernel" else _model_capture(args)
    record = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "environment": environment_metadata(),
        **payload,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(record, args.output)
    write_json(
        args.output.with_suffix(".json"),
        {
            key: value
            for key, value in record.items()
            if key != "tensors"
        }
        | {
            "capture": str(args.output),
            "capture_bytes": args.output.stat().st_size,
            "tensor_count": len(record["tensors"]),
        },
    )
    print(args.output)
    return 0


def _comparison(name: str, reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        return {
            "name": name,
            "shape_match": reference.shape == candidate.shape,
            "dtype_match": reference.dtype == candidate.dtype,
            "passed": False,
        }
    ref = reference.double()
    cand = candidate.double()
    diff = cand - ref
    max_abs = float(diff.abs().max()) if diff.numel() else 0.0
    rel_l2 = float(diff.norm() / ref.norm().clamp_min(1e-30)) if diff.numel() else 0.0
    if name.startswith("grad/"):
        passed = max_abs <= 1e-5 and rel_l2 <= 1e-4
    elif name.startswith("param_after/"):
        passed = max_abs <= 1e-5 and rel_l2 <= 1e-5
    else:
        passed = max_abs <= 1e-5 and rel_l2 <= 1e-5
    return {
        "name": name,
        "shape": list(reference.shape),
        "dtype": str(reference.dtype),
        "exact": bool(torch.equal(reference, candidate)),
        "max_abs": max_abs,
        "relative_l2": rel_l2,
        "finite": bool(torch.isfinite(candidate).all()),
        "passed": bool(passed and torch.isfinite(candidate).all()),
    }


def compare(args: argparse.Namespace) -> int:
    reference = torch.load(args.reference, map_location="cpu", weights_only=False)
    candidate = torch.load(args.candidate, map_location="cpu", weights_only=False)
    if reference["kind"] != candidate["kind"]:
        raise ValueError("capture kind 不一致.")
    reference_tensors = reference["tensors"]
    candidate_tensors = candidate["tensors"]
    names_match = set(reference_tensors) == set(candidate_tensors)
    rows = [
        _comparison(name, reference_tensors[name], candidate_tensors[name])
        for name in sorted(set(reference_tensors) & set(candidate_tensors))
    ]
    loss_abs = abs(float(candidate["loss"]) - float(reference["loss"]))
    passed = names_match and loss_abs <= 1e-6 and all(row["passed"] for row in rows)
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "kind": reference["kind"],
        "model": reference.get("model"),
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "reference_environment": reference["environment"],
        "candidate_environment": candidate["environment"],
        "tensor_names_match": names_match,
        "loss_abs": loss_abs,
        "comparisons": rows,
        "passed": passed,
    }
    write_json(args.output, payload)
    csv_path = args.output.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    keys = ["name", "shape", "dtype", "exact", "max_abs", "relative_l2", "finite", "passed"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})
    print(f"{args.output} passed={passed}")
    return 0 if passed else 1


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="跨 FLA 环境保存并比较精确数值 capture.")
    sub = root.add_subparsers(dest="command", required=True)
    cap = sub.add_parser("capture")
    cap.add_argument("--kind", choices=("kernel", "model"), required=True)
    cap.add_argument("--model", choices=("gdn", "flash"), default="gdn")
    cap.add_argument("--output", type=Path, required=True)
    cap.set_defaults(func=capture)
    comp = sub.add_parser("compare")
    comp.add_argument("--reference", type=Path, required=True)
    comp.add_argument("--candidate", type=Path, required=True)
    comp.add_argument("--output", type=Path, required=True)
    comp.set_defaults(func=compare)
    return root


if __name__ == "__main__":
    args = parser().parse_args()
    raise SystemExit(args.func(args))
