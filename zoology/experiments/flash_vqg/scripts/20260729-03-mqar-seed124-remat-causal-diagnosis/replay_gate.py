#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from common import atomic_write_json, sha256_file
from probe import configure_gate_bwd_runtime, gate_autotune_snapshot, tensor_record


def replay(capsule_path: Path, config_name: str) -> dict:
    configure_gate_bwd_runtime(config_name)
    capsule = torch.load(capsule_path, map_location="cpu", weights_only=False)
    x = capsule["x"].cuda().requires_grad_(True)
    gate = capsule["gate"].cuda().requires_grad_(True)
    weight = capsule["weight"].cuda().requires_grad_(True)
    bias = capsule["bias"]
    if bias is not None:
        bias = bias.cuda().requires_grad_(True)
    grad_output = capsule["grad_output"].cuda()

    from fla.modules.fused_norm_gate import rms_norm_gated

    output = rms_norm_gated(
        x,
        gate,
        weight,
        bias,
        activation=capsule["activation"],
        eps=float(capsule["eps"]),
    )
    variables = (x, gate, weight) if bias is None else (x, gate, weight, bias)
    gradients = torch.autograd.grad(output, variables, grad_outputs=grad_output)
    result = {
        "capsule": str(capsule_path.resolve()),
        "capsule_sha256": sha256_file(capsule_path),
        "gate_bwd_config": config_name,
        "output": tensor_record(output),
        "grad_x": tensor_record(gradients[0]),
        "grad_gate": tensor_record(gradients[1]),
        "grad_weight": tensor_record(gradients[2]),
        "grad_bias": None if bias is None else tensor_record(gradients[3]),
        "gate_autotune": gate_autotune_snapshot(),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--gate-bwd-config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = replay(args.capsule, args.gate_bwd_config)
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
