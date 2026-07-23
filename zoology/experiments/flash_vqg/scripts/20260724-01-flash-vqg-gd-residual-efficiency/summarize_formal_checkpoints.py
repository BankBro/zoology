#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch


EXPERIMENT_ID = "20260724-01-flash-vqg-gd-residual-efficiency"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_hash(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        value = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(",".join(map(str, value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine", choices=("2080ti", "3090"), required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    pattern = (
        f"fvqg-{EXPERIMENT_ID}-{args.machine}-s*-baseline-r16-joint-"
        f"{args.machine}-formal/*/last.pt"
    )
    for path in sorted(args.checkpoint_root.glob(pattern)):
        payload = torch.load(path, map_location="cpu")
        metrics = payload["metrics"]
        run_id = str(payload["run_id"])
        seed_match = re.search(r"efficiency-s(124|125)-baseline", run_id)
        if seed_match is None:
            raise RuntimeError(f"Could not parse training seed from run_id: {run_id}")
        seed = int(seed_match.group(1))
        rows.append(
            {
                "machine": args.machine,
                "training_seed": seed,
                "run_id": run_id,
                "launch_id": payload["launch_id"],
                "checkpoint_path": str(path),
                "checkpoint_bytes": path.stat().st_size,
                "checkpoint_sha256": _file_sha256(path),
                "model_state_sha256": _state_hash(payload["model_state_dict"]),
                "epoch": payload["epoch"],
                "valid_loss": metrics["valid/loss"],
                "valid_accuracy": metrics["valid/accuracy"],
                "accuracy_1024x256": metrics[
                    "valid/mqar_case/accuracy-1024x256"
                ],
            }
        )
    if len(rows) != 2:
        raise RuntimeError(f"Expected two formal checkpoints, found {len(rows)} using {pattern}")
    output = {
        "experiment_id": EXPERIMENT_ID,
        "machine": args.machine,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "rows": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
