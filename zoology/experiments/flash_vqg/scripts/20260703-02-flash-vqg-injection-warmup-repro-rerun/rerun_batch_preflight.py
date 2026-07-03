#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get("ZOOLOGY_REPO_ROOT", "/home/lyj/mnt/project/zoology")).resolve()
SOURCE_SCRIPT = (
    REPO_ROOT
    / "zoology/experiments/flash_vqg/scripts/"
    / "20260702-03-flash-vqg-injection-warmup-screen/injection_warmup_screen.py"
)


def _load_source():
    spec = importlib.util.spec_from_file_location("injection_warmup_screen_source", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load source script: {SOURCE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SRC = _load_source()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _batch_order_hash(dataloader: Any) -> dict[str, Any]:
    sampler = getattr(dataloader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(0)
    order = list(iter(sampler)) if sampler is not None else list(range(len(dataloader)))
    hasher = hashlib.sha256()
    for item in order:
        hasher.update(int(item).to_bytes(8, "little", signed=True))
    return {
        "num_batches": len(order),
        "sha256": hasher.hexdigest(),
        "first_16": order[:16],
    }


def run(args: argparse.Namespace) -> int:
    config = SRC.BASEMOD.BASEMOD.BASE.build_config(
        target=args.target,
        machine_name=args.machine_name,
        variant=args.variant,
        logger_backend="none",
        trace_output_dir=SCRIPT_DIR / "outputs/preflight-traces" / args.machine_name / args.target,
        max_epochs=args.max_epochs,
        max_train_steps=args.max_train_steps,
        max_validation_batches=None,
    )
    if args.run_suffix:
        SRC.BASEMOD.BASEMOD._apply_run_suffix(config, args.run_suffix)
    train_loader, _ = SRC.BASEMOD.BASEMOD.BASE.prepare_data(config.data)
    payload = {
        "schema_version": 1,
        "experiment_id": "20260703-02-flash-vqg-injection-warmup-repro-rerun",
        "source_experiment_id": SRC.EXPERIMENT_ID,
        "machine_name": args.machine_name,
        "target": args.target,
        "variant": args.variant,
        "run_id": config.run_id,
        "launch_id": config.launch_id,
        "cache_dir": config.data.cache_dir,
        "batch_order": _batch_order_hash(train_loader),
        "max_epochs": int(config.max_epochs),
        "max_train_steps": config.max_train_steps,
        "gradient_accumulation_steps": int(config.gradient_accumulation_steps),
        "read_trace_enabled": bool(config.read_trace_enabled),
        "read_trace_train_steps": list(config.read_trace_train_steps),
        "train_inline_event_trace_enabled": bool(config.train_inline_event_trace_enabled),
        "checkpoint_enabled": bool(config.checkpoint.enabled),
        "checkpoint_save_best": bool(config.checkpoint.save_best),
        "checkpoint_save_last": bool(config.checkpoint.save_last),
    }
    if args.output_json:
        _save_json(args.output_json, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine-name", required=True)
    parser.add_argument("--target", required=True, choices=SRC.TARGETS)
    parser.add_argument("--variant", required=True, choices=SRC.TARGETS)
    parser.add_argument("--run-suffix")
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--max-train-steps", type=int, default=704)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    os.chdir(REPO_ROOT)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
