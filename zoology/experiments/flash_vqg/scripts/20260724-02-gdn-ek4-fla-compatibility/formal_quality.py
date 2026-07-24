#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from common import (
    BASE,
    EXPERIMENT_ID,
    configure_numerics,
    environment_metadata,
    sha256_file,
    write_json,
)
from zoology.checkpoints import serialize_train_config
from zoology.data.utils import prepare_data
from zoology.train import train

EXPECTED_GDN_INIT_HASH = "bdba0c19b2530c72c3ae7dd6bd708901c2369f6d3e1da9d850ea8347d5ea60a6"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_config(args: argparse.Namespace):
    flash = BASE._build_flash_config("core", "triton", "triton_remat")
    gdn = BASE._build_gdn_config(flash.data)
    config = flash if args.model == "flash" else gdn
    config.max_epochs = 1
    config.max_train_steps = args.max_train_steps
    config.max_validation_batches = args.max_validation_batches
    config.validations_per_epoch = 4
    config.early_stopping_metric = None
    config.early_stopping_threshold = None
    config.logger.backend = "none"
    config.checkpoint.enabled = True
    config.checkpoint.save_best = True
    config.checkpoint.save_last = True
    config.checkpoint.best_metric = "valid/accuracy"
    config.checkpoint.best_mode = "max"
    config.checkpoint.root_dir = str((args.output_dir / "checkpoint-root").resolve())
    config.launch_id = f"{EXPERIMENT_ID}-{args.machine}-{args.fla_variant}"
    config.sweep_id = EXPERIMENT_ID
    config.run_id = (
        f"{args.model}-s124-{args.machine}-{args.fla_variant}-{args.run_type}"
    )
    config.seed = 124
    if args.model == "flash":
        config.init_checkpoint_path = str(BASE.CANONICAL_INIT)
    else:
        config.init_checkpoint_path = str(BASE.GDN_CANONICAL_INIT)
    config.init_checkpoint_strict = True
    return config


def hard_preflight(config: Any) -> dict[str, Any]:
    cache = BASE._cache_content_hash(config.data)
    train_loader, _ = prepare_data(config.data)
    batch_order = BASE._batch_order_hash(train_loader)
    init_payload = torch.load(config.init_checkpoint_path, map_location="cpu", weights_only=False)
    init_hash = BASE._state_dict_hash(init_payload["model_state_dict"])
    expected_init = (
        BASE.EXPECTED_INIT_HASH
        if config.model.name != "gated_delta_net_expanded_k"
        else EXPECTED_GDN_INIT_HASH
    )
    passed = bool(
        torch.cuda.is_available()
        and cache["match"]
        and batch_order["match"]
        and init_hash == expected_init
    )
    return {
        "passed": passed,
        "cache": cache,
        "batch_order": batch_order,
        "init_path": config.init_checkpoint_path,
        "init_hash": init_hash,
        "expected_init_hash": expected_init,
        "expected_optimizer_steps": (len(train_loader) + config.gradient_accumulation_steps - 1)
        // config.gradient_accumulation_steps,
    }


def run(args: argparse.Namespace) -> int:
    if args.run_type == "formal" and (
        args.max_train_steps is not None or args.max_validation_batches is not None
    ):
        raise ValueError("formal run 不允许截断训练或验证.")
    if args.run_type == "smoke" and args.max_train_steps is None:
        raise ValueError("smoke run 必须显式设置 --max-train-steps.")
    os.environ["FLA_VARIANT"] = args.fla_variant
    configure_numerics()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = build_config(args)
    preflight = hard_preflight(config)
    write_json(args.output_dir / "preflight.json", preflight | {"environment": environment_metadata()})
    if not preflight["passed"]:
        raise RuntimeError("正式质量实验硬预检失败.")
    write_json(args.output_dir / "resolved-config.json", serialize_train_config(config))

    started_at = utc_now()
    started = time.perf_counter()
    result_path = args.output_dir / "result.json"
    try:
        train(config)
        status = "completed"
        error = None
        error_traceback = None
    except BaseException as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        error_traceback = traceback.format_exc()
    ended_at = utc_now()
    wall_clock_sec = time.perf_counter() - started
    run_dir = (
        Path(config.checkpoint.root_dir) / str(config.launch_id) / str(config.run_id)
    ).resolve()
    last_checkpoint = run_dir / "last.pt"
    best_checkpoint = run_dir / "best.pt"
    checkpoint_payload = None
    if status == "completed":
        checkpoint_payload = torch.load(last_checkpoint, map_location="cpu", weights_only=False)
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "machine": args.machine,
        "fla_variant": args.fla_variant,
        "model": args.model,
        "run_type": args.run_type,
        "status": status,
        "error": error,
        "traceback": error_traceback,
        "started_at_utc": started_at,
        "ended_at_utc": ended_at,
        "wall_clock_sec": wall_clock_sec,
        "environment": environment_metadata(),
        "preflight": preflight,
        "configured_max_epochs": 1,
        "max_train_steps": config.max_train_steps,
        "max_validation_batches": config.max_validation_batches,
        "final_epoch": (
            int(checkpoint_payload["epoch"]) + 1 if checkpoint_payload else None
        ),
        "final_epoch_index_zero_based": (
            checkpoint_payload.get("epoch") if checkpoint_payload else None
        ),
        "final_metrics": checkpoint_payload.get("metrics") if checkpoint_payload else None,
        "run_id": config.run_id,
        "launch_id": config.launch_id,
        "train_config_path": str(run_dir / "train_config.json"),
        "last_checkpoint_path": str(last_checkpoint),
        "best_checkpoint_path": str(best_checkpoint),
        "last_checkpoint_sha256": sha256_file(last_checkpoint) if last_checkpoint.exists() else None,
        "best_checkpoint_sha256": sha256_file(best_checkpoint) if best_checkpoint.exists() else None,
        "model_state_sha256": (
            BASE._state_dict_hash(checkpoint_payload["model_state_dict"])
            if checkpoint_payload
            else None
        ),
    }
    write_json(result_path, payload)
    print(json.dumps({"result": str(result_path), "status": status}, ensure_ascii=False))
    if status != "completed":
        raise RuntimeError(error)
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="GDN/Flash 共同环境的正式 1ep 质量回归.")
    root.add_argument("--model", choices=("gdn", "flash"), required=True)
    root.add_argument("--machine", choices=("2080ti", "3090"), required=True)
    root.add_argument("--fla-variant", choices=("current040", "v042", "v050"), required=True)
    root.add_argument("--run-type", choices=("formal", "smoke"), default="formal")
    root.add_argument("--max-train-steps", type=int)
    root.add_argument("--max-validation-batches", type=int)
    root.add_argument("--output-dir", type=Path, required=True)
    return root


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
