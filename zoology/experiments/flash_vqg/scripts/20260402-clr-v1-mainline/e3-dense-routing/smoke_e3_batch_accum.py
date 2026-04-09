#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch


def _load_module(module_name: str, path: Path):
    spec = spec_from_file_location(module_name, path)
    module = module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


_SMOKE_COMMON = _load_module(
    "flash_vqg_e2_smoke_common",
    Path(__file__).resolve().parents[1] / "e2-remote-interface" / "smoke_common.py",
)

CANDIDATE_BATCH_COMBOS = _SMOKE_COMMON.CANDIDATE_BATCH_COMBOS


def _build_args(
    namespace: argparse.Namespace,
    *,
    train_batch_size: int,
    eval_batch_size: int,
    gradient_accumulation_steps: int,
) -> argparse.Namespace:
    args = argparse.Namespace(**vars(namespace))
    args.logger_backend = "none"
    args.analysis = "off"
    args.train_batch_size = train_batch_size
    args.eval_batch_size = eval_batch_size
    args.gradient_accumulation_steps = gradient_accumulation_steps
    args.max_epochs = 1
    args.metrics_white_list = None
    args.metrics_white_list_file = namespace.metrics_white_list_file
    return args


def _load_builder():
    module = _load_module(
        "flash_vqg_e3_builder",
        Path(__file__).resolve().parent / "config_builder.py",
    )
    return module.build_e3_smoke_configs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--backend", type=str, default="torch")
    parser.add_argument("--logger-backend", type=str, default="none")
    parser.add_argument("--block-len", type=str, default="32")
    parser.add_argument("--dmodels", type=str, default="128")
    parser.add_argument("--num-codebook-vectors", type=str, default="128")
    parser.add_argument("--learning-rates", type=str, default="1e-3")
    parser.add_argument("--if-remote-enabled", type=str, default="true")
    parser.add_argument("--local-num-blocks", type=str, default="2")
    parser.add_argument("--train-batch-order", type=str, default="global_shuffle")
    parser.add_argument("--seed-values", type=str, default="123")
    parser.add_argument("--data-seed", type=int, default=123)
    parser.add_argument("--cache-dir", type=str, default="./data/flash_vqg")
    parser.add_argument("--fox-remote-path-backend", type=str, default="torch")
    parser.add_argument("--fox-clr-rank", type=int, default=4)
    parser.add_argument("--fox-clr-use-den-residual", type=str, default="true")
    parser.add_argument("--fox-clr-remat-mode", type=str, default="off")
    parser.add_argument("--vq-topk", type=int, default=4)
    parser.add_argument("--project", type=str, default="smoke")
    parser.add_argument("--entity", type=str, default="smoke")
    parser.add_argument("--launch-id-prefix", type=str, default="flash-vqg-e3-smoke")
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--analysis", type=str, default="off")
    parser.add_argument("--metrics-white-list", type=str, default=None)
    parser.add_argument("--metrics-white-list-file", type=str, default=None)
    namespace = parser.parse_args()

    device = torch.device(f"cuda:{namespace.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(int(namespace.gpu))

    builder = _load_builder()
    results = []
    chosen = None
    started_at = time.time()

    for train_batch_size, eval_batch_size, gradient_accumulation_steps in CANDIDATE_BATCH_COMBOS:
        print(
            f"[smoke] e3 tbs={train_batch_size} ebs={eval_batch_size} ga={gradient_accumulation_steps}",
            flush=True,
        )
        args = _build_args(
            namespace,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        configs = builder(args)
        combo_results = []
        combo_ok = True
        for config in configs:
            result = _SMOKE_COMMON._run_one_config(config, device)
            combo_results.append(result)
            if result["status"] != "pass":
                combo_ok = False
        combo_payload = {
            "train_batch_size": train_batch_size,
            "eval_batch_size": eval_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "effective_train_batch_size": train_batch_size * gradient_accumulation_steps,
            "results": combo_results,
        }
        results.append(combo_payload)
        if combo_ok and chosen is None:
            chosen = {
                "train_batch_size": train_batch_size,
                "eval_batch_size": eval_batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "effective_train_batch_size": train_batch_size * gradient_accumulation_steps,
            }
            break

    payload = {
        "part": "e3",
        "started_at_unix": started_at,
        "duration_sec": time.time() - started_at,
        "chosen": chosen,
        "candidates": results,
    }
    summary_path, env_path = _SMOKE_COMMON._write_smoke_outputs(
        part="e3",
        payload=payload,
        chosen=chosen,
    )
    print(f"SMOKE_SUMMARY_JSON={summary_path}")
    print(f"SMOKE_ENV_FILE={env_path}")
    if chosen is None:
        raise SystemExit("No batch/GA candidate passed E3 smoke.")


if __name__ == "__main__":
    main()
