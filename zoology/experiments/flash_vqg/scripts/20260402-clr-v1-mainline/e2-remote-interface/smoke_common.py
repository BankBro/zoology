#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from einops import rearrange

from zoology.data.utils import prepare_data
from zoology.model import LanguageModel


CANDIDATE_BATCH_COMBOS = [
    (256, 32, 1),
    (128, 16, 2),
    (64, 16, 4),
    (32, 8, 8),
    (16, 8, 16),
]
TRAIN_BATCHES_PER_SEGMENT = 1
EVAL_BATCHES_PER_SEGMENT = 1


def _cleanup(model, optimizer) -> None:
    del model
    del optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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


def _load_builder(part: str):
    from importlib.util import module_from_spec, spec_from_file_location

    builder_path = Path(__file__).resolve().parent / "config_builder.py"
    spec = spec_from_file_location("flash_vqg_e2_builder", builder_path)
    module = module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    if part == "e2_main":
        return module.build_e2_main_smoke_configs
    if part == "e2b":
        return module.build_e2b_smoke_configs
    raise ValueError(f"Unknown part: {part}")


def _run_one_config(config, device: torch.device) -> dict:
    result = {
        "run_id": config.run_id,
        "status": "unknown",
        "oom_phase": None,
        "train_steps": 0,
        "eval_steps": 0,
        "peak_mb": 0,
        "error": None,
    }
    try:
        model = LanguageModel(config.model).to(device)
        train_dl, test_dl = prepare_data(config.data)
    except Exception as exc:
        result["status"] = "build_error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    try:
        model.train()
        accum_steps = int(config.gradient_accumulation_steps)
        num_batches = len(train_dl)
        remainder = num_batches % accum_steps
        partial_start = num_batches - remainder if remainder > 0 else num_batches
        optimizer.zero_grad()
        seg_counts = {}
        for step_idx, (inputs, targets, slices) in enumerate(train_dl):
            seg_idx = slices[0].get("mqar_case", len(seg_counts))
            seg_counts.setdefault(seg_idx, 0)
            if seg_counts[seg_idx] >= TRAIN_BATCHES_PER_SEGMENT:
                continue
            seg_counts[seg_idx] += 1
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = loss_fn(rearrange(logits, "... c -> (...) c"), targets.flatten())
            aux_losses = []
            model.apply(
                lambda m: aux_losses.append(m.get_auxiliary_loss())
                if hasattr(m, "get_auxiliary_loss")
                else None
            )
            if aux_losses:
                loss = loss + sum(aux_losses)
            effective_accum = remainder if step_idx >= partial_start and remainder > 0 else accum_steps
            (loss / effective_accum).backward()
            if (step_idx + 1) % accum_steps == 0 or (step_idx + 1) == num_batches:
                optimizer.step()
                optimizer.zero_grad()
            result["train_steps"] += 1
            if (
                all(v >= TRAIN_BATCHES_PER_SEGMENT for v in seg_counts.values())
                and len(seg_counts) >= 5
            ):
                break
    except torch.cuda.OutOfMemoryError:
        result["status"] = "oom"
        result["oom_phase"] = "train"
        result["peak_mb"] = (
            torch.cuda.max_memory_allocated(device) // (1 << 20)
            if device.type == "cuda"
            else 0
        )
        _cleanup(model, optimizer)
        return result
    except Exception as exc:
        result["status"] = "train_error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        _cleanup(model, optimizer)
        return result

    try:
        model.eval()
        seg_counts_eval = {}
        with torch.no_grad():
            for inputs, targets, slices in test_dl:
                seg_idx = slices[0].get("mqar_case", len(seg_counts_eval))
                seg_counts_eval.setdefault(seg_idx, 0)
                if seg_counts_eval[seg_idx] >= EVAL_BATCHES_PER_SEGMENT:
                    continue
                seg_counts_eval[seg_idx] += 1
                inputs = inputs.to(device)
                targets = targets.to(device)
                logits = model(inputs)
                _ = loss_fn(rearrange(logits, "... c -> (...) c"), targets.flatten())
                result["eval_steps"] += 1
                if (
                    all(v >= EVAL_BATCHES_PER_SEGMENT for v in seg_counts_eval.values())
                    and len(seg_counts_eval) >= 8
                ):
                    break
    except torch.cuda.OutOfMemoryError:
        result["status"] = "oom"
        result["oom_phase"] = "eval"
        result["peak_mb"] = (
            torch.cuda.max_memory_allocated(device) // (1 << 20)
            if device.type == "cuda"
            else 0
        )
        _cleanup(model, optimizer)
        return result
    except Exception as exc:
        result["status"] = "eval_error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        _cleanup(model, optimizer)
        return result

    result["status"] = "pass"
    result["peak_mb"] = (
        torch.cuda.max_memory_allocated(device) // (1 << 20)
        if device.type == "cuda"
        else 0
    )
    _cleanup(model, optimizer)
    return result


def _write_smoke_outputs(*, part: str, payload: dict, chosen: dict | None) -> tuple[Path, Path]:
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_root = Path("/tmp")
    summary_path = output_root / f"flash_vqg_{part}_smoke_{timestamp}.json"
    env_path = output_root / f"flash_vqg_{part}_smoke_{timestamp}.env"
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    env_lines = [
        f"SMOKE_SUMMARY_JSON={summary_path}",
        f"SMOKE_ENV_FILE={env_path}",
    ]
    if chosen is not None:
        env_lines.extend(
            [
                f"TRAIN_BATCH_SIZE={chosen['train_batch_size']}",
                f"EVAL_BATCH_SIZE={chosen['eval_batch_size']}",
                f"GRADIENT_ACCUMULATION_STEPS={chosen['gradient_accumulation_steps']}",
                f"EFFECTIVE_TRAIN_BATCH_SIZE={chosen['effective_train_batch_size']}",
            ]
        )
    env_path.write_text("\n".join(env_lines) + "\n", encoding="utf-8")
    return summary_path, env_path


def main(part: str) -> None:
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
    parser.add_argument("--project", type=str, default="smoke")
    parser.add_argument("--entity", type=str, default="smoke")
    parser.add_argument("--launch-id-prefix", type=str, default="flash-vqg-e2-smoke")
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--analysis", type=str, default="off")
    parser.add_argument("--metrics-white-list", type=str, default=None)
    parser.add_argument("--metrics-white-list-file", type=str, default=None)
    namespace = parser.parse_args()

    device = torch.device(f"cuda:{namespace.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(int(namespace.gpu))
    builder = _load_builder(part)
    results = []
    chosen = None
    started_at = time.time()

    for train_batch_size, eval_batch_size, gradient_accumulation_steps in CANDIDATE_BATCH_COMBOS:
        print(
            f"[smoke] {part} tbs={train_batch_size} ebs={eval_batch_size} ga={gradient_accumulation_steps}",
            flush=True,
        )
        combo_args = _build_args(
            namespace,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        configs = builder(combo_args)
        combo_rows = []
        all_passed = True
        for config in configs:
            row = _run_one_config(config, device)
            row.update(
                {
                    "train_batch_size": train_batch_size,
                    "eval_batch_size": eval_batch_size,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "effective_train_batch_size": train_batch_size * gradient_accumulation_steps,
                    "experiment_part": part,
                }
            )
            combo_rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if row["status"] != "pass":
                all_passed = False
        results.extend(combo_rows)
        if all_passed:
            chosen = {
                "train_batch_size": train_batch_size,
                "eval_batch_size": eval_batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "effective_train_batch_size": train_batch_size * gradient_accumulation_steps,
            }
            break

    payload = {
        "experiment_part": part,
        "chosen": chosen,
        "results": results,
        "elapsed_sec": round(time.time() - started_at, 3),
    }
    summary_path, env_path = _write_smoke_outputs(part=part, payload=payload, chosen=chosen)
    print(f"SMOKE_SUMMARY_JSON={summary_path}")
    print(f"SMOKE_ENV_FILE={env_path}")
    if chosen is None:
        print("未找到可通过的 batch/GA 组合", flush=True)
        raise SystemExit(1)
    print(f"TRAIN_BATCH_SIZE={chosen['train_batch_size']}")
    print(f"EVAL_BATCH_SIZE={chosen['eval_batch_size']}")
    print(f"GRADIENT_ACCUMULATION_STEPS={chosen['gradient_accumulation_steps']}")
