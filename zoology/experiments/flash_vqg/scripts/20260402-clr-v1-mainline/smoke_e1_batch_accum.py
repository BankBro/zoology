#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from einops import rearrange


REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


CANDIDATE_BATCH_COMBOS = [
    (256, 32, 1),
    (128, 16, 2),
    (64, 16, 4),
    (32, 8, 8),
    (16, 8, 16),
]
TRAIN_BATCHES_PER_SEGMENT = 1
EVAL_BATCHES_PER_SEGMENT = 1


def _build_one_config(
    *,
    read_topk: int | None,
    train_batch_size: int,
    eval_batch_size: int,
    gradient_accumulation_steps: int,
    metrics_white_list: list[str],
):
    from zoology.experiments.flash_vqg.flash_vqg_suite import build_configs

    configs = build_configs(
        sweep_id="smoke-clr-v1-e1",
        flash_backend="torch",
        logger_backend="none",
        include_gdn=False,
        block_len=32,
        local_num_blocks=2,
        dmodels=[128],
        learning_rates=[1e-3],
        if_remote_enabled=True,
        train_batch_order="global_shuffle",
        seed_values=[123],
        data_seed=123,
        num_codebook_vectors_values=[128],
        fox_remote_path_backend="torch",
        fox_remote_read_topk_values=[read_topk],
        fox_remote_formula="clr_v1",
        fox_clr_rank=4,
        fox_clr_use_den_residual=True,
        fox_clr_remat_mode="off",
        gradient_accumulation_steps=gradient_accumulation_steps,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        cache_dir="./data/flash_vqg",
        wandb_project="smoke",
        wandb_entity="smoke",
        max_epochs=1,
        metrics_white_list=metrics_white_list,
    )
    if len(configs) != 1:
        raise RuntimeError(f"Expected 1 config, got {len(configs)} for read_topk={read_topk!r}")
    return configs[0]


def _cleanup(model, optimizer):
    del model
    del optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_one_mode(
    *,
    read_topk: int | None,
    train_batch_size: int,
    eval_batch_size: int,
    gradient_accumulation_steps: int,
    device: torch.device,
    metrics_white_list: list[str],
) -> dict:
    from zoology.data.utils import prepare_data
    from zoology.model import LanguageModel

    read_mode = "dense" if read_topk is None else f"top{int(read_topk)}"
    result = {
        "read_mode": read_mode,
        "train_batch_size": int(train_batch_size),
        "eval_batch_size": int(eval_batch_size),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "effective_train_batch_size": int(train_batch_size) * int(gradient_accumulation_steps),
        "status": "unknown",
        "oom_phase": None,
        "train_steps": 0,
        "eval_steps": 0,
        "peak_mb": 0,
        "error": None,
    }

    try:
        config = _build_one_config(
            read_topk=read_topk,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            metrics_white_list=metrics_white_list,
        )
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
        seg_counts: dict[int, int] = {}
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

            is_accum_boundary = (step_idx + 1) % accum_steps == 0
            is_last_batch = (step_idx + 1) == num_batches
            if is_accum_boundary or is_last_batch:
                optimizer.step()
                optimizer.zero_grad()
            result["train_steps"] += 1

            if all(v >= TRAIN_BATCHES_PER_SEGMENT for v in seg_counts.values()) and len(seg_counts) >= 5:
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
        seg_counts_eval: dict[int, int] = {}
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

                if all(v >= EVAL_BATCHES_PER_SEGMENT for v in seg_counts_eval.values()) and len(seg_counts_eval) >= 8:
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


def _write_summary(results: list[dict]) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_path = Path("/tmp") / f"flash_vqg_clr_v1_e1_smoke_{timestamp}.json"
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="clr_v1 实验1 batch/GA smoke")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--metrics-white-list-file", type=str, required=True)
    args = parser.parse_args()

    from zoology.experiments.flash_vqg.metrics_white_list import load_metrics_white_list_file

    metrics_white_list = load_metrics_white_list_file(args.metrics_white_list_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(int(args.gpu))

    results: list[dict] = []
    recommended: tuple[int, int, int] | None = None

    for train_batch_size, eval_batch_size, gradient_accumulation_steps in CANDIDATE_BATCH_COMBOS:
        print(
            f"[smoke] top4 tbs={train_batch_size} ebs={eval_batch_size} ga={gradient_accumulation_steps}",
            flush=True,
        )
        t0 = time.time()
        top4_result = _run_one_mode(
            read_topk=4,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            device=device,
            metrics_white_list=metrics_white_list,
        )
        top4_result["elapsed_s"] = round(time.time() - t0, 2)
        results.append(top4_result)
        print(json.dumps(top4_result, ensure_ascii=False), flush=True)
        if top4_result["status"] != "pass":
            continue

        print(
            f"[smoke] dense sanity tbs={train_batch_size} ebs={eval_batch_size} ga={gradient_accumulation_steps}",
            flush=True,
        )
        t0 = time.time()
        dense_result = _run_one_mode(
            read_topk=None,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            device=device,
            metrics_white_list=metrics_white_list,
        )
        dense_result["elapsed_s"] = round(time.time() - t0, 2)
        results.append(dense_result)
        print(json.dumps(dense_result, ensure_ascii=False), flush=True)
        if dense_result["status"] == "pass":
            recommended = (train_batch_size, eval_batch_size, gradient_accumulation_steps)
            break

    summary_path = _write_summary(results)
    print(f"[smoke] summary={summary_path}", flush=True)
    if recommended is None:
        raise SystemExit(1)

    train_batch_size, eval_batch_size, gradient_accumulation_steps = recommended
    print(
        "RECOMMENDED "
        f"TRAIN_BATCH_SIZE={train_batch_size} "
        f"EVAL_BATCH_SIZE={eval_batch_size} "
        f"GRADIENT_ACCUMULATION_STEPS={gradient_accumulation_steps}",
        flush=True,
    )


if __name__ == "__main__":
    main()
