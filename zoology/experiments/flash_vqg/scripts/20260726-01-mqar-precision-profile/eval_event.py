#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import traceback
from contextlib import nullcontext
from pathlib import Path
from typing import Any

LOCAL_SCRIPT_DIR = Path(__file__).resolve().parent
if str(LOCAL_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_SCRIPT_DIR))

from common import (  # noqa: E402
    GDN_KERNEL_DTYPE,
    REPO_ROOT,
    atomic_write_json,
    load_json,
    sha256_file,
    stable_json_sha256,
    utc_now,
)


def seed_all(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tensor_sha256(*tensors: Any) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(str(tuple(array.shape)).encode("utf-8"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def prediction_sha256(predictions: Any) -> str:
    return tensor_sha256(predictions)


def autocast_context(precision: str):
    import torch

    if precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def prepare_event(event: dict[str, Any]):
    import torch

    from zoology.checkpoints import load_checkpoint
    from zoology.data.multiquery_ar import MQARConfig
    from zoology.experiments.flash_vqg.eval_only import (
        _prepare_test_dataloader_from_data_config,
    )

    seed = int(event.get("eval_seed", 123))
    seed_all(seed)
    bundle = load_checkpoint(
        event["checkpoint_path"],
        which="last",
        device="cuda",
        strict=True,
    )
    config = bundle["config"].model_copy(deep=True)
    template = next(
        (
            item
            for item in list(config.data.test_configs)
            + list(config.data.train_configs)
            if isinstance(item, MQARConfig)
        ),
        None,
    )
    if template is None:
        raise TypeError("Checkpoint config has no MQARConfig.")
    payload = template.model_dump()
    payload.update(
        {
            "vocab_size": 8192,
            "num_examples": int(event["num_examples"]),
            "input_seq_len": int(event["input_seq_len"]),
            "num_kv_pairs": int(event["num_kv_pairs"]),
            "random_non_queries": True,
            "power_a": 0.01,
            "include_slices": True,
        }
    )
    config.data = config.data.model_copy(deep=True)
    config.data.seed = seed
    config.data.cache_dir = None
    config.data.force_cache = False
    config.data.test_configs = [MQARConfig(**payload)]
    config.data.test_batch_segment_order = None
    train_batch = (
        config.data.batch_size
        if isinstance(config.data.batch_size, int)
        else config.data.batch_size[0]
    )
    config.data.batch_size = (train_batch, int(event["eval_batch_size"]))
    seed_all(seed)
    dataloader = _prepare_test_dataloader_from_data_config(config.data)
    segment = dataloader.dataset.segments[0]
    dataset_hash = tensor_sha256(segment.inputs, segment.labels)
    expected_hash = event.get("expected_dataset_hash")
    if expected_hash and dataset_hash != expected_hash:
        raise RuntimeError(
            f"Dataset hash mismatch: expected={expected_hash}, actual={dataset_hash}"
        )
    checkpoint_hash = sha256_file(Path(event["checkpoint_path"]))
    if checkpoint_hash != event["checkpoint_file_sha256"]:
        raise RuntimeError("Checkpoint file hash mismatch.")
    return bundle["model"], dataloader, dataset_hash


def initial_progress(event: dict[str, Any], dataset_hash: str) -> dict[str, Any]:
    return {
        "format_version": 1,
        "event_identity_sha256": stable_json_sha256(event),
        "event_id": event["event_id"],
        "dataset_hash": dataset_hash,
        "next_batch_idx": 0,
        "processed_examples": 0,
        "sample_accuracy_values": [],
        "sample_loss_values": [],
        "prediction_batch_sha256": [],
        "prediction_sample_sha256": [],
        "query_correct": 0,
        "query_count": 0,
        "legacy_batch_loss_sum": 0.0,
        "legacy_batch_count": 0,
        "controlled_interrupt_done": False,
        "started_at_utc": utc_now(),
    }


def load_progress(
    event: dict[str, Any],
    progress_path: Path,
    dataset_hash: str,
) -> dict[str, Any]:
    if not progress_path.exists():
        return initial_progress(event, dataset_hash)
    progress = load_json(progress_path)
    if progress.get("event_identity_sha256") != stable_json_sha256(event):
        raise RuntimeError("Eval progress identity mismatch.")
    if progress.get("dataset_hash") != dataset_hash:
        raise RuntimeError("Eval progress dataset hash mismatch.")
    return progress


def batch_statistics(logits, targets):
    import torch
    import torch.nn.functional as functional

    predictions = logits.argmax(dim=-1)
    mask = targets != -100
    token_loss = functional.cross_entropy(
        logits.float().reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(targets)
    counts = mask.sum(dim=-1).clamp_min(1)
    sample_loss = (token_loss * mask).sum(dim=-1) / counts
    sample_accuracy = ((predictions == targets) & mask).sum(dim=-1).float() / counts
    query_correct = int(((predictions == targets) & mask).sum().item())
    query_count = int(mask.sum().item())
    legacy_loss = functional.cross_entropy(
        logits.float().reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=-100,
    )
    return (
        predictions,
        sample_loss.detach().cpu().tolist(),
        sample_accuracy.detach().cpu().tolist(),
        query_correct,
        query_count,
        float(legacy_loss.detach().cpu().item()),
    )


def collect_scalar_metrics(model) -> dict[str, float]:
    values: dict[str, float] = {}

    def collect(module):
        getter = getattr(module, "get_scalar_metrics", None)
        if getter is None:
            return
        for key, value in (getter() or {}).items():
            values[str(key)] = float(value)

    model.apply(collect)
    return values


def run_event(
    event: dict[str, Any],
    progress_path: Path,
    result_path: Path,
) -> int:
    import torch

    started = time.perf_counter()
    os.environ["GDN_KERNEL_DTYPE"] = GDN_KERNEL_DTYPE[event["eval_precision"]]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model, dataloader, dataset_hash = prepare_event(event)
        model.eval()
        progress = load_progress(event, progress_path, dataset_hash)
        cursor = int(progress["next_batch_idx"])
        with torch.no_grad():
            for batch_idx, (inputs, targets, _slices) in enumerate(dataloader):
                if batch_idx < cursor:
                    continue
                max_batches = int(event.get("max_batches", 0))
                if max_batches > 0 and batch_idx >= max_batches:
                    break
                inputs = inputs.cuda(non_blocking=False)
                targets = targets.cuda(non_blocking=False)
                with autocast_context(event["eval_precision"]):
                    logits = model(inputs)
                (
                    predictions,
                    sample_loss,
                    sample_accuracy,
                    query_correct,
                    query_count,
                    legacy_loss,
                ) = batch_statistics(logits, targets)
                progress["sample_loss_values"].extend(sample_loss)
                progress["sample_accuracy_values"].extend(sample_accuracy)
                progress["prediction_batch_sha256"].append(
                    prediction_sha256(predictions)
                )
                progress["prediction_sample_sha256"].extend(
                    prediction_sha256(row) for row in predictions
                )
                progress["query_correct"] += query_correct
                progress["query_count"] += query_count
                progress["legacy_batch_loss_sum"] += legacy_loss
                progress["legacy_batch_count"] += 1
                progress["processed_examples"] += int(targets.size(0))
                progress["next_batch_idx"] = batch_idx + 1
                progress["updated_at_utc"] = utc_now()
                progress["peak_allocated_mib"] = (
                    torch.cuda.max_memory_allocated() / 1024**2
                )
                progress["peak_reserved_mib"] = (
                    torch.cuda.max_memory_reserved() / 1024**2
                )
                atomic_write_json(progress_path, progress)
                interrupt_after = int(event.get("controlled_interrupt_after_batches", 0))
                if (
                    interrupt_after > 0
                    and not progress["controlled_interrupt_done"]
                    and progress["next_batch_idx"] >= interrupt_after
                ):
                    progress["controlled_interrupt_done"] = True
                    atomic_write_json(progress_path, progress)
                    atomic_write_json(
                        result_path,
                        {
                            "status": "controlled_stop",
                            "event_id": event["event_id"],
                            "next_batch_idx": progress["next_batch_idx"],
                        },
                    )
                    return 75

        dataset_examples = int(event["num_examples"])
        max_batches = int(event.get("max_batches", 0))
        expected_examples = (
            min(dataset_examples, max_batches * int(event["eval_batch_size"]))
            if max_batches > 0
            else dataset_examples
        )
        if int(progress["processed_examples"]) != expected_examples:
            raise RuntimeError(
                f"Processed {progress['processed_examples']} examples, "
                f"expected {expected_examples}."
            )
        sample_accuracy = progress["sample_accuracy_values"]
        sample_loss = progress["sample_loss_values"]
        result = {
            "status": "completed",
            "event_id": event["event_id"],
            "dataset_hash": dataset_hash,
            "checkpoint_file_sha256": event["checkpoint_file_sha256"],
            "num_examples": expected_examples,
            "dataset_num_examples": dataset_examples,
            "eval_batch_size": int(event["eval_batch_size"]),
            "eval_precision": event["eval_precision"],
            "accuracy": sum(sample_accuracy) / len(sample_accuracy),
            "loss_sample_weighted": sum(sample_loss) / len(sample_loss),
            "legacy_loss": (
                progress["legacy_batch_loss_sum"]
                / progress["legacy_batch_count"]
            ),
            "query_accuracy": (
                progress["query_correct"] / progress["query_count"]
            ),
            "query_correct": progress["query_correct"],
            "query_count": progress["query_count"],
            "sample_accuracy_values": sample_accuracy,
            "sample_loss_values": sample_loss,
            "prediction_batch_sha256": progress["prediction_batch_sha256"],
            "prediction_sample_sha256": progress["prediction_sample_sha256"],
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
            "wall_clock_sec": time.perf_counter() - started,
            "started_at_utc": progress["started_at_utc"],
            "ended_at_utc": utc_now(),
            "model_scalar_metrics": collect_scalar_metrics(model),
            "progress_path": str(progress_path.resolve()),
        }
        atomic_write_json(result_path, result)
        return 0
    except BaseException as exc:
        detail = f"{type(exc).__name__}: {exc}"
        trace = traceback.format_exc()
        is_oom = bool(re.search(r"out of memory|CUBLAS_STATUS_ALLOC_FAILED", detail + trace, re.I))
        result = {
            "status": "oom" if is_oom else "failed",
            "event_id": event.get("event_id"),
            "failure_type": "oom" if is_oom else type(exc).__name__,
            "failure_detail": detail,
            "traceback_tail": "\n".join(trace.splitlines()[-30:]),
            "wall_clock_sec": time.perf_counter() - started,
            "ended_at_utc": utc_now(),
        }
        if torch.cuda.is_available():
            result["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 1024**2
            result["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 1024**2
        atomic_write_json(result_path, result)
        print(trace, file=sys.stderr)
        return 2 if is_oom else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    return run_event(load_json(args.event), args.progress, args.result)


if __name__ == "__main__":
    raise SystemExit(main())
