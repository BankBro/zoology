#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


def find_repo_root(start: Path) -> Path:
    current = start if start.is_dir() else start.parent
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists() and (candidate / "zoology").is_dir():
            return candidate
    raise RuntimeError(f"Cannot locate zoology repo root from {start}.")


ROOT = find_repo_root(Path(__file__).resolve())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARTIFACT_DIR = ROOT / "docs/artifacts/longer-mqar/local-window-fairness-20260529"
EVAL_ONLY_ARTIFACT_DIR = ROOT / "docs/artifacts/20260529-flash-local-window-fairness-eval-only"
STAGE3_TRAIN_SUMMARY = ROOT / "docs/artifacts/20260529-flash-local-window-fairness/stage3_train_summary.csv"
OFFICIAL_CORE_DIR = ROOT / "docs/artifacts/longer-mqar/official-core-20260526"
OFFICIAL_MANIFEST = OFFICIAL_CORE_DIR / "manifest.csv"
OFFICIAL_DETAIL = OFFICIAL_CORE_DIR / "longer-mqar-official-core-detail.csv"
FLASH_VQG_ROOT = Path("/home/lyj/mnt/project/Flash-VQG")
EVAL_SEED = 123
SANITY_TOL = 1e-4
DEFAULT_NUM_EXAMPLES = 500
DEFAULT_GROUPS = ["cb64-r16", "cb256-r4", "gdn-h2-ev8", "gdn-h2-ev10", "gdn-h2-ev16"]
DEFAULT_SEEDS = ["123"]
DEFAULT_SLICES = [(1024, 256), (2048, 512), (4096, 512), (4096, 1024)]
BUCKETS = [
    ("<=32", None, 32),
    ("33-64", 33, 64),
    ("65-128", 65, 128),
    ("129-256", 129, 256),
    ("257-512", 257, 512),
    ("513-1024", 513, 1024),
    ("1025-2048", 1025, 2048),
    ("2049-4096", 2049, 4096),
    (">4096", 4097, None),
]
VARIANT_OVERRIDES = {
    "full": {},
    "local_only": {"local_num_blocks": 2, "if_remote_enabled": False},
    "local1": {"local_num_blocks": 1, "if_remote_enabled": True},
    "local4": {"local_num_blocks": 4, "if_remote_enabled": True},
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def run_cmd_text(cmd: list[str], cwd: Path = ROOT) -> str:
    try:
        return subprocess.check_output(cmd, cwd=str(cwd), text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def git_commit(path: Path) -> str:
    return run_cmd_text(["git", "rev-parse", "HEAD"], cwd=path)


def git_dirty(path: Path) -> str:
    return "true" if run_cmd_text(["git", "status", "--short"], cwd=path) else "false"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_sha256(*tensors: torch.Tensor) -> str:
    h = hashlib.sha256()
    for tensor in tensors:
        arr = tensor.detach().cpu().contiguous().numpy()
        h.update(str(arr.dtype).encode("utf-8"))
        h.update(str(tuple(arr.shape)).encode("utf-8"))
        h.update(arr.tobytes())
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_status(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"- {now_utc()} {line}\n")


def parse_csv_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_slices(value: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for raw in parse_csv_list(value):
        match = re.fullmatch(r"(\d+)x(\d+)", raw)
        if not match:
            raise ValueError(f"Invalid slice `{raw}`, expected e.g. 4096x512.")
        out.append((int(match.group(1)), int(match.group(2))))
    return out


def group_label(row: dict[str, str]) -> str:
    family = row.get("source_config_family", "")
    if row.get("source_model_family") == "gdn":
        return family.replace("-usegate0", "")
    return family


def model_label_for(row: dict[str, str]) -> str:
    return f"{group_label(row)}-s{row.get('source_seed', '')}"


def load_official_detail_index(path: Path = OFFICIAL_DETAIL) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    if not path.exists():
        return index
    for row in read_csv(path):
        if row.get("eval_mode") != "formal" or row.get("eval_status") != "completed":
            continue
        source_run_id = row.get("source_run_id", "")
        if source_run_id and source_run_id not in index:
            index[source_run_id] = row
    return index


def load_official_accuracy_refs(path: Path = OFFICIAL_DETAIL) -> dict[tuple[str, int, int], dict[str, str]]:
    refs: dict[tuple[str, int, int], dict[str, str]] = {}
    if not path.exists():
        return refs
    for row in read_csv(path):
        if row.get("eval_mode") != "formal" or row.get("eval_status") != "completed":
            continue
        try:
            key = (row["source_run_id"], int(row["input_seq_len"]), int(row["num_kv_pairs"]))
        except Exception:
            continue
        refs[key] = {
            "official_accuracy_ref": row.get("accuracy", ""),
            "official_dataset_hash_ref": row.get("dataset_hash", ""),
            "official_eval_event_id_ref": row.get("eval_event_id", ""),
        }
    return refs


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


SOURCE_FIELDS = [
    "model_label",
    "kind",
    "checkpoint_path",
    "checkpoint_path_rel",
    "checkpoint_which",
    "checkpoint_file_path",
    "checkpoint_file_path_rel",
    "source_run",
    "source_run_id",
    "source_ledger",
    "seed",
    "git_commit",
    "flash_vqg_commit",
    "machine",
    "dtype_policy",
    "status",
    "source_model_family",
    "source_config_family",
    "source_config",
    "source_scope",
    "source_batch_accum_profile",
    "source_train_config_path",
    "source_train_config_path_abs",
    "source_train_config_sha256",
    "source_ckpt_sha256",
    "source_ckpt_epoch",
    "source_trainable_params",
    "source_dynamic_capacity_total",
    "stage3_variant",
    "stage3_launch_id",
    "stage3_run_id",
    "stage3_valid_accuracy",
]


def prepare_sources(args: argparse.Namespace) -> int:
    manifest_path = Path(args.official_manifest)
    detail_path = Path(args.official_detail)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing official manifest: {manifest_path}")
    detail_index = load_official_detail_index(detail_path)
    groups = set(parse_csv_list(args.groups))
    seeds = set(parse_csv_list(args.seeds))
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for row in read_csv(manifest_path):
        label = group_label(row)
        seed = row.get("source_seed", "")
        if not args.include_all_official and (label not in groups or seed not in seeds):
            continue
        ckpt = resolve_path(row.get("source_ckpt_path_abs") or row.get("source_ckpt_path") or "")
        train_config = resolve_path(row.get("source_train_config_path_abs") or row.get("source_train_config_path") or "")
        detail = detail_index.get(row.get("source_run_id", ""), {})
        status = "completed"
        if not ckpt.exists():
            status = "missing_checkpoint"
            errors.append(f"{row.get('source_run_id')}: checkpoint missing: {ckpt}")
        if not train_config.exists():
            status = "missing_train_config"
            errors.append(f"{row.get('source_run_id')}: train_config missing: {train_config}")
        git_value = detail.get("commit_zoology", "")
        flash_commit = detail.get("commit_flash_vqg", "")
        if not git_value:
            errors.append(f"{row.get('source_run_id')}: missing commit_zoology in official detail")
        if not flash_commit:
            errors.append(f"{row.get('source_run_id')}: missing commit_flash_vqg in official detail")
        if not row.get("source_dtype_policy"):
            errors.append(f"{row.get('source_run_id')}: missing source_dtype_policy")
        rows.append({
            "model_label": model_label_for(row),
            "kind": row.get("source_model_family", ""),
            "checkpoint_path": str(ckpt),
            "checkpoint_path_rel": rel(ckpt),
            "source_run": row.get("source_run_id", ""),
            "source_run_id": row.get("source_run_id", ""),
            "source_ledger": row.get("source_ledger_path", ""),
            "seed": seed,
            "git_commit": git_value,
            "flash_vqg_commit": flash_commit,
            "machine": "mclab-3090",
            "dtype_policy": row.get("source_dtype_policy", ""),
            "status": status,
            "source_model_family": row.get("source_model_family", ""),
            "source_config_family": row.get("source_config_family", ""),
            "source_config": row.get("source_config", ""),
            "source_scope": row.get("source_scope", ""),
            "source_batch_accum_profile": row.get("source_batch_accum_profile", ""),
            "source_train_config_path": rel(train_config),
            "source_train_config_path_abs": str(train_config),
            "source_train_config_sha256": row.get("source_train_config_sha256", ""),
            "source_ckpt_sha256": row.get("source_ckpt_sha256", ""),
            "source_ckpt_epoch": row.get("source_ckpt_epoch", ""),
            "source_trainable_params": row.get("source_trainable_params", ""),
            "source_dynamic_capacity_total": row.get("source_dynamic_capacity_total", ""),
        })
    if not rows:
        raise RuntimeError("No source checkpoints selected.")
    seen: set[str] = set()
    duplicates: list[str] = []
    for row in rows:
        key = row["model_label"]
        if key in seen:
            duplicates.append(key)
        seen.add(key)
    if duplicates:
        errors.append(f"duplicate model_label values: {sorted(duplicates)}")
    if errors and not args.allow_missing:
        raise RuntimeError("source checkpoint prepare failed:\n" + "\n".join(errors))
    output = Path(args.output)
    write_csv(output, rows, SOURCE_FIELDS)
    metadata = {
        "created_at_utc": now_utc(),
        "command": " ".join(sys.argv),
        "source": "prepare_source_checkpoints",
        "official_manifest": rel(manifest_path),
        "official_detail": rel(detail_path),
        "groups": sorted(groups),
        "seeds": sorted(seeds),
        "include_all_official": bool(args.include_all_official),
        "row_count": len(rows),
        "status": "completed_with_warnings" if errors else "completed",
        "errors": errors,
        "git_commit": git_commit(ROOT),
        "git_dirty": git_dirty(ROOT),
        "flash_vqg_commit": git_commit(FLASH_VQG_ROOT),
        "flash_vqg_dirty": git_dirty(FLASH_VQG_ROOT),
    }
    write_json(output.parent / "metadata.json", metadata)
    append_status(output.parent / "status.md", f"prepare_source_checkpoints wrote {rel(output)} rows={len(rows)} status={metadata['status']}")
    return 0

def prepare_stage3_sources(args: argparse.Namespace) -> int:
    summary_path = Path(args.stage3_summary)
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing stage3 summary: {summary_path}")
    checkpoint_kinds = parse_csv_list(args.checkpoint_kinds)
    allowed_kinds = {"best", "last"}
    unknown = sorted(set(checkpoint_kinds) - allowed_kinds)
    if unknown:
        raise ValueError(f"unknown checkpoint kinds: {unknown}; expected best,last")

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for row in read_csv(summary_path):
        if row.get("status") != "completed":
            continue
        variant = row.get("variant", "")
        for checkpoint_which in checkpoint_kinds:
            ckpt_raw = row.get(f"checkpoint_{checkpoint_which}", "")
            if not ckpt_raw:
                errors.append(f"{variant}: missing checkpoint_{checkpoint_which}")
                continue
            ckpt = resolve_path(ckpt_raw)
            train_config = ckpt.parent / "train_config.json"
            status = "completed"
            if not ckpt.exists():
                status = "missing_checkpoint"
                errors.append(f"{variant}:{checkpoint_which}: checkpoint missing: {ckpt}")
            if not train_config.exists():
                status = "missing_train_config"
                errors.append(f"{variant}:{checkpoint_which}: train_config missing: {train_config}")
            model_label = f"stage3-{variant}-{checkpoint_which}"
            run_id = row.get("run_id", "")
            rows.append({
                "model_label": model_label,
                "kind": "flash",
                "checkpoint_path": str(ckpt),
                "checkpoint_path_rel": rel(ckpt),
                "checkpoint_which": checkpoint_which,
                "checkpoint_file_path": str(ckpt),
                "checkpoint_file_path_rel": rel(ckpt),
                "source_run": f"{run_id}:{checkpoint_which}",
                "source_run_id": f"{run_id}:{checkpoint_which}",
                "source_ledger": rel(summary_path),
                "seed": "123",
                "git_commit": git_commit(ROOT),
                "flash_vqg_commit": git_commit(FLASH_VQG_ROOT),
                "machine": "mclab-3090",
                "dtype_policy": "torch-fp32; GDN_KERNEL_DTYPE=float32",
                "status": status,
                "source_model_family": "flash",
                "source_config_family": f"stage3-{variant}",
                "source_config": row.get("run_id", ""),
                "source_scope": "stage3_training_ablation_eval_only",
                "source_batch_accum_profile": row.get("run_id", ""),
                "source_train_config_path": rel(train_config),
                "source_train_config_path_abs": str(train_config),
                "source_train_config_sha256": sha256_file(train_config) if train_config.exists() else "",
                "source_ckpt_sha256": sha256_file(ckpt) if ckpt.exists() else "",
                "source_ckpt_epoch": "",
                "source_trainable_params": "",
                "source_dynamic_capacity_total": "",
                "stage3_variant": variant,
                "stage3_launch_id": row.get("launch_id", ""),
                "stage3_run_id": run_id,
                "stage3_valid_accuracy": row.get("valid_accuracy", ""),
            })
    if not rows:
        raise RuntimeError("No completed stage3 checkpoints selected.")
    if errors and not args.allow_missing:
        raise RuntimeError("stage3 source checkpoint prepare failed:\n" + "\n".join(errors))

    output = Path(args.output)
    write_csv(output, rows, SOURCE_FIELDS)
    metadata = {
        "created_at_utc": now_utc(),
        "command": " ".join(sys.argv),
        "source": "prepare_stage3_source_checkpoints",
        "stage3_summary": rel(summary_path),
        "checkpoint_kinds": checkpoint_kinds,
        "row_count": len(rows),
        "status": "completed_with_warnings" if errors else "completed",
        "errors": errors,
        "git_commit": git_commit(ROOT),
        "git_dirty": git_dirty(ROOT),
        "flash_vqg_commit": git_commit(FLASH_VQG_ROOT),
        "flash_vqg_dirty": git_dirty(FLASH_VQG_ROOT),
    }
    write_json(output.parent / "metadata.json", metadata)
    append_status(output.parent / "status.md", f"prepare_stage3_source_checkpoints wrote {rel(output)} rows={len(rows)} status={metadata['status']}")
    return 0


@dataclass
class MQARWithMetadata:
    inputs: torch.Tensor
    labels: torch.Tensor
    query_pos: torch.Tensor
    key_pos: torch.Tensor
    value_pos: torch.Tensor
    distance_value: torch.Tensor
    distance_key: torch.Tensor
    gaps: torch.Tensor
    seed: int


def set_eval_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def build_near_distance_gaps(
    *,
    num_examples: int,
    input_seq_len: int,
    num_kv_pairs: int,
    seed: int,
    near_pairs_per_bucket: int = 16,
) -> np.ndarray:
    context_size = num_kv_pairs * 2
    space = (input_seq_len - context_size) // 2
    if space < num_kv_pairs:
        raise ValueError("near-distance MQAR requires query slot space >= num_kv_pairs")
    pairs_per_bucket = min(int(near_pairs_per_bucket), 16, num_kv_pairs // 3)
    if pairs_per_bucket <= 0:
        raise ValueError("near_pairs_per_bucket resolved to zero")
    # distance = query_pos - value_pos = 2 * (gap + tail_offset) + 1.
    # Each example enriches one target bucket. The first bucket uses distance 31
    # to keep 16 unique near pairs; the other buckets sample uniformly inside range.
    bucket_margin_ranges = [(15, 15), (16, 31), (32, 63)]
    if pairs_per_bucket - 1 > bucket_margin_ranges[0][1]:
        raise ValueError("near_pairs_per_bucket cannot exceed the <=32 bucket capacity")
    if bucket_margin_ranges[-1][1] >= space:
        raise ValueError(f"near-distance assignment needs gap {bucket_margin_ranges[-1][1]}, but space is {space}")

    rng = np.random.default_rng(seed)
    all_gaps = np.empty((num_examples, num_kv_pairs), dtype=np.int64)
    for example_idx in range(num_examples):
        row = np.full(num_kv_pairs, -1, dtype=np.int64)
        used: set[int] = set()
        margin_lo, margin_hi = bucket_margin_ranges[example_idx % len(bucket_margin_ranges)]
        margin = int(rng.integers(margin_lo, margin_hi + 1))
        enriched_pairs = min(pairs_per_bucket, margin + 1)
        for tail_offset in range(enriched_pairs):
            pair_idx = num_kv_pairs - 1 - tail_offset
            gap = margin - tail_offset
            row[pair_idx] = gap
            used.add(gap)
        remaining_pairs = np.where(row < 0)[0]
        remaining_gaps = np.array([gap for gap in range(space) if gap not in used], dtype=np.int64)
        chosen = rng.choice(remaining_gaps, size=len(remaining_pairs), replace=False)
        rng.shuffle(chosen)
        row[remaining_pairs] = chosen
        all_gaps[example_idx] = row
    return all_gaps


def build_mqar_with_metadata(
    *,
    vocab_size: int,
    num_examples: int,
    input_seq_len: int,
    seed: int,
    power_a: float = 0.01,
    num_kv_pairs: int = 8,
    num_passes: int = 1,
    random_non_queries: bool = True,
    gaps_override: np.ndarray | None = None,
) -> MQARWithMetadata:
    if input_seq_len % 2 != 0:
        raise ValueError("input_seq_len must be even")
    if vocab_size <= input_seq_len:
        raise ValueError("vocab_size must be > input_seq_len")
    if num_kv_pairs * 2 * num_passes + num_kv_pairs * 2 > input_seq_len:
        raise ValueError("MQAR context and query slots do not fit input_seq_len")

    np.random.seed(seed)
    context_size = num_kv_pairs * 2 * num_passes
    key_vocab_size = vocab_size // 2
    key_choices = np.arange(1, key_vocab_size)
    value_choices = np.arange(key_vocab_size, vocab_size)
    keys_unshuffled = np.tile(key_choices, (num_examples, 1))
    keys = np.apply_along_axis(np.random.choice, 1, keys_unshuffled, replace=False, size=num_kv_pairs)
    values_unshuffled = np.tile(value_choices, (num_examples, 1))
    values = np.apply_along_axis(np.random.choice, 1, values_unshuffled, replace=False, size=num_kv_pairs)

    # Mirrors zoology.data.multiquery_ar.multiquery_ar for the pass-1 official task.
    kvs = np.zeros((num_examples, context_size), dtype=np.int64)
    kvs[:, 0::2] = keys
    kvs[:, 1::2] = values
    kvs = np.tile(kvs, (1, num_passes))

    space = (input_seq_len - context_size) // 2
    if gaps_override is None:
        p = power_a * np.arange(1, space + 1) ** (power_a - 1)
        p = p / p.sum()
        x = np.stack([np.arange(space, dtype=int)] * num_examples)
        gaps = np.apply_along_axis(np.random.choice, axis=1, arr=x, replace=False, p=p, size=num_kv_pairs)
    else:
        gaps = np.asarray(gaps_override, dtype=np.int64)
        if gaps.shape != (num_examples, num_kv_pairs):
            raise ValueError(f"gaps_override shape must be {(num_examples, num_kv_pairs)}, got {gaps.shape}")
        if (gaps < 0).any() or (gaps >= space).any():
            raise ValueError("gaps_override contains out-of-range query gaps")
        if any(len(set(row.tolist())) != num_kv_pairs for row in gaps):
            raise ValueError("gaps_override must not repeat query gaps within an example")

    queries = np.zeros((num_examples, input_seq_len - context_size + 1), dtype=np.int64)
    np.put_along_axis(queries, gaps * 2, values=keys, axis=1)
    examples = np.concatenate([kvs, queries], axis=1)
    labels = np.full((num_examples, input_seq_len + 1), -100, dtype=np.int64)
    np.put_along_axis(labels, gaps * 2 + context_size + 1, values=values, axis=1)
    inputs_t = torch.tensor(examples[:, :-1])
    labels_t = torch.tensor(labels[:, 1:])

    if random_non_queries:
        zero_mask = inputs_t == 0
        random_values = torch.randint(vocab_size, size=inputs_t.shape)
        inputs_t[zero_mask] = random_values[zero_mask]

    last_pass_base = (num_passes - 1) * num_kv_pairs * 2
    pair_offsets = np.arange(num_kv_pairs, dtype=np.int64) * 2
    key_pos = last_pass_base + pair_offsets
    value_pos = key_pos + 1
    query_pos = context_size + gaps * 2
    key_pos_2d = np.tile(key_pos, (num_examples, 1))
    value_pos_2d = np.tile(value_pos, (num_examples, 1))
    return MQARWithMetadata(
        inputs=inputs_t,
        labels=labels_t,
        query_pos=torch.tensor(query_pos, dtype=torch.long),
        key_pos=torch.tensor(key_pos_2d, dtype=torch.long),
        value_pos=torch.tensor(value_pos_2d, dtype=torch.long),
        distance_value=torch.tensor(query_pos - value_pos_2d, dtype=torch.long),
        distance_key=torch.tensor(query_pos - key_pos_2d, dtype=torch.long),
        gaps=torch.tensor(gaps, dtype=torch.long),
        seed=int(seed),
    )


def derive_single_test_seed(data_config: Any) -> int:
    max_seed = 2**32
    np.random.seed(int(data_config.seed))
    _ = np.random.randint(0, max_seed // 2, size=len(data_config.train_configs))
    return int(np.random.randint(max_seed // 2, max_seed, size=1)[0])


def find_mqar_template(config: Any) -> Any:
    from zoology.data.multiquery_ar import MQARConfig

    for candidate in list(config.data.test_configs) + list(config.data.train_configs):
        if isinstance(candidate, MQARConfig):
            return candidate
    raise TypeError("checkpoint config does not contain MQARConfig")


def build_eval_dataset(
    config: Any,
    *,
    seq_len: int,
    num_kv_pairs: int,
    num_examples: int,
    eval_seed: int,
    dataset_mode: str = "official",
    near_pairs_per_bucket: int = 16,
) -> MQARWithMetadata:
    from zoology.data.multiquery_ar import MQARConfig

    template = find_mqar_template(config)
    payload = template.model_dump()
    payload.update({
        "vocab_size": 8192,
        "num_examples": int(num_examples),
        "input_seq_len": int(seq_len),
        "num_kv_pairs": int(num_kv_pairs),
        "random_non_queries": True,
        "power_a": 0.01,
        "include_slices": True,
    })
    data_config = config.data.model_copy(deep=True)
    data_config.seed = int(eval_seed)
    data_config.cache_dir = None
    data_config.force_cache = False
    data_config.test_configs = [MQARConfig(**payload)]
    test_seed = derive_single_test_seed(data_config)
    set_eval_seed(eval_seed)
    gaps_override = None
    if dataset_mode == "near_enriched":
        gaps_override = build_near_distance_gaps(
            num_examples=int(payload["num_examples"]),
            input_seq_len=int(payload["input_seq_len"]),
            num_kv_pairs=int(payload["num_kv_pairs"]),
            seed=test_seed,
            near_pairs_per_bucket=int(near_pairs_per_bucket),
        )
    elif dataset_mode != "official":
        raise ValueError(f"unknown dataset_mode `{dataset_mode}`")
    return build_mqar_with_metadata(
        vocab_size=int(payload["vocab_size"]),
        num_examples=int(payload["num_examples"]),
        input_seq_len=int(payload["input_seq_len"]),
        seed=test_seed,
        power_a=float(payload.get("power_a", 0.01)),
        num_kv_pairs=int(payload["num_kv_pairs"]),
        num_passes=int(payload.get("num_passes", 1)),
        random_non_queries=bool(payload.get("random_non_queries", True)),
        gaps_override=gaps_override,
    )


def is_flash_source(source: dict[str, str]) -> bool:
    return source.get("source_model_family") == "flash" or source.get("kind") == "flash"


def apply_flash_override(model: torch.nn.Module, override: dict[str, Any]) -> dict[str, Any]:
    actual: dict[str, Any] = {}
    touched = 0
    for module in model.modules():
        if module.__class__.__name__ != "FlashVQGMixer":
            continue
        touched += 1
        before = {
            "block_len": int(getattr(module, "block_len", getattr(module.attn.config, "block_len", -1))),
            "local_num_blocks": int(getattr(module, "local_num_blocks", getattr(module.attn.config, "local_num_blocks", -1))),
            "if_remote_enabled": bool(getattr(module, "if_remote_enabled", getattr(module.attn.config, "if_remote_enabled", False))),
        }
        for key, raw_value in override.items():
            value = bool(raw_value) if key == "if_remote_enabled" else int(raw_value)
            setattr(module, key, value)
            if hasattr(module, "attn") and hasattr(module.attn, "config"):
                setattr(module.attn.config, key, value)
        after = {
            "block_len": int(getattr(module.attn.config, "block_len", before["block_len"])),
            "local_num_blocks": int(getattr(module.attn.config, "local_num_blocks", before["local_num_blocks"])),
            "if_remote_enabled": bool(getattr(module.attn.config, "if_remote_enabled", before["if_remote_enabled"])),
        }
        actual[f"layer_{touched - 1}"] = {"before": before, "after": after}
    if touched == 0:
        return {}
    return {"override": override, "num_flash_layers": touched, "layers": actual}


def bucket_for_distance(distance: int) -> str:
    for name, lo, hi in BUCKETS:
        if lo is not None and distance < lo:
            continue
        if hi is not None and distance > hi:
            continue
        return name
    raise RuntimeError(f"distance {distance} did not match any bucket")


def binomial_stats(correct: int, n: int) -> tuple[float, float, float, float]:
    if n <= 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    p = correct / n
    stderr = math.sqrt(max(p * (1.0 - p), 0.0) / n)
    return p, stderr, max(0.0, p - 1.96 * stderr), min(1.0, p + 1.96 * stderr)


def is_oom_error(exc: BaseException) -> bool:
    text = "".join(traceback.format_exception_only(type(exc), exc)) + "\n" + traceback.format_exc()
    return bool(re.search(r"out of memory|CUBLAS_STATUS_ALLOC_FAILED|CUDA error", text, re.I))


def evaluate_batch_size(
    *,
    model: torch.nn.Module,
    dataset: MQARWithMetadata,
    batch_size: int,
    device: str,
) -> tuple[dict[str, dict[str, int]], int, int]:
    counts = {name: {"n": 0, "correct": 0} for name, _, _ in BUCKETS}
    total = 0
    correct_total = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(dataset.inputs), batch_size):
            end = min(start + batch_size, len(dataset.inputs))
            inputs = dataset.inputs[start:end].to(device)
            labels = dataset.labels[start:end]
            query_pos = dataset.query_pos[start:end]
            distances = dataset.distance_value[start:end]
            logits = model(inputs)
            preds = logits.argmax(dim=-1).detach().cpu()
            pred_at_query = preds.gather(1, query_pos)
            target_at_query = labels.gather(1, query_pos)
            correct = pred_at_query.eq(target_at_query)
            for distance, ok in zip(distances.reshape(-1).tolist(), correct.reshape(-1).tolist()):
                bucket = bucket_for_distance(int(distance))
                counts[bucket]["n"] += 1
                counts[bucket]["correct"] += int(bool(ok))
                total += 1
                correct_total += int(bool(ok))
            del logits, preds, inputs
    return counts, correct_total, total


def evaluate_with_fallback(
    *,
    model: torch.nn.Module,
    dataset: MQARWithMetadata,
    batch_candidates: list[int],
    device: str,
) -> tuple[dict[str, dict[str, int]], int, int, int, list[dict[str, Any]], float]:
    failures: list[dict[str, Any]] = []
    last_exc: BaseException | None = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    for batch_size in batch_candidates:
        try:
            counts, correct, total = evaluate_batch_size(model=model, dataset=dataset, batch_size=batch_size, device=device)
            peak = float(torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0
            return counts, correct, total, batch_size, failures, peak
        except RuntimeError as exc:
            last_exc = exc
            if not is_oom_error(exc):
                raise
            failures.append({"batch_size": batch_size, "failure_type": "oom", "failure_detail": str(exc).splitlines()[0] if str(exc) else "oom"})
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    raise RuntimeError(f"all eval batch candidates failed: {failures}") from last_exc


def nvidia_query() -> dict[str, str]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,compute_cap,uuid,driver_version",
        "--format=csv,noheader,nounits",
    ]
    out = run_cmd_text(cmd)
    parts = [part.strip() for part in (out.splitlines()[0] if out else "").split(",")]
    if len(parts) < 6:
        return {}
    return {
        "gpu": parts[0],
        "gpu_name": parts[1],
        "gpu_total_memory_mb": parts[2],
        "gpu_compute_capability": parts[3],
        "gpu_uuid": parts[4],
        "driver_version": parts[5],
    }


def load_sources(path: Path, limit: int = 0) -> list[dict[str, str]]:
    rows = [row for row in read_csv(path) if row.get("status") == "completed"]
    if limit:
        rows = rows[:limit]
    if not rows:
        raise RuntimeError(f"No completed source rows found in {path}")
    return rows


def build_source_variants(source: dict[str, str], requested_variants: list[str]) -> list[tuple[str, dict[str, Any]]]:
    variants: list[tuple[str, dict[str, Any]]] = []
    for variant in requested_variants:
        if variant not in VARIANT_OVERRIDES:
            raise ValueError(f"unknown variant `{variant}`, expected one of {sorted(VARIANT_OVERRIDES)}")
        if variant != "full" and not is_flash_source(source):
            continue
        variants.append((variant, dict(VARIANT_OVERRIDES[variant])))
    return variants


def eval_buckets(args: argparse.Namespace) -> int:
    os.environ.setdefault("GDN_KERNEL_DTYPE", "float32")
    from zoology.checkpoints import load_checkpoint

    sources_path = Path(args.sources)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = load_sources(sources_path, limit=int(args.limit))
    slices = parse_slices(args.slices)
    variants = parse_csv_list(args.variants)
    batch_candidates = sorted({int(x) for x in parse_csv_list(args.batch_candidates)}, reverse=True)
    if 1 not in batch_candidates:
        raise ValueError("batch candidates must include 1")
    refs = load_official_accuracy_refs(Path(args.official_detail))
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    if device != "cuda" and not args.cpu:
        raise RuntimeError("CUDA is not available. Pass --cpu only for non-formal smoke.")
    set_eval_seed(int(args.eval_seed))
    gpu_meta = nvidia_query()
    bucket_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    run_status: list[dict[str, Any]] = []
    t0 = time.time()
    for source in sources:
        for variant, override in build_source_variants(source, variants):
            checkpoint_path = source["checkpoint_path"]
            checkpoint_which = source.get("checkpoint_which") or "last"
            source_id = source["source_run_id"]
            model_label = source["model_label"]
            variant_label = f"{model_label}:{variant}"
            load_t0 = time.time()
            bundle = load_checkpoint(checkpoint_path, which=checkpoint_which, device=device, strict=True)
            actual_checkpoint_path = str(bundle.get("checkpoint_path", checkpoint_path))
            checkpoint_sha256 = source.get("source_ckpt_sha256", "")
            if not checkpoint_sha256 and Path(actual_checkpoint_path).exists():
                checkpoint_sha256 = sha256_file(Path(actual_checkpoint_path))
            config_dump = apply_flash_override(bundle["model"], override)
            load_sec = time.time() - load_t0
            for seq_len, num_kv_pairs in slices:
                dataset = build_eval_dataset(
                    bundle["config"],
                    seq_len=seq_len,
                    num_kv_pairs=num_kv_pairs,
                    num_examples=int(args.num_examples),
                    eval_seed=int(args.eval_seed),
                    dataset_mode=str(args.dataset_mode),
                    near_pairs_per_bucket=int(args.near_pairs_per_bucket),
                )
                dataset_hash = tensor_sha256(dataset.inputs, dataset.labels)
                slice_t0 = time.time()
                fallback_failures: list[dict[str, Any]] = []
                try:
                    counts, correct, total, actual_batch, fallback_failures, peak_memory = evaluate_with_fallback(
                        model=bundle["model"],
                        dataset=dataset,
                        batch_candidates=batch_candidates,
                        device=device,
                    )
                    status = "completed"
                    failure_type = ""
                    failure_detail = ""
                except BaseException as exc:
                    counts = {name: {"n": 0, "correct": 0} for name, _, _ in BUCKETS}
                    correct = 0
                    total = 0
                    actual_batch = ""
                    peak_memory = float(torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0
                    status = "oom" if is_oom_error(exc) else "failed"
                    failure_type = "oom" if status == "oom" else type(exc).__name__
                    failure_detail = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
                elapsed = time.time() - slice_t0
                accuracy, stderr, ci_low, ci_high = binomial_stats(correct, total)
                ref = refs.get((source_id, seq_len, num_kv_pairs), {})
                official_ref = ref.get("official_accuracy_ref", "")
                abs_diff: float | str = ""
                sanity_status = "not_applicable"
                if variant == "full" and official_ref:
                    abs_diff = abs(float(accuracy) - float(official_ref))
                    sanity_status = "passed" if abs_diff <= SANITY_TOL else "invalid"
                elif variant == "full":
                    sanity_status = "no_ref"
                run_status_value = status if sanity_status != "invalid" else "invalid"
                summary_row = {
                    "model_label": model_label,
                    "variant": variant,
                    "variant_label": variant_label,
                    "source_run_id": source_id,
                    "source_model_family": source.get("source_model_family", ""),
                    "source_config_family": source.get("source_config_family", ""),
                    "seed": source.get("seed", ""),
                    "slice_seq_len": seq_len,
                    "slice_num_kv_pairs": num_kv_pairs,
                    "eval_seed": args.eval_seed,
                    "dataset_mode": args.dataset_mode,
                    "dataset_hash": dataset_hash,
                    "accuracy": f"{accuracy:.10f}" if total else "",
                    "stderr": f"{stderr:.10f}" if total else "",
                    "ci95_low": f"{ci_low:.10f}" if total else "",
                    "ci95_high": f"{ci_high:.10f}" if total else "",
                    "n": total,
                    "correct": correct,
                    "official_accuracy_ref": official_ref,
                    "official_dataset_hash_ref": ref.get("official_dataset_hash_ref", ""),
                    "abs_diff_from_ref": f"{abs_diff:.10f}" if isinstance(abs_diff, float) else "",
                    "sanity_status": sanity_status,
                    "run_status": run_status_value,
                    "eval_batch_size": actual_batch,
                    "batch_candidates": ";".join(str(x) for x in batch_candidates),
                    "fallback_count": len(fallback_failures),
                    "fallback_failures": json.dumps(fallback_failures, ensure_ascii=True, sort_keys=True),
                    "peak_memory_mb": f"{peak_memory:.3f}",
                    "wall_clock_sec": f"{elapsed:.3f}",
                    "checkpoint_path": actual_checkpoint_path,
                    "checkpoint_which": checkpoint_which,
                    "checkpoint_sha256": checkpoint_sha256,
                    "checkpoint_file_path": source.get("checkpoint_file_path", actual_checkpoint_path),
                    "stage3_variant": source.get("stage3_variant", ""),
                    "config_override": json.dumps(config_dump, ensure_ascii=True, sort_keys=True),
                    "failure_type": failure_type,
                    "failure_detail": failure_detail,
                }
                summary_rows.append(summary_row)
                for bucket_name, _, _ in BUCKETS:
                    bucket_n = counts[bucket_name]["n"]
                    bucket_correct = counts[bucket_name]["correct"]
                    b_acc, b_stderr, b_low, b_high = binomial_stats(bucket_correct, bucket_n)
                    bucket_rows.append({
                        **{k: summary_row[k] for k in [
                            "model_label",
                            "variant",
                            "variant_label",
                            "source_run_id",
                            "source_model_family",
                            "source_config_family",
                            "seed",
                            "slice_seq_len",
                            "slice_num_kv_pairs",
                            "eval_seed",
                            "dataset_mode",
                            "dataset_hash",
                        ]},
                        "distance_def": "query_pos-value_pos",
                        "distance_bucket": bucket_name,
                        "n": bucket_n,
                        "correct": bucket_correct,
                        "accuracy": f"{b_acc:.10f}" if bucket_n else "",
                        "stderr": f"{b_stderr:.10f}" if bucket_n else "",
                        "ci95_low": f"{b_low:.10f}" if bucket_n else "",
                        "ci95_high": f"{b_high:.10f}" if bucket_n else "",
                        "run_status": run_status_value,
                        "eval_batch_size": actual_batch,
                        "config_override": summary_row["config_override"],
                    })
                run_status.append({
                    "variant_label": variant_label,
                    "slice": f"{seq_len}x{num_kv_pairs}",
                    "status": run_status_value,
                    "accuracy": summary_row["accuracy"],
                    "sanity_status": sanity_status,
                    "load_sec": f"{load_sec:.3f}",
                    "eval_sec": f"{elapsed:.3f}",
                })
            del bundle
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    write_csv(output_dir / "slice_summary.csv", summary_rows)
    write_csv(output_dir / "distance_bucket.csv", bucket_rows)
    write_csv(output_dir / "eval_runs.csv", run_status)
    metadata = {
        "created_at_utc": now_utc(),
        "command": " ".join(sys.argv),
        "source": "local_window_bucket_eval",
        "sources": rel(sources_path),
        "slices": [f"{seq}x{kv}" for seq, kv in slices],
        "variants": variants,
        "eval_seed": int(args.eval_seed),
        "num_examples": int(args.num_examples),
        "dataset_mode": args.dataset_mode,
        "near_pairs_per_bucket": int(args.near_pairs_per_bucket),
        "batch_candidates": batch_candidates,
        "sanity_tolerance": SANITY_TOL,
        "duration_sec": time.time() - t0,
        "device": device,
        "gpu": gpu_meta,
        "git_commit": git_commit(ROOT),
        "git_dirty": git_dirty(ROOT),
        "flash_vqg_commit": git_commit(FLASH_VQG_ROOT),
        "flash_vqg_dirty": git_dirty(FLASH_VQG_ROOT),
        "status": "completed",
    }
    write_json(output_dir / "metadata.json", metadata)
    append_status(output_dir / "status.md", f"eval_buckets wrote slice_summary.csv and distance_bucket.csv rows={len(bucket_rows)}")
    return 0


def write_initial_readme(args: argparse.Namespace) -> int:
    path = Path(args.output_dir) / "README.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = """# Local window fairness longer-MQAR diagnostics

This artifact is for 20260529-flash-local-window-fairness.

Rows produced by eval-time overrides are diagnostic, not formal training results.
Formal training checkpoints and ledgers must stay separate from this diagnostic artifact.

Required first-round slices:

- 1024x256
- 2048x512
- 4096x512
- 4096x1024

Distance is computed from sample-level MQAR metadata as query_pos - value_pos.
Token-value reverse lookup is not allowed.
"""
    path.write_text(text, encoding="utf-8")
    append_status(Path(args.output_dir) / "status.md", "initialized README")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Flash local window fairness bucket eval utilities.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_prepare = sub.add_parser("prepare-sources", help="Build source_checkpoints.csv from official core manifest.")
    p_prepare.add_argument("--official-manifest", default=str(OFFICIAL_MANIFEST))
    p_prepare.add_argument("--official-detail", default=str(OFFICIAL_DETAIL))
    p_prepare.add_argument("--groups", default=",".join(DEFAULT_GROUPS))
    p_prepare.add_argument("--seeds", default=",".join(DEFAULT_SEEDS))
    p_prepare.add_argument("--include-all-official", action="store_true")
    p_prepare.add_argument("--allow-missing", action="store_true")
    p_prepare.add_argument("--output", default=str(ARTIFACT_DIR / "source_checkpoints.csv"))
    p_prepare.set_defaults(func=prepare_sources)

    p_prepare_stage3 = sub.add_parser("prepare-stage3-sources", help="Build source_checkpoints.csv from stage3 training summary.")
    p_prepare_stage3.add_argument("--stage3-summary", default=str(STAGE3_TRAIN_SUMMARY))
    p_prepare_stage3.add_argument("--checkpoint-kinds", default="last,best")
    p_prepare_stage3.add_argument("--allow-missing", action="store_true")
    p_prepare_stage3.add_argument("--output", default=str(EVAL_ONLY_ARTIFACT_DIR / "stage3-longer-mqar-bucket/source_checkpoints.csv"))
    p_prepare_stage3.set_defaults(func=prepare_stage3_sources)

    p_eval = sub.add_parser("eval-buckets", help="Evaluate source checkpoints by MQAR distance bucket.")
    p_eval.add_argument("--sources", default=str(ARTIFACT_DIR / "source_checkpoints.csv"))
    p_eval.add_argument("--official-detail", default=str(OFFICIAL_DETAIL))
    p_eval.add_argument("--output-dir", default=str(ARTIFACT_DIR))
    p_eval.add_argument("--slices", default=",".join(f"{seq}x{kv}" for seq, kv in DEFAULT_SLICES))
    p_eval.add_argument("--variants", default="full")
    p_eval.add_argument("--eval-seed", default=str(EVAL_SEED))
    p_eval.add_argument("--num-examples", type=int, default=DEFAULT_NUM_EXAMPLES)
    p_eval.add_argument("--dataset-mode", choices=["official", "near_enriched"], default="official")
    p_eval.add_argument("--near-pairs-per-bucket", type=int, default=16)
    p_eval.add_argument("--batch-candidates", default="8,4,2,1")
    p_eval.add_argument("--limit", type=int, default=0)
    p_eval.add_argument("--cpu", action="store_true", help="CPU smoke only; not formal.")
    p_eval.set_defaults(func=eval_buckets)

    p_readme = sub.add_parser("init-readme", help="Write diagnostic artifact README.")
    p_readme.add_argument("--output-dir", default=str(ARTIFACT_DIR))
    p_readme.set_defaults(func=write_initial_readme)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
