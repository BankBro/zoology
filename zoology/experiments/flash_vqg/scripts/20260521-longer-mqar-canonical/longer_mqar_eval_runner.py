#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

def find_repo_root(start: Path) -> Path:
    current = start if start.is_dir() else start.parent
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists() and (candidate / "zoology").is_dir():
            return candidate
    raise RuntimeError(f"无法从 {start} 定位仓库根目录.")


ROOT = find_repo_root(Path(__file__).resolve())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PRELIM_LEDGER_PATH = ROOT / "docs/artifacts/longer-mqar/longer-mqar-eval-summary.csv"
FLASH_LEDGER = ROOT / "docs/artifacts/gd-residual-v1/rank-seed-effect-summary.csv"
GDN_LEDGER = ROOT / "docs/artifacts/gdn/gdn-hparam-effect-summary.csv"
ARTIFACT_DIR = ROOT / "docs/artifacts/longer-mqar/official-core-20260526"
LEDGER_PATH = ARTIFACT_DIR / "longer-mqar-official-core-detail.csv"
SUMMARY_PATH = ARTIFACT_DIR / "longer-mqar-official-core-summary.csv"
STATUS_CSV_PATH = ARTIFACT_DIR / "status.csv"
TMP_ROOT = ROOT / "tmp/20260526-longer-mqar-official-core"
BATCH_ID = "20260526-longer-mqar-official-core"
EVAL_SCOPE = "longer_mqar_eval_only_vocab8192_official_core_20260526"
EVAL_SEED = 123
FORMAL_SLICES = [(1024, 256), (2048, 512), (4096, 1024), (8190, 512), (8190, 2047)]
BATCH_SEARCH_SLICES = list(FORMAL_SLICES)
SANITY_SLICE = (1024, 256)
LEDGER_LOCK = threading.Lock()

CORE_FLASH_TARGETS = {
    ("cb256-r10", "123"),
    ("cb256-r10", "124"),
    ("cb256-r10", "125"),
    ("cb256-r10", "126"),
    ("cb256-r4", "123"),
    ("cb64-r16", "123"),
}
CORE_GDN_TARGETS = {
    ("gdn-h2-ev8-usegate0", "123"),
    ("gdn-h2-ev8-usegate0", "124"),
    ("gdn-h2-ev8-usegate0", "125"),
    ("gdn-h2-ev8-usegate0", "126"),
    ("gdn-h2-ev8-usegate0", "127"),
    ("gdn-h2-ev10-usegate0", "123"),
    ("gdn-h2-ev10-usegate0", "124"),
    ("gdn-h2-ev10-usegate0", "125"),
    ("gdn-h2-ev10-usegate0", "126"),
    ("gdn-h2-ev10-usegate0", "127"),
    ("gdn-h2-ev16-usegate0", "123"),
    ("gdn-h2-ev16-usegate0", "124"),
    ("gdn-h2-ev16-usegate0", "125"),
}
EXPECTED_CORE_SOURCE_COUNT = len(CORE_FLASH_TARGETS) + len(CORE_GDN_TARGETS)
OFFICIAL_CORE_BATCH_PROFILE = {
    "source_train_batch_size": "64",
    "source_eval_batch_size": "16",
    "source_gradient_accumulation_steps": "4",
    "source_effective_train_batch_size": "256",
    "source_batch_accum_profile": "b64_ga4",
}
EXTRA_DETAIL_FIELDS = [
    "eval_seed",
    "dataset_hash",
    "dataset_hash_algorithm",
    "dataset_input_shape",
    "dataset_label_shape",
    "dataset_num_examples",
    "dataset_dtype",
    "checkpoint_hash",
    "checkpoint_hash_algorithm",
    "official_core_constraint_status",
    "selected_core_subset",
    "repro_check_reference_event_id",
    "repro_check_dataset_hash_match",
    "repro_check_accuracy_delta_abs",
    "repro_check_accuracy_match",
]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def set_eval_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def tensor_sha256(*tensors: Any) -> str:
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


def desired_detail_fields() -> list[str]:
    with PRELIM_LEDGER_PATH.open(newline="", encoding="utf-8") as f:
        fields = list(csv.DictReader(f).fieldnames or [])
    out = list(fields)
    for field in EXTRA_DETAIL_FIELDS:
        if field not in out:
            out.append(field)
    return out


def detail_fields() -> list[str]:
    desired = desired_detail_fields()
    if LEDGER_PATH.exists() and LEDGER_PATH.stat().st_size > 0:
        with LEDGER_PATH.open(newline="", encoding="utf-8") as f:
            fields = list(csv.DictReader(f).fieldnames or [])
        if fields:
            out = list(fields)
            for field in desired:
                if field not in out:
                    out.append(field)
            return out
    return desired


def ensure_detail_ledger() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    fields = detail_fields()
    if LEDGER_PATH.exists() and LEDGER_PATH.stat().st_size > 0:
        with LEDGER_PATH.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            old_fields = list(reader.fieldnames or [])
            rows = list(reader)
        if old_fields and all(field in old_fields for field in fields):
            return
        with LEDGER_PATH.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])
        return
    with LEDGER_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()


def ledger_fields() -> list[str]:
    return detail_fields()


def load_existing_rows() -> dict[str, dict[str, str]]:
    if not LEDGER_PATH.exists():
        return {}
    rows: dict[str, dict[str, str]] = {}
    with LEDGER_PATH.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            event = row.get("eval_event_id", "")
            if event and row.get("eval_status") == "completed":
                rows[event] = row
    return rows


def append_ledger_row(row: dict[str, Any]) -> None:
    fields = ledger_fields()
    clean = {field: row.get(field, "") for field in fields}
    with LEDGER_LOCK:
        ensure_detail_ledger()
        with LEDGER_PATH.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writerow(clean)


def run_cmd_text(cmd: list[str], cwd: Path = ROOT) -> str:
    try:
        return subprocess.check_output(cmd, cwd=str(cwd), text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def git_commit(path: Path) -> str:
    return run_cmd_text(["git", "rev-parse", "HEAD"], cwd=path)


def nvidia_query() -> dict[int, dict[str, str]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,compute_cap,uuid,driver_version",
        "--format=csv,noheader,nounits",
    ]
    out = run_cmd_text(cmd)
    gpus: dict[int, dict[str, str]] = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        idx = int(parts[0])
        gpus[idx] = {
            "gpu": str(idx),
            "cuda_device": str(idx),
            "gpu_name": parts[1],
            "gpu_total_memory_mb": parts[2],
            "gpu_compute_capability": parts[3],
            "gpu_uuid": parts[4],
            "driver_version": parts[5],
        }
    return gpus


def torch_info() -> dict[str, str]:
    code = "import torch, json; print(json.dumps({'torch_version': torch.__version__, 'cuda_version': torch.version.cuda or ''}))"
    out = run_cmd_text([sys.executable, "-c", code])
    try:
        return json.loads(out)
    except Exception:
        return {"torch_version": "", "cuda_version": ""}


def file_mtime_utc(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def source_scope(row: dict[str, str]) -> str:
    return row.get("official_scope") or row.get("comparison_scope") or row.get("dtype_comparison_scope") or ""


def source_model_family(kind: str, row: dict[str, str]) -> str:
    if kind == "flash":
        return "flash"
    return row.get("model_family") or "gdn"


def source_role(kind: str, row: dict[str, str]) -> str:
    if kind == "flash":
        return row.get("config_family") or row.get("config") or "flash"
    return row.get("baseline_role") or row.get("config_family") or row.get("config") or "gdn"


def infer_gdn_total_capacity(row: dict[str, str], train_config_path: Path) -> tuple[str, str]:
    per_layer = row.get("dynamic_state_capacity") or row.get("dynamic_capacity_per_layer") or ""
    total = row.get("dynamic_capacity_total") or ""
    if per_layer:
        # Capacity comparisons use the active GDN memory layer only.
        # Current Hybrid configs are BaseConv + GDN, so multiplying by model.n_layers
        # would incorrectly count the BaseConv layer as another GDN layer.
        return per_layer, per_layer
    return per_layer, total


def is_b64_ga4_fp32_official(row: dict[str, str]) -> bool:
    if row.get("train_batch_size") != OFFICIAL_CORE_BATCH_PROFILE["source_train_batch_size"]:
        return False
    if row.get("eval_batch_size") != OFFICIAL_CORE_BATCH_PROFILE["source_eval_batch_size"]:
        return False
    if row.get("gradient_accumulation_steps") != OFFICIAL_CORE_BATCH_PROFILE["source_gradient_accumulation_steps"]:
        return False
    if row.get("effective_train_batch_size") != OFFICIAL_CORE_BATCH_PROFILE["source_effective_train_batch_size"]:
        return False
    if row.get("batch_accum_profile") != OFFICIAL_CORE_BATCH_PROFILE["source_batch_accum_profile"]:
        return False
    if row.get("dtype_policy") != "float32":
        return False
    if row.get("outer_model_dtype") not in {"", "float32"}:
        return False
    return row.get("official_scope") == "b64_ga4_fp32_official"


def load_ckpt_epoch(path: Path) -> str:
    try:
        import torch

        payload = torch.load(path, map_location="cpu")
        return str(payload.get("epoch", ""))
    except Exception:
        return ""


def build_sources() -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for kind, ledger in [("flash", FLASH_LEDGER), ("gdn", GDN_LEDGER)]:
        for row in read_csv(ledger):
            if (row.get("status") or "completed").lower() != "completed":
                continue
            target_key = (row.get("config_family", ""), row.get("seed", ""))
            if kind == "flash":
                if target_key not in CORE_FLASH_TARGETS:
                    continue
            elif target_key not in CORE_GDN_TARGETS:
                continue
            if not is_b64_ga4_fp32_official(row):
                continue
            tc = row.get("train_config_path") or row.get("source_train_config_path") or ""
            if not tc:
                continue
            train_config_path = (ROOT / tc).resolve()
            ckpt_path = Path(row.get("last_checkpoint_path") or "")
            if str(ckpt_path):
                ckpt_path = (ROOT / ckpt_path).resolve() if not ckpt_path.is_absolute() else ckpt_path
            else:
                ckpt_path = train_config_path.parent / "last.pt"
            best_path = Path(row.get("best_checkpoint_path") or "")
            if str(best_path):
                best_path = (ROOT / best_path).resolve() if not best_path.is_absolute() else best_path
            else:
                best_path = train_config_path.parent / "best.pt"
            if not ckpt_path.exists():
                continue
            source_run_id = row.get("run_id") or row.get("config") or ckpt_path.parent.name
            if kind == "flash":
                dyn_per_layer = ""
                dyn_total = row.get("dynamic_capacity") or row.get("dynamic_capacity_total") or ""
            else:
                dyn_per_layer, dyn_total = infer_gdn_total_capacity(row, train_config_path)
            src = {
                "source_ledger_path": rel(ledger),
                "source_model_family": source_model_family(kind, row),
                "source_run_id": source_run_id,
                "source_role": source_role(kind, row),
                "source_config_id": row.get("config") or row.get("config_family") or source_run_id,
                "source_scope": source_scope(row),
                "source_comparison_scope": row.get("comparison_scope") or source_scope(row),
                "source_run_type": row.get("run_type", ""),
                "source_train_batch_size": row.get("train_batch_size", ""),
                "source_eval_batch_size": row.get("eval_batch_size", ""),
                "source_gradient_accumulation_steps": row.get("gradient_accumulation_steps", ""),
                "source_effective_train_batch_size": row.get("effective_train_batch_size", ""),
                "source_batch_accum_profile": row.get("batch_accum_profile", ""),
                "source_dtype_policy": row.get("dtype_policy", ""),
                "source_outer_model_dtype": row.get("outer_model_dtype", ""),
                "source_hidden_states_dtype": row.get("hidden_states_dtype", ""),
                "source_kernel_input_dtype": row.get("kernel_input_dtype", ""),
                "source_actual_kernel_dtype": row.get("actual_kernel_dtype", ""),
                "source_gdn_kernel_dtype": row.get("gdn_kernel_dtype_policy", "") or row.get("actual_kernel_dtype", ""),
                "source_seed": row.get("seed", ""),
                "source_data_seed": row.get("data_seed", ""),
                "source_config_family": row.get("config_family", ""),
                "source_config": row.get("config", ""),
                "source_num_codebook_vectors": row.get("num_codebook_vectors", ""),
                "source_rank": row.get("rank", ""),
                "source_num_heads": row.get("num_heads", ""),
                "source_expand_v": row.get("expand_v", ""),
                "source_use_gate": row.get("use_gate", ""),
                "source_use_short_conv": row.get("use_short_conv", ""),
                "source_conv_size": row.get("conv_size", ""),
                "source_configured_max_epochs": row.get("configured_max_epochs", ""),
                "source_final_epoch": row.get("final_epoch", ""),
                "source_validations_per_epoch": row.get("validations_per_epoch", ""),
                "source_early_stopping_disabled": row.get("early_stopping_disabled", ""),
                "source_train_config_path": rel(train_config_path),
                "source_train_config_path_abs": str(train_config_path),
                "source_train_config_sha256": sha256_file(train_config_path) if train_config_path.exists() else "",
                "source_ckpt_identity": f"sha256:{sha256_file(ckpt_path)}",
                "source_ckpt_type": "last.pt_epoch4_final",
                "source_ckpt_path": rel(ckpt_path),
                "source_ckpt_path_abs": str(ckpt_path),
                "source_ckpt_exists": str(ckpt_path.exists()).lower(),
                "source_ckpt_sha256": sha256_file(ckpt_path),
                "source_ckpt_size_bytes": str(ckpt_path.stat().st_size),
                "source_ckpt_mtime_utc": file_mtime_utc(ckpt_path),
                "source_ckpt_epoch": load_ckpt_epoch(ckpt_path),
                "source_ckpt_step": "",
                "source_final_valid_loss": row.get("valid_loss", ""),
                "source_final_valid_accuracy": row.get("valid_accuracy", ""),
                "source_final_valid_mqar_case_accuracy_1024x256": row.get("valid_mqar_case_accuracy_1024x256", ""),
                "source_trainable_params": row.get("trainable_params", ""),
                "source_dynamic_capacity_per_layer": dyn_per_layer,
                "source_dynamic_capacity_total": dyn_total,
                "source_artifact": row.get("source_artifact", "") or rel(ledger),
                "checkpoint_hash": sha256_file(ckpt_path),
                "checkpoint_hash_algorithm": "sha256",
                "official_core_constraint_status": "passed",
                "selected_core_subset": "true",
            }
            sources.append(src)
    # Stable order: Flash first, then GDN, within each ledger order.
    seen = set()
    duplicates = []
    for src in sources:
        key = (src["source_model_family"], src["source_config_family"], src["source_seed"])
        if key in seen:
            duplicates.append(key)
        seen.add(key)
    if duplicates:
        raise RuntimeError(f"official core source 出现重复项: {duplicates}")
    if len(sources) != EXPECTED_CORE_SOURCE_COUNT:
        observed = sorted((s["source_model_family"], s["source_config_family"], s["source_seed"], s["source_run_id"]) for s in sources)
        raise RuntimeError(f"official core source 数量应为 {EXPECTED_CORE_SOURCE_COUNT}, 实际 {len(sources)}: {observed}")
    return sources


def write_source_manifest(sources: list[dict[str, Any]]) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = ARTIFACT_DIR / "manifest.json"
    csv_path = ARTIFACT_DIR / "manifest.csv"
    json_path.write_text(json.dumps(sources, ensure_ascii=False, indent=2), encoding="utf-8")
    keys = sorted({k for s in sources for k in s})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(sources)


def protocol_id(input_seq_len: int, num_kv_pairs: int, num_examples: int) -> str:
    return f"mqar-vocab8192-pass1-seed123-rnq1-power0.01-{input_seq_len}x{num_kv_pairs}-n{num_examples}"


def event_id(mode: str, source: dict[str, Any], input_seq_len: int, num_kv_pairs: int, num_examples: int) -> str:
    return f"{BATCH_ID}:{mode}:{source['source_run_id']}:{input_seq_len}x{num_kv_pairs}:n{num_examples}"


def event_id_for_run_id(mode: str, source_run_id: str, input_seq_len: int, num_kv_pairs: int, num_examples: int) -> str:
    return f"{BATCH_ID}:{mode}:{source_run_id}:{input_seq_len}x{num_kv_pairs}:n{num_examples}"


def base_row(
    *,
    mode: str,
    status: str,
    source: dict[str, Any],
    input_seq_len: int,
    num_kv_pairs: int,
    num_examples: int,
    eval_batch_size: int | str,
    gpu_info: dict[str, str],
    torch_meta: dict[str, str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {k: "" for k in ledger_fields()}
    row.update(source)
    row.update({
        "schema_version": "2",
        "eval_event_id": event_id(mode, source, input_seq_len, num_kv_pairs, num_examples),
        "eval_run_id": event_id(mode, source, input_seq_len, num_kv_pairs, num_examples),
        "eval_batch_id": BATCH_ID,
        "run_type": "longer_mqar_eval_only",
        "eval_scope": EVAL_SCOPE,
        "eval_mode": mode,
        "eval_status": status,
        "created_at_utc": now_utc(),
        "eval_protocol_id": protocol_id(input_seq_len, num_kv_pairs, num_examples),
        "eval_dataset": "multiquery_ar",
        "vocab_size": "8192",
        "input_seq_len": str(input_seq_len),
        "num_kv_pairs": str(num_kv_pairs),
        "num_passes": "1",
        "num_examples": str(num_examples),
        "eval_batch_size": str(eval_batch_size),
        "adaptive_batch_search": "true" if mode in {"batch_search", "formal"} else "false",
        "eval_data_seed": str(EVAL_SEED),
        "eval_seed": str(EVAL_SEED),
        "random_non_queries": "true",
        "power_a": "0.01",
        "artifact_dir": rel(ARTIFACT_DIR),
        "commit_zoology": git_commit(ROOT),
        "commit_flash_vqg": git_commit(Path("/home/lyj/mnt/project/Flash-VQG")),
        "runner_commit": git_commit(ROOT),
        "eval_hardware_backfill_status": "measured_current_host",
        **gpu_info,
        **torch_meta,
    })
    row["checkpoint_hash"] = source.get("checkpoint_hash") or source.get("source_ckpt_sha256", "")
    row["checkpoint_hash_algorithm"] = source.get("checkpoint_hash_algorithm") or "sha256"
    hp = hardware_profile_id(row)
    row["eval_hardware_profile_id"] = hp
    if extra:
        row.update(extra)
    return row


def hardware_profile_id(row: dict[str, Any]) -> str:
    name = str(row.get("gpu_name", "unknown")).replace(" ", "_")
    mem = row.get("gpu_total_memory_mb", "")
    cc = row.get("gpu_compute_capability", "")
    drv = row.get("driver_version", "")
    torch_v = row.get("torch_version", "")
    cuda_v = row.get("cuda_version", "")
    return f"{name}-{mem}MiB-cc{cc}-driver{drv}-torch{torch_v}-cuda{cuda_v}"


def batch_search_extra(
    *,
    source: dict[str, Any],
    input_seq_len: int,
    num_kv_pairs: int,
    num_examples: int,
    candidates: list[int],
    best: int | None,
    peak: Any,
    status: str,
    failure_detail: str,
    gpu_info: dict[str, str],
    torch_meta: dict[str, str],
) -> dict[str, Any]:
    kernel_dtype = "float32" if source.get("source_model_family") == "gdn" else (source.get("source_actual_kernel_dtype") or "float32")
    temp_row = {**gpu_info, **torch_meta}
    temp_row["eval_hardware_profile_id"] = hardware_profile_id(temp_row)
    return {
        "batch_search_status": status,
        "batch_search_slice": f"{input_seq_len}x{num_kv_pairs}",
        "batch_search_input_seq_len": str(input_seq_len),
        "batch_search_num_kv_pairs": str(num_kv_pairs),
        "batch_search_num_examples": str(num_examples),
        "batch_search_candidates": ";".join(str(c) for c in candidates),
        "batch_search_best_eval_batch_size": "" if best is None else str(best),
        "batch_search_peak_memory_mb": peak,
        "batch_search_failure_detail": failure_detail,
        "batch_search_reusable_scope": "same_gpu_same_dtype_same_runner_only",
        "batch_search_hardware_dependent": "true",
        "batch_search_gpu": gpu_info.get("gpu", ""),
        "batch_search_cuda_device": gpu_info.get("cuda_device", ""),
        "batch_search_gpu_name": gpu_info.get("gpu_name", ""),
        "batch_search_gpu_total_memory_mb": gpu_info.get("gpu_total_memory_mb", ""),
        "batch_search_gpu_compute_capability": gpu_info.get("gpu_compute_capability", ""),
        "batch_search_gpu_uuid": gpu_info.get("gpu_uuid", ""),
        "batch_search_torch_version": torch_meta.get("torch_version", ""),
        "batch_search_cuda_version": torch_meta.get("cuda_version", ""),
        "batch_search_driver_version": gpu_info.get("driver_version", ""),
        "batch_search_dtype_policy": "float32",
        "batch_search_kernel_dtype": kernel_dtype,
        "batch_search_source_scope": source.get("source_scope", ""),
        "batch_search_hardware_profile_id": temp_row["eval_hardware_profile_id"],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_child_event(event: dict[str, Any], gpu: int, log_path: Path, result_path: Path) -> dict[str, Any]:
    event_path = TMP_ROOT / "events" / f"{event['event_uid']}.json"
    write_json(event_path, event)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["LONGER_MQAR_ORIGINAL_GPU"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("GDN_KERNEL_DTYPE", "float32")
    cmd = [sys.executable, str(Path(__file__).resolve()), "--single-event", str(event_path), "--result", str(result_path)]
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    if result_path.exists():
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception as exc:
            result = {"status": "failed", "failure_type": "bad_result_json", "failure_detail": str(exc)}
    else:
        result = {"status": "failed", "failure_type": "no_result_json", "failure_detail": f"child_returncode={proc.returncode}"}
    result["returncode"] = proc.returncode
    result["log_path"] = rel(log_path)
    return result


def child_single_event(event_path: Path, result_path: Path) -> int:
    event = json.loads(event_path.read_text(encoding="utf-8"))
    result: dict[str, Any] = {}
    t0 = time.time()
    try:
        os.environ.setdefault("GDN_KERNEL_DTYPE", "float32")
        import torch
        import torch.nn as nn

        from zoology.checkpoints import load_checkpoint
        from zoology.data.multiquery_ar import MQARConfig
        from zoology.experiments.flash_vqg.eval_only import _prepare_test_dataloader_from_data_config
        from zoology.train import Trainer

        class NullLogger:
            def log(self, metrics, step=None):
                return None

        set_eval_seed(int(event.get("eval_seed", EVAL_SEED)))
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        device = "cuda"
        bundle = load_checkpoint(event["checkpoint_path"], which="last", device=device, strict=True)
        config = bundle["config"].model_copy(deep=True)
        template = None
        for candidate in list(config.data.test_configs) + list(config.data.train_configs):
            if isinstance(candidate, MQARConfig):
                template = candidate
                break
        if template is None:
            raise TypeError("checkpoint config 中没有 MQARConfig")
        payload = template.model_dump()
        payload.update({
            "vocab_size": 8192,
            "num_examples": int(event["num_examples"]),
            "input_seq_len": int(event["input_seq_len"]),
            "num_kv_pairs": int(event["num_kv_pairs"]),
            "random_non_queries": True,
            "power_a": 0.01,
            "include_slices": True,
        })
        config.data = config.data.model_copy(deep=True)
        config.data.seed = int(event.get("eval_seed", EVAL_SEED))
        config.data.cache_dir = None
        config.data.force_cache = False
        config.data.test_configs = [MQARConfig(**payload)]
        if isinstance(config.data.batch_size, int):
            config.data.batch_size = (config.data.batch_size, int(event["eval_batch_size"]))
        else:
            config.data.batch_size = (config.data.batch_size[0], int(event["eval_batch_size"]))
        config.slice_keys = ["mqar_case", "input_seq_len", "num_kv_pairs"]
        set_eval_seed(int(event.get("eval_seed", EVAL_SEED)))
        test_dataloader = _prepare_test_dataloader_from_data_config(config.data)
        dataset_hash = ""
        dataset_input_shape = ""
        dataset_label_shape = ""
        dataset_num_examples = ""
        dataset_dtype = ""
        try:
            segment = test_dataloader.dataset.segments[0]
            dataset_hash = tensor_sha256(segment.inputs, segment.labels)
            dataset_input_shape = "x".join(str(x) for x in segment.inputs.shape)
            dataset_label_shape = "x".join(str(x) for x in segment.labels.shape)
            dataset_num_examples = str(len(segment))
            dataset_dtype = f"{segment.inputs.dtype}/{segment.labels.dtype}"
        except Exception as exc:
            raise RuntimeError(f"dataset_hash 计算失败: {exc}") from exc
        task = Trainer(
            model=bundle["model"],
            train_dataloader=test_dataloader,
            test_dataloader=test_dataloader,
            input_type=config.input_type,
            max_epochs=1,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            early_stopping_metric=None,
            early_stopping_threshold=None,
            loss_type=config.loss_type,
            slice_keys=config.slice_keys,
            device=device,
            logger=NullLogger(),
            checkpoint_manager=None,
        )
        task.loss_fn = nn.CrossEntropyLoss()
        metrics = task.test(epoch_idx=0)
        elapsed = time.time() - t0
        peak_alloc = torch.cuda.max_memory_allocated() / (1024**2)
        peak_reserved = torch.cuda.max_memory_reserved() / (1024**2)
        case = f"{event['input_seq_len']}x{event['num_kv_pairs']}"
        result = {
            "status": "completed",
            "metrics": {k: float(v) for k, v in metrics.items()},
            "loss": float(metrics.get("valid/loss", float("nan"))),
            "accuracy": float(metrics.get(f"valid/mqar_case/accuracy-{case}", metrics.get("valid/accuracy", float("nan")))),
            "aggregate_loss": float(metrics.get("valid/loss", float("nan"))),
            "aggregate_accuracy": float(metrics.get("valid/accuracy", float("nan"))),
            "wall_clock_sec": elapsed,
            "wall_clock": time.strftime("%H:%M:%S", time.gmtime(elapsed)),
            "peak_memory_mb": peak_alloc,
            "torch_peak_memory_allocated_mib": peak_alloc,
            "torch_peak_memory_reserved_mib": peak_reserved,
            "dataset_hash": dataset_hash,
            "dataset_hash_algorithm": "sha256(inputs,labels)",
            "dataset_input_shape": dataset_input_shape,
            "dataset_label_shape": dataset_label_shape,
            "dataset_num_examples": dataset_num_examples,
            "dataset_dtype": dataset_dtype,
            "eval_seed": str(event.get("eval_seed", EVAL_SEED)),
        }
    except BaseException as exc:
        elapsed = time.time() - t0
        detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        tb = traceback.format_exc()
        oom = bool(re.search(r"out of memory|CUDA error: out of memory|CUBLAS_STATUS_ALLOC_FAILED", detail + "\n" + tb, re.I))
        result = {
            "status": "oom" if oom else "failed",
            "failure_type": "oom" if oom else type(exc).__name__,
            "failure_detail": detail,
            "traceback_tail": "\n".join(tb.splitlines()[-20:]),
            "wall_clock_sec": elapsed,
            "wall_clock": time.strftime("%H:%M:%S", time.gmtime(elapsed)),
        }
        try:
            import torch
            if torch.cuda.is_available():
                result["peak_memory_mb"] = torch.cuda.max_memory_allocated() / (1024**2)
                result["torch_peak_memory_allocated_mib"] = torch.cuda.max_memory_allocated() / (1024**2)
                result["torch_peak_memory_reserved_mib"] = torch.cuda.max_memory_reserved() / (1024**2)
        except Exception:
            pass
    write_json(result_path, result)
    return 0


def result_to_row_extra(result: dict[str, Any], log_path: str) -> dict[str, Any]:
    return {
        "loss": result.get("loss", ""),
        "accuracy": result.get("accuracy", ""),
        "aggregate_loss": result.get("aggregate_loss", ""),
        "aggregate_accuracy": result.get("aggregate_accuracy", ""),
        "wall_clock_sec": result.get("wall_clock_sec", ""),
        "wall_clock": result.get("wall_clock", ""),
        "peak_memory_mb": result.get("peak_memory_mb", ""),
        "torch_peak_memory_allocated_mib": result.get("torch_peak_memory_allocated_mib", ""),
        "torch_peak_memory_reserved_mib": result.get("torch_peak_memory_reserved_mib", ""),
        "log_path": log_path,
        "failure_type": result.get("failure_type", ""),
        "failure_detail": result.get("failure_detail", ""),
        "eval_seed": result.get("eval_seed", str(EVAL_SEED)),
        "dataset_hash": result.get("dataset_hash", ""),
        "dataset_hash_algorithm": result.get("dataset_hash_algorithm", ""),
        "dataset_input_shape": result.get("dataset_input_shape", ""),
        "dataset_label_shape": result.get("dataset_label_shape", ""),
        "dataset_num_examples": result.get("dataset_num_examples", ""),
        "dataset_dtype": result.get("dataset_dtype", ""),
    }


def run_eval_once(
    *,
    source: dict[str, Any],
    gpu: int,
    mode: str,
    input_seq_len: int,
    num_kv_pairs: int,
    num_examples: int,
    eval_batch_size: int,
) -> tuple[dict[str, Any], str]:
    uid = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{mode}-{source['source_run_id']}-{input_seq_len}x{num_kv_pairs}-n{num_examples}-b{eval_batch_size}-gpu{gpu}")
    event = {
        "event_uid": uid,
        "checkpoint_path": source["source_ckpt_path_abs"],
        "input_seq_len": int(input_seq_len),
        "num_kv_pairs": int(num_kv_pairs),
        "num_examples": int(num_examples),
        "eval_batch_size": int(eval_batch_size),
        "eval_seed": EVAL_SEED,
    }
    log_path = TMP_ROOT / "logs" / f"{uid}.log"
    result_path = TMP_ROOT / "results" / f"{uid}.json"
    result = run_child_event(event, gpu=gpu, log_path=log_path, result_path=result_path)
    return result, rel(log_path)


def append_eval_row(
    *,
    mode: str,
    source: dict[str, Any],
    input_seq_len: int,
    num_kv_pairs: int,
    num_examples: int,
    eval_batch_size: int | str,
    result: dict[str, Any],
    log_path: str,
    gpu_info: dict[str, str],
    torch_meta: dict[str, str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row_extra = result_to_row_extra(result, log_path)
    if extra:
        row_extra.update(extra)
    row = base_row(
        mode=mode,
        status=result.get("status", "failed"),
        source=source,
        input_seq_len=input_seq_len,
        num_kv_pairs=num_kv_pairs,
        num_examples=num_examples,
        eval_batch_size=eval_batch_size,
        gpu_info=gpu_info,
        torch_meta=torch_meta,
        extra=row_extra,
    )
    append_ledger_row(row)
    return row


def process_source(
    *,
    source: dict[str, Any],
    gpu: int,
    existing: set[str],
    existing_rows: dict[str, dict[str, str]],
    gpu_info: dict[str, str],
    torch_meta: dict[str, str],
    sanity_examples: int,
    batch_search_examples: int,
    formal_examples: int,
    candidates: list[int],
    phases: set[str],
) -> dict[str, Any]:
    summary = {"source_run_id": source["source_run_id"], "gpu": gpu, "status": "started", "events_appended": 0}
    search_best: dict[tuple[int, int], int] = {}
    search_rows: dict[tuple[int, int], dict[str, Any]] = {}
    observed_rows: dict[str, dict[str, Any]] = {}

    if "sanity" in phases:
        seq, kv = SANITY_SLICE
        eid = event_id("sanity", source, seq, kv, sanity_examples)
        if eid not in existing:
            result, log_path = run_eval_once(source=source, gpu=gpu, mode="sanity", input_seq_len=seq, num_kv_pairs=kv, num_examples=sanity_examples, eval_batch_size=1)
            extra: dict[str, Any] = {}
            ref = source.get("source_final_valid_mqar_case_accuracy_1024x256") or ""
            extra["sanity_reference_accuracy_1024x256"] = ref
            try:
                extra["sanity_delta_abs"] = abs(float(result.get("accuracy", "nan")) - float(ref))
            except Exception:
                extra["sanity_delta_abs"] = ""
            append_eval_row(mode="sanity", source=source, input_seq_len=seq, num_kv_pairs=kv, num_examples=sanity_examples, eval_batch_size=1, result=result, log_path=log_path, gpu_info=gpu_info, torch_meta=torch_meta, extra=extra)
            existing.add(eid)
            summary["events_appended"] += 1

    if "batch-search" in phases:
        for seq, kv in BATCH_SEARCH_SLICES:
            eid = event_id("batch_search", source, seq, kv, batch_search_examples)
            if eid in existing:
                previous = existing_rows.get(eid, {})
                try:
                    best_batch = int(previous.get("batch_search_best_eval_batch_size") or previous.get("eval_batch_size") or "")
                except Exception:
                    best_batch = None
                if best_batch is not None:
                    search_best[(seq, kv)] = best_batch
                    search_rows[(seq, kv)] = {
                        key: previous.get(key, "")
                        for key in ledger_fields()
                        if key.startswith("batch_search_")
                    }
                continue
            best_result = None
            best_log = ""
            best_batch = None
            failures: list[str] = []
            for batch in sorted(candidates, reverse=True):
                if batch > batch_search_examples:
                    continue
                result, log_path = run_eval_once(source=source, gpu=gpu, mode="batch-search-candidate", input_seq_len=seq, num_kv_pairs=kv, num_examples=batch_search_examples, eval_batch_size=batch)
                if result.get("status") == "completed":
                    best_result = result
                    best_log = log_path
                    best_batch = batch
                    break
                failures.append(f"b{batch}:{result.get('status')}:{result.get('failure_type','')}:{result.get('failure_detail','')}")
            if best_result is None:
                best_result = {"status": "oom" if any("oom" in x.lower() for x in failures) else "failed", "failure_type": "batch_search_no_candidate", "failure_detail": " | ".join(failures)}
            b_extra = batch_search_extra(
                source=source,
                input_seq_len=seq,
                num_kv_pairs=kv,
                num_examples=batch_search_examples,
                candidates=[c for c in sorted(candidates, reverse=True) if c <= batch_search_examples],
                best=best_batch,
                peak=best_result.get("peak_memory_mb", ""),
                status="completed" if best_batch is not None else best_result.get("status", "failed"),
                failure_detail=" | ".join(failures),
                gpu_info=gpu_info,
                torch_meta=torch_meta,
            )
            append_eval_row(mode="batch_search", source=source, input_seq_len=seq, num_kv_pairs=kv, num_examples=batch_search_examples, eval_batch_size=best_batch or "", result=best_result, log_path=best_log, gpu_info=gpu_info, torch_meta=torch_meta, extra=b_extra)
            existing.add(eid)
            summary["events_appended"] += 1
            if best_batch is not None:
                search_best[(seq, kv)] = best_batch
                search_rows[(seq, kv)] = b_extra

    if "formal" in phases:
        for seq, kv in FORMAL_SLICES:
            eid = event_id("formal", source, seq, kv, formal_examples)
            if eid in existing:
                continue
            lookup = (seq, kv)
            if lookup == (4096, 512):
                lookup = (4096, 1024)
            elif lookup == (8190, 512):
                lookup = (8190, 2047)
            best_batch = search_best.get(lookup)
            b_extra = deepcopy(search_rows.get(lookup, {}))
            if best_batch is None:
                result = {"status": "skipped", "failure_type": "missing_batch_search", "failure_detail": f"missing batch-search result for {lookup[0]}x{lookup[1]}"}
                append_eval_row(mode="formal", source=source, input_seq_len=seq, num_kv_pairs=kv, num_examples=formal_examples, eval_batch_size="", result=result, log_path="", gpu_info=gpu_info, torch_meta=torch_meta, extra=b_extra)
                existing.add(eid)
                summary["events_appended"] += 1
                continue
            result, log_path = run_eval_once(source=source, gpu=gpu, mode="formal", input_seq_len=seq, num_kv_pairs=kv, num_examples=formal_examples, eval_batch_size=best_batch)
            b_extra["batch_search_slice"] = f"{lookup[0]}x{lookup[1]}"
            b_extra["batch_search_best_eval_batch_size"] = str(best_batch)
            row = append_eval_row(mode="formal", source=source, input_seq_len=seq, num_kv_pairs=kv, num_examples=formal_examples, eval_batch_size=best_batch, result=result, log_path=log_path, gpu_info=gpu_info, torch_meta=torch_meta, extra=b_extra)
            observed_rows[eid] = row
            existing.add(eid)
            summary["events_appended"] += 1

    if "repro" in phases:
        seq, kv = SANITY_SLICE
        repro_examples = formal_examples
        reference_eid = event_id("formal", source, seq, kv, repro_examples)
        eid = event_id("repro", source, seq, kv, repro_examples)
        if eid not in existing:
            best_batch = search_best.get((seq, kv))
            if best_batch is None:
                previous = existing_rows.get(reference_eid, {})
                try:
                    best_batch = int(previous.get("eval_batch_size") or "")
                except Exception:
                    best_batch = None
            if best_batch is None:
                result = {
                    "status": "skipped",
                    "failure_type": "missing_formal_reference",
                    "failure_detail": f"missing formal reference for {seq}x{kv}",
                }
                extra = {
                    "repro_check_reference_event_id": reference_eid,
                    "repro_check_dataset_hash_match": "",
                    "repro_check_accuracy_delta_abs": "",
                    "repro_check_accuracy_match": "",
                }
                append_eval_row(mode="repro", source=source, input_seq_len=seq, num_kv_pairs=kv, num_examples=repro_examples, eval_batch_size="", result=result, log_path="", gpu_info=gpu_info, torch_meta=torch_meta, extra=extra)
            else:
                result, log_path = run_eval_once(source=source, gpu=gpu, mode="repro", input_seq_len=seq, num_kv_pairs=kv, num_examples=repro_examples, eval_batch_size=best_batch)
                reference = observed_rows.get(reference_eid) or existing_rows.get(reference_eid, {})
                ref_hash = reference.get("dataset_hash", "")
                ref_acc = reference.get("accuracy", "")
                delta: str | float = ""
                acc_match = ""
                try:
                    delta_float = abs(float(result.get("accuracy", "nan")) - float(ref_acc))
                    delta = delta_float
                    acc_match = str(delta_float <= 1e-12).lower()
                except Exception:
                    pass
                extra = {
                    "repro_check_reference_event_id": reference_eid,
                    "repro_check_dataset_hash_match": str(bool(ref_hash) and result.get("dataset_hash") == ref_hash).lower(),
                    "repro_check_accuracy_delta_abs": delta,
                    "repro_check_accuracy_match": acc_match,
                }
                append_eval_row(mode="repro", source=source, input_seq_len=seq, num_kv_pairs=kv, num_examples=repro_examples, eval_batch_size=best_batch, result=result, log_path=log_path, gpu_info=gpu_info, torch_meta=torch_meta, extra=extra)
            existing.add(eid)
            summary["events_appended"] += 1
    summary["status"] = "completed"
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fieldnames} for row in rows])


def mean_std(values: list[float]) -> tuple[str, str]:
    if not values:
        return "", ""
    mean = sum(values) / len(values)
    if len(values) == 1:
        return f"{mean:.10f}", ""
    var = sum((value - mean) ** 2 for value in values) / len(values)
    return f"{mean:.10f}", f"{var ** 0.5:.10f}"


def source_group_label(row: dict[str, str]) -> str:
    family = row.get("source_config_family", "")
    if row.get("source_model_family") == "flash":
        return family
    return family.replace("-usegate0", "")


def generate_status_and_summary() -> dict[str, Any]:
    if not LEDGER_PATH.exists():
        return {"status": "missing_detail_csv"}
    rows = read_csv(LEDGER_PATH)
    formal_rows = [row for row in rows if row.get("eval_mode") == "formal"]
    completed_formal = [row for row in formal_rows if row.get("eval_status") == "completed"]
    repro_rows = [row for row in rows if row.get("eval_mode") == "repro"]
    status_rows: list[dict[str, Any]] = []
    for row in rows:
        if row.get("eval_mode") not in {"formal", "repro"}:
            continue
        status_rows.append({
            "eval_event_id": row.get("eval_event_id", ""),
            "eval_mode": row.get("eval_mode", ""),
            "eval_status": row.get("eval_status", ""),
            "source_model_family": row.get("source_model_family", ""),
            "source_config_family": row.get("source_config_family", ""),
            "source_config": row.get("source_config", ""),
            "source_seed": row.get("source_seed", ""),
            "input_seq_len": row.get("input_seq_len", ""),
            "num_kv_pairs": row.get("num_kv_pairs", ""),
            "dataset_hash": row.get("dataset_hash", ""),
            "accuracy": row.get("accuracy", ""),
            "gpu": row.get("gpu", ""),
            "cuda_device": row.get("cuda_device", ""),
            "gpu_name": row.get("gpu_name", ""),
            "failure_type": row.get("failure_type", ""),
            "failure_detail": row.get("failure_detail", ""),
            "repro_check_dataset_hash_match": row.get("repro_check_dataset_hash_match", ""),
            "repro_check_accuracy_delta_abs": row.get("repro_check_accuracy_delta_abs", ""),
            "repro_check_accuracy_match": row.get("repro_check_accuracy_match", ""),
        })
    write_csv(STATUS_CSV_PATH, status_rows)

    by_group_slice: dict[tuple[str, str], list[float]] = {}
    group_meta: dict[str, dict[str, Any]] = {}
    for row in completed_formal:
        group = source_group_label(row)
        slc = f"{row.get('input_seq_len')}x{row.get('num_kv_pairs')}"
        try:
            acc = float(row.get("accuracy", ""))
        except Exception:
            continue
        by_group_slice.setdefault((group, slc), []).append(acc)
        meta = group_meta.setdefault(group, {
            "config_group": group,
            "source_model_family": row.get("source_model_family", ""),
            "seeds": set(),
            "source_trainable_params": row.get("source_trainable_params", ""),
            "source_dynamic_capacity_total": row.get("source_dynamic_capacity_total", ""),
            "source_batch_accum_profile": row.get("source_batch_accum_profile", ""),
        })
        meta["seeds"].add(row.get("source_seed", ""))

    summary_rows: list[dict[str, Any]] = []
    for group in sorted(group_meta):
        meta = group_meta[group]
        out = {
            "config_group": group,
            "source_model_family": meta["source_model_family"],
            "n_checkpoints": len(meta["seeds"]),
            "seeds": ";".join(sorted(meta["seeds"])),
            "source_trainable_params": meta["source_trainable_params"],
            "source_dynamic_capacity_total": meta["source_dynamic_capacity_total"],
            "source_batch_accum_profile": meta["source_batch_accum_profile"],
        }
        for seq, kv in FORMAL_SLICES:
            slc = f"{seq}x{kv}"
            mean, std = mean_std(by_group_slice.get((group, slc), []))
            out[f"accuracy_mean_{slc}"] = mean
            out[f"accuracy_std_{slc}"] = std
        summary_rows.append(out)
    summary_fields = [
        "config_group",
        "source_model_family",
        "n_checkpoints",
        "seeds",
        "source_trainable_params",
        "source_dynamic_capacity_total",
        "source_batch_accum_profile",
    ]
    for seq, kv in FORMAL_SLICES:
        slc = f"{seq}x{kv}"
        summary_fields.extend([f"accuracy_mean_{slc}", f"accuracy_std_{slc}"])
    write_csv(SUMMARY_PATH, summary_rows, summary_fields)

    dataset_hashes_by_slice: dict[str, set[str]] = {}
    for row in completed_formal:
        slc = f"{row.get('input_seq_len')}x{row.get('num_kv_pairs')}"
        dataset_hashes_by_slice.setdefault(slc, set()).add(row.get("dataset_hash", ""))
    status = {
        "status": "completed" if len(completed_formal) == EXPECTED_CORE_SOURCE_COUNT * len(FORMAL_SLICES) else "incomplete",
        "detail_csv": rel(LEDGER_PATH),
        "summary_csv": rel(SUMMARY_PATH),
        "status_csv": rel(STATUS_CSV_PATH),
        "formal_completed": len(completed_formal),
        "formal_expected": EXPECTED_CORE_SOURCE_COUNT * len(FORMAL_SLICES),
        "formal_failed_or_incomplete": len(formal_rows) - len(completed_formal),
        "dataset_hash_unique_count_by_slice": {key: len(value) for key, value in sorted(dataset_hashes_by_slice.items())},
        "dataset_hash_consistent_by_slice": all(len(value) == 1 for value in dataset_hashes_by_slice.values()) and len(dataset_hashes_by_slice) == len(FORMAL_SLICES),
        "repro_rows": len(repro_rows),
        "repro_completed": sum(1 for row in repro_rows if row.get("eval_status") == "completed"),
        "repro_dataset_hash_match": all(row.get("repro_check_dataset_hash_match") == "true" for row in repro_rows if row.get("eval_status") == "completed"),
        "repro_accuracy_match": all(row.get("repro_check_accuracy_match") == "true" for row in repro_rows if row.get("eval_status") == "completed"),
        "updated_at_utc": now_utc(),
    }
    write_json(ARTIFACT_DIR / "verification.json", status)
    return status


def run_queue(args: argparse.Namespace) -> int:
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    (TMP_ROOT / "logs").mkdir(exist_ok=True)
    (TMP_ROOT / "events").mkdir(exist_ok=True)
    (TMP_ROOT / "results").mkdir(exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_detail_ledger()
    sources = build_sources()
    if args.offset:
        sources = sources[int(args.offset):]
    if args.limit:
        sources = sources[: int(args.limit)]
    write_source_manifest(sources)
    gpu_meta = nvidia_query()
    tmeta = torch_info()
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    phases = set(args.phases.split(","))
    if "all" in phases:
        phases = {"sanity", "batch-search", "formal", "repro"}
    candidates = [int(x) for x in args.batch_candidates.split(",") if x.strip()]
    existing_rows = load_existing_rows()
    existing = set(existing_rows)
    status_path = ARTIFACT_DIR / "status.json"
    run_manifest = {
        "batch_id": BATCH_ID,
        "created_at_utc": now_utc(),
        "source_count": len(sources),
        "expected_core_source_count": EXPECTED_CORE_SOURCE_COUNT,
        "formal_slices": [f"{seq}x{kv}" for seq, kv in FORMAL_SLICES],
        "expected_formal_rows": len(sources) * len(FORMAL_SLICES),
        "eval_seed": EVAL_SEED,
        "gpus": gpus,
        "phases": sorted(phases),
        "sanity_examples": args.sanity_examples,
        "batch_search_examples": args.batch_search_examples,
        "formal_examples": args.formal_examples,
        "batch_candidates": candidates,
        "tmp_root": rel(TMP_ROOT),
        "detail_csv": rel(LEDGER_PATH),
        "summary_csv": rel(SUMMARY_PATH),
        "status_csv": rel(STATUS_CSV_PATH),
        "preliminary_ledger_retained": rel(PRELIM_LEDGER_PATH),
    }
    write_json(ARTIFACT_DIR / "manifest-run.json", run_manifest)

    buckets = {gpu: [] for gpu in gpus}
    for idx, src in enumerate(sources):
        buckets[gpus[idx % len(gpus)]].append(src)

    summaries: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    def worker(gpu: int, assigned: list[dict[str, Any]]):
        out = []
        for src in assigned:
            try:
                out.append(process_source(
                    source=src,
                    gpu=gpu,
                    existing=existing,
                    existing_rows=existing_rows,
                    gpu_info=gpu_meta[gpu],
                    torch_meta=tmeta,
                    sanity_examples=args.sanity_examples,
                    batch_search_examples=args.batch_search_examples,
                    formal_examples=args.formal_examples,
                    candidates=candidates,
                    phases=phases,
                ))
                write_json(status_path, {**run_manifest, "updated_at_utc": now_utc(), "completed_sources": len(summaries) + len(out), "errors": errors})
            except BaseException as exc:
                err = {"source_run_id": src.get("source_run_id"), "gpu": gpu, "error": repr(exc), "traceback": traceback.format_exc()}
                errors.append(err)
                write_json(status_path, {**run_manifest, "updated_at_utc": now_utc(), "completed_sources": len(summaries) + len(out), "errors": errors})
        return out

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(worker, gpu, assigned) for gpu, assigned in buckets.items()]
        for fut in as_completed(futures):
            summaries.extend(fut.result())
            write_json(status_path, {**run_manifest, "updated_at_utc": now_utc(), "completed_sources": len(summaries), "errors": errors, "summaries": summaries})
    write_json(status_path, {**run_manifest, "updated_at_utc": now_utc(), "completed_sources": len(summaries), "errors": errors, "summaries": summaries, "status": "completed"})
    generate_status_and_summary()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-event", type=Path, default=None)
    parser.add_argument("--result", type=Path, default=None)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--phases", default="all", help="all or comma list: sanity,batch-search,formal,repro,summary")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sanity-examples", type=int, default=16)
    parser.add_argument("--batch-search-examples", type=int, default=8)
    parser.add_argument("--formal-examples", type=int, default=500)
    parser.add_argument("--batch-candidates", default="1,2,4,8")
    args = parser.parse_args()
    if args.single_event is not None:
        if args.result is None:
            raise SystemExit("--result is required with --single-event")
        return child_single_event(args.single_event, args.result)
    if args.phases.strip() == "summary":
        generate_status_and_summary()
        return 0
    return run_queue(args)


if __name__ == "__main__":
    raise SystemExit(main())
