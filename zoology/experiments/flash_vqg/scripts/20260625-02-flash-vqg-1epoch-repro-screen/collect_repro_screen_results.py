#!/usr/bin/env python3
"""Collect lightweight summaries for the 1-epoch repro screen experiment."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ID = "20260625-02-flash-vqg-1epoch-repro-screen"
REPO_ROOT = Path(os.environ.get("ZOOLOGY_REPO_ROOT", "/home/lyj/mnt/project/zoology")).resolve()
FLASH_VQG_ROOT = Path(os.environ.get("FLASH_VQG_ROOT", "/home/lyj/mnt/project/Flash-VQG")).resolve()
ARTIFACT_DIR = REPO_ROOT / "docs/artifacts/20260625-02-flash-vqg-1epoch-repro-screen"
SOURCE_HOST_BY_MACHINE = {
    "2080ti": "mclab-2080ti",
    "3090": "mclab-3090",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    if fieldnames is None:
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    keys.append(key)
                    seen.add(key)
        fieldnames = keys
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _git_value(repo: Path, args: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _mean(values: list[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def _p05(values: list[float]) -> float | None:
    if not values:
        return None
    values = sorted(values)
    idx = max(0, min(len(values) - 1, int(round(0.05 * (len(values) - 1)))))
    return float(values[idx])


def _path_parts_from_trace_root(path: Path, outputs_dir: Path) -> tuple[str, str]:
    try:
        rel = path.relative_to(outputs_dir / "traces")
        return rel.parts[0], rel.parts[1]
    except Exception:
        return "", ""


def _machine_from_path(path: Path) -> str:
    parts = path.parts
    for machine in ("2080ti", "3090"):
        if machine in parts:
            return machine
        if any(part.startswith(f"{machine}-") for part in parts):
            return machine
        if any(f"rscreen-{machine}-" in part for part in parts):
            return machine
    return ""


def _source_and_mirror_paths(path: Path) -> tuple[str, str, str, str]:
    machine = _machine_from_path(path)
    source_host = SOURCE_HOST_BY_MACHINE.get(machine, "")
    mirror_path = str(path)
    if machine == "3090":
        return machine, source_host, f"{source_host}:{path}", mirror_path
    if machine == "2080ti":
        return machine, source_host, str(path), mirror_path
    return machine, source_host, str(path), mirror_path


def _parse_final_metrics(log_path: Path) -> dict[str, Any]:
    if not log_path.exists():
        return {}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    accuracy_1024 = re.findall(r"valid/mqar_case/accuracy-1024x256=([0-9.]+)", text)
    valid_accuracy = re.findall(r"valid/accuracy=([0-9.]+)", text)
    valid_loss = re.findall(r"valid/loss=([0-9.]+)", text)
    cache_paths = sorted(set(re.findall(r"Loading data from on-disk cache at (.+?\\.pt)\\.\\.\\.", text)))
    return {
        "final_1024x256_accuracy": accuracy_1024[-1] if accuracy_1024 else "",
        "final_valid_accuracy": valid_accuracy[-1] if valid_accuracy else "",
        "final_valid_loss": valid_loss[-1] if valid_loss else "",
        "n_validation_summaries": len(accuracy_1024),
        "cache_paths": cache_paths,
    }


def _target_seed(target: str) -> str:
    match = re.search(r"s(123|124)", target)
    return match.group(1) if match else ""


def _target_repeat(target: str) -> str:
    match = re.search(r"-(r[12])$", target)
    return match.group(1) if match else ""


def _collect_queue_status(outputs_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    invalid: list[dict[str, Any]] = []
    status_rank = {"pending": 0, "started": 1, "completed": 2}
    for status_path in sorted(outputs_dir.glob("*/queue-status.tsv")):
        with status_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                queue = str(row.get("queue", ""))
                target = str(row.get("target", ""))
                key = (queue, target)
                row = dict(row)
                row["status_path"] = str(status_path)
                prev = rows_by_key.get(key)
                current_status = str(row.get("status", ""))
                current_rank = status_rank.get(current_status, 99 if current_status.startswith("failed") else -1)
                prev_rank = -1
                if prev is not None:
                    prev_status = str(prev.get("status", ""))
                    prev_rank = status_rank.get(prev_status, 99 if prev_status.startswith("failed") else -1)
                if prev is None or current_rank >= prev_rank:
                    rows_by_key[key] = row
    rows = sorted(rows_by_key.values(), key=lambda row: (str(row.get("queue", "")), str(row.get("target", ""))))
    for row in rows:
        status = str(row.get("status", ""))
        if status.startswith("failed") or status in {"interrupted", "oom"}:
            invalid.append(row)
    return rows, invalid


def _collect_preflight(outputs_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(outputs_dir.glob("*/preflight-*.json")):
        payload = _read_json(path)
        rows.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "path": str(path),
                "machine_name": payload.get("env", {}).get("machine_name"),
                "mode": path.stem.replace("preflight-", ""),
                "passed": payload.get("passed"),
                "runtime_ready": payload.get("env", {}).get("runtime_ready"),
                "train_batches": payload.get("contract", {}).get("train_batches"),
                "gradient_accumulation_steps": payload.get("contract", {}).get("gradient_accumulation_steps"),
                "num_optimizer_steps": payload.get("contract", {}).get("num_optimizer_steps"),
                "git_commit": payload.get("env", {}).get("git_commit"),
                "torch_version": payload.get("env", {}).get("torch_version"),
                "torch_cuda": payload.get("env", {}).get("torch_cuda"),
                "cuda_available": payload.get("env", {}).get("cuda_available"),
                "device_count": payload.get("env", {}).get("device_count"),
                "nvidia_smi_available": payload.get("env", {}).get("nvidia_smi", {}).get("available"),
                "driver_version": (
                    payload.get("env", {}).get("nvidia_smi", {}).get("gpus", [{}])[0].get("driver_version")
                    if payload.get("env", {}).get("nvidia_smi", {}).get("available")
                    else ""
                ),
            }
        )
    return rows


def _collect_early_metrics(outputs_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted((outputs_dir / "traces").glob("*/*/early_window_metrics.jsonl")):
        machine, target = _path_parts_from_trace_root(metrics_path.parent, outputs_dir)
        for row in _read_jsonl(metrics_path):
            row = dict(row)
            row.setdefault("experiment_id", EXPERIMENT_ID)
            row.setdefault("machine", machine)
            row.setdefault("target", target)
            row.setdefault("source_path", str(metrics_path))
            rows.append(row)
    return rows


def _collect_read_trace_summary(outputs_dir: Path) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int, int, str], dict[str, Any]] = {}
    for trace_path in sorted((outputs_dir / "traces").glob("*/*/train_step_*/read_trace.jsonl*")):
        machine, target = _path_parts_from_trace_root(trace_path.parents[1], outputs_dir)
        rows = _read_jsonl(trace_path)
        if not rows:
            continue
        for row in rows:
            train_step = int(row.get("global_step", -1))
            valid_batch_idx = int(row.get("valid_batch_idx", -1))
            run_id = str(row.get("run_id") or "")
            key = (machine, target, run_id, train_step, valid_batch_idx, str(trace_path))
            bucket = grouped.setdefault(
                key,
                {
                    "experiment_id": EXPERIMENT_ID,
                    "machine": machine,
                    "target": target,
                    "run_id": run_id,
                    "train_step": train_step,
                    "valid_batch_idx": valid_batch_idx,
                    "source_path": str(trace_path),
                    "records": 0,
                    "entropy": [],
                    "selected_mass": [],
                    "margin_top1_top2": [],
                    "top1_ids": set(),
                },
            )
            bucket["records"] += 1
            for metric in ("entropy", "selected_mass", "margin_top1_top2"):
                value = row.get(metric)
                if value is not None:
                    bucket[metric].append(float(value))
            candidates = row.get("topk_candidate_ids") or []
            if candidates:
                bucket["top1_ids"].add(int(candidates[0]))

    summaries: list[dict[str, Any]] = []
    for bucket in grouped.values():
        summaries.append(
            {
                "experiment_id": bucket["experiment_id"],
                "machine": bucket["machine"],
                "target": bucket["target"],
                "run_id": bucket["run_id"],
                "train_step": bucket["train_step"],
                "valid_batch_idx": bucket["valid_batch_idx"],
                "records": bucket["records"],
                "entropy_mean": _mean(bucket["entropy"]),
                "selected_mass_mean": _mean(bucket["selected_mass"]),
                "selected_mass_p05": _p05(bucket["selected_mass"]),
                "margin_top1_top2_mean": _mean(bucket["margin_top1_top2"]),
                "margin_top1_top2_p05": _p05(bucket["margin_top1_top2"]),
                "unique_top1_ids": len(bucket["top1_ids"]),
                "source_path": bucket["source_path"],
            }
        )
    return summaries


def _collect_source_manifest(outputs_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidates = []
    candidates.extend(sorted(outputs_dir.glob("*/preflight-*.json")))
    candidates.extend(sorted(outputs_dir.glob("*/queue-status.tsv")))
    candidates.extend(sorted(outputs_dir.glob("*/queue.log")))
    candidates.extend(sorted(outputs_dir.glob("*/logs/*.log")))
    candidates.extend(sorted((outputs_dir / "traces").glob("*/*/early_window_metrics.jsonl")))
    candidates.extend(sorted((outputs_dir / "traces").glob("*/*/train_step_*/read_trace.jsonl*")))
    generated_dir = REPO_ROOT / "zoology/experiments/flash_vqg/generated"
    candidates.extend(sorted(generated_dir.glob("fvqg-20260625-02-rscreen-*/manifest.json")))
    candidates.extend(sorted(generated_dir.glob("fvqg-20260625-02-rscreen-*/launch_configs.py")))
    for path in candidates:
        if not path.exists() or not path.is_file():
            continue
        machine, source_host, source_path, mirror_path = _source_and_mirror_paths(path)
        rows.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "source_machine": machine,
                "source_host": source_host,
                "source_path": source_path,
                "mirror_path": mirror_path,
                "sha256": _sha256(path),
                "file_size": path.stat().st_size,
                "status": "mirrored-to-2080ti" if machine == "3090" else "source-local",
            }
        )
    return rows


def _build_cache_hash_summary(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in run_rows:
        machine = str(row.get("machine", ""))
        for cache_path in row.get("cache_paths", []):
            key = (machine, cache_path)
            if key in seen:
                continue
            seen.add(key)
            cache_file = REPO_ROOT / cache_path.replace("./", "", 1)
            rows.append(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "machine": machine,
                    "cache_path": cache_path,
                    "exists_local": cache_file.exists(),
                    "sha256_local": _sha256(cache_file) if cache_file.exists() else "",
                    "file_size": cache_file.stat().st_size if cache_file.exists() else "",
                }
            )
    return rows


def _write_metadata(path: Path, outputs_dir: Path, summary: dict[str, Any]) -> None:
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "outputs_dir": str(outputs_dir),
        "zoology_branch": _git_value(REPO_ROOT, ["rev-parse", "--abbrev-ref", "HEAD"]),
        "zoology_commit": _git_value(REPO_ROOT, ["rev-parse", "--short", "HEAD"]),
        "flash_vqg_branch": _git_value(FLASH_VQG_ROOT, ["rev-parse", "--abbrev-ref", "HEAD"]),
        "flash_vqg_commit": _git_value(FLASH_VQG_ROOT, ["rev-parse", "--short", "HEAD"]),
        "dtype_policy": (
            "default torch/zoology runtime dtype; no explicit AMP, bf16, or fp16 override in launch configs; "
            "GD residual builder grouped_chunk_torch_ref with semivec_ref pack mode"
        ),
        "official_ledger": "not recorded; diagnostic/exploratory run",
        **summary,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", type=Path, default=SCRIPT_DIR / "outputs")
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    args = parser.parse_args()

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    preflight_rows = _collect_preflight(args.outputs_dir)
    queue_rows, invalid_rows = _collect_queue_status(args.outputs_dir)
    early_rows = _collect_early_metrics(args.outputs_dir)
    read_trace_rows = _collect_read_trace_summary(args.outputs_dir)
    source_rows = _collect_source_manifest(args.outputs_dir)

    run_rows = []
    for row in queue_rows:
        if str(row.get("status", "")) in {"pending", "started"}:
            continue
        parsed_metrics = _parse_final_metrics(Path(str(row.get("log", ""))))
        target = str(row.get("target", ""))
        run_rows.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "queue": row.get("queue"),
                "machine": str(row.get("queue", "")).replace("-gpu0", "").replace("-smoke", ""),
                "stage": "smoke" if target.startswith("smoke-") else "screen",
                "target": target,
                "seed": _target_seed(target),
                "repeat": _target_repeat(target),
                "gpu": row.get("gpu"),
                "pid": row.get("pid"),
                "status": row.get("status"),
                "log": row.get("log"),
                "trace_output_dir": row.get("trace_output_dir"),
                "started_at": row.get("started_at"),
                "finished_at": row.get("finished_at"),
                **parsed_metrics,
            }
        )

    cache_rows = _build_cache_hash_summary(run_rows)

    _write_csv(args.artifact_dir / "preflight-summary.csv", preflight_rows)
    _write_csv(args.artifact_dir / "machine-summary.csv", queue_rows)
    _write_csv(args.artifact_dir / "invalid-runs.csv", invalid_rows)
    _write_csv(args.artifact_dir / "run-summary.csv", run_rows)
    _write_csv(args.artifact_dir / "early-window-metrics.csv", early_rows)
    _write_csv(args.artifact_dir / "read-trace-summary.csv", read_trace_rows)
    _write_csv(args.artifact_dir / "cache-hash-summary.csv", cache_rows)
    _write_csv(args.artifact_dir / "source-manifest.csv", source_rows)

    step_rows = []
    for row in early_rows:
        step_rows.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "machine": row.get("machine"),
                "target": row.get("target"),
                "run_id": row.get("run_id"),
                "train_step": row.get("train_step"),
                "loss": row.get("loss"),
                "update_norm_p95": row.get("early_window/attn/gd_residual_update_norm_p95"),
                "update_norm_max": row.get("early_window/attn/gd_residual_update_norm_max"),
                "uncapped_sum_zeta_p95": row.get("early_window/attn/gd_residual_uncapped_sum_zeta_p95"),
                "cap_hit_ratio": row.get("early_window/attn/gd_residual_write_strength_cap_hit_ratio"),
                "m_norm_max": row.get("early_window/attn/gd_residual_m_norm_max"),
                "lambda_mean": row.get("early_window/attn/gd_residual_lambda_mean"),
                "inject_ratio": row.get("early_window/attn/gd_residual_inject_ratio"),
            }
        )
    _write_csv(args.artifact_dir / "step-window-summary.csv", step_rows)

    repeat_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        if row.get("stage") != "screen":
            continue
        repeat_groups[(str(row.get("machine", "")), str(row.get("seed", "")))].append(row)

    repeat_rows = []
    for (machine, seed), rows in sorted(repeat_groups.items()):
        accs = [float(row["final_1024x256_accuracy"]) for row in rows if row.get("final_1024x256_accuracy")]
        repeats = {str(row.get("repeat", "")): row.get("final_1024x256_accuracy", "") for row in rows}
        gap = ""
        stable = ""
        if len(accs) >= 2:
            gap_value = max(accs) - min(accs)
            gap = f"{gap_value:.6f}"
            stable = "true" if gap_value <= 0.02 else "false"
        repeat_rows.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "machine": machine,
                "seed": seed,
                "num_runs": len(rows),
                "r1_1024x256_accuracy": repeats.get("r1", ""),
                "r2_1024x256_accuracy": repeats.get("r2", ""),
                "mean_1024x256_accuracy": f"{statistics.fmean(accs):.6f}" if accs else "",
                "repeat_gap": gap,
                "stable_le_0p02": stable,
            }
        )
    _write_csv(args.artifact_dir / "repeat-summary.csv", repeat_rows)

    _write_metadata(
        args.artifact_dir / "metadata.json",
        args.outputs_dir,
        {
            "preflight_rows": len(preflight_rows),
            "queue_rows": len(queue_rows),
            "invalid_rows": len(invalid_rows),
            "run_rows": len(run_rows),
            "repeat_rows": len(repeat_rows),
            "early_window_metric_rows": len(early_rows),
            "read_trace_summary_rows": len(read_trace_rows),
            "cache_hash_rows": len(cache_rows),
            "source_manifest_rows": len(source_rows),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
