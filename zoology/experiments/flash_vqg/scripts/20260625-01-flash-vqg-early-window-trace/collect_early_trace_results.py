#!/usr/bin/env python3
"""Collect lightweight summaries for the early-window trace experiment."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import subprocess
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = Path("/home/lyj/mnt/project/zoology/docs/artifacts/20260625-01-flash-vqg-early-window-trace")
EXPERIMENT_ID = "20260625-01-flash-vqg-early-window-trace"
REPO_ROOT = Path("/home/lyj/mnt/project/zoology")
FLASH_VQG_ROOT = Path("/home/lyj/mnt/project/Flash-VQG")
SOURCE_HOST_BY_MACHINE = {
    "2080ti": "mclab-2080ti",
    "3090": "mclab-3090",
}


def _is_smoke_target(target: str) -> bool:
    return target.startswith("smoke-")


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
        if any(f"etrace-{machine}-" in part for part in parts):
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
    return {
        "final_1024x256_accuracy": accuracy_1024[-1] if accuracy_1024 else "",
        "final_valid_accuracy": valid_accuracy[-1] if valid_accuracy else "",
        "final_valid_loss": valid_loss[-1] if valid_loss else "",
        "n_validation_summaries": len(accuracy_1024),
    }


def _collect_queue_status(outputs_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for status_path in sorted(outputs_dir.glob("*/queue-status.tsv")):
        with status_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                row["status_path"] = str(status_path)
                rows.append(row)
                status = str(row.get("status", ""))
                if status.startswith("failed") or status in {"interrupted", "oom"}:
                    invalid.append(row)
    return rows, invalid


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


def _collect_source_manifest(outputs_dir: Path, artifact_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidates = []
    candidates.extend(sorted(outputs_dir.glob("*/queue-status.tsv")))
    candidates.extend(sorted(outputs_dir.glob("*/queue.log")))
    candidates.extend(sorted(outputs_dir.glob("*/logs/*.log")))
    candidates.extend(sorted((outputs_dir / "traces").glob("*/*/early_window_metrics.jsonl")))
    candidates.extend(sorted((outputs_dir / "traces").glob("*/*/train_step_*/read_trace.jsonl*")))
    generated_dir = REPO_ROOT / "zoology/experiments/flash_vqg/generated"
    candidates.extend(sorted(generated_dir.glob("fvqg-20260625-01-etrace-*/manifest.json")))
    candidates.extend(sorted(generated_dir.glob("fvqg-20260625-01-etrace-*/launch_configs.py")))
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
    queue_rows, invalid_rows = _collect_queue_status(args.outputs_dir)
    early_rows = _collect_early_metrics(args.outputs_dir)
    read_trace_rows = _collect_read_trace_summary(args.outputs_dir)
    source_rows = _collect_source_manifest(args.outputs_dir, args.artifact_dir)

    _write_csv(args.artifact_dir / "machine-summary.csv", queue_rows)
    _write_csv(args.artifact_dir / "invalid-runs.csv", invalid_rows)
    _write_csv(args.artifact_dir / "early-window-metrics.csv", early_rows)
    _write_csv(args.artifact_dir / "read-trace-summary.csv", read_trace_rows)
    _write_csv(args.artifact_dir / "source-manifest.csv", source_rows)

    run_rows = []
    for row in queue_rows:
        if row.get("status") == "started":
            continue
        parsed_metrics = _parse_final_metrics(Path(str(row.get("log", ""))))
        target = str(row.get("target", ""))
        queue = str(row.get("queue", ""))
        run_rows.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "queue": row.get("queue"),
                "stage": "smoke" if _is_smoke_target(target) or "smoke" in queue else "wave1",
                "target": row.get("target"),
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
    _write_csv(args.artifact_dir / "run-summary.csv", run_rows)
    _write_csv(args.artifact_dir / "stage3-run-summary.csv", [r for r in run_rows if r.get("stage") == "wave1"])

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
    _write_csv(args.artifact_dir / "early-window-step-summary.csv", step_rows)
    stage3_step_rows = [r for r in step_rows if not _is_smoke_target(str(r.get("target", "")))]
    _write_csv(args.artifact_dir / "stage3-step-window-summary.csv", stage3_step_rows)

    stage3_read_rows = [r for r in read_trace_rows if not _is_smoke_target(str(r.get("target", "")))]
    _write_csv(args.artifact_dir / "stage3-read-trace-summary.csv", stage3_read_rows)

    read_by_key = {
        (row.get("machine"), row.get("target"), row.get("run_id"), str(row.get("train_step"))): row
        for row in stage3_read_rows
    }
    final_by_key = {
        (row.get("queue", "").replace("-wave1", ""), row.get("target")): row
        for row in run_rows
        if row.get("stage") == "wave1"
    }
    key_rows = []
    for row in stage3_step_rows:
        read_row = read_by_key.get(
            (row.get("machine"), row.get("target"), row.get("run_id"), str(row.get("train_step"))),
            {},
        )
        final_row = final_by_key.get((row.get("machine"), row.get("target")), {})
        key_rows.append(
            {
                **row,
                "read_entropy_mean": read_row.get("entropy_mean", ""),
                "read_selected_mass_mean": read_row.get("selected_mass_mean", ""),
                "read_selected_mass_p05": read_row.get("selected_mass_p05", ""),
                "read_margin_top1_top2_mean": read_row.get("margin_top1_top2_mean", ""),
                "read_margin_top1_top2_p05": read_row.get("margin_top1_top2_p05", ""),
                "read_unique_top1_ids": read_row.get("unique_top1_ids", ""),
                "final_1024x256_accuracy": final_row.get("final_1024x256_accuracy", ""),
                "final_valid_accuracy": final_row.get("final_valid_accuracy", ""),
                "final_valid_loss": final_row.get("final_valid_loss", ""),
            }
        )
    _write_csv(args.artifact_dir / "stage3-key-metrics.csv", key_rows)

    _write_metadata(
        args.artifact_dir / "metadata.json",
        args.outputs_dir,
        {
            "queue_rows": len(queue_rows),
            "invalid_rows": len(invalid_rows),
            "early_window_metric_rows": len(early_rows),
            "read_trace_summary_rows": len(read_trace_rows),
            "source_manifest_rows": len(source_rows),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
