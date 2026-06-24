#!/usr/bin/env python3
"""Collect lightweight summaries for the early-window trace experiment."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = Path("/home/lyj/mnt/project/zoology/docs/artifacts/20260625-01-flash-vqg-early-window-trace")
EXPERIMENT_ID = "20260625-01-flash-vqg-early-window-trace"


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
    candidates.extend(sorted((outputs_dir / "traces").glob("*/*/early_window_metrics.jsonl")))
    candidates.extend(sorted((outputs_dir / "traces").glob("*/*/train_step_*/read_trace.jsonl*")))
    for path in candidates:
        if not path.exists() or not path.is_file():
            continue
        rows.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "source_machine": path.parts[-4] if "traces" in path.parts else "",
                "source_path": str(path),
                "mirror_path": "",
                "sha256": _sha256(path),
                "file_size": path.stat().st_size,
                "status": "source-local",
            }
        )
    return rows


def _write_metadata(path: Path, outputs_dir: Path, summary: dict[str, Any]) -> None:
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "outputs_dir": str(outputs_dir),
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
        run_rows.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "queue": row.get("queue"),
                "target": row.get("target"),
                "gpu": row.get("gpu"),
                "pid": row.get("pid"),
                "status": row.get("status"),
                "log": row.get("log"),
                "trace_output_dir": row.get("trace_output_dir"),
                "started_at": row.get("started_at"),
                "finished_at": row.get("finished_at"),
            }
        )
    _write_csv(args.artifact_dir / "run-summary.csv", run_rows)

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
