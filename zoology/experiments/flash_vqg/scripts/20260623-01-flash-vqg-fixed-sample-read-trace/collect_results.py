#!/usr/bin/env python3
"""Collect metrics and fixed-sample read trace summaries."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


METRIC_KEYS = [
    "valid/accuracy",
    "valid/loss",
    "valid/mqar_case/accuracy-1024x256",
    "valid/attn/gd_residual_read_candidate_probe_count",
    "valid/attn/gd_residual_read_candidate_has_prev",
    "valid/attn/gd_residual_read_candidate_retention_mean",
    "valid/attn/gd_residual_read_candidate_churn_mean",
    "valid/attn/gd_residual_read_candidate_top1_flip_rate",
    "valid/attn/gd_residual_read_margin_top1_top2_mean",
    "valid/attn/gd_residual_read_margin_top1_top2_p05",
    "valid/attn/gd_residual_read_entropy_mean",
    "valid/attn/gd_residual_read_selected_mass_mean",
    "valid/attn/gd_residual_read_selected_mass_p05",
    "valid/attn/gd_residual_remote_read_topk_effective",
    "valid/attn/gd_residual_inject_ratio",
    "valid/attn/gd_residual_lambda_mean",
    "valid/attn/gd_residual_write_strength_mean",
    "valid/attn/gd_residual_write_strength_max",
    "valid/attn/gd_residual_write_strength_p95",
    "valid/attn/gd_residual_sum_zeta_mean",
    "valid/attn/gd_residual_sum_zeta_max",
    "valid/attn/gd_residual_sum_zeta_p95",
    "valid/attn/gd_residual_m_norm_mean",
    "valid/attn/gd_residual_m_norm_max",
    "valid/attn/gd_residual_beta_mean",
    "valid/attn/gd_residual_beta_max",
]

ERROR_PATTERNS = ("Traceback", "CUDA error", "RuntimeError", "ValueError", "Error:", "FAILED")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric_column(metric: str) -> str:
    return metric.replace("/", "__").replace("-", "_")


def _infer_target(launch_id: str, launch_prefix: str) -> str:
    tail = launch_id.removeprefix(f"{launch_prefix}-")
    return re.sub(r"-\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$", "", tail)


def _parse_target(target: str) -> dict[str, int | None]:
    match = re.match(r"cb(?P<cb>\d+)r(?P<rank>\d+)-readk(?P<readk>\d+)-s(?P<seed>\d+)-trace", target)
    if match is None:
        return {"num_codebook_vectors": None, "gd_rank": None, "read_topk": None, "model_seed": None}
    return {
        "num_codebook_vectors": int(match.group("cb")),
        "gd_rank": int(match.group("rank")),
        "read_topk": int(match.group("readk")),
        "model_seed": int(match.group("seed")),
    }


def _load_checkpoint_metadata(path_raw: str | None) -> dict[str, Any] | None:
    if not path_raw:
        return None
    path = Path(path_raw)
    if not path.exists():
        return {"path": str(path), "exists": False, "size_bytes": None, "epoch": None, "metrics": {}}
    payload = torch.load(path, map_location="cpu")
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": path.stat().st_size,
        "epoch": payload.get("epoch"),
        "run_id": payload.get("run_id"),
        "launch_id": payload.get("launch_id"),
        "metrics": payload.get("metrics") or {},
    }


def _load_train_config(path_raw: str | None) -> dict[str, Any]:
    if not path_raw:
        return {}
    path = Path(path_raw)
    if not path.exists():
        return {}
    return _load_json(path)


def _sequence_mixer_kwargs(train_config: dict[str, Any]) -> dict[str, Any]:
    model_cfg = train_config.get("model") or {}
    mixer = model_cfg.get("sequence_mixer") or {}
    for item in reversed((mixer.get("kwargs") or {}).get("configs") or []):
        if item.get("name") == "zoology.mixers.flash_vqg.FlashVQGMixer":
            return item.get("kwargs") or {}
    return {}


def _log_summary(log_path: Path) -> dict[str, Any]:
    if not log_path.exists():
        return {"path": str(log_path), "exists": False, "size_bytes": None, "sha256": None, "error_matches": [], "tail": []}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    return {
        "path": str(log_path),
        "exists": True,
        "size_bytes": log_path.stat().st_size,
        "sha256": _sha256(log_path),
        "error_matches": [
            {"line": idx + 1, "text": line[:500]}
            for idx, line in enumerate(lines)
            if any(pattern in line for pattern in ERROR_PATTERNS)
        ][:50],
        "tail": lines[-80:],
    }


def _source_row(target: str, source_type: str, path: Path) -> dict[str, Any]:
    exists = path.exists()
    return {
        "target": target,
        "source_type": source_type,
        "path": str(path),
        "exists": exists,
        "size_bytes": path.stat().st_size if exists else None,
        "sha256": _sha256(path) if exists else None,
    }


def _compress_trace(src: Path, dst: Path) -> dict[str, Any]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("rb") as fin, gzip.open(dst, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    return {
        "source_path": str(src),
        "archive_path": str(dst),
        "source_size_bytes": src.stat().st_size,
        "archive_size_bytes": dst.stat().st_size,
        "source_sha256": _sha256(src),
        "archive_sha256": _sha256(dst),
    }


def _record_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("input_hash"),
        record.get("valid_batch_idx"),
        record.get("sample_idx"),
        record.get("layer_idx"),
        record.get("head_idx"),
        record.get("query_pos"),
    )


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _summarize_trace(trace_path: Path) -> dict[str, Any]:
    if not trace_path.exists():
        return {"trace_exists": False, "trace_records": 0}
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    steps: set[int] = set()
    margin_values: list[float] = []
    entropy_values: list[float] = []
    selected_mass_values: list[float] = []
    read_topk_values: set[int] = set()
    with trace_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            groups[_record_key(record)].append(record)
            if record.get("global_step") is not None:
                steps.add(int(record["global_step"]))
            if record.get("read_topk") is not None:
                read_topk_values.add(int(record["read_topk"]))
            if record.get("margin_top1_top2") is not None:
                margin_values.append(float(record["margin_top1_top2"]))
            if record.get("entropy") is not None:
                entropy_values.append(float(record["entropy"]))
            if record.get("selected_mass") is not None:
                selected_mass_values.append(float(record["selected_mass"]))

    retention_values: list[float] = []
    churn_values: list[float] = []
    top1_flip_values: list[float] = []
    rank_displacements: list[float] = []
    top2_shadow_retention_values: list[float] = []
    for records in groups.values():
        records.sort(key=lambda item: (int(item.get("global_step") or 0), int(item.get("epoch") or 0)))
        for prev, cur in zip(records, records[1:]):
            prev_ids = [int(v) for v in prev.get("topk_candidate_ids") or []]
            cur_ids = [int(v) for v in cur.get("topk_candidate_ids") or []]
            if not prev_ids or not cur_ids:
                continue
            overlap = len(set(prev_ids) & set(cur_ids))
            denom = max(1, len(cur_ids))
            retention = overlap / denom
            retention_values.append(retention)
            churn_values.append(1.0 - retention)
            top1_flip_values.append(0.0 if prev_ids[0] == cur_ids[0] else 1.0)
            shared = set(prev_ids) & set(cur_ids)
            for candidate in shared:
                rank_displacements.append(abs(prev_ids.index(candidate) - cur_ids.index(candidate)))
            prev_top2 = prev_ids[:2]
            cur_top2 = cur_ids[:2]
            if prev_top2 and cur_top2:
                top2_shadow_retention_values.append(len(set(prev_top2) & set(cur_top2)) / max(1, len(cur_top2)))

    return {
        "trace_exists": True,
        "trace_records": sum(len(records) for records in groups.values()),
        "trace_groups": len(groups),
        "global_steps": json.dumps(sorted(steps)),
        "num_global_steps": len(steps),
        "read_topk_values": json.dumps(sorted(read_topk_values)),
        "retention_mean": _mean(retention_values),
        "churn_mean": _mean(churn_values),
        "top1_flip_rate": _mean(top1_flip_values),
        "rank_displacement_mean": _mean(rank_displacements),
        "top2_shadow_retention_mean": _mean(top2_shadow_retention_values),
        "top2_shadow_churn_mean": (None if not top2_shadow_retention_values else 1.0 - _mean(top2_shadow_retention_values)),
        "margin_mean": _mean(margin_values),
        "entropy_mean": _mean(entropy_values),
        "selected_mass_mean": _mean(selected_mass_values),
    }


def _summarize_trace_steps(trace_path: Path, target: str) -> list[dict[str, Any]]:
    if not trace_path.exists():
        return []
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with trace_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("global_step") is not None:
                groups[int(record["global_step"])].append(record)

    rows: list[dict[str, Any]] = []
    for step, records in sorted(groups.items()):
        selected_mass = [
            float(record["selected_mass"])
            for record in records
            if record.get("selected_mass") is not None
        ]
        margins = [
            float(record["margin_top1_top2"])
            for record in records
            if record.get("margin_top1_top2") is not None
        ]
        entropies = [
            float(record["entropy"])
            for record in records
            if record.get("entropy") is not None
        ]
        rows.append(
            {
                "target": target,
                "global_step": step,
                "records": len(records),
                "selected_mass_mean": _mean(selected_mass),
                "selected_mass_min": min(selected_mass) if selected_mass else None,
                "selected_mass_max": max(selected_mass) if selected_mass else None,
                "margin_mean": _mean(margins),
                "entropy_mean": _mean(entropies),
            }
        )
    return rows


def collect(
    *,
    generated_root: Path,
    output_dir: Path,
    launch_prefix: str,
    launch_id_contains: str | None,
    log_dir: Path | None,
    trace_root: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "traces").mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    raw_runs: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    trace_step_rows: list[dict[str, Any]] = []
    archive_rows: list[dict[str, Any]] = []

    for manifest_path in sorted(generated_root.glob(f"{launch_prefix}*/manifest.json")):
        manifest = _load_json(manifest_path)
        launch_id = str(manifest.get("launch_id"))
        if launch_id_contains is not None and launch_id_contains not in launch_id:
            continue
        target = _infer_target(launch_id, launch_prefix)
        launch_config_path = manifest_path.parent / "launch_configs.py"
        source_rows.append(_source_row(target, "manifest", manifest_path))
        source_rows.append(_source_row(target, "launch_config", launch_config_path))

        for run in manifest.get("runs", []):
            summary = run.get("config_summary") or {}
            local = run.get("local") or {}
            swanlab = run.get("swanlab") or {}
            train_config = _load_train_config(local.get("train_config_json"))
            mixer_kwargs = _sequence_mixer_kwargs(train_config)
            train_config_path = Path(local.get("train_config_json") or "")
            if train_config_path.exists():
                source_rows.append(_source_row(target, "train_config_json", train_config_path))
            log_path = (log_dir / f"{target}.log") if log_dir is not None else Path(f"missing-{target}.log")
            log_info = _log_summary(log_path)
            if log_info["exists"]:
                source_rows.append(_source_row(target, "log", log_path))

            checkpoints = {
                "final": _load_checkpoint_metadata(local.get("last_checkpoint")),
                "best": _load_checkpoint_metadata(local.get("best_checkpoint")),
            }
            trace_path = trace_root / target / "read_trace.jsonl"
            trace_summary = _summarize_trace(trace_path)
            trace_step_rows.extend(_summarize_trace_steps(trace_path, target))
            if trace_path.exists():
                archived_trace = output_dir / "traces" / f"{target}.jsonl.gz"
                archive_info = _compress_trace(trace_path, archived_trace)
                source_rows.append(_source_row(target, "trace_jsonl", trace_path))
                source_rows.append(_source_row(target, "trace_jsonl_gz", archived_trace))
                archive_rows.append({"target": target, **archive_info})
            trace_rows.append({"target": target, **_parse_target(target), **trace_summary})

            base = {
                "target": target,
                **_parse_target(target),
                "machine": "mclab-3090",
                "launch_id": launch_id,
                "run_id": run.get("run_id"),
                "status": run.get("status"),
                "started_at_utc": run.get("started_at_utc"),
                "ended_at_utc": run.get("ended_at_utc"),
                "manifest_path": str(manifest_path),
                "launch_config_path": str(launch_config_path),
                "train_config_json": local.get("train_config_json"),
                "log_path": str(log_path),
                "swanlab_run_url": swanlab.get("run_url"),
                "command": local.get("command"),
                "data_seed": train_config.get("data", {}).get("seed"),
                "max_epochs": train_config.get("max_epochs"),
                "validations_per_epoch": train_config.get("validations_per_epoch"),
                "gradient_accumulation_steps": train_config.get("gradient_accumulation_steps"),
                "read_trace_enabled": train_config.get("read_trace_enabled"),
                "read_trace_valid_batches": json.dumps(train_config.get("read_trace_valid_batches")),
                "read_trace_max_samples": train_config.get("read_trace_max_samples"),
                "read_trace_max_queries_per_sample": train_config.get("read_trace_max_queries_per_sample"),
                "read_trace_output_dir": train_config.get("read_trace_output_dir"),
                "summary_fox_remote_read_topk": summary.get("fox_remote_read_topk"),
                "summary_fox_gd_residual_rank": summary.get("fox_gd_residual_rank"),
                "mixer_fox_remote_formula": mixer_kwargs.get("fox_remote_formula"),
                "mixer_vq_score_mode": mixer_kwargs.get("vq_score_mode"),
                "mixer_vq_weight_mode": mixer_kwargs.get("vq_weight_mode"),
                "mixer_vq_update_mode": mixer_kwargs.get("vq_update_mode"),
                "log_error_count": len(log_info.get("error_matches") or []),
                "log_size_bytes": log_info.get("size_bytes"),
            }
            for kind, checkpoint in checkpoints.items():
                row = dict(base)
                row["checkpoint_kind"] = kind
                if checkpoint is None:
                    checkpoint = {"path": None, "exists": False, "epoch": None, "size_bytes": None, "metrics": {}}
                row.update(
                    {
                        "checkpoint_path": checkpoint.get("path"),
                        "checkpoint_exists": checkpoint.get("exists"),
                        "checkpoint_epoch": checkpoint.get("epoch"),
                        "checkpoint_size_bytes": checkpoint.get("size_bytes"),
                    }
                )
                metrics = checkpoint.get("metrics") or {}
                for metric in METRIC_KEYS:
                    row[_metric_column(metric)] = metrics.get(metric)
                metric_rows.append(row)

            raw_runs.append(
                {
                    "target": target,
                    "manifest": manifest,
                    "checkpoints": checkpoints,
                    "log": log_info,
                    "trace_summary": trace_summary,
                }
            )

    def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    final_rows = [row for row in metric_rows if row.get("checkpoint_kind") == "final"]
    write_csv(output_dir / "final.csv", final_rows)
    write_csv(output_dir / "final_best_metrics.csv", metric_rows)
    write_csv(output_dir / "source_manifest.csv", source_rows)
    write_csv(output_dir / "trace_summary.csv", trace_rows)
    write_csv(output_dir / "trace_step_summary.csv", trace_step_rows)
    write_csv(output_dir / "trace_archives.csv", archive_rows)

    metadata = {
        "launch_prefix": launch_prefix,
        "launch_id_contains": launch_id_contains,
        "generated_root": str(generated_root),
        "trace_root": str(trace_root),
        "num_runs": len(final_rows),
        "targets": sorted({row["target"] for row in metric_rows}),
        "trace_rows": trace_rows,
        "trace_step_rows": trace_step_rows,
        "archive_rows": archive_rows,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "raw_summary.json").write_text(json.dumps({"runs": raw_runs}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# 20260623-01 Flash-VQG fixed-sample read trace artifact\n\n"
        "This artifact contains lightweight CSV/JSON summaries and compressed JSONL read traces. "
        "Checkpoint `.pt` files are not copied into git artifacts.\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-root", type=Path, default=Path("zoology/experiments/flash_vqg/generated"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/artifacts/20260623-01-flash-vqg-fixed-sample-read-trace"))
    parser.add_argument("--launch-prefix", default="flash-vqg-20260623-01-read-trace")
    parser.add_argument("--launch-id-contains", default=None)
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument(
        "--trace-root",
        type=Path,
        default=Path("zoology/experiments/flash_vqg/scripts/20260623-01-flash-vqg-fixed-sample-read-trace/outputs/traces"),
    )
    args = parser.parse_args()
    metadata = collect(
        generated_root=args.generated_root,
        output_dir=args.output_dir,
        launch_prefix=args.launch_prefix,
        launch_id_contains=args.launch_id_contains,
        log_dir=args.log_dir,
        trace_root=args.trace_root,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
