#!/usr/bin/env python3
"""Collect lightweight launch metadata and checkpoint metrics after churn probe runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
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

ERROR_PATTERNS = (
    "Traceback",
    "CUDA error",
    "RuntimeError",
    "ValueError",
    "Error:",
    "FAILED",
)

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
    match = re.match(
        r"cb(?P<cb>\d+)r(?P<rank>\d+)-readk(?P<readk>\d+)-s(?P<seed>\d+)-churn",
        target,
    )
    if match is None:
        return {
            "num_codebook_vectors": None,
            "gd_rank": None,
            "read_topk": None,
            "model_seed": None,
        }
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
        return {
            "path": str(path),
            "exists": False,
            "size_bytes": None,
            "epoch": None,
            "metrics": {},
        }
    payload = torch.load(path, map_location="cpu")
    metrics = payload.get("metrics") or {}
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": path.stat().st_size,
        "epoch": payload.get("epoch"),
        "run_id": payload.get("run_id"),
        "launch_id": payload.get("launch_id"),
        "metrics": metrics,
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
        return {
            "path": str(log_path),
            "exists": False,
            "size_bytes": None,
            "sha256": None,
            "error_matches": [],
            "tail": [],
        }
    text = log_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    error_matches = [
        {"line": idx + 1, "text": line[:500]}
        for idx, line in enumerate(lines)
        if any(pattern in line for pattern in ERROR_PATTERNS)
    ]
    return {
        "path": str(log_path),
        "exists": True,
        "size_bytes": log_path.stat().st_size,
        "sha256": _sha256(log_path),
        "error_matches": error_matches[:50],
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


def _find_swanlab_metadata(swanlab_info: dict[str, Any]) -> tuple[Path | None, dict[str, Any]]:
    experiment_id = swanlab_info.get("experiment_id")
    if not experiment_id:
        return None, {}
    for path in sorted(Path("swanlog").glob(f"run-*-{experiment_id}/files/swanlab-metadata.json")):
        return path, _load_json(path)
    return None, {}


def collect(
    *,
    generated_root: Path,
    output_dir: Path,
    launch_prefix: str,
    log_dir: Path | None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    raw_runs: list[dict[str, Any]] = []
    for manifest_path in sorted(generated_root.glob(f"{launch_prefix}*/manifest.json")):
        manifest = _load_json(manifest_path)
        launch_id = str(manifest.get("launch_id"))
        target = _infer_target(launch_id, launch_prefix)
        launch_config_path = manifest_path.parent / "launch_configs.py"
        source_rows.append(_source_row(target, "manifest", manifest_path))
        source_rows.append(_source_row(target, "launch_config", launch_config_path))
        for run in manifest.get("runs", []):
            summary = run.get("config_summary") or {}
            local = run.get("local") or {}
            swanlab = run.get("swanlab") or {}
            train_config_path = Path(local.get("train_config_json") or "")
            train_config = _load_train_config(local.get("train_config_json"))
            mixer_kwargs = _sequence_mixer_kwargs(train_config)
            swanlab_metadata_path, swanlab_metadata = _find_swanlab_metadata(swanlab)
            if train_config_path.exists():
                source_rows.append(_source_row(target, "train_config_json", train_config_path))
            if swanlab_metadata_path is not None and swanlab_metadata_path.exists():
                source_rows.append(_source_row(target, "swanlab_metadata", swanlab_metadata_path))
            current_log_dir = log_dir
            log_path = (
                current_log_dir / f"{target}.log"
                if current_log_dir is not None
                else Path(swanlab.get("run_dir") or "") / "logs" / "debug.log"
            )
            log_info = _log_summary(log_path)
            if log_info["exists"]:
                source_rows.append(_source_row(target, "log", log_path))

            base = {
                "target": target,
                **_parse_target(target),
                "machine": "mclab-3090",
                "gpu": "NVIDIA GeForce RTX 3090",
                "launch_id": launch_id,
                "run_id": run.get("run_id"),
                "status": run.get("status"),
                "started_at_utc": run.get("started_at_utc"),
                "ended_at_utc": run.get("ended_at_utc"),
                "manifest_path": str(manifest_path),
                "launch_config_path": str(launch_config_path),
                "train_config_json": str(train_config_path) if train_config_path else None,
                "log_path": str(log_path),
                "swanlab_run_url": swanlab.get("run_url"),
                "swanlab_logdir": (swanlab_metadata.get("swanlab") or {}).get("logdir"),
                "command": swanlab_metadata.get("command"),
                "cwd": swanlab_metadata.get("cwd"),
                "hostname": swanlab_metadata.get("hostname"),
                "python": swanlab_metadata.get("python"),
                "executable": swanlab_metadata.get("executable"),
                "os_pretty_name": swanlab_metadata.get("os_pretty_name"),
                "git_branch": (swanlab_metadata.get("git_info") or [None, None])[0],
                "git_commit": (swanlab_metadata.get("git_info") or [None, None])[1],
                "git_remote": swanlab_metadata.get("git_remote"),
                "gpu_driver": ((swanlab_metadata.get("gpu") or {}).get("nvidia") or {}).get("driver"),
                "cuda": ((swanlab_metadata.get("gpu") or {}).get("nvidia") or {}).get("cuda"),
                "data_seed": (train_config.get("data") or {}).get("seed"),
                "max_epochs": train_config.get("max_epochs"),
                "validations_per_epoch": train_config.get("validations_per_epoch"),
                "gradient_accumulation_steps": train_config.get("gradient_accumulation_steps"),
                "read_churn_probe_enabled": train_config.get("read_churn_probe_enabled"),
                "read_churn_probe_valid_batches": json.dumps(
                    train_config.get("read_churn_probe_valid_batches"),
                    ensure_ascii=False,
                ),
                "read_churn_probe_max_samples": train_config.get("read_churn_probe_max_samples"),
                "read_churn_probe_query_only": train_config.get("read_churn_probe_query_only"),
                "summary_fox_remote_read_topk": summary.get("fox_remote_read_topk"),
                "summary_fox_gd_residual_rank": summary.get("fox_gd_residual_rank"),
                "summary_codebook_init_seed": summary.get("codebook_init_seed"),
                "mixer_fox_remote_formula": mixer_kwargs.get("fox_remote_formula"),
                "mixer_vq_score_mode": mixer_kwargs.get("vq_score_mode"),
                "mixer_vq_weight_mode": mixer_kwargs.get("vq_weight_mode"),
                "mixer_vq_update_mode": mixer_kwargs.get("vq_update_mode"),
                "log_error_count": len(log_info["error_matches"]),
                "log_size_bytes": log_info["size_bytes"],
            }
            checkpoints = {
                "best": _load_checkpoint_metadata(local.get("best_checkpoint")),
                "final": _load_checkpoint_metadata(local.get("last_checkpoint")),
            }
            for kind, payload in checkpoints.items():
                metrics = (payload or {}).get("metrics") or {}
                row = dict(base)
                row.update(
                    {
                        "checkpoint_kind": kind,
                        "checkpoint_path": (payload or {}).get("path"),
                        "checkpoint_exists": (payload or {}).get("exists"),
                        "checkpoint_epoch": (payload or {}).get("epoch"),
                        "checkpoint_size_bytes": (payload or {}).get("size_bytes"),
                    }
                )
                for key in METRIC_KEYS:
                    row[_metric_column(key)] = metrics.get(key)
                metric_rows.append(row)
            raw_runs.append(
                {
                    "target": target,
                    "manifest": manifest,
                    "swanlab_metadata_path": str(swanlab_metadata_path) if swanlab_metadata_path else None,
                    "swanlab_metadata": swanlab_metadata,
                    "train_config": train_config,
                    "log_summary": log_info,
                    "checkpoints": checkpoints,
                }
            )

    metric_csv_path = output_dir / "final_best_metrics.csv"
    if metric_rows:
        with metric_csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0].keys()))
            writer.writeheader()
            writer.writerows(metric_rows)

    source_csv_path = output_dir / "source_manifest.csv"
    if source_rows:
        with source_csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(source_rows[0].keys()))
            writer.writeheader()
            writer.writerows(source_rows)

    comparisons = _build_comparisons(metric_rows)
    (output_dir / "comparison.json").write_text(
        json.dumps(comparisons, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "raw_summary.json").write_text(
        json.dumps(
            {
                "runs": raw_runs,
                "comparisons": comparisons,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    summary = {
        "num_runs": len(raw_runs),
        "num_metric_rows": len(metric_rows),
        "num_source_rows": len(source_rows),
        "generated_root": str(generated_root),
        "log_dir": str(log_dir) if log_dir is not None else None,
        "output_dir": str(output_dir),
        "metric_csv_path": str(metric_csv_path),
        "source_csv_path": str(source_csv_path),
        "comparison_path": str(output_dir / "comparison.json"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _find_row(rows: list[dict[str, Any]], target: str, kind: str) -> dict[str, Any] | None:
    for row in rows:
        if row.get("target") == target and row.get("checkpoint_kind") == kind:
            return row
    return None


def _delta(left: dict[str, Any], right: dict[str, Any], key: str) -> float | None:
    a = left.get(key)
    b = right.get(key)
    if a is None or b is None:
        return None
    return float(a) - float(b)


def _build_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for kind in ("best", "final"):
        readk4 = _find_row(rows, "cb256r8-readk4-s123-churn", kind)
        readk2 = _find_row(rows, "cb256r8-readk2-s123-churn", kind)
        if readk4 is None or readk2 is None:
            continue
        comparisons.append(
            {
                "comparison": f"cb256r8_readk4_minus_readk2_{kind}",
                "hard_delta": _delta(
                    readk4,
                    readk2,
                    _metric_column("valid/mqar_case/accuracy-1024x256"),
                ),
                "accuracy_delta": _delta(readk4, readk2, _metric_column("valid/accuracy")),
                "churn_delta": _delta(
                    readk4,
                    readk2,
                    _metric_column("valid/attn/gd_residual_read_candidate_churn_mean"),
                ),
                "selected_mass_delta": _delta(
                    readk4,
                    readk2,
                    _metric_column("valid/attn/gd_residual_read_selected_mass_mean"),
                ),
                "entropy_delta": _delta(
                    readk4,
                    readk2,
                    _metric_column("valid/attn/gd_residual_read_entropy_mean"),
                ),
            }
        )
    return comparisons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generated-root",
        type=Path,
        default=Path("/home/lyj/mnt/project/zoology/zoology/experiments/flash_vqg/generated"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--launch-prefix", default="flash-vqg-20260622-04-churn")
    parser.add_argument("--log-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(
        json.dumps(
            collect(
                generated_root=args.generated_root,
                output_dir=args.output_dir,
                launch_prefix=args.launch_prefix,
                log_dir=args.log_dir,
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
