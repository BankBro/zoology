#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPERIMENT_ID = "20260725-01-current-baselines-longer-mqar"
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path("/home/lyj/mnt/project/zoology").resolve()
OUTPUT_ROOT = SCRIPT_DIR / "outputs"
ARTIFACT_DIR = REPO_ROOT / "docs/artifacts" / EXPERIMENT_ID
SLICES = ("1024x256", "2048x512", "4096x1024", "8190x512", "8190x2047")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_inputs() -> tuple[list[dict[str, Any]], list[dict[str, str]], list[dict[str, Any]]]:
    done = OUTPUT_ROOT / "queue/DONE.json"
    if not done.exists() or json.loads(done.read_text(encoding="utf-8")).get("status") != "completed":
        raise RuntimeError("队列尚未DONE, 禁止生成正式artifact.")
    manifest_path = OUTPUT_ROOT / "formal/source-manifest.json"
    detail_path = OUTPUT_ROOT / "formal-eval/detail.csv"
    if not manifest_path.exists() or not detail_path.exists():
        raise FileNotFoundError("缺少formal manifest或detail.csv.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    detail = read_csv(detail_path)
    training = []
    for model in ("flash", "gdn"):
        for seed in (123, 124, 125):
            path = OUTPUT_ROOT / "formal" / f"{model}-s{seed}-fixedinit-s124-d123-b64ga4-formal" / "result.json"
            training.append(json.loads(path.read_text(encoding="utf-8")))
    return manifest, detail, training


def expand_logical_rows(manifest: list[dict[str, Any]], detail: list[dict[str, str]]) -> list[dict[str, Any]]:
    physical = {
        (row["checkpoint_model_state_sha256"], row["slice"]): row
        for row in detail
        if row.get("eval_mode") == "formal" and row.get("status") == "completed"
    }
    rows: list[dict[str, Any]] = []
    for source in manifest:
        for slc in SLICES:
            key = (source["checkpoint_model_state_sha256"], slc)
            if key not in physical:
                raise RuntimeError(f"缺少formal结果: {source['source_id']} {slc}")
            event = physical[key]
            rows.append({
                "source_id": source["source_id"],
                "model": source["model"],
                "config_family": source["config_family"],
                "seed": int(source["seed"]),
                "checkpoint_role": source["checkpoint_role"],
                "checkpoint_path": source["checkpoint_path"],
                "checkpoint_file_sha256": source["checkpoint_file_sha256"],
                "checkpoint_model_state_sha256": source["checkpoint_model_state_sha256"],
                "checkpoint_epoch": source["checkpoint_epoch"],
                "slice": slc,
                "accuracy": float(event["accuracy"]),
                "loss": float(event["loss"]),
                "dataset_hash": event["dataset_hash"],
                "eval_batch_size": int(event["eval_batch_size"]),
                "wall_clock_sec": float(event["wall_clock_sec"]),
                "peak_memory_mb": float(event["peak_memory_mb"]),
                "physical_event_id": event["event_id"],
            })
    if len(rows) != 60:
        raise RuntimeError(f"逻辑formal矩阵应为60行, 实际{len(rows)}")
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for role in ("last", "best"):
        for model in ("flash", "gdn"):
            base = [row for row in rows if row["checkpoint_role"] == role and row["model"] == model and row["slice"] == "1024x256"]
            base_by_seed = {row["seed"]: row["accuracy"] for row in base}
            for slc in SLICES:
                selected = [row for row in rows if row["checkpoint_role"] == role and row["model"] == model and row["slice"] == slc]
                values = [row["accuracy"] for row in selected]
                retentions = [row["accuracy"] / base_by_seed[row["seed"]] if base_by_seed[row["seed"]] else float("nan") for row in selected]
                out.append({
                    "checkpoint_role": role,
                    "model": model,
                    "slice": slc,
                    "n_seeds": len(values),
                    "seeds": "123;124;125",
                    "accuracy_mean": statistics.fmean(values),
                    "accuracy_population_std": statistics.pstdev(values),
                    "accuracy_min": min(values),
                    "accuracy_max": max(values),
                    "retention_mean_vs_1024": statistics.fmean(retentions),
                    "absolute_drop_mean_vs_1024": statistics.fmean(base_by_seed[row["seed"]] - row["accuracy"] for row in selected),
                })
    return out


def paired_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for role in ("last", "best"):
        for slc in SLICES:
            deltas = []
            row: dict[str, Any] = {"checkpoint_role": role, "slice": slc}
            for seed in (123, 124, 125):
                values = {
                    item["model"]: item["accuracy"]
                    for item in rows
                    if item["checkpoint_role"] == role and item["slice"] == slc and item["seed"] == seed
                }
                delta = values["flash"] - values["gdn"]
                row[f"flash_accuracy_s{seed}"] = values["flash"]
                row[f"gdn_accuracy_s{seed}"] = values["gdn"]
                row[f"paired_delta_s{seed}"] = delta
                deltas.append(delta)
            mean_delta = statistics.fmean(deltas)
            positive = sum(delta > 0 for delta in deltas)
            row.update({
                "mean_paired_delta": mean_delta,
                "positive_seed_count": positive,
                "classification": "稳健领先" if positive == 3 else "混合领先" if mean_delta > 0 else "不支持Flash领先",
            })
            out.append(row)
    return out


def checkpoint_role_comparison(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for model in ("flash", "gdn"):
        for seed in (123, 124, 125):
            for slc in SLICES:
                values = {
                    row["checkpoint_role"]: row["accuracy"]
                    for row in rows
                    if row["model"] == model and row["seed"] == seed and row["slice"] == slc
                }
                out.append({
                    "model": model,
                    "seed": seed,
                    "slice": slc,
                    "last_accuracy": values["last"],
                    "best_accuracy": values["best"],
                    "best_minus_last": values["best"] - values["last"],
                })
    return out


def training_rows(training: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in training:
        last = result["last_checkpoint"]
        best = result["best_checkpoint"]
        metrics = last.get("metrics") or {}
        rows.append({
            "experiment_id": EXPERIMENT_ID,
            "model": result["model"],
            "seed": result["seed"],
            "data_seed": result["data_seed"],
            "status": result["status"],
            "configured_max_epochs": result["configured_max_epochs"],
            "final_epoch": last["epoch"],
            "started_at_utc": result["started_at_utc"],
            "ended_at_utc": result["ended_at_utc"],
            "wall_clock_sec": result["wall_clock_sec"],
            "valid_loss": metrics.get("valid/loss", ""),
            "valid_accuracy": metrics.get("valid/accuracy", ""),
            "valid_1024x256": metrics.get("valid/mqar_case/accuracy-1024x256", ""),
            "last_checkpoint_path": last["path"],
            "last_checkpoint_sha256": last["file_sha256"],
            "last_model_state_sha256": last["model_state_sha256"],
            "best_checkpoint_path": best["path"],
            "best_checkpoint_sha256": best["file_sha256"],
            "best_model_state_sha256": best["model_state_sha256"],
            "resolved_config_path": result["resolved_config_path"],
            "resolved_config_sha256": result["resolved_config_sha256"],
        })
    return rows


def collect() -> dict[str, Any]:
    manifest, detail, training = load_inputs()
    logical = expand_logical_rows(manifest, detail)
    summary = summarize(logical)
    deltas = paired_deltas(logical)
    roles = checkpoint_role_comparison(logical)
    train = training_rows(training)
    write_csv(ARTIFACT_DIR / "training-final.csv", train)
    write_csv(ARTIFACT_DIR / "longer-mqar-detail.csv", logical)
    write_csv(ARTIFACT_DIR / "longer-mqar-summary.csv", summary)
    write_csv(ARTIFACT_DIR / "paired-deltas.csv", deltas)
    write_csv(ARTIFACT_DIR / "checkpoint-role-comparison.csv", roles)
    write_csv(ARTIFACT_DIR / "source-manifest.csv", manifest)
    metadata = {
        "experiment_id": EXPERIMENT_ID,
        "status": "completed",
        "training_runs": len(train),
        "logical_formal_rows": len(logical),
        "summary_rows": len(summary),
        "paired_delta_rows": len(deltas),
        "generated_at_utc": utc_now(),
        "raw_output_root": str(OUTPUT_ROOT),
    }
    write_json(ARTIFACT_DIR / "metadata.json", metadata)
    (ARTIFACT_DIR / "README.md").write_text(
        "# 20260725-01 当前基线 Longer-MQAR\n\n"
        "本目录由完整 `DONE.json` 后的审计collector生成. `last.pt`为主结果, `best.pt`为敏感性结果.\n\n"
        "主要文件: `training-final.csv`, `longer-mqar-detail.csv`, `longer-mqar-summary.csv`, "
        "`paired-deltas.csv`, `checkpoint-role-comparison.csv`, `source-manifest.csv`, `metadata.json`.\n",
        encoding="utf-8",
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="生成当前基线Longer-MQAR正式artifact.")
    parser.parse_args()
    metadata = collect()
    print(json.dumps(metadata, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
