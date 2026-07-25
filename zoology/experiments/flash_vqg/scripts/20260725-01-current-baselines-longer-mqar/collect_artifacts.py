#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
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
GPU_INDEX = "1"
GPU_NAME = "NVIDIA GeForce RTX 2080 Ti"
ZOOLOGY_STATE_SIZE_SEQ1024 = {"flash": 624_640}


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
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_csv_fields(path: Path, additions: list[str]) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        existing = list(reader)
    missing = [field for field in additions if field not in fields]
    if not missing:
        return
    fields.extend(missing)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in existing)
    tmp.replace(path)


def append_unique_csv(path: Path, new_rows: list[dict[str, Any]], key: str) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        existing = list(reader)
    unknown = sorted({column for row in new_rows for column in row if column not in fields})
    if unknown:
        raise RuntimeError(f"{path}出现未知ledger字段: {unknown}")
    by_key = {row[key]: row for row in existing}
    updated = False
    for row in new_rows:
        previous = by_key.get(str(row[key]))
        if previous is None:
            continue
        for field in fields:
            if not previous.get(field) and row.get(field, "") != "":
                previous[field] = row[field]
                updated = True
    existing_keys = set(by_key)
    additions = [row for row in new_rows if str(row[key]) not in existing_keys]
    if not additions and not updated:
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in existing + additions)
    tmp.replace(path)


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
                "started_at_utc": event["started_at_utc"],
                "ended_at_utc": event["ended_at_utc"],
                "status": event["status"],
                "gpu": GPU_INDEX,
                "gpu_name": GPU_NAME,
                "dtype_policy": "float32_ieee_tf32_off",
                "outer_model_dtype": "float32",
                "GDN_KERNEL_DTYPE": "float32" if source["model"] == "gdn" else "not_applicable",
                "TRITON_F32_DEFAULT": "ieee",
                "tf32_enabled": False,
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
            "run_id": result["run_id"],
            "launch_id": result["launch_id"],
            "model": result["model"],
            "seed": result["seed"],
            "data_seed": result["data_seed"],
            "status": result["status"],
            "configured_max_epochs": result["configured_max_epochs"],
            "final_epoch": last["epoch"],
            "started_at_utc": result["started_at_utc"],
            "ended_at_utc": result["ended_at_utc"],
            "wall_clock_sec": result["wall_clock_sec"],
            "gpu": GPU_INDEX,
            "gpu_name": GPU_NAME,
            "dtype_policy": "float32_ieee_tf32_off",
            "outer_model_dtype": "float32",
            "GDN_KERNEL_DTYPE": "float32" if result["model"] == "gdn" else "not_applicable",
            "TRITON_F32_DEFAULT": "ieee",
            "tf32_enabled": False,
            "train_batch_size": 64,
            "eval_batch_size": 16,
            "gradient_accumulation_steps": 4,
            "effective_train_batch_size": 256,
            "validations_per_epoch": 4,
            "trainable_params": 1_160_390 if result["model"] == "flash" else 1_335_942,
            "valid_loss": metrics.get("valid/loss", ""),
            "valid_accuracy": metrics.get("valid/accuracy", ""),
            "valid_1024x256": metrics.get("valid/mqar_case/accuracy-1024x256", ""),
            "valid_512x128": metrics.get("valid/mqar_case/accuracy-512x128", ""),
            "valid_512x64": metrics.get("valid/mqar_case/accuracy-512x64", ""),
            "valid_256x64": metrics.get("valid/mqar_case/accuracy-256x64", ""),
            "valid_input_seq_len_512": metrics.get("valid/input_seq_len/accuracy-512", ""),
            "valid_input_seq_len_1024": metrics.get("valid/input_seq_len/accuracy-1024", ""),
            "valid_num_kv_pairs_128": metrics.get("valid/num_kv_pairs/accuracy-128", ""),
            "valid_num_kv_pairs_256": metrics.get("valid/num_kv_pairs/accuracy-256", ""),
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


def flash_ledger_rows(train: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in train:
        if item["model"] != "flash":
            continue
        seed = int(item["seed"])
        rows.append({
            "summary_scope": "epoch4_final_only",
            "comparison_scope": "gd_residual_v1_current_baseline_longer_mqar_b64_ga4_fp32",
            "config_family": "cb64-r16-joint",
            "config": f"baseline-r16-joint-s{seed}-d123",
            "rank": 16,
            "seed": seed,
            "data_seed": 123,
            "num_codebook_vectors": 64,
            "train_batch_size": 64,
            "eval_batch_size": 16,
            "gradient_accumulation_steps": 4,
            "effective_train_batch_size": 256,
            "batch_accum_profile": "b64_ga4",
            "configured_max_epochs": 4,
            "max_epochs_source": "explicit max_epochs=4; current baseline Longer-MQAR retraining",
            "final_epoch": 4,
            "validations_per_epoch": 4,
            "final_validation_index": 16,
            "final_validation_phase": "epoch_end",
            "checkpoint_label": "epoch4_noearly_b64_ga4_fp32",
            "early_stopping_disabled": "true",
            "replicate_id": "current_lg_20260725",
            "run_type": "current_baseline_length_generalization_4ep",
            "gpu": GPU_INDEX,
            "dynamic_capacity": 131_072,
            "relative_to_gdn": "1.00x",
            "status": item["status"],
            "trainable_params": item["trainable_params"],
            "zoology_state_size_metric_seq1024": ZOOLOGY_STATE_SIZE_SEQ1024["flash"],
            "elapsed_sec": item["wall_clock_sec"],
            "wall_clock": str(datetime.fromtimestamp(float(item["wall_clock_sec"]), timezone.utc).strftime("%H:%M:%S")),
            "started_at_utc": item["started_at_utc"],
            "ended_at_utc": item["ended_at_utc"],
            "gpu_name": GPU_NAME,
            "valid_loss": item["valid_loss"],
            "valid_accuracy": item["valid_accuracy"],
            "valid_mqar_case_accuracy_1024x256": item["valid_1024x256"],
            "valid_mqar_case_accuracy_512x128": item["valid_512x128"],
            "valid_mqar_case_accuracy_512x64": item["valid_512x64"],
            "valid_mqar_case_accuracy_256x64": item["valid_256x64"],
            "valid_input_seq_len_accuracy_512": item["valid_input_seq_len_512"],
            "valid_input_seq_len_accuracy_1024": item["valid_input_seq_len_1024"],
            "valid_num_kv_pairs_accuracy_128": item["valid_num_kv_pairs_128"],
            "valid_num_kv_pairs_accuracy_256": item["valid_num_kv_pairs_256"],
            "source_artifact": f"docs/artifacts/{EXPERIMENT_ID}/training-final.csv",
            "source_precision": "full_precision",
            "source_run_set": EXPERIMENT_ID.replace("-", "_"),
            "run_id": item["run_id"],
            "note": "Current baseline 4ep retraining from fixed family-specific canonical seed124 init; Longer-MQAR source run.",
            "dtype_policy": "float32",
            "outer_model_dtype": "float32",
            "hidden_states_dtype": "float32",
            "kernel_input_dtype": "float32",
            "actual_kernel_dtype": "float32",
            "dtype_comparison_scope": "float32_only",
            "official_scope": "b64_ga4_fp32_official",
            "metadata_verification_level": "verified_runtime_artifact",
            "train_config_path": str(Path(item["last_checkpoint_path"]).parent / "train_config.json"),
            "last_checkpoint_path": item["last_checkpoint_path"],
            "best_checkpoint_path": item["best_checkpoint_path"],
            "metadata_backfill_status": "native_artifact_metadata_normalized",
        })
    return rows


def gdn_ledger_rows(train: list[dict[str, Any]], preflight: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in train:
        if item["model"] != "gdn":
            continue
        rows.append({
            "run_id": item["run_id"],
            "model_family": "GatedDeltaNetExpandedK",
            "seed": item["seed"],
            "data_seed": 123,
            "num_heads": 2,
            "expand_k": 4,
            "expand_v": 4,
            "head_k_dim": 256,
            "head_v_dim": 256,
            "dynamic_state_capacity": 131_072,
            "dtype_policy": "float32_ieee_tf32_off",
            "GDN_KERNEL_DTYPE": "float32",
            "gpu": GPU_INDEX,
            "gpu_name": GPU_NAME,
            "train_batch_size": 64,
            "eval_batch_size": 16,
            "validation_eval_batch_size": 16,
            "gradient_accumulation_steps": 4,
            "effective_train_batch_size": 256,
            "batch_accum_profile": "b64_ga4",
            "configured_max_epochs": 4,
            "final_epoch": 4,
            "checkpoint_path": item["last_checkpoint_path"],
            "checkpoint_hash": item["last_checkpoint_sha256"],
            "started_at_utc": item["started_at_utc"],
            "ended_at_utc": item["ended_at_utc"],
            "wall_clock_sec": item["wall_clock_sec"],
            "status": item["status"],
            "fla_version": "0.4.2",
            "torch_version": "2.6.0+cu118",
            "triton_version": "3.2.0",
            "valid_loss": item["valid_loss"],
            "valid_accuracy": item["valid_accuracy"],
            "valid_mqar_case_accuracy_1024x256": item["valid_1024x256"],
            "source_artifact": f"docs/artifacts/{EXPERIMENT_ID}/training-final.csv",
            "run_type": "current_baseline_length_generalization_4ep",
            "zoology_branch": preflight["environment"]["zoology_branch"],
            "zoology_commit": preflight["environment"]["zoology_commit"],
            "note": "Current GDN ek4-ev4 baseline 4ep retraining from fixed canonical seed124 init; Longer-MQAR source run.",
        })
    return rows


def collect() -> dict[str, Any]:
    manifest, detail, training = load_inputs()
    logical = expand_logical_rows(manifest, detail)
    summary = summarize(logical)
    deltas = paired_deltas(logical)
    roles = checkpoint_role_comparison(logical)
    train = training_rows(training)
    preflight = json.loads((OUTPUT_ROOT / "preflight.json").read_text(encoding="utf-8"))
    queue_done = json.loads((OUTPUT_ROOT / "queue/DONE.json").read_text(encoding="utf-8"))
    smoke_gate = json.loads((OUTPUT_ROOT / "gates/SMOKE_DONE.json").read_text(encoding="utf-8"))
    eval_verification = json.loads((OUTPUT_ROOT / "formal-eval/verification.json").read_text(encoding="utf-8"))
    repro = json.loads((OUTPUT_ROOT / "formal-eval/repro-verification.json").read_text(encoding="utf-8"))
    batch_sizes = json.loads((OUTPUT_ROOT / "formal-eval/batch-sizes.json").read_text(encoding="utf-8"))
    if not all((
        preflight.get("status") == "passed",
        queue_done.get("status") == "completed",
        smoke_gate.get("status") == "passed",
        eval_verification.get("status") == "completed",
        eval_verification.get("all_formal_completed") is True,
        all(row.get("passed") for row in repro),
    )):
        raise RuntimeError("正式artifact输入审计未通过.")
    source_rows = []
    for row in manifest:
        checkpoint_path = Path(row["checkpoint_path"])
        source_rows.append({
            **row,
            "checkpoint_size_bytes": checkpoint_path.stat().st_size,
            "checkpoint_hash_verified": sha256_file(checkpoint_path) == row["checkpoint_file_sha256"],
        })
    write_csv(ARTIFACT_DIR / "training-final.csv", train)
    write_csv(ARTIFACT_DIR / "longer-mqar-detail.csv", logical)
    write_csv(ARTIFACT_DIR / "longer-mqar-summary.csv", summary)
    write_csv(ARTIFACT_DIR / "paired-deltas.csv", deltas)
    write_csv(ARTIFACT_DIR / "checkpoint-role-comparison.csv", roles)
    write_csv(ARTIFACT_DIR / "source-manifest.csv", source_rows)
    write_csv(ARTIFACT_DIR / "batch-sizes.csv", batch_sizes)
    write_csv(ARTIFACT_DIR / "repro-verification.csv", repro)
    write_json(ARTIFACT_DIR / "verification.json", {
        "status": "passed",
        "preflight": preflight["status"],
        "training_runs_completed": sum(row["status"] == "completed" and row["final_epoch"] == 4 for row in train),
        "logical_checkpoint_roles": len(manifest),
        "unique_checkpoint_model_states": len({(row["model"], row["checkpoint_model_state_sha256"]) for row in manifest}),
        "logical_formal_rows": len(logical),
        "physical_formal_rows": eval_verification["formal_rows"],
        "repro_rows": len(repro),
        "repro_all_passed": all(row["passed"] for row in repro),
        "repro_max_accuracy_delta_abs": max(float(row["accuracy_delta_abs"]) for row in repro),
        "dataset_hashes": {slc: next(row["dataset_hash"] for row in logical if row["slice"] == slc) for slc in SLICES},
        "queue": queue_done,
        "smoke_gate": smoke_gate,
    })
    metadata = {
        "experiment_id": EXPERIMENT_ID,
        "status": "completed",
        "training_runs": len(train),
        "logical_formal_rows": len(logical),
        "summary_rows": len(summary),
        "paired_delta_rows": len(deltas),
        "physical_formal_rows": eval_verification["formal_rows"],
        "logical_checkpoint_roles": len(manifest),
        "unique_checkpoint_model_states": len({(row["model"], row["checkpoint_model_state_sha256"]) for row in manifest}),
        "repro_rows": len(repro),
        "repro_all_passed": all(row["passed"] for row in repro),
        "queue_started_at_utc": queue_done["started_at_utc"],
        "queue_ended_at_utc": queue_done["ended_at_utc"],
        "gpu": GPU_INDEX,
        "gpu_name": GPU_NAME,
        "dtype_policy": "float32_ieee_tf32_off",
        "GDN_KERNEL_DTYPE": "float32",
        "TRITON_F32_DEFAULT": "ieee",
        "zoology_branch": preflight["environment"]["zoology_branch"],
        "zoology_commit": preflight["environment"]["zoology_commit"],
        "flash_commit": preflight["environment"]["flash_commit"],
        "preflight_failed_attempts_before_pass": 1,
        "preflight_failure_reason": "Runner入口原名queue.py遮蔽Python标准库queue; formal/smoke均未启动; commit 0dd9572修复后重跑通过.",
        "generated_at_utc": utc_now(),
        "raw_output_root": str(OUTPUT_ROOT),
    }
    write_json(ARTIFACT_DIR / "metadata.json", metadata)
    flash_ledger = REPO_ROOT / "docs/artifacts/gd-residual-v1/rank-seed-effect-summary.csv"
    ensure_csv_fields(flash_ledger, ["started_at_utc", "ended_at_utc", "gpu_name"])
    append_unique_csv(
        flash_ledger,
        flash_ledger_rows(train),
        "run_id",
    )
    append_unique_csv(
        REPO_ROOT / "docs/artifacts/gdn-expanded-k/gdn-expanded-k-summary.csv",
        gdn_ledger_rows(train, preflight),
        "run_id",
    )
    (ARTIFACT_DIR / "README.md").write_text(
        "# 20260725-01 当前基线 Longer-MQAR\n\n"
        "本目录由完整 `DONE.json` 后的审计 collector生成. `last.pt`为预注册主结果, "
        "`best.pt`为 epoch-end checkpoint敏感性结果.\n\n"
        "主结果显示, Flash在 `1024x256` 不支持领先; 四个真正外推 slice中, 三个为 "
        "3/3 seeds稳健领先, `8190x512`为均值领先但 2/3 seeds的混合领先. "
        "`best.pt`敏感性在全部四个外推 slice为 3/3 seeds稳健领先. 主要方差来源是 Flash seed124.\n\n"
        "主要文件:\n\n"
        "- `training-final.csv`: 6条 epoch4训练和时间/dtype/GPU/checkpoint信息.\n"
        "- `longer-mqar-detail.csv`: 60条 last/best逻辑 formal结果.\n"
        "- `longer-mqar-summary.csv`: checkpoint role × model × slice三 seed汇总.\n"
        "- `paired-deltas.csv`: 同 seed Flash-GDN paired delta和预注册分类.\n"
        "- `checkpoint-role-comparison.csv`: best-last敏感性.\n"
        "- `source-manifest.csv`: 12个逻辑角色的 checkpoint来源、hash和大小.\n"
        "- `batch-sizes.csv`, `repro-verification.csv`, `verification.json`, `metadata.json`: 执行与审计证据.\n\n"
        "- `figures/`: 当前两模型 `last.pt` 三 seed Longer-MQAR曲线的 PDF/PNG/SVG、绘图数据.\n\n"
        "完整解释见 `docs/20260725-01-current-baselines-longer-mqar-report.md`. "
        "Raw输出保留在 `zoology/experiments/flash_vqg/scripts/20260725-01-current-baselines-longer-mqar/outputs/`. "
        "本轮使用专用 collector直接生成统计, 未另跑 analysis suite, 因而没有 "
        "`zoology/analysis/flash_vqg/results/<launch_id>/` 目录.\n",
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
