#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPERIMENT_ID = "20260725-01-current-baselines-longer-mqar"
REPO_ROOT = Path("/home/lyj/mnt/project/zoology").resolve()
ARTIFACT_ROOT = REPO_ROOT / "docs/artifacts" / EXPERIMENT_ID
COMBINED_DIR = ARTIFACT_ROOT / "combined"
MACHINES = ("2080ti", "3090")
MODELS = ("flash", "gdn")
SEEDS = (123, 124, 125)
ROLES = ("last", "best")
SLICES = ("1024x256", "2048x512", "4096x1024", "8190x512", "8190x2047")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
    tmp.replace(path)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_unique_csv(path: Path, new_rows: list[dict[str, str]], key: str) -> tuple[int, int]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        existing = list(reader)
    unknown = sorted({column for row in new_rows for column in row if column not in fields})
    if unknown:
        raise RuntimeError(f"{path}出现未知ledger字段: {unknown}")
    by_key = {row[key]: row for row in existing}
    updated = 0
    for row in new_rows:
        previous = by_key.get(row[key])
        if previous is None:
            continue
        changed = False
        for field in fields:
            if not previous.get(field) and row.get(field):
                previous[field] = row[field]
                changed = True
        updated += int(changed)
    existing_keys = set(by_key)
    additions = [row for row in new_rows if row[key] not in existing_keys]
    if additions or updated:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows({field: row.get(field, "") for field in fields} for row in existing + additions)
        tmp.replace(path)
    return len(additions), updated


def validate_machine(machine: str) -> dict[str, Any]:
    root = ARTIFACT_ROOT / "machines" / machine
    required = (
        "training-final.csv",
        "longer-mqar-detail.csv",
        "longer-mqar-summary.csv",
        "paired-deltas.csv",
        "checkpoint-role-comparison.csv",
        "source-manifest.csv",
        "batch-sizes.csv",
        "repro-verification.csv",
        "raw-evidence-manifest.csv",
        "flash-ledger-rows.csv",
        "gdn-ledger-rows.csv",
        "verification.json",
        "metadata.json",
    )
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise RuntimeError(f"{machine}机器artifact缺文件: {missing}")

    verification = json.loads((root / "verification.json").read_text(encoding="utf-8"))
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if verification.get("status") != "passed" or metadata.get("status") != "completed":
        raise RuntimeError(f"{machine}机器artifact未通过: {verification.get('status')}, {metadata.get('status')}")
    if metadata.get("machine") != machine:
        raise RuntimeError(f"{machine} metadata机器字段错误: {metadata.get('machine')}")

    detail = read_csv(root / "longer-mqar-detail.csv")
    keys = {
        (row["machine"], row["model"], int(row["seed"]), row["checkpoint_role"], row["slice"])
        for row in detail
    }
    expected_keys = {
        (machine, model, seed, role, slc)
        for model in MODELS
        for seed in SEEDS
        for role in ROLES
        for slc in SLICES
    }
    if len(detail) != 60 or keys != expected_keys:
        raise RuntimeError(f"{machine} detail矩阵不完整: rows={len(detail)}, keys={len(keys)}")
    if any(row.get("status") != "completed" for row in detail):
        raise RuntimeError(f"{machine} detail存在未完成行.")

    evidence = read_csv(root / "raw-evidence-manifest.csv")
    for row in evidence:
        mirror = REPO_ROOT / row["mirror_path"]
        if not mirror.exists() or sha256_file(mirror) != row["sha256"]:
            raise RuntimeError(f"{machine} raw evidence镜像hash失败: {mirror}")

    return {
        "machine": machine,
        "root": root,
        "detail": detail,
        "metadata": metadata,
        "verification": verification,
        "raw_evidence_files": len(evidence),
        "artifact_hashes": {
            name: sha256_file(root / name)
            for name in required
        },
    }


def cross_machine_deltas(detail: list[dict[str, str]]) -> list[dict[str, Any]]:
    indexed = {
        (row["machine"], row["model"], int(row["seed"]), row["checkpoint_role"], row["slice"]): row
        for row in detail
    }
    rows: list[dict[str, Any]] = []
    for role in ROLES:
        for model in MODELS:
            for seed in SEEDS:
                for slc in SLICES:
                    old = indexed[("2080ti", model, seed, role, slc)]
                    new = indexed[("3090", model, seed, role, slc)]
                    old_accuracy = float(old["accuracy"])
                    new_accuracy = float(new["accuracy"])
                    rows.append({
                        "model": model,
                        "seed": seed,
                        "checkpoint_role": role,
                        "slice": slc,
                        "accuracy_2080ti": old_accuracy,
                        "accuracy_3090": new_accuracy,
                        "accuracy_delta_3090_minus_2080ti": new_accuracy - old_accuracy,
                        "dataset_hash_2080ti": old["dataset_hash"],
                        "dataset_hash_3090": new["dataset_hash"],
                        "dataset_hash_match": old["dataset_hash"] == new["dataset_hash"],
                    })
    return rows


def collect(*, append_ledger: bool) -> dict[str, Any]:
    machines = [validate_machine(machine) for machine in MACHINES]
    detail = [row for item in machines for row in item["detail"]]
    detail_keys = {
        (row["machine"], row["model"], row["seed"], row["checkpoint_role"], row["slice"])
        for row in detail
    }
    if len(detail) != 120 or len(detail_keys) != 120:
        raise RuntimeError(f"combined detail应为120条唯一结果, 实际rows={len(detail)}, keys={len(detail_keys)}")

    combined_files = {
        "training-final.csv": 12,
        "longer-mqar-summary.csv": 40,
        "paired-deltas.csv": 20,
        "checkpoint-role-comparison.csv": 60,
        "source-manifest.csv": 24,
        "batch-sizes.csv": 20,
        "repro-verification.csv": None,
        "raw-evidence-manifest.csv": None,
    }
    combined_rows: dict[str, list[dict[str, str]]] = {}
    for filename, expected in combined_files.items():
        rows = [
            row
            for machine in MACHINES
            for row in read_csv(ARTIFACT_ROOT / "machines" / machine / filename)
        ]
        if expected is not None and len(rows) != expected:
            raise RuntimeError(f"{filename}合并行数应为{expected}, 实际{len(rows)}")
        combined_rows[filename] = rows

    hashes_by_slice: dict[str, set[str]] = {slc: set() for slc in SLICES}
    for row in detail:
        hashes_by_slice[row["slice"]].add(row["dataset_hash"])
    if any(len(values) != 1 for values in hashes_by_slice.values()):
        raise RuntimeError(f"跨机器dataset hash不一致: {hashes_by_slice}")

    deltas = cross_machine_deltas(detail)
    if len(deltas) != 60 or not all(row["dataset_hash_match"] for row in deltas):
        raise RuntimeError("跨机器delta矩阵或dataset hash校验失败.")

    write_csv(COMBINED_DIR / "longer-mqar-detail.csv", detail)
    for filename, rows in combined_rows.items():
        write_csv(COMBINED_DIR / filename, rows)
    write_csv(COMBINED_DIR / "cross-machine-deltas.csv", deltas)

    ledger_result: dict[str, Any] = {"requested": append_ledger, "flash_added": 0, "gdn_added": 0}
    if append_ledger:
        flash_candidates = [
            row
            for machine in MACHINES
            for row in read_csv(ARTIFACT_ROOT / "machines" / machine / "flash-ledger-rows.csv")
        ]
        gdn_candidates = [
            row
            for machine in MACHINES
            for row in read_csv(ARTIFACT_ROOT / "machines" / machine / "gdn-ledger-rows.csv")
        ]
        flash_added, flash_updated = append_unique_csv(
            REPO_ROOT / "docs/artifacts/gd-residual-v1/rank-seed-effect-summary.csv",
            flash_candidates,
            "run_id",
        )
        gdn_added, gdn_updated = append_unique_csv(
            REPO_ROOT / "docs/artifacts/gdn-expanded-k/gdn-expanded-k-summary.csv",
            gdn_candidates,
            "run_id",
        )
        ledger_result.update({
            "flash_added": flash_added,
            "flash_updated": flash_updated,
            "gdn_added": gdn_added,
            "gdn_updated": gdn_updated,
        })

    verification = {
        "experiment_id": EXPERIMENT_ID,
        "status": "passed",
        "machines": list(MACHINES),
        "machine_verification": {
            item["machine"]: {
                "status": item["verification"]["status"],
                "raw_evidence_files": item["raw_evidence_files"],
                "artifact_hashes": item["artifact_hashes"],
            }
            for item in machines
        },
        "training_rows": len(combined_rows["training-final.csv"]),
        "logical_formal_rows": len(detail),
        "unique_logical_keys": len(detail_keys),
        "summary_rows": len(combined_rows["longer-mqar-summary.csv"]),
        "source_manifest_rows": len(combined_rows["source-manifest.csv"]),
        "cross_machine_delta_rows": len(deltas),
        "dataset_hashes": {slc: next(iter(values)) for slc, values in hashes_by_slice.items()},
        "dataset_hashes_match_across_machines": True,
        "ledger": ledger_result,
        "recorded_at_utc": utc_now(),
    }
    write_json(COMBINED_DIR / "verification.json", verification)
    write_json(COMBINED_DIR / "metadata.json", {
        "experiment_id": EXPERIMENT_ID,
        "status": "completed",
        "comparison_scope": "independent_retraining_by_machine_same_protocol",
        "seed_aggregation": "n=3 within each machine; never pooled as n=6",
        "machines": {item["machine"]: item["metadata"] for item in machines},
        "generated_at_utc": utc_now(),
    })
    (COMBINED_DIR / "README.md").write_text(
        "# 跨GPU合并结果\n\n"
        "本目录合并2080 Ti和3090的机器级正式artifact. 相同seed是跨GPU独立重训配对, "
        "统计始终在每张GPU内按3个seed计算, 不合并为n=6.\n\n"
        "- `longer-mqar-detail.csv`: 120条机器×模型×seed×checkpoint role×slice逻辑结果.\n"
        "- `longer-mqar-summary.csv`: 每张机器内的三seed mean和population std.\n"
        "- `paired-deltas.csv`: 每张机器内Flash-GDN同seed差值.\n"
        "- `cross-machine-deltas.csv`: 同模型、seed、role、slice的3090-2080Ti差值.\n"
        "- `verification.json`: 行数、唯一键、dataset hash和机器artifact hash审计.\n",
        encoding="utf-8",
    )
    return verification


def main() -> int:
    parser = argparse.ArgumentParser(description="生成当前基线Longer-MQAR跨机器正式artifact.")
    parser.add_argument("--append-ledger", action="store_true")
    args = parser.parse_args()
    verification = collect(append_ledger=args.append_ledger)
    print(json.dumps(verification, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
