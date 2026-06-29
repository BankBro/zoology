from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_THIS_FILE = Path(__file__).resolve()
for _parent in _THIS_FILE.parents:
    if (_parent / "zoology").is_dir() and (_parent / "docs").is_dir():
        sys.path.insert(0, str(_parent))
        break


EXPERIMENT_ID = "20260629-04-flash-vqg-eval-read-topk-sweep"
METRIC_COLUMNS = [
    "valid_loss",
    "valid_accuracy",
    "valid_mqar_case_accuracy_1024x256",
    "valid_input_seq_len_accuracy_1024",
    "valid_num_kv_pairs_accuracy_256",
    "valid_effective_read_topk",
    "valid_read_selected_mass_mean",
    "valid_read_selected_mass_p05",
]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _dedupe_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keyed: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    unkeyed: list[dict[str, Any]] = []
    for row in rows:
        try:
            key = (
                str(row["checkpoint_id"]),
                str(row["checkpoint_kind"]),
                int(row["eval_read_topk"]),
                str(row["eval_machine"]),
            )
        except (KeyError, TypeError, ValueError):
            unkeyed.append(row)
            continue
        keyed[key] = row
    return unkeyed + [keyed[key] for key in sorted(keyed)]


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _delta(a: Any, b: Any) -> float | None:
    va = _float_or_none(a)
    vb = _float_or_none(b)
    if va is None or vb is None:
        return None
    return va - vb


def _format(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        flat = {
            "experiment_id": row.get("experiment_id"),
            "checkpoint_id": row.get("checkpoint_id"),
            "checkpoint_source_machine": row.get("checkpoint_source_machine"),
            "checkpoint_kind": row.get("checkpoint_kind"),
            "eval_machine": row.get("eval_machine"),
            "eval_read_topk": row.get("eval_read_topk"),
            "status": row.get("status"),
            "duration_seconds": row.get("duration_seconds"),
            "peak_memory_bytes": row.get("peak_memory_bytes"),
            "checkpoint_epoch": row.get("checkpoint_epoch"),
            "checkpoint_saved_valid_accuracy": row.get("checkpoint_saved_valid_accuracy"),
            "checkpoint_saved_1024x256": row.get("checkpoint_saved_1024x256"),
            "checkpoint_sha256": row.get("checkpoint_sha256"),
            "checkpoint_path": row.get("checkpoint_path"),
            "train_config_path": row.get("train_config_path"),
            "error_type": row.get("error_type"),
            "error": row.get("error"),
        }
        for col in METRIC_COLUMNS:
            flat[col] = row.get(col)
        flat["valid_attn_gd_residual_m_norm_mean"] = metrics.get(
            "valid/attn/gd_residual_m_norm_mean"
        )
        flat["valid_attn_gd_residual_m_norm_max"] = metrics.get(
            "valid/attn/gd_residual_m_norm_max"
        )
        out.append(flat)
    out.sort(
        key=lambda r: (
            str(r.get("checkpoint_source_machine")),
            str(r.get("checkpoint_id")),
            str(r.get("checkpoint_kind")),
            str(r.get("eval_machine")),
            int(r.get("eval_read_topk") or 0),
        )
    )
    return out


def _cross_machine_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    completed = [r for r in rows if r.get("status") == "completed"]
    by_key: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in completed:
        key = (
            str(row.get("checkpoint_id")),
            str(row.get("checkpoint_kind")),
            int(row.get("eval_read_topk")),
        )
        by_key[key][str(row.get("eval_machine"))] = row

    out = []
    for (checkpoint_id, checkpoint_kind, topk), machine_rows in sorted(by_key.items()):
        if "2080ti" not in machine_rows or "3090" not in machine_rows:
            continue
        a = machine_rows["2080ti"]
        b = machine_rows["3090"]
        out.append(
            {
                "checkpoint_id": checkpoint_id,
                "checkpoint_source_machine": a.get("checkpoint_source_machine"),
                "checkpoint_kind": checkpoint_kind,
                "eval_read_topk": topk,
                "valid_accuracy_2080ti_eval": a.get("valid_accuracy"),
                "valid_accuracy_3090_eval": b.get("valid_accuracy"),
                "valid_accuracy_delta_3090_minus_2080ti": _delta(
                    b.get("valid_accuracy"), a.get("valid_accuracy")
                ),
                "hard_1024x256_2080ti_eval": a.get("valid_mqar_case_accuracy_1024x256"),
                "hard_1024x256_3090_eval": b.get("valid_mqar_case_accuracy_1024x256"),
                "hard_1024x256_delta_3090_minus_2080ti": _delta(
                    b.get("valid_mqar_case_accuracy_1024x256"),
                    a.get("valid_mqar_case_accuracy_1024x256"),
                ),
                "loss_2080ti_eval": a.get("valid_loss"),
                "loss_3090_eval": b.get("valid_loss"),
                "loss_delta_3090_minus_2080ti": _delta(
                    b.get("valid_loss"), a.get("valid_loss")
                ),
            }
        )
    return out


def _topk_reference_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    completed = [r for r in rows if r.get("status") == "completed"]
    by_ref: dict[tuple[str, str, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in completed:
        key = (
            str(row.get("checkpoint_id")),
            str(row.get("checkpoint_kind")),
            str(row.get("eval_machine")),
        )
        by_ref[key][int(row.get("eval_read_topk"))] = row

    out = []
    for (checkpoint_id, checkpoint_kind, eval_machine), topk_rows in sorted(by_ref.items()):
        ref = topk_rows.get(64)
        if ref is None:
            continue
        for topk, row in sorted(topk_rows.items()):
            out.append(
                {
                    "checkpoint_id": checkpoint_id,
                    "checkpoint_source_machine": row.get("checkpoint_source_machine"),
                    "checkpoint_kind": checkpoint_kind,
                    "eval_machine": eval_machine,
                    "eval_read_topk": topk,
                    "valid_accuracy": row.get("valid_accuracy"),
                    "valid_accuracy_delta_vs_topk64": _delta(
                        row.get("valid_accuracy"), ref.get("valid_accuracy")
                    ),
                    "hard_1024x256": row.get("valid_mqar_case_accuracy_1024x256"),
                    "hard_1024x256_delta_vs_topk64": _delta(
                        row.get("valid_mqar_case_accuracy_1024x256"),
                        ref.get("valid_mqar_case_accuracy_1024x256"),
                    ),
                    "loss": row.get("valid_loss"),
                    "loss_delta_vs_topk64": _delta(row.get("valid_loss"), ref.get("valid_loss")),
                    "selected_mass_mean": row.get("valid_read_selected_mass_mean"),
                    "selected_mass_p05": row.get("valid_read_selected_mass_p05"),
                }
            )
    return out


def _aggregate_by_topk(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    completed = [r for r in rows if r.get("status") == "completed"]
    buckets: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in completed:
        buckets[(str(row.get("eval_machine")), int(row.get("eval_read_topk")))].append(row)
    out = []
    for (eval_machine, topk), vals in sorted(buckets.items()):
        hard = [
            float(v["valid_mqar_case_accuracy_1024x256"])
            for v in vals
            if v.get("valid_mqar_case_accuracy_1024x256") is not None
        ]
        acc = [
            float(v["valid_accuracy"])
            for v in vals
            if v.get("valid_accuracy") is not None
        ]
        out.append(
            {
                "eval_machine": eval_machine,
                "eval_read_topk": topk,
                "n": len(vals),
                "valid_accuracy_mean": statistics.fmean(acc) if acc else None,
                "valid_accuracy_min": min(acc) if acc else None,
                "valid_accuracy_max": max(acc) if acc else None,
                "hard_1024x256_mean": statistics.fmean(hard) if hard else None,
                "hard_1024x256_min": min(hard) if hard else None,
                "hard_1024x256_max": max(hard) if hard else None,
            }
        )
    return out


def _write_readme(artifact_dir: Path, metadata: dict[str, Any]) -> None:
    text = f"""# {EXPERIMENT_ID}

本目录保存 Flash-VQG dense-read 4ep checkpoint 的 evaluation read-topk sweep 轻量结果.

本实验不重新训练, 不保存新 checkpoint. 每条记录只加载已有 checkpoint, 覆盖评估阶段 `fox_remote_read_topk`, 然后跑完整 validation.

## 文件

- `eval-summary.csv`: 每个 checkpoint/topk/eval machine 的评估结果.
- `topk-vs-64.csv`: 同一 checkpoint 同一 eval machine 下, 各 topk 相对 topk=64 的差值.
- `cross-machine-eval-comparison.csv`: 同一 checkpoint 同一 topk 在 2080ti 与 3090 eval 的差值.
- `aggregate-by-topk.csv`: 按 eval machine 和 topk 聚合的均值/范围.
- `checkpoint-manifest.csv`: 本轮 checkpoint 输入清单.
- `source-manifest.csv`: 原始 JSONL/status 文件的来源与 sha256.
- `metadata.json`: 运行元信息.

## 汇总

```json
{json.dumps(metadata, ensure_ascii=False, indent=2)}
```
"""
    (artifact_dir / "README.md").write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    record_paths = [Path(p) for p in args.records]
    rows: list[dict[str, Any]] = []
    for path in record_paths:
        rows.extend(_read_jsonl(path))
    raw_record_count = len(rows)
    rows = _dedupe_records(rows)

    summary = _summary_rows(rows)
    cross = _cross_machine_rows(rows)
    topk_ref = _topk_reference_rows(rows)
    aggregate = _aggregate_by_topk(rows)

    summary_fields = [
        "experiment_id",
        "checkpoint_id",
        "checkpoint_source_machine",
        "checkpoint_kind",
        "eval_machine",
        "eval_read_topk",
        "status",
        *METRIC_COLUMNS,
        "duration_seconds",
        "peak_memory_bytes",
        "checkpoint_epoch",
        "checkpoint_saved_valid_accuracy",
        "checkpoint_saved_1024x256",
        "valid_attn_gd_residual_m_norm_mean",
        "valid_attn_gd_residual_m_norm_max",
        "checkpoint_sha256",
        "checkpoint_path",
        "train_config_path",
        "error_type",
        "error",
    ]
    _write_csv(artifact_dir / "eval-summary.csv", summary, summary_fields)
    _write_csv(
        artifact_dir / "cross-machine-eval-comparison.csv",
        cross,
        [
            "checkpoint_id",
            "checkpoint_source_machine",
            "checkpoint_kind",
            "eval_read_topk",
            "valid_accuracy_2080ti_eval",
            "valid_accuracy_3090_eval",
            "valid_accuracy_delta_3090_minus_2080ti",
            "hard_1024x256_2080ti_eval",
            "hard_1024x256_3090_eval",
            "hard_1024x256_delta_3090_minus_2080ti",
            "loss_2080ti_eval",
            "loss_3090_eval",
            "loss_delta_3090_minus_2080ti",
        ],
    )
    _write_csv(
        artifact_dir / "topk-vs-64.csv",
        topk_ref,
        [
            "checkpoint_id",
            "checkpoint_source_machine",
            "checkpoint_kind",
            "eval_machine",
            "eval_read_topk",
            "valid_accuracy",
            "valid_accuracy_delta_vs_topk64",
            "hard_1024x256",
            "hard_1024x256_delta_vs_topk64",
            "loss",
            "loss_delta_vs_topk64",
            "selected_mass_mean",
            "selected_mass_p05",
        ],
    )
    _write_csv(
        artifact_dir / "aggregate-by-topk.csv",
        aggregate,
        [
            "eval_machine",
            "eval_read_topk",
            "n",
            "valid_accuracy_mean",
            "valid_accuracy_min",
            "valid_accuracy_max",
            "hard_1024x256_mean",
            "hard_1024x256_min",
            "hard_1024x256_max",
        ],
    )

    checkpoint_manifest = Path(args.checkpoint_manifest)
    if checkpoint_manifest.exists():
        (artifact_dir / "checkpoint-manifest.csv").write_text(
            checkpoint_manifest.read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    source_rows = []
    for path in record_paths:
        source_rows.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size if path.exists() else None,
                "sha256": _sha256_file(path) if path.exists() else None,
            }
        )
        status_path = path.with_name("status.json")
        if status_path.exists():
            source_rows.append(
                {
                    "path": str(status_path),
                    "bytes": status_path.stat().st_size,
                    "sha256": _sha256_file(status_path),
                }
            )
    _write_csv(artifact_dir / "source-manifest.csv", source_rows, ["path", "bytes", "sha256"])

    completed = [r for r in rows if r.get("status") == "completed"]
    failed = [r for r in rows if r.get("status") == "failed"]
    metadata = {
        "experiment_id": EXPERIMENT_ID,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "record_files": [str(p) for p in record_paths],
        "raw_records": raw_record_count,
        "total_records": len(rows),
        "completed_records": len(completed),
        "failed_records": len(failed),
        "expected_records": 112,
        "all_expected_completed": len(completed) == 112 and len(failed) == 0,
        "eval_machines": sorted({str(r.get("eval_machine")) for r in completed}),
        "topks": sorted({int(r.get("eval_read_topk")) for r in completed if r.get("eval_read_topk") is not None}),
    }
    (artifact_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_readme(artifact_dir, metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


def _sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect eval read-topk sweep records.")
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--checkpoint-manifest", required=True)
    parser.add_argument("records", nargs="+")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
