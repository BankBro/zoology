from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from experiment_lib import ARTIFACT_DIR, REPO_ROOT, RESULTS_ROOT, ensure_artifact_dirs, write_csv, write_json


KEY_METRICS = [
    "valid/loss",
    "valid/accuracy",
    "valid/mqar_case/accuracy-1024x256",
    "valid/mqar_case/accuracy-512x128",
    "valid/mqar_case/accuracy-256x64",
    "valid/attn/gd_residual_write_strength_mean",
    "valid/attn/gd_residual_raw_topk_mass_mean",
    "valid/attn/gd_residual_write_q_entropy_mean",
    "valid/vq/relative_err_mean",
    "valid/vq/write_entropy_mean",
    "valid/vq/write_top1_mass_mean",
]


def _read_matrix_rows(path: Path) -> list[dict[str, Any]]:
    import csv

    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _load_checkpoint_metrics(launch_id: str, run_id: str) -> dict[str, Any]:
    ckpt_dir = REPO_ROOT / "checkpoints" / launch_id / run_id
    payload_path = ckpt_dir / "best.pt"
    if not payload_path.exists():
        payload_path = ckpt_dir / "last.pt"
    if not payload_path.exists():
        return {"checkpoint_status": "missing", "checkpoint_path": None}
    payload = torch.load(payload_path, map_location="cpu")
    metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
    row: dict[str, Any] = {
        "checkpoint_status": "loaded",
        "checkpoint_path": str(payload_path.resolve()),
        "checkpoint_epoch": payload.get("epoch") if isinstance(payload, dict) else None,
    }
    for key in KEY_METRICS:
        row[key] = metrics.get(key)
    return row


def collect(launch_id: str, *, mode: str, matrix: str) -> tuple[Path, list[dict[str, Any]]]:
    ensure_artifact_dirs()
    matrix_path = ARTIFACT_DIR / f"{mode}-{matrix}-matrix.csv"
    rows = _read_matrix_rows(matrix_path)
    final_rows: list[dict[str, Any]] = []
    for row in rows:
        run_id = str(row["run_id"])
        merged = dict(row)
        merged.update(_load_checkpoint_metrics(launch_id, run_id))
        final_rows.append(merged)
    output_path = ARTIFACT_DIR / f"{mode}-{matrix}-final.csv"
    write_csv(output_path, final_rows)
    return output_path, final_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总 init-transplant checkpoint final metrics.")
    parser.add_argument("--launch-id", required=True)
    parser.add_argument("--mode", choices=["early", "train"], required=True)
    parser.add_argument("--matrix", choices=["core", "extended"], default="core")
    args = parser.parse_args()

    output_path, rows = collect(args.launch_id, mode=args.mode, matrix=args.matrix)
    write_json(
        ARTIFACT_DIR / f"{args.mode}-{args.matrix}-collect-status.json",
        {
            "status": "collected",
            "launch_id": args.launch_id,
            "output_path": str(output_path.resolve()),
            "num_rows": len(rows),
            "analysis_dir": str((RESULTS_ROOT / args.launch_id).resolve()),
        },
    )
    print(json.dumps({"output_path": str(output_path.resolve()), "num_rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
