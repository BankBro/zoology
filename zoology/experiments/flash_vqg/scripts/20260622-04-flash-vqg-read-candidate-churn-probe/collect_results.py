#!/usr/bin/env python3
"""Collect lightweight launch metadata after churn probe runs finish."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def collect(*, generated_root: Path, output_dir: Path, launch_prefix: str) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for manifest_path in sorted(generated_root.glob(f"{launch_prefix}*/manifest.json")):
        manifest = _load_json(manifest_path)
        for run in manifest.get("runs", []):
            summary = run.get("config_summary") or {}
            rows.append(
                {
                    "launch_id": manifest.get("launch_id"),
                    "run_id": run.get("run_id"),
                    "status": run.get("status"),
                    "manifest_path": str(manifest_path),
                    "fox_remote_read_topk": summary.get("fox_remote_read_topk"),
                    "read_churn_probe_enabled": summary.get("read_churn_probe_enabled"),
                    "read_churn_probe_valid_batches": summary.get("read_churn_probe_valid_batches"),
                    "best_checkpoint": (run.get("local") or {}).get("best_checkpoint"),
                    "last_checkpoint": (run.get("local") or {}).get("last_checkpoint"),
                }
            )
    csv_path = output_dir / "launch_index.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "launch_id",
                "run_id",
                "status",
                "manifest_path",
                "fox_remote_read_topk",
                "read_churn_probe_enabled",
                "read_churn_probe_valid_batches",
                "best_checkpoint",
                "last_checkpoint",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "num_runs": len(rows),
        "generated_root": str(generated_root),
        "output_dir": str(output_dir),
        "csv_path": str(csv_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generated-root",
        type=Path,
        default=Path("/home/lyj/mnt/project/zoology/zoology/experiments/flash_vqg/generated"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--launch-prefix", default="flash-vqg-20260622-04-churn")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(
        json.dumps(
            collect(
                generated_root=args.generated_root,
                output_dir=args.output_dir,
                launch_prefix=args.launch_prefix,
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

