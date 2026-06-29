from __future__ import annotations

import argparse
import csv
import hashlib
import socket
from pathlib import Path


EXPERIMENT_ID = "20260629-04-flash-vqg-eval-read-topk-sweep"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _infer_repeat(run_dir: Path) -> str:
    name = run_dir.name
    if "-r1-" in name or name.endswith("-r1"):
        return "r1"
    if "-r2-" in name or name.endswith("-r2"):
        return "r2"
    return ""


def _row_for_run_dir(run_dir: Path, source_machine: str, source_host: str) -> dict[str, str | int]:
    best = run_dir / "best.pt"
    last = run_dir / "last.pt"
    config = run_dir / "train_config.json"
    missing = [str(p) for p in (best, last, config) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing checkpoint files for {run_dir}: {missing}")
    repeat = _infer_repeat(run_dir)
    checkpoint_id = f"{source_machine}-{repeat}" if repeat else f"{source_machine}-{run_dir.name}"
    return {
        "experiment_id": EXPERIMENT_ID,
        "checkpoint_id": checkpoint_id,
        "source_machine": source_machine,
        "source_host": source_host,
        "run_repeat": repeat,
        "run_dir": str(run_dir.resolve()),
        "best_checkpoint": str(best.resolve()),
        "last_checkpoint": str(last.resolve()),
        "train_config": str(config.resolve()),
        "best_bytes": best.stat().st_size,
        "last_bytes": last.stat().st_size,
        "best_sha256": _sha256_file(best),
        "last_sha256": _sha256_file(last),
    }


def run(args: argparse.Namespace) -> int:
    output = Path(args.output)
    rows = [
        _row_for_run_dir(Path(raw), args.source_machine, args.source_host or socket.gethostname())
        for raw in args.run_dirs
    ]
    rows.sort(key=lambda r: str(r["checkpoint_id"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment_id",
        "checkpoint_id",
        "source_machine",
        "source_host",
        "run_repeat",
        "run_dir",
        "best_checkpoint",
        "last_checkpoint",
        "train_config",
        "best_bytes",
        "last_bytes",
        "best_sha256",
        "last_sha256",
    ]
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(output)
    for row in rows:
        print(row["checkpoint_id"], row["run_dir"])
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build checkpoint manifest for eval read-topk sweep.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-machine", required=True)
    parser.add_argument("--source-host", default="")
    parser.add_argument("run_dirs", nargs="+")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
