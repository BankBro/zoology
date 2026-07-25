#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPERIMENT_ID = "20260725-01-current-baselines-longer-mqar"
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get("ZOOLOGY_REPO_ROOT", "/home/lyj/mnt/project/zoology")).resolve()
MACHINE = os.environ.get("LONGER_MQAR_MACHINE", "2080ti").strip().lower()
if MACHINE not in {"2080ti", "3090"}:
    raise RuntimeError(f"LONGER_MQAR_MACHINE必须为2080ti或3090, 实际为{MACHINE!r}.")
OUTPUT_ROOT = SCRIPT_DIR / "outputs" if MACHINE == "2080ti" else SCRIPT_DIR / "outputs/machines" / MACHINE
GATE_DIR = OUTPUT_ROOT / "gates"
LEGACY_RUNNER = (
    REPO_ROOT
    / "zoology/experiments/flash_vqg/scripts/"
    / "20260521-longer-mqar-canonical/longer_mqar_eval_runner.py"
)
SLICES = ((1024, 256), (2048, 512), (4096, 1024), (8190, 512), (8190, 2047))
EXPECTED_DATASET_HASHES = {
    "1024x256": "f30f1e09e1300deb2e0f430d7c50c47a77f58f43657aa4d6f425f938a91a9acb",
    "2048x512": "e446efc9f4dc774377c56c403a468c747870beb18ff0320f15df794dcb69f015",
    "4096x1024": "0981292ce8ebea1402a4c3a7a6655dfc1de42688c329bd5ea921541a0f1d16ed",
    "8190x512": "37a8533a985fabd3db03daf9dfd3a8e0936513ab724039c63b9a22bf352d002d",
    "8190x2047": "8c16a91ea127bb7bf2130238ecd56b49fb1fd73bb3e0c2f4c586ddba8e9b97a9",
}
BATCH_CANDIDATES = (32, 16, 8, 4, 2, 1)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def slice_name(seq: int, kv: int) -> str:
    return f"{seq}x{kv}"


def load_manifest(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    logical = json.loads(path.read_text(encoding="utf-8"))
    if len(logical) != 12:
        raise RuntimeError(f"source manifest逻辑角色应为12, 实际{len(logical)}")
    seen_ids = set()
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in logical:
        if str(row.get("machine", "2080ti")) != MACHINE:
            raise RuntimeError(f"source manifest机器不匹配: expected={MACHINE}, row={row.get('machine')}")
        source_id = str(row["source_id"])
        if source_id in seen_ids:
            raise RuntimeError(f"重复source_id: {source_id}")
        seen_ids.add(source_id)
        checkpoint = Path(row["checkpoint_path"])
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        if sha256_file(checkpoint) != row["checkpoint_file_sha256"]:
            raise RuntimeError(f"checkpoint文件hash漂移: {checkpoint}")
        key = (str(row["model"]), str(row["checkpoint_model_state_sha256"]))
        if key not in grouped:
            grouped[key] = {
                **row,
                "unique_source_id": f"{row['model']}-{str(row['checkpoint_model_state_sha256'])[:16]}",
                "aliases": [],
                "checkpoint_roles": [],
            }
        grouped[key]["aliases"].append(source_id)
        grouped[key]["checkpoint_roles"].append(str(row["checkpoint_role"]))
        if row["checkpoint_role"] == "last":
            grouped[key]["checkpoint_path"] = row["checkpoint_path"]
            grouped[key]["checkpoint_file_sha256"] = row["checkpoint_file_sha256"]
    unique = sorted(grouped.values(), key=lambda row: (row["model"], int(row["seed"]), row["unique_source_id"]))
    if not unique or len(unique) > 12:
        raise RuntimeError(f"唯一checkpoint数量异常: {len(unique)}")
    return logical, unique


class EvalQueue:
    def __init__(self, *, manifest: Path, output_dir: Path, mode: str, resume: bool):
        self.manifest = manifest
        self.output_dir = output_dir
        self.mode = mode
        self.resume = resume
        self.logical_sources, self.sources = load_manifest(manifest)
        self.events_dir = output_dir / "events"
        self.results_dir = output_dir / "results"
        self.logs_dir = output_dir / "logs"
        self.records_dir = output_dir / "records"
        for path in (self.events_dir, self.results_dir, self.logs_dir, self.records_dir):
            path.mkdir(parents=True, exist_ok=True)
        self.status_path = output_dir / "status.json"
        self.batch_sizes: dict[tuple[str, str], int] = {}

    def update_status(self, phase: str, current: str = "", **extra: Any) -> None:
        write_json(self.status_path, {
            "experiment_id": EXPERIMENT_ID,
            "machine": MACHINE,
            "mode": self.mode,
            "status": "running",
            "phase": phase,
            "current": current,
            "logical_sources": len(self.logical_sources),
            "unique_sources": len(self.sources),
            "updated_at_utc": utc_now(),
            **extra,
        })

    def event_paths(self, uid: str) -> tuple[Path, Path, Path, Path]:
        name = safe_name(uid)
        return (
            self.events_dir / f"{name}.json",
            self.results_dir / f"{name}.json",
            self.logs_dir / f"{name}.log",
            self.records_dir / f"{name}.json",
        )

    def run_event(
        self,
        *,
        source: dict[str, Any],
        event_mode: str,
        seq: int,
        kv: int,
        num_examples: int,
        batch_size: int,
    ) -> dict[str, Any]:
        uid = f"{event_mode}-{source['unique_source_id']}-{seq}x{kv}-n{num_examples}-b{batch_size}"
        event_path, result_path, log_path, record_path = self.event_paths(uid)
        if self.resume and record_path.exists():
            record = json.loads(record_path.read_text(encoding="utf-8"))
            reusable = all((
                record.get("status") == "completed",
                record.get("machine", "2080ti") == MACHINE,
                record.get("eval_mode") == event_mode,
                record.get("checkpoint_model_state_sha256") == source["checkpoint_model_state_sha256"],
                int(record.get("input_seq_len", -1)) == seq,
                int(record.get("num_kv_pairs", -1)) == kv,
                int(record.get("num_examples", -1)) == num_examples,
                int(record.get("eval_batch_size", -1)) == batch_size,
                Path(record.get("result_path") or "").exists(),
                Path(record.get("log_path") or "").exists(),
            ))
            if reusable:
                return record
        event = {
            "event_uid": uid,
            "checkpoint_path": source["checkpoint_path"],
            "input_seq_len": seq,
            "num_kv_pairs": kv,
            "num_examples": num_examples,
            "eval_batch_size": batch_size,
            "eval_seed": 123,
        }
        write_json(event_path, event)
        started_at = utc_now()
        started = time.perf_counter()
        env = os.environ.copy()
        env["TRITON_F32_DEFAULT"] = "ieee"
        env["GDN_KERNEL_DTYPE"] = "float32"
        env["NVIDIA_TF32_OVERRIDE"] = "0"
        cmd = [sys.executable, str(LEGACY_RUNNER), "--single-event", str(event_path), "--result", str(result_path)]
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        ended_at = utc_now()
        elapsed = time.perf_counter() - started
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
        else:
            result = {"status": "failed", "failure_type": "missing_result", "failure_detail": f"returncode={proc.returncode}"}
        record = {
            "event_id": uid,
            "machine": MACHINE,
            "eval_mode": event_mode,
            "status": result.get("status", "failed"),
            "model": source["model"],
            "seed": source["seed"],
            "unique_source_id": source["unique_source_id"],
            "source_aliases": ";".join(sorted(source["aliases"])),
            "checkpoint_roles": ";".join(sorted(set(source["checkpoint_roles"]))),
            "checkpoint_path": source["checkpoint_path"],
            "checkpoint_file_sha256": source["checkpoint_file_sha256"],
            "checkpoint_model_state_sha256": source["checkpoint_model_state_sha256"],
            "input_seq_len": seq,
            "num_kv_pairs": kv,
            "slice": slice_name(seq, kv),
            "num_examples": num_examples,
            "eval_batch_size": batch_size,
            "eval_seed": 123,
            "accuracy": result.get("accuracy", ""),
            "loss": result.get("loss", ""),
            "aggregate_accuracy": result.get("aggregate_accuracy", ""),
            "aggregate_loss": result.get("aggregate_loss", ""),
            "dataset_hash": result.get("dataset_hash", ""),
            "dataset_hash_algorithm": result.get("dataset_hash_algorithm", ""),
            "dataset_input_shape": result.get("dataset_input_shape", ""),
            "dataset_label_shape": result.get("dataset_label_shape", ""),
            "wall_clock_sec": result.get("wall_clock_sec", elapsed),
            "peak_memory_mb": result.get("peak_memory_mb", ""),
            "failure_type": result.get("failure_type", ""),
            "failure_detail": result.get("failure_detail", ""),
            "returncode": proc.returncode,
            "started_at_utc": started_at,
            "ended_at_utc": ended_at,
            "event_path": str(event_path),
            "result_path": str(result_path),
            "log_path": str(log_path),
        }
        write_json(record_path, record)
        return record

    def all_records(self) -> list[dict[str, Any]]:
        return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(self.records_dir.glob("*.json"))]

    def save_detail(self) -> None:
        write_csv(self.output_dir / "detail.csv", self.all_records())

    def representatives(self) -> dict[str, dict[str, Any]]:
        reps: dict[str, dict[str, Any]] = {}
        for source in self.sources:
            reps.setdefault(str(source["model"]), source)
        if set(reps) != {"flash", "gdn"}:
            raise RuntimeError(f"缺少模型代表source: {sorted(reps)}")
        return reps

    def batch_search(self) -> None:
        self.update_status("batch_search")
        reps = self.representatives()
        search_examples = 2 if self.mode == "smoke-dag" else 32
        candidates = tuple(value for value in BATCH_CANDIDATES if value <= search_examples)
        rows: list[dict[str, Any]] = []
        for model in ("flash", "gdn"):
            for seq, kv in SLICES:
                slc = slice_name(seq, kv)
                selected = None
                failures: list[str] = []
                for batch in candidates:
                    self.update_status("batch_search", f"{model}:{slc}:b{batch}")
                    record = self.run_event(
                        source=reps[model], event_mode="batch-search-candidate", seq=seq, kv=kv,
                        num_examples=search_examples, batch_size=batch,
                    )
                    if record["status"] == "completed":
                        selected = batch
                        break
                    if record["status"] == "oom" or str(record.get("failure_type", "")).lower() == "oom":
                        failures.append(f"b{batch}:oom")
                        continue
                    raise RuntimeError(f"batch-search非OOM失败: {record}")
                if selected is None:
                    raise RuntimeError(f"没有可用batch: {model}:{slc}, {failures}")
                self.batch_sizes[(model, slc)] = selected
                rows.append({"model": model, "slice": slc, "selected_batch_size": selected, "failures": ";".join(failures)})
        write_json(self.output_dir / "batch-sizes.json", rows)
        write_csv(self.output_dir / "batch-sizes.csv", rows)
        self.save_detail()

    def source_smoke(self) -> None:
        total = len(self.sources) * len(SLICES)
        completed = 0
        for source in self.sources:
            for seq, kv in SLICES:
                slc = slice_name(seq, kv)
                batch = self.batch_sizes[(source["model"], slc)]
                self.update_status("source_smoke", f"{source['unique_source_id']}:{slc}", completed=completed, expected=total)
                record = self.run_event(
                    source=source, event_mode="source-smoke", seq=seq, kv=kv,
                    num_examples=batch, batch_size=batch,
                )
                if record["status"] != "completed":
                    raise RuntimeError(f"source smoke失败: {record}")
                completed += 1
                self.save_detail()
        if self.mode == "formal":
            write_json(GATE_DIR / "EVAL_SMOKE_PASSED.json", {
                "status": "passed", "machine": MACHINE, "completed": completed, "expected": total,
                "unique_sources": len(self.sources), "recorded_at_utc": utc_now(),
            })

    def formal_events(self) -> dict[tuple[str, str], dict[str, Any]]:
        event_mode = "formal-probe" if self.mode == "smoke-dag" else "formal"
        examples = 2 if self.mode == "smoke-dag" else 500
        total = len(self.sources) * len(SLICES)
        completed = 0
        references: dict[tuple[str, str], dict[str, Any]] = {}
        dataset_hashes: dict[str, set[str]] = {slice_name(*slc): set() for slc in SLICES}
        for source in self.sources:
            for seq, kv in SLICES:
                slc = slice_name(seq, kv)
                batch = min(examples, self.batch_sizes[(source["model"], slc)])
                self.update_status(event_mode, f"{source['unique_source_id']}:{slc}", completed=completed, expected=total)
                record = self.run_event(
                    source=source, event_mode=event_mode, seq=seq, kv=kv,
                    num_examples=examples, batch_size=batch,
                )
                if record["status"] != "completed":
                    raise RuntimeError(f"formal event失败: {record}")
                if self.mode == "formal" and record["dataset_hash"] != EXPECTED_DATASET_HASHES[slc]:
                    raise RuntimeError(f"dataset hash不匹配: {slc} {record['dataset_hash']}")
                dataset_hashes[slc].add(str(record["dataset_hash"]))
                references[(source["unique_source_id"], slc)] = record
                completed += 1
                self.save_detail()
        if any(len(values) != 1 for values in dataset_hashes.values()):
            raise RuntimeError(f"同slice dataset hash不一致: {dataset_hashes}")
        return references

    def repro(self, references: dict[tuple[str, str], dict[str, Any]]) -> None:
        examples = 2 if self.mode == "smoke-dag" else 500
        seq, kv = SLICES[0]
        slc = slice_name(seq, kv)
        rows: list[dict[str, Any]] = []
        for source in self.sources:
            batch = min(examples, self.batch_sizes[(source["model"], slc)])
            self.update_status("repro", source["unique_source_id"])
            record = self.run_event(
                source=source, event_mode="repro", seq=seq, kv=kv,
                num_examples=examples, batch_size=batch,
            )
            reference = references[(source["unique_source_id"], slc)]
            try:
                delta = abs(float(record["accuracy"]) - float(reference["accuracy"]))
            except Exception:
                delta = float("inf")
            hash_match = record.get("dataset_hash") == reference.get("dataset_hash")
            passed = record.get("status") == "completed" and hash_match and delta <= 1e-12
            rows.append({
                "unique_source_id": source["unique_source_id"],
                "status": record.get("status"),
                "dataset_hash_match": hash_match,
                "accuracy_delta_abs": delta,
                "passed": passed,
            })
            if not passed:
                raise RuntimeError(f"repro失败: {rows[-1]}")
            self.save_detail()
        write_json(self.output_dir / "repro-verification.json", rows)
        write_csv(self.output_dir / "repro-verification.csv", rows)

    def run(self) -> dict[str, Any]:
        self.update_status("start")
        self.batch_search()
        self.source_smoke()
        references = self.formal_events()
        self.repro(references)
        records = self.all_records()
        physical_formal_mode = "formal-probe" if self.mode == "smoke-dag" else "formal"
        formal_rows = [row for row in records if row.get("eval_mode") == physical_formal_mode]
        verification = {
            "experiment_id": EXPERIMENT_ID,
            "machine": MACHINE,
            "mode": self.mode,
            "status": "completed",
            "logical_sources": len(self.logical_sources),
            "unique_sources": len(self.sources),
            "formal_rows": len(formal_rows),
            "formal_expected": len(self.sources) * len(SLICES),
            "all_formal_completed": all(row.get("status") == "completed" for row in formal_rows),
            "recorded_at_utc": utc_now(),
        }
        if verification["formal_rows"] != verification["formal_expected"] or not verification["all_formal_completed"]:
            raise RuntimeError(f"formal完成数异常: {verification}")
        write_json(self.output_dir / "verification.json", verification)
        write_json(self.status_path, {**verification, "phase": "done", "updated_at_utc": utc_now()})
        self.save_detail()
        return verification


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="当前基线manifest-driven Longer-MQAR runner.")
    root.add_argument("--manifest", type=Path, required=True)
    root.add_argument("--output-dir", type=Path, required=True)
    root.add_argument("--mode", choices=("smoke-dag", "formal"), required=True)
    root.add_argument("--resume", action="store_true")
    return root


def main() -> int:
    args = parser().parse_args()
    queue = EvalQueue(manifest=args.manifest, output_dir=args.output_dir, mode=args.mode, resume=args.resume)
    verification = queue.run()
    print(json.dumps(verification, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
