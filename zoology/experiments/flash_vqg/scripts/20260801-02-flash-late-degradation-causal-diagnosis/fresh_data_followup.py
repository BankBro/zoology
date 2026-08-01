#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

LOCAL_DIR = Path(__file__).resolve().parent
if str(LOCAL_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_DIR))

import analyze
import experiment
from causal_common import atomic_write_json, run_root, sha256_file, stable_json_sha256, utc_now


ARMS = ("ctrl-bridge", "factor-block")
SEED = 123
EXPOSURE = "fresh_per_epoch"
SEED_STRIDE = 1_000_003
TORCH_SEED_STRIDE = 10_000_019


def fresh_root() -> Path:
    return run_root() / "fresh-data"


def result_path(arm: str) -> Path:
    return fresh_root() / "training" / arm / "result.json"


def cache_path(cache_dir: Path, config: Any, seed: int) -> Path:
    payload = {**config.model_dump(), "_seed": int(seed)}
    digest = hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return cache_dir / f"data_{digest}.pt"


def fresh_seed(base_seed: int, epoch: int) -> int:
    if epoch == 0:
        return int(base_seed)
    modulus = 2**31
    return int((base_seed + epoch * SEED_STRIDE) % modulus)


def torch_seed(segment_idx: int, epoch: int) -> int:
    return int(SEED + epoch * TORCH_SEED_STRIDE + segment_idx * 1009)


def build_segment(config: Any, seed: int, cache_dir: Path, cpu_seed: int):
    from zoology.data.utils import DataSegment

    numpy_state = np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(cpu_seed)
            return DataSegment.from_config(
                config,
                cache_dir=str(cache_dir),
                force_cache=False,
                seed=seed,
            )
    finally:
        np.random.set_state(numpy_state)


class RefreshingSyntheticDataset:
    def __init__(self, base_dataset: Any, configs: list[Any], base_seeds: list[int], cache_dirs: dict[int, Path]):
        from zoology.data.utils import _SyntheticDataset

        self._dataset_type = _SyntheticDataset
        self.configs = configs
        self.base_seeds = base_seeds
        self.cache_dirs = cache_dirs
        self.batch_size = int(base_dataset.batch_size)
        self.current_epoch = 0
        self.epoch_manifests: dict[int, list[dict[str, Any]]] = {}
        self._install_segments(list(base_dataset.segments), epoch=0)

    def _install_segments(self, segments: list[Any], epoch: int) -> None:
        holder = self._dataset_type(segments=segments, batch_size=self.batch_size)
        self.segments = holder.segments
        self.batches = holder.batches
        self.segment_to_batch_indices = holder.segment_to_batch_indices
        self.current_epoch = int(epoch)
        self.epoch_manifests[epoch] = self._manifest(epoch)

    def _manifest(self, epoch: int) -> list[dict[str, Any]]:
        rows = []
        cache_dir = self.cache_dirs[epoch]
        for idx, config in enumerate(self.configs):
            seed = fresh_seed(self.base_seeds[idx], epoch)
            path = cache_path(cache_dir, config, seed)
            rows.append(
                {
                    "segment": idx,
                    "seed": seed,
                    "torch_seed": torch_seed(idx, epoch),
                    "cache_path": str(path.resolve()),
                    "cache_sha256": sha256_file(path),
                    "num_examples": int(config.num_examples),
                    "input_seq_len": int(config.input_seq_len),
                }
            )
        return rows

    def set_epoch(self, epoch: int) -> None:
        epoch = int(epoch)
        if epoch == self.current_epoch:
            return
        cache_dir = self.cache_dirs[epoch]
        segments = [
            build_segment(
                config,
                fresh_seed(self.base_seeds[idx], epoch),
                cache_dir,
                torch_seed(idx, epoch),
            )
            for idx, config in enumerate(self.configs)
        ]
        self._install_segments(segments, epoch)

    def __getitem__(self, batch_idx: int):
        segment_idx, batch_start = self.batches[batch_idx]
        segment = self.segments[segment_idx]
        slc = slice(batch_start, batch_start + self.batch_size)
        batch_len = len(segment.inputs[slc])
        slices = [segment.slices if segment.slices is not None else {}] * batch_len
        return segment.inputs[slc], segment.labels[slc], slices

    def __len__(self) -> int:
        return len(self.batches)


class RefreshingBatchOrderSampler:
    def __init__(self, dataset: RefreshingSyntheticDataset, mode: str, seed: int):
        from zoology.data.utils import _BatchOrderSampler

        self._sampler = _BatchOrderSampler(dataset, mode=mode, seed=seed, segment_order=None)
        self.dataset = dataset

    def set_epoch(self, epoch: int) -> None:
        self.dataset.set_epoch(epoch)
        self._sampler.set_epoch(epoch)

    def __iter__(self):
        return iter(self._sampler)

    def __len__(self) -> int:
        return len(self._sampler)


def base_train_seeds(data_config: Any) -> list[int]:
    state = np.random.get_state()
    try:
        np.random.seed(data_config.seed)
        values = np.random.randint(0, 2**31, size=len(data_config.train_configs))
        return [int(value) for value in values]
    finally:
        np.random.set_state(state)


def make_fresh_prepare_data(controller: dict[str, Any]):
    from zoology.data.utils import prepare_data as original_prepare_data

    def prepare_data(data_config: Any):
        base_train, test_loader = original_prepare_data(data_config)
        canonical_cache = Path(data_config.cache_dir).resolve()
        generated_cache = fresh_root() / "cache"
        generated_cache.mkdir(parents=True, exist_ok=True)
        cache_dirs = {0: canonical_cache, 1: generated_cache, 2: generated_cache, 3: generated_cache}
        dataset = RefreshingSyntheticDataset(
            base_dataset=base_train.dataset,
            configs=list(data_config.train_configs),
            base_seeds=base_train_seeds(data_config),
            cache_dirs=cache_dirs,
        )
        sampler = RefreshingBatchOrderSampler(dataset, data_config.train_batch_order, data_config.seed)
        controller["dataset"] = dataset
        return DataLoader(dataset, batch_size=None, num_workers=0, sampler=sampler), test_loader

    return prepare_data


def configure_fresh_run(arm: str):
    config = experiment.build_config(arm, SEED, "formal", "fixed")
    config.run_id = f"{arm}-s{SEED}-bf16-fresh-per-epoch"
    config.launch_id = f"{experiment.EXPERIMENT_ID}-{experiment.run_tag()}-fresh-data"
    config.checkpoint.root_dir = str(fresh_root() / "checkpoints")
    config.training_telemetry_path = str(result_path(arm).parent / "telemetry.jsonl")
    identity = dict(config.resume_identity)
    identity["data_exposure"] = EXPOSURE
    identity["zoology_commit"] = experiment.git_value(experiment.REPO_ROOT, "rev-parse", "HEAD")
    config.resume_identity = identity
    return config


def normalized_fresh_config(config: Any) -> dict[str, Any]:
    payload = experiment.normalized_config(config)
    payload["data_exposure"] = EXPOSURE
    payload["fresh_seed_stride"] = SEED_STRIDE
    payload["fresh_torch_seed_stride"] = TORCH_SEED_STRIDE
    return payload


def result_payload(arm: str, config: Any, resolved: Path, started_at: str, elapsed: float, status: str, error: str | None, controller: dict[str, Any]) -> dict[str, Any]:
    dataset = controller.get("dataset")
    return {
        **experiment.descriptor(arm, SEED, "fixed"),
        "experiment_id": experiment.EXPERIMENT_ID,
        "run_tag": experiment.run_tag(),
        "phase": "formal",
        "exposure": EXPOSURE,
        "status": status,
        "error": error,
        "started_at_utc": started_at,
        "ended_at_utc": utc_now(),
        "wall_clock_sec": elapsed,
        "resolved_config_path": str(resolved.resolve()),
        "resolved_config_sha256": sha256_file(resolved),
        "normalized_config_sha256": stable_json_sha256(normalized_fresh_config(config)),
        "checkpoint_dir": str(experiment.checkpoint_run_dir(config).resolve()),
        "telemetry": experiment.UPSTREAM.telemetry_summary(Path(config.training_telemetry_path)),
        "epoch_data_manifests": None if dataset is None else dataset.epoch_manifests,
    }


def run_training(arm: str) -> int:
    experiment.configure_numerics()
    experiment.configure_gate_bwd_runtime("fixed")
    config = configure_fresh_run(arm)
    resolved = experiment.write_resolved_config(config)
    controller: dict[str, Any] = {}
    started_at, started = utc_now(), time.perf_counter()
    import zoology.train as train_module

    original_prepare_data = train_module.prepare_data
    train_module.prepare_data = make_fresh_prepare_data(controller)
    try:
        train_module.train(config)
        status, error, return_code = "completed", None, 0
    except BaseException as exc:
        status, error, return_code = "failed", f"{type(exc).__name__}: {exc}", 1
        traceback.print_exc()
    finally:
        train_module.prepare_data = original_prepare_data
    payload = result_payload(arm, config, resolved, started_at, time.perf_counter() - started, status, error, controller)
    if status == "completed" and not experiment._completion_result(payload, config):
        payload["status"] = "failed"
        payload["error"] = "Fresh-data training failed completion audit."
        return_code = 1
    atomic_write_json(result_path(arm), payload)
    print(json.dumps({"status": payload["status"], "result": str(result_path(arm))}, ensure_ascii=False))
    return return_code


def curve_from_telemetry(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    valid = [row for row in rows if "valid/mqar_case/accuracy-1024x256" in row]
    values = [float(row["valid/mqar_case/accuracy-1024x256"]) for row in valid]
    best_idx = max(range(len(values)), key=values.__getitem__)
    train = [row for row in rows if "train/loss" in row]
    return {
        "validations": len(valid),
        "best_index": best_idx,
        "best": values[best_idx],
        "final": values[-1],
        "drop": values[-1] - values[best_idx],
        "terminal_train_loss": float(train[-1]["train/loss"]),
        "terminal_valid_loss": float(valid[-1]["valid/loss"]),
    }


def fixed_telemetry(arm: str) -> Path:
    return run_root() / "training" / "formal" / f"{arm}-s{SEED}-bf16-b64ga4-fixed-formal" / "telemetry.jsonl"


def analyze_followup() -> dict[str, Any]:
    rows = {}
    for arm in ARMS:
        fixed = curve_from_telemetry(fixed_telemetry(arm))
        fresh = curve_from_telemetry(result_path(arm).parent / "telemetry.jsonl")
        rows[arm] = {
            "fixed_repeat": fixed,
            "fresh_per_epoch": fresh,
            "fresh_minus_fixed_final": fresh["final"] - fixed["final"],
            "fresh_minus_fixed_drop": fresh["drop"] - fixed["drop"],
        }
    degraded = rows["factor-block"]
    eliminated = degraded["fresh_per_epoch"]["drop"] >= -0.02
    improved = degraded["fresh_minus_fixed_drop"] >= 0.05
    decision = "repeat_data_interaction" if eliminated else ("attenuated_not_eliminated" if improved else "persistent_window_dynamics")
    payload = {
        "status": "completed",
        "decision": decision,
        "rows": rows,
        "completed_at_utc": utc_now(),
    }
    atomic_write_json(run_root() / "analysis" / "fresh-data-followup.json", payload)
    return payload


def source_clean() -> bool:
    roots = (experiment.REPO_ROOT, experiment.FLASH_ROOT)
    return all(not experiment.git_value(root, "status", "--short") for root in roots)


def preflight() -> dict[str, Any]:
    main_done = json.loads((run_root() / "DONE.json").read_text())
    checks = {
        "main_complete": main_done.get("status") == "completed",
        "source_clean": source_clean(),
        "gpu": torch.cuda.is_available() and torch.cuda.get_device_name(0) == "NVIDIA GeForce RTX 3090",
        "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES") == "0",
        "arms": all(experiment._contract_checks(arm, experiment.scientific_contract(configure_fresh_run(arm))).values() for arm in ARMS),
    }
    payload = {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "zoology_commit": experiment.git_value(experiment.REPO_ROOT, "rev-parse", "HEAD"),
        "flash_commit": experiment.git_value(experiment.FLASH_ROOT, "rev-parse", "HEAD"),
        "recorded_at_utc": utc_now(),
    }
    atomic_write_json(fresh_root() / "preflight.json", payload)
    if payload["status"] != "passed":
        raise RuntimeError(f"Fresh-data preflight failed: {payload}")
    return payload


def run_queue() -> int:
    preflight()
    logs = fresh_root() / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    jobs = []
    for arm in ARMS:
        command = [sys.executable, str(Path(__file__).resolve()), "train", "--arm", arm]
        log_path = logs / f"train-{arm}.log"
        started = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as handle:
            process = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
        jobs.append({"arm": arm, "return_code": process.returncode, "wall_clock_sec": time.perf_counter() - started, "log": str(log_path)})
        if process.returncode:
            atomic_write_json(fresh_root() / "status.json", {"status": "failed", "jobs": jobs})
            return process.returncode
    summary = analyze_followup()
    status = {"status": "completed", "jobs": jobs, "summary": summary, "completed_at_utc": utc_now()}
    atomic_write_json(fresh_root() / "DONE.json", status)
    atomic_write_json(fresh_root() / "status.json", status)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    train = sub.add_parser("train")
    train.add_argument("--arm", choices=ARMS, required=True)
    sub.add_parser("run")
    sub.add_parser("analyze")
    args = parser.parse_args()
    if args.command == "preflight":
        preflight()
        return 0
    if args.command == "train":
        return run_training(args.arm)
    if args.command == "analyze":
        print(json.dumps(analyze_followup(), ensure_ascii=False, indent=2))
        return 0
    return run_queue()


if __name__ == "__main__":
    raise SystemExit(main())
