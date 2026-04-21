#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

from zoology.experiments.flash_vqg.manifest import MANIFEST_ENV_VAR, initialize_manifest, manifest_path_for_launch
from zoology.experiments.flash_vqg.run_flash_vqg_suite import (
    GENERATED_DIR,
    REPO_ROOT,
    _build_analysis_command,
    _build_launch_id,
    _load_config_builder,
    _render_generated_config_from_builder,
)
from zoology.train import train


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-builder", type=str, required=True)
    parser.add_argument("--launch-id-prefix", type=str, default=None)
    parser.add_argument("--gpus", type=str, default=None)
    parser.add_argument("--analysis", type=str, default="local")
    parser.add_argument("--backend", type=str, default="torch")
    parser.add_argument("--logger-backend", type=str, default="swanlab")
    parser.add_argument("--dmodels", type=str, default="128")
    parser.add_argument("--learning-rates", type=str, default="1e-3")
    parser.add_argument("--max-epochs", type=int, default=32)
    parser.add_argument("--train-batch-order", type=str, default="global_shuffle")
    parser.add_argument("--seed-values", type=str, default="123")
    parser.add_argument("--data-seed", type=int, default=123)
    parser.add_argument("--num-codebook-vectors", type=str, default="128")
    parser.add_argument("--fox-remote-path-backend", type=str, default="torch")
    parser.add_argument("--fox-clr-rank", type=int, default=4)
    parser.add_argument("--fox-clr-use-den-residual", type=str, default="true")
    parser.add_argument("--fox-clr-remat-mode", type=str, default="off")
    parser.add_argument("--fox-clr-state-write-topk", type=int, default=4)
    parser.add_argument("--fox-clr-delta-target-mode", type=str, default="residual_to_coarse")
    parser.add_argument("--vq-topk", type=int, default=4)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--cache-dir", type=str, default="./data/flash_vqg")
    parser.add_argument("--metrics-white-list", type=str, default=None)
    parser.add_argument("--metrics-white-list-file", type=str, default=None)
    parser.add_argument("--project", type=str, default="flash_vqg_clr_v1_mainline")
    parser.add_argument("--entity", type=str, default="scu-mclab")
    parser.add_argument("--status-interval-sec", type=int, default=600)

    parser.add_argument("--worker-index", type=int, default=None)
    parser.add_argument("--worker-gpu", type=str, default=None)
    parser.add_argument("--builder-args-file", type=str, default=None)
    parser.add_argument("--launch-id", type=str, default=None)
    parser.add_argument("--manifest-path", type=str, default=None)
    return parser


def _builder_args_from_file(path: Path) -> argparse.Namespace:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return argparse.Namespace(**payload)


def _orchestrator(args: argparse.Namespace) -> int:
    if args.launch_id_prefix is None or args.gpus is None:
        raise ValueError("orchestrator 模式要求提供 launch-id-prefix 和 gpus.")
    if (
        args.train_batch_size is None
        or args.eval_batch_size is None
        or args.gradient_accumulation_steps is None
    ):
        raise ValueError(
            "orchestrator 模式要求提供 train-batch-size, eval-batch-size, gradient-accumulation-steps."
        )
    builder = _load_config_builder(args.config_builder)
    configs = builder(args)
    if not configs:
        raise ValueError("builder 返回了空配置列表.")

    launch_id = _build_launch_id(args.launch_id_prefix)
    generated_launch_dir = GENERATED_DIR / launch_id
    generated_launch_dir.mkdir(parents=True, exist_ok=True)
    builder_args_path = generated_launch_dir / "builder_args.json"
    builder_args_path.write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    generated_config_path = generated_launch_dir / "launch_configs.py"
    generated_config_path.write_text(
        _render_generated_config_from_builder(
            builder_spec=args.config_builder,
            builder_args=vars(args),
        ),
        encoding="utf-8",
    )
    manifest_path = manifest_path_for_launch(launch_id)
    initialize_manifest(
        manifest_path=manifest_path,
        launch_id=launch_id,
        sweep_id=args.launch_id_prefix,
        logger_backend=args.logger_backend,
        project=args.project,
        entity=args.entity,
        run_ids=[config.run_id for config in configs],
        launch_config_file=generated_config_path,
    )

    gpu_ids = [gpu.strip() for gpu in str(args.gpus).split(",") if gpu.strip()]
    if not gpu_ids:
        raise ValueError("gpus 不能为空.")

    pending = deque(range(len(configs)))
    running: dict[int, tuple[subprocess.Popen, str, int]] = {}
    finished: dict[int, int] = {}
    script_path = Path(__file__).resolve()
    base_env = os.environ.copy()
    exit_code = 0
    status_interval_sec = max(30, int(args.status_interval_sec))
    last_status_report_ts = 0.0
    start_ts = time.time()

    print(f"[local-launch] launch_id={launch_id} configs={len(configs)} gpus={gpu_ids}", flush=True)
    while pending or running:
        busy_gpus = {gpu for _, gpu, _ in running.values()}
        idle_gpus = [gpu for gpu in gpu_ids if gpu not in busy_gpus]
        while pending and idle_gpus:
            config_idx = pending.popleft()
            gpu_id = idle_gpus.pop(0)
            worker_cmd = [
                sys.executable,
                str(script_path),
                "--worker-index",
                str(config_idx),
                "--worker-gpu",
                gpu_id,
                "--builder-args-file",
                str(builder_args_path),
                "--config-builder",
                args.config_builder,
                "--launch-id",
                launch_id,
                "--manifest-path",
                str(manifest_path),
            ]
            worker_env = dict(base_env)
            worker_env[MANIFEST_ENV_VAR] = str(manifest_path)
            worker_env["CUDA_VISIBLE_DEVICES"] = gpu_id
            proc = subprocess.Popen(worker_cmd, cwd=REPO_ROOT, env=worker_env)
            running[proc.pid] = (proc, gpu_id, config_idx)
            print(
                f"[local-launch] started run_id={configs[config_idx].run_id} gpu={gpu_id} pid={proc.pid}",
                flush=True,
            )

        now = time.time()
        if now - last_status_report_ts >= status_interval_sec:
            running_names = [configs[idx].run_id for _, _, idx in running.values()]
            pending_names = [configs[idx].run_id for idx in pending]
            completed_names = [configs[idx].run_id for idx, rc in sorted(finished.items()) if rc == 0]
            failed_names = [configs[idx].run_id for idx, rc in sorted(finished.items()) if rc != 0]
            print(
                "[local-launch][status] "
                f"elapsed_sec={int(now - start_ts)} "
                f"pending={pending_names} running={running_names} "
                f"completed={completed_names} failed={failed_names}",
                flush=True,
            )
            last_status_report_ts = now

        time.sleep(5)
        finished_pids = []
        for pid, (proc, gpu_id, config_idx) in running.items():
            ret = proc.poll()
            if ret is None:
                continue
            finished_pids.append(pid)
            finished[config_idx] = ret
            print(
                f"[local-launch] finished run_id={configs[config_idx].run_id} gpu={gpu_id} rc={ret}",
                flush=True,
            )
            if ret != 0:
                exit_code = ret
        for pid in finished_pids:
            running.pop(pid, None)

    if args.analysis != "off":
        analysis_cmd = _build_analysis_command(launch_id=launch_id, source=args.analysis)
        print("[local-launch] start analysis:", " ".join(analysis_cmd), flush=True)
        analysis_rc = subprocess.run(analysis_cmd, check=False, cwd=REPO_ROOT, env=base_env).returncode
        print(f"[local-launch] analysis finished rc={analysis_rc}", flush=True)
        if analysis_rc != 0 and exit_code == 0:
            exit_code = analysis_rc
    return exit_code


def _worker(args: argparse.Namespace) -> int:
    if args.worker_index is None or args.worker_gpu is None:
        raise ValueError("worker 模式缺少 worker-index 或 worker-gpu.")
    if args.builder_args_file is None or args.launch_id is None or args.manifest_path is None:
        raise ValueError("worker 模式缺少 builder-args-file / launch-id / manifest-path.")
    builder_args = _builder_args_from_file(Path(args.builder_args_file))
    builder = _load_config_builder(args.config_builder)
    configs = builder(builder_args)
    config = configs[int(args.worker_index)]
    config.launch_id = args.launch_id
    os.environ[MANIFEST_ENV_VAR] = str(Path(args.manifest_path).resolve())
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.worker_gpu)
    train(config)
    return 0


def main() -> None:
    args = _build_parser().parse_args()
    if args.worker_index is not None:
        raise SystemExit(_worker(args))
    raise SystemExit(_orchestrator(args))


if __name__ == "__main__":
    main()
