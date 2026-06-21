from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

from experiment_lib import MANIFEST_ENV_VAR, PYTHON_BIN, REPO_ROOT


def _load_configs(path: Path):
    spec = importlib.util.spec_from_file_location("gd_init_transplant_config_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 config: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.configs)


def _parse_only_indices(value: str, *, total: int) -> list[int]:
    if not value:
        return list(range(total))
    indices: list[int] = []
    for part in value.split(","):
        token = part.strip()
        if not token:
            continue
        index = int(token)
        if index < 0 or index >= total:
            raise ValueError(f"--only-indices 包含越界 index: {index}, total={total}.")
        indices.append(index)
    if not indices:
        raise ValueError("--only-indices 没有解析出任何 index.")
    return indices


def _worker(args: argparse.Namespace) -> int:
    if args.gpu is None:
        raise ValueError("--worker-index 模式必须指定 --gpu.")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if args.manifest_path:
        os.environ[MANIFEST_ENV_VAR] = str(Path(args.manifest_path).resolve())
    if os.environ.get("TORCH_DETERMINISTIC") == "1":
        raise RuntimeError("本实验禁止启用 TORCH_DETERMINISTIC=1.")
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from zoology.train import train

    configs = _load_configs(Path(args.launch_config))
    config = configs[int(args.worker_index)]
    config.launch_id = str(args.launch_id)
    print(f"[worker] gpu={args.gpu} index={args.worker_index} run_id={config.run_id}", flush=True)
    train(config)
    return 0


def _parent(args: argparse.Namespace) -> int:
    launch_config = Path(args.launch_config).resolve()
    configs = _load_configs(launch_config)
    gpu_ids = [part.strip() for part in str(args.gpus).split(",") if part.strip()]
    if not gpu_ids:
        raise ValueError("--gpus 不能为空.")
    log_dir = launch_config.parent / "local_parallel_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    pending = _parse_only_indices(args.only_indices, total=len(configs))
    scheduled_total = len(pending)
    running: dict[subprocess.Popen, tuple[int, str, Path]] = {}
    completed = 0
    failed = 0
    env_base = os.environ.copy()
    env_base["PYTHONPATH"] = f"{REPO_ROOT}:{env_base.get('PYTHONPATH', '')}"
    if args.manifest_path:
        env_base[MANIFEST_ENV_VAR] = str(Path(args.manifest_path).resolve())
    if env_base.get("TORCH_DETERMINISTIC") == "1":
        raise RuntimeError("本实验禁止启用 TORCH_DETERMINISTIC=1.")

    print(f"[local-parallel] launch_id={args.launch_id} runs={len(configs)} gpus={','.join(gpu_ids)}", flush=True)
    if args.only_indices:
        print(f"[local-parallel] only_indices={','.join(str(index) for index in pending)}", flush=True)
    while pending or running:
        busy_gpus = {gpu for _, gpu, _ in running.values()}
        free_gpus = [gpu for gpu in gpu_ids if gpu not in busy_gpus]
        while pending and free_gpus:
            index = pending.pop(0)
            gpu = free_gpus.pop(0)
            run_id = str(configs[index].run_id)
            log_path = log_dir / f"{index:02d}-{run_id}.log"
            cmd = [
                PYTHON_BIN,
                str(Path(__file__).resolve()),
                "--launch-config",
                str(launch_config),
                "--launch-id",
                str(args.launch_id),
                "--manifest-path",
                str(Path(args.manifest_path).resolve()) if args.manifest_path else "",
                "--worker-index",
                str(index),
                "--gpu",
                str(gpu),
            ]
            env = env_base.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            log_file = log_path.open("a" if args.append_logs else "w", encoding="utf-8")
            if args.append_logs:
                print(
                    f"\n[local-parallel] append launch start index={index} gpu={gpu} run_id={run_id} "
                    f"time={time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
                    file=log_file,
                    flush=True,
                )
            process = subprocess.Popen(
                cmd,
                cwd=REPO_ROOT,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
            log_file.close()
            running[process] = (index, gpu, log_path)
            print(f"[local-parallel] start index={index} gpu={gpu} run_id={run_id} log={log_path}", flush=True)

        time.sleep(float(args.poll_seconds))
        for process in list(running):
            code = process.poll()
            if code is None:
                continue
            index, gpu, log_path = running.pop(process)
            completed += 1
            if code != 0:
                failed += 1
                print(
                    f"[local-parallel] failed index={index} gpu={gpu} code={code} log={log_path}",
                    flush=True,
                )
            else:
                print(
                    f"[local-parallel] done index={index} gpu={gpu} completed={completed}/{scheduled_total}",
                    flush=True,
                )
    print(f"[local-parallel] complete completed={completed}/{scheduled_total} failed={failed}", flush=True)
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="本实验专用本地多 GPU launcher.")
    parser.add_argument("--launch-config", required=True)
    parser.add_argument("--launch-id", required=True)
    parser.add_argument("--manifest-path", default="")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--only-indices", default="")
    parser.add_argument("--append-logs", action="store_true")
    parser.add_argument("--worker-index", type=int, default=None)
    parser.add_argument("--gpu", default=None)
    args = parser.parse_args()
    if args.worker_index is not None:
        return _worker(args)
    return _parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
