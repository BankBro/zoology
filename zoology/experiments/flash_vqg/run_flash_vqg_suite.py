import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from zoology.experiments.flash_vqg.flash_vqg_suite import build_configs
from zoology.experiments.flash_vqg.manifest import MANIFEST_ENV_VAR, initialize_manifest


REPO_ROOT = Path(__file__).resolve().parents[3]
GENERATED_DIR = Path(__file__).resolve().parent / "generated"


def _parse_csv_ints(raw: str) -> list[int]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("dmodels 不能为空.")
    return [int(v) for v in values]


def _parse_csv_floats(raw: str) -> list[float]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("learning rates 不能为空.")
    return [float(v) for v in values]


def _parse_csv_bools(raw: str) -> list[bool]:
    values = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("if_remote_enabled 不能为空.")

    mapping = {
        "1": True,
        "0": False,
        "true": True,
        "false": False,
        "yes": True,
        "no": False,
    }
    normalized: list[bool] = []
    seen: set[bool] = set()
    for value in values:
        if value not in mapping:
            raise ValueError(f"if_remote_enabled 只能是 true/false 或 1/0, 当前收到: {value}")
        parsed = mapping[value]
        if parsed not in seen:
            normalized.append(parsed)
            seen.add(parsed)
    return normalized


def _parse_train_batch_orders(raw: str) -> list[str]:
    values = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("train_batch_order 不能为空.")

    valid_orders = {"sequential", "global_shuffle", "balanced_interleave"}
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in valid_orders:
            raise ValueError(
                f"train_batch_order 只能是 {sorted(valid_orders)}, 当前收到: {value}"
            )
        if value not in seen:
            normalized.append(value)
            seen.add(value)
    return normalized


def _normalize_launch_id_prefix(prefix: str) -> str:
    normalized = str(prefix).strip()
    if not normalized:
        raise ValueError("launch id prefix 不能为空.")
    return normalized


def _build_launch_id(launch_id_prefix: str) -> str:
    timestamp = datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S")
    return f"{launch_id_prefix}-{timestamp}"


def _render_generated_config(
    *,
    sweep_id: str,
    backend: str,
    logger_backend: str,
    include_gdn: bool,
    block_len: int,
    dmodels: list[int],
    learning_rates: list[float],
    if_remote_enabled_values: list[bool],
    local_num_blocks_values: list[int],
    train_batch_orders: list[str],
    cache_dir: str,
    wandb_project: str,
    wandb_entity: str,
    max_epochs: int,
) -> str:
    lines = [
        "# -*- coding: utf-8 -*-",
        "# 此文件由 run_flash_vqg_suite.py 自动生成.",
        "# 如需调整实验参数, 请修改 wrapper 入参后重新生成.",
        "",
        "from zoology.experiments.flash_vqg.flash_vqg_suite import build_configs",
        "",
        "configs = build_configs(",
        f"    sweep_id={sweep_id!r},",
        f"    flash_backend={backend!r},",
        f"    logger_backend={logger_backend!r},",
        f"    include_gdn={include_gdn!r},",
        f"    block_len={block_len!r},",
        f"    dmodels={dmodels!r},",
        f"    learning_rates={learning_rates!r},",
        f"    if_remote_enabled_values={if_remote_enabled_values!r},",
        f"    local_num_blocks_values={local_num_blocks_values!r},",
        f"    train_batch_orders={train_batch_orders!r},",
        f"    cache_dir={cache_dir!r},",
        f"    wandb_project={wandb_project!r},",
        f"    wandb_entity={wandb_entity!r},",
        f"    max_epochs={max_epochs!r},",
        ")",
        "",
    ]
    return "\n".join(lines)


def _build_manifest_run_ids(
    *,
    sweep_id: str,
    backend: str,
    logger_backend: str,
    include_gdn: bool,
    block_len: int,
    dmodels: list[int],
    learning_rates: list[float],
    if_remote_enabled_values: list[bool],
    local_num_blocks_values: list[int],
    train_batch_orders: list[str],
    cache_dir: str,
    project: str,
    entity: str,
    max_epochs: int,
) -> list[str]:
    configs = build_configs(
        sweep_id=sweep_id,
        flash_backend=backend,
        logger_backend=logger_backend,
        include_gdn=include_gdn,
        block_len=block_len,
        dmodels=dmodels,
        learning_rates=learning_rates,
        if_remote_enabled_values=if_remote_enabled_values,
        local_num_blocks_values=local_num_blocks_values,
        train_batch_orders=train_batch_orders,
        cache_dir=cache_dir,
        wandb_project=project,
        wandb_entity=entity,
        max_epochs=max_epochs,
    )
    return [config.run_id for config in configs]


def _build_launch_command(
    *,
    generated_config_path: Path,
    launch_id: str,
    outdir: str | None,
    parallelize: bool,
    gpus: str | None,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "zoology.launch",
        str(generated_config_path),
        "--launch-id",
        launch_id,
    ]
    if outdir is not None:
        cmd.extend(["--outdir", outdir])
    if parallelize:
        cmd.append("-p")
    if gpus is not None:
        cmd.extend(["--gpus", gpus])
    return cmd


def _build_analysis_command(*, launch_id: str, source: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "zoology.analysis.flash_vqg.run_flash_vqg_analysis",
        "--launch-id",
        launch_id,
        "--source",
        source,
    ]


def main():
    parser = argparse.ArgumentParser(description="生成并启动 Flash-VQG MQAR 实验配置.")
    parser.add_argument("--backend", choices=["accel", "torch"], default="accel")
    parser.add_argument("--logger-backend", choices=["wandb", "swanlab", "none"], default="wandb")
    parser.add_argument("--include-gdn", dest="include_gdn", action="store_true", default=True)
    parser.add_argument("--flash-only", dest="include_gdn", action="store_false")
    parser.add_argument("--block-len", type=int, default=8)
    parser.add_argument("--dmodels", type=str, default="128")
    parser.add_argument("--learning-rates", type=str, default="1e-4,3e-4,1e-3,3e-3")
    parser.add_argument("--if-remote-enabled", type=str, default="true")
    parser.add_argument("--local-num-blocks", type=str, default="2")
    parser.add_argument("--train-batch-order", type=str, default="sequential")
    parser.add_argument("--cache-dir", type=str, default="./data/flash_vqg")
    parser.add_argument("--project", type=str, default="flash_vqg_vs_gdn")
    parser.add_argument("--entity", type=str, default="scu-mclab")
    parser.add_argument("--max-epochs", type=int, default=32)
    parser.add_argument("--launch-id-prefix", type=str, default="flash-vqg-suite")
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--gpus", type=str, default=None)
    parser.add_argument("-p", "--parallelize", action="store_true")
    parser.add_argument(
        "--analysis",
        choices=["off", "remote", "local"],
        default="off",
        help="训练完成后是否自动执行 analysis, 以及使用的数据源.",
    )
    args = parser.parse_args()

    dmodels = _parse_csv_ints(args.dmodels)
    learning_rates = _parse_csv_floats(args.learning_rates)
    if_remote_enabled_values = _parse_csv_bools(args.if_remote_enabled)
    local_num_blocks_values = _parse_csv_ints(args.local_num_blocks)
    train_batch_orders = _parse_train_batch_orders(args.train_batch_order)
    launch_id_prefix = _normalize_launch_id_prefix(args.launch_id_prefix)
    launch_id = _build_launch_id(launch_id_prefix)

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    generated_launch_dir = GENERATED_DIR / launch_id
    generated_launch_dir.mkdir(parents=True, exist_ok=True)
    generated_path = generated_launch_dir / "launch_configs.py"
    manifest_path = generated_launch_dir / "manifest.json"
    run_ids = _build_manifest_run_ids(
        sweep_id=launch_id_prefix,
        backend=args.backend,
        logger_backend=args.logger_backend,
        include_gdn=args.include_gdn,
        block_len=args.block_len,
        dmodels=dmodels,
        learning_rates=learning_rates,
        if_remote_enabled_values=if_remote_enabled_values,
        local_num_blocks_values=local_num_blocks_values,
        train_batch_orders=train_batch_orders,
        cache_dir=args.cache_dir,
        project=args.project,
        entity=args.entity,
        max_epochs=args.max_epochs,
    )
    generated_path.write_text(
        _render_generated_config(
            sweep_id=launch_id_prefix,
            backend=args.backend,
            logger_backend=args.logger_backend,
            include_gdn=args.include_gdn,
            block_len=args.block_len,
            dmodels=dmodels,
            learning_rates=learning_rates,
            if_remote_enabled_values=if_remote_enabled_values,
            local_num_blocks_values=local_num_blocks_values,
            train_batch_orders=train_batch_orders,
            cache_dir=args.cache_dir,
            wandb_project=args.project,
            wandb_entity=args.entity,
            max_epochs=args.max_epochs,
        ),
        encoding="utf-8",
    )
    initialize_manifest(
        manifest_path=manifest_path,
        launch_id=launch_id,
        sweep_id=launch_id_prefix,
        logger_backend=args.logger_backend,
        project=args.project,
        entity=args.entity,
        run_ids=run_ids,
        launch_config_file=generated_path,
    )

    launch_cmd = _build_launch_command(
        generated_config_path=generated_path,
        launch_id=launch_id,
        outdir=args.outdir,
        parallelize=args.parallelize,
        gpus=args.gpus,
    )
    print(f"已生成配置脚本: {generated_path}")
    print(f"已生成 manifest: {manifest_path}")
    print("即将执行命令:")
    print(" ".join(shlex.quote(part) for part in launch_cmd))
    env = os.environ.copy()
    env[MANIFEST_ENV_VAR] = str(manifest_path.resolve())
    subprocess.run(launch_cmd, check=True, cwd=REPO_ROOT, env=env)

    if args.analysis != "off":
        analysis_cmd = _build_analysis_command(launch_id=launch_id, source=args.analysis)
        print("训练已完成, 即将自动执行 analysis:")
        print(" ".join(shlex.quote(part) for part in analysis_cmd))
        subprocess.run(analysis_cmd, check=True, cwd=REPO_ROOT, env=env)


if __name__ == "__main__":
    main()
