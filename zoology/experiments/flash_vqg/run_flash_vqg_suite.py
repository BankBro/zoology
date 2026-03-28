import argparse
import importlib
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from zoology.experiments.flash_vqg.flash_vqg_suite import build_configs
from zoology.experiments.flash_vqg.manifest import MANIFEST_ENV_VAR, initialize_manifest
from zoology.experiments.flash_vqg.metrics_white_list import (
    load_metrics_white_list_file,
    merge_metrics_white_lists,
    parse_metrics_white_list_csv,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
GENERATED_DIR = Path(__file__).resolve().parent / "generated"
E7_MODE_NAMES = ("dense", "top2", "top4")


def _parse_csv_ints(raw: str) -> list[int]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("dmodels 不能为空.")
    return [int(v) for v in values]


def _parse_codebook_vectors_map(raw: str) -> dict[int, int]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("num_codebook_vectors_map 不能为空.")

    normalized: dict[int, int] = {}
    for value in values:
        if ":" not in value:
            raise ValueError(
                "num_codebook_vectors_map 的格式必须是 d_model:num_codebook_vectors, "
                f"当前收到: {value}"
            )
        d_model_raw, num_codes_raw = [part.strip() for part in value.split(":", maxsplit=1)]
        d_model = int(d_model_raw)
        num_codes = int(num_codes_raw)
        if d_model <= 0:
            raise ValueError(f"d_model 必须是正整数, 当前收到: {d_model_raw}")
        if num_codes <= 0:
            raise ValueError(
                f"num_codebook_vectors 必须是正整数, 当前收到: {num_codes_raw}"
            )
        normalized[d_model] = num_codes
    return normalized


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


def _parse_paired_block_local(raw: str) -> list[tuple[int, int]]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("paired_block_local 不能为空.")

    normalized: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for value in values:
        if ":" not in value:
            raise ValueError(
                f"paired_block_local 的格式必须是 block_len:local_num_blocks, 当前收到: {value}"
            )
        block_len_raw, local_num_blocks_raw = [part.strip() for part in value.split(":", maxsplit=1)]
        pair = (int(block_len_raw), int(local_num_blocks_raw))
        if pair not in seen:
            normalized.append(pair)
            seen.add(pair)
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


def _resolve_metrics_white_list(
    *,
    metrics_white_list_raw: str | None,
    metrics_white_list_file: str | None,
) -> list[str]:
    file_values = (
        load_metrics_white_list_file(metrics_white_list_file)
        if metrics_white_list_file is not None
        else []
    )
    inline_values = parse_metrics_white_list_csv(metrics_white_list_raw)
    return merge_metrics_white_lists(file_values, inline_values)


def _normalize_launch_id_prefix(prefix: str) -> str:
    normalized = str(prefix).strip()
    if not normalized:
        raise ValueError("launch id prefix 不能为空.")
    return normalized


def _build_launch_id(launch_id_prefix: str) -> str:
    timestamp = datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S")
    return f"{launch_id_prefix}-{timestamp}"


def _build_eval_run_id(checkpoint_run_id: str, eval_task: str) -> str:
    return f"eval_{eval_task}_{checkpoint_run_id}"


def _build_e7_eval_run_ids(checkpoint_run_id: str) -> list[str]:
    return [f"eval_e7_{mode_name}_{checkpoint_run_id}" for mode_name in E7_MODE_NAMES]


def _render_eval_request(
    *,
    eval_task: str,
    eval_launch_id: str,
    eval_sweep_id: str,
    eval_run_id: str,
    eval_run_ids: list[str] | None,
    checkpoint_launch_id: str,
    checkpoint_run_id: str,
    logger_backend: str,
    project: str | None,
    entity: str | None,
    metrics_white_list: list[str],
) -> str:
    lines = [
        "# -*- coding: utf-8 -*-",
        "# 此文件由 run_flash_vqg_suite.py 自动生成.",
        "# 当前 launch 为 eval-only 请求, 不会交给 zoology.launch 执行.",
        "",
        "eval_request = {",
        f"    'eval_task': {eval_task!r},",
        f"    'eval_launch_id': {eval_launch_id!r},",
        f"    'eval_sweep_id': {eval_sweep_id!r},",
        f"    'eval_run_id': {eval_run_id!r},",
        f"    'eval_run_ids': {eval_run_ids!r},",
        f"    'checkpoint_launch_id': {checkpoint_launch_id!r},",
        f"    'checkpoint_run_id': {checkpoint_run_id!r},",
        f"    'logger_backend': {logger_backend!r},",
        f"    'project': {project!r},",
        f"    'entity': {entity!r},",
        f"    'metrics_white_list': {metrics_white_list!r},",
        "}",
        "",
    ]
    return "\n".join(lines)


def _render_generated_config(
    *,
    sweep_id: str,
    backend: str,
    logger_backend: str,
    include_gdn: bool,
    block_lens: list[int] | None,
    paired_block_local_values: list[tuple[int, int]] | None,
    dmodels: list[int],
    learning_rates: list[float],
    if_remote_enabled_values: list[bool],
    local_num_blocks_values: list[int] | None,
    train_batch_orders: list[str],
    num_codebook_vectors_values: list[int] | None,
    num_codebook_vectors_map: dict[int, int] | None,
    cache_dir: str,
    wandb_project: str,
    wandb_entity: str,
    max_epochs: int,
    metrics_white_list: list[str],
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
    ]
    if paired_block_local_values is not None:
        lines.append(f"    paired_block_local_values={paired_block_local_values!r},")
    else:
        lines.extend(
            [
                f"    block_len_values={block_lens!r},",
                f"    local_num_blocks_values={local_num_blocks_values!r},",
            ]
        )
    lines.extend(
        [
        f"    dmodels={dmodels!r},",
        f"    learning_rates={learning_rates!r},",
        f"    if_remote_enabled_values={if_remote_enabled_values!r},",
        f"    train_batch_orders={train_batch_orders!r},",
        f"    num_codebook_vectors_values={num_codebook_vectors_values!r},",
        f"    num_codebook_vectors_map={num_codebook_vectors_map!r},",
        f"    cache_dir={cache_dir!r},",
        f"    wandb_project={wandb_project!r},",
        f"    wandb_entity={wandb_entity!r},",
        f"    max_epochs={max_epochs!r},",
        f"    metrics_white_list={metrics_white_list!r},",
        ")",
        "",
        ]
    )
    return "\n".join(lines)


def _build_manifest_run_ids(
    *,
    sweep_id: str,
    backend: str,
    logger_backend: str,
    include_gdn: bool,
    block_lens: list[int] | None,
    paired_block_local_values: list[tuple[int, int]] | None,
    dmodels: list[int],
    learning_rates: list[float],
    if_remote_enabled_values: list[bool],
    local_num_blocks_values: list[int] | None,
    train_batch_orders: list[str],
    num_codebook_vectors_values: list[int] | None,
    num_codebook_vectors_map: dict[int, int] | None,
    cache_dir: str,
    project: str,
    entity: str,
    max_epochs: int,
    metrics_white_list: list[str],
) -> list[str]:
    configs = build_configs(
        sweep_id=sweep_id,
        flash_backend=backend,
        logger_backend=logger_backend,
        include_gdn=include_gdn,
        **(
            {"paired_block_local_values": paired_block_local_values}
            if paired_block_local_values is not None
            else {
                "block_len_values": block_lens,
                "local_num_blocks_values": local_num_blocks_values,
            }
        ),
        dmodels=dmodels,
        learning_rates=learning_rates,
        if_remote_enabled_values=if_remote_enabled_values,
        train_batch_orders=train_batch_orders,
        num_codebook_vectors_values=num_codebook_vectors_values,
        num_codebook_vectors_map=num_codebook_vectors_map,
        cache_dir=cache_dir,
        wandb_project=project,
        wandb_entity=entity,
        max_epochs=max_epochs,
        metrics_white_list=metrics_white_list,
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


def _run_eval_only(
    *,
    eval_task: str,
    eval_launch_id: str,
    eval_sweep_id: str,
    eval_run_id: str,
    eval_run_ids: list[str] | None,
    checkpoint_launch_id: str,
    checkpoint_run_id: str,
    logger_backend: str,
    project: str,
    entity: str,
    manifest_path: Path,
    gpus: str | None,
    metrics_white_list: list[str] | None,
) -> dict:
    if gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    eval_only_module = importlib.import_module("zoology.experiments.flash_vqg.eval_only")

    eval_fns = {
        "e4a": getattr(eval_only_module, "run_e4a_eval", None),
        "e7": getattr(eval_only_module, "run_e7_eval", None),
    }
    if eval_task not in eval_fns:
        raise ValueError(f"不支持的 eval task: {eval_task}")
    if eval_fns[eval_task] is None:
        raise RuntimeError(f"eval_only 模块缺少 {eval_task} 对应的执行入口.")

    print(f"已进入 eval-only 模式, 将直接加载历史 checkpoint 并执行 {eval_task.upper()} 测试.")
    print(f"eval_launch_id={eval_launch_id}")
    print(f"eval_sweep_id={eval_sweep_id}")
    if eval_run_ids is None:
        print(f"eval_run_id={eval_run_id}")
    else:
        print(f"eval_run_ids={','.join(eval_run_ids)}")
    print(f"eval_task={eval_task}")
    print(f"checkpoint_launch_id={checkpoint_launch_id}")
    print(f"checkpoint_run_id={checkpoint_run_id}")
    if gpus is not None:
        print(f"CUDA_VISIBLE_DEVICES={gpus}")
    eval_kwargs = dict(
        checkpoint_launch_id=checkpoint_launch_id,
        checkpoint_run_id=checkpoint_run_id,
        eval_launch_id=eval_launch_id,
        eval_sweep_id=eval_sweep_id,
        logger_backend=logger_backend,
        project=project,
        entity=entity,
        manifest_path=manifest_path,
        metrics_white_list=metrics_white_list,
    )
    if eval_task == "e7":
        eval_kwargs["eval_run_ids"] = eval_run_ids
    else:
        eval_kwargs["eval_run_id"] = eval_run_id
    result = eval_fns[eval_task](**eval_kwargs)
    print(f"{eval_task.upper()} eval-only 已完成.")
    print(f"结果目录: {result['output_dir']}")
    return result


def main():
    parser = argparse.ArgumentParser(description="生成并启动 Flash-VQG MQAR 实验配置.")
    parser.add_argument("--backend", choices=["accel", "torch"], default="accel")
    parser.add_argument("--logger-backend", choices=["wandb", "swanlab", "none"], default="wandb")
    parser.add_argument("--include-gdn", dest="include_gdn", action="store_true", default=True)
    parser.add_argument("--flash-only", dest="include_gdn", action="store_false")
    parser.add_argument(
        "--eval-only",
        choices=["e4a", "e7"],
        default=None,
        help="跳过训练, 直接加载历史 checkpoint 执行指定 eval task.",
    )
    parser.add_argument("--checkpoint-launch-id", type=str, default=None, help="eval-only 模式下要加载的 checkpoint launch_id.")
    parser.add_argument("--checkpoint-run-id", type=str, default=None, help="eval-only 模式下要加载的 checkpoint run_id.")
    parser.add_argument("--block-len", type=str, default="8")
    parser.add_argument(
        "--paired-block-local",
        type=str,
        default=None,
        help="按 block_len:local_num_blocks 的形式做配对扫描, 例如 8:8,16:4,32:2,64:1.",
    )
    parser.add_argument("--dmodels", type=str, default="128")
    parser.add_argument(
        "--num-codebook-vectors",
        type=str,
        default=None,
        help="逗号分隔的 codebook size sweep, 例如 64,128,256,512.",
    )
    parser.add_argument(
        "--num-codebook-vectors-map",
        type=str,
        default=None,
        help="按 d_model:num_codebook_vectors 指定映射, 例如 128:128,256:256.",
    )
    parser.add_argument("--learning-rates", type=str, default="1e-4,3e-4,1e-3,3e-3")
    parser.add_argument("--if-remote-enabled", type=str, default="true")
    parser.add_argument("--local-num-blocks", type=str, default="2")
    parser.add_argument("--train-batch-order", type=str, default="sequential")
    parser.add_argument("--cache-dir", type=str, default="./data/flash_vqg")
    parser.add_argument(
        "--metrics-white-list",
        type=str,
        default=None,
        help="逗号分隔的 metrics 白名单模式串, 例如 valid/accuracy,valid/mqar_case/*.",
    )
    parser.add_argument(
        "--metrics-white-list-file",
        type=str,
        default=None,
        help="JSON/YAML 格式的 metrics 白名单文件路径.",
    )
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
    metrics_white_list_provided = (
        args.metrics_white_list is not None or args.metrics_white_list_file is not None
    )
    metrics_white_list = _resolve_metrics_white_list(
        metrics_white_list_raw=args.metrics_white_list,
        metrics_white_list_file=args.metrics_white_list_file,
    )

    if args.eval_only is not None:
        if not args.checkpoint_launch_id or not args.checkpoint_run_id:
            raise ValueError("--eval-only 模式下必须同时提供 --checkpoint-launch-id 和 --checkpoint-run-id.")
        launch_id_prefix = _normalize_launch_id_prefix(args.launch_id_prefix)
        launch_id = _build_launch_id(launch_id_prefix)
        eval_run_ids = (
            _build_e7_eval_run_ids(args.checkpoint_run_id)
            if args.eval_only == "e7"
            else [_build_eval_run_id(args.checkpoint_run_id, args.eval_only)]
        )
        eval_run_id = eval_run_ids[0]
        GENERATED_DIR.mkdir(parents=True, exist_ok=True)
        generated_launch_dir = GENERATED_DIR / launch_id
        generated_launch_dir.mkdir(parents=True, exist_ok=True)
        generated_path = generated_launch_dir / "launch_configs.py"
        manifest_path = generated_launch_dir / "manifest.json"
        generated_path.write_text(
            _render_eval_request(
                eval_task=args.eval_only,
                eval_launch_id=launch_id,
                eval_sweep_id=launch_id_prefix,
                eval_run_id=eval_run_id,
                eval_run_ids=eval_run_ids,
                checkpoint_launch_id=args.checkpoint_launch_id,
                checkpoint_run_id=args.checkpoint_run_id,
                logger_backend=args.logger_backend,
                project=args.project,
                entity=args.entity,
                metrics_white_list=metrics_white_list,
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
            run_ids=eval_run_ids,
            launch_config_file=generated_path,
            eval_task=args.eval_only,
            eval_sources={
                run_id: {
                    "checkpoint_launch_id": args.checkpoint_launch_id,
                    "checkpoint_run_id": args.checkpoint_run_id,
                    "best_checkpoint": None,
                }
                for run_id in eval_run_ids
            },
        )
        print(f"已生成 eval-only 请求: {generated_path}")
        print(f"已生成 manifest: {manifest_path}")
        _run_eval_only(
            eval_task=args.eval_only,
            eval_launch_id=launch_id,
            eval_sweep_id=launch_id_prefix,
            eval_run_id=eval_run_id,
            eval_run_ids=eval_run_ids if args.eval_only == "e7" else None,
            checkpoint_launch_id=args.checkpoint_launch_id,
            checkpoint_run_id=args.checkpoint_run_id,
            logger_backend=args.logger_backend,
            project=args.project,
            entity=args.entity,
            manifest_path=manifest_path,
            gpus=args.gpus,
            metrics_white_list=metrics_white_list if metrics_white_list_provided else None,
        )
        if args.analysis != "off":
            analysis_cmd = _build_analysis_command(launch_id=launch_id, source=args.analysis)
            print("评测已完成, 即将自动执行 analysis:")
            print(" ".join(shlex.quote(part) for part in analysis_cmd))
            subprocess.run(analysis_cmd, check=True, cwd=REPO_ROOT, env=os.environ.copy())
        return

    if args.checkpoint_launch_id is not None or args.checkpoint_run_id is not None:
        raise ValueError("--checkpoint-launch-id 和 --checkpoint-run-id 只能与 --eval-only 一起使用.")

    if args.paired_block_local is not None and (
        args.block_len != "8" or args.local_num_blocks != "2"
    ):
        raise ValueError("--paired-block-local 不能与 --block-len 或 --local-num-blocks 同时使用.")

    paired_block_local_values = (
        _parse_paired_block_local(args.paired_block_local)
        if args.paired_block_local is not None
        else None
    )
    if args.num_codebook_vectors is not None and args.num_codebook_vectors_map is not None:
        raise ValueError(
            "--num-codebook-vectors 和 --num-codebook-vectors-map 不能同时使用."
        )
    block_lens = _parse_csv_ints(args.block_len) if paired_block_local_values is None else None
    dmodels = _parse_csv_ints(args.dmodels)
    num_codebook_vectors_values = (
        _parse_csv_ints(args.num_codebook_vectors)
        if args.num_codebook_vectors is not None
        else None
    )
    num_codebook_vectors_map = (
        _parse_codebook_vectors_map(args.num_codebook_vectors_map)
        if args.num_codebook_vectors_map is not None
        else None
    )
    learning_rates = _parse_csv_floats(args.learning_rates)
    if_remote_enabled_values = _parse_csv_bools(args.if_remote_enabled)
    local_num_blocks_values = (
        _parse_csv_ints(args.local_num_blocks) if paired_block_local_values is None else None
    )
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
        block_lens=block_lens,
        paired_block_local_values=paired_block_local_values,
        dmodels=dmodels,
        learning_rates=learning_rates,
        if_remote_enabled_values=if_remote_enabled_values,
        local_num_blocks_values=local_num_blocks_values,
        train_batch_orders=train_batch_orders,
        num_codebook_vectors_values=num_codebook_vectors_values,
        num_codebook_vectors_map=num_codebook_vectors_map,
        cache_dir=args.cache_dir,
        project=args.project,
        entity=args.entity,
        max_epochs=args.max_epochs,
        metrics_white_list=metrics_white_list,
    )
    generated_path.write_text(
        _render_generated_config(
            sweep_id=launch_id_prefix,
            backend=args.backend,
            logger_backend=args.logger_backend,
            include_gdn=args.include_gdn,
            block_lens=block_lens,
            paired_block_local_values=paired_block_local_values,
            dmodels=dmodels,
            learning_rates=learning_rates,
            if_remote_enabled_values=if_remote_enabled_values,
            local_num_blocks_values=local_num_blocks_values,
            train_batch_orders=train_batch_orders,
            num_codebook_vectors_values=num_codebook_vectors_values,
            num_codebook_vectors_map=num_codebook_vectors_map,
            cache_dir=args.cache_dir,
            wandb_project=args.project,
            wandb_entity=args.entity,
            max_epochs=args.max_epochs,
            metrics_white_list=metrics_white_list,
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
