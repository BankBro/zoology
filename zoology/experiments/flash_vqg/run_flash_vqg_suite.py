import argparse
import importlib
import importlib.util
import os
import shlex
import subprocess
import sys
from datetime import UTC, datetime
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


def _parse_seed_values(raw: str) -> list[int]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("seed_values 不能为空.")

    normalized: list[int] = []
    seen: set[int] = set()
    for value in values:
        parsed = int(value)
        if parsed < 0:
            raise ValueError(f"seed 必须是非负整数, 当前收到: {value}")
        if parsed not in seen:
            normalized.append(parsed)
            seen.add(parsed)
    return normalized


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


def _parse_remote_read_topk_values(raw: str) -> list[int | None]:
    values = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("fox_remote_read_topk_values 不能为空.")

    normalized: list[int | None] = []
    seen: set[int | None] = set()
    for value in values:
        if value in {"dense", "none", "null"}:
            parsed = None
        else:
            try:
                parsed = int(value)
            except ValueError as exc:
                raise ValueError(
                    f"fox_remote_read_topk_values 只能包含 dense 或正整数, 当前收到: {value}"
                ) from exc
            if parsed <= 0:
                raise ValueError(
                    f"fox_remote_read_topk_values 只能包含 dense 或正整数, 当前收到: {value}"
                )
        if parsed not in seen:
            normalized.append(parsed)
            seen.add(parsed)
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
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d-%H-%M-%S")
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
    seed_values: list[int] | None,
    data_seed: int,
    num_codebook_vectors_values: list[int] | None,
    num_codebook_vectors_map: dict[int, int] | None,
    fox_remote_path_backend: str | None,
    fox_remote_read_topk_values: list[int | None] | None,
    fox_remote_formula: str,
    fox_clr_rank: int,
    fox_clr_use_den_residual: bool,
    fox_clr_remat_mode: str,
    fox_clr_residual_update_mode: str = "additive",
    fox_clr_residual_forget_mode: str = "global",
    fox_clr_state_write_topk: int = 4,
    fox_clr_delta_target_mode: str = "residual_to_coarse",
    fox_gd_residual_rank: int = 16,
    fox_gd_residual_write_topk: int = 4,
    fox_gd_residual_builder: str = "grouped_chunk_torch_ref",
    fox_gd_residual_pack_mode: str = "semivec_ref",
    fox_gd_residual_chunk_size: int = 64,
    fox_gd_residual_mu_min_count: float = 1.0,
    fox_gd_residual_addr_eps: float = 1e-6,
    fox_gd_residual_den_eps: float = 1e-6,
    fox_gd_residual_rho_eps: float = 1e-12,
    fox_gd_residual_beta_init: float = 0.5,
    fox_gd_residual_lambda_init: float = 0.05,
    fox_gd_residual_norm_with_gain: bool = False,
    fox_gd_residual_use_separate_addr_codebook: bool = False,
    vq_score_mode: str = "l2",
    vq_weight_mode: str = "one-hot",
    vq_update_mode: str = "ema",
    vq_softmax_tau: float = 1.0,
    vq_topk: int = 4,
    gradient_accumulation_steps: int,
    train_batch_size: int | None,
    eval_batch_size: int | None,
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
        f"    seed_values={seed_values!r},",
        f"    data_seed={data_seed!r},",
        f"    num_codebook_vectors_values={num_codebook_vectors_values!r},",
        f"    num_codebook_vectors_map={num_codebook_vectors_map!r},",
        f"    fox_remote_path_backend={fox_remote_path_backend!r},",
        f"    fox_remote_read_topk_values={fox_remote_read_topk_values!r},",
        f"    fox_remote_formula={fox_remote_formula!r},",
        f"    fox_clr_rank={fox_clr_rank!r},",
        f"    fox_clr_use_den_residual={fox_clr_use_den_residual!r},",
        f"    fox_clr_remat_mode={fox_clr_remat_mode!r},",
        f"    fox_clr_residual_update_mode={fox_clr_residual_update_mode!r},",
        f"    fox_clr_residual_forget_mode={fox_clr_residual_forget_mode!r},",
        f"    fox_clr_state_write_topk={fox_clr_state_write_topk!r},",
        f"    fox_clr_delta_target_mode={fox_clr_delta_target_mode!r},",
        f"    fox_gd_residual_rank={fox_gd_residual_rank!r},",
        f"    fox_gd_residual_write_topk={fox_gd_residual_write_topk!r},",
        f"    fox_gd_residual_builder={fox_gd_residual_builder!r},",
        f"    fox_gd_residual_pack_mode={fox_gd_residual_pack_mode!r},",
        f"    fox_gd_residual_chunk_size={fox_gd_residual_chunk_size!r},",
        f"    fox_gd_residual_mu_min_count={fox_gd_residual_mu_min_count!r},",
        f"    fox_gd_residual_addr_eps={fox_gd_residual_addr_eps!r},",
        f"    fox_gd_residual_den_eps={fox_gd_residual_den_eps!r},",
        f"    fox_gd_residual_rho_eps={fox_gd_residual_rho_eps!r},",
        f"    fox_gd_residual_beta_init={fox_gd_residual_beta_init!r},",
        f"    fox_gd_residual_lambda_init={fox_gd_residual_lambda_init!r},",
        f"    fox_gd_residual_norm_with_gain={fox_gd_residual_norm_with_gain!r},",
        "    fox_gd_residual_use_separate_addr_codebook="
        f"{fox_gd_residual_use_separate_addr_codebook!r},",
        f"    vq_score_mode={vq_score_mode!r},",
        f"    vq_weight_mode={vq_weight_mode!r},",
        f"    vq_update_mode={vq_update_mode!r},",
        f"    vq_softmax_tau={vq_softmax_tau!r},",
        f"    vq_topk={vq_topk!r},",
        f"    gradient_accumulation_steps={gradient_accumulation_steps!r},",
        f"    train_batch_size={train_batch_size!r},",
        f"    eval_batch_size={eval_batch_size!r},",
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


def _builder_args_dict(args: argparse.Namespace) -> dict:
    return {
        key: value
        for key, value in vars(args).items()
        if key not in {"eval_only", "checkpoint_launch_id", "checkpoint_run_id"}
    }


def _load_config_builder(builder_spec: str):
    if ":" not in builder_spec:
        raise ValueError("--config-builder 必须使用 <module_or_file>:<callable> 格式.")
    target, callable_name = builder_spec.split(":", maxsplit=1)
    target = target.strip()
    callable_name = callable_name.strip()
    if not target or not callable_name:
        raise ValueError("--config-builder 必须使用 <module_or_file>:<callable> 格式.")

    if target.endswith(".py") or "/" in target or target.startswith("."):
        builder_path = Path(target).expanduser().resolve()
        if not builder_path.exists():
            raise FileNotFoundError(f"未找到 config builder 文件: {builder_path}")
        module_name = f"flash_vqg_builder_{abs(hash(str(builder_path)))}"
        spec = importlib.util.spec_from_file_location(module_name, builder_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法从文件加载 config builder: {builder_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)

    builder = getattr(module, callable_name, None)
    if builder is None or not callable(builder):
        raise AttributeError(f"config builder 中未找到可调用对象: {builder_spec}")
    return builder


def _build_configs_from_builder(*, builder_spec: str, args: argparse.Namespace):
    builder = _load_config_builder(builder_spec)
    configs = builder(args)
    if not isinstance(configs, list) or not configs:
        raise ValueError("config builder 必须返回非空 list[TrainConfig].")
    return configs


def _render_generated_config_from_builder(*, builder_spec: str, builder_args: dict) -> str:
    lines = [
        "# -*- coding: utf-8 -*-",
        "# 此文件由 run_flash_vqg_suite.py 自动生成.",
        "# 如需调整实验参数, 请修改 wrapper 入参后重新生成.",
        "",
        "import argparse",
        "",
        "from zoology.experiments.flash_vqg.run_flash_vqg_suite import _load_config_builder",
        "",
        f"_builder = _load_config_builder({builder_spec!r})",
        f"_builder_args = argparse.Namespace(**{builder_args!r})",
        "configs = _builder(_builder_args)",
        "",
    ]
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
    seed_values: list[int] | None,
    data_seed: int,
    num_codebook_vectors_values: list[int] | None,
    num_codebook_vectors_map: dict[int, int] | None,
    fox_remote_path_backend: str | None,
    fox_remote_read_topk_values: list[int | None] | None,
    fox_remote_formula: str,
    fox_clr_rank: int,
    fox_clr_use_den_residual: bool,
    fox_clr_remat_mode: str,
    fox_clr_residual_update_mode: str = "additive",
    fox_clr_residual_forget_mode: str = "global",
    fox_clr_state_write_topk: int = 4,
    fox_clr_delta_target_mode: str = "residual_to_coarse",
    fox_gd_residual_rank: int = 16,
    fox_gd_residual_write_topk: int = 4,
    fox_gd_residual_builder: str = "grouped_chunk_torch_ref",
    fox_gd_residual_pack_mode: str = "semivec_ref",
    fox_gd_residual_chunk_size: int = 64,
    fox_gd_residual_mu_min_count: float = 1.0,
    fox_gd_residual_addr_eps: float = 1e-6,
    fox_gd_residual_den_eps: float = 1e-6,
    fox_gd_residual_rho_eps: float = 1e-12,
    fox_gd_residual_beta_init: float = 0.5,
    fox_gd_residual_lambda_init: float = 0.05,
    fox_gd_residual_norm_with_gain: bool = False,
    fox_gd_residual_use_separate_addr_codebook: bool = False,
    vq_score_mode: str = "l2",
    vq_weight_mode: str = "one-hot",
    vq_update_mode: str = "ema",
    vq_softmax_tau: float = 1.0,
    vq_topk: int = 4,
    gradient_accumulation_steps: int,
    train_batch_size: int | None,
    eval_batch_size: int | None,
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
        seed_values=seed_values,
        data_seed=data_seed,
        num_codebook_vectors_values=num_codebook_vectors_values,
        num_codebook_vectors_map=num_codebook_vectors_map,
        fox_remote_path_backend=fox_remote_path_backend,
        fox_remote_read_topk_values=fox_remote_read_topk_values,
        fox_remote_formula=fox_remote_formula,
        fox_clr_rank=fox_clr_rank,
        fox_clr_use_den_residual=fox_clr_use_den_residual,
        fox_clr_remat_mode=fox_clr_remat_mode,
        fox_clr_residual_update_mode=fox_clr_residual_update_mode,
        fox_clr_residual_forget_mode=fox_clr_residual_forget_mode,
        fox_clr_state_write_topk=fox_clr_state_write_topk,
        fox_clr_delta_target_mode=fox_clr_delta_target_mode,
        fox_gd_residual_rank=fox_gd_residual_rank,
        fox_gd_residual_write_topk=fox_gd_residual_write_topk,
        fox_gd_residual_builder=fox_gd_residual_builder,
        fox_gd_residual_pack_mode=fox_gd_residual_pack_mode,
        fox_gd_residual_chunk_size=fox_gd_residual_chunk_size,
        fox_gd_residual_mu_min_count=fox_gd_residual_mu_min_count,
        fox_gd_residual_addr_eps=fox_gd_residual_addr_eps,
        fox_gd_residual_den_eps=fox_gd_residual_den_eps,
        fox_gd_residual_rho_eps=fox_gd_residual_rho_eps,
        fox_gd_residual_beta_init=fox_gd_residual_beta_init,
        fox_gd_residual_lambda_init=fox_gd_residual_lambda_init,
        fox_gd_residual_norm_with_gain=fox_gd_residual_norm_with_gain,
        fox_gd_residual_use_separate_addr_codebook=(
            fox_gd_residual_use_separate_addr_codebook
        ),
        vq_score_mode=vq_score_mode,
        vq_weight_mode=vq_weight_mode,
        vq_update_mode=vq_update_mode,
        vq_softmax_tau=vq_softmax_tau,
        vq_topk=vq_topk,
        gradient_accumulation_steps=gradient_accumulation_steps,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
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
        "e5a": getattr(eval_only_module, "run_e5a_eval", None),
        "e5b": getattr(eval_only_module, "run_e5b_eval", None),
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
        choices=["e4a", "e5a", "e5b", "e7"],
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
    parser.add_argument("--seed-values", type=str, default=None)
    parser.add_argument("--data-seed", type=int, default=123)
    parser.add_argument("--cache-dir", type=str, default="./data/flash_vqg")
    parser.add_argument(
        "--fox-remote-path-backend",
        choices=["torch", "triton"],
        default=None,
        help="训练模式下 Flash-VQG remote reduce backend. 默认跟随 --backend.",
    )
    parser.add_argument(
        "--fox-remote-read-topk-values",
        type=str,
        default=None,
        help="逗号分隔的 read-side top-k 扫描, 例如 dense,2,4.",
    )
    parser.add_argument(
        "--fox-remote-formula",
        choices=["legacy", "clr_v1", "clr_delta_v1", "gd_residual_v1"],
        default="legacy",
        help="remote 分支读出公式. legacy 为当前 U/L 方案, clr_v1 为 softmax-like CLR 一阶近似, clr_delta_v1 为 Delta memory residual, gd_residual_v1 为 gated residual reference.",
    )
    parser.add_argument(
        "--fox-clr-rank",
        type=int,
        default=4,
        help="CLR v1 residual 坐标秩.",
    )
    parser.add_argument(
        "--fox-clr-use-den-residual",
        type=str,
        choices=["true", "false"],
        default="true",
        help="CLR v1 是否启用分母 residual 修正.",
    )
    parser.add_argument(
        "--fox-clr-remat-mode",
        choices=["off", "post_phase1"],
        default="off",
        help="CLR v1 materialize 路径的 remat 模式.",
    )
    parser.add_argument(
        "--fox-clr-residual-update-mode",
        choices=["additive", "delta"],
        default="additive",
        help="CLR residual state 的写入更新模式.",
    )
    parser.add_argument(
        "--fox-clr-residual-forget-mode",
        choices=["global", "code_aware"],
        default="global",
        help="CLR residual state 的 forget 模式.",
    )
    parser.add_argument(
        "--fox-clr-state-write-topk",
        type=int,
        default=4,
        help="weighted write 时 residual state 使用的 write-side top-k.",
    )
    parser.add_argument(
        "--fox-clr-delta-target-mode",
        choices=["residual_to_coarse"],
        default="residual_to_coarse",
        help="delta residual 写入目标定义.",
    )
    parser.add_argument("--fox-gd-residual-rank", type=int, default=16)
    parser.add_argument("--fox-gd-residual-write-topk", type=int, default=4)
    parser.add_argument(
        "--fox-gd-residual-builder",
        choices=["token_step_ref", "grouped_chunk_torch_ref"],
        default="grouped_chunk_torch_ref",
    )
    parser.add_argument(
        "--fox-gd-residual-pack-mode",
        choices=["loop_ref", "semivec_ref"],
        default="semivec_ref",
    )
    parser.add_argument("--fox-gd-residual-chunk-size", type=int, default=64)
    parser.add_argument("--fox-gd-residual-mu-min-count", type=float, default=1.0)
    parser.add_argument("--fox-gd-residual-addr-eps", type=float, default=1e-6)
    parser.add_argument("--fox-gd-residual-den-eps", type=float, default=1e-6)
    parser.add_argument("--fox-gd-residual-rho-eps", type=float, default=1e-12)
    parser.add_argument("--fox-gd-residual-beta-init", type=float, default=0.5)
    parser.add_argument("--fox-gd-residual-lambda-init", type=float, default=0.05)
    parser.add_argument(
        "--fox-gd-residual-norm-with-gain",
        choices=["true", "false"],
        default="false",
    )
    parser.add_argument(
        "--fox-gd-residual-use-separate-addr-codebook",
        choices=["true", "false"],
        default="false",
    )
    parser.add_argument(
        "--vq-score-mode",
        choices=["l2", "attn_dot", "mlp", "codebook_dot"],
        default="l2",
        help="VQ write-side 打分模式.",
    )
    parser.add_argument(
        "--vq-weight-mode",
        choices=["one-hot", "dense_softmax", "topk_softmax"],
        default="one-hot",
        help="VQ write-side 分配模式.",
    )
    parser.add_argument(
        "--vq-update-mode",
        choices=["ema", "grad"],
        default="ema",
        help="VQ codebook 更新模式.",
    )
    parser.add_argument(
        "--vq-softmax-tau",
        type=float,
        default=1.0,
        help="Routing VQ softmax temperature.",
    )
    parser.add_argument(
        "--vq-topk",
        type=int,
        default=4,
        help="Routing VQ top-k write mode 的 k 值.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="每多少个 train micro-batches 做一次 optimizer step.",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=None,
        help="覆盖默认 train micro-batch 大小.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="覆盖默认 eval micro-batch 大小.",
    )
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
    parser.add_argument(
        "--config-builder",
        type=str,
        default=None,
        help="实验专用预组装配置入口, 格式为 <module_or_file>:<callable>.",
    )
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

    if args.logger_backend == "none" and args.analysis != "off":
        raise ValueError(
            "logger_backend='none' 时无法生成结构化 metrics, analysis 阶段将无法工作. "
            "请设置一个 logger backend (如 --logger-backend swanlab), "
            "或设置 --analysis off 跳过 analysis."
        )

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
    seed_values = (
        _parse_seed_values(args.seed_values)
        if args.seed_values is not None
        else None
    )
    fox_remote_read_topk_values = (
        _parse_remote_read_topk_values(args.fox_remote_read_topk_values)
        if args.fox_remote_read_topk_values is not None
        else None
    )
    launch_id_prefix = _normalize_launch_id_prefix(args.launch_id_prefix)
    launch_id = _build_launch_id(launch_id_prefix)

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    generated_launch_dir = GENERATED_DIR / launch_id
    generated_launch_dir.mkdir(parents=True, exist_ok=True)
    generated_path = generated_launch_dir / "launch_configs.py"
    manifest_path = generated_launch_dir / "manifest.json"
    if args.config_builder is not None:
        configs = _build_configs_from_builder(builder_spec=args.config_builder, args=args)
        run_ids = [config.run_id for config in configs]
        generated_path.write_text(
            _render_generated_config_from_builder(
                builder_spec=args.config_builder,
                builder_args=_builder_args_dict(args),
            ),
            encoding="utf-8",
        )
    else:
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
            seed_values=seed_values,
            data_seed=args.data_seed,
            num_codebook_vectors_values=num_codebook_vectors_values,
            num_codebook_vectors_map=num_codebook_vectors_map,
            fox_remote_path_backend=args.fox_remote_path_backend,
            fox_remote_read_topk_values=fox_remote_read_topk_values,
            fox_remote_formula=args.fox_remote_formula,
            fox_clr_rank=args.fox_clr_rank,
            fox_clr_use_den_residual=(args.fox_clr_use_den_residual == "true"),
            fox_clr_remat_mode=args.fox_clr_remat_mode,
            fox_clr_residual_update_mode=args.fox_clr_residual_update_mode,
            fox_clr_residual_forget_mode=args.fox_clr_residual_forget_mode,
            fox_clr_state_write_topk=args.fox_clr_state_write_topk,
            fox_clr_delta_target_mode=args.fox_clr_delta_target_mode,
            fox_gd_residual_rank=args.fox_gd_residual_rank,
            fox_gd_residual_write_topk=args.fox_gd_residual_write_topk,
            fox_gd_residual_builder=args.fox_gd_residual_builder,
            fox_gd_residual_pack_mode=args.fox_gd_residual_pack_mode,
            fox_gd_residual_chunk_size=args.fox_gd_residual_chunk_size,
            fox_gd_residual_mu_min_count=args.fox_gd_residual_mu_min_count,
            fox_gd_residual_addr_eps=args.fox_gd_residual_addr_eps,
            fox_gd_residual_den_eps=args.fox_gd_residual_den_eps,
            fox_gd_residual_rho_eps=args.fox_gd_residual_rho_eps,
            fox_gd_residual_beta_init=args.fox_gd_residual_beta_init,
            fox_gd_residual_lambda_init=args.fox_gd_residual_lambda_init,
            fox_gd_residual_norm_with_gain=(args.fox_gd_residual_norm_with_gain == "true"),
            fox_gd_residual_use_separate_addr_codebook=(
                args.fox_gd_residual_use_separate_addr_codebook == "true"
            ),
            vq_score_mode=args.vq_score_mode,
            vq_weight_mode=args.vq_weight_mode,
            vq_update_mode=args.vq_update_mode,
            vq_softmax_tau=args.vq_softmax_tau,
            vq_topk=args.vq_topk,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
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
                seed_values=seed_values,
                data_seed=args.data_seed,
                num_codebook_vectors_values=num_codebook_vectors_values,
                num_codebook_vectors_map=num_codebook_vectors_map,
                fox_remote_path_backend=args.fox_remote_path_backend,
                fox_remote_read_topk_values=fox_remote_read_topk_values,
                fox_remote_formula=args.fox_remote_formula,
                fox_clr_rank=args.fox_clr_rank,
                fox_clr_use_den_residual=(args.fox_clr_use_den_residual == "true"),
                fox_clr_remat_mode=args.fox_clr_remat_mode,
                fox_clr_residual_update_mode=args.fox_clr_residual_update_mode,
                fox_clr_residual_forget_mode=args.fox_clr_residual_forget_mode,
                fox_clr_state_write_topk=args.fox_clr_state_write_topk,
                fox_clr_delta_target_mode=args.fox_clr_delta_target_mode,
                fox_gd_residual_rank=args.fox_gd_residual_rank,
                fox_gd_residual_write_topk=args.fox_gd_residual_write_topk,
                fox_gd_residual_builder=args.fox_gd_residual_builder,
                fox_gd_residual_pack_mode=args.fox_gd_residual_pack_mode,
                fox_gd_residual_chunk_size=args.fox_gd_residual_chunk_size,
                fox_gd_residual_mu_min_count=args.fox_gd_residual_mu_min_count,
                fox_gd_residual_addr_eps=args.fox_gd_residual_addr_eps,
                fox_gd_residual_den_eps=args.fox_gd_residual_den_eps,
                fox_gd_residual_rho_eps=args.fox_gd_residual_rho_eps,
                fox_gd_residual_beta_init=args.fox_gd_residual_beta_init,
                fox_gd_residual_lambda_init=args.fox_gd_residual_lambda_init,
                fox_gd_residual_norm_with_gain=(args.fox_gd_residual_norm_with_gain == "true"),
                fox_gd_residual_use_separate_addr_codebook=(
                    args.fox_gd_residual_use_separate_addr_codebook == "true"
                ),
                vq_score_mode=args.vq_score_mode,
                vq_weight_mode=args.vq_weight_mode,
                vq_update_mode=args.vq_update_mode,
                vq_softmax_tau=args.vq_softmax_tau,
                vq_topk=args.vq_topk,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                train_batch_size=args.train_batch_size,
                eval_batch_size=args.eval_batch_size,
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
