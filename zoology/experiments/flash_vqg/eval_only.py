from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from zoology.checkpoints import load_checkpoint
from zoology.config import LoggerConfig
from zoology.data.multiquery_ar import MQARConfig
from zoology.data.utils import DataSegment, _SyntheticDataset
from zoology.experiments.flash_vqg.manifest import update_manifest_for_run
from zoology.logger import build_logger
from zoology.train import Trainer


RESULTS_ROOT = Path(__file__).resolve().parents[2] / "analysis" / "flash_vqg" / "results"
GENERATED_ROOT = Path(__file__).resolve().parent / "generated"
E5A_SCRIPT_PATH = (
    Path(__file__).resolve().parent
    / "scripts"
    / "20260402-clr-v1-mainline"
    / "e5a-top2-audit"
    / "e5a_audit.py"
)
E5B_SCRIPT_PATH = (
    Path(__file__).resolve().parent
    / "scripts"
    / "20260402-clr-v1-mainline"
    / "e5a-top2-audit"
    / "e5b_partial_override.py"
)

E4A_LENGTH_PRIMARY_CASES = [(64, 16), (128, 16), (256, 16), (512, 16), (1024, 16)]
E4A_LENGTH_AUX_CASES = [(64, 8), (128, 8), (256, 8), (512, 8), (1024, 8)]
E4A_CAPACITY_PRIMARY_CASES = [(256, 4), (256, 8), (256, 16), (256, 32), (256, 64), (256, 128)]
E5A_SMOKE_CASES = [(512, 128), (1024, 256)]
E7_READ_MODES = [("dense", None), ("top2", 2), ("top4", 4)]
_E5A_RUNNER = None
_E5B_RUNNER = None


def generated_launch_dir(launch_id: str) -> Path:
    return GENERATED_ROOT / launch_id


def _load_e5a_audit_runner():
    global _E5A_RUNNER
    if _E5A_RUNNER is not None:
        return _E5A_RUNNER

    module_name = "flash_vqg_e5a_audit"
    if not E5A_SCRIPT_PATH.exists():
        raise FileNotFoundError(f"未找到 E5A 审计脚本: {E5A_SCRIPT_PATH}")
    spec = importlib.util.spec_from_file_location(module_name, E5A_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法从 {E5A_SCRIPT_PATH} 加载 E5A 审计模块.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    _E5A_RUNNER = getattr(module, "run_e5a_audit")
    return _E5A_RUNNER


def _load_e5b_partial_override_runner():
    global _E5B_RUNNER
    if _E5B_RUNNER is not None:
        return _E5B_RUNNER

    module_name = "flash_vqg_e5b_partial_override"
    if not E5B_SCRIPT_PATH.exists():
        raise FileNotFoundError(f"未找到 E5B 审计脚本: {E5B_SCRIPT_PATH}")
    spec = importlib.util.spec_from_file_location(module_name, E5B_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法从 {E5B_SCRIPT_PATH} 加载 E5B 审计模块.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    _E5B_RUNNER = getattr(module, "run_e5b_partial_override")
    return _E5B_RUNNER


def manifest_path_for_launch(launch_id: str) -> Path:
    return generated_launch_dir(launch_id) / "manifest.json"


def load_manifest(launch_id: str) -> dict[str, Any]:
    path = manifest_path_for_launch(launch_id)
    if not path.exists():
        raise FileNotFoundError(f"未找到 manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_run_entry(manifest: dict[str, Any], run_id: str) -> dict[str, Any]:
    for run in manifest.get("runs", []):
        if run.get("run_id") == run_id:
            return run
    raise ValueError(f"launch_id={manifest.get('launch_id')} 中未找到 run_id={run_id}.")


def resolve_best_checkpoint_from_manifest(manifest: dict[str, Any], run_id: str) -> Path:
    run_entry = _resolve_run_entry(manifest, run_id)
    local_info = run_entry.get("local") or {}
    checkpoint_path_raw = local_info.get("best_checkpoint")
    if not checkpoint_path_raw:
        raise ValueError(
            "manifest 中缺少 `local.best_checkpoint`. 该 launch 可能是旧 manifest, "
            "请使用扩展后的新 run 重新生成 manifest."
        )
    checkpoint_path = Path(checkpoint_path_raw)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"manifest 记录的 checkpoint 不存在: {checkpoint_path}")
    return checkpoint_path


def e4a_test_cases() -> dict[str, list[tuple[int, int]]]:
    return {
        "length_axis_primary": E4A_LENGTH_PRIMARY_CASES,
        "length_axis_aux": E4A_LENGTH_AUX_CASES,
        "capacity_axis_primary": E4A_CAPACITY_PRIMARY_CASES,
    }


def build_eval_run_id(checkpoint_run_id: str) -> str:
    return f"eval_{checkpoint_run_id}"


def build_e7_eval_run_id(checkpoint_run_id: str, mode_name: str) -> str:
    return f"eval_e7_{mode_name}_{checkpoint_run_id}"


def _unique_cases_preserve_order(cases: list[tuple[int, int]]) -> list[tuple[int, int]]:
    unique_cases: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for case in cases:
        if case in seen:
            continue
        unique_cases.append(case)
        seen.add(case)
    return unique_cases


def _build_mqar_eval_configs(template: MQARConfig, cases: list[tuple[int, int]]) -> list[MQARConfig]:
    base_payload = template.model_dump()
    return [
        MQARConfig(
            **{
                **base_payload,
                "input_seq_len": int(input_seq_len),
                "num_kv_pairs": int(num_kv_pairs),
            }
        )
        for input_seq_len, num_kv_pairs in cases
    ]


def _is_valid_mqar_case(template: MQARConfig, input_seq_len: int, num_kv_pairs: int) -> bool:
    num_passes = int(getattr(template, "num_passes", 1))
    return (int(num_kv_pairs) * 2 * num_passes) + (int(num_kv_pairs) * 2) <= int(input_seq_len)


def _resolve_template_segment(bundle_config) -> MQARConfig:
    candidates = list(bundle_config.data.test_configs) + list(bundle_config.data.train_configs)
    if not candidates:
        raise ValueError("checkpoint config 中没有任何数据 segment, 无法构造 E4-A 测试集.")

    template = candidates[0]
    if not isinstance(template, MQARConfig):
        raise TypeError(f"E4-A 目前只支持 MQARConfig, 当前收到: {type(template).__name__}")
    if bundle_config.input_type != "discrete":
        raise ValueError(f"E4-A 目前只支持 discrete 输入, 当前收到: {bundle_config.input_type}")
    return template


def _prepare_test_dataloader_from_data_config(data_config) -> DataLoader:
    if isinstance(data_config.batch_size, int):
        _, test_batch_size = (data_config.batch_size, data_config.batch_size)
    else:
        _, test_batch_size = data_config.batch_size

    max_seed = 2**32
    np.random.seed(data_config.seed)
    _ = np.random.randint(0, max_seed // 2, size=len(data_config.train_configs))
    test_seeds = np.random.randint(max_seed // 2, max_seed, size=len(data_config.test_configs))
    factory_kwargs = {"cache_dir": data_config.cache_dir, "force_cache": data_config.force_cache}
    test_segments = _SyntheticDataset(
        [
            DataSegment.from_config(segment_config, seed=int(seed), **factory_kwargs)
            for segment_config, seed in zip(data_config.test_configs, test_seeds)
        ],
        batch_size=test_batch_size,
    )
    return DataLoader(test_segments, batch_size=None, num_workers=0, shuffle=False)


def _build_eval_config(
    *,
    source_config,
    eval_launch_id: str,
    eval_sweep_id: str,
    eval_run_id: str,
    logger_backend: str,
    project: str | None,
    entity: str | None,
    valid_cases: list[tuple[int, int]],
    template: MQARConfig,
    metrics_white_list: list[str] | None,
):
    eval_config = source_config.model_copy(deep=True)
    eval_config.launch_id = eval_launch_id
    eval_config.sweep_id = eval_sweep_id
    eval_config.run_id = eval_run_id
    eval_config.logger = LoggerConfig(
        backend=logger_backend,
        project_name=project,
        entity=entity,
    )
    eval_config.checkpoint = source_config.checkpoint.model_copy(deep=True)
    eval_config.checkpoint.enabled = False
    eval_config.data = source_config.data.model_copy(deep=True)
    eval_config.data.test_configs = _build_mqar_eval_configs(template, valid_cases)
    if metrics_white_list is not None:
        eval_config.metrics_white_list = list(metrics_white_list)
    return eval_config


def _build_passthrough_eval_config(
    *,
    source_config,
    eval_launch_id: str,
    eval_sweep_id: str,
    eval_run_id: str,
    logger_backend: str,
    project: str | None,
    entity: str | None,
    metrics_white_list: list[str] | None,
):
    eval_config = source_config.model_copy(deep=True)
    eval_config.launch_id = eval_launch_id
    eval_config.sweep_id = eval_sweep_id
    eval_config.run_id = eval_run_id
    eval_config.logger = LoggerConfig(
        backend=logger_backend,
        project_name=project,
        entity=entity,
    )
    eval_config.checkpoint = source_config.checkpoint.model_copy(deep=True)
    eval_config.checkpoint.enabled = False
    eval_config.data = source_config.data.model_copy(deep=True)
    if metrics_white_list is not None:
        eval_config.metrics_white_list = list(metrics_white_list)
    return eval_config


class _PrefixedLogger:
    def __init__(self, base_logger, *, prefix: str, step_offset: int):
        self.base_logger = base_logger
        self.prefix = prefix
        self.step_offset = int(step_offset)

    def _patched_metrics_white_list(self) -> list[str] | None:
        config = getattr(self.base_logger, "config", None)
        if config is None or not hasattr(config, "metrics_white_list"):
            return None

        raw_patterns = getattr(config, "metrics_white_list", None)
        if not raw_patterns:
            return []

        prefixed_patterns: list[str] = []
        seen: set[str] = set()
        for pattern in raw_patterns:
            value = str(pattern).strip()
            if not value:
                continue
            prefixed = f"{self.prefix}{value}"
            if prefixed in seen:
                continue
            prefixed_patterns.append(prefixed)
            seen.add(prefixed)
        return prefixed_patterns

    def log(self, metrics: dict[str, float | int], *, step: int | None = None):
        prefixed = {
            f"{self.prefix}{key}": value
            for key, value in metrics.items()
            if key != "epoch"
        }
        resolved_step = self.step_offset if step is None else self.step_offset + int(step)
        config = getattr(self.base_logger, "config", None)
        if config is None or not hasattr(config, "metrics_white_list"):
            self.base_logger.log(prefixed, step=resolved_step)
            return

        original_patterns = getattr(config, "metrics_white_list", None)
        setattr(config, "metrics_white_list", self._patched_metrics_white_list())
        try:
            self.base_logger.log(prefixed, step=resolved_step)
        finally:
            setattr(config, "metrics_white_list", original_patterns)


def _iter_flash_vqg_remote_configs(model):
    for module in model.modules():
        cfg = getattr(module, "config", None)
        if cfg is None:
            continue
        required_attrs = (
            "if_remote_enabled",
            "fox_remote_path_backend",
            "local_num_blocks",
            "num_codebook_vectors",
        )
        if not all(hasattr(cfg, attr) for attr in required_attrs):
            continue
        yield cfg


def _override_flash_vqg_remote_read(model, *, read_topk: int | None) -> dict[str, int]:
    flash_layers = 0
    remote_enabled_layers = 0
    for cfg in _iter_flash_vqg_remote_configs(model):
        flash_layers += 1
        if not bool(getattr(cfg, "if_remote_enabled", False)):
            continue
        remote_enabled_layers += 1
        cfg.fox_remote_path_backend = "torch"
        cfg.fox_remote_read_topk = None if read_topk is None else int(read_topk)
    return {
        "flash_layers": flash_layers,
        "remote_enabled_layers": remote_enabled_layers,
    }


def _evaluate_metrics(bundle: dict[str, Any], eval_config, test_dataloader: DataLoader, logger) -> dict[str, float]:
    device = next(bundle["model"].parameters()).device
    task = Trainer(
        model=bundle["model"],
        train_dataloader=test_dataloader,
        test_dataloader=test_dataloader,
        input_type=eval_config.input_type,
        max_epochs=1,
        learning_rate=eval_config.learning_rate,
        weight_decay=eval_config.weight_decay,
        early_stopping_metric=eval_config.early_stopping_metric,
        early_stopping_threshold=eval_config.early_stopping_threshold,
        loss_type=eval_config.loss_type,
        slice_keys=eval_config.slice_keys,
        device=str(device),
        logger=logger,
        checkpoint_manager=None,
    )
    task.loss_fn = nn.CrossEntropyLoss()
    return task.test(epoch_idx=0)


def _table_from_metrics(
    *,
    cases: list[tuple[int, int]],
    metrics: dict[str, float],
    axis_name: str,
) -> pd.DataFrame:
    rows = []
    for input_seq_len, num_kv_pairs in cases:
        metric_name = f"valid/mqar_case/accuracy-{input_seq_len}x{num_kv_pairs}"
        rows.append(
            {
                "axis_name": axis_name,
                "input_seq_len": input_seq_len,
                "num_kv_pairs": num_kv_pairs,
                "accuracy": float(metrics.get(metric_name, float("nan"))),
            }
        )
    return pd.DataFrame(rows)


def _plot_axis_table(df: pd.DataFrame, *, x_key: str, title: str, output_path: Path):
    plt.figure(figsize=(6, 4))
    plt.plot(df[x_key], df["accuracy"], marker="o")
    plt.title(title)
    plt.xlabel(x_key)
    plt.ylabel("accuracy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def _plot_e7_metric_compare(
    metrics_df: pd.DataFrame,
    *,
    metric_key: str,
    title: str,
    output_path: Path,
):
    if metric_key not in metrics_df.columns:
        return False
    plot_df = metrics_df[["mode", metric_key]].copy()
    plot_df = plot_df.dropna(subset=[metric_key])
    if plot_df.empty:
        return False

    plt.figure(figsize=(6, 4))
    plt.bar(plot_df["mode"], plot_df[metric_key])
    plt.title(title)
    plt.xlabel("mode")
    plt.ylabel(metric_key)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    return True


def _sanitize_metric_filename(metric: str) -> str:
    return (
        str(metric)
        .replace("/", "__")
        .replace("\\", "__")
        .replace(":", "_")
        .replace("*", "_")
        .replace("?", "_")
        .replace('"', "_")
        .replace("<", "_")
        .replace(">", "_")
        .replace("|", "_")
        .replace(".", "_")
    )


def _write_local_e4a_outputs(
    *,
    output_dir: Path,
    checkpoint_launch_id: str,
    checkpoint_run_id: str,
    checkpoint_path: Path,
    eval_launch_id: str,
    eval_run_id: str,
    skipped_invalid_cases: list[tuple[int, int]],
    metrics: dict[str, float],
):
    data_dir = output_dir / "data"
    pics_dir = output_dir / "pics"
    data_dir.mkdir(parents=True, exist_ok=True)
    pics_dir.mkdir(parents=True, exist_ok=True)

    length_primary_df = _table_from_metrics(
        cases=E4A_LENGTH_PRIMARY_CASES,
        metrics=metrics,
        axis_name="length_axis_primary",
    )
    length_aux_df = _table_from_metrics(
        cases=E4A_LENGTH_AUX_CASES,
        metrics=metrics,
        axis_name="length_axis_aux",
    )
    capacity_primary_df = _table_from_metrics(
        cases=E4A_CAPACITY_PRIMARY_CASES,
        metrics=metrics,
        axis_name="capacity_axis_primary",
    )

    length_primary_df.to_csv(data_dir / "e4a_length_axis_primary.csv", index=False)
    length_aux_df.to_csv(data_dir / "e4a_length_axis_aux.csv", index=False)
    capacity_primary_df.to_csv(data_dir / "e4a_capacity_axis_primary.csv", index=False)

    _plot_axis_table(
        length_primary_df,
        x_key="input_seq_len",
        title="E4-A Length Axis Primary",
        output_path=pics_dir / "e4a_length_axis_primary.png",
    )
    _plot_axis_table(
        length_aux_df,
        x_key="input_seq_len",
        title="E4-A Length Axis Aux",
        output_path=pics_dir / "e4a_length_axis_aux.png",
    )
    _plot_axis_table(
        capacity_primary_df,
        x_key="num_kv_pairs",
        title="E4-A Capacity Axis Primary",
        output_path=pics_dir / "e4a_capacity_axis_primary.png",
    )

    summary = {
        "checkpoint_launch_id": checkpoint_launch_id,
        "checkpoint_run_id": checkpoint_run_id,
        "checkpoint_path": str(checkpoint_path),
        "eval_launch_id": eval_launch_id,
        "eval_run_id": eval_run_id,
        "output_dir": str(output_dir.resolve()),
        "skipped_invalid_cases": skipped_invalid_cases,
        "metrics": metrics,
    }
    (data_dir / "e4a_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _write_local_e7_outputs(
    *,
    output_dir: Path,
    checkpoint_launch_id: str,
    checkpoint_run_id: str,
    checkpoint_path: Path,
    eval_launch_id: str,
    eval_run_id: str,
    layer_override: dict[str, int],
    mode_rows: list[dict[str, Any]],
):
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame(mode_rows)
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    generated_plots: list[str] = []
    for metric_key, title in [
        ("valid/accuracy", "E7 Accuracy Compare"),
        ("valid/loss", "E7 Loss Compare"),
        ("valid/attn/den_cache_ratio", "E7 Den Cache Ratio Compare"),
        (
            "valid/attn/o_remote_energy_ratio",
            "E7 Remote Energy Ratio Compare",
        ),
    ]:
        filename = f"{_sanitize_metric_filename(metric_key)}.png"
        if _plot_e7_metric_compare(
            metrics_df,
            metric_key=metric_key,
            title=title,
            output_path=output_dir / filename,
        ):
            generated_plots.append(filename)

    summary = {
        "eval_task": "e7",
        "checkpoint_launch_id": checkpoint_launch_id,
        "checkpoint_run_id": checkpoint_run_id,
        "checkpoint_path": str(checkpoint_path),
        "eval_launch_id": eval_launch_id,
        "eval_run_id": eval_run_id,
        "output_dir": str(output_dir.resolve()),
        "backend_override": "torch",
        "layer_override": layer_override,
        "generated_plots": generated_plots,
        "modes": [
            {
                "mode": row["mode"],
                "read_topk": row["read_topk"],
                "metrics": {
                    key: value
                    for key, value in row.items()
                    if key not in {"mode", "read_topk"}
                },
            }
            for row in mode_rows
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def run_e4a_eval(
    *,
    checkpoint_launch_id: str,
    checkpoint_run_id: str,
    eval_launch_id: str,
    eval_sweep_id: str,
    eval_run_id: str,
    logger_backend: str,
    project: str | None,
    entity: str | None,
    manifest_path: Path | None = None,
    metrics_white_list: list[str] | None = None,
) -> dict[str, Any]:
    source_manifest = load_manifest(checkpoint_launch_id)
    checkpoint_path = resolve_best_checkpoint_from_manifest(source_manifest, checkpoint_run_id)
    bundle = load_checkpoint(
        checkpoint_path,
        which="best",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    template = _resolve_template_segment(bundle["config"])
    requested_cases = _unique_cases_preserve_order(
        E4A_LENGTH_PRIMARY_CASES + E4A_LENGTH_AUX_CASES + E4A_CAPACITY_PRIMARY_CASES
    )
    valid_cases = [case for case in requested_cases if _is_valid_mqar_case(template, *case)]
    skipped_invalid_cases = [case for case in requested_cases if case not in valid_cases]

    eval_config = _build_eval_config(
        source_config=bundle["config"],
        eval_launch_id=eval_launch_id,
        eval_sweep_id=eval_sweep_id,
        eval_run_id=eval_run_id,
        logger_backend=logger_backend,
        project=project,
        entity=entity,
        valid_cases=valid_cases,
        template=template,
        metrics_white_list=metrics_white_list,
    )
    eval_source = {
        "checkpoint_launch_id": checkpoint_launch_id,
        "checkpoint_run_id": checkpoint_run_id,
        "best_checkpoint": str(checkpoint_path),
    }

    logger = None
    try:
        logger = build_logger(eval_config)
        logger.log_config(eval_config)
        logger.log_model(bundle["model"], config=eval_config)
        update_manifest_for_run(
            config=eval_config,
            logger_summary=logger.get_summary(),
            status="running",
            manifest_path=manifest_path,
            eval_source=eval_source,
        )

        test_dataloader = _prepare_test_dataloader_from_data_config(eval_config.data)
        metrics = _evaluate_metrics(bundle, eval_config, test_dataloader, logger)
        output_dir = RESULTS_ROOT / eval_launch_id / eval_run_id
        summary = _write_local_e4a_outputs(
            output_dir=output_dir,
            checkpoint_launch_id=checkpoint_launch_id,
            checkpoint_run_id=checkpoint_run_id,
            checkpoint_path=checkpoint_path,
            eval_launch_id=eval_launch_id,
            eval_run_id=eval_run_id,
            skipped_invalid_cases=skipped_invalid_cases,
            metrics=metrics,
        )
        update_manifest_for_run(
            config=eval_config,
            logger_summary=logger.get_summary(),
            status="completed",
            manifest_path=manifest_path,
            eval_source=eval_source,
        )
        return summary
    except Exception as exc:
        if logger is not None:
            update_manifest_for_run(
                config=eval_config,
                logger_summary=logger.get_summary(),
                status="failed",
                error=str(exc),
                manifest_path=manifest_path,
                eval_source=eval_source,
            )
        raise
    finally:
        if logger is not None:
            logger.finish()


def run_e7_eval(
    *,
    checkpoint_launch_id: str,
    checkpoint_run_id: str,
    eval_launch_id: str,
    eval_sweep_id: str,
    eval_run_ids: list[str] | None = None,
    logger_backend: str,
    project: str | None,
    entity: str | None,
    manifest_path: Path | None = None,
    metrics_white_list: list[str] | None = None,
) -> dict[str, Any]:
    source_manifest = load_manifest(checkpoint_launch_id)
    checkpoint_path = resolve_best_checkpoint_from_manifest(source_manifest, checkpoint_run_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    first_bundle = load_checkpoint(checkpoint_path, which="best", device=device)
    layer_override = _override_flash_vqg_remote_read(first_bundle["model"], read_topk=None)
    if layer_override["flash_layers"] == 0 or layer_override["remote_enabled_layers"] == 0:
        raise ValueError("E7 只支持包含已启用 remote 分支的 Flash-VQG 模型.")
    resolved_run_ids = (
        list(eval_run_ids)
        if eval_run_ids is not None
        else [build_e7_eval_run_id(checkpoint_run_id, mode_name) for mode_name, _ in E7_READ_MODES]
    )
    if len(resolved_run_ids) != len(E7_READ_MODES):
        raise ValueError(
            f"E7 需要 {len(E7_READ_MODES)} 个 eval_run_id, 当前收到 {len(resolved_run_ids)} 个."
        )
    eval_source = {
        "checkpoint_launch_id": checkpoint_launch_id,
        "checkpoint_run_id": checkpoint_run_id,
        "best_checkpoint": str(checkpoint_path),
    }
    test_dataloader = _prepare_test_dataloader_from_data_config(first_bundle["config"].data)
    mode_rows: list[dict[str, Any]] = []

    for mode_idx, ((mode_name, read_topk), eval_run_id) in enumerate(zip(E7_READ_MODES, resolved_run_ids)):
        bundle = (
            first_bundle
            if mode_idx == 0
            else load_checkpoint(checkpoint_path, which="best", device=device)
        )
        current_override = _override_flash_vqg_remote_read(bundle["model"], read_topk=read_topk)
        if current_override["remote_enabled_layers"] == 0:
            raise ValueError("E7 评测目标没有启用 remote 分支, 无法执行 read-side top-k probe.")

        eval_config = _build_passthrough_eval_config(
            source_config=bundle["config"],
            eval_launch_id=eval_launch_id,
            eval_sweep_id=eval_sweep_id,
            eval_run_id=eval_run_id,
            logger_backend=logger_backend,
            project=project,
            entity=entity,
            metrics_white_list=metrics_white_list,
        )

        logger = None
        try:
            logger = build_logger(eval_config)
            logger.log_config(eval_config)
            logger.log_model(bundle["model"], config=eval_config)
            update_manifest_for_run(
                config=eval_config,
                logger_summary=logger.get_summary(),
                status="running",
                manifest_path=manifest_path,
                eval_source=eval_source,
            )

            metrics = _evaluate_metrics(bundle, eval_config, test_dataloader, logger)
            row: dict[str, Any] = {
                "mode": mode_name,
                "read_topk": read_topk,
                "eval_run_id": eval_run_id,
            }
            row.update({key: float(value) for key, value in metrics.items()})
            mode_rows.append(row)

            update_manifest_for_run(
                config=eval_config,
                logger_summary=logger.get_summary(),
                status="completed",
                manifest_path=manifest_path,
                eval_source=eval_source,
            )
        except Exception as exc:
            if logger is not None:
                update_manifest_for_run(
                    config=eval_config,
                    logger_summary=logger.get_summary(),
                    status="failed",
                    error=str(exc),
                    manifest_path=manifest_path,
                    eval_source=eval_source,
                )
            raise
        finally:
            if logger is not None:
                logger.finish()

    compare_output_dir = RESULTS_ROOT / eval_launch_id / "launch_analysis"
    compare_summary = _write_local_e7_outputs(
        output_dir=compare_output_dir,
        checkpoint_launch_id=checkpoint_launch_id,
        checkpoint_run_id=checkpoint_run_id,
        checkpoint_path=checkpoint_path,
        eval_launch_id=eval_launch_id,
        eval_run_id="launch_analysis",
        layer_override=layer_override,
        mode_rows=mode_rows,
    )
    compare_summary["output_dir"] = str((RESULTS_ROOT / eval_launch_id).resolve())
    compare_summary["compare_output_dir"] = str(compare_output_dir.resolve())
    compare_summary["mode_run_ids"] = {
        row["mode"]: row["eval_run_id"]
        for row in mode_rows
    }
    return compare_summary


def run_e5a_eval(
    *,
    checkpoint_launch_id: str,
    checkpoint_run_id: str,
    eval_launch_id: str,
    eval_sweep_id: str,
    eval_run_id: str,
    logger_backend: str,
    project: str | None,
    entity: str | None,
    manifest_path: Path | None = None,
    metrics_white_list: list[str] | None = None,
) -> dict[str, Any]:
    run_e5a_audit = _load_e5a_audit_runner()

    source_manifest = load_manifest(checkpoint_launch_id)
    checkpoint_path = resolve_best_checkpoint_from_manifest(source_manifest, checkpoint_run_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = load_checkpoint(checkpoint_path, which="best", device=device)

    template = _resolve_template_segment(bundle["config"])
    valid_cases = [case for case in E5A_SMOKE_CASES if _is_valid_mqar_case(template, *case)]
    if len(valid_cases) != len(E5A_SMOKE_CASES):
        raise ValueError(f"E5A 固定 smoke case 不完整, 当前可用 case={valid_cases}.")

    eval_config = _build_eval_config(
        source_config=bundle["config"],
        eval_launch_id=eval_launch_id,
        eval_sweep_id=eval_sweep_id,
        eval_run_id=eval_run_id,
        logger_backend=logger_backend,
        project=project,
        entity=entity,
        valid_cases=valid_cases,
        template=template,
        metrics_white_list=metrics_white_list,
    )
    eval_source = {
        "checkpoint_launch_id": checkpoint_launch_id,
        "checkpoint_run_id": checkpoint_run_id,
        "best_checkpoint": str(checkpoint_path),
    }
    logger = None
    try:
        logger = build_logger(eval_config)
        logger.log_config(eval_config)
        logger.log_model(bundle["model"], config=eval_config)
        update_manifest_for_run(
            config=eval_config,
            logger_summary=logger.get_summary(),
            status="running",
            manifest_path=manifest_path,
            eval_source=eval_source,
        )

        test_dataloader = _prepare_test_dataloader_from_data_config(eval_config.data)
        output_dir = RESULTS_ROOT / eval_launch_id / eval_run_id / "e5a_outputs"
        summary = run_e5a_audit(
            bundle=bundle,
            test_dataloader=test_dataloader,
            checkpoint_launch_id=checkpoint_launch_id,
            checkpoint_run_id=checkpoint_run_id,
            checkpoint_path=checkpoint_path,
            eval_launch_id=eval_launch_id,
            eval_run_id=eval_run_id,
            output_dir=output_dir,
            logger=logger,
        )
        update_manifest_for_run(
            config=eval_config,
            logger_summary=logger.get_summary(),
            status="completed",
            manifest_path=manifest_path,
            eval_source=eval_source,
        )
        return summary
    except Exception as exc:
        if logger is not None:
            update_manifest_for_run(
                config=eval_config,
                logger_summary=logger.get_summary(),
                status="failed",
                error=str(exc),
                manifest_path=manifest_path,
                eval_source=eval_source,
            )
        raise
    finally:
        if logger is not None:
            logger.finish()


def run_e5b_eval(
    *,
    checkpoint_launch_id: str,
    checkpoint_run_id: str,
    eval_launch_id: str,
    eval_sweep_id: str,
    eval_run_id: str,
    logger_backend: str,
    project: str | None,
    entity: str | None,
    manifest_path: Path | None = None,
    metrics_white_list: list[str] | None = None,
) -> dict[str, Any]:
    run_e5b_partial_override = _load_e5b_partial_override_runner()

    source_manifest = load_manifest(checkpoint_launch_id)
    checkpoint_path = resolve_best_checkpoint_from_manifest(source_manifest, checkpoint_run_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = load_checkpoint(checkpoint_path, which="best", device=device)

    template = _resolve_template_segment(bundle["config"])
    valid_cases = [case for case in E5A_SMOKE_CASES if _is_valid_mqar_case(template, *case)]
    if len(valid_cases) != len(E5A_SMOKE_CASES):
        raise ValueError(f"E5B 固定 smoke case 不完整, 当前可用 case={valid_cases}.")

    eval_config = _build_eval_config(
        source_config=bundle["config"],
        eval_launch_id=eval_launch_id,
        eval_sweep_id=eval_sweep_id,
        eval_run_id=eval_run_id,
        logger_backend=logger_backend,
        project=project,
        entity=entity,
        valid_cases=valid_cases,
        template=template,
        metrics_white_list=metrics_white_list,
    )
    eval_source = {
        "checkpoint_launch_id": checkpoint_launch_id,
        "checkpoint_run_id": checkpoint_run_id,
        "best_checkpoint": str(checkpoint_path),
    }
    logger = None
    try:
        logger = build_logger(eval_config)
        logger.log_config(eval_config)
        logger.log_model(bundle["model"], config=eval_config)
        update_manifest_for_run(
            config=eval_config,
            logger_summary=logger.get_summary(),
            status="running",
            manifest_path=manifest_path,
            eval_source=eval_source,
        )

        test_dataloader = _prepare_test_dataloader_from_data_config(eval_config.data)
        output_dir = RESULTS_ROOT / eval_launch_id / eval_run_id / "e5b_outputs"
        summary = run_e5b_partial_override(
            bundle=bundle,
            test_dataloader=test_dataloader,
            checkpoint_launch_id=checkpoint_launch_id,
            checkpoint_run_id=checkpoint_run_id,
            checkpoint_path=checkpoint_path,
            eval_launch_id=eval_launch_id,
            eval_run_id=eval_run_id,
            output_dir=output_dir,
            logger=logger,
        )
        update_manifest_for_run(
            config=eval_config,
            logger_summary=logger.get_summary(),
            status="completed",
            manifest_path=manifest_path,
            eval_source=eval_source,
        )
        return summary
    except Exception as exc:
        if logger is not None:
            update_manifest_for_run(
                config=eval_config,
                logger_summary=logger.get_summary(),
                status="failed",
                error=str(exc),
                manifest_path=manifest_path,
                eval_source=eval_source,
            )
        raise
    finally:
        if logger is not None:
            logger.finish()
