from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from zoology.experiments.flash_vqg.metrics_white_list import (
    default_flash_vqg_metric_universe,
    filter_metric_names,
    has_metrics_white_list,
    metric_chart_type,
    metrics_white_list_from_config,
)


RESULTS_ROOT = Path(__file__).resolve().parent / "results"
GENERATED_ROOT = Path(__file__).resolve().parents[2] / "experiments" / "flash_vqg" / "generated"
DEFAULT_SOURCE = "remote"
DEFAULT_TRAIN_CASES = [(64, 4), (128, 8), (256, 16), (256, 32), (256, 64)]
DEFAULT_TEST_CASES = [(64, 4), (64, 8), (64, 16), (128, 32), (256, 64), (512, 64), (512, 128), (1024, 256)]


@dataclass(frozen=True)
class MetricSpec:
    metric: str
    chart_type: Literal["line", "bar"] = "line"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize_filename(value: str) -> str:
    return (
        str(value)
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


def _to_jsonable(value: Any):
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    return value


def _write_json(path: Path, payload: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _unwrap_swanlab_config(value: Any):
    if isinstance(value, dict):
        if "value" in value and set(value.keys()).issubset({"value", "desc", "sort"}):
            return _unwrap_swanlab_config(value["value"])
        return {k: _unwrap_swanlab_config(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_unwrap_swanlab_config(v) for v in value]
    return value


def generated_launch_dir(launch_id: str) -> Path:
    return GENERATED_ROOT / launch_id


def manifest_path_for_launch(launch_id: str) -> Path:
    return generated_launch_dir(launch_id) / "manifest.json"


def load_manifest(launch_id: str) -> dict[str, Any]:
    path = manifest_path_for_launch(launch_id)
    if not path.exists():
        raise FileNotFoundError(f"未找到 manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def completed_runs(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    completed = [run for run in manifest.get("runs", []) if run.get("status") == "completed"]
    if not completed:
        raise ValueError(f"launch_id={manifest.get('launch_id')} 没有任何 completed run.")
    skipped = [run["run_id"] for run in manifest.get("runs", []) if run.get("status") != "completed"]
    if skipped:
        print(f"[warn] 以下 run 不是 completed, 已跳过: {', '.join(skipped)}")
    return completed


def _cases_from_data_config(data: dict[str, Any], key: str, defaults: list[tuple[int, int]]) -> list[tuple[int, int]]:
    configs = data.get(key) or []
    parsed_cases: list[tuple[int, int]] = []
    for item in configs:
        seq_len = item.get("input_seq_len")
        num_kv_pairs = item.get("num_kv_pairs")
        if seq_len is not None and num_kv_pairs is not None:
            parsed_cases.append((int(seq_len), int(num_kv_pairs)))
    return parsed_cases or defaults[:]


def _metric_universe_from_config(config_dict: dict[str, Any]) -> set[str]:
    data = config_dict.get("data") or {}
    model = config_dict.get("model") or {}
    n_layers = int(model.get("n_layers") or 0)
    train_cases = _cases_from_data_config(data, "train_configs", DEFAULT_TRAIN_CASES)
    test_cases = _cases_from_data_config(data, "test_configs", DEFAULT_TEST_CASES)

    metrics = {
        "train/loss",
        "valid/loss",
        "valid/accuracy",
        "num_parameters",
        "state_size",
    }
    for seq_len, num_kv_pairs in train_cases:
        metrics.add(f"train/mqar_case/loss-{seq_len}x{num_kv_pairs}")
    for seq_len, num_kv_pairs in test_cases:
        metrics.add(f"valid/mqar_case/accuracy-{seq_len}x{num_kv_pairs}")
        metrics.add(f"valid/input_seq_len/accuracy-{seq_len}")
        metrics.add(f"valid/num_kv_pairs/accuracy-{num_kv_pairs}")
    metrics.update(default_flash_vqg_metric_universe(layer_count=n_layers))
    return metrics


def _metric_specs_from_config(config_dict: dict[str, Any], *, eval_task: str | None = None) -> dict[str, MetricSpec]:
    metrics_white_list = metrics_white_list_from_config(config_dict)
    universe = sorted(_metric_universe_from_config(config_dict))
    if has_metrics_white_list(metrics_white_list):
        selected_metrics = filter_metric_names(universe, metrics_white_list)
        selected_metrics.extend(
            metric
            for metric in metrics_white_list
            if "*" not in metric and metric not in selected_metrics
        )
    else:
        selected_metrics = universe

    return {
        metric: MetricSpec(
            metric=metric,
            chart_type=metric_chart_type(metric),
        )
        for metric in selected_metrics
    }


def _candidate_metrics_from_config(config_dict: dict[str, Any], *, eval_task: str | None = None) -> list[str]:
    return ["epoch", *sorted(_metric_specs_from_config(config_dict, eval_task=eval_task).keys())]


def _filter_model_metrics(history: pd.DataFrame, metric_specs: dict[str, MetricSpec] | None = None) -> pd.DataFrame:
    if history.empty:
        return history
    if metric_specs is None:
        allowed_metrics = {metric for metric in history["metric"].unique().tolist() if not str(metric).startswith("__swanlab__.")}
    else:
        allowed_metrics = set(metric_specs)
    filtered = history[history["metric"].isin(allowed_metrics)].copy()
    if filtered.empty:
        return pd.DataFrame(columns=history.columns)
    return filtered.sort_values(["metric", "step"]).reset_index(drop=True)


def _parse_remote_timestamp(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.to_datetime(value, unit="ms", utc=True).isoformat()


def _normalize_remote_metric_name(metric: str, run_id: str) -> str:
    prefix = f"{run_id}-"
    if metric.startswith(prefix):
        return metric[len(prefix) :]
    return metric


def _history_from_remote_wide(df: pd.DataFrame, run_id: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["metric", "step", "epoch", "timestamp", "value"])

    epoch_series = df["epoch"] if "epoch" in df.columns else None
    rows = []
    for metric in sorted(c for c in df.columns if c != "epoch" and not c.endswith("_timestamp")):
        timestamp_col = f"{metric}_timestamp"
        metric_series = df[metric].dropna()
        normalized_metric = _normalize_remote_metric_name(metric, run_id)
        for step, value in metric_series.items():
            epoch_value = None if epoch_series is None else epoch_series.get(step)
            timestamp_value = None if timestamp_col not in df.columns else _parse_remote_timestamp(df.loc[step, timestamp_col])
            rows.append(
                {
                    "metric": normalized_metric,
                    "step": int(step),
                    "epoch": None if pd.isna(epoch_value) else int(epoch_value),
                    "timestamp": timestamp_value,
                    "value": float(value),
                }
            )
    history = pd.DataFrame(rows)
    if history.empty:
        return pd.DataFrame(columns=["metric", "step", "epoch", "timestamp", "value"])
    return history.sort_values(["metric", "step"]).reset_index(drop=True)


def _history_from_remote_single_metric(
    df: pd.DataFrame,
    *,
    metric: str,
    run_id: str,
    epoch_by_step: dict[int, int] | None = None,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["metric", "step", "epoch", "timestamp", "value"])

    normalized_metric = _normalize_remote_metric_name(metric, run_id)
    value_columns = [column for column in df.columns if not column.endswith("_timestamp")]
    if not value_columns:
        return pd.DataFrame(columns=["metric", "step", "epoch", "timestamp", "value"])

    value_column = value_columns[0]
    timestamp_column = f"{value_column}_timestamp"
    value_series = df[value_column].dropna()
    rows = []
    for step, value in value_series.items():
        step_int = int(step)
        timestamp_value = None
        if timestamp_column in df.columns:
            timestamp_value = _parse_remote_timestamp(df.loc[step, timestamp_column])
        epoch_value = None if epoch_by_step is None else epoch_by_step.get(step_int)
        rows.append(
            {
                "metric": normalized_metric,
                "step": step_int,
                "epoch": epoch_value,
                "timestamp": timestamp_value,
                "value": float(value),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["metric", "step", "epoch", "timestamp", "value"])
    return pd.DataFrame(rows).sort_values(["metric", "step"]).reset_index(drop=True)


def fetch_remote_run(run_entry: dict[str, Any], launch_id: str) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    try:
        import swanlab
    except ImportError as e:
        raise ImportError("source=remote 需要安装 swanlab.") from e

    swanlab_info = run_entry.get("swanlab") or {}
    experiment_id = swanlab_info.get("experiment_id")
    project = swanlab_info.get("project")
    entity = swanlab_info.get("entity")
    if not experiment_id or not project or not entity:
        raise ValueError(f"run_id={run_entry.get('run_id')} 缺少 remote 所需的 experiment_id/project/entity.")

    api = swanlab.OpenApi()
    experiment_resp = api.get_experiment(project, experiment_id, username=entity)
    experiment = experiment_resp.data.model_dump()
    config_dict = _unwrap_swanlab_config((experiment.get("profile") or {}).get("config") or {})
    eval_task = run_entry.get("eval_task")
    metric_specs = _metric_specs_from_config(config_dict, eval_task=eval_task)
    candidate_metrics = ["epoch", *sorted(metric_specs)]
    epoch_by_step: dict[int, int] = {}
    epoch_resp = api.get_metrics(experiment_id, "epoch")
    if not epoch_resp.data.empty and "epoch" in epoch_resp.data.columns:
        epoch_by_step = {int(step): int(value) for step, value in epoch_resp.data["epoch"].dropna().items()}

    history_frames = []
    for metric in candidate_metrics:
        if metric == "epoch":
            continue
        metric_resp = api.get_metrics(experiment_id, metric)
        metric_history = _history_from_remote_single_metric(
            metric_resp.data,
            metric=metric,
            run_id=run_entry["run_id"],
            epoch_by_step=epoch_by_step,
        )
        if not metric_history.empty:
            history_frames.append(metric_history)

    if history_frames:
        history = pd.concat(history_frames, ignore_index=True).sort_values(["metric", "step"]).reset_index(drop=True)
    else:
        history = pd.DataFrame(columns=["metric", "step", "epoch", "timestamp", "value"])
    history = _filter_model_metrics(history, metric_specs)

    summary = {
        "launch_id": launch_id,
        "sweep_id": config_dict.get("sweep_id"),
        "run_id": run_entry["run_id"],
        "source": "remote",
        "status": run_entry.get("status"),
        "project": project,
        "entity": entity,
        "experiment_id": experiment_id,
        "run_url": swanlab_info.get("run_url"),
        "run_dir": None,
        "backup_file": None,
        "available_metrics": sorted(history["metric"].unique().tolist()),
        "created_at_utc": experiment.get("createdAt"),
        "finished_at_utc": experiment.get("finishedAt"),
    }
    metadata = {
        "fetched_at_utc": _utc_now(),
        "manifest_entry": run_entry,
        "experiment": experiment,
        "config": config_dict,
        "candidate_metrics": candidate_metrics,
        "metric_specs": {metric: {"chart_type": spec.chart_type} for metric, spec in metric_specs.items()},
        "fetched_metric_count": len(history_frames),
        "eval_task": eval_task,
    }
    return summary, metadata, history


def fetch_local_run(run_entry: dict[str, Any], launch_id: str) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    local_info = run_entry.get("local") or {}
    run_dir_raw = local_info.get("run_dir")
    backup_file_raw = local_info.get("backup_file")
    checkpoint_run_dir_raw = local_info.get("checkpoint_run_dir")
    if run_dir_raw:
        run_dir = Path(run_dir_raw)
    elif checkpoint_run_dir_raw:
        run_dir = Path(checkpoint_run_dir_raw)
    elif backup_file_raw:
        run_dir = Path(backup_file_raw).resolve().parent
    else:
        raise ValueError(
            f"run_id={run_entry.get('run_id')} 缺少 local 所需的 run_dir/checkpoint_run_dir/backup_file."
        )

    backup_file = Path(backup_file_raw) if backup_file_raw else run_dir / "backup.swanlab"
    config_file_raw = local_info.get("config_file") or local_info.get("train_config_json")
    config_file = Path(config_file_raw) if config_file_raw else run_dir / "files" / "config.yaml"
    metadata_file = Path(local_info["metadata_file"]) if local_info.get("metadata_file") else run_dir / "files" / "swanlab-metadata.json"
    parsed_config = {}
    if config_file.exists():
        parsed_config = _unwrap_swanlab_config(yaml.safe_load(config_file.read_text(encoding="utf-8")) or {})
    swanlab_metadata = {}
    if metadata_file.exists():
        swanlab_metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    eval_task = run_entry.get("eval_task")
    metric_specs = _metric_specs_from_config(parsed_config, eval_task=eval_task)

    if backup_file.exists():
        from swanlab.data.porter.datastore import DataStore
        from swanlab.proto.v0 import BaseModel, Scalar

        ds = DataStore()
        ds.open_for_scan(str(backup_file))
        rows = []
        for raw in ds:
            record = BaseModel.from_record(raw)
            if isinstance(record, Scalar):
                rows.append(
                    {
                        "metric": record.key,
                        "step": int(record.step),
                        "epoch": int(record.epoch),
                        "timestamp": record.metric.get("create_time"),
                        "value": float(record.metric.get("data")),
                    }
                )
        history = pd.DataFrame(rows)
        if history.empty:
            history = pd.DataFrame(columns=["metric", "step", "epoch", "timestamp", "value"])
        else:
            history = history.sort_values(["metric", "step"]).reset_index(drop=True)
    else:
        log_file_raw = local_info.get("log_file")
        if not log_file_raw:
            raise FileNotFoundError(
                f"run_id={run_entry.get('run_id')} 缺少 backup 文件: {backup_file}, 且 manifest 未提供 local.log_file."
            )
        history = _history_from_local_log(Path(log_file_raw), run_entry["run_id"])
    history = _filter_model_metrics(history, metric_specs)

    summary = {
        "launch_id": launch_id,
        "sweep_id": parsed_config.get("sweep_id"),
        "run_id": run_entry["run_id"],
        "source": "local",
        "status": run_entry.get("status"),
        "project": (run_entry.get("swanlab") or {}).get("project"),
        "entity": (run_entry.get("swanlab") or {}).get("entity"),
        "experiment_id": (run_entry.get("swanlab") or {}).get("experiment_id"),
        "run_url": (run_entry.get("swanlab") or {}).get("run_url"),
        "run_dir": str(run_dir.resolve()),
        "backup_file": str(backup_file.resolve()),
        "available_metrics": sorted(history["metric"].unique().tolist()),
        "created_at_utc": None,
        "finished_at_utc": None,
    }
    metadata = {
        "fetched_at_utc": _utc_now(),
        "manifest_entry": run_entry,
        "swanlab_metadata": swanlab_metadata,
        "config": parsed_config,
        "metric_specs": {metric: {"chart_type": spec.chart_type} for metric, spec in metric_specs.items()},
        "eval_task": eval_task,
    }
    return summary, metadata, history


def _history_from_local_log(log_file: Path, run_id: str) -> pd.DataFrame:
    if not log_file.exists():
        raise FileNotFoundError(f"run_id={run_id} 的 local.log_file 不存在: {log_file}")

    text = log_file.read_text(encoding="utf-8", errors="ignore").replace("\r", "\n")
    run_matches = list(re.finditer(r"run_id='([^']+)'", text, flags=re.S))
    if not run_matches:
        raise ValueError(f"log_file={log_file} 中未找到任何 run_id 记录.")

    normalized_target = re.sub(r"\s+", "", run_id)
    chunk = None
    for idx, match in enumerate(run_matches):
        normalized_found = re.sub(r"\s+", "", match.group(1))
        if normalized_found != normalized_target:
            continue
        start = match.end()
        end = run_matches[idx + 1].start() if idx + 1 < len(run_matches) else len(text)
        chunk = text[start:end]
        break

    if chunk is None:
        raise ValueError(f"log_file={log_file} 中未找到 run_id={run_id} 的日志分段.")

    epoch_metrics: dict[int, dict[str, float]] = {}
    for line in chunk.splitlines():
        if "Valid Epoch " not in line or "valid/" not in line:
            continue
        epoch_match = re.search(r"Valid Epoch (\d+)/(\d+)", line)
        if epoch_match is None:
            continue
        epoch = int(epoch_match.group(1))
        metrics = {
            metric: float(value)
            for metric, value in re.findall(r"(valid/[A-Za-z0-9_./-]+)=(-?\d+(?:\.\d+)?)", line)
        }
        if not metrics:
            continue
        epoch_metrics[epoch] = metrics

    rows = []
    for epoch, metrics in sorted(epoch_metrics.items()):
        for metric, value in sorted(metrics.items()):
            rows.append(
                {
                    "metric": metric,
                    "step": epoch,
                    "epoch": epoch,
                    "timestamp": None,
                    "value": value,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["metric", "step", "epoch", "timestamp", "value"])
    return pd.DataFrame(rows).sort_values(["metric", "step"]).reset_index(drop=True)


def _run_metrics_index(history: pd.DataFrame, metric_specs: dict[str, MetricSpec]) -> list[dict[str, Any]]:
    rows = []
    for metric, group in history.groupby("metric"):
        chart_type = metric_specs.get(metric, MetricSpec(metric=metric)).chart_type
        if len(group) <= 1:
            chart_type = "bar"
        rows.append(
            {
                "metric": metric,
                "chart_type": chart_type,
                "num_points": int(len(group)),
                "first_step": int(group["step"].min()),
                "last_step": int(group["step"].max()),
                "picture_file": f"{_sanitize_filename(metric)}.png",
            }
        )
    return sorted(rows, key=lambda item: item["metric"])


def _plot_line_metric(group: pd.DataFrame, *, title: str, output_path: Path):
    plt.figure(figsize=(8, 4))
    plt.plot(group["step"], group["value"], linewidth=1.5)
    plt.xlabel("step")
    plt.ylabel("value")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def _plot_run_bar_metric(group: pd.DataFrame, *, title: str, run_id: str, output_path: Path):
    latest = group.sort_values(["step", "epoch"]).iloc[-1]
    plt.figure(figsize=(6, 4))
    plt.bar([run_id], [latest["value"]], width=0.6)
    plt.ylabel("value")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def _plot_run_metrics(history: pd.DataFrame, run_id: str, pics_dir: Path, metric_specs: dict[str, MetricSpec]):
    pics_dir.mkdir(parents=True, exist_ok=True)
    for metric, group in history.groupby("metric"):
        output_path = pics_dir / f"{_sanitize_filename(metric)}.png"
        chart_type = metric_specs.get(metric, MetricSpec(metric=metric)).chart_type
        sorted_group = group.sort_values("step")
        if chart_type == "bar" or len(sorted_group) <= 1:
            _plot_run_bar_metric(group, title=f"{run_id} | {metric}", run_id=run_id, output_path=output_path)
        else:
            _plot_line_metric(sorted_group, title=f"{run_id} | {metric}", output_path=output_path)


def _write_run_outputs(
    *,
    launch_root: Path,
    summary: dict[str, Any],
    metadata: dict[str, Any],
    history: pd.DataFrame,
    metric_specs: dict[str, MetricSpec],
):
    run_root = launch_root / summary["run_id"]
    data_dir = run_root / "data"
    pics_dir = run_root / "pics"
    data_dir.mkdir(parents=True, exist_ok=True)
    pics_dir.mkdir(parents=True, exist_ok=True)

    for filename in ("history.csv", "summary.json", "metadata.json", "metrics_index.json"):
        target = data_dir / filename
        if target.exists():
            target.unlink()
    for metric in metric_specs:
        plot_path = pics_dir / f"{_sanitize_filename(metric)}.png"
        if plot_path.exists():
            plot_path.unlink()

    history.to_csv(data_dir / "history.csv", index=False)
    _write_json(data_dir / "summary.json", summary)
    _write_json(data_dir / "metadata.json", metadata)
    _write_json(data_dir / "metrics_index.json", {"metrics": _run_metrics_index(history, metric_specs)})
    _plot_run_metrics(history, summary["run_id"], pics_dir, metric_specs)


def _find_flash_vqg_kwargs(node: Any) -> dict[str, Any] | None:
    if isinstance(node, dict):
        if node.get("name") == "zoology.mixers.flash_vqg.FlashVQGMixer":
            kwargs = node.get("kwargs")
            return kwargs if isinstance(kwargs, dict) else {}
        for value in node.values():
            found = _find_flash_vqg_kwargs(value)
            if found is not None:
                return found
    elif isinstance(node, list):
        for item in node:
            found = _find_flash_vqg_kwargs(item)
            if found is not None:
                return found
    return None


def _build_run_summary_row(summary: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    row = dict(summary)
    manifest_entry = metadata.get("manifest_entry") or {}
    config = metadata.get("config") or {}
    model = config.get("model") or {}
    data = config.get("data") or {}
    flash_kwargs = _find_flash_vqg_kwargs(model) or {}
    config_summary = manifest_entry.get("config_summary") or {}
    fox_remote_read_topk = flash_kwargs.get("fox_remote_read_topk")
    row.update(
        {
            "seed": config.get("seed"),
            "data_seed": data.get("seed"),
            "learning_rate": config.get("learning_rate"),
            "d_model": model.get("d_model"),
            "n_layers": model.get("n_layers"),
            "train_batch_order": data.get("train_batch_order"),
            "block_len": flash_kwargs.get("block_len"),
            "local_num_blocks": flash_kwargs.get("local_num_blocks"),
            "if_remote_enabled": flash_kwargs.get("if_remote_enabled"),
            "num_codebook_vectors": flash_kwargs.get("num_codebook_vectors"),
            "num_heads": flash_kwargs.get("num_heads"),
            "use_time_mixing": flash_kwargs.get("use_time_mixing"),
            "fox_remote_path_backend": flash_kwargs.get("fox_remote_path_backend"),
            "fox_remote_read_topk": fox_remote_read_topk,
            "read_mode": "dense" if fox_remote_read_topk is None else f"top{int(fox_remote_read_topk)}",
            "experiment_part": config_summary.get("experiment_part", flash_kwargs.get("experiment_part")),
            "experiment_mode": config_summary.get("experiment_mode", flash_kwargs.get("experiment_mode")),
            "fox_clr_selector_mode": config_summary.get("fox_clr_selector_mode", flash_kwargs.get("fox_clr_selector_mode")),
            "fox_clr_merge_mode": config_summary.get("fox_clr_merge_mode", flash_kwargs.get("fox_clr_merge_mode")),
            "fox_clr_gate_mode": config_summary.get("fox_clr_gate_mode", flash_kwargs.get("fox_clr_gate_mode")),
            "fox_clr_lambda_remote": config_summary.get("fox_clr_lambda_remote", flash_kwargs.get("fox_clr_lambda_remote")),
            "fox_clr_gate_init_bias": config_summary.get("fox_clr_gate_init_bias", flash_kwargs.get("fox_clr_gate_init_bias")),
        }
    )
    return row


def _launch_metrics_index(history_by_run: dict[str, pd.DataFrame], metric_specs: dict[str, MetricSpec]) -> list[dict[str, Any]]:
    rows = []
    metrics = sorted({metric for df in history_by_run.values() for metric in df["metric"].unique().tolist()})
    for metric in metrics:
        runs_with_metric = []
        total_points = 0
        max_points_per_run = 0
        for run_id, history in history_by_run.items():
            metric_history = history[history["metric"] == metric]
            if metric_history.empty:
                continue
            runs_with_metric.append(run_id)
            num_points = len(metric_history)
            total_points += num_points
            max_points_per_run = max(max_points_per_run, num_points)
        if not runs_with_metric:
            continue
        chart_type = metric_specs.get(metric, MetricSpec(metric=metric)).chart_type
        if max_points_per_run <= 1:
            chart_type = "bar"
        rows.append(
            {
                "metric": metric,
                "chart_type": chart_type,
                "runs_with_data": runs_with_metric,
                "num_runs": len(runs_with_metric),
                "num_points": int(total_points),
                "picture_file": f"{_sanitize_filename(metric)}.png",
            }
        )
    return rows


def _plot_launch_metrics(
    launch_id: str,
    history_by_run: dict[str, pd.DataFrame],
    output_dir: Path,
    metric_specs: dict[str, MetricSpec],
):
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = sorted({metric for df in history_by_run.values() for metric in df["metric"].unique().tolist()})
    for metric in metrics:
        chart_type = metric_specs.get(metric, MetricSpec(metric=metric)).chart_type
        max_points_per_run = max(
            (
                len(history[history["metric"] == metric])
                for history in history_by_run.values()
            ),
            default=0,
        )
        if max_points_per_run <= 1:
            chart_type = "bar"
        if chart_type == "bar":
            run_ids = []
            values = []
            for run_id, history in sorted(history_by_run.items()):
                metric_history = history[history["metric"] == metric].sort_values(["step", "epoch"])
                if metric_history.empty:
                    continue
                run_ids.append(run_id)
                values.append(metric_history.iloc[-1]["value"])
            if not run_ids:
                continue
            plt.figure(figsize=(9, 5))
            plt.bar(run_ids, values, width=0.65)
            plt.xticks(rotation=15, ha="right")
            plt.ylabel("value")
            plt.title(f"{launch_id} | {metric}")
            plt.tight_layout()
            plt.savefig(output_dir / f"{_sanitize_filename(metric)}.png", dpi=200, bbox_inches="tight")
            plt.close()
            continue

        plt.figure(figsize=(9, 5))
        plotted = False
        for run_id, history in sorted(history_by_run.items()):
            metric_history = history[history["metric"] == metric].sort_values("step")
            if metric_history.empty:
                continue
            plt.plot(metric_history["step"], metric_history["value"], linewidth=1.3, label=run_id)
            plotted = True
        if not plotted:
            plt.close()
            continue
        plt.xlabel("step")
        plt.ylabel("value")
        plt.title(f"{launch_id} | {metric}")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(output_dir / f"{_sanitize_filename(metric)}.png", dpi=200, bbox_inches="tight")
        plt.close()


def _sync_launch_summary_generated_plots(launch_analysis_dir: Path):
    summary_path = launch_analysis_dir / "summary.json"
    if not summary_path.exists():
        return

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        return

    summary["generated_plots"] = sorted(path.name for path in launch_analysis_dir.glob("*.png") if path.is_file())
    _write_json(summary_path, summary)


def run_launch_analysis(*, launch_id: str, source: str = DEFAULT_SOURCE) -> dict[str, Any]:
    if source not in {"remote", "local"}:
        raise ValueError(f"source 只能是 ['local', 'remote'], 当前收到: {source}")

    manifest = load_manifest(launch_id)
    runs = completed_runs(manifest)
    launch_root = RESULTS_ROOT / launch_id
    launch_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    run_summary_rows = []
    history_by_run: dict[str, pd.DataFrame] = {}
    launch_metric_specs: dict[str, MetricSpec] = {}
    for run_entry in runs:
        if source == "remote":
            summary, metadata, history = fetch_remote_run(run_entry, launch_id)
        else:
            summary, metadata, history = fetch_local_run(run_entry, launch_id)
        metric_specs = _metric_specs_from_config(metadata.get("config") or {})
        launch_metric_specs.update(metric_specs)
        _write_run_outputs(
            launch_root=launch_root,
            summary=summary,
            metadata=metadata,
            history=history,
            metric_specs=metric_specs,
        )
        summaries.append(summary)
        run_summary_rows.append(_build_run_summary_row(summary, metadata))
        history_by_run[summary["run_id"]] = history

    launch_analysis_dir = launch_root / "launch_analysis"
    launch_analysis_dir.mkdir(parents=True, exist_ok=True)
    for target in (
        launch_analysis_dir / "run_summary.csv",
        launch_analysis_dir / "metrics_index.json",
    ):
        if target.exists():
            target.unlink()
    for plot_path in launch_analysis_dir.glob("*.png"):
        if plot_path.is_file():
            plot_path.unlink()
    _plot_launch_metrics(launch_id, history_by_run, launch_analysis_dir, launch_metric_specs)
    pd.DataFrame(run_summary_rows).sort_values("run_id").to_csv(launch_analysis_dir / "run_summary.csv", index=False)
    _write_json(
        launch_analysis_dir / "metrics_index.json",
        {
            "launch_id": launch_id,
            "source": source,
            "metrics": _launch_metrics_index(history_by_run, launch_metric_specs),
        },
    )
    _write_json(
        launch_analysis_dir / "summary.json",
        {
            "launch_id": launch_id,
            "source": source,
            "run_count": len(run_summary_rows),
            "experiment_modes": sorted(
                {
                    str(row["experiment_mode"])
                    for row in run_summary_rows
                    if row.get("experiment_mode") not in {None, ""}
                }
            ),
            "fox_remote_read_topk_values": sorted(
                {
                    "dense" if row.get("fox_remote_read_topk") is None else int(row["fox_remote_read_topk"])
                    for row in run_summary_rows
                },
                key=lambda value: (-1 if value == "dense" else int(value)),
            ),
            "generated_plots": [],
        },
    )
    _sync_launch_summary_generated_plots(launch_analysis_dir)
    return {
        "launch_id": launch_id,
        "source": source,
        "output_dir": launch_root,
        "run_summaries": summaries,
    }
