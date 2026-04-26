from __future__ import annotations

import fnmatch
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, TypeVar

import yaml


DEFAULT_METRICS_WHITE_LIST: list[str] = []
BAR_CHART_METRICS = {"num_parameters", "state_size"}
DEFAULT_LAYER_SENTINEL_COUNT = 8

DEFAULT_ATTN_METRICS = [
    "attn/den_min",
    "attn/nan_inf_count",
    "attn/o_remote_energy_ratio",
    "attn/clr_alpha_norm_mean",
    "attn/clr_den_neg_ratio",
    "attn/remote_win_rate",
    "attn/den_cache_ratio",
    "attn/remote_routing_entropy",
    "attn/remote_top1_top2_margin",
    "attn/remote_topk_den_capture_ratio",
    "attn/q_rms_mean",
    "attn/k_rms_mean",
    "attn/k_hat_rms_mean",
    "attn/clr_h_norm_mean",
    "attn/gd_residual_inject_ratio",
    "attn/gd_residual_lambda_mean",
    "attn/gd_residual_write_strength_mean",
    "attn/gd_residual_m_norm_mean",
    "attn/gd_residual_m_norm_max",
    "attn/gd_residual_mu_valid_ratio",
]
DEFAULT_VQ_METRICS = [
    "vq/c_sim_min",
    "vq/c_sim_mean",
    "vq/c_sim_max",
    "vq/c_dist_min",
    "vq/c_dist_mean",
    "vq/c_dist_max",
    "vq/c_norm_min",
    "vq/c_norm_mean",
    "vq/c_norm_max",
    "vq/c_usage_min",
    "vq/c_usage_mean",
    "vq/c_usage_max",
    "vq/c_entropy",
    "vq/k_norm_mean",
    "vq/k_hat_norm_mean",
    "vq/relative_err_min",
    "vq/relative_err_mean",
    "vq/relative_err_max",
    "vq/c_rms_mean",
    "vq/c_usage_min_batch",
    "vq/c_usage_mean_batch",
    "vq/c_usage_max_batch",
    "vq/c_entropy_batch",
    "vq/c_usage_small_ratio",
    "vq/c_usage_large_ratio",
    "vq/write_entropy_mean",
    "vq/write_top1_mass_mean",
]
REMOTE_LITE_BASE_KEYS = {
    "attn/den_min",
    "attn/nan_inf_count",
    "attn/o_remote_energy_ratio",
    "attn/clr_alpha_norm_mean",
    "attn/clr_den_neg_ratio",
}
REMOTE_FULL_ONLY_BASE_KEYS = {
    "attn/remote_win_rate",
    "attn/den_cache_ratio",
    "attn/remote_routing_entropy",
    "attn/remote_top1_top2_margin",
    "attn/remote_topk_den_capture_ratio",
    "attn/q_rms_mean",
    "attn/k_rms_mean",
    "attn/k_hat_rms_mean",
    "attn/clr_h_norm_mean",
}
GD_RESIDUAL_BASE_KEYS = {
    "attn/gd_residual_inject_ratio",
    "attn/gd_residual_lambda_mean",
    "attn/gd_residual_write_strength_mean",
    "attn/gd_residual_m_norm_mean",
    "attn/gd_residual_m_norm_max",
    "attn/gd_residual_mu_valid_ratio",
}

T = TypeVar("T")


def normalize_metrics_white_list(raw_values: Iterable[str] | None) -> list[str]:
    if raw_values is None:
        return []

    normalized: list[str] = []
    seen: set[str] = set()
    for raw_value in raw_values:
        value = str(raw_value).strip()
        if not value:
            continue
        if value not in seen:
            normalized.append(value)
            seen.add(value)
    return normalized


def parse_metrics_white_list_csv(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return normalize_metrics_white_list(part for part in str(raw).split(","))


def load_metrics_white_list_file(path: str | Path) -> list[str]:
    resolved_path = Path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"未找到 metrics white list 文件: {resolved_path}")

    text = resolved_path.read_text(encoding="utf-8")
    suffix = resolved_path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(text)
    else:
        payload = yaml.safe_load(text)

    if payload is None:
        return []
    if isinstance(payload, dict):
        payload = payload.get("metrics_white_list", [])
    if not isinstance(payload, list):
        raise TypeError("metrics white list 文件顶层必须是 list, 或包含 metrics_white_list 的 dict.")
    return normalize_metrics_white_list(payload)


def merge_metrics_white_lists(*value_groups: Iterable[str] | None) -> list[str]:
    merged: list[str] = []
    for group in value_groups:
        merged.extend(normalize_metrics_white_list(group))
    return normalize_metrics_white_list(merged)


def has_metrics_white_list(metrics_white_list: Iterable[str] | None) -> bool:
    return bool(normalize_metrics_white_list(metrics_white_list))


def metric_matches_white_list(metric: str, metrics_white_list: Iterable[str] | None) -> bool:
    normalized = normalize_metrics_white_list(metrics_white_list)
    if not normalized:
        return True
    metric = str(metric)
    return any(fnmatch.fnmatchcase(metric, pattern) for pattern in normalized)


def filter_metric_names(metrics: Iterable[str], metrics_white_list: Iterable[str] | None) -> list[str]:
    normalized = normalize_metrics_white_list(metrics_white_list)
    names = [str(metric) for metric in metrics]
    if not normalized:
        return names
    return [metric for metric in names if metric_matches_white_list(metric, normalized)]


def filter_metrics_dict(metrics: Mapping[str, T], metrics_white_list: Iterable[str] | None) -> dict[str, T]:
    normalized = normalize_metrics_white_list(metrics_white_list)
    if not normalized:
        return dict(metrics)
    return {
        str(key): value
        for key, value in metrics.items()
        if metric_matches_white_list(str(key), normalized)
    }


def metrics_white_list_from_config(config: Any) -> list[str]:
    if config is None:
        return []
    if isinstance(config, Mapping):
        raw = config.get("metrics_white_list", [])
    else:
        raw = getattr(config, "metrics_white_list", [])
    if raw is None:
        return []
    if not isinstance(raw, list):
        raw = list(raw)
    return normalize_metrics_white_list(raw)


def metric_chart_type(metric: str) -> str:
    return "bar" if str(metric) in BAR_CHART_METRICS else "line"


def _expand_metric_variants(base_metrics: Iterable[str], *, layer_count: int = DEFAULT_LAYER_SENTINEL_COUNT) -> set[str]:
    variants: set[str] = set()
    for metric in base_metrics:
        metric = str(metric)
        variants.add(metric)
        variants.add(f"valid/{metric}")
        for layer_idx in range(layer_count):
            variants.add(f"layer_{layer_idx}/{metric}")
            variants.add(f"valid/layer_{layer_idx}/{metric}")
    return variants


def default_flash_vqg_metric_universe(*, layer_count: int = DEFAULT_LAYER_SENTINEL_COUNT) -> set[str]:
    return _expand_metric_variants(
        [*DEFAULT_ATTN_METRICS, *DEFAULT_VQ_METRICS],
        layer_count=layer_count,
    )


def derive_flash_metric_controls(
    metrics_white_list: Iterable[str] | None,
    *,
    layer_count: int = DEFAULT_LAYER_SENTINEL_COUNT,
) -> dict[str, Any]:
    normalized = normalize_metrics_white_list(metrics_white_list)
    if not normalized:
        return {
            "enable_layer_metrics": True,
            "fox_phase2_metrics_mode": "full",
        }

    model_universe = default_flash_vqg_metric_universe(layer_count=layer_count)
    enable_layer_metrics = bool(filter_metric_names(model_universe, normalized))

    remote_lite_variants = _expand_metric_variants(REMOTE_LITE_BASE_KEYS, layer_count=layer_count)
    remote_full_only_variants = _expand_metric_variants(REMOTE_FULL_ONLY_BASE_KEYS, layer_count=layer_count)
    gd_residual_variants = _expand_metric_variants(GD_RESIDUAL_BASE_KEYS, layer_count=layer_count)

    if filter_metric_names(remote_full_only_variants, normalized):
        phase2_mode = "full"
    elif filter_metric_names(remote_lite_variants, normalized) or filter_metric_names(
        gd_residual_variants,
        normalized,
    ):
        phase2_mode = "lite"
    else:
        phase2_mode = "off"

    return {
        "enable_layer_metrics": enable_layer_metrics,
        "fox_phase2_metrics_mode": phase2_mode,
    }
