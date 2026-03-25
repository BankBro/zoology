from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from zoology.config import (
    CheckpointConfig,
    DataConfig,
    DataSegmentConfig,
    LoggerConfig,
    ModelConfig,
    TrainConfig,
)
from zoology.model import ContinuousInputModel, LanguageModel


def serialize_train_config(config: TrainConfig) -> dict[str, Any]:
    payload = config.model_dump(mode="json")
    payload["data"]["train_configs"] = [
        segment.model_dump(mode="json")
        for segment in config.data.train_configs
    ]
    payload["data"]["test_configs"] = [
        segment.model_dump(mode="json")
        for segment in config.data.test_configs
    ]
    return payload


def resolve_checkpoint_path(path_or_dir: str | Path, which: str = "best") -> Path:
    """
    Resolve a checkpoint target from either a concrete `.pt` file or a run directory.

    Examples:
        resolve_checkpoint_path("checkpoints/manual/run-1/best.pt")
        resolve_checkpoint_path("checkpoints/manual/run-1", which="last")
    """
    path = Path(path_or_dir)
    which_alias = _normalize_checkpoint_alias(which)

    if path.is_file():
        return path

    if path.is_dir():
        checkpoint_path = path / f"{which_alias}.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"在 run 目录 `{path}` 下没有找到 `{checkpoint_path.name}`."
            )
        return checkpoint_path

    if path.suffix == ".pt":
        raise FileNotFoundError(f"checkpoint 文件不存在: `{path}`.")

    raise FileNotFoundError(f"checkpoint 路径不存在: `{path}`.")


def load_checkpoint_payload(
    path_or_dir: str | Path,
    which: str = "best",
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """
    Load a saved checkpoint payload without rebuilding the model.
    """
    checkpoint_path = resolve_checkpoint_path(path_or_dir=path_or_dir, which=which)
    payload = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(payload, dict):
        raise TypeError(f"checkpoint `{checkpoint_path}` 不是预期的 dict payload.")
    return payload


def load_checkpoint(
    path_or_dir: str | Path,
    which: str = "best",
    device: str | torch.device = "cpu",
    eval_mode: bool = True,
    strict: bool = True,
) -> dict[str, Any]:
    """
    Rebuild a model from a saved checkpoint directory or checkpoint file.

    Examples:
        bundle = load_checkpoint("checkpoints/manual/run-1", which="best")
        bundle = load_checkpoint("checkpoints/manual/run-1/last.pt", device="cuda")
    """
    checkpoint_path = resolve_checkpoint_path(path_or_dir=path_or_dir, which=which)
    run_dir = checkpoint_path.parent
    payload = load_checkpoint_payload(checkpoint_path, map_location="cpu")
    config = _load_train_config_from_run_dir(run_dir)

    model = _build_model_from_config(config)
    incompatible = model.load_state_dict(payload["model_state_dict"], strict=strict)
    model = model.to(device)

    if eval_mode:
        model.eval()

    return {
        "model": model,
        "config": config,
        "payload": payload,
        "checkpoint_path": checkpoint_path,
        "run_dir": run_dir,
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def _normalize_checkpoint_alias(which: str) -> str:
    alias = which.strip().lower()
    if alias in {"best", "best.pt"}:
        return "best"
    if alias in {"last", "last.pt"}:
        return "last"
    raise ValueError("`which` 只支持 `best` 或 `last`.")


def _load_train_config_from_run_dir(run_dir: Path) -> TrainConfig:
    config_path = run_dir / "train_config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"在 checkpoint 目录 `{run_dir}` 下没有找到 `train_config.json`, 无法重建模型."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    data_payload = payload["data"]
    data_config = DataConfig(
        train_configs=[_load_data_segment_config(item) for item in data_payload["train_configs"]],
        test_configs=[_load_data_segment_config(item) for item in data_payload["test_configs"]],
        **{k: v for k, v in data_payload.items() if k not in {"train_configs", "test_configs"}},
    )

    return TrainConfig(
        data=data_config,
        model=ModelConfig.model_validate(payload["model"]),
        logger=LoggerConfig.model_validate(payload.get("logger", {})),
        checkpoint=CheckpointConfig.model_validate(payload.get("checkpoint", {})),
        **{k: v for k, v in payload.items() if k not in {"data", "model", "logger", "checkpoint"}},
    )


def _load_data_segment_config(payload: dict[str, Any] | DataSegmentConfig) -> DataSegmentConfig:
    if isinstance(payload, DataSegmentConfig):
        return payload
    if not isinstance(payload, dict):
        raise TypeError(f"data segment payload 必须是 dict, 当前收到: {type(payload).__name__}")

    segment_name = payload.get("name")
    if segment_name is None:
        return DataSegmentConfig.model_validate(payload)

    registry = _data_segment_registry()
    if segment_name not in registry:
        raise ValueError(
            f"未知的数据 segment 类型 `{segment_name}`. 已知类型: {sorted(registry.keys())}"
        )
    return registry[segment_name].model_validate(payload)


def _data_segment_registry() -> dict[str, type[DataSegmentConfig]]:
    from zoology.data.circuits import (
        CumulativeMajorityConfig,
        CumulativeParityConfig,
        MajorityConfig,
        ParityConfig,
        VocabMajorityConfig,
    )
    from zoology.data.compositional_mqar import CompositionalMQARConfig
    from zoology.data.forgetting_mqar import ForgettingMQARConfig
    from zoology.data.multiquery_ar import MQARConfig
    from zoology.data.stacked_mqar import ContinuousMQARConfig

    return {
        "multiquery_ar": MQARConfig,
        "continuous_mqar": ContinuousMQARConfig,
        "compositional_mqar": CompositionalMQARConfig,
        "forgetting_mqar": ForgettingMQARConfig,
        "parity": ParityConfig,
        "majority": MajorityConfig,
        "vocab_majority": VocabMajorityConfig,
        "cumulative_parity": CumulativeParityConfig,
        "cumulative_majority": CumulativeMajorityConfig,
    }


def _build_model_from_config(config: TrainConfig):
    if config.input_type == "continuous":
        return ContinuousInputModel(config.model)
    return LanguageModel(config.model)
