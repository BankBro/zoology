import sys
from types import SimpleNamespace

import pytest

from zoology.config import DataConfig, DataSegmentConfig, LoggerConfig, ModelConfig, TrainConfig
from zoology.logger import NoOpLogger, SwanLabLogger, WandbLogger, build_logger


def _build_config(*, backend: str = "wandb") -> TrainConfig:
    data_segment = DataSegmentConfig(input_seq_len=16, num_examples=4)
    return TrainConfig(
        model=ModelConfig(name="toy"),
        data=DataConfig(train_configs=[data_segment], test_configs=[data_segment]),
        logger=LoggerConfig(
            backend=backend,
            project_name="demo-project",
            entity="demo-entity",
        ),
        run_id="demo-run",
    )


class _FakeWandbRun:
    def __init__(self):
        self.entity = "demo-entity"
        self.id = "wandb-run-id"
        self.name = "demo-run"
        self.url = "https://wandb.example/run"
        self.config = SimpleNamespace(update=lambda *_args, **_kwargs: None)

    def project(self):
        return "demo-project"

    def finish(self):
        return None


class _FakeWandbModule:
    def __init__(self):
        self.run = _FakeWandbRun()

    def init(self, **_kwargs):
        return self.run

    def log(self, *_args, **_kwargs):
        return None

    def watch(self, *_args, **_kwargs):
        return None


class _FakeSwanConfig:
    def update(self, *_args, **_kwargs):
        return None


class _FakeSwanRun:
    def __init__(self):
        self.public = SimpleNamespace(
            project_name="demo-project",
            run_id="swan-run-id",
            run_dir="/tmp/swan-run",
            backup_file="/tmp/swan-run/backup.swanlab",
            cloud=SimpleNamespace(
                experiment_name="demo-run",
                experiment_url="https://swanlab.example/run",
            ),
        )


class _FakeSwanLabModule:
    def __init__(self):
        self.config = _FakeSwanConfig()

    def init(self, **_kwargs):
        return _FakeSwanRun()

    def log(self, *_args, **_kwargs):
        return None

    def finish(self, *_args, **_kwargs):
        return None


def test_build_logger_returns_wandb_logger(monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", _FakeWandbModule())

    logger = build_logger(_build_config(backend="wandb"))

    assert isinstance(logger, WandbLogger)
    assert logger.get_summary()["backend"] == "wandb"


def test_build_logger_returns_swanlab_logger(monkeypatch):
    monkeypatch.setitem(sys.modules, "swanlab", _FakeSwanLabModule())

    logger = build_logger(_build_config(backend="swanlab"))

    assert isinstance(logger, SwanLabLogger)
    assert logger.get_summary()["backend"] == "swanlab"


def test_build_logger_returns_noop_logger():
    logger = build_logger(_build_config(backend="none"))

    assert isinstance(logger, NoOpLogger)
    assert logger.get_summary() == {
        "backend": "none",
        "enabled": False,
        "project": None,
        "entity": None,
        "run_id": None,
        "run_name": None,
        "run_url": None,
        "run_dir": None,
        "backup_file": None,
        "config_file": None,
        "metadata_file": None,
    }


def test_build_logger_rejects_unknown_backend():
    config = _build_config(backend="wandb")
    config.logger.backend = "mystery"

    with pytest.raises(ValueError, match="Unsupported logger backend"):
        build_logger(config)
