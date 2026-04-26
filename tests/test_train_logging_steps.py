import torch
import torch.nn as nn
import pytest

import zoology.train as train_module
from zoology.config import DataConfig, DataSegmentConfig, LoggerConfig, ModelConfig, TrainConfig
from zoology.train import Trainer


class _ToyModel(nn.Module):
    def __init__(self, vocab_size: int = 8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, vocab_size)

    def forward(self, inputs):
        return self.embed(inputs)


class _CaptureLogger:
    def __init__(self):
        self.records = []

    def log_config(self, config):
        return None

    def log_model(self, model, config):
        return None

    def log(self, metrics, *, step=None):
        self.records.append((dict(metrics), step))

    def finish(self):
        return None

    def get_summary(self):
        return {}


class _InterruptingLogger(_CaptureLogger):
    def log_model(self, model, config):
        return None

    def log_config(self, config):
        return None


def _build_interrupt_config() -> TrainConfig:
    data_segment = DataSegmentConfig(input_seq_len=16, num_examples=4)
    return TrainConfig(
        model=ModelConfig(name="toy"),
        data=DataConfig(train_configs=[data_segment], test_configs=[data_segment]),
        logger=LoggerConfig(
            backend="swanlab",
            project_name="demo-project",
            entity="demo-entity",
        ),
        launch_id="demo-launch",
        sweep_id="demo-sweep",
        run_id="demo-run",
    )


def _make_batch(token: int, case: str):
    inputs = torch.tensor([[token, token + 1]], dtype=torch.long)
    targets = inputs.clone()
    return inputs, targets, [{"mqar_case": case, "input_seq_len": 2, "num_kv_pairs": 1}]


def test_trainer_logs_train_and_valid_metrics_on_shared_global_step_axis():
    logger = _CaptureLogger()
    trainer = Trainer(
        model=_ToyModel(),
        train_dataloader=[
            _make_batch(0, "2x1-a"),
            _make_batch(2, "2x1-b"),
        ],
        test_dataloader=[_make_batch(4, "2x1-valid")],
        max_epochs=2,
        learning_rate=1e-2,
        slice_keys=["mqar_case", "input_seq_len", "num_kv_pairs"],
        device="cpu",
        logger=logger,
    )

    trainer.fit()

    train_steps = [step for metrics, step in logger.records if "train/loss" in metrics]
    valid_steps = [step for metrics, step in logger.records if "valid/accuracy" in metrics]

    assert train_steps == [0, 1, 3, 4]
    assert valid_steps == [2, 5]


def test_trainer_can_validate_mid_epoch_without_checkpoint_semantics():
    logger = _CaptureLogger()
    trainer = Trainer(
        model=_ToyModel(),
        train_dataloader=[
            _make_batch(0, "2x1-a"),
            _make_batch(2, "2x1-b"),
            _make_batch(4, "2x1-c"),
            _make_batch(6, "2x1-d"),
        ],
        test_dataloader=[_make_batch(0, "2x1-valid")],
        max_epochs=1,
        learning_rate=1e-2,
        validations_per_epoch=2,
        device="cpu",
        logger=logger,
    )

    trainer.fit()

    train_steps = [step for metrics, step in logger.records if "train/loss" in metrics]
    valid_steps = [step for metrics, step in logger.records if "valid/accuracy" in metrics]

    assert train_steps == [0, 1, 3, 4]
    assert valid_steps == [2, 5]


def test_train_marks_manifest_failed_on_keyboard_interrupt(monkeypatch):
    statuses = []
    logger = _InterruptingLogger()

    class _InterruptingTrainer:
        def __init__(self, **_kwargs):
            return None

        def fit(self):
            raise KeyboardInterrupt()

    monkeypatch.setattr(train_module, "build_logger", lambda _config: logger)
    monkeypatch.setattr(train_module, "LanguageModel", lambda _model_cfg: object())
    monkeypatch.setattr(train_module, "prepare_data", lambda _data_cfg: ([], []))
    monkeypatch.setattr(train_module, "Trainer", _InterruptingTrainer)
    monkeypatch.setattr(
        train_module,
        "update_manifest_for_run",
        lambda *, status, error=None, **_kwargs: statuses.append((status, error)),
    )

    with pytest.raises(KeyboardInterrupt):
        train_module.train(_build_interrupt_config())

    assert statuses == [
        ("running", None),
        ("failed", "KeyboardInterrupt"),
    ]
