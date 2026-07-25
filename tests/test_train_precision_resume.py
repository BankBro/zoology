import copy
import random

import numpy as np
import pytest
import torch
import torch.nn as nn

from zoology.config import (
    CheckpointConfig,
    DataConfig,
    DataSegmentConfig,
    LoggerConfig,
    ModelConfig,
    TrainConfig,
)
from zoology.logger import NoOpLogger
from zoology.train import CheckpointManager, Trainer, TrainingInterrupted


class _RuntimeToyModel(nn.Module):
    def __init__(self, vocab_size: int = 16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, vocab_size)
        self.dropout = nn.Dropout(0.2)
        self.forward_count = 0

    def forward(self, inputs):
        if self.training and torch.is_grad_enabled():
            self.forward_count += 1
        return self.dropout(self.embedding(inputs))

    def get_training_runtime_state(self):
        return {"forward_count": self.forward_count}

    def load_training_runtime_state(self, state):
        self.forward_count = int(state["forward_count"])


def _batch(token: int):
    inputs = torch.tensor([[token, token + 1]], dtype=torch.long)
    return inputs, inputs.clone(), [{"mqar_case": f"case-{token}"}]


def _config(tmp_path, *, run_id: str, controlled_stop: int | None):
    segment = DataSegmentConfig(input_seq_len=2, num_examples=4)
    return TrainConfig(
        data=DataConfig(train_configs=[segment], test_configs=[segment]),
        model=ModelConfig(name="runtime-toy"),
        logger=LoggerConfig(backend="none"),
        checkpoint=CheckpointConfig(root_dir=str(tmp_path)),
        launch_id="resume-test",
        run_id=run_id,
        max_epochs=1,
        validations_per_epoch=4,
        early_stopping_metric=None,
        resume_enabled=True,
        resume_identity={"cache_sha256": "canonical-cache"},
        resume_stop_after_optimizer_step=controlled_stop,
    )


def _trainer(model, config, batches, validation):
    return Trainer(
        model=model,
        train_dataloader=batches,
        test_dataloader=[validation],
        max_epochs=config.max_epochs,
        validations_per_epoch=config.validations_per_epoch,
        precision=config.precision,
        resume_stop_after_optimizer_step=(
            config.resume_stop_after_optimizer_step
        ),
        max_grad_scaler_skips=config.max_grad_scaler_skips,
        max_consecutive_grad_scaler_skips=(
            config.max_consecutive_grad_scaler_skips
        ),
        early_stopping_metric=None,
        device="cpu",
        logger=NoOpLogger(),
        checkpoint_manager=CheckpointManager(config),
    )


def _seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def test_train_config_precision_defaults_are_backward_compatible(tmp_path):
    config = _config(tmp_path, run_id="defaults", controlled_stop=None)
    assert config.precision == "float32"
    assert config.max_grad_scaler_skips == 0
    assert config.max_consecutive_grad_scaler_skips == 0
    assert config.resume_path is None


def test_resume_restores_exact_training_trajectory_and_runtime_state(tmp_path):
    batches = [_batch(token) for token in (0, 2, 4, 6)]
    validation = _batch(8)
    _seed_all(19)
    initial = copy.deepcopy(_RuntimeToyModel().state_dict())

    interrupted_config = _config(
        tmp_path,
        run_id="interrupted",
        controlled_stop=1,
    )
    interrupted_model = _RuntimeToyModel()
    interrupted_model.load_state_dict(initial)
    _seed_all(101)
    with pytest.raises(TrainingInterrupted, match="Controlled stop"):
        _trainer(
            interrupted_model,
            interrupted_config,
            batches,
            validation,
        ).fit()

    resume_path = (
        tmp_path / "resume-test" / "interrupted" / "resume.pt"
    )
    payload = torch.load(resume_path, map_location="cpu", weights_only=False)
    assert payload["next_train_batch_idx"] == 1
    assert payload["optimizer_step"] == 1
    assert next(iter(payload["model_runtime_state"].values()))["forward_count"] == 1

    resumed_model = _RuntimeToyModel()
    resumed_trainer = _trainer(
        resumed_model,
        interrupted_config,
        batches,
        validation,
    )
    resumed_trainer.fit()

    reference_config = _config(
        tmp_path,
        run_id="reference",
        controlled_stop=None,
    )
    reference_model = _RuntimeToyModel()
    reference_model.load_state_dict(initial)
    _seed_all(101)
    reference_trainer = _trainer(
        reference_model,
        reference_config,
        batches,
        validation,
    )
    reference_trainer.fit()

    for name, expected in reference_model.state_dict().items():
        torch.testing.assert_close(
            resumed_model.state_dict()[name],
            expected,
            atol=0,
            rtol=0,
        )
    assert resumed_model.forward_count == reference_model.forward_count == 4
    assert resumed_trainer.optimizer_step == reference_trainer.optimizer_step == 4
    assert resumed_trainer.global_step == reference_trainer.global_step
    assert resumed_trainer.scheduler.state_dict() == reference_trainer.scheduler.state_dict()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_amp_float16_keeps_master_weights_fp32_and_steps_on_cuda():
    model = _RuntimeToyModel().cuda()
    trainer = Trainer(
        model=model,
        train_dataloader=[_batch(0), _batch(2)],
        test_dataloader=[_batch(4)],
        max_epochs=1,
        precision="amp_float16",
        max_grad_scaler_skips=2,
        max_consecutive_grad_scaler_skips=2,
        early_stopping_metric=None,
        device="cuda",
        logger=NoOpLogger(),
    )
    trainer.fit()
    assert trainer.scaler.is_enabled()
    assert trainer.optimizer_step == 2
    assert trainer.grad_scaler_skips == 0
    assert {parameter.dtype for parameter in model.parameters()} == {torch.float32}
    assert all(
        torch.isfinite(parameter).all()
        for parameter in model.parameters()
    )
