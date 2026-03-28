import torch
import torch.nn as nn

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
