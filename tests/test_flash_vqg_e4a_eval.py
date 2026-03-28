import json
from pathlib import Path

import pandas as pd
import torch

from zoology.checkpoints import serialize_train_config
from zoology.config import CheckpointConfig, DataConfig, LoggerConfig, ModelConfig, ModuleConfig, TrainConfig
from zoology.data.multiquery_ar import MQARConfig
from zoology.experiments.flash_vqg import eval_only as e4a_eval
from zoology.experiments.flash_vqg.manifest import initialize_manifest, update_manifest_for_run
from zoology.model import LanguageModel


class _FakeLogger:
    def __init__(self):
        self.logged_metrics = []

    def log_config(self, config):
        self.config = config

    def log_model(self, model, config):
        self.model_name = config.model.name

    def log(self, metrics: dict, *, step: int | None = None):
        self.logged_metrics.append((dict(metrics), step))

    def finish(self):
        self.finished = True

    def get_summary(self):
        return {
            "project": "demo-project",
            "entity": "demo-entity",
            "run_id": "eval-exp-123",
            "run_url": "https://swanlab.example/run/eval-exp-123",
            "run_dir": "/tmp/eval-run",
            "backup_file": "/tmp/eval-run/backup.swanlab",
            "config_file": "/tmp/eval-run/files/config.yaml",
            "metadata_file": "/tmp/eval-run/files/swanlab-metadata.json",
        }


def _build_checkpointed_run(
    tmp_path: Path,
    *,
    metrics_white_list: list[str] | None = None,
) -> tuple[str, str, Path]:
    launch_id = "demo-launch"
    run_id = "demo-run"
    generated_root = tmp_path / "generated"
    manifest_path = generated_root / launch_id / "manifest.json"
    checkpoint_root = tmp_path / "checkpoints"
    run_dir = checkpoint_root / launch_id / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    config = TrainConfig(
        data=DataConfig(
            train_configs=[MQARConfig(vocab_size=2048, input_seq_len=64, num_examples=2, num_kv_pairs=4)],
            test_configs=[MQARConfig(vocab_size=2048, input_seq_len=64, num_examples=2, num_kv_pairs=4)],
            batch_size=(2, 2),
            cache_dir=None,
        ),
        model=ModelConfig(
            d_model=8,
            n_layers=1,
            vocab_size=2048,
            max_position_embeddings=2048,
            sequence_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
            state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
        ),
        logger=LoggerConfig(backend="none"),
        checkpoint=CheckpointConfig(root_dir=str(checkpoint_root)),
        metrics_white_list=[] if metrics_white_list is None else metrics_white_list,
        max_epochs=1,
        slice_keys=["num_kv_pairs", "input_seq_len", "mqar_case"],
        launch_id=launch_id,
        sweep_id="demo-sweep",
        run_id=run_id,
    )

    model = LanguageModel(config.model)
    payload = {
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "epoch": 0,
        "metrics": {"valid/accuracy": 0.5},
        "run_id": run_id,
        "launch_id": launch_id,
        "sweep_id": "demo-sweep",
        "model_name": config.model.name,
    }
    torch.save(payload, run_dir / "best.pt")
    torch.save(payload, run_dir / "last.pt")
    (run_dir / "train_config.json").write_text(
        json.dumps(serialize_train_config(config), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    initialize_manifest(
        manifest_path=manifest_path,
        launch_id=launch_id,
        sweep_id="demo-sweep",
        logger_backend="none",
        project=None,
        entity=None,
        run_ids=[run_id],
        launch_config_file=generated_root / launch_id / "launch_configs.py",
    )
    update_manifest_for_run(
        config=config,
        logger_summary=None,
        status="completed",
        manifest_path=manifest_path,
    )
    return launch_id, run_id, generated_root


def test_e4a_unique_cases_preserve_order_deduplicates_overlap():
    cases = e4a_eval._unique_cases_preserve_order(
        e4a_eval.E4A_LENGTH_PRIMARY_CASES
        + e4a_eval.E4A_LENGTH_AUX_CASES
        + e4a_eval.E4A_CAPACITY_PRIMARY_CASES
    )

    assert cases == [
        (64, 16),
        (128, 16),
        (256, 16),
        (512, 16),
        (1024, 16),
        (64, 8),
        (128, 8),
        (256, 8),
        (512, 8),
        (1024, 8),
        (256, 4),
        (256, 32),
        (256, 64),
        (256, 128),
    ]


def test_e4a_eval_writes_standard_launch_outputs_and_manifest(tmp_path, monkeypatch):
    checkpoint_launch_id, checkpoint_run_id, generated_root = _build_checkpointed_run(tmp_path)
    results_root = tmp_path / "results"
    eval_launch_id = "flash-vqg-e4-2026-03-27-00-00-00"
    eval_run_id = "eval_demo-run"
    eval_manifest_path = generated_root / eval_launch_id / "manifest.json"

    initialize_manifest(
        manifest_path=eval_manifest_path,
        launch_id=eval_launch_id,
        sweep_id="flash-vqg-e4",
        logger_backend="none",
        project="demo-project",
        entity="demo-entity",
        run_ids=[eval_run_id],
        launch_config_file=generated_root / eval_launch_id / "launch_configs.py",
        eval_sources={
            eval_run_id: {
                "checkpoint_launch_id": checkpoint_launch_id,
                "checkpoint_run_id": checkpoint_run_id,
                "best_checkpoint": None,
            }
        },
    )

    monkeypatch.setattr(e4a_eval, "GENERATED_ROOT", generated_root)
    monkeypatch.setattr(e4a_eval, "RESULTS_ROOT", results_root)

    result = e4a_eval.run_e4a_eval(
        checkpoint_launch_id=checkpoint_launch_id,
        checkpoint_run_id=checkpoint_run_id,
        eval_launch_id=eval_launch_id,
        eval_sweep_id="flash-vqg-e4",
        eval_run_id=eval_run_id,
        logger_backend="none",
        project="demo-project",
        entity="demo-entity",
        manifest_path=eval_manifest_path,
    )

    output_dir = results_root / eval_launch_id / eval_run_id
    manifest = json.loads(eval_manifest_path.read_text(encoding="utf-8"))
    run_entry = manifest["runs"][0]

    assert result["checkpoint_launch_id"] == checkpoint_launch_id
    assert result["checkpoint_run_id"] == checkpoint_run_id
    assert result["eval_launch_id"] == eval_launch_id
    assert result["eval_run_id"] == eval_run_id
    assert result["skipped_invalid_cases"] == [(256, 128)]
    assert (output_dir / "data" / "e4a_summary.json").exists()
    assert (output_dir / "data" / "e4a_length_axis_primary.csv").exists()
    assert (output_dir / "data" / "e4a_length_axis_aux.csv").exists()
    assert (output_dir / "data" / "e4a_capacity_axis_primary.csv").exists()
    assert (output_dir / "pics" / "e4a_length_axis_primary.png").exists()
    assert (output_dir / "pics" / "e4a_capacity_axis_primary.png").exists()
    assert run_entry["status"] == "completed"
    assert run_entry["local"]["best_checkpoint"] is None
    assert run_entry["eval_source"]["checkpoint_launch_id"] == checkpoint_launch_id
    assert run_entry["eval_source"]["checkpoint_run_id"] == checkpoint_run_id
    assert run_entry["eval_source"]["best_checkpoint"].endswith("/checkpoints/demo-launch/demo-run/best.pt")

    length_primary = pd.read_csv(output_dir / "data" / "e4a_length_axis_primary.csv")
    capacity_primary = pd.read_csv(output_dir / "data" / "e4a_capacity_axis_primary.csv")
    assert length_primary["input_seq_len"].tolist() == [64, 128, 256, 512, 1024]
    assert capacity_primary["num_kv_pairs"].tolist() == [4, 8, 16, 32, 64, 128]
    assert pd.isna(capacity_primary.iloc[-1]["accuracy"])


def test_e4a_eval_logs_metrics_with_formal_logger(monkeypatch, tmp_path):
    checkpoint_launch_id, checkpoint_run_id, generated_root = _build_checkpointed_run(tmp_path)
    results_root = tmp_path / "results"
    eval_launch_id = "flash-vqg-e4-2026-03-27-00-00-01"
    eval_run_id = "eval_demo-run"
    eval_manifest_path = generated_root / eval_launch_id / "manifest.json"
    fake_logger = _FakeLogger()

    initialize_manifest(
        manifest_path=eval_manifest_path,
        launch_id=eval_launch_id,
        sweep_id="flash-vqg-e4",
        logger_backend="swanlab",
        project="demo-project",
        entity="demo-entity",
        run_ids=[eval_run_id],
        launch_config_file=generated_root / eval_launch_id / "launch_configs.py",
        eval_sources={
            eval_run_id: {
                "checkpoint_launch_id": checkpoint_launch_id,
                "checkpoint_run_id": checkpoint_run_id,
                "best_checkpoint": None,
            }
        },
    )

    monkeypatch.setattr(e4a_eval, "GENERATED_ROOT", generated_root)
    monkeypatch.setattr(e4a_eval, "RESULTS_ROOT", results_root)
    monkeypatch.setattr(e4a_eval, "build_logger", lambda config: fake_logger)

    result = e4a_eval.run_e4a_eval(
        checkpoint_launch_id=checkpoint_launch_id,
        checkpoint_run_id=checkpoint_run_id,
        eval_launch_id=eval_launch_id,
        eval_sweep_id="flash-vqg-e4",
        eval_run_id=eval_run_id,
        logger_backend="swanlab",
        project="demo-project",
        entity="demo-entity",
        manifest_path=eval_manifest_path,
    )

    manifest = json.loads(eval_manifest_path.read_text(encoding="utf-8"))
    run_entry = manifest["runs"][0]
    metric_keys = {
        key
        for metrics, _ in fake_logger.logged_metrics
        for key in metrics.keys()
    }

    assert result["output_dir"].endswith(f"/{eval_launch_id}/{eval_run_id}")
    assert any(step == 0 for _, step in fake_logger.logged_metrics)
    assert "valid/accuracy" in metric_keys
    assert "valid/input_seq_len/accuracy-64" in metric_keys
    assert "valid/num_kv_pairs/accuracy-4" in metric_keys
    assert "valid/mqar_case/accuracy-64x16" in metric_keys
    assert run_entry["status"] == "completed"
    assert run_entry["swanlab"]["experiment_id"] == "eval-exp-123"
    assert run_entry["swanlab"]["run_url"] == "https://swanlab.example/run/eval-exp-123"
    assert run_entry["local"]["run_dir"] == "/tmp/eval-run"
    assert getattr(fake_logger, "finished", False) is True


def test_e4a_eval_fails_on_old_manifest_without_best_checkpoint(tmp_path, monkeypatch):
    launch_id, run_id, generated_root = _build_checkpointed_run(tmp_path)
    manifest_path = generated_root / launch_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["runs"][0]["local"]["best_checkpoint"]
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    monkeypatch.setattr(e4a_eval, "GENERATED_ROOT", generated_root)
    monkeypatch.setattr(e4a_eval, "RESULTS_ROOT", tmp_path / "results")

    try:
        e4a_eval.run_e4a_eval(
            checkpoint_launch_id=launch_id,
            checkpoint_run_id=run_id,
            eval_launch_id="flash-vqg-e4-2026-03-27-00-00-02",
            eval_sweep_id="flash-vqg-e4",
            eval_run_id="eval_demo-run",
            logger_backend="none",
            project="demo-project",
            entity="demo-entity",
            manifest_path=tmp_path / "generated" / "flash-vqg-e4-2026-03-27-00-00-02" / "manifest.json",
        )
    except ValueError as exc:
        assert "best_checkpoint" in str(exc)
    else:
        raise AssertionError("expected ValueError for old manifest without best_checkpoint")


def test_e4a_eval_overrides_checkpoint_metrics_white_list(monkeypatch, tmp_path):
    checkpoint_launch_id, checkpoint_run_id, generated_root = _build_checkpointed_run(
        tmp_path,
        metrics_white_list=["train/loss"],
    )
    fake_logger = _FakeLogger()
    eval_launch_id = "flash-vqg-e4-2026-03-27-00-00-03"
    eval_manifest_path = generated_root / eval_launch_id / "manifest.json"

    initialize_manifest(
        manifest_path=eval_manifest_path,
        launch_id=eval_launch_id,
        sweep_id="flash-vqg-e4",
        logger_backend="swanlab",
        project="demo-project",
        entity="demo-entity",
        run_ids=["eval_demo-run"],
        launch_config_file=generated_root / eval_launch_id / "launch_configs.py",
        eval_sources={
            "eval_demo-run": {
                "checkpoint_launch_id": checkpoint_launch_id,
                "checkpoint_run_id": checkpoint_run_id,
                "best_checkpoint": None,
            }
        },
    )

    monkeypatch.setattr(e4a_eval, "GENERATED_ROOT", generated_root)
    monkeypatch.setattr(e4a_eval, "RESULTS_ROOT", tmp_path / "results")
    monkeypatch.setattr(e4a_eval, "build_logger", lambda config: fake_logger)

    e4a_eval.run_e4a_eval(
        checkpoint_launch_id=checkpoint_launch_id,
        checkpoint_run_id=checkpoint_run_id,
        eval_launch_id=eval_launch_id,
        eval_sweep_id="flash-vqg-e4",
        eval_run_id="eval_demo-run",
        logger_backend="swanlab",
        project="demo-project",
        entity="demo-entity",
        manifest_path=eval_manifest_path,
        metrics_white_list=["valid/accuracy", "valid/mqar_case/*"],
    )

    assert fake_logger.config.metrics_white_list == ["valid/accuracy", "valid/mqar_case/*"]
