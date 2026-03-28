import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
import torch.nn as nn

from zoology.config import CheckpointConfig, DataConfig, LoggerConfig, ModelConfig, ModuleConfig, TrainConfig
from zoology.data.multiquery_ar import MQARConfig
from zoology.experiments.flash_vqg import eval_only as e7_eval
from zoology.experiments.flash_vqg.manifest import initialize_manifest


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
            "run_id": "eval-e7-123",
            "run_url": "https://swanlab.example/run/eval-e7-123",
            "run_dir": "/tmp/eval-e7-run",
            "backup_file": "/tmp/eval-e7-run/backup.swanlab",
            "config_file": "/tmp/eval-e7-run/files/config.yaml",
            "metadata_file": "/tmp/eval-e7-run/files/swanlab-metadata.json",
        }


class _FakeFlashLayer(nn.Module):
    def __init__(self, *, if_remote_enabled: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(
            if_remote_enabled=if_remote_enabled,
            fox_remote_path_backend="triton",
            fox_remote_read_topk=None,
            local_num_blocks=2,
            num_codebook_vectors=16,
        )


class _FakeModel(nn.Module):
    def __init__(self, *, if_remote_enabled: bool = True):
        super().__init__()
        self.remote = _FakeFlashLayer(if_remote_enabled=if_remote_enabled)


def _source_config() -> TrainConfig:
    return TrainConfig(
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
        checkpoint=CheckpointConfig(root_dir="./checkpoints"),
        metrics_white_list=[],
        max_epochs=1,
        slice_keys=["num_kv_pairs", "input_seq_len", "mqar_case"],
        launch_id="demo-launch",
        sweep_id="demo-sweep",
        run_id="demo-run",
    )


def test_e7_eval_reloads_checkpoint_per_mode_and_writes_outputs(tmp_path, monkeypatch):
    generated_root = tmp_path / "generated"
    results_root = tmp_path / "results"
    eval_launch_id = "flash-vqg-e7-2026-03-27-00-00-00"
    eval_run_ids = [
        "eval_e7_dense_demo-run",
        "eval_e7_top2_demo-run",
        "eval_e7_top4_demo-run",
    ]
    eval_manifest_path = generated_root / eval_launch_id / "manifest.json"
    fake_loggers = []
    load_calls = []
    observed_modes = []

    initialize_manifest(
        manifest_path=eval_manifest_path,
        launch_id=eval_launch_id,
        sweep_id="flash-vqg-e7",
        logger_backend="swanlab",
        project="demo-project",
        entity="demo-entity",
        run_ids=eval_run_ids,
        launch_config_file=generated_root / eval_launch_id / "launch_configs.py",
        eval_task="e7",
        eval_sources={
            run_id: {
                "checkpoint_launch_id": "demo-launch",
                "checkpoint_run_id": "demo-run",
                "best_checkpoint": None,
            }
            for run_id in eval_run_ids
        },
    )

    def fake_load_checkpoint(*args, **kwargs):
        load_calls.append((args, kwargs))
        return {
            "model": _FakeModel(if_remote_enabled=True),
            "config": _source_config(),
        }

    def fake_evaluate_metrics(bundle, eval_config, test_dataloader, logger):
        cfg = bundle["model"].remote.config
        observed_modes.append(
            {
                "backend": cfg.fox_remote_path_backend,
                "read_topk": cfg.fox_remote_read_topk,
                "metrics_white_list": list(eval_config.metrics_white_list),
            }
        )
        accuracy = {
            None: 0.80,
            2: 0.82,
            4: 0.81,
        }[cfg.fox_remote_read_topk]
        den_cache_ratio = {
            None: 0.37,
            2: 0.32,
            4: 0.35,
        }[cfg.fox_remote_read_topk]
        remote_energy_ratio = {
            None: 0.36,
            2: 0.35,
            4: 0.355,
        }[cfg.fox_remote_read_topk]
        logger.log(
            {
                "valid/accuracy": accuracy,
                "valid/loss": 1.0 - accuracy,
                "valid/attn/den_cache_ratio": den_cache_ratio,
                "valid/attn/o_remote_energy_ratio": remote_energy_ratio,
            },
            step=0,
        )
        return {
            "valid/accuracy": accuracy,
            "valid/loss": 1.0 - accuracy,
            "valid/attn/den_cache_ratio": den_cache_ratio,
            "valid/attn/o_remote_energy_ratio": remote_energy_ratio,
        }

    monkeypatch.setattr(e7_eval, "RESULTS_ROOT", results_root)
    monkeypatch.setattr(
        e7_eval,
        "build_logger",
        lambda config: fake_loggers.append(_FakeLogger()) or fake_loggers[-1],
    )
    monkeypatch.setattr(e7_eval, "load_manifest", lambda launch_id: {"launch_id": launch_id})
    monkeypatch.setattr(
        e7_eval,
        "resolve_best_checkpoint_from_manifest",
        lambda manifest, run_id: tmp_path / "checkpoints" / "demo-run" / "best.pt",
    )
    monkeypatch.setattr(e7_eval, "load_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(e7_eval, "_prepare_test_dataloader_from_data_config", lambda data_config: object())
    monkeypatch.setattr(e7_eval, "_evaluate_metrics", fake_evaluate_metrics)

    result = e7_eval.run_e7_eval(
        checkpoint_launch_id="demo-launch",
        checkpoint_run_id="demo-run",
        eval_launch_id=eval_launch_id,
        eval_sweep_id="flash-vqg-e7",
        eval_run_ids=eval_run_ids,
        logger_backend="swanlab",
        project="demo-project",
        entity="demo-entity",
        manifest_path=eval_manifest_path,
        metrics_white_list=["valid/accuracy"],
    )

    compare_output_dir = results_root / eval_launch_id / "launch_analysis"
    manifest = json.loads(eval_manifest_path.read_text(encoding="utf-8"))
    metrics_df = pd.read_csv(compare_output_dir / "metrics.csv")
    summary = json.loads((compare_output_dir / "summary.json").read_text(encoding="utf-8"))

    assert len(load_calls) == 3
    assert len(fake_loggers) == 3
    assert [item["backend"] for item in observed_modes] == ["torch", "torch", "torch"]
    assert [item["read_topk"] for item in observed_modes] == [None, 2, 4]
    assert all(item["metrics_white_list"] == ["valid/accuracy"] for item in observed_modes)
    assert [logger.model_name for logger in fake_loggers] == [
        "default",
        "default",
        "default",
    ]
    assert metrics_df["mode"].tolist() == ["dense", "top2", "top4"]
    assert metrics_df["valid/accuracy"].round(2).tolist() == [0.80, 0.82, 0.81]
    assert metrics_df["eval_run_id"].tolist() == eval_run_ids
    assert summary["eval_task"] == "e7"
    assert summary["backend_override"] == "torch"
    assert sorted(summary["generated_plots"]) == [
        "valid__accuracy.png",
        "valid__attn__den_cache_ratio.png",
        "valid__attn__o_remote_energy_ratio.png",
        "valid__loss.png",
    ]
    assert result["output_dir"].endswith(f"/{eval_launch_id}")
    assert result["compare_output_dir"].endswith(f"/{eval_launch_id}/launch_analysis")
    assert [run["status"] for run in manifest["runs"]] == ["completed", "completed", "completed"]
    assert manifest["eval_task"] == "e7"
    assert [run["eval_task"] for run in manifest["runs"]] == ["e7", "e7", "e7"]
    assert all(getattr(logger, "finished", False) is True for logger in fake_loggers)
    assert any("valid/accuracy" in metrics for logger in fake_loggers for metrics, _ in logger.logged_metrics)
    assert (compare_output_dir / "valid__accuracy.png").exists()
    assert (compare_output_dir / "valid__loss.png").exists()
    for eval_run_id in eval_run_ids:
        assert not (results_root / eval_launch_id / eval_run_id / "data" / "e7_metrics.csv").exists()
        assert not (results_root / eval_launch_id / eval_run_id / "data" / "e7_summary.json").exists()


def test_e7_eval_rejects_models_without_remote_enabled(monkeypatch, tmp_path):
    monkeypatch.setattr(e7_eval, "load_manifest", lambda launch_id: {"launch_id": launch_id})
    monkeypatch.setattr(
        e7_eval,
        "resolve_best_checkpoint_from_manifest",
        lambda manifest, run_id: tmp_path / "checkpoints" / "demo-run" / "best.pt",
    )
    monkeypatch.setattr(
        e7_eval,
        "load_checkpoint",
        lambda *args, **kwargs: {
            "model": _FakeModel(if_remote_enabled=False),
            "config": _source_config(),
        },
    )

    with pytest.raises(ValueError, match="remote 分支"):
        e7_eval.run_e7_eval(
            checkpoint_launch_id="demo-launch",
            checkpoint_run_id="demo-run",
            eval_launch_id="flash-vqg-e7-2026-03-27-00-00-01",
            eval_sweep_id="flash-vqg-e7",
            eval_run_ids=["eval_e7_dense_demo-run", "eval_e7_top2_demo-run", "eval_e7_top4_demo-run"],
            logger_backend="none",
            project="demo-project",
            entity="demo-entity",
            manifest_path=tmp_path / "generated" / "manifest.json",
        )
