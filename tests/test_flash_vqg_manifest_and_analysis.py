import json
from pathlib import Path

import pandas as pd
from swanlab.data.porter.datastore import DataStore
from swanlab.proto.v0 import Experiment, Footer, Header, Project, Scalar

from zoology.analysis.flash_vqg.flash_vqg_analysis_suite import (
    _history_from_remote_single_metric,
    _history_from_remote_wide,
    run_launch_analysis,
)
from zoology.config import DataConfig, DataSegmentConfig, LoggerConfig, ModelConfig, TrainConfig
from zoology.experiments.flash_vqg.manifest import (
    MANIFEST_SCHEMA_VERSION,
    checkpoint_local_paths_from_config,
    initialize_manifest,
    update_manifest_for_run,
)


def _build_config() -> TrainConfig:
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


def _write_scalar_backup(
    run_dir: Path,
    *,
    train_loss: float = 1.0,
    valid_accuracy: float = 0.5,
    extra_scalars: list[dict] | None = None,
):
    ds = DataStore()
    backup_file = run_dir / "backup.swanlab"
    ds.open_for_write(str(backup_file))
    ds.write(Header.model_validate({"create_time": "2026-03-26T00:00:00+00:00", "backup_type": "DEFAULT"}).to_record())
    ds.write(Project.model_validate({"name": "demo-project", "workspace": "demo-entity", "public": True}).to_record())
    ds.write(
        Experiment.model_validate(
            {
                "id": "exp-123",
                "name": "demo-run",
                "colors": [],
                "description": "",
                "tags": [],
            }
        ).to_record()
    )
    ds.write(
        Scalar.model_validate(
            {
                "metric": {"index": 0, "data": train_loss, "create_time": "2026-03-26T00:00:01+00:00"},
                "key": "train/loss",
                "step": 0,
                "epoch": 1,
            }
        ).to_record()
    )
    ds.write(
        Scalar.model_validate(
            {
                "metric": {"index": 1, "data": valid_accuracy, "create_time": "2026-03-26T00:00:02+00:00"},
                "key": "valid/accuracy",
                "step": 1,
                "epoch": 1,
            }
        ).to_record()
    )
    for item in extra_scalars or []:
        ds.write(Scalar.model_validate(item).to_record())
    ds.write(Footer.model_validate({"create_time": "2026-03-26T00:00:03+00:00", "success": True}).to_record())
    ds.close()


def test_initialize_and_update_manifest(tmp_path):
    manifest_path = tmp_path / "generated" / "demo-launch" / "manifest.json"
    initialize_manifest(
        manifest_path=manifest_path,
        launch_id="demo-launch",
        sweep_id="demo-sweep",
        logger_backend="swanlab",
        project="demo-project",
        entity="demo-entity",
        run_ids=["demo-run"],
        launch_config_file=tmp_path / "generated" / "demo-launch" / "launch_configs.py",
    )

    update_manifest_for_run(
        config=_build_config(),
        logger_summary={
            "project": "demo-project",
            "entity": "demo-entity",
            "run_id": "exp-123",
            "run_url": "https://swanlab.example/run/exp-123",
            "run_dir": "/tmp/demo-run",
            "backup_file": "/tmp/demo-run/backup.swanlab",
            "config_file": "/tmp/demo-run/files/config.yaml",
            "metadata_file": "/tmp/demo-run/files/swanlab-metadata.json",
        },
        status="completed",
        manifest_path=manifest_path,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == MANIFEST_SCHEMA_VERSION
    run = manifest["runs"][0]
    assert run["status"] == "completed"
    assert run["swanlab"]["experiment_id"] == "exp-123"
    assert run["local"]["backup_file"] == "/tmp/demo-run/backup.swanlab"
    assert run["local"]["checkpoint_run_dir"].endswith("/checkpoints/demo-launch/demo-run")
    assert run["local"]["best_checkpoint"].endswith("/checkpoints/demo-launch/demo-run/best.pt")
    assert run["local"]["last_checkpoint"].endswith("/checkpoints/demo-launch/demo-run/last.pt")
    assert run["local"]["train_config_json"].endswith("/checkpoints/demo-launch/demo-run/train_config.json")
    assert run["eval_source"] == {
        "checkpoint_launch_id": None,
        "checkpoint_run_id": None,
        "best_checkpoint": None,
    }


def test_checkpoint_local_paths_are_empty_when_checkpoint_disabled():
    config = _build_config()
    config.checkpoint.enabled = False

    paths = checkpoint_local_paths_from_config(config)

    assert paths == {
        "checkpoint_run_dir": None,
        "best_checkpoint": None,
        "last_checkpoint": None,
        "train_config_json": None,
    }


def test_local_analysis_writes_run_and_launch_outputs(tmp_path, monkeypatch):
    generated_root = tmp_path / "generated"
    results_root = tmp_path / "analysis-results"
    launch_id = "demo-launch"
    launch_dir = generated_root / launch_id
    run_dir = tmp_path / "swanlog" / "run-demo"
    files_dir = run_dir / "files"
    files_dir.mkdir(parents=True, exist_ok=True)
    (files_dir / "config.yaml").write_text(
        """launch_id:
  value: demo-launch
run_id:
  value: demo-run
sweep_id:
  value: demo-sweep
seed:
  value: 456
learning_rate:
  value: 0.001
data:
  value:
    seed: 123
    train_batch_order: global_shuffle
model:
  value:
    d_model: 128
    n_layers: 2
    sequence_mixer:
      kwargs:
        configs:
          - name: zoology.mixers.base_conv.BaseConv
            kwargs: {}
          - name: zoology.mixers.flash_vqg.FlashVQGMixer
            kwargs:
              block_len: 32
              local_num_blocks: 2
              if_remote_enabled: true
              num_codebook_vectors: 128
              num_heads: 2
              use_time_mixing: kv_shift
              fox_remote_path_backend: torch
              fox_remote_read_topk: 2
""",
        encoding="utf-8",
    )
    (files_dir / "swanlab-metadata.json").write_text(
        json.dumps({"swanlab": {"logdir": str(run_dir)}}, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_scalar_backup(run_dir)

    initialize_manifest(
        manifest_path=launch_dir / "manifest.json",
        launch_id=launch_id,
        sweep_id="demo-sweep",
        logger_backend="swanlab",
        project="demo-project",
        entity="demo-entity",
        run_ids=["demo-run"],
        launch_config_file=launch_dir / "launch_configs.py",
    )
    manifest = json.loads((launch_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["runs"][0]["status"] = "completed"
    manifest["runs"][0]["swanlab"]["experiment_id"] = "exp-123"
    manifest["runs"][0]["swanlab"]["run_url"] = "https://swanlab.example/run/exp-123"
    manifest["runs"][0]["local"]["run_dir"] = str(run_dir)
    manifest["runs"][0]["local"]["backup_file"] = str(run_dir / "backup.swanlab")
    manifest["runs"][0]["local"]["config_file"] = str(files_dir / "config.yaml")
    manifest["runs"][0]["local"]["metadata_file"] = str(files_dir / "swanlab-metadata.json")
    (launch_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    preserved_data_file = results_root / launch_id / "demo-run" / "data" / "e7_metrics.csv"
    preserved_pic_file = results_root / launch_id / "demo-run" / "pics" / "e7_accuracy_compare.png"
    preserved_data_file.parent.mkdir(parents=True, exist_ok=True)
    preserved_pic_file.parent.mkdir(parents=True, exist_ok=True)
    preserved_data_file.write_text("mode,valid/accuracy\ndense,0.8\n", encoding="utf-8")
    preserved_pic_file.write_bytes(b"demo")

    monkeypatch.setattr("zoology.analysis.flash_vqg.flash_vqg_analysis_suite.GENERATED_ROOT", generated_root)
    monkeypatch.setattr("zoology.analysis.flash_vqg.flash_vqg_analysis_suite.RESULTS_ROOT", results_root)

    result = run_launch_analysis(launch_id=launch_id, source="local")

    run_root = results_root / launch_id / "demo-run"
    assert result["launch_id"] == launch_id
    assert (run_root / "data" / "history.csv").exists()
    assert (run_root / "data" / "summary.json").exists()
    assert (run_root / "pics" / "train__loss.png").exists()
    assert preserved_data_file.exists()
    assert preserved_pic_file.exists()
    assert (results_root / launch_id / "launch_analysis" / "train__loss.png").exists()

    run_summary = pd.read_csv(results_root / launch_id / "launch_analysis" / "run_summary.csv")
    row = run_summary.iloc[0].to_dict()
    assert row["block_len"] == 32
    assert row["local_num_blocks"] == 2
    assert row["if_remote_enabled"] in {True, 1, 1.0}
    assert row["num_codebook_vectors"] == 128
    assert row["train_batch_order"] == "global_shuffle"
    assert row["seed"] == 456
    assert row["data_seed"] == 123
    assert row["fox_remote_path_backend"] == "torch"
    assert row["fox_remote_read_topk"] == 2
    assert row["read_mode"] == "top2"


def test_e7_local_analysis_keeps_default_metric_names_and_launch_outputs(tmp_path, monkeypatch):
    generated_root = tmp_path / "generated"
    results_root = tmp_path / "analysis-results"
    launch_id = "flash-vqg-e7-demo"
    launch_dir = generated_root / launch_id
    run_ids = [
        "eval_e7_dense_demo-run",
        "eval_e7_top2_demo-run",
        "eval_e7_top4_demo-run",
    ]
    run_payloads = {
        run_ids[0]: (1.0, 0.50),
        run_ids[1]: (0.9, 0.55),
        run_ids[2]: (0.95, 0.53),
    }

    initialize_manifest(
        manifest_path=launch_dir / "manifest.json",
        launch_id=launch_id,
        sweep_id="flash-vqg-e7",
        logger_backend="swanlab",
        project="demo-project",
        entity="demo-entity",
        run_ids=run_ids,
        launch_config_file=launch_dir / "launch_configs.py",
        eval_task="e7",
    )
    manifest = json.loads((launch_dir / "manifest.json").read_text(encoding="utf-8"))

    for run in manifest["runs"]:
        run_id = run["run_id"]
        run_dir = tmp_path / "swanlog" / run_id
        files_dir = run_dir / "files"
        files_dir.mkdir(parents=True, exist_ok=True)
        (files_dir / "config.yaml").write_text(
            f"""launch_id:
  value: {launch_id}
run_id:
  value: {run_id}
sweep_id:
  value: flash-vqg-e7
metrics_white_list:
  value:
    - valid/accuracy
    - valid/loss
model:
  value:
    d_model: 128
    n_layers: 1
""",
            encoding="utf-8",
        )
        (files_dir / "swanlab-metadata.json").write_text(
            json.dumps({"swanlab": {"logdir": str(run_dir)}}, ensure_ascii=False),
            encoding="utf-8",
        )
        train_loss, valid_accuracy = run_payloads[run_id]
        _write_scalar_backup(
            run_dir,
            train_loss=train_loss,
            valid_accuracy=valid_accuracy,
            extra_scalars=[
                {
                    "metric": {"index": 2, "data": 1.5 - valid_accuracy, "create_time": "2026-03-26T00:00:02+00:00"},
                    "key": "valid/loss",
                    "step": 1,
                    "epoch": 1,
                }
            ],
        )
        run["status"] = "completed"
        run["swanlab"]["experiment_id"] = f"exp-{run_id}"
        run["local"]["run_dir"] = str(run_dir)
        run["local"]["backup_file"] = str(run_dir / "backup.swanlab")
        run["local"]["config_file"] = str(files_dir / "config.yaml")
        run["local"]["metadata_file"] = str(files_dir / "swanlab-metadata.json")

    (launch_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    launch_analysis_dir = results_root / launch_id / "launch_analysis"
    launch_analysis_dir.mkdir(parents=True, exist_ok=True)
    (launch_analysis_dir / "metrics.csv").write_text("mode,valid/accuracy\ndense,0.5\n", encoding="utf-8")
    (launch_analysis_dir / "summary.json").write_text(json.dumps({"eval_task": "e7"}, ensure_ascii=False), encoding="utf-8")

    monkeypatch.setattr("zoology.analysis.flash_vqg.flash_vqg_analysis_suite.GENERATED_ROOT", generated_root)
    monkeypatch.setattr("zoology.analysis.flash_vqg.flash_vqg_analysis_suite.RESULTS_ROOT", results_root)

    result = run_launch_analysis(launch_id=launch_id, source="local")

    assert result["launch_id"] == launch_id
    for run_id in run_ids:
        run_root = results_root / launch_id / run_id
        history = pd.read_csv(run_root / "data" / "history.csv")
        assert sorted(history["metric"].unique().tolist()) == ["valid/accuracy", "valid/loss"]
        assert (run_root / "pics" / "valid__accuracy.png").exists()
        assert (run_root / "pics" / "valid__loss.png").exists()

    assert (launch_analysis_dir / "valid__accuracy.png").exists()
    assert (launch_analysis_dir / "valid__loss.png").exists()
    assert (launch_analysis_dir / "metrics.csv").exists()
    assert (launch_analysis_dir / "summary.json").exists()
    summary = json.loads((launch_analysis_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["generated_plots"] == ["valid__accuracy.png", "valid__loss.png"]


def test_remote_history_normalizes_run_prefix():
    df = pd.DataFrame(
        {
            "demo-run-train/loss": {0: 1.0, 1: 0.8},
            "demo-run-valid/accuracy": {1: 0.5},
            "epoch": {0: 1, 1: 1},
            "demo-run-train/loss_timestamp": {0: 0, 1: 1000},
            "demo-run-valid/accuracy_timestamp": {0: None, 1: 1000},
        }
    )

    history = _history_from_remote_wide(df, "demo-run")

    assert sorted(history["metric"].unique().tolist()) == ["train/loss", "valid/accuracy"]


def test_remote_single_metric_keeps_full_series_without_inner_join():
    train_df = pd.DataFrame(
        {
            "demo-run-train/loss": {0: 1.0, 1: 0.8, 2: 0.6},
            "demo-run-train/loss_timestamp": {0: 0, 1: 1000, 2: 2000},
        }
    )
    valid_df = pd.DataFrame(
        {
            "demo-run-valid/loss": {0: 1.5, 1: 1.2},
            "demo-run-valid/loss_timestamp": {0: 0, 1: 1000},
        }
    )
    summary_df = pd.DataFrame(
        {
            "demo-run-num_parameters": {0: 123.0},
            "demo-run-num_parameters_timestamp": {0: 0},
        }
    )

    epoch_by_step = {0: 1, 1: 1, 2: 1}
    history = pd.concat(
        [
            _history_from_remote_single_metric(train_df, metric="train/loss", run_id="demo-run", epoch_by_step=epoch_by_step),
            _history_from_remote_single_metric(valid_df, metric="valid/loss", run_id="demo-run", epoch_by_step=epoch_by_step),
            _history_from_remote_single_metric(
                summary_df,
                metric="num_parameters",
                run_id="demo-run",
                epoch_by_step=epoch_by_step,
            ),
        ],
        ignore_index=True,
    ).sort_values(["metric", "step"])

    assert len(history[history["metric"] == "train/loss"]) == 3
    assert len(history[history["metric"] == "valid/loss"]) == 2
    assert len(history[history["metric"] == "num_parameters"]) == 1


def test_local_analysis_respects_metrics_white_list(tmp_path, monkeypatch):
    generated_root = tmp_path / "generated"
    results_root = tmp_path / "analysis-results"
    launch_id = "demo-launch"
    launch_dir = generated_root / launch_id
    run_dir = tmp_path / "swanlog" / "run-demo"
    files_dir = run_dir / "files"
    files_dir.mkdir(parents=True, exist_ok=True)
    (files_dir / "config.yaml").write_text(
        """launch_id:
  value: demo-launch
run_id:
  value: demo-run
metrics_white_list:
  value:
    - valid/accuracy
model:
  value:
    d_model: 128
    n_layers: 2
""",
        encoding="utf-8",
    )
    (files_dir / "swanlab-metadata.json").write_text(
        json.dumps({"swanlab": {"logdir": str(run_dir)}}, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_scalar_backup(run_dir)

    initialize_manifest(
        manifest_path=launch_dir / "manifest.json",
        launch_id=launch_id,
        sweep_id="demo-sweep",
        logger_backend="swanlab",
        project="demo-project",
        entity="demo-entity",
        run_ids=["demo-run"],
        launch_config_file=launch_dir / "launch_configs.py",
    )
    manifest = json.loads((launch_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["runs"][0]["status"] = "completed"
    manifest["runs"][0]["local"]["run_dir"] = str(run_dir)
    manifest["runs"][0]["local"]["backup_file"] = str(run_dir / "backup.swanlab")
    manifest["runs"][0]["local"]["config_file"] = str(files_dir / "config.yaml")
    manifest["runs"][0]["local"]["metadata_file"] = str(files_dir / "swanlab-metadata.json")
    (launch_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    monkeypatch.setattr("zoology.analysis.flash_vqg.flash_vqg_analysis_suite.GENERATED_ROOT", generated_root)
    monkeypatch.setattr("zoology.analysis.flash_vqg.flash_vqg_analysis_suite.RESULTS_ROOT", results_root)

    run_launch_analysis(launch_id=launch_id, source="local")

    metrics_index = json.loads(
        (results_root / launch_id / "launch_analysis" / "metrics_index.json").read_text(encoding="utf-8")
    )
    assert [item["metric"] for item in metrics_index["metrics"]] == ["valid/accuracy"]
