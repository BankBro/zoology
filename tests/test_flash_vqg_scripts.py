import importlib.util
from pathlib import Path

from zoology.analysis.flash_vqg.flash_vqg_analysis_suite import fetch_local_run
from zoology.experiments.flash_vqg.scripts.compare_remat_training import _build_config as build_remat_config
from zoology.experiments.flash_vqg.scripts.smoke_clr_oom_grid import _build_one_config as build_smoke_config


def test_compare_remat_training_build_config_supports_batch_and_gradient_accumulation():
    config = build_remat_config(
        "post_phase1",
        train_batch_size=16,
        eval_batch_size=24,
        gradient_accumulation_steps=8,
    )

    assert config.data.batch_size == (16, 24)
    assert config.gradient_accumulation_steps == 8
    assert config.run_id.endswith("-rformula-clr1-r4-den1-rremat-postp1-tbs16-ebs24-ga8")


def test_smoke_clr_oom_grid_build_config_supports_batch_and_gradient_accumulation():
    config = build_smoke_config(
        4,
        True,
        "off",
        train_batch_size=32,
        eval_batch_size=12,
        gradient_accumulation_steps=4,
    )

    assert config.data.batch_size == (32, 12)
    assert config.gradient_accumulation_steps == 4
    assert config.run_id.endswith("-rformula-clr1-r4-den1-rremat-off-tbs32-ebs12-ga4")


def _load_e1_smoke_module():
    script_path = Path(
        "/home/lyj/mnt/project/zoology/zoology/experiments/flash_vqg/scripts/20260402-clr-v1-mainline/smoke_e1_batch_accum.py"
    )
    spec = importlib.util.spec_from_file_location("flash_vqg_e1_smoke", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_e1_smoke_build_config_supports_top4_and_batch_overrides():
    module = _load_e1_smoke_module()
    config = module._build_one_config(
        read_topk=4,
        train_batch_size=32,
        eval_batch_size=8,
        gradient_accumulation_steps=8,
        metrics_white_list=["train/loss", "valid/accuracy", "attn/remote_topk_den_capture_ratio"],
    )

    flash_kwargs = config.model.sequence_mixer.kwargs["configs"][-1]["kwargs"]
    assert flash_kwargs["fox_remote_formula"] == "clr_v1"
    assert flash_kwargs["fox_remote_read_topk"] == 4
    assert config.data.batch_size == (32, 8)
    assert config.gradient_accumulation_steps == 8
    assert config.run_id.endswith("-rformula-clr1-r4-den1-rremat-off-rread-top4-seed123-tbs32-ebs8-ga8")


def test_e1_train_script_uses_local_analysis_and_dense_top1_top2_top4_modes():
    script_path = Path(
        "/home/lyj/mnt/project/zoology/zoology/experiments/flash_vqg/scripts/20260402-clr-v1-mainline/run_e1_train.sh"
    )
    content = script_path.read_text(encoding="utf-8")

    assert "--analysis \"${ANALYSIS_SOURCE}\"" in content
    assert "--project \"${PROJECT}\"" in content
    assert "--fox-remote-read-topk-values \"${REMOTE_READ_TOPK_VALUES}\"" in content
    assert "REMOTE_READ_TOPK_VALUES=\"${REMOTE_READ_TOPK_VALUES:-dense,1,2,4}\"" in Path(
        "/home/lyj/mnt/project/zoology/zoology/experiments/flash_vqg/scripts/20260402-clr-v1-mainline/common_env.sh"
    ).read_text(encoding="utf-8")


def test_fetch_local_run_falls_back_to_worker_log_when_backup_is_missing(tmp_path):
    run_id = "demo-run"
    checkpoint_dir = tmp_path / "checkpoints" / run_id
    checkpoint_dir.mkdir(parents=True)
    config_path = checkpoint_dir / "train_config.json"
    config_path.write_text(
        """
{
  "seed": 123,
  "learning_rate": 0.001,
  "model": {
    "d_model": 128,
    "n_layers": 2,
    "sequence_mixer": {
      "kwargs": {
        "configs": [
          {
            "name": "zoology.mixers.flash_vqg.FlashVQGMixer",
            "kwargs": {
              "block_len": 32,
              "local_num_blocks": 2,
              "if_remote_enabled": true,
              "num_codebook_vectors": 128,
              "num_heads": 2,
              "use_time_mixing": "kv_shift",
              "fox_remote_path_backend": "torch"
            }
          }
        ]
      }
    }
  },
  "data": {
    "seed": 123,
    "train_batch_order": "global_shuffle",
    "test_configs": [
      {
        "input_seq_len": 64,
        "num_kv_pairs": 4
      }
    ]
  }
}
""".strip(),
        encoding="utf-8",
    )
    log_path = tmp_path / "worker.log"
    log_path.write_text(
        (
            "run_id='demo-run'\n"
            "Valid Epoch 0/2: 100%|##########| 1/1 [00:00<00:00, valid/loss=1.23, valid/accuracy=0.45]\n"
            "Valid Epoch 1/2: 100%|##########| 1/1 [00:00<00:00, valid/loss=0.98, valid/accuracy=0.67, valid/input_seq_len/accuracy-64=0.67]\n"
        ),
        encoding="utf-8",
    )

    summary, metadata, history = fetch_local_run(
        {
            "run_id": run_id,
            "status": "completed",
            "local": {
                "checkpoint_run_dir": str(checkpoint_dir),
                "train_config_json": str(config_path),
                "log_file": str(log_path),
            },
            "swanlab": {},
        },
        launch_id="launch-x",
    )

    assert summary["run_dir"] == str(checkpoint_dir.resolve())
    assert summary["backup_file"].endswith("backup.swanlab")
    assert metadata["config"]["seed"] == 123
    assert sorted(history["metric"].unique().tolist()) == [
        "valid/accuracy",
        "valid/input_seq_len/accuracy-64",
        "valid/loss",
    ]
    latest_acc = history[history["metric"] == "valid/accuracy"].sort_values("step").iloc[-1]["value"]
    assert latest_acc == 0.67
