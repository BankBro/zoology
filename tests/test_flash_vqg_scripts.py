import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch

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


def _load_e2_builder_module():
    script_path = Path(
        "/home/lyj/mnt/project/zoology/zoology/experiments/flash_vqg/scripts/20260402-clr-v1-mainline/e2-remote-interface/config_builder.py"
    )
    spec = importlib.util.spec_from_file_location("flash_vqg_e2_builder", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_e3_builder_module():
    script_path = Path(
        "/home/lyj/mnt/project/zoology/zoology/experiments/flash_vqg/scripts/20260402-clr-v1-mainline/e3-dense-routing/config_builder.py"
    )
    spec = importlib.util.spec_from_file_location("flash_vqg_e3_builder", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_gd_residual_builder_module():
    script_path = Path(
        "/home/lyj/mnt/project/zoology/zoology/experiments/flash_vqg/scripts/20260425-gd-residual-v1-mqar/config_builder.py"
    )
    spec = importlib.util.spec_from_file_location("flash_vqg_gd_residual_builder", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_gd_residual_short_run_module():
    script_path = Path(
        "/home/lyj/mnt/project/zoology/zoology/experiments/flash_vqg/scripts/20260425-gd-residual-v1-mqar/short_run_gd_residual_v1.py"
    )
    spec = importlib.util.spec_from_file_location("flash_vqg_gd_residual_short_run", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _build_e2_args() -> Namespace:
    return Namespace(
        launch_id_prefix="flash-vqg-e2-test",
        backend="torch",
        logger_backend="none",
        dmodels="128",
        learning_rates="1e-3",
        train_batch_order="global_shuffle",
        seed_values="123",
        data_seed=123,
        num_codebook_vectors="128",
        fox_remote_path_backend="torch",
        fox_clr_rank=4,
        fox_clr_use_den_residual="true",
        fox_clr_remat_mode="off",
        gradient_accumulation_steps=8,
        train_batch_size=16,
        eval_batch_size=8,
        cache_dir="./data/flash_vqg",
        project="flash_vqg_clr_v1_mainline",
        entity="scu-mclab",
        max_epochs=32,
        metrics_white_list=None,
        metrics_white_list_file=str(
            Path(
                "/home/lyj/mnt/project/zoology/zoology/experiments/flash_vqg/scripts/20260402-clr-v1-mainline/e2-remote-interface/metrics.yaml"
            )
        ),
    )


def _build_e3_args() -> Namespace:
    return Namespace(
        launch_id_prefix="flash-vqg-e3-test",
        backend="torch",
        logger_backend="none",
        dmodels="128",
        learning_rates="1e-3",
        train_batch_order="global_shuffle",
        seed_values="123",
        data_seed=123,
        num_codebook_vectors="128",
        fox_remote_path_backend="torch",
        fox_clr_rank=4,
        fox_clr_use_den_residual="true",
        fox_clr_remat_mode="off",
        vq_topk=8,
        gradient_accumulation_steps=8,
        train_batch_size=16,
        eval_batch_size=8,
        cache_dir="./data/flash_vqg",
        project="flash_vqg_clr_v1_mainline",
        entity="scu-mclab",
        max_epochs=32,
        metrics_white_list=None,
        metrics_white_list_file=str(
            Path(
                "/home/lyj/mnt/project/zoology/zoology/experiments/flash_vqg/scripts/20260402-clr-v1-mainline/e3-dense-routing/metrics.yaml"
            )
        ),
    )


def _build_gd_residual_args() -> Namespace:
    return Namespace(
        launch_id_prefix="flash-vqg-gdr1-test",
        backend="torch",
        logger_backend="none",
        dmodels="128",
        learning_rates="1e-3",
        train_batch_order="global_shuffle",
        seed_values="123",
        data_seed=123,
        num_codebook_vectors="128",
        fox_remote_path_backend="torch",
        fox_remote_read_topk_values="2",
        fox_remote_formula="gd_residual_v1",
        fox_gd_residual_rank=16,
        fox_gd_residual_write_topk=4,
        fox_gd_residual_builder="grouped_chunk_torch_ref",
        fox_gd_residual_pack_mode="semivec_ref",
        fox_gd_residual_chunk_size=64,
        fox_gd_residual_mu_min_count=1.0,
        fox_gd_residual_addr_eps=1e-6,
        fox_gd_residual_den_eps=1e-6,
        fox_gd_residual_rho_eps=1e-12,
        fox_gd_residual_beta_init=0.5,
        fox_gd_residual_lambda_init=0.05,
        fox_gd_residual_norm_with_gain="false",
        fox_gd_residual_use_separate_addr_codebook="false",
        vq_score_mode="codebook_dot",
        vq_weight_mode="dense_softmax",
        vq_update_mode="grad",
        vq_softmax_tau=1.0,
        vq_topk=4,
        gradient_accumulation_steps=8,
        train_batch_size=16,
        eval_batch_size=8,
        cache_dir="./data/flash_vqg",
        project="flash_vqg_gd_residual_v1_mqar",
        entity="scu-mclab",
        max_epochs=32,
        metrics_white_list=None,
        metrics_white_list_file=str(
            Path(
                "/home/lyj/mnt/project/zoology/zoology/experiments/flash_vqg/scripts/20260425-gd-residual-v1-mqar/metrics.yaml"
            )
        ),
    )


def _extract_flash_kwargs(config):
    return config.model.sequence_mixer.kwargs["configs"][-1]["kwargs"]


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


def test_e2_main_builder_returns_expected_smoke_and_train_matrix():
    module = _load_e2_builder_module()
    args = _build_e2_args()

    smoke_configs = module.build_e2_main_smoke_configs(args)
    train_configs = module.build_e2_main_train_configs(args)

    assert [config.run_id for config in smoke_configs] == ["baseline", "2a-l050", "2b-l050", "2c-lmax050"]
    assert len(train_configs) == 10
    assert [config.run_id for config in train_configs][:4] == ["baseline", "2a-l025", "2b-l025", "2c-lmax025"]

    for config in train_configs:
        flash_kwargs = _extract_flash_kwargs(config)
        assert flash_kwargs["fox_remote_read_topk"] == 2
        assert flash_kwargs["experiment_part"] == "e2_main"

    baseline_kwargs = _extract_flash_kwargs(train_configs[0])
    assert baseline_kwargs["fox_clr_selector_mode"] == "den_aware"
    assert baseline_kwargs["fox_clr_merge_mode"] == "shared_den"

    two_a_kwargs = _extract_flash_kwargs(train_configs[1])
    assert two_a_kwargs["fox_clr_selector_mode"] == "score_only"
    assert two_a_kwargs["fox_clr_merge_mode"] == "residual_add"
    assert two_a_kwargs["fox_clr_lambda_remote"] == 0.25

    two_c_kwargs = _extract_flash_kwargs(train_configs[3])
    assert two_c_kwargs["fox_clr_gate_mode"] == "shared_query_linear"
    assert two_c_kwargs["fox_clr_gate_init_bias"] == -2.0


def test_e2b_builder_returns_expected_smoke_and_train_matrix():
    module = _load_e2_builder_module()
    args = _build_e2_args()

    smoke_configs = module.build_e2b_smoke_configs(args)
    train_configs = module.build_e2b_train_configs(args)

    assert [config.run_id for config in smoke_configs] == [
        "baseline",
        "e2b-2a-l050",
        "e2b-2b-l050",
        "e2b-2c-lmax050",
    ]
    assert len(train_configs) == 10

    for config in train_configs:
        flash_kwargs = _extract_flash_kwargs(config)
        assert flash_kwargs["fox_remote_read_topk"] == 2
        assert flash_kwargs["experiment_part"] == "e2b"
        assert flash_kwargs["fox_clr_selector_mode"] == "den_aware"

    two_b_kwargs = _extract_flash_kwargs(train_configs[2])
    assert two_b_kwargs["fox_clr_merge_mode"] == "shared_local_den"
    assert two_b_kwargs["fox_clr_lambda_remote"] == 0.25


def test_e2b_decoupled_baseline_probe_builder_allows_seed_data_seed_mismatch():
    module = _load_e2_builder_module()
    args = _build_e2_args()
    args.seed_values = "123"
    args.data_seed = 124

    configs = module.build_e2b_baseline_probe_decoupled_configs(args)

    assert [config.run_id for config in configs] == ["baseline-s123-d124"]
    flash_kwargs = _extract_flash_kwargs(configs[0])
    assert flash_kwargs["experiment_part"] == "e2b_probe"
    assert flash_kwargs["experiment_mode"] == "baseline"
    assert flash_kwargs["fox_clr_selector_mode"] == "den_aware"
    assert flash_kwargs["fox_clr_merge_mode"] == "shared_den"
    assert flash_kwargs["fox_clr_gate_mode"] == "off"
    assert flash_kwargs["fox_clr_lambda_remote"] == 1.0


def test_e3_builder_returns_expected_smoke_and_train_matrix():
    module = _load_e3_builder_module()
    args = _build_e3_args()

    smoke_configs = module.build_e3_smoke_configs(args)
    train_configs = module.build_e3_train_configs(args)

    assert [config.run_id for config in smoke_configs] == ["baseline", "dense-t100"]
    assert [config.run_id for config in train_configs] == [
        "baseline",
        "dense-t050",
        "dense-t100",
        "dense-t200",
    ]

    baseline_kwargs = _extract_flash_kwargs(train_configs[0])
    assert baseline_kwargs["vq_score_mode"] == "l2"
    assert baseline_kwargs["vq_weight_mode"] == "one-hot"
    assert baseline_kwargs["vq_update_mode"] == "ema"
    assert baseline_kwargs["fox_remote_read_topk"] == 2
    assert baseline_kwargs["fox_clr_selector_mode"] == "den_aware"
    assert baseline_kwargs["fox_clr_merge_mode"] == "shared_den"

    dense_kwargs = _extract_flash_kwargs(train_configs[1])
    assert dense_kwargs["vq_score_mode"] == "codebook_dot"
    assert dense_kwargs["vq_weight_mode"] == "dense_softmax"
    assert dense_kwargs["vq_update_mode"] == "grad"
    assert dense_kwargs["vq_softmax_tau"] == 0.5
    assert dense_kwargs["vq_topk"] == 8
    assert dense_kwargs["experiment_part"] == "e3_dense"


def test_gd_residual_builder_returns_expected_smoke_and_train_configs():
    module = _load_gd_residual_builder_module()
    args = _build_gd_residual_args()

    smoke_configs = module.build_gd_residual_v1_smoke_configs(args)
    train_configs = module.build_gd_residual_v1_train_configs(args)

    assert [config.run_id for config in smoke_configs] == [
        "gd-residual-v1-smoke-s123-d123-rread-top2-r16-wk4-b16"
    ]
    assert [config.run_id for config in train_configs] == [
        "gd-residual-v1-train-s123-d123-rread-top2-r16-wk4-b16"
    ]

    smoke_kwargs = _extract_flash_kwargs(smoke_configs[0])
    train_kwargs = _extract_flash_kwargs(train_configs[0])

    assert smoke_kwargs["fox_remote_formula"] == "gd_residual_v1"
    assert smoke_kwargs["fox_remote_read_topk"] == 2
    assert smoke_kwargs["fox_gd_residual_rank"] == 16
    assert smoke_kwargs["fox_gd_residual_builder"] == "grouped_chunk_torch_ref"
    assert smoke_kwargs["fox_gd_residual_pack_mode"] == "semivec_ref"
    assert smoke_kwargs["vq_score_mode"] == "codebook_dot"
    assert smoke_kwargs["vq_weight_mode"] == "dense_softmax"
    assert smoke_kwargs["vq_update_mode"] == "grad"
    assert smoke_kwargs["experiment_part"] == "gd_residual_v1_mqar"
    assert smoke_kwargs["experiment_mode"] == "smoke"
    assert train_kwargs["experiment_mode"] == "train"

    smoke_train_examples = [segment.num_examples for segment in smoke_configs[0].data.train_configs]
    train_train_examples = [segment.num_examples for segment in train_configs[0].data.train_configs]
    smoke_test_examples = [segment.num_examples for segment in smoke_configs[0].data.test_configs]
    train_test_examples = [segment.num_examples for segment in train_configs[0].data.test_configs]

    assert smoke_train_examples == [128, 64, 64, 64, 64]
    assert train_train_examples == [100000, 20000, 20000, 20000, 20000]
    assert smoke_test_examples == [4, 4, 4, 4, 4, 4, 4, 4]
    assert train_test_examples == [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
    assert smoke_configs[0].data.batch_size == (16, 1)
    assert train_configs[0].data.batch_size == (16, 8)


def test_gd_residual_builder_supports_dense_read_topk():
    module = _load_gd_residual_builder_module()
    args = _build_gd_residual_args()
    args.fox_remote_read_topk_values = "dense"

    configs = module.build_gd_residual_v1_train_configs(args)
    flash_kwargs = _extract_flash_kwargs(configs[0])

    assert configs[0].run_id == "gd-residual-v1-train-s123-d123-rread-dense-r16-wk4-b16"
    assert flash_kwargs["fox_remote_read_topk"] is None


def _build_short_run_args(tmp_path):
    return Namespace(
        train_batches=1,
        seeds="123",
        variant="gd_r8_wk4",
        valid_every=0,
        output_dir=str(tmp_path / "out"),
        enable_gd_diagnostics=True,
        append=False,
        batch_size=1,
        eval_batch_size=1,
        test_examples_per_segment=1,
        eval_batches=1,
        d_model=64,
        num_codebook_vectors=8,
        block_len=32,
        local_num_blocks=2,
        learning_rate=1e-3,
        weight_decay=0.1,
        data_seed=123,
        remote_read_topk="2",
        builder="grouped_chunk_torch_ref",
        pack_mode="semivec_ref",
        chunk_size=64,
        fox_gd_residual_mu_min_count=0.5,
        cache_dir=str(tmp_path / "cache"),
        train_batch_order="sequential",
        device="cuda",
    )


def test_gd_residual_short_run_build_config_passes_mu_min_count(tmp_path):
    module = _load_gd_residual_short_run_module()
    args = _build_short_run_args(tmp_path)

    config = module.build_short_run_config(args, module.VARIANTS["gd_r8_wk4"], seed=123)
    flash_kwargs = _extract_flash_kwargs(config)

    assert flash_kwargs["fox_gd_residual_mu_min_count"] == 0.5


def test_gd_residual_short_run_valid_record_includes_phase_step_and_splits(tmp_path, monkeypatch):
    module = _load_gd_residual_short_run_module()
    records_path = tmp_path / "records.jsonl"

    def fake_evaluate(**_kwargs):
        return (
            {
                "valid/attn/gd_residual_mu_valid_ratio": 0.25,
                "valid/attn/gd_residual_debug_event_count": 2.0,
                "valid/attn/gd_residual_debug_l_state_mean": 0.12,
                "valid/layer_0/attn/gd_residual_mu_valid_ratio": 0.5,
            },
            1.25,
            0.75,
            0.125,
        )

    monkeypatch.setattr(module, "_evaluate", fake_evaluate)
    monkeypatch.setattr(
        module,
        "_memory_snapshot",
        lambda _device: {"peak_allocated_bytes": None, "peak_reserved_bytes": None},
    )

    _, valid_loss, valid_accuracy, valid_sec, nan_or_inf = module._append_valid_record(
        records_path=records_path,
        model=object(),
        dataloader=object(),
        loss_fn=object(),
        device=torch.device("cpu"),
        max_batches=1,
        variant=module.VARIANTS["gd_r16_wk4"],
        seed=123,
        run_id="short-run-test",
        step=10,
        phase="periodic_valid",
        last_train_loss=2.0,
    )

    record = json.loads(records_path.read_text(encoding="utf-8"))
    assert (valid_loss, valid_accuracy, valid_sec, nan_or_inf) == (1.25, 0.75, 0.125, False)
    assert record["record_type"] == "valid"
    assert record["phase"] == "periodic_valid"
    assert record["step"] == 10
    assert record["train_loss"] == 2.0
    assert record["gd_residual_metrics"]["valid/attn/gd_residual_mu_valid_ratio"] == 0.25
    assert record["layer_metrics"]["valid/layer_0/attn/gd_residual_mu_valid_ratio"] == 0.5
    assert record["event_diagnostics"]["valid/attn/gd_residual_debug_event_count"] == 2.0
    assert record["l_state_diagnostics"]["valid/attn/gd_residual_debug_l_state_mean"] == 0.12


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gd_residual short-run smoke needs CUDA")
def test_gd_residual_short_run_runner_writes_summary_and_records(tmp_path):
    module = _load_gd_residual_short_run_module()
    args = _build_short_run_args(tmp_path)

    summary = module.run_short_run(args)
    records_path = Path(summary["records_path"])
    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train_record = next(record for record in records if record["record_type"] == "train")
    valid_records = [record for record in records if record["record_type"] == "valid"]
    valid_record = valid_records[-1]

    assert summary["runs"][0]["status"] == "completed"
    assert summary["fox_gd_residual_mu_min_count"] == 0.5
    assert summary["runs"][0]["init_check"]["checked"] is True
    assert [record["phase"] for record in valid_records] == ["initial_valid", "final_valid"]
    assert [record["step"] for record in valid_records] == [0, 1]
    assert "metrics_collect_sec" in train_record
    assert "peak_allocated_bytes" in train_record["memory"]
    assert train_record["gd_residual_metrics"]
    assert train_record["layer_metrics"]
    assert train_record["event_diagnostics"]
    assert train_record["l_state_diagnostics"]
    assert "attn/gd_residual_debug_avg_events_per_group" in train_record["metrics"]
    assert valid_record["valid_loss"] is not None
    assert valid_record["valid_accuracy"] is not None


def test_gd_residual_scripts_and_gitignores_track_only_configs_and_numeric_results():
    base_dir = Path(
        "/home/lyj/mnt/project/zoology/zoology/experiments/flash_vqg/scripts/20260425-gd-residual-v1-mqar"
    )
    readme = (base_dir / "README.md").read_text(encoding="utf-8")
    common_env = (base_dir / "common_env.sh").read_text(encoding="utf-8")
    smoke = (base_dir / "run_smoke.sh").read_text(encoding="utf-8")
    train = (base_dir / "run_train.sh").read_text(encoding="utf-8")
    profile = (base_dir / "run_profile.sh").read_text(encoding="utf-8")
    short_run = (base_dir / "run_short_run.sh").read_text(encoding="utf-8")
    profile_py = (base_dir / "profile_gd_residual_v1.py").read_text(encoding="utf-8")
    short_run_py = (base_dir / "short_run_gd_residual_v1.py").read_text(encoding="utf-8")
    metrics_yaml = (base_dir / "metrics.yaml").read_text(encoding="utf-8")
    generated_gitignore = Path(
        "/home/lyj/mnt/project/zoology/zoology/experiments/flash_vqg/generated/.gitignore"
    ).read_text(encoding="utf-8")
    results_gitignore = Path(
        "/home/lyj/mnt/project/zoology/zoology/analysis/flash_vqg/results/.gitignore"
    ).read_text(encoding="utf-8")

    assert "PROJECT=\"${PROJECT:-flash_vqg_gd_residual_v1_mqar}\"" in common_env
    assert "SWANLAB_MODE=\"${SWANLAB_MODE:-cloud}\"" in common_env
    assert "FOX_REMOTE_READ_TOPK=\"${FOX_REMOTE_READ_TOPK:-2}\"" in common_env
    assert "LAUNCH_ID_PREFIX_SMOKE" in common_env
    assert "LAUNCH_ID_PREFIX_TRAIN" in common_env
    assert "LAUNCH_ID_PREFIX_PROFILE" in common_env
    assert "--config-builder \"${BUILDER_SPEC}\"" in smoke
    assert "--config-builder \"${BUILDER_SPEC}\"" in train
    assert "--fox-remote-read-topk-values \"${FOX_REMOTE_READ_TOPK}\"" in smoke
    assert "--fox-remote-read-topk-values \"${FOX_REMOTE_READ_TOPK}\"" in train
    assert "--fox-gd-residual-rank \"${FOX_GD_RESIDUAL_RANK}\"" in smoke
    assert "--fox-gd-residual-lambda-init \"${FOX_GD_RESIDUAL_LAMBDA_INIT}\"" in train
    assert "PROFILE_ENABLE_TORCH_PROFILER" in profile
    assert "PROFILE_ENABLE_GD_DIAGNOSTICS" in profile
    assert "profile_gd_residual_v1.py" in profile
    assert "SHORT_RUN_TRAIN_BATCHES" in short_run
    assert "SHORT_RUN_SEEDS" in short_run
    assert "SHORT_RUN_VARIANT" in short_run
    assert "SHORT_RUN_VALID_EVERY" in short_run
    assert "SHORT_RUN_OUTPUT_DIR" in short_run
    assert "export FOX_GD_RESIDUAL_MU_MIN_COUNT" in short_run
    assert "short_run_gd_residual_v1.py" in short_run
    assert "metrics_collect_sec" in profile_py
    assert "logged_step_sec" not in profile_py
    assert "metrics_collect_sec" in short_run_py
    assert "initial_valid" in short_run_py
    assert "periodic_valid" in short_run_py
    assert "final_valid" in short_run_py
    assert "records.jsonl" in short_run_py
    assert "peak_allocated_bytes" in short_run_py
    assert "gd_residual_debug_avg_events_per_group" in short_run_py
    assert "run_smoke.sh" in readme
    assert "run_train.sh" in readme
    assert "run_profile.sh" in readme
    assert "run_short_run.sh" in readme
    assert "PROFILE_ENABLE_GD_DIAGNOSTICS=1" in readme
    assert "`PROFILE_ENABLE_GD_DIAGNOSTICS`: 默认 `0`" in readme
    assert "SHORT_RUN_VALID_EVERY" in readme
    assert "FOX_GD_RESIDUAL_MU_MIN_COUNT" in readme
    assert "flash_vqg_gd_residual_v1_mqar" in readme
    assert "attn/gd_residual_lambda_mean" in metrics_yaml
    assert "attn/gd_residual_debug_avg_events_per_group" in metrics_yaml
    assert "layer_*/attn/gd_residual_mu_valid_ratio" in metrics_yaml
    assert "valid/attn/gd_residual_mu_valid_ratio" in metrics_yaml
    assert "valid/layer_*/attn/gd_residual_inject_ratio" in metrics_yaml
    assert "debug/gd_token_chunk_max_diff" not in metrics_yaml

    for content in (generated_gitignore, results_gitignore):
        assert "!flash-vqg-20260425-gd-residual-v1-mqar*/" in content
        assert "flash-vqg-20260425-gd-residual-v1-mqar*/**/*.png" in content
        assert "flash-vqg-20260425-gd-residual-v1-mqar*/**/*.log" in content


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


def test_e2_scripts_use_config_builder_and_smoke_env_files():
    base_dir = Path(
        "/home/lyj/mnt/project/zoology/zoology/experiments/flash_vqg/scripts/20260402-clr-v1-mainline/e2-remote-interface"
    )
    smoke_main = (base_dir / "run_e2_main_smoke.sh").read_text(encoding="utf-8")
    smoke_e2b = (base_dir / "run_e2b_smoke.sh").read_text(encoding="utf-8")
    train_main = (base_dir / "run_e2_main_train.sh").read_text(encoding="utf-8")
    train_e2b = (base_dir / "run_e2b_train.sh").read_text(encoding="utf-8")
    dual_train = (base_dir / "run_e2_dual_train.sh").read_text(encoding="utf-8")
    common_env = Path(
        "/home/lyj/mnt/project/zoology/zoology/experiments/flash_vqg/scripts/20260402-clr-v1-mainline/common_env.sh"
    ).read_text(encoding="utf-8")

    assert "SMOKE_ENV_FILE" in smoke_main
    assert "e2_main_smoke.env" in smoke_main
    assert "SMOKE_ENV_FILE" in smoke_e2b
    assert "e2b_smoke.env" in smoke_e2b

    assert "--config-builder \"${BUILDER_SPEC}\"" in train_main
    assert "source \"${ENV_FILE}\"" in train_main
    assert "--analysis \"${ANALYSIS_SOURCE}\"" in train_main
    assert "--config-builder \"${BUILDER_SPEC}\"" in train_e2b
    assert "source \"${ENV_FILE}\"" in train_e2b

    assert "GPU_ID_E2_MAIN" in dual_train
    assert "TRAIN_BATCH_SIZE_E2_MAIN" in dual_train
    assert "TRAIN_BATCH_SIZE_E2B" in dual_train

    assert "LAUNCH_ID_PREFIX_E2_MAIN" in common_env
    assert "LAUNCH_ID_PREFIX_E2B" in common_env
    assert "METRICS_WHITE_LIST_FILE_E2" in common_env


def test_e2_decoupled_baseline_probe_script_uses_dedicated_builder():
    base_dir = Path(
        "/home/lyj/mnt/project/zoology/zoology/experiments/flash_vqg/scripts/20260402-clr-v1-mainline/e2-remote-interface"
    )
    script = (base_dir / "run_e2b_baseline_probe_decoupled.sh").read_text(encoding="utf-8")
    readme = (base_dir / "README.md").read_text(encoding="utf-8")

    assert "build_e2b_baseline_probe_decoupled_configs" in script
    assert "decoupled baseline probe 只接受单个 SEED_VALUES" in script
    assert "run_e2b_baseline_probe_decoupled.sh" in readme
    assert "允许 `SEED_VALUES != DATA_SEED`" in readme


def test_e3_scripts_use_config_builder_and_smoke_env_files():
    base_dir = Path(
        "/home/lyj/mnt/project/zoology/zoology/experiments/flash_vqg/scripts/20260402-clr-v1-mainline/e3-dense-routing"
    )
    readme = (base_dir / "README.md").read_text(encoding="utf-8")
    smoke = (base_dir / "run_e3_smoke.sh").read_text(encoding="utf-8")
    train = (base_dir / "run_e3_train.sh").read_text(encoding="utf-8")
    metrics_yaml = (base_dir / "metrics.yaml").read_text(encoding="utf-8")

    assert "dense write + top2 read" in readme
    assert "e3_smoke.env" in smoke
    assert "SMOKE_ENV_FILE" in smoke
    assert "--config-builder \"${BUILDER_SPEC}\"" in train
    assert "--logger-backend swanlab" in train
    assert "source \"${ENV_FILE}\"" in train
    assert "--analysis \"${ANALYSIS_SOURCE}\"" in train
    assert "vq/write_entropy_mean" in metrics_yaml
    assert "valid/vq/write_top1_mass_mean" in metrics_yaml


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
