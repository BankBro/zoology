import json
import os
import sys
import types
from argparse import Namespace
from pathlib import Path

import pytest

from zoology.experiments.flash_vqg.flash_vqg_suite import build_configs
import zoology.experiments.flash_vqg.run_flash_vqg_suite as suite
from zoology.experiments.flash_vqg.run_flash_vqg_suite import (
    _build_configs_from_builder,
    _build_e7_eval_run_ids,
    _build_eval_run_id,
    _builder_args_dict,
    _load_config_builder,
    _parse_codebook_vectors_map,
    _parse_csv_ints,
    _parse_paired_block_local,
    _parse_remote_read_topk_values,
    _parse_seed_values,
    _render_generated_config_from_builder,
    _resolve_metrics_white_list,
    _render_generated_config,
)


def test_parse_csv_ints_supports_multi_block_len_values():
    assert _parse_csv_ints("8,16,32,64") == [8, 16, 32, 64]


def test_parse_codebook_vectors_map_supports_dmodel_mapping():
    assert _parse_codebook_vectors_map("128:128,256:256") == {
        128: 128,
        256: 256,
    }


def test_parse_seed_values_supports_multi_seed_scan():
    assert _parse_seed_values("123,456,789") == [123, 456, 789]


def test_parse_paired_block_local_supports_zip_scan_values():
    assert _parse_paired_block_local("8:8,16:4,32:2,64:1") == [
        (8, 8),
        (16, 4),
        (32, 2),
        (64, 1),
    ]


def test_parse_remote_read_topk_values_supports_dense_and_sparse_modes():
    assert _parse_remote_read_topk_values("dense,2,4") == [None, 2, 4]


def test_build_eval_run_id_prefixes_checkpoint_run_id():
    assert _build_eval_run_id("flash_vqg_h2_accel-block32", "e4a") == "eval_e4a_flash_vqg_h2_accel-block32"


def test_build_e7_eval_run_ids_creates_three_mode_runs():
    assert _build_e7_eval_run_ids("flash_vqg_h2_accel-block32") == [
        "eval_e7_dense_flash_vqg_h2_accel-block32",
        "eval_e7_top2_flash_vqg_h2_accel-block32",
        "eval_e7_top4_flash_vqg_h2_accel-block32",
    ]


def test_resolve_metrics_white_list_merges_file_and_inline(tmp_path):
    metrics_file = tmp_path / "e1.yaml"
    metrics_file.write_text("- valid/accuracy\n- valid/mqar_case/*\n", encoding="utf-8")

    resolved = _resolve_metrics_white_list(
        metrics_white_list_raw="valid/accuracy,attn/remote_win_rate",
        metrics_white_list_file=str(metrics_file),
    )

    assert resolved == ["valid/accuracy", "valid/mqar_case/*", "attn/remote_win_rate"]


def test_load_config_builder_supports_file_path():
    builder_path = Path(
        "/home/lyj/mnt/project/zoology/zoology/experiments/flash_vqg/scripts/20260402-clr-v1-mainline/e2-remote-interface/config_builder.py"
    )
    builder = _load_config_builder(f"{builder_path}:build_e2_main_smoke_configs")

    assert callable(builder)
    assert builder.__name__ == "build_e2_main_smoke_configs"


def test_builder_args_dict_excludes_eval_only_fields():
    args = Namespace(
        backend="torch",
        eval_only=True,
        checkpoint_launch_id="launch-x",
        checkpoint_run_id="run-y",
        dmodels="128",
    )

    payload = _builder_args_dict(args)

    assert payload == {"backend": "torch", "dmodels": "128"}


def test_render_generated_config_from_builder_writes_builder_loader():
    rendered = _render_generated_config_from_builder(
        builder_spec="/tmp/demo_builder.py:build_demo",
        builder_args={"backend": "torch", "dmodels": "128"},
    )

    assert "_load_config_builder('/tmp/demo_builder.py:build_demo')" in rendered
    assert "_builder_args = argparse.Namespace(**{'backend': 'torch', 'dmodels': '128'})" in rendered
    assert "configs = _builder(_builder_args)" in rendered


def test_build_configs_from_builder_requires_non_empty_list(tmp_path):
    builder_path = tmp_path / "bad_builder.py"
    builder_path.write_text(
        "def build_configs(args):\n    return []\n",
        encoding="utf-8",
    )
    args = Namespace()

    with pytest.raises(ValueError, match="非空 list"):
        _build_configs_from_builder(builder_spec=f"{builder_path}:build_configs", args=args)


def test_render_generated_config_writes_block_len_values_scan():
    rendered = _render_generated_config(
        sweep_id="flash-vqg-e3",
        backend="accel",
        logger_backend="swanlab",
        include_gdn=False,
        block_lens=[8, 16, 32, 64],
        paired_block_local_values=None,
        dmodels=[128],
        learning_rates=[1e-3],
        if_remote_enabled_values=[False],
        local_num_blocks_values=[1, 2],
        train_batch_orders=["global_shuffle"],
        seed_values=None,
        data_seed=123,
        num_codebook_vectors_values=None,
        num_codebook_vectors_map=None,
        fox_remote_path_backend=None,
        fox_remote_read_topk_values=None,
        fox_remote_formula="legacy",
        fox_clr_rank=4,
        fox_clr_use_den_residual=True,
        fox_clr_remat_mode="off",
        gradient_accumulation_steps=1,
        train_batch_size=None,
        eval_batch_size=None,
        cache_dir="./data/flash_vqg",
        wandb_project="flash_vqg_mqar",
        wandb_entity="scu-mclab",
        max_epochs=32,
        metrics_white_list=["valid/accuracy", "valid/mqar_case/*"],
    )

    assert "block_len_values=[8, 16, 32, 64]" in rendered
    assert "block_len=" not in rendered
    assert "metrics_white_list=['valid/accuracy', 'valid/mqar_case/*']" in rendered
    assert "fox_remote_formula='legacy'" in rendered
    assert "fox_clr_remat_mode='off'" in rendered


def test_render_generated_config_writes_paired_block_local_scan():
    rendered = _render_generated_config(
        sweep_id="flash-vqg-e3",
        backend="accel",
        logger_backend="swanlab",
        include_gdn=False,
        block_lens=None,
        paired_block_local_values=[(8, 8), (16, 4), (32, 2), (64, 1)],
        dmodels=[128],
        learning_rates=[1e-3],
        if_remote_enabled_values=[False],
        local_num_blocks_values=None,
        train_batch_orders=["global_shuffle"],
        seed_values=None,
        data_seed=123,
        num_codebook_vectors_values=None,
        num_codebook_vectors_map=None,
        fox_remote_path_backend=None,
        fox_remote_read_topk_values=None,
        fox_remote_formula="legacy",
        fox_clr_rank=4,
        fox_clr_use_den_residual=True,
        fox_clr_remat_mode="off",
        gradient_accumulation_steps=1,
        train_batch_size=None,
        eval_batch_size=None,
        cache_dir="./data/flash_vqg",
        wandb_project="flash_vqg_mqar",
        wandb_entity="scu-mclab",
        max_epochs=32,
        metrics_white_list=["valid/accuracy"],
    )

    assert "paired_block_local_values=[(8, 8), (16, 4), (32, 2), (64, 1)]" in rendered
    assert "block_len_values=" not in rendered
    assert "local_num_blocks_values=" not in rendered
    assert "fox_clr_rank=4" in rendered


def test_render_generated_config_writes_codebook_sweep_and_map():
    rendered = _render_generated_config(
        sweep_id="flash-vqg-e5",
        backend="accel",
        logger_backend="swanlab",
        include_gdn=False,
        block_lens=[32],
        paired_block_local_values=None,
        dmodels=[128],
        learning_rates=[1e-3],
        if_remote_enabled_values=[True],
        local_num_blocks_values=[2],
        train_batch_orders=["global_shuffle"],
        seed_values=None,
        data_seed=123,
        num_codebook_vectors_values=[64, 128, 256, 512],
        num_codebook_vectors_map=None,
        fox_remote_path_backend=None,
        fox_remote_read_topk_values=None,
        fox_remote_formula="legacy",
        fox_clr_rank=4,
        fox_clr_use_den_residual=True,
        fox_clr_remat_mode="off",
        gradient_accumulation_steps=1,
        train_batch_size=None,
        eval_batch_size=None,
        cache_dir="./data/flash_vqg",
        wandb_project="flash_vqg_mqar",
        wandb_entity="scu-mclab",
        max_epochs=32,
        metrics_white_list=["valid/accuracy"],
    )

    assert "num_codebook_vectors_values=[64, 128, 256, 512]" in rendered
    assert "num_codebook_vectors_map=None" in rendered
    assert "fox_clr_use_den_residual=True" in rendered

    rendered_map = _render_generated_config(
        sweep_id="flash-vqg-default",
        backend="accel",
        logger_backend="swanlab",
        include_gdn=False,
        block_lens=[32],
        paired_block_local_values=None,
        dmodels=[128, 256],
        learning_rates=[1e-3],
        if_remote_enabled_values=[True],
        local_num_blocks_values=[2],
        train_batch_orders=["global_shuffle"],
        seed_values=None,
        data_seed=123,
        num_codebook_vectors_values=None,
        num_codebook_vectors_map={128: 128, 256: 256},
        fox_remote_path_backend=None,
        fox_remote_read_topk_values=None,
        fox_remote_formula="legacy",
        fox_clr_rank=4,
        fox_clr_use_den_residual=True,
        fox_clr_remat_mode="off",
        gradient_accumulation_steps=1,
        train_batch_size=None,
        eval_batch_size=None,
        cache_dir="./data/flash_vqg",
        wandb_project="flash_vqg_mqar",
        wandb_entity="scu-mclab",
        max_epochs=32,
        metrics_white_list=["valid/accuracy"],
    )

    assert "num_codebook_vectors_values=None" in rendered_map
    assert "num_codebook_vectors_map={128: 128, 256: 256}" in rendered_map


def test_render_generated_config_writes_batch_and_gradient_accumulation_overrides():
    rendered = _render_generated_config(
        sweep_id="flash-vqg-ga",
        backend="torch",
        logger_backend="none",
        include_gdn=False,
        block_lens=[32],
        paired_block_local_values=None,
        dmodels=[128],
        learning_rates=[1e-3],
        if_remote_enabled_values=[True],
        local_num_blocks_values=[2],
        train_batch_orders=["global_shuffle"],
        seed_values=None,
        data_seed=123,
        num_codebook_vectors_values=[128],
        num_codebook_vectors_map=None,
        fox_remote_path_backend="torch",
        fox_remote_read_topk_values=None,
        fox_remote_formula="clr_v1",
        fox_clr_rank=4,
        fox_clr_use_den_residual=True,
        fox_clr_remat_mode="off",
        gradient_accumulation_steps=8,
        train_batch_size=16,
        eval_batch_size=24,
        cache_dir="./data/flash_vqg",
        wandb_project="flash_vqg_mqar",
        wandb_entity="scu-mclab",
        max_epochs=32,
        metrics_white_list=["valid/accuracy"],
    )

    assert "gradient_accumulation_steps=8" in rendered
    assert "train_batch_size=16" in rendered
    assert "eval_batch_size=24" in rendered


def _flash_num_codebook_vectors(config) -> int:
    return config.model.sequence_mixer.kwargs["configs"][-1]["kwargs"]["num_codebook_vectors"]


def _flash_remote_read_topk(config):
    return config.model.sequence_mixer.kwargs["configs"][-1]["kwargs"]["fox_remote_read_topk"]


def _flash_remote_path_backend(config) -> str:
    return config.model.sequence_mixer.kwargs["configs"][-1]["kwargs"]["fox_remote_path_backend"]


def _flash_remote_formula(config) -> str:
    return config.model.sequence_mixer.kwargs["configs"][-1]["kwargs"]["fox_remote_formula"]


def _flash_clr_remat_mode(config) -> str:
    return config.model.sequence_mixer.kwargs["configs"][-1]["kwargs"]["fox_clr_remat_mode"]


def test_build_configs_sweeps_num_codebook_vectors_values():
    configs = build_configs(
        include_gdn=False,
        block_len=32,
        dmodels=[128],
        learning_rates=[1e-3],
        if_remote_enabled=True,
        local_num_blocks=2,
        train_batch_order="global_shuffle",
        num_codebook_vectors_values=[64, 128, 256, 512],
        metrics_white_list=["valid/accuracy"],
    )

    assert len(configs) == 4
    assert [_flash_num_codebook_vectors(config) for config in configs] == [64, 128, 256, 512]
    assert [config.run_id for config in configs] == [
        "flash_vqg_h2_accel-block32-dmodel128-cb64-lr1.0e-03-local2-remote1-sampler-gshuffle-rformula-legacy",
        "flash_vqg_h2_accel-block32-dmodel128-cb128-lr1.0e-03-local2-remote1-sampler-gshuffle-rformula-legacy",
        "flash_vqg_h2_accel-block32-dmodel128-cb256-lr1.0e-03-local2-remote1-sampler-gshuffle-rformula-legacy",
        "flash_vqg_h2_accel-block32-dmodel128-cb512-lr1.0e-03-local2-remote1-sampler-gshuffle-rformula-legacy",
    ]


def test_build_configs_applies_num_codebook_vectors_map_for_selected_dmodels():
    configs = build_configs(
        include_gdn=False,
        block_len=32,
        dmodels=[128, 256],
        learning_rates=[1e-3],
        if_remote_enabled=True,
        local_num_blocks=2,
        train_batch_order="global_shuffle",
        num_codebook_vectors_map={64: 64, 128: 96, 256: 192},
        metrics_white_list=["valid/accuracy"],
    )

    assert len(configs) == 2
    assert [(config.model.d_model, _flash_num_codebook_vectors(config)) for config in configs] == [
        (128, 96),
        (256, 192),
    ]
    assert [config.run_id for config in configs] == [
        "flash_vqg_h2_accel-block32-dmodel128-cb96-lr1.0e-03-local2-remote1-sampler-gshuffle-rformula-legacy",
        "flash_vqg_h2_accel-block32-dmodel256-cb192-lr1.0e-03-local2-remote1-sampler-gshuffle-rformula-legacy",
    ]


def test_build_configs_requires_selected_dmodels_in_num_codebook_vectors_map():
    with pytest.raises(ValueError, match="缺少这些 d_model"):
        build_configs(
            include_gdn=False,
            block_len=32,
            dmodels=[128, 256],
            learning_rates=[1e-3],
            if_remote_enabled=True,
            local_num_blocks=2,
            train_batch_order="global_shuffle",
            num_codebook_vectors_map={128: 128},
            metrics_white_list=["valid/accuracy"],
        )


def test_build_configs_sweeps_seed_and_remote_read_topk_values():
    configs = build_configs(
        include_gdn=False,
        block_len=32,
        dmodels=[128],
        learning_rates=[1e-3],
        if_remote_enabled=True,
        local_num_blocks=2,
        train_batch_order="global_shuffle",
        seed_values=[123, 456, 789],
        data_seed=123,
        num_codebook_vectors_values=[128],
        fox_remote_path_backend="torch",
        fox_remote_read_topk_values=[None, 2, 4],
        metrics_white_list=["valid/accuracy"],
    )

    assert len(configs) == 9
    assert [config.seed for config in configs] == [123, 456, 789, 123, 456, 789, 123, 456, 789]
    assert [config.data.seed for config in configs] == [123] * 9
    assert [_flash_remote_path_backend(config) for config in configs] == ["torch"] * 9
    assert [_flash_remote_formula(config) for config in configs] == ["legacy"] * 9
    assert [_flash_remote_read_topk(config) for config in configs] == [None, None, None, 2, 2, 2, 4, 4, 4]
    assert [config.run_id for config in configs] == [
        "flash_vqg_h2_accel-block32-dmodel128-cb128-lr1.0e-03-local2-remote1-sampler-gshuffle-rformula-legacy-rread-dense-seed123",
        "flash_vqg_h2_accel-block32-dmodel128-cb128-lr1.0e-03-local2-remote1-sampler-gshuffle-rformula-legacy-rread-dense-seed456",
        "flash_vqg_h2_accel-block32-dmodel128-cb128-lr1.0e-03-local2-remote1-sampler-gshuffle-rformula-legacy-rread-dense-seed789",
        "flash_vqg_h2_accel-block32-dmodel128-cb128-lr1.0e-03-local2-remote1-sampler-gshuffle-rformula-legacy-rread-top2-seed123",
        "flash_vqg_h2_accel-block32-dmodel128-cb128-lr1.0e-03-local2-remote1-sampler-gshuffle-rformula-legacy-rread-top2-seed456",
        "flash_vqg_h2_accel-block32-dmodel128-cb128-lr1.0e-03-local2-remote1-sampler-gshuffle-rformula-legacy-rread-top2-seed789",
        "flash_vqg_h2_accel-block32-dmodel128-cb128-lr1.0e-03-local2-remote1-sampler-gshuffle-rformula-legacy-rread-top4-seed123",
        "flash_vqg_h2_accel-block32-dmodel128-cb128-lr1.0e-03-local2-remote1-sampler-gshuffle-rformula-legacy-rread-top4-seed456",
        "flash_vqg_h2_accel-block32-dmodel128-cb128-lr1.0e-03-local2-remote1-sampler-gshuffle-rformula-legacy-rread-top4-seed789",
    ]


def test_build_configs_supports_clr_formula_suffix():
    configs = build_configs(
        include_gdn=False,
        flash_backend="torch",
        block_len=32,
        dmodels=[128],
        learning_rates=[1e-3],
        if_remote_enabled=True,
        local_num_blocks=2,
        train_batch_order="global_shuffle",
        num_codebook_vectors_values=[128],
        fox_remote_formula="clr_v1",
        fox_clr_rank=4,
        fox_clr_use_den_residual=True,
        fox_clr_remat_mode="off",
        metrics_white_list=["valid/accuracy"],
    )

    assert len(configs) == 1
    assert _flash_remote_formula(configs[0]) == "clr_v1"
    assert _flash_clr_remat_mode(configs[0]) == "off"
    assert configs[0].run_id.endswith("-rformula-clr1-r4-den1-rremat-off")


def test_build_configs_supports_clr_v1_remote_read_topk_suffixes():
    configs = build_configs(
        include_gdn=False,
        flash_backend="torch",
        block_len=32,
        dmodels=[128],
        learning_rates=[1e-3],
        if_remote_enabled=True,
        local_num_blocks=2,
        train_batch_order="global_shuffle",
        num_codebook_vectors_values=[128],
        fox_remote_formula="clr_v1",
        fox_clr_rank=4,
        fox_clr_use_den_residual=True,
        fox_clr_remat_mode="off",
        fox_remote_read_topk_values=[None, 1, 2, 4],
        fox_remote_path_backend="torch",
        metrics_white_list=["valid/accuracy"],
    )

    assert len(configs) == 4
    assert [_flash_remote_read_topk(config) for config in configs] == [None, 1, 2, 4]
    assert [config.run_id for config in configs] == [
        "flash_vqg_h2_torch-block32-dmodel128-cb128-lr1.0e-03-local2-remote1-sampler-gshuffle-rformula-clr1-r4-den1-rremat-off-rread-dense",
        "flash_vqg_h2_torch-block32-dmodel128-cb128-lr1.0e-03-local2-remote1-sampler-gshuffle-rformula-clr1-r4-den1-rremat-off-rread-top1",
        "flash_vqg_h2_torch-block32-dmodel128-cb128-lr1.0e-03-local2-remote1-sampler-gshuffle-rformula-clr1-r4-den1-rremat-off-rread-top2",
        "flash_vqg_h2_torch-block32-dmodel128-cb128-lr1.0e-03-local2-remote1-sampler-gshuffle-rformula-clr1-r4-den1-rremat-off-rread-top4",
    ]


def test_build_configs_supports_batch_and_gradient_accumulation_overrides():
    configs = build_configs(
        include_gdn=False,
        flash_backend="torch",
        block_len=32,
        dmodels=[128],
        learning_rates=[1e-3],
        if_remote_enabled=True,
        local_num_blocks=2,
        train_batch_order="global_shuffle",
        num_codebook_vectors_values=[128],
        fox_remote_formula="clr_v1",
        fox_clr_rank=4,
        fox_clr_use_den_residual=True,
        fox_clr_remat_mode="off",
        train_batch_size=16,
        eval_batch_size=24,
        gradient_accumulation_steps=8,
        metrics_white_list=["valid/accuracy"],
    )

    assert len(configs) == 1
    assert configs[0].data.batch_size == (16, 24)
    assert configs[0].gradient_accumulation_steps == 8
    assert configs[0].run_id.endswith("-rformula-clr1-r4-den1-rremat-off-tbs16-ebs24-ga8")


def test_build_configs_keeps_default_run_id_when_batch_and_gradient_accumulation_match_defaults():
    configs = build_configs(
        include_gdn=False,
        flash_backend="torch",
        block_len=32,
        dmodels=[128],
        learning_rates=[1e-3],
        if_remote_enabled=True,
        local_num_blocks=2,
        train_batch_order="global_shuffle",
        num_codebook_vectors_values=[128],
        fox_remote_formula="clr_v1",
        fox_clr_rank=4,
        fox_clr_use_den_residual=True,
        fox_clr_remat_mode="off",
        train_batch_size=256,
        eval_batch_size=32,
        gradient_accumulation_steps=1,
        metrics_white_list=["valid/accuracy"],
    )

    assert len(configs) == 1
    assert configs[0].run_id.endswith("-rformula-clr1-r4-den1-rremat-off")
    assert "-tbs256-ebs32-ga1" not in configs[0].run_id


def test_build_configs_rejects_non_positive_batch_or_gradient_accumulation():
    with pytest.raises(ValueError, match="train_batch_size"):
        build_configs(
            include_gdn=False,
            block_len=32,
            dmodels=[128],
            learning_rates=[1e-3],
            if_remote_enabled=True,
            local_num_blocks=2,
            train_batch_order="global_shuffle",
            train_batch_size=0,
            metrics_white_list=["valid/accuracy"],
        )

    with pytest.raises(ValueError, match="gradient_accumulation_steps"):
        build_configs(
            include_gdn=False,
            block_len=32,
            dmodels=[128],
            learning_rates=[1e-3],
            if_remote_enabled=True,
            local_num_blocks=2,
            train_batch_order="global_shuffle",
            gradient_accumulation_steps=0,
            metrics_white_list=["valid/accuracy"],
        )


def test_build_configs_supports_clr_rank_zero_suffix():
    configs = build_configs(
        include_gdn=False,
        flash_backend="torch",
        block_len=32,
        dmodels=[128],
        learning_rates=[1e-3],
        if_remote_enabled=True,
        local_num_blocks=2,
        train_batch_order="global_shuffle",
        num_codebook_vectors_values=[128],
        fox_remote_formula="clr_v1",
        fox_clr_rank=0,
        fox_clr_use_den_residual=False,
        fox_clr_remat_mode="off",
        metrics_white_list=["valid/accuracy"],
    )

    assert len(configs) == 1
    assert _flash_remote_formula(configs[0]) == "clr_v1"
    assert _flash_clr_remat_mode(configs[0]) == "off"
    assert configs[0].run_id.endswith("-rformula-clr1-r0-den0-rremat-off")


def test_build_configs_rejects_clr_rank_zero_with_den_residual():
    with pytest.raises(ValueError, match="fox_clr_rank=0"):
        build_configs(
            include_gdn=False,
            flash_backend="torch",
            block_len=32,
            dmodels=[128],
            learning_rates=[1e-3],
            if_remote_enabled=True,
            local_num_blocks=2,
            train_batch_order="global_shuffle",
            num_codebook_vectors_values=[128],
            fox_remote_formula="clr_v1",
            fox_clr_rank=0,
            fox_clr_use_den_residual=True,
            fox_clr_remat_mode="off",
            metrics_white_list=["valid/accuracy"],
        )


def test_build_configs_rejects_clr_with_accel_backend():
    with pytest.raises(ValueError, match="flash_backend='torch'"):
        build_configs(
            include_gdn=False,
            flash_backend="accel",
            block_len=32,
            dmodels=[128],
            learning_rates=[1e-3],
            if_remote_enabled=True,
            local_num_blocks=2,
            train_batch_order="global_shuffle",
            num_codebook_vectors_values=[128],
            fox_remote_formula="clr_v1",
            fox_clr_rank=4,
            fox_clr_use_den_residual=True,
            fox_clr_remat_mode="off",
            metrics_white_list=["valid/accuracy"],
        )


def test_build_configs_rejects_clr_with_triton_remote_backend():
    with pytest.raises(ValueError, match="fox_remote_path_backend='torch'"):
        build_configs(
            include_gdn=False,
            flash_backend="torch",
            block_len=32,
            dmodels=[128],
            learning_rates=[1e-3],
            if_remote_enabled=True,
            local_num_blocks=2,
            train_batch_order="global_shuffle",
            num_codebook_vectors_values=[128],
            fox_remote_formula="clr_v1",
            fox_clr_rank=4,
            fox_clr_use_den_residual=True,
            fox_clr_remat_mode="off",
            fox_remote_path_backend="triton",
            metrics_white_list=["valid/accuracy"],
        )


def test_build_configs_rejects_clr_delta_v1_remote_topk():
    with pytest.raises(ValueError, match="clr_delta_v1"):
        build_configs(
            include_gdn=False,
            flash_backend="torch",
            block_len=32,
            dmodels=[128],
            learning_rates=[1e-3],
            if_remote_enabled=True,
            local_num_blocks=2,
            train_batch_order="global_shuffle",
            num_codebook_vectors_values=[128],
            fox_remote_formula="clr_delta_v1",
            fox_clr_rank=4,
            fox_clr_use_den_residual=True,
            fox_clr_remat_mode="off",
            fox_remote_path_backend="torch",
            fox_remote_read_topk_values=[None, 2],
            metrics_white_list=["valid/accuracy"],
        )


def test_build_configs_supports_clr_remat_suffix():
    configs = build_configs(
        include_gdn=False,
        flash_backend="torch",
        block_len=32,
        dmodels=[128],
        learning_rates=[1e-3],
        if_remote_enabled=True,
        local_num_blocks=2,
        train_batch_order="global_shuffle",
        num_codebook_vectors_values=[128],
        fox_remote_formula="clr_v1",
        fox_clr_rank=4,
        fox_clr_use_den_residual=True,
        fox_clr_remat_mode="post_phase1",
        metrics_white_list=["valid/accuracy"],
    )

    assert len(configs) == 1
    assert _flash_clr_remat_mode(configs[0]) == "post_phase1"
    assert configs[0].run_id.endswith("-rformula-clr1-r4-den1-rremat-postp1")


def test_build_configs_rejects_legacy_with_clr_remat():
    with pytest.raises(ValueError, match="fox_clr_remat_mode"):
        build_configs(
            include_gdn=False,
            flash_backend="torch",
            block_len=32,
            dmodels=[128],
            learning_rates=[1e-3],
            if_remote_enabled=True,
            local_num_blocks=2,
            train_batch_order="global_shuffle",
            num_codebook_vectors_values=[128],
            fox_remote_formula="legacy",
            fox_clr_rank=4,
            fox_clr_use_den_residual=True,
            fox_clr_remat_mode="post_phase1",
            metrics_white_list=["valid/accuracy"],
        )


def test_build_configs_rejects_clr_remat_when_layer_metrics_enabled():
    with pytest.raises(ValueError, match="enable_layer_metrics=True"):
        build_configs(
            include_gdn=False,
            flash_backend="torch",
            block_len=32,
            dmodels=[128],
            learning_rates=[1e-3],
            if_remote_enabled=True,
            local_num_blocks=2,
            train_batch_order="global_shuffle",
            num_codebook_vectors_values=[128],
            fox_remote_formula="clr_v1",
            fox_clr_rank=4,
            fox_clr_use_den_residual=True,
            fox_clr_remat_mode="post_phase1",
            metrics_white_list=["attn/remote_win_rate"],
        )


def test_main_eval_only_requires_checkpoint_ids(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_flash_vqg_suite.py", "--eval-only", "e4a"])

    with pytest.raises(ValueError, match="checkpoint-launch-id"):
        suite.main()


def test_main_eval_only_creates_eval_launch_and_dispatches(monkeypatch, tmp_path):
    captured = {}
    generated_root = tmp_path / "generated"

    def fake_run_eval_only(
        *,
        eval_task: str,
        eval_launch_id: str,
        eval_sweep_id: str,
        eval_run_id: str,
        eval_run_ids: list[str] | None,
        checkpoint_launch_id: str,
        checkpoint_run_id: str,
        logger_backend: str,
        project: str,
        entity: str,
        manifest_path,
        gpus: str | None,
        metrics_white_list: list[str] | None,
    ):
        captured.update(
            {
                "eval_task": eval_task,
                "eval_launch_id": eval_launch_id,
                "eval_sweep_id": eval_sweep_id,
                "eval_run_id": eval_run_id,
                "eval_run_ids": eval_run_ids,
                "checkpoint_launch_id": checkpoint_launch_id,
                "checkpoint_run_id": checkpoint_run_id,
                "logger_backend": logger_backend,
                "project": project,
                "entity": entity,
                "manifest_path": str(manifest_path),
                "gpus": gpus,
                "metrics_white_list": metrics_white_list,
            }
        )
        return {"output_dir": "/tmp/e4a-results"}

    monkeypatch.setattr(suite, "GENERATED_DIR", generated_root)
    monkeypatch.setattr(suite, "_build_launch_id", lambda prefix: f"{prefix}-2026-03-27-00-00-00")
    monkeypatch.setattr(suite, "_run_eval_only", fake_run_eval_only)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_flash_vqg_suite.py",
            "--eval-only",
            "e4a",
            "--checkpoint-launch-id",
            "demo-launch",
            "--checkpoint-run-id",
            "demo-run",
            "--logger-backend",
            "swanlab",
            "--project",
            "demo-project",
            "--entity",
            "demo-entity",
            "--launch-id-prefix",
            "flash-vqg-e4",
            "--gpus",
            "3",
            "--metrics-white-list",
            "valid/accuracy,valid/mqar_case/*",
        ],
    )

    suite.main()

    manifest_path = generated_root / "flash-vqg-e4-2026-03-27-00-00-00" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert captured == {
        "eval_task": "e4a",
        "eval_launch_id": "flash-vqg-e4-2026-03-27-00-00-00",
        "eval_sweep_id": "flash-vqg-e4",
        "eval_run_id": "eval_e4a_demo-run",
        "eval_run_ids": None,
        "checkpoint_launch_id": "demo-launch",
        "checkpoint_run_id": "demo-run",
        "logger_backend": "swanlab",
        "project": "demo-project",
        "entity": "demo-entity",
        "manifest_path": str(manifest_path),
        "gpus": "3",
        "metrics_white_list": ["valid/accuracy", "valid/mqar_case/*"],
    }
    assert manifest["launch_id"] == "flash-vqg-e4-2026-03-27-00-00-00"
    assert manifest["sweep_id"] == "flash-vqg-e4"
    assert manifest["eval_task"] == "e4a"
    assert manifest["runs"][0]["run_id"] == "eval_e4a_demo-run"
    assert manifest["runs"][0]["eval_task"] == "e4a"
    assert manifest["runs"][0]["eval_source"] == {
        "checkpoint_launch_id": "demo-launch",
        "checkpoint_run_id": "demo-run",
        "best_checkpoint": None,
    }


def test_main_e7_eval_only_creates_three_runs(monkeypatch, tmp_path):
    captured = {}
    generated_root = tmp_path / "generated"

    def fake_run_eval_only(**kwargs):
        captured.update(kwargs)
        return {"output_dir": "/tmp/e7-results"}

    monkeypatch.setattr(suite, "GENERATED_DIR", generated_root)
    monkeypatch.setattr(suite, "_build_launch_id", lambda prefix: f"{prefix}-2026-03-27-00-00-00")
    monkeypatch.setattr(suite, "_run_eval_only", fake_run_eval_only)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_flash_vqg_suite.py",
            "--eval-only",
            "e7",
            "--checkpoint-launch-id",
            "demo-launch",
            "--checkpoint-run-id",
            "demo-run",
            "--launch-id-prefix",
            "flash-vqg-e7",
        ],
    )

    suite.main()

    manifest_path = generated_root / "flash-vqg-e7-2026-03-27-00-00-00" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert captured["eval_task"] == "e7"
    assert captured["eval_run_id"] == "eval_e7_dense_demo-run"
    assert captured["eval_run_ids"] == [
        "eval_e7_dense_demo-run",
        "eval_e7_top2_demo-run",
        "eval_e7_top4_demo-run",
    ]
    assert [run["run_id"] for run in manifest["runs"]] == captured["eval_run_ids"]
    assert all(run["eval_task"] == "e7" for run in manifest["runs"])


def test_main_eval_only_runs_analysis_on_eval_launch(monkeypatch, tmp_path):
    generated_root = tmp_path / "generated"
    subprocess_calls = []

    monkeypatch.setattr(suite, "GENERATED_DIR", generated_root)
    monkeypatch.setattr(suite, "_build_launch_id", lambda prefix: f"{prefix}-2026-03-27-00-00-01")
    monkeypatch.setattr(suite, "_run_eval_only", lambda **kwargs: {"output_dir": "/tmp/e4a-results"})
    monkeypatch.setattr(
        suite.subprocess,
        "run",
        lambda cmd, check, cwd, env: subprocess_calls.append(
            {
                "cmd": cmd,
                "check": check,
                "cwd": cwd,
                "env_launch_manifest": env.get(suite.MANIFEST_ENV_VAR),
            }
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_flash_vqg_suite.py",
            "--eval-only",
            "e4a",
            "--checkpoint-launch-id",
            "demo-launch",
            "--checkpoint-run-id",
            "demo-run",
            "--launch-id-prefix",
            "flash-vqg-e4",
            "--analysis",
            "remote",
        ],
    )

    suite.main()

    assert len(subprocess_calls) == 1
    assert subprocess_calls[0]["cmd"][-2:] == ["--source", "remote"]
    assert subprocess_calls[0]["cmd"][-4:-2] == ["--launch-id", "flash-vqg-e4-2026-03-27-00-00-01"]


def test_run_eval_only_sets_cuda_visible_devices(monkeypatch):
    captured = {}
    manifest_path = suite.GENERATED_DIR / "dummy" / "manifest.json"

    def fake_run_e4a_eval(**kwargs):
        captured["kwargs"] = kwargs
        captured["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        return {"output_dir": "/tmp/e4a-results"}

    monkeypatch.setitem(
        sys.modules,
        "zoology.experiments.flash_vqg.eval_only",
        types.SimpleNamespace(run_e4a_eval=fake_run_e4a_eval),
    )
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    result = suite._run_eval_only(
        eval_task="e4a",
        eval_launch_id="flash-vqg-e4-2026-03-27-00-00-02",
        eval_sweep_id="flash-vqg-e4",
        eval_run_id="eval_e4a_demo-run",
        eval_run_ids=None,
        checkpoint_launch_id="demo-launch",
        checkpoint_run_id="demo-run",
        logger_backend="swanlab",
        project="demo-project",
        entity="demo-entity",
        manifest_path=manifest_path,
        gpus="5",
        metrics_white_list=["valid/accuracy"],
    )

    assert result["output_dir"] == "/tmp/e4a-results"
    assert captured["cuda_visible_devices"] == "5"
    assert captured["kwargs"]["eval_run_id"] == "eval_e4a_demo-run"
    assert captured["kwargs"]["manifest_path"] == manifest_path
    assert captured["kwargs"]["metrics_white_list"] == ["valid/accuracy"]


def test_run_eval_only_dispatches_e7(monkeypatch):
    captured = {}

    def fake_run_e7_eval(**kwargs):
        captured.update(kwargs)
        return {"output_dir": "/tmp/e7-results"}

    monkeypatch.setitem(
        sys.modules,
        "zoology.experiments.flash_vqg.eval_only",
        types.SimpleNamespace(run_e4a_eval=lambda **kwargs: {"output_dir": "/tmp/e4a-results"}, run_e7_eval=fake_run_e7_eval),
    )

    result = suite._run_eval_only(
        eval_task="e7",
        eval_launch_id="flash-vqg-e7-2026-03-27-00-00-03",
        eval_sweep_id="flash-vqg-e7",
        eval_run_id="eval_e7_dense_demo-run",
        eval_run_ids=["eval_e7_dense_demo-run", "eval_e7_top2_demo-run", "eval_e7_top4_demo-run"],
        checkpoint_launch_id="demo-launch",
        checkpoint_run_id="demo-run",
        logger_backend="none",
        project="demo-project",
        entity="demo-entity",
        manifest_path=suite.GENERATED_DIR / "dummy" / "manifest.json",
        gpus=None,
        metrics_white_list=None,
    )

    assert result["output_dir"] == "/tmp/e7-results"
    assert captured["eval_run_ids"] == [
        "eval_e7_dense_demo-run",
        "eval_e7_top2_demo-run",
        "eval_e7_top4_demo-run",
    ]


def test_main_training_writes_e7_train_sweep(monkeypatch, tmp_path):
    generated_root = tmp_path / "generated"
    subprocess_calls = []

    monkeypatch.setattr(suite, "GENERATED_DIR", generated_root)
    monkeypatch.setattr(suite, "_build_launch_id", lambda prefix: f"{prefix}-2026-03-28-00-00-00")
    monkeypatch.setattr(
        suite.subprocess,
        "run",
        lambda cmd, check, cwd, env: subprocess_calls.append(
            {
                "cmd": cmd,
                "check": check,
                "cwd": cwd,
                "manifest_env": env.get(suite.MANIFEST_ENV_VAR),
            }
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_flash_vqg_suite.py",
            "--flash-only",
            "--backend",
            "accel",
            "--block-len",
            "32",
            "--dmodels",
            "128",
            "--learning-rates",
            "1e-3",
            "--local-num-blocks",
            "2",
            "--if-remote-enabled",
            "true",
            "--train-batch-order",
            "global_shuffle",
            "--seed-values",
            "123,456,789",
            "--data-seed",
            "123",
            "--fox-remote-path-backend",
            "torch",
            "--fox-remote-read-topk-values",
            "dense,2,4",
            "--num-codebook-vectors",
            "128",
            "--train-batch-size",
            "16",
            "--eval-batch-size",
            "24",
            "--gradient-accumulation-steps",
            "8",
            "--launch-id-prefix",
            "flash-vqg-e7-train",
        ],
    )

    suite.main()

    generated_dir = generated_root / "flash-vqg-e7-train-2026-03-28-00-00-00"
    manifest = json.loads((generated_dir / "manifest.json").read_text(encoding="utf-8"))
    generated_config = (generated_dir / "launch_configs.py").read_text(encoding="utf-8")

    assert len(manifest["runs"]) == 9
    assert manifest["runs"][0]["run_id"].endswith("-rread-dense-seed123-tbs16-ebs24-ga8")
    assert manifest["runs"][-1]["run_id"].endswith("-rread-top4-seed789-tbs16-ebs24-ga8")
    assert "seed_values=[123, 456, 789]" in generated_config
    assert "data_seed=123" in generated_config
    assert "fox_remote_path_backend='torch'" in generated_config
    assert "fox_remote_read_topk_values=[None, 2, 4]" in generated_config
    assert "fox_remote_formula='legacy'" in generated_config
    assert "train_batch_size=16" in generated_config
    assert "eval_batch_size=24" in generated_config
    assert "gradient_accumulation_steps=8" in generated_config
    assert len(subprocess_calls) == 1
    assert subprocess_calls[0]["manifest_env"] == str((generated_dir / "manifest.json").resolve())
